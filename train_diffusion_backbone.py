import os
import random

import flax
import numpy as np
import jax
from jax import config
# config.update("jax_debug_nans", True)
from jax.experimental.compilation_cache import compilation_cache as cc
import jax.numpy as jnp
from einops import rearrange
from flax.training import train_state
import optax
import flax.linen as nn
import orbax.checkpoint
from tqdm import tqdm
import wandb
import jmp
import cv2


from vae import Decoder, Encoder
from dit import DiTBLock
from diffusers import FlaxDDIMScheduler, FlaxDDPMScheduler
from utils import FrozenModel, create_image_mosaic, flatten_dict, unflatten_dict
from streaming_dataloader import threading_dataloader, collate_labeled_imagenet_fn, SquareImageNetDataset, rando_colours

# sharding
from jax.sharding import Mesh
from jax.sharding import PartitionSpec
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils
cc.initialize_cache("jax_cache")

# adjust this sharding mesh to create appropriate sharding rule
devices = mesh_utils.create_device_mesh((jax.device_count(), 1))
# create axis name on how many parallelism slice you want on your model
mesh = Mesh(devices, axis_names=("data_parallel", "model_parallel"))

        # just fancy wrapper
mixed_precision_policy = jmp.Policy(
    compute_dtype=jnp.bfloat16,
    param_dtype=jnp.float32,
    output_dtype=jnp.float32
)

def checkpoint_manager(save_path, max_to_keep=2):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
    return orbax.checkpoint.CheckpointManager(os.path.abspath(save_path), orbax_checkpointer, options)


def init_model(batch_size = 256, training_res = 256, seed = 42, learning_rate = 10e-3):
    # TODO: move all hardcoded value as config
    with jax.default_device(jax.devices("cpu")[0]):
        enc_rng, dec_rng, dit_rng = jax.random.split(jax.random.PRNGKey(seed), 3)

        enc = Encoder(
            output_features = 768,
            down_layer_contraction_factor = ( (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)),
            down_layer_dim = (128, 256, 512, 512, 1024),
            down_layer_kernel_size = ( 3, 3, 3, 3, 3),
            down_layer_blocks = (4, 4, 4, 4, 2),
            down_layer_ordinary_conv = (True, True, True, True, False),
            down_layer_residual = (True, True, True, True, True),
            use_bias = False,
            conv_expansion_factor = (1, 1, 1, 1, 2),
            eps = 1e-6,
            group_count = 16,
            last_layer = "linear",
        )
        dec = Decoder(
            output_features = 3,
            up_layer_contraction_factor = ( (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)),
            up_layer_dim = (1024, 512, 512, 256, 128),
            up_layer_kernel_size = ( 3, 3, 3, 3, 3),
            up_layer_blocks = (2, 4, 4, 4, 4),
            up_layer_ordinary_conv = (False, True, True, True, True),
            up_layer_residual = (True, True, True, True, True),
            use_bias = True,
            conv_expansion_factor = (2, 1, 1, 1, 1),
            eps = 1e-6,
            group_count = 16,
        )

        dit_backbone = DiTBLock(
            n_layers=10, 
            embed_dim=768, 
            n_heads=8, 
            use_flash_attention=False, 
            latent_size=training_res, 
            n_class=1001, # last class is a null class where it's untrained and serve as random vector
            pixel_based=True
        )

        # init model params
        # create param for each model
        # encoder
        image = jnp.ones((batch_size, training_res, training_res, 3))
        enc_params = enc.init(enc_rng, image)
        # decoder
        dummy_latent = jnp.ones((batch_size, training_res // 32, training_res // 32, 768))
        dec_params = dec.init(dec_rng, dummy_latent)
        # dit
        latent = dummy_latent
        # img_pos = dit_backbone.create_2d_sinusoidal_pos(training_res // 32, training_res // 32)
        timesteps = jnp.ones([batch_size]).astype(jnp.int32)
        conds = jnp.ones([batch_size]).astype(jnp.int32)
        #  x, timestep, cond=None, image_pos=None, extra_pos=None
        latent = jnp.ones((batch_size, training_res, training_res, 3))
        dit_params = dit_backbone.init(dit_rng, latent, timesteps, conds)


        enc_param_count = sum(list(flatten_dict(jax.tree_map(lambda x: x.size, enc_params)).values()))
        dec_param_count = sum(list(flatten_dict(jax.tree_map(lambda x: x.size, dec_params)).values()))
        dit_param_count = sum(list(flatten_dict(jax.tree_map(lambda x: x.size, dit_params)).values()))
        print("encoder param count:", f"{enc_param_count:,}")
        print("decoder param count:", f"{dec_param_count:,}")
        print("transformer param count:", f"{dit_param_count:,}")
        tabulate_dit_backbone = nn.tabulate(dit_backbone, jax.random.key(0), compute_flops=True, compute_vjp_flops=True)

        print(tabulate_dit_backbone(latent, timesteps, conds))
        # create callable optimizer chain
        def adam_wrapper(mask):
            constant_scheduler = optax.constant_schedule(learning_rate)
            adamw = optax.adamw(
                learning_rate=constant_scheduler,
                b1=0.9,
                b2=0.999,
                eps=1e-08,
                mask=mask,
            )
            u_net_optimizer = optax.chain(
                optax.clip_by_global_norm(1),  # prevent explosion
                adamw,
            )
            return u_net_optimizer

        # do not apply weight decay to norm layer
        # trained
        dit_state = train_state.TrainState.create(
            apply_fn=dit_backbone.apply,
            params=dit_params,
            tx=adam_wrapper(
                jax.tree_util.tree_map_with_path(lambda path, var: path[-1].key != "scale" and path[-1].key != "bias", dit_params)
            ),
        )
        # frozen
        enc_state = FrozenModel(
            call=enc.apply,
            params=enc_params,
        )
        # frozen
        dec_state = FrozenModel(
            call=dec.apply,
            params=dec_params,
        )
        
        # put everything in accelerator in data parallel mode
        enc_state = jax.tree_map(
            lambda leaf: jax.device_put(jmp.cast_to_half(leaf), device=NamedSharding(mesh, PartitionSpec())),
            enc_state,
        )
        dec_state = jax.tree_map(
            lambda leaf: jax.device_put(jmp.cast_to_half(leaf), device=NamedSharding(mesh, PartitionSpec())),
            dec_state,
        )
        dit_state = jax.tree_map(
            lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, PartitionSpec())),
            dit_state,
        )
        # TODO: yoink diffusers scheduler as separate module and simplify it for further tweaking 
        training_scheduler = FlaxDDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="squaredcos_cap_v2", # <<<< there's better schedule for this gonna tweak it later
            num_train_timesteps=1000,
            prediction_type="epsilon",
        )
        training_scheduler_params = training_scheduler.create_state()
        training_scheduler_params = jax.tree_map(
            lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, PartitionSpec())),
            training_scheduler_params,
        )
        training_scheduler_state = FrozenModel(
            call=training_scheduler, # just pass the whole object, diffusers make things complicated
            params=training_scheduler_params,
        )

        # just use good ol ddim for now
        inference_scheduler = FlaxDDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="squaredcos_cap_v2", # <<<< there's better schedule for this gonna tweak it later
            num_train_timesteps=1000,
            prediction_type="epsilon",
        )
        inference_scheduler_params = inference_scheduler.create_state()
        inference_scheduler_params = jax.tree_map(
            lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, PartitionSpec())),
            inference_scheduler_params,
        )
        inference_scheduler_state = FrozenModel(
            call=inference_scheduler, # just pass the whole object, diffusers make things complicated
            params=inference_scheduler_params,
        )

        return [dit_state, enc_state, dec_state, training_scheduler_state, inference_scheduler_state]


@jax.jit
def train(dit_state, frozen_models, batch, train_rng):

    # always create new RNG!
    sample_rng, new_train_rng = jax.random.split(train_rng, num=2)
    
    # unpack
    enc_state, training_scheduler_state = frozen_models

    def _compute_loss(
        dit_params, enc_params, training_scheduler_params, batch, rng_key
    ):
        dit_params, enc_params, batch = mixed_precision_policy.cast_to_compute((dit_params, enc_params, batch))
    
        images = batch["images"]
        class_cond = batch["labels"]
        n, h, w, c = images.shape

        latents = enc_state.call(enc_params, images)

        # logvar is not used
        # we dont do sampling here, we just want mean value
        # sampling only useful for decoder because decoder need to be robust to noise
        # we want the distribution to be exact dead on the mean for the diffusion backbone 
        # think of it like diffusion model doing sloppy job denoising the image and the 
        # decoder cleaning up the remaining residual noise
        latent_mean, latent_logvar = rearrange(latents, "b h w (c split) -> split b h w c", split = 2)

        # Sample noise that we'll add to the images
        # I think I should combine this with the first noise seed generator
        noise_rng, timestep_rng = jax.random.split(
            key=rng_key, num=2
        )
        
        noise = jax.random.normal(key=noise_rng, shape=latent_mean.shape)

        # Sample a random timestep for each image
        timesteps = jax.random.randint(
            key=timestep_rng,
            shape=(n,),
            minval=0,
            maxval=training_scheduler_state.call.config.num_train_timesteps,
        )

        # Add noise to the images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_image = training_scheduler_state.call.add_noise(
            state=training_scheduler_params,
            original_samples=latent_mean,
            noise=noise,
            timesteps=timesteps,
        )
        #  x, timestep, cond, image_pos, extra_pos=None
        # TODO: add a way to interpolate positional embedding
        predicted_noise = dit_state.apply_fn(dit_params, noisy_image, timesteps, class_cond)

        # MSE loss
        loss = (noise - predicted_noise) ** 2
        loss = loss.mean()
        return mixed_precision_policy.cast_to_output(loss)

    grad_fn = jax.value_and_grad(
        fun=_compute_loss, argnums=[0,]  # differentiate first param only
    )

    loss, grad = grad_fn(
        dit_state.params,
        enc_state.params,
        training_scheduler_state.params, 
        batch, 
        sample_rng,
    )
    # update weight and bias value
    dit_state = dit_state.apply_gradients(grads=jmp.cast_to_full(grad[0]))

    # calculate loss
    metrics = {"mse_loss": loss}
    return (
        dit_state,
        metrics,
        new_train_rng,
    )

@jax.jit
def train_pixel_based(dit_state, frozen_models, batch, train_rng):

    # always create new RNG!
    sample_rng, new_train_rng = jax.random.split(train_rng, num=2)
    
    # unpack
    enc_state, training_scheduler_state = frozen_models

    def _compute_loss(
        dit_params, enc_params, training_scheduler_params, batch, rng_key
    ):
        dit_params, enc_params, batch = mixed_precision_policy.cast_to_compute((dit_params, enc_params, batch))
    
        images = batch["images"]
        class_cond = batch["labels"]
        n, h, w, c = images.shape

        # latents = enc_state.call(enc_params, images)

        # logvar is not used
        # we dont do sampling here, we just want mean value
        # sampling only useful for decoder because decoder need to be robust to noise
        # we want the distribution to be exact dead on the mean for the diffusion backbone 
        # think of it like diffusion model doing sloppy job denoising the image and the 
        # decoder cleaning up the remaining residual noise
        # latent_mean, latent_logvar = rearrange(latents, "b h w (c split) -> split b h w c", split = 2)

        # Sample noise that we'll add to the images
        # I think I should combine this with the first noise seed generator
        noise_rng, timestep_rng = jax.random.split(
            key=rng_key, num=2
        )
        
        noise = jax.random.normal(key=noise_rng, shape=images.shape)

        # Sample a random timestep for each image
        timesteps = jax.random.randint(
            key=timestep_rng,
            shape=(n,),
            minval=0,
            maxval=training_scheduler_state.call.config.num_train_timesteps,
        )

        # Add noise to the images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_image = training_scheduler_state.call.add_noise(
            state=training_scheduler_params,
            original_samples=images,
            noise=noise,
            timesteps=timesteps,
        )
        #  x, timestep, cond, image_pos, extra_pos=None
        # TODO: add a way to interpolate positional embedding
        predicted_noise = dit_state.apply_fn(dit_params, noisy_image, timesteps, class_cond)

        # MSE loss
        loss = (noise - predicted_noise) ** 2
        loss = loss.mean()
        return mixed_precision_policy.cast_to_output(loss)

    grad_fn = jax.value_and_grad(
        fun=_compute_loss, argnums=[0,]  # differentiate first param only
    )

    loss, grad = grad_fn(
        dit_state.params,
        enc_state.params,
        training_scheduler_state.params, 
        batch, 
        sample_rng,
    )
    # update weight and bias value
    dit_state = dit_state.apply_gradients(grads=jmp.cast_to_full(grad[0]))

    # calculate loss
    metrics = {"mse_loss": loss}
    return (
        dit_state,
        metrics,
        new_train_rng,
    )

@jax.jit
def inference(
    dit_state,
    frozen_models,
    class_cond,
    seed = jax.random.key(0),
    width = 32,
    height = 32,
    n_latent_dim = 768,
    guidance_scale = 1,
    num_inference_steps = 50,
):  
    # unpack
    dec_state, inference_scheduler_state = frozen_models
    batch_count = len(class_cond)

    # generate random latent images 
    # number of element in class_cond will determine the number of images generated 
    latents_shape = (batch_count, height, width, n_latent_dim)
    latents = jax.random.normal(seed, shape=latents_shape, dtype=jnp.float32)


    scheduler_state = inference_scheduler_state.call.set_timesteps(
        inference_scheduler_state.params, 
        num_inference_steps=num_inference_steps, 
        shape=latents_shape
    )

    # scale the initial noise by the scale required by the scheduler
    latents = latents * inference_scheduler_state.params.init_noise_sigma

    # pack it so jax for i loop can work on this 
    loop_state = (scheduler_state, latents)

    def single_step_pass(loop_counter, loop_state):  
        scheduler_state, latents = loop_state
        # need 2 images one for prompt and the other for neg prompt so just duplicate this
        # this is used for classifier free guidance, to contrast the vector towards bad stuff

        # get scheduler timestep (reverse time step) from this loop step
        t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[loop_counter]
        # create broadcastable array for the batch dim
        timestep = jnp.broadcast_to(t, latents.shape[0])

        # get sample noised latent from the schedule
        latents = inference_scheduler_state.call.scale_model_input(
            scheduler_state, latents, timestep
        )

        # predict the noise residual to be substracted
        #  x, timestep, cond, image_pos, extra_pos=None
        predicted_noise = dit_state.apply_fn(dit_state.params, latents, timestep, class_cond)
        predicted_noise_uncond = dit_state.apply_fn(
            dit_state.params, 
            latents, 
            timestep, 
            jnp.array([1000] * batch_count).astype(jnp.int32) # condition on null class (untrained embedding vector)
        )

        # classifier free guidance
        noise_pred = predicted_noise_uncond + guidance_scale * (predicted_noise - predicted_noise_uncond)

        # "subtract" the noise and return less noised sample  and the state back
        latents, scheduler_state = inference_scheduler_state.call.step(
            scheduler_state, noise_pred, t, latents
        ).to_tuple()

        return scheduler_state, latents

    # functional way to write for loops (don't judge)
    loop_state = jax.lax.fori_loop(0, num_inference_steps, single_step_pass, loop_state)

    # unpack loop_state
    scheduler_state, latents = loop_state

    # decode back to image space
    images = dec_state.call(dec_state.params, latents)
    return images


@jax.jit
def inference_pixel_space(
    dit_state,
    frozen_models,
    class_cond,
    seed = jax.random.key(0),
    width = 32,
    height = 32,
    n_latent_dim = 3,
    guidance_scale = 3,
    num_inference_steps = 100,
):  
    guidance_scale = jnp.array([guidance_scale], dtype=jnp.float32)
    # unpack
    dec_state, inference_scheduler_state = frozen_models
    batch_count = len(class_cond)

    # generate random latent images 
    # number of element in class_cond will determine the number of images generated 
    latents_shape = (batch_count, height, width, n_latent_dim)
    latents = jax.random.normal(seed, shape=latents_shape, dtype=jnp.float32)

    # scale the initial noise by the scale required by the scheduler
    latents = latents * inference_scheduler_state.params.init_noise_sigma

    scheduler_state = inference_scheduler_state.call.set_timesteps(
        inference_scheduler_state.params, 
        num_inference_steps=num_inference_steps, 
        shape=latents_shape
    )


    # pack it so jax for i loop can work on this 
    loop_state = (scheduler_state, latents)

    def single_step_pass(loop_counter, loop_state):  
        scheduler_state, latents = loop_state
        # need 2 images one for prompt and the other for neg prompt so just duplicate this
        # this is used for classifier free guidance, to contrast the vector towards bad stuff

        # get scheduler timestep (reverse time step) from this loop step
        t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[loop_counter]
        # create broadcastable array for the batch dim
        timestep = jnp.broadcast_to(t, latents.shape[0])

        # get sample noised latent from the schedule
        latents = inference_scheduler_state.call.scale_model_input(
            scheduler_state, latents, timestep
        )

        # predict the noise residual to be substracted
        #  x, timestep, cond, image_pos, extra_pos=None
        predicted_noise = dit_state.apply_fn(dit_state.params, latents, timestep, class_cond)
        predicted_noise_uncond = dit_state.apply_fn(
            dit_state.params, 
            latents, 
            timestep, 
            jnp.array([1000] * batch_count).astype(jnp.int32) # condition on null class (untrained embedding vector)
        )

        # classifier free guidance
        noise_pred = predicted_noise_uncond + guidance_scale * (predicted_noise - predicted_noise_uncond)

        # "subtract" the noise and return less noised sample  and the state back
        latents, scheduler_state = inference_scheduler_state.call.step(
            scheduler_state, noise_pred, t, latents
        ).to_tuple()

        return scheduler_state, latents

    # functional way to write for loops (don't judge)
    # run with python for loop
    # for i in range(num_inference_steps):
    #     latents, scheduler_state = single_step_pass(i, loop_state)
    loop_state = jax.lax.fori_loop(0, num_inference_steps, single_step_pass, loop_state)
# 
    # unpack loop_state
    scheduler_state, latents = loop_state

    return latents

def main():
    BATCH_SIZE = 128
    SEED = 0
    SAVE_MODEL_PATH = "dit_ckpt"
    IMAGE_RES = 32
    SAVE_EVERY = 500
    LEARNING_RATE = 1e-4
    WANDB_PROJECT_NAME = "DiT"
    WANDB_RUN_NAME = "test"
    WANDB_LOG_INTERVAL = 100

    # wandb logging
    if WANDB_PROJECT_NAME:
        wandb.init(project=WANDB_PROJECT_NAME, name=WANDB_RUN_NAME)

    # init seed
    train_rng = jax.random.PRNGKey(SEED)
    # init checkpoint manager
    ckpt_manager = checkpoint_manager(SAVE_MODEL_PATH)
    # init models
    models = init_model(batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, training_res=IMAGE_RES)
    # unpack models
    dit_state, enc_state, dec_state, training_scheduler_state, inference_scheduler_state = models
    # pack for training
    frozen_training_state = [enc_state, training_scheduler_state]
    # pack for inference
    frozen_inference_state = [dec_state, inference_scheduler_state]


    # Open the text file in read mode
    image_paths = ["ramdisk/train_images"] * 10

    STEPS = 0
    try:
        for image_path in image_paths:
            # dataset = CustomDataset(parquet_url, square_size=IMAGE_RES)
            dataset = SquareImageNetDataset(image_path, square_size=IMAGE_RES, seed=STEPS)
            t_dl = threading_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_labeled_imagenet_fn,  num_workers=100, prefetch_factor=0.5, seed=SEED)
            # Initialize the progress bar
            progress_bar = tqdm(total=len(dataset) // BATCH_SIZE, position=0)

            for i, batch in enumerate(t_dl):
                STEPS += 1
                if i > len(dataset) // BATCH_SIZE -10:
                    # i should fix my dataloader instead of doing this
                    break          
                # image batch already rescaled inside collate_labeled_imagenet_fn
                # regularization to flat colours
                # batch["images"][0] = rando_colours(IMAGE_RES)
                # batch["labels"][0] = 1000

                batch["images"] = jax.tree_map(
                    lambda leaf: jax.device_put(
                        leaf, device=NamedSharding(mesh, PartitionSpec("data_parallel", None, None, None))
                    ),
                    batch["images"],
                )
                batch["labels"] = jax.tree_map(
                    lambda leaf: jax.device_put(
                        leaf, device=NamedSharding(mesh, PartitionSpec("data_parallel"))
                    ),
                    batch["labels"],
                )

                dit_state, metrics, train_rng = train_pixel_based(dit_state, frozen_training_state, batch, train_rng)
                # dit_state, metrics, train_rng = train(dit_state, frozen_training_state, batch, train_rng)

                if jnp.isnan(metrics["mse_loss"]).any():
                    raise ValueError("The array contains NaN values")

                if i % WANDB_LOG_INTERVAL == 0:
                    progress_bar.set_description(f"{metrics}")
                    wandb.log(metrics, step=STEPS)
                    preview = inference_pixel_space(dit_state, frozen_inference_state, batch["labels"][:BATCH_SIZE//4])
                    preview = np.array((jnp.concatenate([preview, batch["images"][:BATCH_SIZE//4]], axis=0) + 1) / 2 * 255, dtype=np.uint8)
                    create_image_mosaic(preview, 8, 8, f"{STEPS}.png")

                # save every n steps
                if i % SAVE_EVERY == 0:
                    # preview = inference(dit_state, frozen_inference_state, batch["labels"][:4])
                    preview = inference_pixel_space(dit_state, frozen_inference_state, batch["labels"][:4])
                    preview = np.array((jnp.concatenate([preview, batch["images"][:4]], axis=0) + 1) / 2 * 255, dtype=np.uint8)
                    create_image_mosaic(preview, 2, 4, f"{STEPS}.png")
                    wandb.log({"image": wandb.Image(f'{STEPS}.png')}, step=STEPS)

                    ckpt_manager.save(STEPS, models)


                progress_bar.update(1)

    except KeyboardInterrupt:
        STEPS += 1
        print("Ctrl+C command detected. saving model before exiting...")
        ckpt_manager.save(STEPS, models)

main()
