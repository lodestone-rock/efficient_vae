import os
import random
from functools import partial
import time

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
from flax import struct
from typing import Callable
import dm_pix as pix

from vae import Decoder, Encoder
from dit import DiTBLockContinuous, rand_stratified_cosine
from utils import FrozenModel, create_image_mosaic, flatten_dict, unflatten_dict
from streaming_dataloader import threading_dataloader, collate_labeled_imagenet_fn, SquareImageNetDataset, rando_colours, OxfordFlowersDataset

# sharding
from jax.sharding import Mesh
from jax.sharding import PartitionSpec
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils
cc.initialize_cache("jax_cache")

print(jax.devices())
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


def init_model(batch_size = 256, training_res = 256, latent_ratio = 32, seed = 42, learning_rate = 10e-3, pretrained_vae_ckpt=None, vae_ckpt_steps=1, latent_depth=256, patch_size=1):
    # TODO: move all hardcoded value as config
    with jax.default_device(jax.devices("cpu")[0]):
        enc_rng, dec_rng, dit_rng = jax.random.split(jax.random.PRNGKey(seed), 3)

        enc = Encoder(
            output_features = latent_depth,
            down_layer_contraction_factor = ((2, 2), (2, 2), (2, 2), (2, 2)),
            down_layer_dim = (128, 128, 128, 256), # deliberate expansion at the bottleneck
            down_layer_kernel_size = ( 3, 3, 3, 3),
            down_layer_blocks = (4, 4, 4, 4),
            down_layer_ordinary_conv = (True, True, True, True),
            down_layer_residual = (True, True, True, True),
            use_bias = False,
            conv_expansion_factor = (1, 1, 1, 1),
            eps = 1e-6,
            group_count = 16,
            last_layer = "conv",
        )
        dec = Decoder(
            output_features = 3,
            up_layer_contraction_factor = ((2, 2), (2, 2), (2, 2), (2, 2)),
            up_layer_dim = (256, 128, 128, 128), # deliberate expansion at the bottleneck
            up_layer_kernel_size = ( 3, 3, 3, 3),
            up_layer_blocks = (4, 4, 4, 4),
            up_layer_ordinary_conv = (True, True, True, True),
            up_layer_residual = (True, True, True, True),
            use_bias = True,
            conv_expansion_factor = (1, 1, 1, 1),
            eps = 1e-6,
            group_count = 16,
        )

        dit_backbone = DiTBLockContinuous(
            n_layers=8, 
            embed_dim=512, 
            output_dim=latent_depth,
            n_heads=8, 
            use_flash_attention=False, 
            latent_size=training_res // latent_ratio, 
            n_class=1001, # last class is a null class where it's untrained and serve as random vector
            pixel_based=False,
            attn_expansion_factor=1,
            patch_size=patch_size,
        )

        # init model params
        # create param for each model
        # encoder
        image = jnp.ones((batch_size, training_res, training_res, 3))
        enc_params = enc.init(enc_rng, image)
        # decoder
        dummy_latent = jnp.ones((batch_size, training_res // latent_ratio, training_res // latent_ratio, latent_depth))
        dec_params = dec.init(dec_rng, dummy_latent)
        # dit
        timesteps = jnp.ones([batch_size]).astype(jnp.int32)
        conds = jnp.ones([batch_size]).astype(jnp.int32)
        #  x, timestep, cond=None, image_pos=None, extra_pos=None
        dit_params = dit_backbone.init(dit_rng, dummy_latent, timesteps, conds, toggle_cond=conds)
        # load vae ckpt
        vae_manager = checkpoint_manager(pretrained_vae_ckpt,2)
        raw_restored = vae_manager.restore(directory=os.path.abspath(pretrained_vae_ckpt), step=vae_ckpt_steps)
        if pretrained_vae_ckpt is not None:
            if vae_ckpt_steps < 100:
                print("ARE YOU FREAKING SURE THAT VAE IS TRAINED?")
            # overwrite vae params with pretrained
            enc_params_flatten = flatten_dict(enc_params)
            dec_params_flatten = flatten_dict(dec_params)

            orbax_enc_params_flatten = flatten_dict(raw_restored[0]["params"])
            orbax_dec_params_flatten = flatten_dict(raw_restored[1]["params"])

            enc_params = unflatten_dict({key: orbax_enc_params_flatten[key] for key in enc_params_flatten})
            dec_params = unflatten_dict({key: orbax_dec_params_flatten[key] for key in dec_params_flatten})


            enc_params = jax.tree_map(lambda x: jax.device_put(x, jax.devices()[0]), enc_params)
            dec_params = jax.tree_map(lambda x: jax.device_put(x, jax.devices()[0]), dec_params)

        enc_param_count = sum(list(flatten_dict(jax.tree_map(lambda x: x.size, enc_params)).values()))
        dec_param_count = sum(list(flatten_dict(jax.tree_map(lambda x: x.size, dec_params)).values()))
        dit_param_count = sum(list(flatten_dict(jax.tree_map(lambda x: x.size, dit_params)).values()))
        print("encoder param count:", f"{enc_param_count:,}")
        print("decoder param count:", f"{dec_param_count:,}")
        print("transformer param count:", f"{dit_param_count:,}")
        # tabulate_dit_backbone = nn.tabulate(dit_backbone, jax.random.key(0), compute_flops=True, compute_vjp_flops=True)

        # print(tabulate_dit_backbone(dummy_latent, timesteps, conds))
        # create callable optimizer chain
        def adam_wrapper(mask):
            constant_scheduler = optax.constant_schedule(learning_rate)
            adamw = optax.adamw(
                learning_rate=constant_scheduler,
                b1=0.9,
                b2=0.95,
                eps=1e-08,
                weight_decay=1e-3,
                mask=mask,
            )
            u_net_optimizer = optax.chain(
                optax.clip_by_global_norm(1),  # prevent explosion
                adamw,
            )
            return u_net_optimizer
        
        class CustomDiTState(train_state.TrainState):
            # use pytree_node=False to indicate an attribute should not be touched
            # by Jax transformations.
            loss: Callable = struct.field(pytree_node=False)
            pred: Callable = struct.field(pytree_node=False)
            sigma_data: float = struct.field(pytree_node=False)
            rectified_flow_loss: Callable = struct.field(pytree_node=False)
            pred_rectified_flow: Callable = struct.field(pytree_node=False)


        # do not apply weight decay to norm layer
        # trained
        dit_state = CustomDiTState.create(
            apply_fn=dit_backbone.apply,
            params=dit_params,
            tx=adam_wrapper(
                jax.tree_util.tree_map_with_path(lambda path, var: path[-1].key != "scale" and path[-1].key != "bias", dit_params)
            ),
            loss=dit_backbone.loss,
            rectified_flow_loss=dit_backbone.rectified_flow_loss,
            pred=dit_backbone.pred,
            pred_rectified_flow=dit_backbone.pred_rectified_flow,
            sigma_data=dit_backbone.sigma_data

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

        return [dit_state, enc_state, dec_state]


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


# @jax.jit
def train_edm_based(dit_state, batch, train_rng):

    # always create new RNG!
    sample_rng, new_train_rng = jax.random.split(train_rng, num=2)
    
    # unpack

    def _compute_loss(
        dit_params, batch, rng_key
    ):
        dit_params, batch = mixed_precision_policy.cast_to_compute((dit_params, batch))
    
        images = batch["images"]
        class_cond = batch["labels"]
        n, h, w, c = images.shape      

        # Sample noise that we'll add to the images
        # I think I should combine this with the first noise seed generator
        noise_rng, timestep_rng = jax.random.split(
            key=rng_key, num=2
        )
        timesteps = rand_stratified_cosine(timestep_rng, n, dit_state.sigma_data)
        # rng_key, model_apply_fn, model_params, images, timesteps, conds, image_pos=None, extra_pos=None
        loss, debug_image = dit_state.loss(noise_rng, dit_state.apply_fn, dit_params, images, timesteps, class_cond)

        return mixed_precision_policy.cast_to_output(loss), debug_image

    grad_fn = jax.value_and_grad(
        fun=_compute_loss, argnums=[0,], has_aux=True  # differentiate first param only
    )

    (loss, debug_image), grad = grad_fn(
        dit_state.params,
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
        debug_image
    )


# @jax.jit
def train_flow_based(dit_state, batch, train_rng):

    # always create new RNG!
    sample_rng, new_train_rng = jax.random.split(train_rng, num=2)
    
    # unpack

    def _compute_loss(
        dit_params, batch, rng_key
    ):
        dit_params, batch = mixed_precision_policy.cast_to_compute((dit_params, batch))
    
        images = batch["images"]
        class_cond = batch["labels"]
        n, h, w, c = images.shape      

        # Sample noise that we'll add to the images
        # I think I should combine this with the first noise seed generator
        noise_rng, timestep_rng, cond = jax.random.split(
            key=rng_key, num=3
        )
        timesteps = jax.numpy.sort(jax.random.uniform(timestep_rng, [n])) 
        toggle_cond = jax.random.choice(cond,2,[n])
        # rng_key, model_apply_fn, model_params, images, timesteps, conds, image_pos=None, extra_pos=None
        loss, debug_image = dit_state.rectified_flow_loss(noise_rng, dit_state.apply_fn, dit_params, images, timesteps, class_cond, toggle_cond=toggle_cond)

        return mixed_precision_policy.cast_to_output(loss), debug_image

    grad_fn = jax.value_and_grad(
        fun=_compute_loss, argnums=[0,], has_aux=True  # differentiate first param only
    )

    (loss, debug_image), grad = grad_fn(
        dit_state.params,
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
        debug_image
    )


# @jax.jit
def edm_sampler(key,
    model, latents, class_labels=None,
    num_steps=200, sigma_min=0.001, sigma_max=1000, rho=7,
    S_churn=0, S_min=0, S_max=jnp.float32('inf'), S_noise=1,
):

    # Time step discretization.

    step_indices = jnp.arange(num_steps)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = jnp.concatenate([t_steps, jnp.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, jnp.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = t_cur + gamma * t_cur
        x_hat = x_cur + jnp.sqrt(t_hat ** 2 - t_cur ** 2) * S_noise * jax.random.normal(key=key, shape=x_cur.shape)

        # Euler step.
        denoised = model(images=x_hat, timesteps=t_hat[None], conds=class_labels)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = model(images=x_next, timesteps=t_next[None], conds=class_labels)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def euler_solver(func, init_cond, t_span, dt, conds=None, model_params=None,  model_apply_fn=None, cfg_scale=None):
    """
    Euler method solver for ODE: dZ/dt = v(Z, t)

    Parameters:
        func: Function representing dZ/dt = v(Z, t)
        Z0: Initial condition for Z
        t_span: Tuple (t0, tf) specifying initial and final time
        dt: Step size

    Returns:
        Z: Array of approximated solutions
        t: Array of time points
    """
    t0, tf = t_span
    num_steps = abs(int((tf - t0) / dt) + 1)  # Number of time steps
    t = jnp.linspace(t0, tf, num_steps)   # Time array
    Z = init_cond
    toggle_cond_on = jnp.ones_like(conds)
    toggle_cond_off = jnp.zeros_like(conds)

    # wraps model vector CFG
    def _func_cfg(init_cond, t, conds, model_params, model_apply_fn):
        cond_vector = func(init_cond, t, conds=conds, model_params=model_params,  model_apply_fn=model_apply_fn, toggle_cond=toggle_cond_on)
        uncond_vector = func(init_cond, t, conds=conds, model_params=model_params,  model_apply_fn=model_apply_fn, toggle_cond=toggle_cond_off)
        return uncond_vector + (cond_vector - uncond_vector) * cfg_scale


    # Euler method iteration
    for i in range(1, num_steps):
        Z = Z - _func_cfg(Z, t[i - 1][None], conds=conds, model_params=model_params,  model_apply_fn=model_apply_fn) * dt


    return Z


def main():
    BATCH_SIZE = 32
    SEED = 0
    SAVE_MODEL_PATH = "dit_ckpt"
    IMAGE_RES = 256
    LATENT_RATIO = 16
    SAVE_EVERY = 500
    LEARNING_RATE = 5e-4
    WANDB_PROJECT_NAME = "DiT"
    WANDB_RUN_NAME = "oxford_flowers"
    WANDB_LOG_INTERVAL = 100
    LATENT_BASED = True
    VAE_CKPT_STEPS = 401181
    VAE_CKPT = "vae_small_ckpt"
    LATENT_DEPTH = 64
    IMAGE_OUTPUT_PATH = "dit_output"
    PATCH_SIZE = 1
    IMAGE_PATH = "ramdisk/train_images"


    # Open the text file in read mode
    STEPS = 0
    
    dataset = SquareImageNetDataset(IMAGE_PATH, square_size=IMAGE_RES, seed=STEPS)
    try:
        while True:
            t_dl = threading_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_labeled_imagenet_fn,  num_workers=100, prefetch_factor=0.5, seed=SEED)
            # Initialize the progress bar
            progress_bar = tqdm(total=len(dataset) // BATCH_SIZE, position=0)

            for i, batch in enumerate(t_dl):
                batch_size,_,_,_ = batch["images"].shape

                if batch_size < BATCH_SIZE:
                    continue

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
                batch["og_images"] = batch["images"]

                print(batch["images"].shape)
                print(batch["images"].device())
                time.sleep(10)
                
                

       
                STEPS += 1

    except KeyboardInterrupt:
        pass

main()