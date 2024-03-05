import os
import random

import flax
import numpy as np
import jax
from jax.experimental.compilation_cache import compilation_cache as cc
import jax.numpy as jnp
from einops import rearrange
from flax.training import train_state
import optax
from lpips_j.lpips import LPIPS
import flax.linen as nn
import orbax.checkpoint
from tqdm import tqdm
import wandb
import jmp

from vae import Decoder, Encoder, Discriminator, UNetDiscriminator
from utils import FrozenModel, create_image_mosaic, flatten_dict, unflatten_dict
from streaming_dataloader import CustomDataset, threading_dataloader, collate_fn, ImageFolderDataset

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
    with jax.default_device(jax.devices("cpu")[0]):
        enc_rng, dec_rng, disc_rng, lpips_rng = jax.random.split(jax.random.PRNGKey(seed), 4)

        enc = Encoder(
            down_layer_contraction_factor = ( (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)),
            down_layer_dim = (128, 192, 256, 512, 1024),
            down_layer_kernel_size = ( 3, 3, 3, 3, 3),
            down_layer_blocks = (2, 2, 2, 2, 1),
            down_layer_ordinary_conv = (False, False, False, False, False),
            down_layer_residual = (True, True, True, True, True),
            use_bias = False,
            conv_expansion_factor = 4,
            eps = 1e-6,
            group_count = 16,
            last_layer = "conv",
        )
        dec = Decoder(
            output_features = 3,
            up_layer_contraction_factor = ( (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)),
            up_layer_dim = (1024, 512, 256, 192, 128),
            up_layer_kernel_size = ( 3, 3, 3, 3, 3),
            up_layer_blocks = (2, 2, 2, 2, 2),
            up_layer_ordinary_conv = (False, False, False, False, False),
            up_layer_residual = (True, True, True, True, True),
            use_bias = False,
            conv_expansion_factor = 4,
            eps = 1e-6,
            group_count = 16,
        )

        disc = UNetDiscriminator(
            input_features = 3,
            down_layer_contraction_factor = ((2, 2), (2, 2), (2, 2), (2, 2), (2, 2)),
            down_layer_dim = (32, 64, 96, 128, 192),
            down_layer_kernel_size = (3, 3, 3, 3, 3),
            down_layer_blocks = (2, 2, 2, 2, 1),
            down_layer_ordinary_conv = (False, False, False, False, False),
            down_layer_residual = (True, True, True, True, True),

            output_features = 3,
            up_layer_contraction_factor = ((2, 2), (2, 2), (2, 2), (2, 2), (2, 2)),
            up_layer_dim = (192, 128, 96, 64, 32),
            up_layer_kernel_size = (3, 3, 3, 3, 3),
            up_layer_blocks = (1, 2, 2, 2, 2),
            up_layer_ordinary_conv = (False, False, False, False, False),
            up_layer_residual = (True, True, True, True, True),

            use_bias = False,
            conv_expansion_factor = 4,
            eps = 1e-6,
            group_count = 16,

        )
        # disc = Discriminator(
        #     down_layer_contraction_factor = ( (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)),
        #     down_layer_dim = (128, 192, 256, 512, 1024),
        #     down_layer_kernel_size = ( 3, 3, 3, 3, 3),
        #     down_layer_blocks = (2, 2, 2, 2, 2),
        #     down_layer_ordinary_conv = (False, False, False, False, False),
        #     down_layer_residual = (True, True, True, True, True),
        #     use_bias = False,
        #     conv_expansion_factor = 4,
        #     eps = 1e-6,
        #     group_count = 16,
        # )
        lpips = LPIPS()

        # init model params
        image = jnp.ones((batch_size, training_res, training_res, 3))
        # create param for each model
        # encoder
        enc_params = enc.init(enc_rng, image)
        # decoder
        dummy_latent = enc.apply(enc_params, image)
        # TODO: replace this CPU forward with proper empty latent tensor
        dummy_latent_mean, dummy_latent_log_var = rearrange(dummy_latent, "n h w (c split) -> split n h w c", split=2)
        dec_params = dec.init(dec_rng, dummy_latent_mean)
        # discriminator
        disc_params = disc.init(disc_rng, image)
        # LPIPS VGG16
        lpips_params = lpips.init(lpips_rng, image, image)

        enc_param_count = sum(list(flatten_dict(jax.tree_map(lambda x: x.size, enc_params)).values()))
        dec_param_count = sum(list(flatten_dict(jax.tree_map(lambda x: x.size, dec_params)).values()))
        disc_param_count = sum(list(flatten_dict(jax.tree_map(lambda x: x.size, disc_params)).values()))
        print("encoder param count:", enc_param_count)
        print("decoder param count:", dec_param_count)
        print("discriminator param count:", disc_param_count)
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

        # trained

        # do not apply weight decay to norm layer
        enc_state = train_state.TrainState.create(
            apply_fn=enc.apply,
            params=enc_params,
            tx=adam_wrapper(
                jax.tree_util.tree_map_with_path(lambda path, var: path[-1].key != "scale", enc_params)
            ),
        )
        # trained
        dec_state = train_state.TrainState.create(
            apply_fn=dec.apply,
            params=dec_params,
            tx=adam_wrapper(
                jax.tree_util.tree_map_with_path(lambda path, var: path[-1].key != "scale", dec_params)
            ),
        )
        # trained
        disc_state = train_state.TrainState.create(
            apply_fn=disc.apply,
            params=disc_params,
            tx=adam_wrapper(
                jax.tree_util.tree_map_with_path(lambda path, var: path[-1].key != "scale", disc_params)
            ),
        )
        # frozen
        lpips_state = FrozenModel(
            call=lpips.apply,
            params=lpips_params,
        )
        
        # put everything in accelerator in data parallel mode
        enc_state = jax.tree_map(
            lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, PartitionSpec())),
            enc_state,
        )
        dec_state = jax.tree_map(
            lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, PartitionSpec())),
            dec_state,
        )
        disc_state = jax.tree_map(
            lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, PartitionSpec())),
            disc_state,
        )
        lpips_state = jax.tree_map(
            lambda leaf: jax.device_put(jmp.cast_to_half(leaf), device=NamedSharding(mesh, PartitionSpec())),
            lpips_state,
        )
        return [enc_state, dec_state, disc_state, lpips_state]

@jax.jit
def train_vae_only(models, batch, loss_scale, train_rng):
    # always create new RNG!
    vae_rng, new_train_rng = jax.random.split(train_rng, num=2)

    # unpack model
    enc_state, dec_state, lpips_state = models

    def _vae_loss(enc_params, dec_params, lpips_params, batch, loss_scale, vae_rng):
        # cast input
        enc_params, dec_params, lpips_params, batch = mixed_precision_policy.cast_to_compute((enc_params, dec_params, lpips_params, batch))
        latents = enc_state.apply_fn(enc_params, batch)

        # encoder is learning logvar instead of std 
        latent_mean, latent_logvar = rearrange(latents, "b h w (c split) -> split b h w c", split = 2)

        # KL loss
        kl_loss = 0.5 * jnp.sum(latent_mean**2 + jnp.exp(latent_logvar) - 1.0 - latent_logvar, axis=[1, 2, 3]).mean()  

        # reparameterization using logvar
        sample = latent_mean + jnp.exp(0.5 * latent_logvar) * jax.random.normal(vae_rng, latent_mean.shape)
        pred_batch = dec_state.apply_fn(dec_params, sample)

        # MSE loss
        mse_loss = ((batch - pred_batch) ** 2).mean()
        # lpips loss
        lpips_loss = lpips_state.call(lpips_params, batch, pred_batch).mean()

        vae_loss = mixed_precision_policy.cast_to_output(
            mse_loss * loss_scale["mse_loss_scale"] + 
            lpips_loss * loss_scale["lpips_loss_scale"] + 
            kl_loss * loss_scale["kl_loss_scale"]
        )

        vae_loss_stats = {
            "mse_loss": mse_loss * loss_scale["mse_loss_scale"],
            "lpips_loss": lpips_loss * loss_scale["lpips_loss_scale"],
            "kl_div_loss":  kl_loss * loss_scale["kl_loss_scale"]
        }
        return vae_loss, (vae_loss_stats, pred_batch)
    
    vae_loss_grad_fn = jax.value_and_grad(
        fun=_vae_loss, argnums=[0, 1], has_aux=True  # differentiate encoder and decoder
    )
   
    (vae_loss, (vae_loss_stats, pred_batch)), vae_grad = vae_loss_grad_fn(
        enc_state.params, 
        dec_state.params, 
        lpips_state.params, 
        batch, 
        loss_scale, 
        vae_rng
    )
    # update vae params
    new_enc_state = enc_state.apply_gradients(grads=jmp.cast_to_full(vae_grad[0]))
    new_dec_state = dec_state.apply_gradients(grads=jmp.cast_to_full(vae_grad[1]))

    

    # pack models 
    new_models_state = new_enc_state, new_dec_state, lpips_state

    loss_stats =  {"vae_loss": vae_loss}

    return new_models_state, new_train_rng, pred_batch, {**vae_loss_stats, **loss_stats}

@jax.jit
def train(models, batch, loss_scale, train_rng):
    # always create new RNG!
    vae_rng, new_train_rng = jax.random.split(train_rng, num=2)

    # unpack model
    enc_state, dec_state, disc_state, lpips_state = models

    def _vae_loss(enc_params, dec_params, disc_params, lpips_params, batch, loss_scale, vae_rng):
        # cast input
        enc_params, dec_params, disc_params, lpips_params, batch = mixed_precision_policy.cast_to_compute((enc_params, dec_params, disc_params, lpips_params, batch))
        latents = enc_state.apply_fn(enc_params, batch)

        # encoder is learning logvar instead of std 
        latent_mean, latent_logvar = rearrange(latents, "b h w (c split) -> split b h w c", split = 2)

        # KL encoder loss
        kl_loss = 0.5 * jnp.sum(latent_mean**2 + jnp.exp(latent_logvar) - 1.0 - latent_logvar, axis=[1, 2, 3]).mean()

        # reparameterization using logvar
        sample = latent_mean + jnp.exp(0.5 * latent_logvar) * jax.random.normal(vae_rng, latent_mean.shape)

        pred_batch = dec_state.apply_fn(dec_params, sample)

        # wraps some of the loss function because we want to compute the gradient of each 
        # we want a proportional scaling of the loss gradient from generator and discriminator.
        # by using the last decoder gradient on each loss we can compute the ratio and rebalance it.
        def _decoder_reconstruction_loss(
            batch,
            pred_batch
        ):
            # pixelwise loss
            mse_loss = ((batch - pred_batch) ** 2).mean() * loss_scale["mse_loss_scale"]
            mae_loss = (jnp.abs(batch - pred_batch)).mean() * loss_scale["mae_loss_scale"]
            # global lpips loss
            lpips_loss = lpips_state.call(lpips_params, batch, pred_batch).mean() * loss_scale["lpips_loss_scale"]
            # return a tuple of individual losses just for statistics
            return  mse_loss + mae_loss + lpips_loss, (mse_loss, mae_loss, lpips_loss)

        def _pixelwise_discriminator_critic(pred_batch):
            # disc loss
            pixelwise_critic = disc_state.apply_fn(disc_params, pred_batch)
            return nn.softplus(-pixelwise_critic).mean() * loss_scale["vae_disc_loss_scale"] 
        
        def adaptive_disc_scale(sample):
            flattened_dec_params = flatten_dict(dec_params)
            last_layer_dec_params = flattened_dec_params.pop("params.final_conv.kernel")

            # do decoder forward but split the last layer params as separate args 
            # this is usefull to calculate gradient proportion of discriminator later
            # by doing this jax autograd can compute the gradient with respect to that last layer
            # jax tracer probably smart enough and fuse this 3 forward pass (2 here and 1 above) into 1
            def _decoder_apply(
                last_layer_params,
                remainder_params,
                sample: jax.Array
            ) -> jax.Array:
                # merge back the params 
                remainder_params["params.final_conv.kernel"] = last_layer_params
                params = unflatten_dict(remainder_params)
                return dec_state.apply_fn(params, sample)

            def _vae_rec_loss_contrb(
                last_layer_dec_params, # <<< we compute grad with respect to this variable only
                remainder_dec_params,
                sample,
                batch
            ):
                pred_batch = _decoder_apply(last_layer_dec_params, remainder_dec_params, sample)
                return _decoder_reconstruction_loss(batch, pred_batch)
            
            
            def _disc_loss_contrb(
                last_layer_dec_params, # <<< we compute grad with respect to this variable only
                remainder_dec_params,
                sample,
            ) -> jax.Array:
                pred_batch = _decoder_apply(last_layer_dec_params, remainder_dec_params, sample)
                return _pixelwise_discriminator_critic(pred_batch)

            # get the gradient matrix
            _vae_rec_loss_contrb_grad, _ = jax.grad(_vae_rec_loss_contrb, argnums=[0], has_aux=True)(last_layer_dec_params, flattened_dec_params, sample, batch)
            _disc_loss_contrb_grad = jax.grad(_disc_loss_contrb, argnums=[0])(last_layer_dec_params, flattened_dec_params, sample)

            # any scaling method will do here 
            d_weight = jnp.linalg.norm(_vae_rec_loss_contrb_grad[0]) / (jnp.linalg.norm(_disc_loss_contrb_grad[0]) + 1e-4)
            d_weight = jnp.clip(d_weight, 0.0, 1e4)

            # prevent gradient to propagate here because we only want a scale 
            return jax.lax.stop_gradient(d_weight)


        # compute the loss as usual
        dec_loss, (mse_loss, mae_loss, lpips_loss)  = _decoder_reconstruction_loss(batch, pred_batch)
        disc_loss = _pixelwise_discriminator_critic(pred_batch)

        disc_scale = adaptive_disc_scale(sample)
        
        vae_loss = mixed_precision_policy.cast_to_output(
            dec_loss + # already scaled on each
            kl_loss * loss_scale["kl_loss_scale"] +
            disc_loss * disc_scale * loss_scale["toggle_gan"]
        )
        
        vae_loss_stats = {
            "mse_loss": mse_loss,
            "mae_loss": mae_loss,
            "lpips_loss": lpips_loss,
            "critic_loss": disc_loss * disc_scale * loss_scale["toggle_gan"],
            "kl_divergence_loss":  kl_loss * loss_scale["kl_loss_scale"]
        }
        return vae_loss, (vae_loss_stats, pred_batch)
    
    def _disc_loss(disc_params, batch, reconstructed_batch, loss_scale):
        # cast input
        disc_params, batch, reconstructed_batch = mixed_precision_policy.cast_to_compute((disc_params,  batch, reconstructed_batch))
        # wasserstein GAN loss
        disc_fake_scores = disc_state.apply_fn(disc_params, reconstructed_batch)
        disc_real_scores = disc_state.apply_fn(disc_state.params, batch)
        loss_fake = nn.softplus(disc_fake_scores)
        loss_real = nn.softplus(-disc_real_scores)
        wgan_loss = jnp.mean(loss_fake + loss_real)

        def _disc_gradient_penalty(disc_state, batch):
            # a regularization based on real image
            return disc_state.apply_fn(disc_state.params, batch).mean()
        
        regularization = (jax.grad(_disc_gradient_penalty, argnums=[1])(disc_state, batch)[0] ** 2).mean()
        # wgan
        disc_loss = mixed_precision_policy.cast_to_output((wgan_loss +  regularization * loss_scale["reg_1_scale"]))


        # just for monitoring
        disc_loss_stats = {
            "prob_prediction_is_real": jnp.exp(-loss_real).mean(),  # p = 1 -> predict real is real
            "prob_prediction_is_fake": jnp.exp(-loss_fake).mean(),  # p = 1 -> predict fake is fake
            "loss_real": loss_real.mean(),
            "loss_fake": loss_fake.mean(),
            "discriminator_training_loss": disc_loss,
        }        
        return disc_loss, disc_loss_stats

    vae_loss_grad_fn = jax.value_and_grad(
        fun=_vae_loss, argnums=[0, 1], has_aux=True  # differentiate encoder and decoder
    )
    disc_loss_grad_fn = jax.value_and_grad(
        fun=_disc_loss, argnums=[0], has_aux=True  # differentiate discriminator
    )

    (vae_loss, (vae_loss_stats, pred_batch)), vae_grad = vae_loss_grad_fn(
        enc_state.params, 
        dec_state.params, 
        disc_state.params, 
        lpips_state.params, 
        batch, 
        loss_scale, 
        vae_rng
    )
    (disc_loss, disc_loss_stats), disc_grad = disc_loss_grad_fn(
        disc_state.params, 
        batch, 
        pred_batch,
        loss_scale, 
    )

    # update vae params
    new_enc_state = enc_state.apply_gradients(grads=jmp.cast_to_full(vae_grad[0]))
    new_dec_state = dec_state.apply_gradients(grads=jmp.cast_to_full(vae_grad[1]))

    # update discriminator params
    # disc_grad = jax.tree_map(lambda x: x * loss_scale["toggle_gan"], disc_grad)
    new_disc_state = disc_state.apply_gradients(grads=jmp.cast_to_full(disc_grad[0]))

    # pack models 
    new_models_state = new_enc_state, new_dec_state, new_disc_state, lpips_state

    loss_stats =  {"vae_loss": vae_loss, "disc_loss": disc_loss}

    return new_models_state, new_train_rng, pred_batch, {**disc_loss_stats, **vae_loss_stats, **loss_stats}


def main():
    BATCH_SIZE = 64
    SEED = 0
    URL_TXT = "datacomp_1b.txt"
    SAVE_MODEL_PATH = "orbax_ckpt"
    IMAGE_RES = 256
    SAVE_EVERY = 500
    LEARNING_RATE = 5e-5
    LOSS_SCALE = {
        "mse_loss_scale": 0.0,
        "mae_loss_scale": 1,
        "lpips_loss_scale": 0.25,
        "kl_loss_scale": 1e-6,
        "vae_disc_loss_scale": 0.25,
        "reg_1_scale": 1e7,
        "toggle_gan": 1
    }
    GAN_TRAINING_START= 0
    NO_GAN = False
    WANDB_PROJECT_NAME = "vae"
    WANDB_RUN_NAME = "kl[1e-6]_lpips[0.25]_mse[0]_mae[1]_disc[0.5]_reg[1e7]_gan[pixel]_lr[5e-5]_b1[0.5]_b2[0.9]_imagenet-1k"
    WANDB_LOG_INTERVAL = 100

    # wandb logging
    if WANDB_PROJECT_NAME:
        wandb.init(project=WANDB_PROJECT_NAME, name=WANDB_RUN_NAME)

    # init seed
    train_rng = jax.random.PRNGKey(SEED)
    # init checkpoint manager
    ckpt_manager = checkpoint_manager(SAVE_MODEL_PATH)
    # init model
    models = init_model(batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
    # remove gan discriminator params if VAE only training
    if NO_GAN:
        disc_params = models.pop(2)
        del disc_params

    # Open the text file in read mode
    with open(URL_TXT, 'r') as file:
        # Read the lines of the file and store them in a list
        parquet_urls = file.readlines()

    # Remove newline characters from each line and create a list
    parquet_urls = [parquet_url.strip() for parquet_url in parquet_urls]
    parquet_urls = ["ramdisk/train_images"] * 10

    def rando_colours(IMAGE_RES):
        
        max_colour = np.full([1, IMAGE_RES, IMAGE_RES, 1], 255)
        min_colour = np.zeros((1, IMAGE_RES, IMAGE_RES, 1))

        black = np.concatenate([min_colour,min_colour,min_colour],axis=-1) / 255 * 2 - 1 
        white = np.concatenate([max_colour,max_colour,max_colour],axis=-1) / 255 * 2 - 1 
        red = np.concatenate([max_colour,min_colour,min_colour],axis=-1) / 255 * 2 - 1 
        green = np.concatenate([min_colour,max_colour,min_colour],axis=-1) / 255 * 2 - 1 
        blue = np.concatenate([min_colour,min_colour,max_colour],axis=-1) / 255 * 2 - 1 
        magenta = np.concatenate([max_colour,min_colour,max_colour],axis=-1) / 255 * 2 - 1 
        cyan = np.concatenate([min_colour,max_colour,max_colour],axis=-1) / 255 * 2 - 1 
        yellow = np.concatenate([max_colour,max_colour,min_colour],axis=-1) / 255 * 2 - 1 

        r = np.random.randint(0, 255) * np.ones((1, IMAGE_RES, IMAGE_RES, 1))
        g = np.random.randint(0, 255) * np.ones((1, IMAGE_RES, IMAGE_RES, 1))
        b = np.random.randint(0, 255) * np.ones((1, IMAGE_RES, IMAGE_RES, 1))
        rando_colour = np.concatenate([r,g,b],axis=-1) / 255 * 2 - 1 

        pallete = [black, white, red, green, blue, magenta, cyan, yellow, rando_colour]

        return random.choice(pallete)

        

    STEPS = 0
    _gan_start = 0
    try:
        for parquet_url in parquet_urls:
            # dataset = CustomDataset(parquet_url, square_size=IMAGE_RES)
            dataset = ImageFolderDataset(parquet_url, square_size=IMAGE_RES, seed=STEPS)
            t_dl = threading_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,  num_workers=100, prefetch_factor=1, seed=SEED)
            # Initialize the progress bar
            progress_bar = tqdm(total=len(dataset) // BATCH_SIZE, position=0)

            for i, batch in enumerate(t_dl):
                STEPS += 1
                if i > len(dataset) // BATCH_SIZE -10:
                    break
                
                _gan_start += 1

                progress_bar.set_description(f"steps until gan training: {_gan_start}")

                if _gan_start > GAN_TRAINING_START:
                    print("GAN TRAINING MODE START NOW")
                    LOSS_SCALE["toggle_gan"] = 1
                else:
                    LOSS_SCALE["toggle_gan"] = 0 

                batch = batch / 255 * 2 - 1

                batch[0] = rando_colours(IMAGE_RES)

                batch = jax.tree_map(
                    lambda leaf: jax.device_put(
                        leaf, device=NamedSharding(mesh, PartitionSpec("data_parallel", None, None, None))
                    ),
                    batch,
                )

                if NO_GAN:
                    # new_models_state, new_train_rng, pred_batch, (vae_loss, vae_loss_stats, disc_loss, disc_loss_stats)
                    models, train_rng, output, stats = train_vae_only(models, batch, LOSS_SCALE, train_rng)
                else:
                    models, train_rng, output, stats = train(models, batch, LOSS_SCALE, train_rng)


                if i % WANDB_LOG_INTERVAL == 0:
                    wandb.log(stats, step=STEPS)
                    stats_rounded = {key: round(value, 3) for (key, value) in stats.items()}
                    # progress_bar.set_description(f"{stats_rounded}")


                # save every n steps
                if i % SAVE_EVERY == 0:
                    preview = jnp.concatenate([batch[:4], output[:4]], axis = 0)
                    preview = np.array((preview + 1) / 2 * 255, dtype=np.uint8)

                    create_image_mosaic(preview, 2, len(preview)//2, f"{STEPS}.png")
                    wandb.log({"image": wandb.Image(f'{STEPS}.png')}, step=STEPS)

                    ckpt_manager.save(i, models)


                progress_bar.update(1)
    except KeyboardInterrupt:
        i = -1
        print("Ctrl+C command detected. saving model before exiting...")
        ckpt_manager.save(i, models)

main()
