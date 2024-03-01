import os

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

from vae import Decoder, Encoder, Discriminator
from utils import FrozenModel, create_image_mosaic
from streaming_dataloader import CustomDataset, threading_dataloader, collate_fn

# sharding
from jax.sharding import Mesh
from jax.sharding import PartitionSpec
from jax.sharding import NamedSharding
from jax.experimental import mesh_utils


# adjust this sharding mesh to create appropriate sharding rule
devices = mesh_utils.create_device_mesh((jax.device_count(), 1))
# create axis name on how many parallelism slice you want on your model
mesh = Mesh(devices, axis_names=("data_parallel", "model_parallel"))

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
        disc = Discriminator(
            down_layer_contraction_factor = ( (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)),
            down_layer_dim = (128, 192, 256, 512, 1024),
            down_layer_kernel_size = ( 3, 3, 3, 3, 3),
            down_layer_blocks = (2, 2, 2, 2, 2),
            down_layer_ordinary_conv = (False, False, False, False, False),
            down_layer_residual = (True, True, True, True, True),
            use_bias = False,
            conv_expansion_factor = 4,
            eps = 1e-6,
            group_count = 16,
        )
        lpips = LPIPS()

        # init model params
        image = jnp.ones((batch_size, training_res, training_res, 3))
        # create param for each model
        # encoder
        enc_params = enc.init(enc_rng, image)
        # decoder
        dummy_latent = enc.apply(enc_params, image)
        dummy_latent_mean, dummy_latent_log_var = rearrange(dummy_latent, "n h w (c split) -> split n h w c", split=2)
        dec_params = dec.init(dec_rng, dummy_latent_mean)
        # discriminator
        disc_params = disc.init(disc_rng, image)
        # LPIPS VGG16
        lpips_params = lpips.init(lpips_rng, image, image)

        # create callable optimizer chain
        constant_scheduler = optax.constant_schedule(learning_rate)
        adamw = optax.adamw(
            learning_rate=constant_scheduler,
            b1=0.9,
            b2=0.999,
            eps=1e-08,
            weight_decay=1e-2,
        )
        u_net_optimizer = optax.chain(
            optax.clip_by_global_norm(1),  # prevent explosion
            adamw,
        )

        # trained
        enc_state = train_state.TrainState.create(
            apply_fn=enc.apply,
            params=enc_params,
            tx=u_net_optimizer,
        )
        # trained
        dec_state = train_state.TrainState.create(
            apply_fn=dec.apply,
            params=dec_params,
            tx=u_net_optimizer,
        )
        # trained
        disc_state = train_state.TrainState.create(
            apply_fn=disc.apply,
            params=disc_params,
            tx=u_net_optimizer,
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
            lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, PartitionSpec())),
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

        vae_loss = (
            mse_loss * loss_scale["mse_loss_scale"] + 
            lpips_loss * loss_scale["lpips_loss_scale"] + 
            kl_loss * loss_scale["kl_loss_scale"]
        )

        vae_loss_stats = {
            "pixelwise_reconstruction_loss": mse_loss * loss_scale["mse_loss_scale"],
            "lpips_loss": lpips_loss * loss_scale["lpips_loss_scale"],
            "kl_divergence_loss":  kl_loss * loss_scale["kl_loss_scale"]
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
    new_enc_state = enc_state.apply_gradients(grads=vae_grad[0])
    new_dec_state = dec_state.apply_gradients(grads=vae_grad[1])


    # pack models 
    new_models_state = new_enc_state, new_dec_state, lpips_state

    loss_stats =  {"vae_loss": vae_loss}

    return new_models_state, new_train_rng, pred_batch, {**vae_loss_stats, **loss_stats}


def train(models, batch, loss_scale, train_rng):
    # always create new RNG!
    vae_rng, new_train_rng = jax.random.split(train_rng, num=2)

    # unpack model
    enc_state, dec_state, disc_state, lpips_state = models

    def _vae_loss(enc_params, dec_params, disc_params, lpips_params, batch, loss_scale, vae_rng):
        latents = enc_state.apply_fn(enc_params, batch)

        # encoder is learning logvar instead of std 
        latent_mean, latent_logvar = rearrange(latents, "b h w (c split) -> split b h w c", split = 2)

        # KL loss
        kl_loss = 0.5 * jnp.sum(latent_mean**2 + jnp.exp(latent_logvar) - 1.0 - latent_logvar, axis=[1, 2, 3])      

        # reparameterization using logvar
        sample = latent_mean + jnp.exp(0.5 * latent_logvar) * jax.random.normal(vae_rng, latent_mean.shape)
        pred_batch = dec_state.apply_fn(dec_params, sample)

        # MSE loss
        mse_loss = ((batch - pred_batch) ** 2).mean()
        # lpips loss
        lpips_loss = lpips_state.call(lpips_params, batch, pred_batch)
        # disc loss (frozen)
        disc_fake_scores = disc_state.apply_fn(disc_params, pred_batch)
        disc_loss = nn.softplus(-disc_fake_scores)

        vae_loss = (
            mse_loss * loss_scale["mse_loss_scale"] + 
            lpips_loss * loss_scale["lpips_loss_scale"] + 
            disc_loss * loss_scale["vae_disc_loss_scale"] +
            kl_loss * loss_scale["kl_loss_scale"]
        )

        vae_loss_stats = {
            "pixelwise_reconstruction_loss": mse_loss * loss_scale["mse_loss_scale"],
            "lpips_loss": lpips_loss * loss_scale["lpips_loss_scale"],
            "discriminator_loss": disc_loss * loss_scale["vae_disc_loss_scale"],
            "kl_divergence_loss":  kl_loss * loss_scale["kl_loss_scale"]
        }
        return vae_loss, (vae_loss_stats, pred_batch)
    
    def _disc_loss(disc_state, batch, reconstructed_batch, loss_scale):
        # wasserstein GAN loss
        disc_fake_scores = disc_state.apply_fn(disc_state.params, reconstructed_batch)
        disc_real_scores = disc_state.apply_fn(disc_state.params, batch)
        loss_fake = nn.softplus(disc_fake_scores)
        loss_real = nn.softplus(-disc_real_scores)
        wgan_loss = jnp.mean(loss_fake + loss_real)

        def _disc_gradient_penalty(disc_state, batch):
            # a regularization based on real image
            return disc_state.apply_fn(disc_state.params, batch).mean()
        
        regularization = jax.grad(_disc_gradient_penalty, argnums=[1])(disc_state, batch) ** 2
        # wgan
        disc_loss = wgan_loss +  regularization * loss_scale["reg_1_scale"]


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
    new_enc_state = enc_state.apply_gradients(vae_grad[0])
    new_dec_state = dec_state.apply_gradients(vae_grad[1])

    # update discriminator params
    new_disc_state = disc_state.apply_gradients(disc_grad[0])

    # pack models 
    new_models_state = new_enc_state, new_dec_state, new_disc_state, lpips_state

    loss_stats =  {"vae_loss": vae_loss, "disc_loss": disc_loss}

    return new_models_state, new_train_rng, pred_batch, {**disc_loss_stats, **vae_loss_stats, **loss_stats}


def main():
    BATCH_SIZE = 16
    SEED = 0
    URL_TXT = "datacomp_1b.txt"
    SAVE_MODEL_PATH = "orbax_ckpt"
    IMAGE_RES = 256
    SAVE_EVERY = 500
    LEARNING_RATE = 1e-4
    LOSS_SCALE = {
        "mse_loss_scale": 1.0,
        "lpips_loss_scale": 1.0,
        "kl_loss_scale": 2e-5,
        "vae_disc_loss_scale": 0.5,
        "reg_1_scale": 1e4
    }
    NO_GAN = True
    WANDB_PROJECT_NAME = "vae"
    WANDB_RUN_NAME = "testing"
    WANDB_LOG_INTERVAL = 100
    JIT = True

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

    for parquet_url in parquet_urls:
        dataset = CustomDataset(parquet_url, square_size=IMAGE_RES)
        t_dl = threading_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,  num_workers=100, prefetch_factor=1, seed=SEED)
        # Initialize the progress bar
        progress_bar = tqdm(total=len(dataset), position=0)

        for i, batch in enumerate(t_dl):

            batch = batch / 255 * 2 - 1

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
                wandb.log(stats)


            # save every n steps
            if i % SAVE_EVERY == 0:
                preview = jnp.concatenate([batch, output], axis = 1)
                preview = np.array(((preview / 255 * 2 - 1) + 1) / 2 * 255, dtype=np.uint8)

                create_image_mosaic(preview, len(preview), 2, f"{i}.png")
                wandb.log({"image": wandb.Image(f'{i}.png')})

                ckpt_manager.save(i, models)

            progress_bar.set_description(f'Loss: {stats}')
            progress_bar.update(1)

main()