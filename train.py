import flax
import numpy as np
import jax
import jax.numpy as jnp
from vae import Decoder, Encoder, Discriminator
from einops import rearrange
from flax.training import train_state
import optax
from lpips_j.lpips import LPIPS
from utils import FrozenModel
import flax.linen as nn

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

        return enc_state, dec_state, disc_state, lpips_state


def train(models, batch, loss_scale, train_rng):
    # always create new RNG!
    vae_rng, new_train_rng = jax.random.split(train_rng, num=2)

    # unpack model
    enc_state, dec_state, disc_state, lpips_state = models

    def _vae_loss(enc_state, dec_state, disc_state, lpips_state, batch, loss_scale, vae_rng):
        latents = enc_state.apply(enc_state.params, batch)

        # encoder is learning logvar instead of std 
        latent_mean, latent_logvar = rearrange(latents, "b h w (c split) -> split b h w c", split = 2)

        # KL loss
        kl_loss = 0.5 * jnp.sum(latent_mean**2 + jnp.exp(latent_logvar) - 1.0 - latent_logvar, axis=[1, 2, 3])      

        # reparameterization using logvar
        sample = latent_mean + jnp.exp(0.5 * latent_logvar) * jax.random.normal(vae_rng, latent_mean.shape)
        pred_batch = dec_state.apply(dec_state.params, sample)

        # MSE loss
        mse_loss = ((batch - pred_batch) ** 2).mean()
        # lpips loss
        lpips_loss = lpips_state.call(lpips_state.params, batch, pred_batch)
        # disc loss (frozen)
        disc_fake_scores = disc_state.apply(disc_state.params, pred_batch)
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
        return vae_loss, vae_loss_stats
    
    def _disc_loss(disc_state, batch, reconstructed_batch):
        # GAN loss
        disc_fake_scores = disc_state.apply(disc_state.params, reconstructed_batch)
        disc_real_scores = disc_state.apply(disc_state.params, batch)
        loss_fake = nn.softplus(disc_fake_scores)
        loss_real = nn.softplus(-disc_real_scores)
        disc_loss = jnp.mean(loss_fake + loss_real)
        # just for monitoring
        disc_loss_stats = {
            "prob_prediction_is_real": jnp.exp(-loss_real).mean(),  # p = 1 -> predict real is real
            "prob_prediction_is_fake": jnp.exp(-loss_fake).mean(),  # p = 1 -> predict fake is fake
            "loss_real": loss_real.mean(),
            "loss_fake": loss_fake.mean(),
            "discriminator_training_loss": disc_loss,
        }
        return disc_loss, disc_loss_stats
    
    def _disc_gradient_penalty(disc_state, batch):
        # a scaling penalty for discriminator training loss
        return disc_state.apply(disc_state.params, batch).mean()


    vae_loss_grad_fn = jax.value_and_grad(
        fun=_vae_loss, argnums=[0, 1]  # differentiate encoder and decoder
    )
    disc_loss_grad_fn = jax.value_and_grad(
        fun=_disc_loss, argnums=[0]  # differentiate discriminator
    )
    disc_gradient_penalty_grad_fn = jax.value_and_grad(
        fun=_disc_gradient_penalty, argnums=[1]  # differentiate input
    )



models = init_model(batch_size=4)
print()