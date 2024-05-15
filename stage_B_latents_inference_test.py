import os
import random

import flax
import numpy as np
import jax
from safetensors.numpy import save_file, load_file
from jax.experimental.compilation_cache import compilation_cache as cc
import jax.numpy as jnp
from einops import rearrange

import flax.linen as nn
from tqdm import tqdm
import wandb
import cv2
import dm_pix as pix

from cascade import DecoderStageA, EncoderStageA, UNetStageB
from utils import FrozenModel, create_image_mosaic, flatten_dict, unflatten_dict
from streaming_dataloader import CustomDataset, threading_dataloader, collate_fn, ImageFolderDataset, scale


cc.initialize_cache("jax_cache")


training_res = 256
compression_ratio = 4
latent_dim = 4

# offload this into json

enc = EncoderStageA(
    first_layer_output_features = 24,
    output_features = 4,
    down_layer_dim = (48, 96),
    down_layer_kernel_size = (3, 3),
    down_layer_blocks = (8, 8),
    use_bias = False ,
    conv_expansion_factor = (4, 4),
    eps = 1e-6,
    group_count = (-1, -1)


)
dec = DecoderStageA(
    last_upsample_layer_output_features = 24,
    output_features = 3,
    up_layer_dim = (96, 48),
    up_layer_kernel_size = (3, 3),
    up_layer_blocks = (8, 8),
    use_bias = False ,
    conv_expansion_factor = (4, 4),
    eps = 1e-6,
    group_count = (-1, -1)
)

controlnet = EncoderStageA(
    first_layer_output_features = 128,
    output_features = 128,
    down_layer_dim = (128,),
    down_layer_kernel_size = (3,),
    down_layer_blocks = (15,),
    use_bias = False,
    conv_expansion_factor = (4,),
    eps = 1e-6,
    group_count = (-1,),
    downscale = False
)

unet = UNetStageB(
    down_layer_dim = (128, 192, 256),
    down_layer_kernel_size = (3, 3, 3, 3),
    down_layer_blocks = (8, 12, 14),
    down_group_count = (-1, -1, -1),
    down_conv_expansion_factor = (4, 4, 4),

    up_layer_dim = (256, 192, 128),
    up_layer_kernel_size = (3, 3, 3),
    up_layer_blocks = (14, 12, 8),
    up_group_count = (-1, -1, -1),
    up_conv_expansion_factor = (4, 4, 4),

    output_features = latent_dim,
    use_bias = False,
    timestep_dim = 320,
    eps = 1e-6,
    checkpoint=True
)

stage_a_path = "stage_a_safetensors/119092"
stage_b_path = "stage_b_safetensors/100000"
# init model params
# encoder
# image = jnp.ones((1, training_res, training_res, 3))
# enc_params = enc.init(jax.random.PRNGKey(0), image)
enc_params = unflatten_dict(load_file(f"{stage_a_path}/enc_params.safetensors"))
# decoder
# dummy_latent = jnp.ones((1, training_res // compression_ratio, training_res // compression_ratio, latent_dim))
# dec_params = dec.init(jax.random.PRNGKey(0), dummy_latent)
dec_params = unflatten_dict(load_file(f"{stage_a_path}/dec_params.safetensors"))

controlnet_params = unflatten_dict(load_file(f"{stage_b_path}/controlnet_params.safetensors"))
unet_params = unflatten_dict(load_file(f"{stage_b_path}/unet_params.safetensors"))


unet_state = FrozenModel(
    call=unet.apply,
    params=jax.device_put(unet_params, device=jax.devices()[0]),
)
controlnet_state = FrozenModel(
    call=controlnet.apply,
    params=jax.device_put(controlnet_params, device=jax.devices()[0]),
)

enc_state = FrozenModel(
    call=enc.apply,
    params=jax.device_put(enc_params, device=jax.devices()[0]),
)

dec_state = FrozenModel(
    call=dec.apply,
    params=jax.device_put(dec_params, device=jax.devices()[0]),
)



def euler_solver(init_cond, t_span, dt, model_state=None, conds=None, cfg_scale=None):
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

    # simple wrapper to make less cluttered on ODE loop
    def _func_cfg(init_cond, t):
        # UNJITTED
        cond_vector = jax.jit(model_state.call)(model_state.params, init_cond, t, conds)
        uncond_vector = jax.jit(model_state.call)(model_state.params, init_cond, t, conds*0)
        # cond_vector = model_state.apply_fn(model_state.params, init_cond, t, conds)
        # uncond_vector = model_state.apply_fn(model_state.params, init_cond, t, conds*0)
        return uncond_vector + (cond_vector - uncond_vector) * cfg_scale 

    # Euler method iteration
    for i in range(1, num_steps):
        Z = Z - _func_cfg(Z, jnp.stack([t[i - 1][None]]*Z.shape[0])) * dt

    return Z

def inference(unet_state, controlnet_state, enc_state, dec_state, batch, stage_a_compression_ratio, stage_a_latent_size, seed, t_span, dt, cfg_scale=1):

    n, h, w, c = batch.shape
    small_latents = jax.jit(enc_state.call)(enc_state.params, batch)
    n, h, w, c = small_latents.shape
    # initial noise + low res latent
    # UNJITTED
    cond_latents = jax.jit(controlnet_state.call)(controlnet_state.params, small_latents)
    # cond_latents = controlnet_state.apply_fn(controlnet_state.params, small_latents)
    init_cond = jax.random.normal(key=jax.random.PRNGKey(seed), shape=(n, h*stage_a_compression_ratio, w*stage_a_compression_ratio, stage_a_latent_size))

    # solve the model
    latents = euler_solver(init_cond, t_span, dt, model_state=unet_state, conds=cond_latents, cfg_scale=cfg_scale)

    # convert back to pixel space
    logits = jax.jit(dec_state.call)(dec_state.params, latents)
    
    
    return logits

image_path = 'a70460fe4349ffc329aca41e45a91e5e.jpg'
image_path = "thispersondoesnotexist.jpeg"
image = cv2.imread(image_path) / 255 * 2 - 1
image = scale(image, 0.125)


preview = inference(unet_state, controlnet_state, enc_state, dec_state, image[None, ...], 4, 4, 0, (1, 0.00001), 0.01, 2)

preview = np.array((preview + 1) / 2 * 255, dtype=np.uint8)
cv2.imwrite("test2.png", preview[0])

print()