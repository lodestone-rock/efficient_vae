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

from cascade import DecoderStageA, EncoderStageA
from utils import FrozenModel, create_image_mosaic, flatten_dict, unflatten_dict
from streaming_dataloader import CustomDataset, threading_dataloader, collate_fn, ImageFolderDataset


cc.initialize_cache("jax_cache")


# offload this into json

enc = EncoderStageA(
    first_layer_output_features = 24,
    output_features = 4,
    down_layer_dim = (48, 96),
    down_layer_kernel_size = (3, 3),
    down_layer_blocks = (8, 8),
    # down_layer_ordinary_conv = (True, True),
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
    # up_layer_ordinary_conv = (True, True) ,
    use_bias = False ,
    conv_expansion_factor = (4, 4),
    eps = 1e-6,
    group_count = (-1, -1)
)
training_res = 256
compression_ratio = 4
latent_dim = 4



model_path = "stage_a_training_history/119092"

# init model params
image = jnp.ones((1, training_res, training_res, 3))
# encoder
enc_params = enc.init(jax.random.PRNGKey(0), image)
# decoder
dummy_latent = jnp.ones((1, training_res // compression_ratio, training_res // compression_ratio, latent_dim))
dec_params = dec.init(jax.random.PRNGKey(0), dummy_latent)

enc_params_flatten = unflatten_dict(load_file(f"{model_path}/enc_params.safetensors"))
dec_params_flatten = unflatten_dict(load_file(f"{model_path}/dec_params.safetensors"))

image_path = 'a70460fe4349ffc329aca41e45a91e5e.jpg'
image = cv2.imread(image_path) / 255 * 2 - 1

latents = jax.jit(enc.apply)(enc_params_flatten, image[None,...])
image_logits = jax.jit(dec.apply)(dec_params_flatten, latents)

preview = np.array((image_logits + 1) / 2 * 255, dtype=np.uint8)
cv2.imwrite("test2.png", preview[0])

print()