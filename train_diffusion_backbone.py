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
import flax.linen as nn
import orbax.checkpoint
from tqdm import tqdm
import wandb
import jmp
import cv2

from vae import Decoder, Encoder
from utils import FrozenModel, create_image_mosaic, flatten_dict, unflatten_dict
from streaming_dataloader import CustomDataset, threading_dataloader, collate_fn, SquareImageDataset

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


        enc_param_count = sum(list(flatten_dict(jax.tree_map(lambda x: x.size, enc_params)).values()))
        dec_param_count = sum(list(flatten_dict(jax.tree_map(lambda x: x.size, dec_params)).values()))
        print("encoder param count:", enc_param_count)
        print("decoder param count:", dec_param_count)
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

        # frozen

        # do not apply weight decay to norm layer
        enc_state = FrozenModel(
            call=enc.apply,
            params=enc_params,
        )
        # frozen
        dec_state = train_state.TrainState.create(
            apply_fn=dec.apply,
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
        return [enc_state, dec_state]



def main():
    BATCH_SIZE = 128
    SEED = 0
    URL_TXT = "datacomp_1b.txt"
    SAVE_MODEL_PATH = "vae_ckpt"
    IMAGE_RES = 256
    SAVE_EVERY = 500
    LEARNING_RATE = 2e-4
    LOSS_SCALE = {
        "mse_loss_scale": 1,
        "mae_loss_scale": 0,
        "lpips_loss_scale": 0.25,
        "kl_loss_scale": 1e-6,
        "vae_disc_loss_scale": 0.0,
        "reg_1_scale": 0,
        "toggle_gan": 0
    }
    GAN_TRAINING_START= 0
    NO_GAN = True
    WANDB_PROJECT_NAME = "vae"
    WANDB_RUN_NAME = "final"#"kl[1e-6]_lpips[0.25]_mse[1]_mae[0]_lr[1e-4]_b1[0.5]_b2[0.9]_gn[32]_c[768]_imagenet-1k"
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


        absolute = [black, white] * 4
        pallete = [red, green, blue, magenta, cyan, yellow, rando_colour] + absolute

        return random.choice(pallete)

    sample_image = np.concatenate([cv2.imread(f"sample_{x}.jpg")[None, ...] for x in range(4)] * int(BATCH_SIZE//4), axis=0) / 255 * 2 - 1

    STEPS = 0
    _gan_start = 0
    try:
        for parquet_url in parquet_urls:
            # dataset = CustomDataset(parquet_url, square_size=IMAGE_RES)
            dataset = ImageFolderDataset(parquet_url, square_size=IMAGE_RES, seed=STEPS)
            t_dl = threading_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,  num_workers=100, prefetch_factor=3, seed=SEED)
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
                    LEARNING_RATE = 1e-6
                    if LOSS_SCALE["toggle_gan"] < 1:
                        LOSS_SCALE["toggle_gan"] += 0.00001
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

                    preview_test = inference(models, sample_image)
                    preview = jnp.concatenate([batch[:4], output[:4], preview_test[:4]], axis = 0)
                    preview = np.array((preview + 1) / 2 * 255, dtype=np.uint8)

                    create_image_mosaic(preview, 3, len(preview)//3, f"{STEPS}.png")
                    # progress_bar.set_description(f"{stats_rounded}")


                # save every n steps
                if i % SAVE_EVERY == 0:
                    preview_test = inference(models, sample_image)
                    preview = jnp.concatenate([batch[:4], output[:4], preview_test[:4]], axis = 0)
                    preview = np.array((preview + 1) / 2 * 255, dtype=np.uint8)

                    create_image_mosaic(preview, 3, len(preview)//3, f"{STEPS}.png")
                    wandb.log({"image": wandb.Image(f'{STEPS}.png')}, step=STEPS)

                    ckpt_manager.save(STEPS, models)


                progress_bar.update(1)
    except KeyboardInterrupt:
        i = -1
        print("Ctrl+C command detected. saving model before exiting...")
        ckpt_manager.save(i, models)

main()
