import os
import random

import flax
import numpy as np
import jax
from safetensors.numpy import save_file, load_file
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
import cv2
import dm_pix as pix

from vae import DecoderStageA, EncoderStageA
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


def init_model(batch_size = 256, training_res = 256, latent_dim=4, compression_ratio=4, seed = 42, learning_rate = 10e-3):
    with jax.default_device(jax.devices("cpu")[0]):
        enc_rng, dec_rng, lpips_rng = jax.random.split(jax.random.PRNGKey(seed), 3)

        enc = EncoderStageA(
            first_layer_output_features = 24,
            output_features = 4,
            down_layer_dim = (48, 96),
            down_layer_kernel_size = (3, 3),
            down_layer_blocks = (8, 8),
            down_layer_ordinary_conv = (True, True),
            use_bias = False ,
            conv_expansion_factor = (4, 4),
            eps = 1e-6,
            group_count = 16,


        )
        dec = DecoderStageA(
            last_upsample_layer_output_features = 24,
            output_features = 3,
            up_layer_dim = (96, 48),
            up_layer_kernel_size = (3, 3),
            up_layer_blocks = (8, 8),
            up_layer_ordinary_conv = (True, True) ,
            use_bias = False ,
            conv_expansion_factor = (4, 4),
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
        dummy_latent = jnp.ones((batch_size, training_res // compression_ratio, training_res // compression_ratio, latent_dim))
        dec_params = dec.init(dec_rng, dummy_latent)
        # LPIPS VGG16
        lpips_params = lpips.init(lpips_rng, image, image)

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

        # trained

        # do not apply weight decay to norm layer
        enc_state = train_state.TrainState.create(
            apply_fn=enc.apply,
            params=enc_params,
            tx=adam_wrapper(
                jax.tree_util.tree_map_with_path(lambda path, var: path[-1].key != "scale" and path[-1].key != "bias", enc_params)
            ),
        )
        # trained
        dec_state = train_state.TrainState.create(
            apply_fn=dec.apply,
            params=dec_params,
            tx=adam_wrapper(
                jax.tree_util.tree_map_with_path(lambda path, var: path[-1].key != "scale" and path[-1].key != "bias", dec_params)
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
    
        lpips_state = jax.tree_map(
            lambda leaf: jax.device_put(jmp.cast_to_half(leaf), device=NamedSharding(mesh, PartitionSpec())),
            lpips_state,
        )
        return [enc_state, dec_state, lpips_state]

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

        pred_batch = dec_state.apply_fn(dec_params, latents)

        # MSE loss
        mse_loss = ((batch - pred_batch) ** 2).mean()
        # lpips loss
        lpips_loss = lpips_state.call(lpips_params, batch, pred_batch).mean()

        vae_loss = mixed_precision_policy.cast_to_output(
            mse_loss * loss_scale["mse_loss_scale"] + 
            lpips_loss * loss_scale["lpips_loss_scale"]
        )

        vae_loss_stats = {
            "mse_loss": mse_loss * loss_scale["mse_loss_scale"],
            "lpips_loss": lpips_loss * loss_scale["lpips_loss_scale"],
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
def inference(models, batch):
    enc_state, dec_state, _ = models

    enc_params, dec_params, batch = mixed_precision_policy.cast_to_compute((enc_state.params, dec_state.params, batch))
    latents = enc_state.apply_fn(enc_params, batch)
    return dec_state.apply_fn(dec_params, latents)

def main():
    BATCH_SIZE = 256
    SEED = 0
    EPOCHS = 100
    SAVE_MODEL_PATH = "vae_small_ckpt"
    TRAINING_IMAGE_PATH = "ramdisk/train_images"
    IMAGE_RES = 256
    SAVE_EVERY = 500
    LEARNING_RATE = 1e-4
    LOSS_SCALE = {
        "mse_loss_scale": 1,
        "lpips_loss_scale": 0.25,

    }
    WANDB_PROJECT_NAME = "vae"
    WANDB_RUN_NAME = "PINVAE"
    WANDB_LOG_INTERVAL = 100
    LOAD_CHECKPOINTS = 0

    # wandb logging
    if WANDB_PROJECT_NAME:
        wandb.init(project=WANDB_PROJECT_NAME, name=WANDB_RUN_NAME)

    # init seed
    train_rng = jax.random.PRNGKey(SEED)
    # init model
    models = init_model(batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)

    if LOAD_CHECKPOINTS != 0:
        print(f"RESUMING FROM CHECKPOINT:{LOAD_CHECKPOINTS}")
        STEPS = LOAD_CHECKPOINTS
        # load from safetensors
        enc_params = unflatten_dict(load_file(f"{SAVE_MODEL_PATH}/{STEPS}/enc_params.safetensors"))
        enc_mu = unflatten_dict(load_file(f"{SAVE_MODEL_PATH}/{STEPS}/enc_mu.safetensors"))
        enc_nu = unflatten_dict(load_file(f"{SAVE_MODEL_PATH}/{STEPS}/enc_nu.safetensors"))

        models[0].params.update(jax.tree_map(lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, PartitionSpec())), enc_params))
        models[0].opt_state[1][0].mu.update(jax.tree_map(lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, PartitionSpec())), enc_mu))
        models[0].opt_state[1][0].nu.update(jax.tree_map(lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, PartitionSpec())), enc_nu))

        dec_params = unflatten_dict(load_file(f"{SAVE_MODEL_PATH}/{STEPS}/dec_params.safetensors"))
        dec_mu = unflatten_dict(load_file(f"{SAVE_MODEL_PATH}/{STEPS}/dec_mu.safetensors"))
        dec_nu = unflatten_dict(load_file(f"{SAVE_MODEL_PATH}/{STEPS}/dec_nu.safetensors"))

        models[0].params.update(jax.tree_map(lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, PartitionSpec())), dec_params))
        models[0].opt_state[1][0].mu.update(jax.tree_map(lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, PartitionSpec())), dec_mu))
        models[0].opt_state[1][0].nu.update(jax.tree_map(lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, PartitionSpec())), dec_nu))
        del (
            enc_params,
            enc_mu,
            enc_nu,
            dec_params,
            dec_mu,
            dec_nu,
        ) # flush


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

    for _ in range(EPOCHS):
        # dataset = CustomDataset(parquet_url, square_size=IMAGE_RES)
        dataset = ImageFolderDataset(TRAINING_IMAGE_PATH, square_size=IMAGE_RES, seed=STEPS)
        t_dl = threading_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,  num_workers=100, prefetch_factor=3, seed=SEED)
        # Initialize the progress bar
        progress_bar = tqdm(total=len(dataset) // BATCH_SIZE, position=0)

        for i, batch in enumerate(t_dl):
            batch_size,_,_,_ = batch.shape
            
            if batch_size < BATCH_SIZE:
                continue
            
            batch[0] = rando_colours(IMAGE_RES)
            batch_og = batch / 255 * 2 - 1

            # batch_blur = pix.gaussian_blur(batch_og, 1, 3)
            # batch = jnp.clip(batch_og + (batch_og - batch_blur) * 1.5, -1, 1)
            batch = batch_og

            batch = jax.tree_map(
                lambda leaf: jax.device_put(
                    leaf, device=NamedSharding(mesh, PartitionSpec("data_parallel", None, None, None))
                ),
                batch,
            )

            models, train_rng, output, stats = train_vae_only(models, batch, LOSS_SCALE, train_rng)

            if i % WANDB_LOG_INTERVAL == 0:
                wandb.log(stats, step=STEPS)
                preview_test = inference(models, sample_image)
                preview = jnp.concatenate([batch_og[:4], output[:4], preview_test[:4]], axis = 0)
                preview = jnp.clip(preview, -1, 1)
                preview = np.array((preview + 1) / 2 * 255, dtype=np.uint8)

                create_image_mosaic(preview, 3, len(preview)//3, f"output/{STEPS}.png")

                preview = jnp.concatenate([batch[:4], output[:4], preview_test[:4]], axis = 0)
                preview = jnp.clip(preview, -1, 1)
                preview = np.array((preview + 1) / 2 * 255, dtype=np.uint8)

                create_image_mosaic(preview, 3, len(preview)//3, f"output_scaled/{STEPS}.png")



            # save every n steps
            if i % SAVE_EVERY == 0:
                try:
                    preview_test = inference(models, sample_image)
                    preview = jnp.concatenate([batch_og[:4], output[:4], preview_test[:4]], axis = 0)
                    preview = jnp.clip(preview, -1, 1)
                    preview = np.array((preview + 1) / 2 * 255, dtype=np.uint8)


                    create_image_mosaic(preview, 3, len(preview)//3, f"output/{STEPS}.png")
                    wandb.log({"image": wandb.Image(f'output/{STEPS}.png')}, step=STEPS)
                except Exception as e:
                    print(e)



                try:
                    if not os.path.exists(f"{SAVE_MODEL_PATH}/{STEPS}"):
                        os.makedirs(f"{SAVE_MODEL_PATH}/{STEPS}")
                    save_file(flatten_dict(models[0].params), f"{SAVE_MODEL_PATH}/{STEPS}/enc_params.safetensors")
                    save_file(flatten_dict(models[0].opt_state[1][0].mu), f"{SAVE_MODEL_PATH}/{STEPS}/enc_mu.safetensors")
                    save_file(flatten_dict(models[0].opt_state[1][0].nu), f"{SAVE_MODEL_PATH}/{STEPS}/enc_nu.safetensors")

                    save_file(flatten_dict(models[1].params), f"{SAVE_MODEL_PATH}/{STEPS}/dec_params.safetensors")
                    save_file(flatten_dict(models[1].opt_state[1][0].mu), f"{SAVE_MODEL_PATH}/{STEPS}/dec_mu.safetensors")
                    save_file(flatten_dict(models[1].opt_state[1][0].nu), f"{SAVE_MODEL_PATH}/{STEPS}/dec_nu.safetensors")
                except Exception as e:
                    print(e)


            progress_bar.update(1)
            STEPS += 1


main()
