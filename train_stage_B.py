import os
import random
import shutil

import flax
import numpy as np
import jax
from safetensors.numpy import save_file, load_file
from jax.experimental.compilation_cache import compilation_cache as cc
import jax.numpy as jnp
from einops import rearrange
from flax.training import train_state
import optax
from tqdm import tqdm
import wandb
import jmp
import cv2
import dm_pix as pix

from cascade import DecoderStageA, EncoderStageA, UNetStageB
from utils import FrozenModel, create_image_mosaic, flatten_dict, unflatten_dict, get_overlapping_keys, exclude_keys_from_list
from streaming_dataloader import  threading_dataloader, collate_fn, ImageFolderDataset

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

 # checked
def init_model(batch_size = 256, training_res = 256, latent_dim=4, compression_ratio=4, controlnet_input_compression=2, controlnet_compression_ratio=16, controlnet_latent_dim=16, stage_a_path=None, seed = 42, learning_rate = 10e-3):
    with jax.default_device(jax.devices("cpu")[0]):
        unet_rng, _ = jax.random.split(jax.random.PRNGKey(seed), 2)

        enc = EncoderStageA(
            first_layer_output_features = 24,
            output_features = latent_dim,
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
            first_layer_output_features = 24,
            output_features = 16,
            down_layer_dim = (48, 96, 96, 128, 192),
            down_layer_kernel_size = (3, 3, 3, 3, 3),
            down_layer_blocks = (8, 8, 8, 10, 12),
            use_bias = False,
            conv_expansion_factor = (4, 4, 4, 4, 4),
            eps = 1e-6,
            group_count = (-1, -1, -1, -1, -1)
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


        # init model params
        image = jnp.ones((batch_size, training_res, training_res, 3))
        # encoder
        enc_params = enc.init(jax.random.PRNGKey(0), image)
        controlnet_image = jnp.ones((batch_size, training_res//controlnet_input_compression, training_res//controlnet_input_compression, 3))
        controlnet_params = controlnet.init(jax.random.PRNGKey(0), controlnet_image)
        # decoder
        dummy_latent = jnp.ones((batch_size, training_res // compression_ratio, training_res // compression_ratio, latent_dim))

        dec_params = dec.init(jax.random.PRNGKey(0), dummy_latent)
        # unet
        timesteps = jnp.ones([batch_size])[:,None].astype(jnp.int32)
        #  x, timestep, cond=None, image_pos=None, extra_pos=None
        controlnet_output_latent = jnp.ones(
            (
                batch_size, 
                int(training_res//controlnet_input_compression/controlnet_compression_ratio), 
                int(training_res//controlnet_input_compression/controlnet_compression_ratio), 
                controlnet_latent_dim,
            )
        )
        unet_params = unet.init(unet_rng, dummy_latent, timesteps, controlnet_output_latent)        # TODO! CONTROL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        enc_params = unflatten_dict(load_file(f"{stage_a_path}/enc_params.safetensors"))
        dec_params = unflatten_dict(load_file(f"{stage_a_path}/dec_params.safetensors"))

        # assume the shape match for now
        # init some of the controlnet layer from stage A
        flattened_controlnet_params = flatten_dict(controlnet_params)
        flattened_enc_params = flatten_dict(enc_params)
        overlapping_keys = get_overlapping_keys(flattened_controlnet_params, flattened_enc_params)
        excluded_keys = ['params.final_norm.scale', 'params.projections.bias', 'params.projections.kernel']
        transfered_params = exclude_keys_from_list(overlapping_keys, excluded_keys)

        # overwrite randomly init-ed params with pretrained stage A params
        for param_key in transfered_params:
            if flattened_controlnet_params[param_key].shape != flattened_enc_params[param_key].shape:
                print(f"shape missmatch at {param_key}")
            flattened_controlnet_params[param_key] = flattened_enc_params[param_key]
        
        controlnet_params = unflatten_dict(flattened_controlnet_params)

        controlnet_param_count = sum(list(flatten_dict(jax.tree_map(lambda x: x.size, controlnet_params)).values()))
        enc_param_count = sum(list(flatten_dict(jax.tree_map(lambda x: x.size, enc_params)).values()))
        dec_param_count = sum(list(flatten_dict(jax.tree_map(lambda x: x.size, dec_params)).values()))
        unet_param_count = sum(list(flatten_dict(jax.tree_map(lambda x: x.size, unet_params)).values()))
        print("control net param count:", controlnet_param_count)
        print("encoder param count:", enc_param_count)
        print("decoder param count:", dec_param_count)
        print("unet param count:", unet_param_count)
        
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
        unet_state = train_state.TrainState.create(
            apply_fn=unet.apply,
            params=unet_params,
            tx=adam_wrapper(
                jax.tree_util.tree_map_with_path(lambda path, var: path[-1].key != "scale" and path[-1].key != "bias", unet_params)
            ),
        )

        controlnet_state = train_state.TrainState.create(
            apply_fn=controlnet.apply,
            params=controlnet_params,
            tx=adam_wrapper(
                jax.tree_util.tree_map_with_path(lambda path, var: path[-1].key != "scale" and path[-1].key != "bias", controlnet_params)
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
        
        unet_state = jax.tree_map(
            lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, PartitionSpec())),
            unet_state,
        )
        controlnet_state = jax.tree_map(
            lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, PartitionSpec())),
            controlnet_state,
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

        return [controlnet_state, unet_state, enc_state, dec_state]

# checked
def train_flow_based(unet_state, controlnet_state, enc_state, upscale_factor, batch, train_rng):

    # always create new RNG!
    sample_rng, new_train_rng = jax.random.split(train_rng, num=2)

    # reminder NHWC
    n, h, w, c = batch.shape
    normal_latents = enc_state.call(enc_state.params, batch)


    def _compute_loss(
        unet_params, controlnet_params, latents, batch, rng_key
    ):
        unet_params, controlnet_params, latents, batch = mixed_precision_policy.cast_to_compute((unet_params, controlnet_params, latents, batch))
        noise_rng, timestep_rng, cond_rng = jax.random.split(key=rng_key, num=3)
        
        # lower res image is piped to the controlnet
        conditional = controlnet_state.apply_fn(controlnet_params, jax.image.resize(batch, shape=(n, int(h/upscale_factor), int(w/upscale_factor), c), method="bicubic"))
        # cfg training
        toggle_cond = jax.random.choice(cond_rng, jnp.array([0, 1]), [n], p=jnp.array([0.1, 0.9]))
        conditional = conditional * toggle_cond[:, None, None, None]


        # Sample noise that we'll add to the images
        # I think I should combine this with the first noise seed generator
        # might need to play with this noise distribution weighting instead of uniform sampling
        timesteps = jax.numpy.sort(jax.random.uniform(timestep_rng, [n]))[:,None]
        # rectified flow loss wrt guided image
        # generate noise to be denoised with some garbage interpolated latent of original images
        noises = jax.random.normal(key=noise_rng, shape=latents.shape) 
        # compute interpolation
        noise_image_lerp = noises * timesteps[:, None, None] + latents * (1-timesteps[:, None, None])
        flow_path = noises - latents # midpoint velocity
        model_pred = unet_state.apply_fn(unet_params, noise_image_lerp, timesteps, conditional)
        loss = jnp.mean((model_pred - flow_path)** 2)

        return mixed_precision_policy.cast_to_output(loss)

    grad_fn = jax.value_and_grad(
        fun=_compute_loss, argnums=[0,1]  # differentiate first param only
    )

    loss, grad = grad_fn(
        unet_state.params,
        controlnet_state.params,
        normal_latents,
        batch,
        sample_rng,
    )
    # update weight and bias value
    unet_state = unet_state.apply_gradients(grads=jmp.cast_to_full(grad[0]))
    controlnet_state = controlnet_state.apply_gradients(grads=jmp.cast_to_full(grad[1]))

    # calculate loss
    metrics = {"mse_loss": loss}
    return (
        unet_state,
        controlnet_state,
        metrics,
        new_train_rng,
    )

# checked
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
        cond_vector = jax.jit(model_state.apply_fn)(model_state.params, init_cond, t, conds)
        uncond_vector = jax.jit(model_state.apply_fn)(model_state.params, init_cond, t, conds*0)
        # cond_vector = model_state.apply_fn(model_state.params, init_cond, t, conds)
        # uncond_vector = model_state.apply_fn(model_state.params, init_cond, t, conds*0)
        return uncond_vector + (cond_vector - uncond_vector) * cfg_scale 

    # Euler method iteration
    for i in range(1, num_steps):
        Z = Z - _func_cfg(Z, jnp.stack([t[i - 1][None]]*Z.shape[0])) * dt

    return Z

# checked
def rando_colours(image_res):
    
    max_colour = np.full([1, image_res, image_res, 1], 255)
    min_colour = np.zeros((1, image_res, image_res, 1))

    black = np.concatenate([min_colour,min_colour,min_colour],axis=-1) / 255 * 2 - 1 
    white = np.concatenate([max_colour,max_colour,max_colour],axis=-1) / 255 * 2 - 1 
    red = np.concatenate([max_colour,min_colour,min_colour],axis=-1) / 255 * 2 - 1 
    green = np.concatenate([min_colour,max_colour,min_colour],axis=-1) / 255 * 2 - 1 
    blue = np.concatenate([min_colour,min_colour,max_colour],axis=-1) / 255 * 2 - 1 
    magenta = np.concatenate([max_colour,min_colour,max_colour],axis=-1) / 255 * 2 - 1 
    cyan = np.concatenate([min_colour,max_colour,max_colour],axis=-1) / 255 * 2 - 1 
    yellow = np.concatenate([max_colour,max_colour,min_colour],axis=-1) / 255 * 2 - 1 

    r = np.random.randint(0, 255) * np.ones((1, image_res, image_res, 1))
    g = np.random.randint(0, 255) * np.ones((1, image_res, image_res, 1))
    b = np.random.randint(0, 255) * np.ones((1, image_res, image_res, 1))
    rando_colour = np.concatenate([r,g,b],axis=-1) / 255 * 2 - 1 


    absolute = [black, white] * 4
    pallete = [red, green, blue, magenta, cyan, yellow, rando_colour] + absolute

    return random.choice(pallete)


def inference(unet_state, controlnet_state, dec_state, batch, stage_a_compression_ratio, stage_a_latent_size, controlnet_scale_factor, seed, t_span, dt, cfg_scale=1):

    n, h, w, c = batch.shape
    # initial noise + low res latent
    # UNJITTED
    cond_latents = jax.jit(controlnet_state.apply_fn)(controlnet_state.params, jax.image.resize(batch, shape=(n, int(h/controlnet_scale_factor), int(w/controlnet_scale_factor), c), method="bicubic"))
    # cond_latents = controlnet_state.apply_fn(controlnet_state.params, jax.image.resize(batch, shape=(n, h//controlnet_scale_factor, w//controlnet_scale_factor, c), method="bicubic"))
    init_cond = jax.random.normal(key=jax.random.PRNGKey(seed), shape=(n, h//stage_a_compression_ratio, w//stage_a_compression_ratio, stage_a_latent_size))

    # solve the model
    latents = euler_solver(init_cond, t_span, dt, model_state=unet_state, conds=cond_latents, cfg_scale=cfg_scale)

    # convert back to pixel space
    logits = jax.jit(dec_state.call)(dec_state.params, latents)
    
    return logits

# checked
def main():
    BATCH_SIZE = 256
    SEED = 0
    EPOCHS = 100
    SAVE_MODEL_PATH = "stage_b"
    STAGE_A_PATH = "stage_a_safetensors/119092"
    TRAINING_IMAGE_PATH = "ramdisk/train_images"
    IMAGE_OUTPUT_PATH = "output_stage_b"
    IMAGE_RES = 256
    LATENT_DIM = 4
    COMPRESSION_RATIO = 4
    UPSCALE_FACTOR = 4/3
    SAVE_EVERY = 5000
    LEARNING_RATE = 1e-4
    WANDB_PROJECT_NAME = "cascade"
    WANDB_RUN_NAME = "stage-b"
    WANDB_LOG_INTERVAL = 100
    LOAD_CHECKPOINTS = 0
    PERMANENT_EPOCH_STORE = 50000
    MAX_TO_SAVE = 3

    if not os.path.exists(IMAGE_OUTPUT_PATH):
        os.makedirs(IMAGE_OUTPUT_PATH)
    # wandb logging
    if WANDB_PROJECT_NAME:
        wandb.init(project=WANDB_PROJECT_NAME, name=WANDB_RUN_NAME)

    # init seed
    train_rng = jax.random.PRNGKey(SEED)
    # init model
    controlnet_state, unet_state, enc_state, dec_state = init_model(
        batch_size=BATCH_SIZE, 
        training_res=IMAGE_RES, 
        latent_dim=LATENT_DIM, 
        compression_ratio=COMPRESSION_RATIO,
        stage_a_path=STAGE_A_PATH, 
        seed=SEED, 
        learning_rate=LEARNING_RATE
    )
    STEPS = 0
    if LOAD_CHECKPOINTS != 0:
        print(f"RESUMING FROM CHECKPOINT:{LOAD_CHECKPOINTS}")
        STEPS = LOAD_CHECKPOINTS
        # load from safetensors
        unet_params = unflatten_dict(load_file(f"{SAVE_MODEL_PATH}/{STEPS}/unet_params.safetensors"))
        unet_mu = unflatten_dict(load_file(f"{SAVE_MODEL_PATH}/{STEPS}/unet_mu.safetensors"))
        unet_nu = unflatten_dict(load_file(f"{SAVE_MODEL_PATH}/{STEPS}/unet_nu.safetensors"))

        unet_state.params.update(jax.tree_map(lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, PartitionSpec())), unet_params))
        unet_state.opt_state[1][0].mu.update(jax.tree_map(lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, PartitionSpec())), unet_mu))
        unet_state.opt_state[1][0].nu.update(jax.tree_map(lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, PartitionSpec())), unet_nu))

        controlnet_params = unflatten_dict(load_file(f"{SAVE_MODEL_PATH}/{STEPS}/controlnet_params.safetensors"))
        controlnet_mu = unflatten_dict(load_file(f"{SAVE_MODEL_PATH}/{STEPS}/controlnet_mu.safetensors"))
        controlnet_nu = unflatten_dict(load_file(f"{SAVE_MODEL_PATH}/{STEPS}/controlnet_nu.safetensors"))

        controlnet_state.params.update(jax.tree_map(lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, PartitionSpec())), controlnet_params))
        controlnet_state.opt_state[1][0].mu.update(jax.tree_map(lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, PartitionSpec())), controlnet_mu))
        controlnet_state.opt_state[1][0].nu.update(jax.tree_map(lambda leaf: jax.device_put(leaf, device=NamedSharding(mesh, PartitionSpec())), controlnet_nu))


        del unet_params, unet_mu, unet_nu, controlnet_params, controlnet_mu, controlnet_nu

    sample_image = np.concatenate([cv2.imread(f"sample_{x}.jpg")[None, ...] for x in range(4)] * int(BATCH_SIZE//4), axis=0) / 255 * 2 - 1

    # UNJITTED
    train_flow_based_jit = jax.jit(train_flow_based, static_argnames=["upscale_factor"])
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
            
            # extra dataset aug and preprocess
            batch[0] = rando_colours(IMAGE_RES)
            batch = batch / 255 * 2 - 1

            # put batch to device
            batch = jax.tree_map(
                lambda leaf: jax.device_put(
                    leaf, device=NamedSharding(mesh, PartitionSpec("data_parallel", None, None, None))
                ),
                batch,
            )

            # train
            unet_state, controlnet_state, metrics, train_rng = train_flow_based_jit(unet_state, controlnet_state, enc_state, UPSCALE_FACTOR, batch, train_rng)


            if STEPS % (WANDB_LOG_INTERVAL//10) == 0:
                if jnp.isnan(metrics["mse_loss"]).any():
                    raise ValueError("The array contains NaN values")
        
                progress_bar.set_description(f"{metrics}")
                wandb.log(metrics, step=STEPS)
                

            if STEPS % WANDB_LOG_INTERVAL == 0:
                wandb.log(metrics, step=STEPS)
                # grab a handfull of images to test model perf
                eval_sample = jnp.concatenate([batch[:4], sample_image[:4]], axis = 0)

                model_eval = inference(unet_state, controlnet_state, dec_state, eval_sample, COMPRESSION_RATIO, LATENT_DIM, UPSCALE_FACTOR, 0, (1, 0.001), 0.01, 1.1)

                preview = jnp.concatenate([eval_sample, model_eval], axis=0)
                preview = (preview + 1) / 2 * 255
                preview  = np.array(preview, dtype=np.uint8)
                create_image_mosaic(preview, 4, 4, f"{IMAGE_OUTPUT_PATH}/{STEPS}.png")

            # save every n steps
            if STEPS % SAVE_EVERY == 0:
                try:

                    wandb.log({"image": wandb.Image(f'{IMAGE_OUTPUT_PATH}/{STEPS}.png')}, step=STEPS)
                except Exception as e:
                    print(e)



                try:
                    if not os.path.exists(f"{SAVE_MODEL_PATH}/{STEPS}"):
                        os.makedirs(f"{SAVE_MODEL_PATH}/{STEPS}")
                    save_file(flatten_dict(unet_state.params), f"{SAVE_MODEL_PATH}/{STEPS}/unet_params.safetensors")
                    save_file(flatten_dict(unet_state.opt_state[1][0].mu), f"{SAVE_MODEL_PATH}/{STEPS}/unet_mu.safetensors")
                    save_file(flatten_dict(unet_state.opt_state[1][0].nu), f"{SAVE_MODEL_PATH}/{STEPS}/unet_nu.safetensors")
                    save_file(flatten_dict(controlnet_state.params), f"{SAVE_MODEL_PATH}/{STEPS}/controlnet_params.safetensors")
                    save_file(flatten_dict(controlnet_state.opt_state[1][0].mu), f"{SAVE_MODEL_PATH}/{STEPS}/controlnet_mu.safetensors")
                    save_file(flatten_dict(controlnet_state.opt_state[1][0].nu), f"{SAVE_MODEL_PATH}/{STEPS}/controlnet_nu.safetensors")


                except Exception as e:
                    print(e)


                try:
                    # delete checkpoint 
                    if (STEPS - MAX_TO_SAVE * SAVE_EVERY) % PERMANENT_EPOCH_STORE == 0:
                        pass
                    else:
                        shutil.rmtree(f"{SAVE_MODEL_PATH}/{STEPS - MAX_TO_SAVE * SAVE_EVERY}")
                except:

                    pass

            progress_bar.update(1)
            STEPS += 1


main()
