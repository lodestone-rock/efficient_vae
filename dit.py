import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
import math
from typing import Tuple, Tuple
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention
import numpy as np

# def rand_stratified_cosine(rng_key, batch, sigma_data=1., min_value=1e-3, max_value=1e3):
#     # do stratified sampling on cosine timesteps, ensuring within each batch 
#     # it will always has representation for each section of the timesteps
#     offsets = jnp.arange(0, batch)
#     u = jax.random.uniform(rng_key, [batch])
#     u = (offsets + u) / batch
#     # u = jnp.linspace(0,1, 10000)

#     def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
#         t_min = jnp.atan(jnp.exp(-0.5 * logsnr_max))
#         t_max = jnp.atan(jnp.exp(-0.5 * logsnr_min))
#         return -2 * jnp.log(jnp.tan(t_min + t * (t_max - t_min)))

#     logsnr_min = -2 * jnp.log(min_value / sigma_data)
#     logsnr_max = -2 * jnp.log(max_value / sigma_data)
#     logsnr = logsnr_schedule_cosine(u, logsnr_min, logsnr_max)
#     return jnp.exp(-logsnr / 2) * sigma_data

def rand_stratified_cosine(rng_key, batch, sigma_data=1., min_value=1e-3, max_value=1e3):
    # do stratified sampling on cosine timesteps, ensuring within each batch 
    # it will always has representation for each section of the timesteps
    offsets = jnp.arange(0, batch)
    u = jax.random.uniform(rng_key, [batch])
    u = (offsets + u) / batch
    # u = jnp.linspace(0,1, 10000)

    min_cdf = jnp.atan(min_value / sigma_data) * 2 / jnp.pi
    max_cdf = jnp.atan(max_value / sigma_data) * 2 / jnp.pi
    u = u * (max_cdf - min_cdf) + min_cdf
    return jnp.tan(u * jnp.pi / 2) * sigma_data

# for rope, time, and 2d pos
def create_sinusoidal_positions_fn(pos, dim=768, freq_scale=None, concat_sin_cos=True):
    with jax.default_device(jax.devices("cpu")[0]):
        assert dim % 2 == 0
        # just a list of number to enumerate each dim (divided by 2 because we use a half for sine and other for cosine)
        hidden_dim_pos = jnp.arange(dim//2)
        # scale it down to 0-1
        hidden_dim_pos = hidden_dim_pos / dim * 2
        # inverse exponential decay scaling from good ol' transformer paper
        # ps. no one bothered to tweak the constant 10000 because practically it makes no difference
        # but we need to scale it wee bit because deeper the dimension it will just collapsed to 1 and -1 for 2d position
        # especially when we're dealing with extremely small latent 
        if freq_scale is None:
            inv_freq = 1.0 / ((dim//10) ** hidden_dim_pos)
        else:
            inv_freq = 1.0 / (freq_scale ** hidden_dim_pos)
        
        freqs = jnp.einsum("i , j -> i j", pos, inv_freq).astype("float32")

        if concat_sin_cos:
            return jnp.concatenate([jnp.sin(freqs), jnp.cos(freqs)],axis=-1)
        else:
            return jnp.sin(freqs), jnp.cos(freqs)
    

def create_1d_sinusoidal_positions_fn(num_pos=1024, dim=768, freq_scale=10000):
    with jax.default_device(jax.devices("cpu")[0]):
        return create_sinusoidal_positions_fn(jnp.arange(num_pos), dim, freq_scale)


def create_2d_sinusoidal_positions_fn(embed_dim, grid_size, freq_scale=None, relative_center=False):
    # assume square image for now
    if freq_scale is None:
        freq_scale = embed_dim // 10
    if relative_center:
        h = jnp.linspace(1, -1, grid_size) * grid_size / 2
        w = jnp.linspace(1, -1, grid_size) * grid_size / 2
    else:
        h = jnp.arange(grid_size, dtype=jnp.float32)
        w = jnp.arange(grid_size, dtype=jnp.float32)

    # practically just replicating w and h from (1, grid_size) to (grid_size, grid_size)
    h, w = jnp.meshgrid(w, h)

    pos_w_sin, pos_w_cos = create_sinusoidal_positions_fn(w.reshape(-1), embed_dim, freq_scale, False)
    pos_h_sin, pos_h_cos = create_sinusoidal_positions_fn(h.reshape(-1), embed_dim, freq_scale, False)
    return jnp.concatenate([pos_h_sin, pos_w_sin, pos_h_cos, pos_w_cos], axis=-1)


# def create_2d_sinusoidal_positions_fn(width_len=8, height_len=8, dim=768, max_freq=None):
#     with jax.default_device(jax.devices("cpu")[0]):

#         # max frequency must be at most a half of your width and height whichever come first
#         # nyquist sampling theorem thingy
#         # to interpolate width and height, keep the max freq fixed 
#         if max_freq is None:
#             max_freq = height_len / 2 if height_len < width_len else width_len / 2

#         freqs = jnp.linspace(1, max_freq, dim)

#         # stack of position embedding for each dim
#         sinusoidal_2d = []

#         for freq in freqs:
#             # create wavy plane
#             height = jnp.sin(jnp.stack([jnp.linspace(1, -1, height_len)] * width_len, axis=0).T * freq)
#             width = jnp.cos(jnp.stack([jnp.linspace(1, -1, width_len)] * height_len, axis=0) * freq)
#             # slant it diagonally
#             height_slant = jnp.stack([jnp.linspace(1, -1, height_len)] * width_len, axis=0).T
#             width_slant = jnp.stack([jnp.linspace(1, -1, width_len)] * height_len, axis=0)
#             # combine it
#             compound = height - width + height_slant + width_slant
#             # rescale it
#             compound = (compound- compound.min())/(compound.max()-compound.min())
#             sinusoidal_2d.append(compound)
        
#         sinusoidal_2d = jnp.stack(sinusoidal_2d, axis=-1)
#         return sinusoidal_2d



# yoinked from hf transformers
def rotate_half(tensor):
    """Rotates half the hidden dims of the input."""
    rotate_half_tensor = jnp.concatenate(
        (-tensor[..., tensor.shape[-1] // 2 :], tensor[..., : tensor.shape[-1] // 2]), axis=-1
    )
    return rotate_half_tensor


# yoinked from hf transformers
def apply_rotary_pos_emb(tensor, pos):
    sin_pos, cos_pos = rearrange(pos[...,None,:], "b l h (d sincos)-> sincos b l h d", sincos=2)
    return (tensor * cos_pos) + (rotate_half(tensor) * sin_pos)


def embedding_modulation(x, scale, shift):
    # basically applies time embedding and conditional to each input for each layer
    return x * (1 + scale) + shift 

class SelfAttention(nn.Module):
    n_heads: int = 8
    expansion_factor: int = 1
    use_bias: bool = False # removing bias won't affect the model much
    eps:float = 1e-6
    embed_dim: int = 768
    use_flash_attention: bool = False

    def setup(self):

        self.rms_norm = nn.RMSNorm(
            epsilon=self.eps, 
            dtype=jnp.float32,
            param_dtype=jnp.float32
        )
        self.cond_projection =  nn.Dense(
            features=self.embed_dim * 3, 
            use_bias=self.use_bias
        )
        # pointwise conv reformulated as matmul
        self.qkv = nn.Dense(
            features=self.embed_dim * self.expansion_factor * 3, 
            use_bias=self.use_bias
        )
        self.out = nn.Dense(
            features=self.embed_dim, 
            use_bias=self.use_bias,
            kernel_init=jax.nn.initializers.zeros
        )

    def __call__(self, x, cond, pos=None):

        # store input as residual identity
        residual = x
        x = self.rms_norm(x)
        scale, shift, gate = rearrange(self.cond_projection(cond), "b l (d split) -> split b l d", split=3)
        x = embedding_modulation(x, scale, shift)
        # query, key, value
        q, k, v = rearrange(self.qkv(x), "b l (d split) -> split b l d", split=3)
        # if using rotary
        if self.use_flash_attention:
            # [batch_size, num_heads, q_seq_len, d_model]
            q = rearrange(q, "b l (d n_head) -> b n_head l d", n_head=self.n_heads)
            # [batch_size, num_heads, kv_seq_len, d_model]
            k = rearrange(k, "b l (d n_head) -> b n_head l d", n_head=self.n_heads)
            # [batch_size, num_heads, kv_seq_len, d_model]
            v = rearrange(v, "b l (d n_head) -> b n_head l d", n_head=self.n_heads)
            if pos is not None:
                # pos is theta angle for rotary embedding
                q = apply_rotary_pos_emb(q, pos)
                k = apply_rotary_pos_emb(k, pos)
            # use pallas flash attention kernel
            # https://github.com/google/jax/issues/18590#issuecomment-1830671863
            # turns out no need to use this since we're dealing with seq length smaller than 4096
            out = flash_attention(q,k,v)
            # output projection
            out = rearrange(out, "b n_head l d -> b l (n_head d)")
        else:
            # [batch_size, num_heads, q_seq_len, d_model]
            q = rearrange(q, "b l (d n_head) -> b l n_head d", n_head=self.n_heads)
            # [batch_size, num_heads, kv_seq_len, d_model]
            k = rearrange(k, "b l (d n_head) -> b l n_head d", n_head=self.n_heads)
            # [batch_size, num_heads, kv_seq_len, d_model]
            v = rearrange(v, "b l (d n_head) -> b l n_head d", n_head=self.n_heads)
            if pos is not None:
                # pos is theta angle for rotary embedding
                q = apply_rotary_pos_emb(q, pos)
                k = apply_rotary_pos_emb(k, pos)
            out = nn.dot_product_attention(q,k,v)
            # output projection
            out = rearrange(out, "b l n_head d -> b l (n_head d)")
        x = self.out(out)
        return  x + residual + gate
    

class GLU(nn.Module):
    embed_dim: int = 768
    expansion_factor: int = 4
    use_bias: bool = False
    eps:float = 1e-6
    cond: bool = True

    def setup(self):
        self.rms_norm = nn.RMSNorm(
            epsilon=self.eps, 
            dtype=jnp.float32,
            param_dtype=jnp.float32
        )
        if self.cond:
            self.cond_projection =  nn.Dense(
                features=self.embed_dim * 3, 
                use_bias=self.use_bias
            )
        # inverted bottleneck with gating
        # practically mimicing mobilenet except only pointwise conv
        self.up_proj = nn.Dense(self.embed_dim * self.expansion_factor, use_bias=False)
        self.gate_proj = nn.Dense(self.embed_dim * self.expansion_factor, use_bias=False)
        self.down_proj = nn.Dense(self.embed_dim, use_bias=False, kernel_init=jax.nn.initializers.zeros)

    def __call__(self, x, cond=None):
        residual = x
        x = self.rms_norm(x)
        if self.cond:
            scale, shift, gate = rearrange(self.cond_projection(cond), "b l (d split) -> split b l d", split=3)
            x = embedding_modulation(x, scale, shift)
        x = self.down_proj(self.up_proj(x) * nn.silu(self.gate_proj(x)))
        if self.cond:
            return x + residual + gate
        else:
            return x + residual


class FourierLayers(nn.Module):
    # linear layer probably sufficient but eh why not
    features: int = 768
    keep_random: bool = False # random fourier features mode

    def setup(self):
        self.freq =  nn.Dense(
            features=self.features // 2, 
            use_bias=False
        )

    def __call__(self, timesteps):
        
        if self.keep_random:
            freq = jax.lax.stop_gradient(nn.freq(timesteps * jnp.pi * 2))
        else:    
            freq = self.freq(timesteps * jnp.pi * 2)
        return jnp.concatenate([jnp.sin(freq), jnp.cos(freq)], axis=-1)



class DiTBLock(nn.Module):
    embed_dim: int = 768

    attn_expansion_factor: int = 2
    glu_expansion_factor: int = 4
    use_bias: bool = False
    eps:float = 1e-6
    n_heads: int = 8
    n_layers: int = 24
    n_time_embed_layers: int = 3
    time_embed_expansion_factor: int = 2
    diffusion_timesteps: int = 1000
    use_flash_attention: bool = False
    n_class: int = 1000
    latent_size: int = 8
    pixel_based: bool = False

    def setup(self):
        

        self.linear_proj_input = nn.Dense(
            features=self.embed_dim, 
            use_bias=self.use_bias
        )

        if self.latent_size is not None:
            self.latent_positional_embedding = self.create_2d_sinusoidal_pos(self.latent_size)
        if self.use_flash_attention:
            raise Exception("not supported yet! still have to figure out padding shape")

        # time embed lut
        self.time_embed = nn.Embed(
            self.diffusion_timesteps, 
            features=self.embed_dim
        )
        # class embed just an ordinary lookup table
        if self.n_class is not None:
            self.class_embed = nn.Embed(
                self.n_class, 
                features=self.embed_dim
            )

        # transformers
        dit_blocks = []
        for layer in range(self.n_layers):
            attn = SelfAttention(
                n_heads=self.n_heads, 
                expansion_factor=self.attn_expansion_factor, 
                use_bias=self.use_bias, 
                eps=self.eps,
                embed_dim=self.embed_dim,
                use_flash_attention=self.use_flash_attention
            )
            glu = GLU(embed_dim=self.embed_dim, expansion_factor=self.glu_expansion_factor, use_bias=self.use_bias, eps=self.eps)

            dit_blocks.append([attn, glu])

        self.dit_blocks = dit_blocks

        # time embedding
        time_embedding_stack = [] 
        for layer in range(self.n_layers):
            # overkill but eh whatever
            glu_time = GLU(embed_dim=self.embed_dim, expansion_factor=self.time_embed_expansion_factor, use_bias=self.use_bias, eps=self.eps, cond=False)

            time_embedding_stack.append(glu_time)
        self.time_embedding_stack = time_embedding_stack

        self.final_norm = nn.RMSNorm(
            epsilon=self.eps, 
            dtype=jnp.float32,
            param_dtype=jnp.float32
        )
        self.cond_projection =  nn.Dense(
            features=self.embed_dim * 2, 
            use_bias=self.use_bias
        )
        if self.pixel_based:
            self.output = nn.Dense(
                features=3, 
                use_bias=self.use_bias
            )
        else:
            self.output = nn.Dense(
                features=self.embed_dim, 
                use_bias=self.use_bias
            )

        
    def create_2d_sinusoidal_pos(self, width_len=8, height_len=8):
        pos = create_2d_sinusoidal_positions_fn(self.embed_dim, width_len, freq_scale=None, relative_center=False, for_rope=False)
        # pos = create_2d_sinusoidal_positions_fn(
        #     width_len=width_len, 
        #     height_len=height_len, 
        #     dim=self.embed_dim, 
        #     max_freq=self.embed_dim // 2, 
        #     epsilon=self.eps
        # )
        return pos[None, ...] # add batch dim HWC -> NHWC


    def __call__(self, x, timestep, cond=None, image_pos=None, extra_pos=None):
        # NOTE: flax uses NHWC convention so the entire ops is in NHWC
        # merge height and weight and treat it as a token sequence
        # NHWC => BLD 
        n, h, w, c = x.shape
        x = rearrange(x, "n h w c-> n (h w) c")
        x = self.linear_proj_input(x)
        # use default image position latent if not provided
        if self.latent_size is not None:
            x = x + jax.lax.stop_gradient(self.latent_positional_embedding)
        else:
            x = x + image_pos
        
        # time loop 
        time = self.time_embed(timestep)[:,None,:] # grab the time
        if self.n_class is not None:
            class_cond = self.class_embed(cond)[:,None,:]
            cond = time + class_cond
        # mebbe ditching this loop is simpler 
        # simple time embedding vector should suffice
        # silly ideas, rescale the input vector relative to this 
        # so the model pay more attention to the timesteps or class
        for glu_time in self.time_embedding_stack:
            cond = glu_time(cond)
        cond = nn.silu(cond)

        # TODO: add rope

        # if self.n_class is not None:
        #     x = jnp.concatenate([time, class_cond, x], axis=-2) # put time embedding in the seq dim
        # else:
        #     x = jnp.concatenate([time, x], axis=-2) # put time embedding in the seq dim
        # attention block loop
        for attn, glu in self.dit_blocks:
            x = attn(x, cond, extra_pos) # NOTE: if using this pos put image token in 1 position!
            x = glu(x, cond)
        # if self.n_class is not None:
        #     x = x[:,2:,:] # pop time and class embedding
        # else:
        #     x = x[:,1:,:] # pop time embedding
        x = self.final_norm(x)
        scale, shift = rearrange(self.cond_projection(cond), "b l (d split) -> split b l d", split=2)
        x = embedding_modulation(x, scale, shift)
        x = self.output(x)
        x = rearrange(x, "n (h w) c -> n h w c", h=h)
        return x


class DiTBLockContinuous(nn.Module):
    # to use this model during inference please wrap model.apply with model.pred method
    # model.pred(model.apply, x, timesteps, conds, *rest_of_args)
    sigma_data: float = 0.5

    embed_dim: int = 768
    output_dim: int = 256

    attn_expansion_factor: int = 2
    glu_expansion_factor: int = 4
    use_bias: bool = False
    eps:float = 1e-6
    n_heads: int = 8
    n_layers: int = 24
    n_time_embed_layers: int = 3
    time_embed_expansion_factor: int = 2
    diffusion_timesteps: int = 1000
    use_flash_attention: bool = False
    n_class: int = 1000
    latent_size: int = 8
    pixel_based: bool = False
    random_fourier_features: bool = False
    use_rope: bool = True

    def setup(self):
        

        self.linear_proj_input = nn.Dense(
            features=self.embed_dim, 
            use_bias=self.use_bias
        )

        if self.latent_size is not None:
            self.latent_positional_embedding = self.create_2d_sinusoidal_pos(self.latent_size)
        if self.use_flash_attention:
            raise Exception("not supported yet! still have to figure out padding shape")

        self.time_embed = FourierLayers(
            features=self.embed_dim, 
            keep_random=self.random_fourier_features,
        )

        # class embed just an ordinary lookup table
        if self.n_class is not None:
            self.class_embed = nn.Embed(
                self.n_class, 
                features=self.embed_dim
            )

        # transformers
        dit_blocks = []
        for layer in range(self.n_layers):
            attn = SelfAttention(
                n_heads=self.n_heads, 
                expansion_factor=self.attn_expansion_factor, 
                use_bias=self.use_bias, 
                eps=self.eps,
                embed_dim=self.embed_dim,
                use_flash_attention=self.use_flash_attention
            )
            glu = GLU(embed_dim=self.embed_dim, expansion_factor=self.glu_expansion_factor, use_bias=self.use_bias, eps=self.eps)

            dit_blocks.append([attn, glu])

        self.dit_blocks = dit_blocks

        # time embedding
        time_embedding_stack = [] 
        for layer in range(self.n_layers):
            # overkill but eh whatever
            glu_time = GLU(embed_dim=self.embed_dim, expansion_factor=self.time_embed_expansion_factor, use_bias=self.use_bias, eps=self.eps, cond=False)

            time_embedding_stack.append(glu_time)
        self.time_embedding_stack = time_embedding_stack

        self.final_norm = nn.RMSNorm(
            epsilon=self.eps, 
            dtype=jnp.float32,
            param_dtype=jnp.float32
        )
        self.cond_projection =  nn.Dense(
            features=self.embed_dim * 2, 
            use_bias=self.use_bias
        )
        if self.pixel_based:
            self.output = nn.Dense(
                features=3, 
                use_bias=self.use_bias
            )
        else:
            self.output = nn.Dense(
                features=self.output_dim, 
                use_bias=self.use_bias
            )

        
    def create_2d_sinusoidal_pos(self, width_len=8, use_rope=True):
        pos = create_2d_sinusoidal_positions_fn(self.embed_dim * self.attn_expansion_factor // self.n_heads, width_len, freq_scale=None, relative_center=use_rope)
        # pos = create_2d_sinusoidal_positions_fn(
        #     width_len=width_len, 
        #     height_len=height_len, 
        #     dim=self.embed_dim, 
        #     max_freq=self.embed_dim // 2, 
        #     epsilon=self.eps
        # )
        return pos[None, ...] # add batch dim HWC -> NHWC

    def sigma_scaling(self, timesteps):
        # this is karras preconditioning formula to help the model stays in unit variance
        # better alternatives than replacing the noise from standard gaussian to have unit variance 
        c_skip = self.sigma_data ** 2 / (timesteps ** 2 + self.sigma_data ** 2)
        c_out = timesteps * self.sigma_data / (timesteps ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (timesteps ** 2 + self.sigma_data ** 2) ** 0.5
        # reweight loss so it focus more on low frequency denoising part
        soft_weighting = (timesteps * self.sigma_data) ** 2 / (timesteps ** 2 + self.sigma_data ** 2) ** 2
        return c_skip, c_out, c_in, soft_weighting


    def loss(self, rng_key, model_apply_fn, model_params, images, timesteps, conds, image_pos=None, extra_pos=None):
        c_skip, c_out, c_in, soft_weighting = [scale[:, None, None, None] for scale in self.sigma_scaling(timesteps)]
        noises = jax.random.normal(key=rng_key, shape=images.shape)
        noised_images = (images + noises * timesteps[:, None, None, None])
        model_predictions = model_apply_fn(model_params, noised_images * c_in, timesteps, conds, image_pos, extra_pos)
        # target "noise"
        # this scaling has interesting dynamics if you print out the image
        # mainly the model returned a prediction based on frequency that remains after noise is added
        target_predictions = (images - noised_images * c_skip)/c_out
        loss = jnp.mean((model_predictions-target_predictions)** 2, axis=[1,2,3]) * soft_weighting.reshape(-1)
        return jnp.mean(loss), (model_predictions, target_predictions, model_predictions - target_predictions, noised_images, images)  # flatten


    def pred(self, model_apply_fn, model_params, images, timesteps, conds, image_pos=None, extra_pos=None):
        c_skip, c_out, c_in, _ = [scale[:, None, None, None] for scale in self.sigma_scaling(timesteps)]
        return model_apply_fn(model_params, images * c_in, timesteps, conds, image_pos, extra_pos) * c_out + images * c_skip
        

    def __call__(self, x, timestep, cond=None, image_pos=None, extra_pos=None):
        # NOTE: flax uses NHWC convention so the entire ops is in NHWC
        # merge height and weight and treat it as a token sequence
        # NHWC => BLD 
        n, h, w, c = x.shape
        x = rearrange(x, "n h w c-> n (h w) c")
        x = self.linear_proj_input(x)

        # time loop 
        time = self.time_embed(timestep[:,None])[:,None,:] # grab the time
        if self.n_class and cond is not None :
            class_cond = self.class_embed(cond)[:,None,:]
            cond = time + class_cond
        else:
            cond = time 
        # mebbe ditching this loop is simpler 
        # simple time embedding vector should suffice
        # silly ideas, rescale the input vector relative to this 
        # so the model pay more attention to the timesteps or class
        for glu_time in self.time_embedding_stack:
            cond = glu_time(cond)
        # jax.debug.print("max,{max} min,{min} mean,{mean}", max=cond.max(), min=cond.min(), mean=cond.mean())

        # attention block loop
        for attn, glu in self.dit_blocks:
            x = attn(x, cond, jax.lax.stop_gradient(self.latent_positional_embedding) if self.use_rope else None)
            x = glu(x, cond)
        # jax.debug.print("X max,{max} min,{min} mean,{mean}", max=x.max(), min=x.min(), mean=x.mean())

        x = self.final_norm(x)
        scale, shift = rearrange(self.cond_projection(cond), "b l (d split) -> split b l d", split=2)
        x = embedding_modulation(x, scale, shift)
        x = self.output(x)
        x = rearrange(x, "n (h w) c -> n h w c", h=h)
        return x


# model = DiTBLock(n_layers=3, embed_dim=10, n_heads=2, use_flash_attention=False, latent_size=8)

# img = jnp.ones((128,8,8,10))
# img_pos = model.create_2d_sinusoidal_pos(16,16)
# timesteps = jnp.ones([128]).astype(jnp.int32)
# cond = jnp.ones([128]).astype(jnp.int32)
# model_params = model.init(jax.random.PRNGKey(2), img, cond, timesteps)
# print()
