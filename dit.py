import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
import math
from typing import Tuple, Tuple
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention


# for rope, time, and 2d pos
def create_sinusoidal_positions_fn(pos, dim=768, freq_scale=None):
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

        return jnp.concatenate([jnp.cos(freqs), jnp.sin(freqs)],axis=-1)
    

def create_1d_sinusoidal_positions_fn(num_pos=1024, dim=768, freq_scale=10000):
    with jax.default_device(jax.devices("cpu")[0]):
        return create_sinusoidal_positions_fn(jnp.arange(num_pos), dim, freq_scale)


def create_2d_sinusoidal_positions_fn(embed_dim, grid_size, freq_scale=None, relative_center=False, for_rope=False):
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

    pos_w = create_sinusoidal_positions_fn(w.reshape(-1), embed_dim // 2, freq_scale)
    pos_h = create_sinusoidal_positions_fn(h.reshape(-1), embed_dim // 2, freq_scale)
    if for_rope:
        # just use sine part as theta angle rotation
        pos_h = jnp.concatenate([pos_h[..., :embed_dim // 4]] * 2, axis = -1)
        pos_w = jnp.concatenate([pos_w[..., :embed_dim // 4]] * 2, axis = -1)

    return jnp.concatenate([pos_h, pos_w], axis=-1)


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
    sin_pos, cos_pos = pos
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
        if pos is not None:
            # pos is theta angle for rotary embedding
            q = apply_rotary_pos_emb(q, pos)
            k = apply_rotary_pos_emb(k, pos)
        if self.use_flash_attention:
            # [batch_size, num_heads, q_seq_len, d_model]
            q = rearrange(q, "b l (d n_head) -> b n_head l d", n_head=self.n_heads)
            # [batch_size, num_heads, kv_seq_len, d_model]
            k = rearrange(k, "b l (d n_head) -> b n_head l d", n_head=self.n_heads)
            # [batch_size, num_heads, kv_seq_len, d_model]
            v = rearrange(v, "b l (d n_head) -> b n_head l d", n_head=self.n_heads)
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
        if self.cond:
            scale, shift, gate = rearrange(self.cond_projection(cond), "b l (d split) -> split b l d", split=3)
            x = embedding_modulation(x, scale, shift)
        x = self.rms_norm(x)
        x = self.down_proj(self.up_proj(x) * nn.silu(self.gate_proj(x)))
        if self.cond:
            return x + residual + gate
        else:
            return x + residual


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
    
    def create_sinusoidal_rope_pos(self):
        # theta pos for sine and cosine rope
        pos = create_sinusoidal_positions_fn(
            num_pos=self.diffusion_timesteps,
            dim=self.embed_dim,
        )
        return pos


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

# model = DiTBLock(n_layers=3, embed_dim=10, n_heads=2, use_flash_attention=False, latent_size=8)

# img = jnp.ones((128,8,8,10))
# img_pos = model.create_2d_sinusoidal_pos(16,16)
# timesteps = jnp.ones([128]).astype(jnp.int32)
# cond = jnp.ones([128]).astype(jnp.int32)
# model_params = model.init(jax.random.PRNGKey(2), img, cond, timesteps)
# print()
