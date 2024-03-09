import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
import math
from typing import Tuple, Tuple
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention


# yoinked from hf transformers
# for rope and time embed
def create_sinusoidal_positions_fn(num_pos=1000, dim=768):
    with jax.default_device(jax.devices("cpu")[0]):
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2)[: (dim // 2)] / dim))
        freqs = jnp.einsum("i , j -> i j", jnp.arange(num_pos), inv_freq).astype("float32")

        emb = jnp.concatenate((freqs, freqs), axis=-1)
        return jnp.sin(emb)


def create_2d_sinusoidal_positions_fn(width_len=8, height_len=8, dim=768, max_freq=None, epsilon=1e-6):
    with jax.default_device(jax.devices("cpu")[0]):

        # max frequency must be at most a half of your width and height whichever come first
        # nyquist sampling theorem thingy
        # to interpolate width and height, keep the max freq fixed 
        if max_freq is None:
            max_freq = height_len / 2 if height_len < width_len else width_len / 2

        freqs = jnp.linspace(epsilon, max_freq, dim)

        # stack of position embedding for each dim
        sinusoidal_2d = []

        for freq in freqs:
            # create wavy plane
            height = jnp.sin(jnp.stack([jnp.linspace(1, -1, height_len)] * width_len, axis=0).T * freq)
            width = jnp.cos(jnp.stack([jnp.linspace(1, -1, width_len)] * height_len, axis=0) * freq)
            # slant it diagonally
            height_slant = jnp.stack([jnp.linspace(1, -1, height_len)] * width_len, axis=0).T
            width_slant = jnp.stack([jnp.linspace(1, -1, width_len)] * height_len, axis=0)
            # combine it
            compound = height - width + height_slant + width_slant
            # rescale it
            compound = (compound- compound.min())/(compound.max()-compound.min())
            sinusoidal_2d.append(compound)
        
        sinusoidal_2d = jnp.stack(sinusoidal_2d, axis=-1)
        return sinusoidal_2d


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

        # pointwise conv reformulated as matmul
        self.qkv = nn.Dense(
            features=self.embed_dim * self.expansion_factor * 3, 
            use_bias=self.use_bias
        )
        self.out = nn.Dense(
            features=self.embed_dim, 
            use_bias=self.use_bias
        )

    def __call__(self, x, pos=None):

        # store input as residual identity
        residual = x
        x = self.rms_norm(x)
        # query, key, value
        q, k, v = rearrange(self.qkv(x), "b l (d split) -> split b l d", split=3)
        # if using rotary
        if pos is not None:
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

        return  x + residual
    

class GLU(nn.Module):
    embed_dim: int = 768
    expansion_factor: int = 4
    use_bias: bool = False

    def setup(self):
        # inverted bottleneck with gating
        # practically mimicing mobilenet except only pointwise conv
        self.up_proj = nn.Dense(self.embed_dim * self.expansion_factor, use_bias=False)
        self.gate_proj = nn.Dense(self.embed_dim * self.expansion_factor, use_bias=False)
        self.down_proj = nn.Dense(self.embed_dim, use_bias=False)

    def __call__(self, x):
        residual = x
        x = self.down_proj(self.up_proj(x) * nn.silu(self.gate_proj(x)))
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

    def setup(self):
        
        if self.latent_size is not None:
            self.latent_positional_embedding = self.create_2d_sinusoidal_pos(self.latent_size, self.latent_size)
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
            glu = GLU(embed_dim=self.embed_dim, expansion_factor=self.glu_expansion_factor, use_bias=self.use_bias)

            dit_blocks.append([attn, glu])

        self.dit_blocks = dit_blocks

        # time embedding
        time_embedding_stack = [] 
        for layer in range(self.n_layers):
            # overkill but eh whatever
            glu_time = GLU(embed_dim=self.embed_dim, expansion_factor=self.time_embed_expansion_factor, use_bias=self.use_bias)

            time_embedding_stack.append(glu_time)
        self.time_embedding_stack = time_embedding_stack
        
    def create_2d_sinusoidal_pos(self, width_len=8, height_len=8):

        pos = create_2d_sinusoidal_positions_fn(
            width_len=width_len, 
            height_len=height_len, 
            dim=self.embed_dim, 
            max_freq=self.embed_dim // 2, 
            epsilon=self.eps
        )
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

        # use default image position latent if not provided
        if self.latent_size is not None:
            x = x + self.latent_positional_embedding
        else:
            x = x + image_pos
        x = rearrange(x, "n h w c-> n (h w) c")
        # time loop 
        time = self.time_embed(timestep) # grab the time
        if self.n_class is not None:
            class_cond = self.class_embed(cond)
        # mebbe ditching this loop is simpler 
        # simple time embedding vector should suffice
        # silly ideas, rescale the input vector relative to this 
        # so the model pay more attention to the timesteps or class
        for glu_time in self.time_embedding_stack:
            time = glu_time(time)

        # TODO: add rope

        if self.n_class is not None:
            x = jnp.concatenate([time[:,None,:], class_cond[:,None,:], x], axis=-2) # put time embedding in the seq dim
        else:
            x = jnp.concatenate([time[:,None,:], x], axis=-2) # put time embedding in the seq dim
        # attention block loop
        for attn, glu in self.dit_blocks:
            x = attn(x, extra_pos) # NOTE: if using this pos put image token in 1 position!
            x = glu(x)
        if self.n_class is not None:
            x = x[:,2:,:] # pop time and class embedding
        else:
            x = x[:,1:,:] # pop time embedding
        x = rearrange(x, "n (h w) c -> n h w c", h=h)
        return x

# model = DiTBLock(n_layers=3, embed_dim=10, n_heads=2, use_flash_attention=False, latent_size=8)

# img = jnp.ones((128,8,8,10))
# img_pos = model.create_2d_sinusoidal_pos(16,16)
# timesteps = jnp.ones([128]).astype(jnp.int32)
# cond = jnp.ones([128]).astype(jnp.int32)
# model_params = model.init(jax.random.PRNGKey(2), img, cond, timesteps)
# print()
