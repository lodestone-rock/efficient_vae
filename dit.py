import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
import math
from typing import Tuple, Tuple
from jax.experimental.pallas.ops.tpu.flash_attention import flash_attention


# yoinked from hf transformers
def create_sinusoidal_positions(num_pos, dim):
    with jax.default_device(jax.devices("cpu")[0]):
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2)[: (dim // 2)] / dim))
        freqs = jnp.einsum("i , j -> i j", jnp.arange(num_pos), inv_freq).astype("float32")

        emb = jnp.concatenate((freqs, freqs), axis=-1)
        out = jnp.concatenate((jnp.sin(emb)[:, None, :], jnp.cos(emb)[:, None, :]), axis=-1)
        return out[:, :, :num_pos]


# yoinked from hf transformers
def rotate_half(tensor):
    """Rotates half the hidden dims of the input."""
    rotate_half_tensor = jnp.concatenate(
        (-tensor[..., tensor.shape[-1] // 2 :], tensor[..., : tensor.shape[-1] // 2]), axis=-1
    )
    return rotate_half_tensor


# yoinked from hf transformers
def apply_rotary_pos_emb(tensor, sin_pos, cos_pos):
    return (tensor * cos_pos) + (rotate_half(tensor) * sin_pos)


class SelfAttention(nn.Module):
    # TODO: 
    features: int
    n_heads: int = 8
    expansion_factor: int = 1
    use_bias: bool = False # removing bias won't affect the model much
    eps:float = 1e-6
    embed_dim: int = 768

    def setup(self):

        self.rms_norm = nn.RMSNorm(
            epsilon=self.eps, 
            dtype=jnp.float32,
            param_dtype=jnp.float32
        )

        # pointwise conv reformulated as matmul
        self.qkv = nn.Dense(
            features=self.features * self.expansion_factor * 3, 
            use_bias=self.use_bias
        )
        self.out = nn.Dense(
            features=self.features, 
            use_bias=self.use_bias
        )

    def __call__(self, x,):
        # NOTE: flax uses NHWC convention so the entire ops is in NHWC
        # store input as residual identity
        residual = x
        x = self.rms_norm(x)
        # query, key, value
        q, k, v = rearrange(self.qkv(x), "n h w (c split) -> split n h w c", split=3)
        # merge height and weight and treat it as a token sequence
        # NHWC => BLD 
        # [batch_size, num_heads, q_seq_len, d_model]
        q = rearrange(q, "n h w (c n_head) -> n n_head (h w) c", n_head=self.n_head)
        # [batch_size, num_heads, kv_seq_len, d_model]
        k = rearrange(k, "n h w (c n_head) -> n n_head (h w) c", n_head=self.n_head)
        # [batch_size, num_heads, kv_seq_len, d_model]
        v = rearrange(v, "n h w (c n_head) -> n n_head (h w) c", n_head=self.n_head)
        # use pallas flash attention kernel
        out = flash_attention(q,k,v)
        # output projection
        x = self.out(x)
        return  x + residual
    

class GLU(nn.Module):
    embed_dim: int = 768
    inner_dim_multiplier: int = 4
    use_bias: bool = False

    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # inverted bottleneck with gating
        # practically mimicing mobilenet except only pointwise conv
        self.up_proj = nn.Dense(self.embed_dim * self.inner_dim_multiplier, use_bias=False, dtype=self.dtype)
        self.gate_proj = nn.Dense(self.embed_dim * self.inner_dim_multiplier, use_bias=False, dtype=self.dtype)
        self.down_proj = nn.Dense(self.embed_dim, use_bias=False, dtype=self.dtype)

    def __call__(self, x):
        residual = x
        x = self.down_proj(self.up_proj(x) * nn.silu(self.gate_proj(x)))
        return x + residual