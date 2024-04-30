from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from torch.utils.checkpoint import checkpoint

def space_to_depth(x, h=2, w=2):
    return rearrange(x, '... c (h dh) (w dw) -> ... (c dh dw) h w', dh=h, dw=w)


# pixel shuffle
def depth_to_space(x, h=2, w=2):
    return rearrange(x, '... (c dh dw) h w  -> ... c (h dh) (w dw)', dh=h, dw=w)


def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype)**2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


def create_sinusoidal_positions_fn(pos, dim=768, freq_scale=None, concat_sin_cos=True):
    assert dim % 2 == 0
    hidden_dim_pos = torch.arange(dim//2)
    hidden_dim_pos = hidden_dim_pos / dim * 2
    if freq_scale is None:
        inv_freq = 1.0 / ((dim//10) ** hidden_dim_pos)
    else:
        inv_freq = 1.0 / (freq_scale ** hidden_dim_pos)
    freqs = torch.einsum("i, j -> i j", pos, inv_freq)
    if concat_sin_cos:
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)
    else:
        return torch.sin(freqs), torch.cos(freqs)
    

def create_1d_sinusoidal_positions_fn(num_pos=1024, dim=768, freq_scale=10000):
    return create_sinusoidal_positions_fn(torch.arange(num_pos), dim, freq_scale)


def create_2d_sinusoidal_positions_fn(embed_dim, grid_size, freq_scale=None, relative_center=False):
    if freq_scale is None:
        freq_scale = embed_dim // 10
    if relative_center:
        h = torch.linspace(1, -1, grid_size) * grid_size / 2
        w = torch.linspace(1, -1, grid_size) * grid_size / 2
    else:
        h = torch.arange(grid_size, dtype=torch.float32)
        w = torch.arange(grid_size, dtype=torch.float32)

    w, h = torch.meshgrid(w, h)
    pos_w_sin, pos_w_cos = create_sinusoidal_positions_fn(w.reshape(-1), embed_dim, freq_scale, False)
    pos_h_sin, pos_h_cos = create_sinusoidal_positions_fn(h.reshape(-1), embed_dim, freq_scale, False)
    return torch.cat([pos_h_sin, pos_w_sin, pos_h_cos, pos_w_cos], dim=-1)


def rotate_half(tensor):
    """Rotates half the hidden dims of the input."""
    rotate_half_tensor = torch.cat(
        (-tensor[..., tensor.shape[-1] // 2 :], tensor[..., : tensor.shape[-1] // 2]), dim=-1
    )
    return rotate_half_tensor


def apply_rotary_pos_emb(tensor, pos):
    sin_pos, cos_pos = pos.view(*pos.shape[:-1], -1, 2).permute(3, 0, 1, 2)
    return (tensor * cos_pos) + (rotate_half(tensor) * sin_pos)


def embedding_modulation(x, scale, shift):
    # basically applies time embedding and conditional to each input for each layer
    return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    def __init__(self, shape, epsilon=1e-6):
        super().__init__()
        self.eps = epsilon
        self.scale = nn.Parameter(torch.ones(shape))

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)
    

class SelfAttention(nn.Module):
    def __init__(self, embed_dim=768, cond_dim=768, n_heads=8, expansion_factor=1, use_bias=False, eps=1e-6, downsample_kv=False, use_checkpoint=False):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.expansion_factor = expansion_factor
        self.use_bias = use_bias
        self.eps = eps
        self.downsample_kv = downsample_kv
        self.cond_dim = cond_dim
        self.use_checkpoint = use_checkpoint

        self.rms_norm = RMSNorm(shape=self.embed_dim, epsilon=self.eps)
        self.cond_projection = nn.Linear(self.cond_dim, self.embed_dim * 2, bias=self.use_bias)
        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * self.expansion_factor * 3, bias=self.use_bias)
        self.out = nn.Linear(self.embed_dim * self.expansion_factor, self.embed_dim, bias=self.use_bias)

    def forward(self, x, cond, pos=None, og_image_shape=None):
        n, c, h, w = og_image_shape
        residual = x
        x = self.rms_norm(x)
        scale, shift = rearrange(self.cond_projection(cond), "b l (d split) -> split b l d", split=2)
        x = embedding_modulation(x, scale, shift)
        # query, key, value
        q, k, v = rearrange(checkpoint(self.qkv, x) if self.use_checkpoint else self.qkv(x), "b l (d split) -> split b l d", split=3)

        # [batch_size, num_heads, q_seq_len, d_model]
        q = rearrange(q, "b l (d n_head) -> b n_head l d", n_head=self.n_heads)
        # [batch_size, num_heads, kv_seq_len, d_model]
        k = rearrange(k, "b l (d n_head) -> b n_head l d", n_head=self.n_heads)
        # [batch_size, num_heads, kv_seq_len, d_model]
        v = rearrange(v, "b l (d n_head) -> b n_head l d", n_head=self.n_heads)
        if pos is not None:
            q = apply_rotary_pos_emb(q, pos)
            k = apply_rotary_pos_emb(k, pos)

        if self.downsample_kv:
            raise NotImplementedError
            k = k.view(-1, h, w, self.n_heads, self.embed_dim // self.n_heads * self.expansion_factor)
            v = v.view(-1, h, w, self.n_heads, self.embed_dim // self.n_heads * self.expansion_factor)
            k = F.interpolate(k, size=(h//2, w//2), mode='linear', align_corners=True)
            v = F.interpolate(v, size=(h//2, w//2), mode='linear', align_corners=True)
            k = k.view(-1, k.shape[-3] * k.shape[-2], self.n_heads, k.shape[-1])
            v = v.view(-1, v.shape[-3] * v.shape[-2], self.n_heads, v.shape[-1])

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b n_head l d -> b l (n_head d)")
        if self.use_checkpoint:
            x = checkpoint(self.out, out)
        else:
            x = self.out(out)
        return x + residual
    

class GLU(nn.Module):
    def __init__(self, embed_dim=768, cond_dim=768, expansion_factor=4, use_bias=False, eps=1e-6, cond=True, use_checkpoint=False):
        super(GLU, self).__init__()
        self.embed_dim = embed_dim
        self.expansion_factor = expansion_factor
        self.use_bias = use_bias
        self.eps = eps
        self.cond = cond
        self.cond_dim = cond_dim

        if use_checkpoint:
            raise NotImplementedError

        self.rms_norm = RMSNorm(shape=self.embed_dim, epsilon=self.eps)
        if self.cond:
            self.cond_projection = nn.Linear(self.cond_dim, self.embed_dim * 2, bias=self.use_bias)
        self.up_proj = nn.Linear(self.embed_dim, int(self.embed_dim * self.expansion_factor), bias=False)
        self.gate_proj = nn.Linear(self.embed_dim, int(self.embed_dim * self.expansion_factor), bias=False)
        self.down_proj = nn.Linear(int(self.embed_dim * self.expansion_factor), self.embed_dim, bias=False)

    def forward(self, x, cond=None):
        residual = x
        x = self.rms_norm(x)
        if self.cond:
            scale, shift = rearrange(self.cond_projection(cond), "b l (d split) -> split b l d", split=2)
            x = embedding_modulation(x, scale, shift)
        x = self.down_proj((self.up_proj(x) * F.silu(self.gate_proj(x))))
        return x + residual
    

class FourierLayers(nn.Module):
    def __init__(self, in_features=1, features=768, keep_random=False):
        super(FourierLayers, self).__init__()
        self.features = features
        self.keep_random = keep_random
        self.freq = nn.Linear(in_features, features // 2, bias=False)

    def forward(self, timesteps):
        if self.keep_random:
            freq = torch.stop_gradient(torch.randn_like(timesteps) * timesteps * torch.pi * 2)
        else:
            freq = self.freq(timesteps * torch.pi * 2)
        return torch.cat([torch.sin(freq), torch.cos(freq)], dim=-1)
    

class DiTBLockContinuous(nn.Module):
    def __init__(
        self,
        sigma_data: float = 0.5,
        embed_dim: int = 768,
        output_dim: int = 256,
        attn_expansion_factor: int = 2,
        glu_expansion_factor: int = 4,
        use_bias: bool = False,
        eps:float = 1e-6,
        n_heads: int = 8,
        n_layers: int = 24,
        n_time_embed_layers: int = 3,
        cond_embed_dim: int = 768,
        time_embed_expansion_factor: int = 2,
        diffusion_timesteps: int = 1000,
        use_flash_attention: bool = False,
        n_class: int = 1000,
        latent_size: int = 8,
        latent_dim: int = 64,
        pixel_based: bool = False,
        random_fourier_features: bool = False,
        use_rope: bool = True,
        patch_size: int = 1,
        downsample_kv: bool = False,
        split_time_embed: int = 1,
        denseformers: bool = False,
        checkpoint_glu: bool = False,
        checkpoint_attn: bool = False,
        rope_device: str = "cpu"

    ):
        super(DiTBLockContinuous, self).__init__()
        self.sigma_data = sigma_data 
        self.embed_dim = embed_dim 
        self.output_dim = output_dim 
        self.attn_expansion_factor = attn_expansion_factor 
        self.glu_expansion_factor = glu_expansion_factor 
        self.use_bias = use_bias 
        self.eps = eps
        self.n_heads = n_heads 
        self.n_layers = n_layers 
        self.n_time_embed_layers = n_time_embed_layers 
        self.cond_embed_dim = cond_embed_dim 
        self.time_embed_expansion_factor = time_embed_expansion_factor 
        self.diffusion_timesteps = diffusion_timesteps 
        self.use_flash_attention = use_flash_attention 
        self.n_class = n_class 
        self.latent_size = latent_size 
        self.latent_dim = latent_dim
        self.pixel_based = pixel_based 
        self.random_fourier_features = random_fourier_features 
        self.use_rope = use_rope 
        self.patch_size = patch_size 
        self.downsample_kv = downsample_kv 
        self.split_time_embed = split_time_embed 
        self.denseformers = denseformers 
        self.checkpoint_glu = checkpoint_glu 
        self.checkpoint_attn = checkpoint_attn
        self.rope_device = rope_device

        self.linear_proj_input = nn.Linear(self.latent_dim, self.embed_dim, bias=self.use_bias)

        if self.latent_size is not None:
            self.latent_positional_embedding = self.create_2d_sinusoidal_pos(self.latent_size // self.patch_size, use_rope=self.use_rope, device=self.rope_device)

        self.time_embed = FourierLayers(in_features=1, features=self.cond_embed_dim, keep_random=self.random_fourier_features)

        if self.n_class is not None:
            self.class_embed = nn.Embedding(self.n_class, embedding_dim=self.cond_embed_dim)

        dit_blocks = []
        for layer in range(self.n_layers):
            attn = SelfAttention(
                embed_dim=self.embed_dim, 
                cond_dim=self.cond_embed_dim,
                n_heads=self.n_heads, 
                expansion_factor=self.attn_expansion_factor,
                use_bias=self.use_bias, 
                eps=self.eps,
                downsample_kv=self.downsample_kv,
                use_checkpoint=self.checkpoint_attn
            )
            glu = GLU(
                embed_dim=self.embed_dim, 
                cond_dim=self.cond_embed_dim,
                expansion_factor=self.glu_expansion_factor, 
                use_bias=self.use_bias, 
                eps=self.eps, 
                cond=True,
                use_checkpoint=self.checkpoint_glu
            )

            if self.denseformers:
                scaler_layers = [nn.Parameter(torch.ones(()))]
                scaler_layers += [nn.Parameter(torch.zeros(())) for _ in range(layer+1)]
            else:
                scaler_layers = []

            dit_blocks.append(nn.ModuleList([attn, glu, nn.ParameterList(scaler_layers)]))

        self.dit_blocks = nn.ModuleList(dit_blocks)

        time_embedding_stack = []
        for layer in range(self.n_time_embed_layers):
            glu_time = GLU(
                embed_dim=self.cond_embed_dim, 
                expansion_factor=self.time_embed_expansion_factor, 
                use_bias=self.use_bias, 
                eps=self.eps, 
                cond=False
            )
            time_embedding_stack.append(glu_time)
        self.time_embedding_stack = nn.ModuleList(time_embedding_stack)

        self.final_norm = RMSNorm(shape=self.embed_dim, epsilon=self.eps)

        if self.pixel_based:
            self.output = nn.Linear(in_features=self.embed_dim, out_features=3 * self.patch_size * 2 if self.patch_size > 1 else 3, bias=self.use_bias)
        else:
            self.output = nn.Linear(in_features=self.embed_dim, out_features=self.output_dim * self.patch_size * 2 if self.patch_size > 1 else self.output_dim, bias=self.use_bias)

    def create_2d_sinusoidal_pos(self, width_len=8, use_rope=True, device=None):
        if use_rope:
            pos = create_2d_sinusoidal_positions_fn(self.embed_dim * self.attn_expansion_factor // self.n_heads, width_len, freq_scale=None, relative_center=False)
        else:
            pos = create_2d_sinusoidal_positions_fn(self.embed_dim // 2, width_len, freq_scale=None, relative_center=False)
        return pos[None,...].to(device=device)  # add batch dim HWC -> NHWC

    def rectified_flow_loss(self, images, timesteps, conds, image_pos=None, extra_pos=None, toggle_cond=None):
        noises = torch.randn_like(images)
        noise_to_image_flow = noises * timesteps[:, None, None, None] + images * (1 - timesteps[:, None, None, None])
        flow_path = noises - images
        model_trajectory_predictions = self.forward(noise_to_image_flow, timesteps, conds, image_pos, extra_pos, toggle_cond)
        loss = torch.mean((model_trajectory_predictions - flow_path) ** 2)
        return loss, (model_trajectory_predictions, model_trajectory_predictions - flow_path, flow_path, images)

    def forward(self, x, timestep, cond=None, image_pos=None, extra_pos=None, toggle_cond=None):
        if self.patch_size > 1:
            x = space_to_depth(x, self.patch_size, self.patch_size)
        n, c, h, w = x.shape
        x = rearrange(x, "n c h w -> n (h w) c")
        x = self.linear_proj_input(x)

        if not self.use_rope:
            x = x  + self.latent_positional_embedding.detach()

        time = self.time_embed(timestep[:, None])[:, None, :]
        if self.n_class and cond is not None:
            class_cond = self.class_embed(cond)[:, None, :]
            if toggle_cond is not None:
                cond = time + class_cond * toggle_cond[:, None, None]
            else:
                cond = time + class_cond
        else:
            cond = time

        for glu_time in self.time_embedding_stack:
            cond = glu_time(cond)

        if self.split_time_embed > 1:
            cond_split = rearrange(x, "b l (d split) -> split b l d", split=self.split_time_embed)

        if self.denseformers:
            skips = [x]         
        for i, (attn, glu, skip_scaler) in enumerate(self.dit_blocks):
            if self.split_time_embed > 1:
                cond = cond_split[i % self.split_time_embed]

            x = attn(x, cond, self.latent_positional_embedding.detach() if self.use_rope else None, [n, h, w, c])
            x = glu(x, cond)

            if self.denseformers:
                skips.append(x)
                x *= skip_scaler[0]
                for skip, scaler in zip(skips[:i], skip_scaler[1:]):
                    x += skip * scaler

        x = self.final_norm(x)
        x = self.output(x)
        x = rearrange(x, "n (h w) c -> n c h w", h=h)
        if self.patch_size > 1:
            x = depth_to_space(x, self.patch_size, self.patch_size)
        return x