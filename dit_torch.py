from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

def space_to_depth(x, h=2, w=2):
    return rearrange(x, '... (h dh) (w dw) c -> ... h w (c dh dw)', dh=h, dw=w)


# pixel shuffle
def depth_to_space(x, h=2, w=2):
    return rearrange(x, '... h w (c dh dw) -> ... (h dh) (w dw) c', dh=h, dw=w)


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

    h, w = torch.meshgrid(w, h)
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


class SelfAttention(nn.Module):
    def __init__(self, embed_dim=768, n_heads=8, expansion_factor=1, use_bias=False, eps=1e-6, use_flash_attention=False, downsample_kv=False):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.expansion_factor = expansion_factor
        self.use_bias = use_bias
        self.eps = eps
        self.use_flash_attention = use_flash_attention
        self.downsample_kv = downsample_kv

        self.rms_norm = nn.RMSNorm(epsilon=self.eps)
        self.cond_projection = nn.Linear(self.embed_dim * 2, self.embed_dim * 2, bias=self.use_bias)
        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * self.expansion_factor * 3, bias=self.use_bias)
        self.out = nn.Linear(self.embed_dim * self.expansion_factor, self.embed_dim, bias=self.use_bias)

    def forward(self, x, cond, pos=None, og_image_shape=None):
        n, h, w, c = og_image_shape
        residual = x
        x = self.rms_norm(x)
        scale, shift = self.cond_projection(cond).view(-1, 2, self.embed_dim).permute(1, 0, 2)
        x = embedding_modulation(x, scale, shift)
        q, k, v = self.qkv(x).view(-1, 3, self.embed_dim * self.expansion_factor).permute(1, 0, 2)

        if self.use_flash_attention:
            q = q.view(-1, self.n_heads, q.shape[-2], q.shape[-1])
            k = k.view(-1, self.n_heads, k.shape[-2], k.shape[-1])
            v = v.view(-1, self.n_heads, v.shape[-2], v.shape[-1])
            if pos is not None:
                q = apply_rotary_pos_emb(q, pos)
                k = apply_rotary_pos_emb(k, pos)
            out = F.scaled_dot_product_attention(q, k, v, dropout=0.0)
            out = out.view(-1, out.shape[-2], out.shape[-1] * self.n_heads)
        else:
            q = q.view(-1, q.shape[-2], self.n_heads, q.shape[-1])
            k = k.view(-1, k.shape[-2], self.n_heads, k.shape[-1])
            v = v.view(-1, v.shape[-2], self.n_heads, v.shape[-1])
            if pos is not None:
                q = apply_rotary_pos_emb(q, pos)
                k = apply_rotary_pos_emb(k, pos)
            if self.downsample_kv:
                k = k.view(-1, h, w, self.n_heads, self.embed_dim // self.n_heads * self.expansion_factor)
                v = v.view(-1, h, w, self.n_heads, self.embed_dim // self.n_heads * self.expansion_factor)
                k = F.interpolate(k, size=(h//2, w//2), mode='linear', align_corners=True)
                v = F.interpolate(v, size=(h//2, w//2), mode='linear', align_corners=True)
                k = k.view(-1, k.shape[-3] * k.shape[-2], self.n_heads, k.shape[-1])
                v = v.view(-1, v.shape[-3] * v.shape[-2], self.n_heads, v.shape[-1])
            out = F.scaled_dot_product_attention(q, k, v, dropout=0.0)
            out = out.view(-1, out.shape[-2], out.shape[-1] * self.n_heads)

        x = self.out(out)
        return x + residual
    

class GLU(nn.Module):
    def __init__(self, embed_dim=768, expansion_factor=4, use_bias=False, eps=1e-6, cond=True, checkpoint=False):
        super(GLU, self).__init__()
        self.embed_dim = embed_dim
        self.expansion_factor = expansion_factor
        self.use_bias = use_bias
        self.eps = eps
        self.cond = cond
        self.checkpoint = checkpoint

        self.rms_norm = nn.RMSNorm(epsilon=self.eps)
        if self.cond:
            self.cond_projection = nn.Linear(self.embed_dim * 2, self.embed_dim * 2, bias=self.use_bias)
        if self.checkpoint:
            self.up_proj = nn.utils.checkpoint(nn.Linear)(int(self.embed_dim * self.expansion_factor), bias=False)
            self.gate_proj = nn.utils.checkpoint(nn.Linear)(int(self.embed_dim * self.expansion_factor), bias=False)
        else:
            self.up_proj = nn.Linear(int(self.embed_dim * self.expansion_factor), bias=False)
            self.gate_proj = nn.Linear(int(self.embed_dim * self.expansion_factor), bias=False)
        self.down_proj = nn.Linear(self.embed_dim, bias=False)

    def forward(self, x, cond=None):
        residual = x
        x = self.rms_norm(x)
        if self.cond:
            scale, shift = self.cond_projection(cond).view(-1, 2, self.embed_dim).permute(1, 0, 2)
            x = embedding_modulation(x, scale, shift)
        x = self.down_proj(F.silu(self.up_proj(x) * F.silu(self.gate_proj(x))))
        return x + residual
    

class FourierLayers(nn.Module):
    def __init__(self, features=768, keep_random=False):
        super(FourierLayers, self).__init__()
        self.features = features
        self.keep_random = keep_random
        self.freq = nn.Linear(features // 2, bias=False)

    def forward(self, timesteps):
        if self.keep_random:
            freq = torch.stop_gradient(torch.randn_like(timesteps) * timesteps * torch.pi * 2)
        else:
            freq = self.freq(timesteps * torch.pi * 2)
        return torch.cat([torch.sin(freq), torch.cos(freq)], dim=-1)
    

class DiTBLockContinuous(nn.Module):
    def __init__(self, **kwargs):
        super(DiTBLockContinuous, self).__init__()
        self.sigma_data = kwargs.get('sigma_data', 0.5)
        self.embed_dim = kwargs.get('embed_dim', 768)
        self.output_dim = kwargs.get('output_dim', 256)
        self.attn_expansion_factor = kwargs.get('attn_expansion_factor', 2)
        self.glu_expansion_factor = kwargs.get('glu_expansion_factor', 4)
        self.use_bias = kwargs.get('use_bias', False)
        self.eps = kwargs.get('eps', 1e-6)
        self.n_heads = kwargs.get('n_heads', 8)
        self.n_layers = kwargs.get('n_layers', 24)
        self.n_time_embed_layers = kwargs.get('n_time_embed_layers', 3)
        self.cond_embed_dim = kwargs.get('cond_embed_dim', 768)
        self.time_embed_expansion_factor = kwargs.get('time_embed_expansion_factor', 2)
        self.diffusion_timesteps = kwargs.get('diffusion_timesteps', 1000)
        self.use_flash_attention = kwargs.get('use_flash_attention', False)
        self.n_class = kwargs.get('n_class', 1000)
        self.latent_size = kwargs.get('latent_size', 8)
        self.pixel_based = kwargs.get('pixel_based', False)
        self.random_fourier_features = kwargs.get('random_fourier_features', False)
        self.use_rope = kwargs.get('use_rope', True)
        self.patch_size = kwargs.get('patch_size', 1)
        self.downsample_kv = kwargs.get('downsample_kv', False)
        self.split_time_embed = kwargs.get('split_time_embed', 1)
        self.denseformers = kwargs.get('denseformers', False)
        self.checkpoint_glu = kwargs.get('checkpoint_glu', False)

        self.linear_proj_input = nn.Linear(self.embed_dim, bias=self.use_bias)

        if self.latent_size is not None:
            self.latent_positional_embedding = self.create_2d_sinusoidal_pos(self.latent_size // self.patch_size, use_rope=self.use_rope)

        self.time_embed = FourierLayers(features=self.cond_embed_dim, keep_random=self.random_fourier_features)

        if self.n_class is not None:
            self.class_embed = nn.Embedding(self.n_class, features=self.cond_embed_dim)

        dit_blocks = []
        for layer in range(self.n_layers):
            attn = SelfAttention(n_heads=self.n_heads, expansion_factor=self.attn_expansion_factor, use_bias=self.use_bias, eps=self.eps, embed_dim=self.embed_dim, use_flash_attention=self.use_flash_attention, downsample_kv=self.downsample_kv)
            glu = GLU(embed_dim=self.embed_dim, expansion_factor=self.glu_expansion_factor, use_bias=self.use_bias, eps=self.eps, checkpoint=self.checkpoint_glu)

            if self.denseformers:
                scaler_layers = [nn.Parameter(torch.ones(())) for _ in range(layer+1)]
            else:
                scaler_layers = None

            dit_blocks.append([attn, glu, scaler_layers])

        self.dit_blocks = nn.ModuleList(dit_blocks)

        time_embedding_stack = []
        for layer in range(self.n_time_embed_layers):
            glu_time = GLU(embed_dim=self.cond_embed_dim, expansion_factor=self.time_embed_expansion_factor, use_bias=self.use_bias, eps=self.eps, cond=False)
            time_embedding_stack.append(glu_time)
        self.time_embedding_stack = nn.ModuleList(time_embedding_stack)

        self.final_norm = nn.RMSNorm(epsilon=self.eps)

        if self.pixel_based:
            self.output = nn.Linear(features=3 * self.patch_size * 2 if self.patch_size > 1 else 3, bias=self.use_bias)
        else:
            self.output = nn.Linear(features=self.output_dim * self.patch_size * 2 if self.patch_size > 1 else self.output_dim, bias=self.use_bias)

    def create_2d_sinusoidal_pos(self, width_len=8, use_rope=True):
        if use_rope:
            pos = create_2d_sinusoidal_positions_fn(self.embed_dim * self.attn_expansion_factor // self.n_heads, width_len, freq_scale=None, relative_center=False)
        else:
            pos = create_2d_sinusoidal_positions_fn(self.embed_dim // 2, width_len, freq_scale=None, relative_center=False)
        return pos[None,...]  # add batch dim HWC -> NHWC

    def sigma_scaling(self, timesteps):
        c_skip = self.sigma_data ** 2 / (timesteps ** 2 + self.sigma_data ** 2)
        c_out = timesteps * self.sigma_data / (timesteps ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (timesteps ** 2 + self.sigma_data ** 2) ** 0.5
        soft_weighting = (timesteps * self.sigma_data) ** 2 / (timesteps ** 2 + self.sigma_data ** 2) ** 2
        return c_skip, c_out, c_in, soft_weighting

    def loss(self, rng_key, model_apply_fn, model_params, images, timesteps, conds, image_pos=None, extra_pos=None):
        c_skip, c_out, c_in, soft_weighting = [scale[:, None, None, None] for scale in self.sigma_scaling(timesteps)]
        noises = torch.randn_like(images)
        noised_images = (images + noises * timesteps[:, None, None, None])
        model_predictions = model_apply_fn(model_params, noised_images * c_in, timesteps, conds, image_pos, extra_pos)
        target_predictions = (images - noised_images * c_skip) / c_out
        loss = torch.mean((model_predictions - target_predictions) ** 2, axis=[1, 2, 3]) * soft_weighting.reshape(-1)
        return torch.mean(loss), (model_predictions, target_predictions, model_predictions - target_predictions, noised_images, images)

    def pred(self, model_apply_fn, model_params, images, timesteps, conds, image_pos=None, extra_pos=None):
        c_skip, c_out, c_in, _ = [scale[:, None, None, None] for scale in self.sigma_scaling(timesteps)]
        return model_apply_fn(model_params, images * c_in, timesteps, conds, image_pos, extra_pos) * c_out + images * c_skip

    def rectified_flow_loss(self, rng_key, model_apply_fn, model_params, images, timesteps, conds, image_pos=None, extra_pos=None, toggle_cond=None):
        noises = torch.randn_like(images)
        noise_to_image_flow = noises * timesteps[:, None, None, None] + images * (1 - timesteps[:, None, None, None])
        flow_path = noises - images
        model_trajectory_predictions = model_apply_fn(model_params, noise_to_image_flow, timesteps, conds, image_pos, extra_pos, toggle_cond)
        loss = torch.mean((model_trajectory_predictions - flow_path) ** 2)
        return loss, (model_trajectory_predictions, model_trajectory_predictions - flow_path, flow_path, images)

    def pred_rectified_flow(self, images, timesteps, conds, model_apply_fn, model_params, image_pos=None, extra_pos=None, toggle_cond=None):
        return model_apply_fn(model_params, images, timesteps, conds, image_pos, extra_pos, toggle_cond)

    def forward(self, x, timestep, cond=None, image_pos=None, extra_pos=None, toggle_cond=None):
        if self.patch_size > 1:
            x = space_to_depth(x, self.patch_size, self.patch_size)
        n, h, w, c = x.shape
        x = x.view(n, h * w, c)
        x = self.linear_proj_input(x)

        if not self.use_rope:
            x += torch.stop_gradient(self.latent_positional_embedding)

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
            cond_split = cond.view(-1, self.split_time_embed, cond.shape[-1])

        for i, (attn, glu, skip_scaler) in enumerate(self.dit_blocks):
            if self.split_time_embed > 1:
                cond = cond_split[i % self.split_time_embed]

            x = attn(x, cond, torch.stop_gradient(self.latent_positional_embedding) if self.use_rope else None, [n, h, w, c])
            x = glu(x, cond)

            if self.denseformers:
                x *= skip_scaler[0]
                for skip, scaler in zip(self.dit_blocks[:i], skip_scaler[1:]):
                    x += skip * scaler

        x = self.final_norm(x)
        x = self.output(x)
        x = x.view(n, h, w, c)
        if self.patch_size > 1:
            x = depth_to_space(x, self.patch_size, self.patch_size)
        return x