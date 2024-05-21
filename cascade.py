import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
from typing import Tuple, Tuple


def rectified_flow_loss(rng_key, model, model_params, images, conditions, timesteps):
    noises = jax.random.normal(key=rng_key, shape=images.shape)
    noise_to_image_flow = noises * timesteps[:, None, None, None] + images * (1-timesteps[:, None, None, None]) # lerp
    flow_path = noises - images # noise >>>>towards>>>> image
    model_trajectory_predictions = model(model_params, noise_to_image_flow, conditions, timesteps)

    loss = jnp.mean((model_trajectory_predictions - flow_path)** 2)
    return loss


class EfficientConvRMS(nn.Module):
    features: int
    kernel_size: int
    expansion_factor: int = 4 # inverted bottleneck scale factor to up project inner conv
    group_count: int = -1
    use_bias: bool = False # removing bias won't affect the model much
    eps:float = 1e-6
    residual: bool = True # toggle residual identity path (useful for first layer)
    checkpoint: bool = True

    def setup(self):
        if self.checkpoint:
            conv = nn.checkpoint(nn.Conv)
            dense = nn.checkpoint(nn.Dense)
        else:
            conv = nn.Conv
            dense = nn.Dense
        self.group_norm = nn.RMSNorm(
            epsilon=self.eps, 
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )
        # using classical conv on early layer will increase flops but make it faster to train 
        if self.group_count == -1:
            self.conv = conv(
                features=self.features * self.expansion_factor,
                kernel_size=(self.kernel_size, self.kernel_size), 
                strides=(1, 1),
                padding="SAME",
                feature_group_count=1,
                use_bias=self.use_bias,
            )
            pass
        else:
            # pointwise conv reformulated as matmul
            self.pointwise_expand = dense(
                features=self.features * self.expansion_factor, 
                use_bias=self.use_bias
            )
            self.depthwise = conv(
                features=self.features * self.expansion_factor,
                kernel_size=(self.kernel_size, self.kernel_size), 
                strides=(1, 1),
                padding="SAME",
                feature_group_count=self.group_count,
                use_bias=self.use_bias,
            )
    
        # activation
        # pointwise conv reformulated as matmul
        self.pointwise_contract = dense(
            features=self.features, 
            use_bias=self.use_bias,
            kernel_init=jax.nn.initializers.zeros
        )

    def __call__(self, x,):
        # NOTE: flax uses NHWC convention so the entire ops is in NHWC
        # store input as residual identity
        if self.residual:
            residual = x
        x = self.group_norm(x)
        # use conv for early layer if possible
        if self.group_count == -1:
            x = self.conv(x) # up-project & transform in higher dim manifold
            x = nn.silu(x)
        else:
            x = self.depthwise(x) # up-project
            x = self.pointwise_expand(x) # transform in higher dim manifold
            x = nn.silu(x) # nonlinearity 
            # projection back to input space
        x = self.pointwise_contract(x)

        if self.residual:
            x = x + residual
        return x


class SelfAttention(nn.Module):
    n_heads: int = 8
    expansion_factor: int = 1
    use_bias: bool = False # removing bias won't affect the model much
    eps:float = 1e-6
    embed_dim: int = 768
    use_flash_attention: bool = False
    downsample_kv: bool = False

    def setup(self):

        self.rms_norm = nn.RMSNorm(
            epsilon=self.eps, 
            dtype=jnp.float32,
            param_dtype=jnp.float32
        )
        # pointwise conv reformulated as matmul
        self.qkv = nn.Dense(
            features=int(self.embed_dim * self.expansion_factor * 3), 
            use_bias=self.use_bias
        )
        self.out = nn.Dense(
            features=self.embed_dim, 
            use_bias=self.use_bias,
            kernel_init=jax.nn.initializers.zeros
        )

    def __call__(self, x):
        residual = x
        x = self.rms_norm(x)
        # query, key, value
        q, k, v = rearrange(self.qkv(x), "b l (d split) -> split b l d", split=3)

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


class CrossAttention(nn.Module):
    n_heads: int = 8
    use_bias: bool = False
    features: int = 768
    eps:float = 1e-6

    def setup(self):

        self.rms_norm_q = nn.RMSNorm(
            epsilon=self.eps, 
            dtype=jnp.float32,
            param_dtype=jnp.float32
        )
        self.rms_norm_cond = nn.RMSNorm(
            epsilon=self.eps, 
            dtype=jnp.float32,
            param_dtype=jnp.float32
        )
        self.q = nn.Dense(
            features=int(self.features), 
            use_bias=self.use_bias
        )
        self.kv = nn.Dense(
            features=int(self.features * 2), 
            use_bias=self.use_bias
        )
        self.out = nn.Dense(
            features=self.features, 
            use_bias=self.use_bias,
            kernel_init=jax.nn.initializers.zeros
        )

    def __call__(self, x, cond):
        n, h, w, c = x.shape
        residual = x
        x = rearrange(x, "n h w c -> n (h w) c") # to B L D
        x = self.rms_norm_q(x)
        cond = self.rms_norm_cond(cond)
        q = self.q(x)
        # query, key, value
        k, v = rearrange(self.kv(cond), "b l (d split) -> split b l d", split=2)

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
        x = rearrange(x, "n (h w) c -> n h w c", h=h, w=w)
        return  x + residual
    

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


class Modulator(nn.Module):
    features: int = 768
    use_bias: bool = False

    def setup(self):

        self.modulator = nn.Dense(
            features=self.features * 2, 
            use_bias=self.use_bias,
            kernel_init=jax.nn.initializers.zeros
        )
    def __call__(self, x, t):
        n, h, w, c = x.shape
        modulation = nn.silu(self.modulator(t))
        scale, shift = rearrange(modulation, "b (d split) -> split b d", split=2)
        return x * (1 + scale[:,None, None,:]) + shift[:,None, None,:]


class Upsample(nn.Module):

    features: int
    use_bias: False

    def setup(self):
        self.conv = nn.Conv(
            self.features,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=self.use_bias
        )

    def __call__(self, x):
        batch, height, width, channels = x.shape
        x = jax.image.resize(
            x,
            shape=(batch, height * 2, width * 2, channels),
            method="bicubic",
        )
        x = self.conv(x)
        return x


class Downsample(nn.Module):

    features: int
    use_bias: False

    def setup(self):
        self.conv = nn.Conv(
            self.features,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="VALID",
            use_bias=self.use_bias
        )

    def __call__(self, x):
        pad = ((0, 0), (0, 1), (0, 1), (0, 0))
        x = jnp.pad(x, pad_width=pad)
        x = self.conv(x)
        return x


class EncoderStageA(nn.Module):
    first_layer_output_features: int = 24
    output_features: int = 4
    downscale: float = True
    down_layer_dim: Tuple = (48, 96)
    down_layer_kernel_size: Tuple = (3, 3)
    down_layer_blocks: Tuple = (2, 2)
    use_bias: bool = False 
    conv_expansion_factor: Tuple = (2, 2)
    eps:float = 1e-6
    group_count: int = (-1, -1)
    checkpoint: bool = True

    def setup(self):

        self.input_conv = nn.Conv(
            features=self.first_layer_output_features,
            kernel_size=(3, 3), 
            strides=(1, 1),
            padding="SAME",
            feature_group_count=1,
            use_bias=True,
        )

        # down
        down_projections = []
        down_blocks = []

        for stage, layer_count in enumerate(self.down_layer_blocks):
            if self.downscale:
                input_proj = Downsample(
                    features=self.down_layer_dim[stage],
                    use_bias=self.use_bias,
                )
            else:
                input_proj = None
            down_projections.append(input_proj)
            
            down_layers = []
            for layer in range(layer_count):
                down_layer = EfficientConvRMS(
                    features=self.down_layer_dim[stage],
                    kernel_size=self.down_layer_kernel_size[stage],
                    expansion_factor=self.conv_expansion_factor[stage],
                    group_count=self.group_count[stage],
                    use_bias=self.use_bias,
                    eps=self.eps,
                    checkpoint=self.checkpoint,
                    residual=True,
                )

                down_layers.append(down_layer)
            down_blocks.append(down_layers)

        self.blocks = list(zip(down_projections, down_blocks))

        self.final_norm = nn.RMSNorm(
            epsilon=self.eps, 
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )
        self.projections = nn.Dense(
            features=self.output_features,
            use_bias=True,
        )

    def __call__(self, image):
        image = self.input_conv(image)
        for downsample, conv_layers in self.blocks:
            if self.downscale:
                image = downsample(image)
            for conv_layer in conv_layers:
                image = conv_layer(image)
        image = self.final_norm(image)
        image = self.projections(image)

        image = nn.tanh(image)
        return image


class DecoderStageA(nn.Module):
    last_upsample_layer_output_features: int = 24
    output_features: int = 3
    upscale: float = True
    up_layer_dim: Tuple = (96, 24)
    up_layer_kernel_size: Tuple = (3, 3)
    up_layer_blocks: Tuple = (2, 2)
    use_bias: bool = False 
    conv_expansion_factor: Tuple = (2, 2)
    eps:float = 1e-6
    group_count: int = (-1, -1)
    checkpoint: bool = True

    def setup(self):
        
        self.projections = nn.Dense(
            features=self.up_layer_dim[0],
            use_bias=True,
        )

        # up
        up_blocks = []
        up_projections = []

        for stage, layer_count in enumerate(self.up_layer_blocks):
            up_layers = []
            for layer in range(layer_count):
                up_layer = EfficientConvRMS(
                    features=self.up_layer_dim[stage],
                    kernel_size=self.up_layer_kernel_size[stage],
                    expansion_factor=self.conv_expansion_factor[stage],
                    group_count=self.group_count[stage],
                    use_bias=self.use_bias,
                    eps=self.eps,
                    checkpoint=self.checkpoint,
                    residual=True,
                )

                up_layers.append(up_layer)
            up_blocks.append(up_layers)

            # TODO: add a way to disable this projection so the identity path is uninterrupted
            # projection layer (pointwise conv) 
            if self.upscale:
                if stage + 1 == len(self.up_layer_blocks):
                    output_proj = Upsample(
                        features=self.up_layer_dim[-1],
                        use_bias=self.use_bias,
                    )

                else:

                    output_proj = Upsample(
                        features=self.up_layer_dim[stage + 1],
                        use_bias=self.use_bias,
                    )
            else:
                output_proj = None
            up_projections.append(output_proj)
            

        self.blocks = list(zip(up_projections, up_blocks))
        self.final_norm = nn.RMSNorm(
            epsilon=self.eps, 
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        self.final_conv = nn.Conv(
            features=self.output_features,
            kernel_size=(3, 3), 
            strides=(1, 1),
            padding="SAME",
            feature_group_count=1,
            use_bias=True,
        )

    def __call__(self, image):

        image = self.projections(image)
        for  upsample, conv_layers in self.blocks:
            for conv_layer in conv_layers:
                image = conv_layer(image)
            
            image = upsample(image)
        image = self.final_norm(image)
        image = self.final_conv(image) 
        image = nn.tanh(image)

        return image


class ControlNet(nn.Module):

    first_layer_output_features: int = 24
    output_features: int = 4
    down_layer_dim: Tuple = (48, 96)
    down_layer_kernel_size: Tuple = (3, 3)
    down_layer_blocks: Tuple = (2, 2)
    use_bias: bool = False 
    conv_expansion_factor: Tuple = (2, 2)
    group_count: int = (16, 16) # not used, TODO add if statement for this lol
    eps: float = 1e-6

    def setup(self):
        # wrapper
        # reusing arch to unlock possibility of reusing weights from stage A
        self.control_encoder = EncoderStageA(
            first_layer_output_features=self.first_layer_output_features,
            output_features=self.output_features,
            down_layer_dim=self.down_layer_dim,
            down_layer_kernel_size=self.down_layer_kernel_size,
            down_layer_blocks=self.down_layer_blocks,
            use_bias=self.use_bias,
            conv_expansion_factor=self.conv_expansion_factor,
            group_count=self.group_count,
            eps=self.eps,
        )

    def __call__(self, x):
        return self.control_encoder(x)


class UNetEncoderStageB(nn.Module):
    down_layer_dim: Tuple = (48, 96)
    down_layer_kernel_size: Tuple = (3, 3)
    down_layer_blocks: Tuple = (2, 2)
    use_bias: bool = False 
    conv_expansion_factor: Tuple = (2, 2)
    eps:float = 1e-6
    group_count: int = (-1, -1)
    checkpoint: bool = True
    def setup(self):

        self.projections = nn.Dense(
            features=self.down_layer_dim[0],
            use_bias=True,
        )

        # down
        down_projections = []
        down_blocks = []
        control_blocks = []

        for stage, layer_count in enumerate(self.down_layer_blocks):
            control_projections = nn.Dense(
                features=self.down_layer_dim[stage],
                use_bias=self.use_bias,
            )
            control_blocks.append(control_projections)           
            if stage + 1 != len(self.down_layer_blocks):
                input_proj = Downsample(
                    features=self.down_layer_dim[stage + 1],
                    use_bias=self.use_bias,
                )
            else:
                input_proj = None
            down_projections.append(input_proj)
            
            down_layers = []
            for layer in range(layer_count):
                down_layer = EfficientConvRMS(
                    features=self.down_layer_dim[stage],
                    kernel_size=self.down_layer_kernel_size[stage],
                    expansion_factor=self.conv_expansion_factor[stage],
                    group_count=self.group_count[stage],
                    use_bias=self.use_bias,
                    eps=self.eps,
                    residual=True,
                    checkpoint=self.checkpoint,
                )

                modulator = Modulator(
                    features=self.down_layer_dim[stage],
                    use_bias=self.use_bias,
                )
                down_layers.append([down_layer, modulator])
            down_blocks.append(down_layers)

        self.blocks = list(zip(down_projections, down_blocks, control_blocks))

    def __call__(self, image, timestep, control=None):
        image = self.projections(image)
        skips_for_decoder = []
        for i, (downsample, conv_layers, control_proj) in enumerate(self.blocks):
            # skip connection for the decoder
            for conv_layer, modulator in conv_layers:
                # timestep information is inserted here
                image = modulator(image, timestep)
                image = conv_layer(image)
            # controlnet injection
            if control is not None:
                n, h, w, c = image.shape
                nc, hc, wc, cc = control.shape
                image = image + control_proj(jax.image.resize(control, shape=(n, h, w, cc), method="bicubic"))
            skips_for_decoder.insert(0, image)

            # no downsample last layer
            if i + 1 == len(self.down_layer_blocks):
                pass
            else:
                image = downsample(image)
        return skips_for_decoder


class UNetDecoderStageB(nn.Module):
    output_features: int = 3
    up_layer_dim: Tuple = (96, 24)
    up_layer_kernel_size: Tuple = (3, 3)
    up_layer_blocks: Tuple = (2, 2)
    use_bias: bool = False 
    conv_expansion_factor: Tuple = (2, 2)
    eps:float = 1e-6
    group_count: int = (-1, -1)
    checkpoint: bool = True

    def setup(self):

        # up
        up_blocks = []
        up_projections = []

        for stage, layer_count in enumerate(self.up_layer_blocks):
            up_layers = []
            for layer in range(layer_count):
                up_layer = EfficientConvRMS(
                    features=self.up_layer_dim[stage],
                    kernel_size=self.up_layer_kernel_size[stage],
                    expansion_factor=self.conv_expansion_factor[stage],
                    group_count=self.group_count[stage],
                    use_bias=self.use_bias,
                    eps=self.eps,
                    residual=True,
                    checkpoint=self.checkpoint,
                )
                modulator = Modulator(
                    features=self.up_layer_dim[stage],
                    use_bias=self.use_bias,
                )
                up_layers.append([up_layer, modulator])
            up_blocks.append(up_layers)


            if stage + 1 != len(self.up_layer_blocks):
                output_proj = Upsample(
                    features=self.up_layer_dim[stage + 1],
                    use_bias=self.use_bias,
                )
            else:
                output_proj = None
                
            up_projections.append(output_proj)
            

        self.blocks = list(zip(up_projections, up_blocks))

        # latent decoder
        self.final_norm = nn.RMSNorm(
            epsilon=self.eps, 
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        self.final_conv = nn.Conv(
            features=self.output_features,
            kernel_size=(3, 3), 
            strides=(1, 1),
            padding="SAME",
            feature_group_count=1,
            use_bias=True,
        )

    def __call__(self, skips, timestep):

        for  i, (upsample, conv_layers) in enumerate(self.blocks):
            # skip connections from encoder with injected image information
            if i > 0:
                image = image + skips[i]
            else:
                image = skips[i]
            for conv_layer, modulator in conv_layers:
                # timestep information is inserted here
                image = modulator(image, timestep)
                image = conv_layer(image)
            # no upsample last layer
            if i + 1 == len(self.up_layer_blocks):
                pass
            else:
                image = upsample(image)

        # decoder for final layer
        image = self.final_norm(image)
        image = self.final_conv(image) 
        # image = nn.tanh(image)

        return image


class UNetStageB(nn.Module):

    down_layer_dim: Tuple = (48, 96)
    down_layer_kernel_size: Tuple = (3, 3)
    down_layer_blocks: Tuple = (2, 2)
    down_group_count: int = (-1, -1)
    down_conv_expansion_factor: Tuple = (2, 2)

    up_layer_dim: Tuple = (96, 24)
    up_layer_kernel_size: Tuple = (3, 3)
    up_layer_blocks: Tuple = (2, 2)
    up_group_count: int = (-1, -1)
    up_conv_expansion_factor: Tuple = (2, 2)

    output_features: int = 3
    use_bias: bool = False 
    timestep_dim: int = 320
    eps:float = 1e-6
    checkpoint: bool = True

    def setup(self):
        self.time_embed = FourierLayers(
            features=self.timestep_dim, 
            keep_random=False,
        )


        self.encoder = UNetEncoderStageB(
            down_layer_dim=self.down_layer_dim,
            down_layer_kernel_size=self.down_layer_kernel_size,
            down_layer_blocks=self.down_layer_blocks,
            use_bias=self.use_bias, 
            conv_expansion_factor=self.down_conv_expansion_factor,
            eps=self.eps,
            group_count=self.down_group_count,
            checkpoint=self.checkpoint,
        )

        self.decoder = UNetDecoderStageB(
            output_features=self.output_features,
            up_layer_dim=self.up_layer_dim,
            up_layer_kernel_size=self.up_layer_kernel_size,
            up_layer_blocks=self.up_layer_blocks,
            use_bias=self.use_bias, 
            conv_expansion_factor=self.up_conv_expansion_factor,
            eps=self.eps,
            group_count=self.up_group_count,
            checkpoint=self.checkpoint,
        )

    def __call__(self, image, timestep, control=None):
        timestep = self.time_embed(timestep)
        skips = self.encoder(image, timestep, control)
        image = self.decoder(skips, timestep)
        return image


class EfficientCrossAttnStageC(nn.Module):
    first_layer_output_features: int = 24
    output_features: int = 4
    layer_dim: int = 320
    timestep_dim: int = 320
    cross_attn_heads: Tuple = (8, 8)
    layer_kernel_size: Tuple = (3, 3)
    layer_blocks: Tuple = (2, 2) # only 1 cross attn for each block
    use_bias: bool = False 
    conv_expansion_factor: Tuple = (2, 2)
    eps:float = 1e-6
    group_count: int = (-1, -1)
    checkpoint: bool = True

    def setup(self):
        self.time_embed = FourierLayers(
            features=self.timestep_dim, 
            keep_random=False,
        )

        self.input_conv = nn.Conv(
            features=self.first_layer_output_features,
            kernel_size=(3, 3), 
            strides=(1, 1),
            padding="SAME",
            feature_group_count=1,
            use_bias=True,
        )

        # down
        cross_modulators = []
        blocks = []

        for stage, layer_count in enumerate(self.layer_blocks):
            cross_modulator = CrossAttention(
                n_heads=self.cross_attn_heads[stage],
                features=self.layer_dim,
                use_bias=self.use_bias,
                eps=self.eps,
            )
            cross_modulators.append(cross_modulator)
            
            layers = []
            for layer in range(layer_count):

                layer = EfficientConvRMS(
                    features=self.layer_dim,
                    kernel_size=self.layer_kernel_size[stage],
                    expansion_factor=self.conv_expansion_factor[stage],
                    group_count=self.group_count[stage],
                    use_bias=self.use_bias,
                    eps=self.eps,
                    residual=True,
                    checkpoint=self.checkpoint,
                )
                modulator = Modulator(
                    features=self.layer_dim,
                    use_bias=self.use_bias,
                )

                layers.append([layer, modulator])
            blocks.append(layers)

        self.blocks = list(zip(cross_modulators, blocks))

        self.final_norm = nn.RMSNorm(
            epsilon=self.eps, 
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )

        self.final_conv = nn.Conv(
            features=self.output_features,
            kernel_size=(3, 3), 
            strides=(1, 1),
            padding="SAME",
            feature_group_count=1,
            use_bias=True,
        )


    def __call__(self, image, cond, timestep):
        timestep = self.time_embed(timestep)
        image = self.input_conv(image)

        for cross_modulators, conv_layers in self.blocks:
            image = cross_modulators(image, cond)
            for conv_layer, modulator in conv_layers:
                image = modulator(image, timestep)
                image = conv_layer(image)

        image = self.final_norm(image)
        image = self.final_conv(image)

        image = nn.tanh(image)
        return image


class GuidanceProjectorStageD(nn.Module):
    pass