import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
from typing import Tuple, Tuple


class EfficientConvRMS(nn.Module):
    features: int
    kernel_size: int
    expansion_factor: int = 4 # inverted bottleneck scale factor to up project inner conv
    group_count: int = -1
    use_bias: bool = False # removing bias won't affect the model much
    eps:float = 1e-6
    residual: bool = True # toggle residual identity path (useful for first layer)

    def setup(self):

        self.group_norm = nn.RMSNorm(
            epsilon=self.eps, 
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )
        # using classical conv on early layer will increase flops but make it faster to train 
        if self.group_count == -1:
            self.conv = nn.Conv(
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
            self.pointwise_expand = nn.Dense(
                features=self.features * self.expansion_factor, 
                use_bias=self.use_bias
            )
            self.depthwise = nn.Conv(
                features=self.features * self.expansion_factor,
                kernel_size=(self.kernel_size, self.kernel_size), 
                strides=(1, 1),
                padding="SAME",
                feature_group_count=self.features * self.expansion_factor // self.group_count,
                use_bias=self.use_bias,
            )
    
        # activation
        # pointwise conv reformulated as matmul
        self.pointwise_contract = nn.Dense(
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
            method="nearest",
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
    down_layer_dim: Tuple = (48, 96)
    down_layer_kernel_size: Tuple = (3, 3)
    down_layer_blocks: Tuple = (2, 2)
    use_bias: bool = False 
    conv_expansion_factor: Tuple = (2, 2)
    eps:float = 1e-6
    group_count: int = (-1, -1)


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
            input_proj = Downsample(
                features=self.down_layer_dim[stage],
                use_bias=self.use_bias,
            )
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
                    # classic_conv=self.down_layer_ordinary_conv[stage],
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
    up_layer_dim: Tuple = (96, 24)
    up_layer_kernel_size: Tuple = (3, 3)
    up_layer_blocks: Tuple = (2, 2)
    use_bias: bool = False 
    conv_expansion_factor: Tuple = (2, 2)
    eps:float = 1e-6
    group_count: int = (-1, -1)

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
                    # classic_conv=self.up_layer_ordinary_conv[stage],
                    residual=True,
                )

                up_layers.append(up_layer)
            up_blocks.append(up_layers)

            # TODO: add a way to disable this projection so the identity path is uninterrupted
            # projection layer (pointwise conv) 
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

    def setup(self):

        # down
        down_projections = []
        down_blocks = []

        for stage, layer_count in enumerate(self.down_layer_blocks):
            if stage + 1 != len(self.down_layer_blocks):
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
                    residual=True,
                )

                modulator = Modulator(
                    features=self.down_layer_dim[stage],
                    use_bias=self.use_bias,
                )
                down_layers.append([down_layer, modulator])
            down_blocks.append(down_layers)

        self.blocks = list(zip(down_projections, down_blocks))

    def __call__(self, image, control, timestep):

        # insert upscaled image as a seed image
        image = image + jax.image.resize(control, image.shape)
        skips_for_decoder = []
        for downsample, conv_layers in self.blocks:
            # skip connection for the decoder
            for i, (conv_layer, modulator) in enumerate(conv_layers):
                # timestep information is inserted here
                image = modulator(image, timestep)
                image = conv_layer(image)
            # ensure each skip connection also has the same original information
            image = image + jax.image.resize(control, image.shape)
            skips_for_decoder.insert(0, image)

            # no downsample last layer
            if i + 1 != len(self.down_layer_blocks):
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

        for  i, upsample, conv_layers in enumerate(self.blocks, 0):
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
            if i + 1 != len(self.up_layer_blocks):
                image = upsample(image)

        # decoder for final layer
        image = self.final_norm(image)
        image = self.final_conv(image) 
        image = nn.tanh(image)

        return image


class UNetStageB(nn.Module):

    down_layer_dim: Tuple = (48, 96)
    down_layer_kernel_size: Tuple = (3, 3)
    down_layer_blocks: Tuple = (2, 2)
    use_bias: bool = False 
    conv_expansion_factor: Tuple = (2, 2)

    output_features: int = 3
    up_layer_dim: Tuple = (96, 24)
    up_layer_kernel_size: Tuple = (3, 3)
    up_layer_blocks: Tuple = (2, 2)
    use_bias: bool = False 
    conv_expansion_factor: Tuple = (2, 2)

    eps:float = 1e-6
    group_count: int = (-1, -1)

    def setup(self):

        self.encoder = UNetEncoderStageB(
            down_layer_dim=self.down_layer_dim,
            down_layer_kernel_size=self.down_layer_kernel_size,
            down_layer_blocks=self.down_layer_blocks,
            use_bias=self.use_bias, 
            conv_expansion_factor=self.conv_expansion_factor,
            eps=self.eps,
            group_count=self.group_count,
        )

        self.decoder = UNetDecoderStageB(
            output_features=self.output_features,
            up_layer_dim=self.up_layer_dim,
            up_layer_kernel_size=self.up_layer_kernel_size,
            up_layer_blocks=self.up_layer_blocks,
            use_bias=self.use_bias, 
            conv_expansion_factor=self.conv_expansion_factor,
            eps=self.eps,
            group_count=self.group_count,
        )

    def rectified_flow_loss(self, rng_key, model, model_params, images, conditions, timesteps):
        noises = jax.random.normal(key=rng_key, shape=images.shape)
        noise_to_image_flow = noises * timesteps[:, None, None, None] + images * (1-timesteps[:, None, None, None]) # lerp
        flow_path = noises - images # noise >>>>towards>>>> image
        model_trajectory_predictions = model(model_params, noise_to_image_flow, conditions, timesteps)

        loss = jnp.mean((model_trajectory_predictions - flow_path)** 2)
        return loss

    def __init__(self, image, control, timestep):

        skips = self.encoder(image, control, timestep)
        image = self.decoder(skips, timestep)
        return image