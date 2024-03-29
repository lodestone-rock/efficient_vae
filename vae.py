import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
import math
from typing import Tuple, Tuple

# TODO: custom initializer for each parameters

### upscale and downscale stuff ###
# pixel unshuffle
def space_to_depth(x, h=2, w=2):
    return rearrange(x, '... (h dh) (w dw) c -> ... h w (c dh dw)', dh=h, dw=w)


# pixel shuffle
def depth_to_space(x, h=2, w=2):
    return rearrange(x, '... h w (c dh dw) -> ... (h dh) (w dw) c', dh=h, dw=w)


class EfficientConv(nn.Module):
    features: int
    kernel_size: int
    expansion_factor: int = 4 # inverted bottleneck scale factor to up project inner conv
    group_count: int = 16
    use_bias: bool = False # removing bias won't affect the model much
    eps:float = 1e-6
    classic_conv: bool = False # highly recommended toggling this on early layer
    residual: bool = True # toggle residual identity path (useful for first layer)

    def setup(self):

        self.rms_norm = nn.RMSNorm(
            epsilon=self.eps, 
            dtype=jnp.float32,
            param_dtype=jnp.float32
        )
        # using classical conv on early layer will increase flops but make it faster to train 
        if self.classic_conv:
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
                features=self.features * self.expansion_factor * 2, 
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
        x = self.rms_norm(x)
        # use conv for early layer if possible
        if self.classic_conv:
            x = self.conv(x) 
            x = nn.silu(x)
        else:
            x = self.pointwise_expand(x)
            # split the channel into 2 groups instead of doing individual linear projection for each
            x, bypass = rearrange(x, "n h w (t nc) -> t n h w nc", t=2)
            x = self.depthwise(x)
            # non linearity selection in higher dimensional space
            x = nn.silu(bypass) * x
            # projection back to input space
        x = self.pointwise_contract(x)

        if self.residual:
            x = x + residual
        return x

 
class Encoder(nn.Module):
    # granular configuration to experiment with
    down_layer_contraction_factor: Tuple = ((2, 2), (2, 2), (2, 2)) # (h, w) patched input will reduce computation in exchange of accuracy
    down_layer_dim: Tuple = (256, 512, 1024)
    down_layer_kernel_size: Tuple = (7, 7, 7)
    down_layer_blocks: Tuple = (2, 2, 2)
    down_layer_ordinary_conv: Tuple = (False, False, False) # convert block to ordinary conv
    down_layer_residual: Tuple = (True, True, True) # toggle it off just for fun :P https://arxiv.org/abs/2108.08810
    use_bias: bool = False 
    conv_expansion_factor: int = 4
    eps:float = 1e-6
    group_count: int = 16
    last_layer: str = "linear"


    def setup(self):
        
        assert (
            len(self.down_layer_contraction_factor) == len(self.down_layer_dim) == 
            len(self.down_layer_kernel_size) == len(self.down_layer_blocks) == 
            len(self.down_layer_ordinary_conv) == len(self.down_layer_residual)
        ), (
            "down_layer_contraction_factor, down_layer_dim, down_layer_kernel_size, "
            "down_layer_blocks, down_layer_ordinary_conv, down_layer_residual "
            "must have equal amount of elements!"
            )

        # down
        down_projections = []
        down_blocks = []

        for stage, layer_count in enumerate(self.down_layer_blocks):
            # TODO: add a way to disable this projection so the identity path is uninterrupted
            # projection layer (pointwise conv) 
            input_proj = nn.Dense(
                features=self.down_layer_dim[stage],
                use_bias=self.use_bias,
            )
            down_projections.append(input_proj)
            
            down_layers = []
            for layer in range(layer_count):
                down_layer = EfficientConv(
                    features=self.down_layer_dim[stage],
                    kernel_size=self.down_layer_kernel_size[stage],
                    expansion_factor=self.conv_expansion_factor,
                    group_count=self.group_count,
                    use_bias=self.use_bias,
                    eps=self.eps,
                    classic_conv=self.down_layer_ordinary_conv[stage],
                    residual=self.down_layer_residual[stage],
                )

                down_layers.append(down_layer)
            down_blocks.append(down_layers)

        self.blocks = list(zip(self.down_layer_contraction_factor, down_projections, down_blocks))

        # cant decide which is which so gonna put it in the config
        if self.last_layer == "conv":
            self.projections = EfficientConv(
                features=self.down_layer_dim[-1] * 2,
                kernel_size=self.down_layer_kernel_size[-1],
                expansion_factor=self.conv_expansion_factor,
                group_count=self.group_count,
                use_bias=self.use_bias,
                eps=self.eps,
                classic_conv=self.down_layer_ordinary_conv[-1],
                residual=False,
            )

        elif self.last_layer == "linear":
            self.projections = nn.Dense(
                features=self.down_layer_dim[-1] * 2,
                use_bias=self.use_bias,
            )
        self.final_norm = nn.RMSNorm(
            epsilon=self.eps, 
            dtype=jnp.float32,
            param_dtype=jnp.float32
        )


    def __call__(self, image):

        for patch, pointwise, conv_layers in self.blocks:
            image = space_to_depth(image, h=patch[0], w=patch[1]) 
            image = pointwise(image)
            for conv_layer in conv_layers:
                image = conv_layer(image)
        image = self.final_norm(image)
        image = self.projections(image)
        return image


class Discriminator(nn.Module):
    # basically a copy of encoder block with simple linear layer at the end
    # granular configuration to experiment with
    down_layer_contraction_factor: Tuple = ((2, 2), (2, 2), (2, 2)) # (h, w) patched input will reduce computation in exchange of accuracy
    down_layer_dim: Tuple = (256, 512, 1024)
    down_layer_kernel_size: Tuple = (7, 7, 7)
    down_layer_blocks: Tuple = (2, 2, 2)
    down_layer_ordinary_conv: Tuple = (False, False, False) # convert block to ordinary conv
    down_layer_residual: Tuple = (True, True, True) # toggle it off just for fun :P https://arxiv.org/abs/2108.08810
    use_bias: bool = False 
    conv_expansion_factor: int = 4
    eps:float = 1e-6
    group_count: int = 16

    def setup(self):
        
        assert (
            len(self.down_layer_contraction_factor) == len(self.down_layer_dim) == 
            len(self.down_layer_kernel_size) == len(self.down_layer_blocks) == 
            len(self.down_layer_ordinary_conv) == len(self.down_layer_residual)
        ), (
            "down_layer_contraction_factor, down_layer_dim, down_layer_kernel_size, "
            "down_layer_blocks, down_layer_ordinary_conv, down_layer_residual "
            "must have equal amount of elements!"
            )

        # down
        down_projections = []
        down_blocks = []

        for stage, layer_count in enumerate(self.down_layer_blocks):
            # TODO: add a way to disable this projection so the identity path is uninterrupted
            # projection layer (pointwise conv) 
            input_proj = nn.Dense(
                features=self.down_layer_dim[stage],
                use_bias=self.use_bias,
            )
            down_projections.append(input_proj)
            
            down_layers = []
            for layer in range(layer_count):
                down_layer = EfficientConv(
                    features=self.down_layer_dim[stage],
                    kernel_size=self.down_layer_kernel_size[stage],
                    expansion_factor=self.conv_expansion_factor,
                    group_count=self.group_count,
                    use_bias=self.use_bias,
                    eps=self.eps,
                    classic_conv=self.down_layer_ordinary_conv[stage],
                    residual=self.down_layer_residual[stage],
                )

                down_layers.append(down_layer)
            down_blocks.append(down_layers)

        self.blocks = list(zip(self.down_layer_contraction_factor, down_projections, down_blocks))
        self.final_norm = nn.RMSNorm(
            epsilon=self.eps, 
            dtype=jnp.float32,
            param_dtype=jnp.float32
        )
        self.classifier = nn.Dense(
                features=1,
                use_bias=self.use_bias,
            )


    def __call__(self, image):

        for patch, pointwise, conv_layers in self.blocks:
            image = space_to_depth(image, h=patch[0], w=patch[1]) 
            image = pointwise(image)
            for conv_layer in conv_layers:
                image = conv_layer(image)
        image = self.final_norm(image)
        logits = self.classifier(image)
        return logits


class Decoder(nn.Module):
    # granular configuration to experiment with
    output_features: int = 3
    up_layer_contraction_factor: Tuple = ((2, 2), (2, 2), (2, 2)) # (h, w) patched input will reduce computation in exchange of accuracy
    up_layer_dim: Tuple = (1024, 512, 256)
    up_layer_kernel_size: Tuple = (7, 7, 7)
    up_layer_blocks: Tuple = (2, 2, 2)
    up_layer_ordinary_conv: Tuple = (False, False, False) # convert block to ordinary conv
    up_layer_residual: Tuple = (True, True, True) # toggle it off just for fun :P https://arxiv.org/abs/2108.08810
    use_bias: bool = False 
    conv_expansion_factor: int = 4
    eps:float = 1e-6
    group_count: int = 16


    def setup(self):
        
        assert (
            len(self.up_layer_contraction_factor) == len(self.up_layer_dim) == 
            len(self.up_layer_kernel_size) == len(self.up_layer_blocks) == 
            len(self.up_layer_ordinary_conv) == len(self.up_layer_residual)
        ), (
            "up_layer_contraction_factor, up_layer_dim, up_layer_kernel_size, "
            "up_layer_blocks, up_layer_ordinary_conv, up_layer_residual "
            "must have equal amount of elements!"
            )

        # up
        up_blocks = []
        up_projections = []

        for stage, layer_count in enumerate(self.up_layer_blocks):
            up_layers = []
            for layer in range(layer_count):
                up_layer = EfficientConv(
                    features=self.up_layer_dim[stage],
                    kernel_size=self.up_layer_kernel_size[stage],
                    expansion_factor=self.conv_expansion_factor,
                    group_count=self.group_count,
                    use_bias=self.use_bias,
                    eps=self.eps,
                    classic_conv=self.up_layer_ordinary_conv[stage],
                    residual=self.up_layer_residual[stage],
                )

                up_layers.append(up_layer)
            up_blocks.append(up_layers)

            # TODO: add a way to disable this projection so the identity path is uninterrupted
            # projection layer (pointwise conv) 
            if stage + 1 == len(self.up_layer_blocks):
                 output_proj = nn.Dense(
                    features=self.output_features * self.up_layer_contraction_factor[stage][0] * self.up_layer_contraction_factor[stage][1],
                    use_bias=self.use_bias,
                )
            else:
                output_proj = nn.Dense(
                    features=self.up_layer_dim[stage + 1] * (
                        self.up_layer_contraction_factor[stage][0] * self.up_layer_contraction_factor[stage][1]),
                    use_bias=self.use_bias,
                )
            up_projections.append(output_proj)
            

        self.blocks = list(zip(self.up_layer_contraction_factor, up_projections, up_blocks))
        self.final_norm = nn.RMSNorm(
            epsilon=self.eps, 
            dtype=jnp.float32,
            param_dtype=jnp.float32
        )

    def __call__(self, image):

        for i, (patch, pointwise, conv_layers) in enumerate(self.blocks):
            for conv_layer in conv_layers:
                image = conv_layer(image)
            
            if len(self.blocks)-1 == i:
                image = self.final_norm(image)
            image = pointwise(image)
            image = depth_to_space(image, h=patch[0], w=patch[1]) 

        return image