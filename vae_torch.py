import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class EfficientConv(nn.Module):
    def __init__(self, features, kernel_size, expansion_factor=4, group_count=16, use_bias=False, eps=1e-6, classic_conv=False, residual=True):
        super(EfficientConv, self).__init__()
        self.features = features
        self.kernel_size = kernel_size
        self.expansion_factor = expansion_factor
        self.group_count = group_count
        self.use_bias = use_bias
        self.eps = eps
        self.classic_conv = classic_conv
        self.residual = residual

        self.group_norm = nn.GroupNorm(num_groups=32, num_channels=features, eps=eps)

        if classic_conv:
            self.conv = nn.Conv2d(features, features * expansion_factor, kernel_size=kernel_size, padding=kernel_size // 2, bias=use_bias)
        else:
            self.pointwise_expand = nn.Linear(features, features * expansion_factor, bias=use_bias)
            self.depthwise = nn.Conv2d(features * expansion_factor, features * expansion_factor, kernel_size=kernel_size, padding=kernel_size // 2, groups=features * expansion_factor // group_count, bias=use_bias)

        self.pointwise_contract = nn.Linear(features * expansion_factor, features, bias=use_bias)

    def forward(self, x):
        if self.residual:
            residual = x

        x = self.group_norm(x)

        if self.classic_conv:
            x = F.silu(self.conv(x))
        else:
            x = F.silu(self.pointwise_expand(x))
            x = F.silu(self.depthwise(x))

        x = self.pointwise_contract(rearrange(x, "n c h w -> n h w c"))
        x = rearrange(x, "n h w c-> n c h w")

        if self.residual:
            x = x + residual

        return x
    

class Upsample(nn.Module):
    def __init__(self, in_features, features, use_bias=False):
        super(Upsample, self).__init__()
        self.features = features
        self.use_bias = use_bias
        self.conv = nn.Conv2d(in_features, features, kernel_size=3, padding=1, bias=use_bias)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_features, features, use_bias=False):
        super(Downsample, self).__init__()
        self.features = features
        self.use_bias = use_bias
        self.conv = nn.Conv2d(in_features, features, kernel_size=3, stride=2, padding=0, bias=use_bias)

    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1))
        x = self.conv(x)
        return x
    

class Encoder(nn.Module):
    def __init__(self, output_features=3, down_layer_contraction_factor=((2, 2), (2, 2), (2, 2)), 
                 down_layer_dim=(256, 512, 1024), down_layer_kernel_size=(7, 7, 7), 
                 down_layer_blocks=(2, 2, 2), down_layer_ordinary_conv=(False, False, False), 
                 down_layer_residual=(True, True, True), use_bias=False, conv_expansion_factor=(2, 2, 2), 
                 eps=1e-6, group_count=16, last_layer="linear"):
        super(Encoder, self).__init__()
        self.output_features = output_features
        self.down_layer_contraction_factor = down_layer_contraction_factor
        self.down_layer_dim = down_layer_dim
        self.down_layer_kernel_size = down_layer_kernel_size
        self.down_layer_blocks = down_layer_blocks
        self.down_layer_ordinary_conv = down_layer_ordinary_conv
        self.down_layer_residual = down_layer_residual
        self.use_bias = use_bias
        self.conv_expansion_factor = conv_expansion_factor
        self.eps = eps
        self.group_count = group_count
        self.last_layer = last_layer

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
        # down_projections = []
        self.blocks  = nn.ModuleList()

        for stage, layer_count in enumerate(self.down_layer_blocks):
            # projection layer (pointwise conv) 
            input_proj = Downsample(
                in_features=self.down_layer_dim[stage] if stage == 0 else self.down_layer_dim[stage-1], 
                features=self.down_layer_dim[stage],
                use_bias=self.use_bias,
            )
            # down_projections.append(input_proj)
            
            down_layers = nn.ModuleList()
            for layer in range(layer_count):
                down_layer = EfficientConv(
                    features=self.down_layer_dim[stage],
                    kernel_size=self.down_layer_kernel_size[stage],
                    expansion_factor=self.conv_expansion_factor[stage],
                    group_count=self.group_count,
                    use_bias=self.use_bias,
                    eps=self.eps,
                    classic_conv=self.down_layer_ordinary_conv[stage],
                    residual=self.down_layer_residual[stage],
                )

                down_layers.append(down_layer)
            self.blocks.append(nn.ModuleList([input_proj, down_layers]))

        # self.blocks = nn.ModuleList(list(zip(down_projections, down_blocks)))

        # cant decide which is which so gonna put it in the config
        if self.last_layer == "conv":
            self.projections = nn.Conv2d(
                self.down_layer_dim[-1], 
                self.output_features * 2, 
                kernel_size=3, 
                padding=1, 
                bias=True,
            )
        elif self.last_layer == "linear":
            self.projections = nn.Linear(
                self.down_layer_dim[-1], 
                self.output_features * 2, 
                bias=True,
            )
        self.final_norm = nn.GroupNorm(
            num_groups=32, 
            num_channels=self.down_layer_dim[-1], 
            eps=self.eps,
        )

        self.input_conv = nn.Conv2d(
            3, 
            self.down_layer_dim[0], 
            kernel_size=3, 
            padding=1, 
            bias=True,
        )

    def forward(self, image):
        image = self.input_conv(image)
        for pointwise, conv_layers in self.blocks:
            image = pointwise(image)
            for conv_layer in conv_layers:
                image = conv_layer(image)
        image = self.final_norm(image)
        image = self.projections(image)
        image = torch.tanh(image)
        return image


class Decoder(nn.Module):
    def __init__(self, intermediate_features=64, output_features=3, up_layer_contraction_factor=((2, 2), (2, 2), (2, 2)), 
                 up_layer_dim=(1024, 512, 256), up_layer_kernel_size=(7, 7, 7), 
                 up_layer_blocks=(2, 2, 2), up_layer_ordinary_conv=(False, False, False), 
                 up_layer_residual=(True, True, True), use_bias=False, conv_expansion_factor=(2, 2, 2), 
                 eps=1e-6, group_count=16):
        super(Decoder, self).__init__()
        self.intermediate_features = intermediate_features
        self.output_features = output_features
        self.up_layer_contraction_factor = up_layer_contraction_factor
        self.up_layer_dim = up_layer_dim
        self.up_layer_kernel_size = up_layer_kernel_size
        self.up_layer_blocks = up_layer_blocks
        self.up_layer_ordinary_conv = up_layer_ordinary_conv
        self.up_layer_residual = up_layer_residual
        self.use_bias = use_bias
        self.conv_expansion_factor = conv_expansion_factor
        self.eps = eps
        self.group_count = group_count

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
        self.blocks = nn.ModuleList()
        # up_projections = []

        for stage, layer_count in enumerate(self.up_layer_blocks):
            up_layers = nn.ModuleList()
            for layer in range(layer_count):
                up_layer = EfficientConv(
                    features=self.up_layer_dim[stage],
                    kernel_size=self.up_layer_kernel_size[stage],
                    expansion_factor=self.conv_expansion_factor[stage],
                    group_count=self.group_count,
                    use_bias=self.use_bias,
                    eps=self.eps,
                    classic_conv=self.up_layer_ordinary_conv[stage],
                    residual=self.up_layer_residual[stage],
                )

                up_layers.append(up_layer)

            # TODO: add a way to disable this projection so the identity path is uninterrupted
            # projection layer (pointwise conv) 
            if stage + 1 == len(self.up_layer_blocks):
                output_proj = Upsample(
                    in_features=self.up_layer_dim[-1] if stage == 0 else self.up_layer_dim[-2], 
                    features=self.up_layer_dim[-1],
                    use_bias=self.use_bias,
                )
            else:
                output_proj = Upsample(
                    in_features=self.up_layer_dim[stage],
                    features=self.up_layer_dim[stage + 1],
                    use_bias=self.use_bias,
                )
            # up_projections.append(output_proj)
            self.blocks.append(nn.ModuleList([output_proj, up_layers]))
            

        # self.blocks = nn.ModuleList(list(zip(up_projections, up_blocks)))
        self.final_norm = nn.GroupNorm(
            num_groups=1, 
            num_channels=self.up_layer_dim[-1], 
            eps=self.eps,
        )
        self.final_conv = nn.Conv2d(
            self.up_layer_dim[-1], 
            self.output_features, 
            kernel_size=3, 
            padding=1, 
            bias=True,
        )
        self.projections = nn.Linear(
            self.intermediate_features, 
            self.up_layer_dim[0], 
            bias=True,
        )

    def forward(self, image):
        image = self.projections(rearrange(image, "n c h w -> n h w c"))
        image = rearrange(image, "n h w c-> n c h w")

        for i, (pointwise, conv_layers) in enumerate(self.blocks):
            for conv_layer in conv_layers:
                image = conv_layer(image)
            
            image = pointwise(image)
        image = self.final_norm(image)
        image = self.final_conv(image)
        image = torch.tanh(image)

        return image