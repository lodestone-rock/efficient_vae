


def embedding_modulation(x, scale, shift):
    # basically applies time embedding and conditional to each input for each layer
    return x * (1 + scale) + shift 


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

        self.group_norm = nn.GroupNorm(
            epsilon=self.eps, 
            dtype=jnp.float32,
            param_dtype=jnp.float32,
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
        if self.classic_conv:
            x = self.conv(x) 
            x = nn.silu(x)
        else:
            x = self.pointwise_expand(x)
            x = nn.silu(x)
            x = self.depthwise(x)
            x = nn.silu(x)
            # projection back to input space
        x = self.pointwise_contract(x)

        if self.residual:
            x = x + residual
        return x

