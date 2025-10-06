from re import L
from typing import Callable, Union, Sequence
import numpy as np
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import SqueezeExcitation

from ..utils.utils import make_tuple
from .ops import *
from .norm import LayerNorm2d

# TODO: implement FNO block with  differential kernels (i.e. conv when the kernel is rescaled by 1/h and the mean is substracted)
# It would allows to capture the high frequencies while keeping the resolution invariance, far better and more grounded than  the U-FNO! see neuralop for an implementation

DOWNSAMPLING_LAYERS = {
    "convpixunshuffle": ConvPixelUnshuffleDownSampleLayer,
    "pixunshuffleaveraging": PixelUnshuffleChannelAveragingDownSampleLayer,
    "stridedconv": StridedConvDownsamplingLayer,
    "maxpooling": MaxPoolConv,
}

UPSAMPLING_LAYERS = {
    "convpixshuffle": ConvPixelShuffleUpSampleLayer,
    "duplicatingpixshuffle": ChannelDuplicatingPixelUnshuffleUpSampleLayer,
    "interpolate": InterpolateConvUpSampleLayer,
}


class ResBlock(nn.Module):
    """Residual block with n convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        nconv=2,
        stride=1,
        kernel_size=3,
        groups=1,
        norm=partial(nn.GroupNorm, num_channels=32),
        activation=nn.GELU,
        use_bias=False,
        se=False,
        se_ratio=8,
        downsampling_method="maxpooling",
    ):

        super().__init__()

        use_bias = make_tuple(use_bias, nconv)
        norm = make_tuple(norm, nconv)
        activation = make_tuple(activation, nconv)

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.ops = nn.ModuleList()
        for i in range(nconv):
            input_channels = in_channels if i == 0 else out_channels
            conv_stride = stride if i == 0 else 1
            self.ops.append(
                ConvLayer(
                    input_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=conv_stride,
                    groups=groups,
                    use_bias=use_bias[i],
                    norm=norm[i],
                    activation=activation[i],
                )
            )

        if se:
            se_act = [act for act in activation if act is not None][-1]
            self.ops.append(SqueezeExcitation(out_channels, out_channels // se_ratio, activation=se_act))

        if stride > 1:
            if downsampling_method == "pixunshuffleaveraging":
                self.shortconv = PixelUnshuffleChannelAveragingDownSampleLayer(
                    in_channels=in_channels, out_channels=out_channels, factor=stride
                )
            else:

                self.shortconv = DOWNSAMPLING_LAYERS.get(downsampling_method, StridedConvDownsamplingLayer)(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=1, factor=stride
                )
        elif in_channels != out_channels:
            self.shortconv = ConvLayer(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                groups=1,
                use_bias=False,
                norm=None,
                activation=None,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        for op in self.ops:
            x = op(x)
        if self.stride > 1 or (self.in_channels != self.out_channels):
            shortcut = self.shortconv(shortcut)

        return x + shortcut


class SepResBlock(nn.Module):
    """Residual block with n separable convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        nconv=2,
        stride=1,
        kernel_size=3,
        norm=partial(nn.GroupNorm, num_channels=32),
        activation=nn.GELU,
        use_bias=False,
        se=False,
        se_ratio=8,
        downsampling_method="maxpooling",
    ):

        super().__init__()

        use_bias = make_tuple(use_bias, nconv)
        norm = make_tuple(norm, nconv)
        activation = make_tuple(activation, nconv)

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.ops = nn.ModuleList()
        for i in range(nconv):
            input_channels = in_channels if i == 0 else out_channels
            conv_stride = stride if i == 0 else 1
            # depthwise convolution
            self.ops.append(nn.Conv2d(in_channels=input_channels,
                                      out_channels=out_channels,
                                      stride=conv_stride,
                                      kernel_size=kernel_size,
                                      groups=input_channels,
                                      padding='same',
                                      bias=use_bias[i]))
            self.ops.append(nn.Conv2d(in_channels=out_channels,
                                      out_channels=out_channels,
                                      stride=conv_stride,
                                      kernel_size=1,
                                      padding='same',
                                      bias=use_bias[i]))
            self.ops.append(norm[i])
            self.ops.append(activation[i])

        if se:
            se_act = [act for act in activation if act is not None][-1]
            self.ops.append(SqueezeExcitation(out_channels, out_channels // se_ratio, activation=se_act))

        if stride > 1:
            if downsampling_method == "pixunshuffleaveraging":
                self.shortconv = PixelUnshuffleChannelAveragingDownSampleLayer(
                    in_channels=in_channels, out_channels=out_channels, factor=stride
                )
            else:

                self.shortconv = DOWNSAMPLING_LAYERS.get(downsampling_method, StridedConvDownsamplingLayer)(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=1, factor=stride
                )
        elif in_channels != out_channels:
            self.shortconv = ConvLayer(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                groups=1,
                use_bias=False,
                norm=None,
                activation=None,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        for op in self.ops:
            x = op(x)
        if self.stride > 1 or (self.in_channels != self.out_channels):
            shortcut = self.shortconv(shortcut)

        return x + shortcut


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_ratio=4,
        stride=1,
        kernel_size=3,
        groups=1,
        norm=partial(nn.GroupNorm, num_channels=32),
        activation=nn.GELU,
        se=True,
        se_ratio=4,
        use_bias=False,
        downsampling_method="maxpooling",
    ):

        super().__init__()

        use_bias = make_tuple(use_bias, 3)
        norm_list = make_tuple(norm, 3)
        activation_list = make_tuple(activation, 3)

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = out_channels // bottleneck_ratio

        self.proj = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            use_bias=use_bias[0],
            norm=norm_list[0],
            activation=activation_list[0],
        )

        self.conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            use_bias=use_bias[1],
            norm=norm_list[1],
            activation=activation_list[1],
        )
        if se:
            se_act = [act for act in activation_list if act is not None][-1]
            self.se = SqueezeExcitation(mid_channels, mid_channels // se_ratio, activation=se_act)
        else:
            self.se = None

        self.expand = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            use_bias=use_bias[2],
            norm=norm_list[2],
            activation=activation_list[2],
        )

        if stride > 1:
            if downsampling_method == "pixunshuffleaveraging":
                self.shortconv = PixelUnshuffleChannelAveragingDownSampleLayer(
                    in_channels=in_channels, out_channels=out_channels, factor=stride
                )
            else:
                self.shortconv = DOWNSAMPLING_LAYERS.get(downsampling_method, StridedConvDownsamplingLayer)(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=1, factor=stride
                )

        elif in_channels != out_channels:
            self.shortconv = ConvLayer(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                groups=1,
                use_bias=False,
                norm=None,
                activation=None,
            )

    def forward(self, x):

        shortcut = x

        x = self.proj(x)
        x = self.conv(x)

        if self.se is not None:
            x = self.se(x)

        x = self.expand(x)

        if self.stride > 1:
            shortcut = self.shortconv(shortcut)
        elif self.in_channels != self.out_channels:
            shortcut = self.shortconv(shortcut)

        x = x + shortcut
        return x


class ConvNextv2Block(nn.Module):
    """https://github.com/facebookresearch/ConvNeXt-V2"""

    def __init__(
        self, in_channels, activation=nn.GELU, drop_path=0.0, **kwargs
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels
        )  # depthwise conv
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.pwconv1 = nn.Linear(in_channels, 4 * in_channels)  # pointwise/1x1 convs, implemented with linear layers
        self.act = activation()
        self.grn = GRN(4 * in_channels)
        self.pwconv2 = nn.Linear(4 * in_channels, in_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = shortcut + self.drop_path(x)
        return x


class FNOBlock(nn.Module):
    """
    A single block of a Fourier Neural Operator (FNO) with a residual connection.
    Args:
        in_channels (int): Number of input channels for the block.
        hidden_channels:Number of output channels for the SpectralConv
        out_channels (int): Number of output channels for the block.
        modes (tuple): Tuple (modes_x, modes_y) for the spectral convolution.
        spectral_layer_type (str): Type of spectral layer ('standard' or 'tucker').
        ranks (tuple, optional): Ranks (r1, r2, r3) for Tucker factorization, required if spectral_layer_type is 'tucker'.
        A way to set (r1, r2, r3) is to fix a compression factor and to set
        r1 = cin / k
        r2 = cout / k
        r3 = np.prod(modes) / k
        For a k=2 the number of parameters is decreased by approx 5 times relatively to a standard SpectralConv
        for k=4 it's 10 times, etc.
        even with no compression, (k=1) the number of parameters is 2.5 times less
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: tuple[int, int],
        hidden_channels: Union[int, None] = None,
        activation: Callable = nn.GELU,
        normalization: Union[Callable, None] = LayerNorm2d,
        spectral_layer_type: str = "standard",
        ranks: Union[tuple[int, int, int], np.ndarray, None] = None,
        scaling: Union[int, float] = 1,
    ):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if hidden_channels is None:
            self.hidden_channels = out_channels
        else:
            self.hidden_channels = hidden_channels

        self.modes = modes
        self.spectral_layer_type = spectral_layer_type
        self.ranks = ranks
        self.scaling = scaling

        # Core Spectral Convolution Layer
        if spectral_layer_type == "standard":
            self.spectral_conv = SpectralConv2d(in_channels, self.hidden_channels, modes, scaling=scaling)
        elif spectral_layer_type == "tucker":
            if ranks is None:
                raise ValueError("Ranks must be provided for TuckerSpectralConv2d.")
            self.spectral_conv = TuckerSpectralConv2d(in_channels, self.hidden_channels, modes, ranks, scaling=scaling)
        else:
            raise ValueError(f"Unknown spectral_layer_type: {spectral_layer_type}. Choose 'standard' or 'tucker'.")

        # Activation & normalization
        self.activation = activation()
        if normalization is not None:
            self.norm = normalization(out_channels)
        else:
            self.norm = None

        # mix channels
        self.linear = nn.Conv2d(self.hidden_channels, out_channels, kernel_size=1)

        # Shortcut Branch Layer
        # 1x1 Conv to potentially change input channels to output channels
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C_in, H, W) - spatial domain

        # Shortcut branch computation
        x_shortcut = self.shortcut(x)  # (B, out_channels, H, W)
        if self.scaling != 1:
            x_shortcut = F.interpolate(x_shortcut, scale_factor=self.scaling, mode="bilinear", align_corners=False)

        # Main branch computation
        x = self.spectral_conv(x)  # (B, hidden_channels, H, W)
        x = self.activation(x)  # (B, hidden_channels, H, W)
        x = self.linear(x)  # (B, out_channels, H, W)

        if self.norm is not None:
            x = self.norm(x)

        x = x + x_shortcut  # (B, out_channels, H, W)

        if self.activation is not None:
            x = self.activation(x)

        return x


class FNOBlockv2(nn.Module):
    """
    Improved  Fourier Neural Operator (FNO) block with a double residual connection.

    see
    Multi-Grid Tensorized Fourier Neural Operator for High-Resolution PDEs,
    Jean Kossaifi, Nikola Kovachki, Kamyar Azizzadenesheli, Anima Anandkumar
    https://arxiv.org/abs/2310.00120
    Args:
        in_channels (int): Number of input channels for the block.
        out_channels (int): Number of output channels for the block.
        modes (tuple): Tuple (modes_x, modes_y) for the spectral convolution.
        spectral_layer_type (str): Type of spectral layer ('standard' or 'tucker').
        ranks (tuple, optional): Ranks (r1, r2, r3) for Tucker factorization, required if spectral_layer_type is 'tucker'.
        A way to set (r1, r2, r3) is to fix a compression factor and to set
        r1 = cin / k
        r2 = cout / k
        r3 = np.prod(modes) / k
        For a k=2 the number of parameters is decreased by approx 5 times relatively to a standard SpectralConv
        for k=4 it's 10 times, etc.
        even with no compression, (k=1) the number of parameters is 2.5 times less
        mlp_layers: number of linear layers after the SpectralConv (default 1)

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: tuple[int, int],
        activation: Callable = nn.GELU,
        normalization: Union[Callable, None] = LayerNorm2d,
        spectral_layer_type: str = "standard",
        mlp_layers=2,
        ranks: Union[tuple[int, int, int], np.ndarray, None] = None,
        scaling: Union[int, float] = 1,
    ):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.spectral_layer_type = spectral_layer_type
        self.ranks = ranks
        self.scaling = scaling

        # Core Spectral Convolution Layer
        if spectral_layer_type == "standard":
            self.spectral_conv = SpectralConv2d(in_channels, out_channels, modes, scaling=scaling)
        elif spectral_layer_type == "tucker":
            if ranks is None:
                raise ValueError("Ranks must be provided for TuckerSpectralConv2d.")
            self.spectral_conv = TuckerSpectralConv2d(in_channels, out_channels, modes, ranks, scaling=scaling)
        else:
            raise ValueError(f"Unknown spectral_layer_type: {spectral_layer_type}. Choose 'standard' or 'tucker'.")

        # Activation & normalization
        self.activation1 = activation()
        self.norm1 = normalization(out_channels) if normalization is not None else None
        self.activation2 = activation()
        self.norm2 = normalization(out_channels) if normalization is not None else None

        # mix channels with n mlp_layers of 1x1 conv.
        # The hidden dimension is out_channels // 2
        self.mlp = nn.ModuleList()
        for i in range(mlp_layers):
            if mlp_layers == 1:
                self.mlp.append(nn.Conv2d(out_channels, out_channels, kernel_size=1))
            else:
                out_mlp = out_channels if i == mlp_layers - 1 else out_channels // 2
                in_mlp = out_channels if i == 0 else out_channels // 2
                self.mlp.append(nn.Conv2d(in_mlp, out_mlp, kernel_size=1))

        # Shortcut Branch Layer
        # 1x1 Conv to potentially change input channels to output channels
        # If in_channels == out_channels, this acts as a trainable identity or scaling
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C_in, H, W) - spatial domain

        # Shortcut branch computation
        x_shortcut = self.shortcut(x)  # (B, out_channels, H, W)
        if self.scaling != 1:
            x_shortcut = F.interpolate(x_shortcut, scale_factor=self.scaling, mode="bilinear", align_corners=False)

        x = self.spectral_conv(x)

        if self.norm1 is not None:
            x = self.norm1(x)

        x = x + x_shortcut

        x = self.activation1(x)

        for op in self.mlp:
            x = op(x)

        if self.norm2 is not None:
            x = self.norm2(x)

        x = x + x_shortcut

        x = self.activation2(x)

        return x


class ConvFNOBlock(nn.Module):
    """
    A single block of a Fourier Neural Operator (FNO) with a local convolutional branch using Differential Kernel [1] and a residual connection.
    This block combines the spectral convolution with a local convolutional layer to capture both low and high frequencies in the input data.
    Args:
        in_channels (int): Number of input channels for the block.
        hidden_channels:Number of output channels for the SpectralConv
        out_channels (int): Number of output channels for the block.
        nconv (int): Number of local convolutional layers in the local branch.
        activation (Callable): Activation function to use after the spectral convolution and local convolution.
        normalization (Callable, optional): Normalization layer to apply after the spectral convolution.
        resampling (Union[str, None]): either 'up' (x2), 'down' (//2) or None (no scaling is applied)
        modes (tuple): Tuple (modes_x, modes_y) for the spectral convolution.
        spectral_layer_type (str): Type of spectral layer ('standard' or 'tucker').
        ranks (tuple, optional): Ranks (r1, r2, r3) for Tucker factorization, required if spectral_layer_type is 'tucker'.
        A way to set (r1, r2, r3) is to fix a compression factor and to set
        r1 = cin / k
        r2 = cout / k
        r3 = np.prod(modes) / k
        For a k=2 the number of parameters is decreased by approx 5 times relatively to a standard SpectralConv
        for k=4 it's 10 times, etc.
        even with no compression, (k=1) the number of parameters is 2.5 times less

    .. [1] : Liu-Schiaffini, M., et al. (2024). "Neural Operators with
        Localized Integral and Differential Kernels".
        ICML 2024, https://arxiv.org/abs/2402.16845.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: tuple[int, int],
        hidden_channels: Union[int, None] = None,
        nconv=1,
        kernel_size: int = 3,
        activation: Callable = nn.GELU,
        normalization: Union[Callable, None] = LayerNorm2d,
        spectral_layer_type: str = "standard",
        ranks: Union[tuple[int, int, int], np.ndarray, None] = None,
        resampling: Union[str, None] = None,
        grid_width: Union[int, float] = 1.0,
    ):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if hidden_channels is None:
            self.hidden_channels = out_channels
        else:
            self.hidden_channels = hidden_channels

        self.modes = modes
        self.spectral_layer_type = spectral_layer_type
        self.ranks = ranks
        self.resampling = resampling
        self.nconv = nconv
        self.grid_width = grid_width

        if self.resampling is not None:
            if self.resampling == "up":
                self.scaling = 2
                self.resampling_module = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            elif self.resampling == "down":
                self.scaling = 0.5
                self.resampling_module = nn.MaxPool2d(kernel_size=2)
            else:
                self.resampling_module = None
                self.scaling = 1
                raise ValueError(f"Unknown resampling method: {self.resampling}. Choose 'up', 'down' or None.")
        else:
            self.scaling = 1
            self.resampling_module = None
        # Core Spectral Convolution Layer
        if spectral_layer_type == "standard":
            self.spectral_conv = SpectralConv2d(in_channels, self.hidden_channels, modes, scaling=self.scaling)
        elif spectral_layer_type == "tucker":
            if ranks is None:
                raise ValueError("Ranks must be provided for TuckerSpectralConv2d.")
            self.spectral_conv = TuckerSpectralConv2d(
                in_channels, self.hidden_channels, modes, ranks, scaling=self.scaling
            )
        else:
            raise ValueError(f"Unknown spectral_layer_type: {spectral_layer_type}. Choose 'standard' or 'tucker'.")

        # Activation & normalization
        self.activation = activation()
        if normalization is not None:
            self.norm = normalization(out_channels)
        else:
            self.norm = None

        # mix channels
        self.linear = nn.Conv2d(self.hidden_channels, out_channels, kernel_size=1)

        self.conv = nn.ModuleList()

        for i in range(nconv):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = out_channels if i == nconv - 1 else hidden_channels
            self.conv.append(
                FiniteDifferenceLayer(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    padding="same",
                    norm=normalization,
                    activation=activation,
                    grid_width=self.grid_width * self.scaling,
                )
            )

        # Shortcut Branch Layer
        # 1x1 Conv to potentially change input channels to output channels
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C_in, H, W) - spatial domain

        # Shortcut branch computation
        x_shortcut = self.shortcut(x)  # (B, out_channels, H, W)
        x_local = x
        if self.scaling != 1:
            x_shortcut = F.interpolate(x_shortcut, scale_factor=self.scaling, mode="bilinear", align_corners=False)

        # Main branch computation
        x = self.spectral_conv(x)  # (B, hidden_channels, H, W)
        x = self.activation(x)  # (B, hidden_channels, H, W)
        x = self.linear(x)  # (B, out_channels, H, W)
        if self.norm is not None:
            x = self.norm(x)

        # local branch computation
        if self.resampling_module is not None:
            x_local = self.resampling_module(x_local)
        for op in self.conv:
            x_local = op(x_local)

        x = x + x_shortcut + x_local  # (B, out_channels, H, W)

        if self.activation is not None:
            x = self.activation(x)

        return x


class MLPBlock(nn.Module):
    """
    A small MLP that takes a normalized 1D coordinate (real) and
    produces (complex) values for a set of basis functions.
    Args:
        n_dim (int): Dimension of the input coordinates (default: 2).
        Note: the kernel can be made non linear by using an input with the evaluation of the funciton at the coordinates.
        num_modes (int): Number of modes to learn.
        hidden_dim (int): Hidden dimension of the MLP.
        num_layers (int): Number of hidden layers in the MLP.
        activation (callable): Activation function to use (default: nn.GELU).
        norm (callable): Normalization layer to use (default: nn.LayerNorm).
        dropout (float): Dropout rate (default 0.0).
    """

    def __init__(self, in_ch, out_ch, hidden_dim=64, num_layers=2, activation=nn.GELU, norm=nn.LayerNorm, dropout=0.0):

        super().__init__()
        self.num_modes = out_ch

        layers = []
        # The first layer takes 1 coordinate (e.g., normalized)
        layers.append(nn.Linear(in_ch, hidden_dim))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        if norm is not None:
            layers.append(norm(hidden_dim))
        if activation is not None:
            layers.append(activation())  # Or ReLU, LeakyReLU, etc.

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            if norm is not None:
                layers.append(norm(hidden_dim))
            if activation is not None:
                layers.append(activation())

        # The last layer produces 2 * num_modes real values (for real and imaginary parts)
        layers.append(nn.Linear(hidden_dim, 2 * out_ch))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x is a tensor of shape (N, n_dim) where N is the number of sampling points and n_dim the dimension of the input
        # for example, N = H or N = W

        # Pass the coordinates through the MLP
        output = self.mlp(x)  # (N, 2 * num_modes)

        # Separate real and imaginary parts
        real_part = output[..., : self.num_modes]  # (N, num_modes)
        imag_part = output[..., self.num_modes :]  # (N, num_modes)

        # Combine to form a complex tensor
        complex_output = torch.complex(real_part, imag_part)  # (N, num_modes)

        return complex_output


class LITBlock(nn.Module):
    """LITBlock (Linear Integral Transform Block)
    A PyTorch module implementing a learned, resolution-invariant 2D integral transform using continuous basis functions parameterized by MLPs.
    Inspired by IAE-NET, this block projects input features onto learned bases, applies learned complex weights in the transformed domain, and reconstructs the spatial representation via inverse transforms.
    Supports variable input resolutions and includes normalization, activation, and residual connections.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        m1 (int): Number of modes for the height dimension.
        m2 (int): Number of modes for the width dimension.
        mlp_hidden_dim (int, optional): Hidden dimension for the basis MLPs. Default: 64.
        mlp_num_layers (int, optional): Number of layers in the basis MLPs. Default: 2.
        activation (callable, optional): Activation function. Default: nn.GELU.
        norm (callable, optional): Normalization layer. Default: LayerNorm2d.
    Inputs:
        x (torch.Tensor): Input tensor of shape (B, C, H, W), where H and W may vary.
        torch.Tensor: Output tensor of shape (B, out_channels, H, W)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        m1,
        m2,
        mlp_hidden_dim=64,
        mlp_num_layers=2,
        activation=nn.GELU,
        norm=LayerNorm2d,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.m1 = m1  # Number of modes kept in height (u)
        self.m2 = m2  # Number of modes kept in width (v)

        # The bases are MLPs that generate the basis values.
        # These MLPs are the learnable parameters.
        self.basis_h_fn = MLPBlock(
            out_ch=self.m1, in_ch=1, hidden_dim=mlp_hidden_dim, num_layers=mlp_num_layers, activation=activation
        )
        self.basis_w_fn = MLPBlock(
            out_ch=self.m2, in_ch=1, hidden_dim=mlp_hidden_dim, num_layers=mlp_num_layers, activation=activation
        )

        # Learned parameters for multiplication in the transformed space (still per channel and mode).
        self.learned_weights_freq = nn.Parameter(
            torch.randn(out_channels, self.in_channels, self.m1, self.m2, dtype=torch.cfloat)
        )

        # Optional: Initialization of learned_weights_freq for a good starting point (e.g., all to 1.0)
        nn.init.constant_(self.learned_weights_freq, 1.0)  # Initialize to 1+0j

        self.mixer = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        # Activation & normalization
        self.activation = activation()

        if norm is not None:
            self.norm = norm(out_channels)
        else:
            self.norm = None

        # Shortcut Branch Layer
        # 1x1 Conv to potentially change input channels to output channels
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Performs the forward pass of the module for variable resolution input.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
                                H and W may vary.

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W), real part.
        """
        batch_size, channels, H, W = x.shape

        # Convert input to complex numbers if it is real
        x_complex = torch.complex(x, torch.zeros_like(x)) if not x.is_complex() else x

        # Generate normalized coordinates for the current height and width.
        # These coordinates are passed to our MLPs to generate the bases dynamically.
        h_coords = torch.linspace(0, 1, H, device=x.device, dtype=torch.float).unsqueeze(1)  # (H, 1)
        w_coords = torch.linspace(0, 1, W, device=x.device, dtype=torch.float).unsqueeze(1)  # (W, 1)

        # Dynamically generate the basis matrices for the current resolution
        # base_values_h: (H, m1)
        # base_values_w: (W, m2)
        base_values_h = self.basis_h_fn(h_coords)
        base_values_w = self.basis_w_fn(w_coords)

        # Transpose the generated bases so they have shape (m1, H) and (m2, W)
        # for matrix multiplication as in the previous approach.
        # transform_h_basis_runtime: (m1, H)
        # transform_w_basis_runtime: (m2, W)
        transform_h_basis_runtime = base_values_h.T
        transform_w_basis_runtime = base_values_w.T

        # --- Step 1: Direct transform (projection onto learned modes) ---
        # F_truncated = W'_M * f * W'_N

        # Transform along rows (dimension H -> m1 modes)
        # x_complex: (B, C, H, W)
        # transform_h_basis_runtime: (m1, H)
        # Result: (B, C, m1, W)
        transformed_freq_h = torch.einsum("bchw,mh->bcmw", x_complex, transform_h_basis_runtime)

        # Transform along columns (dimension W -> m2 modes)
        # transformed_freq_h: (B, C, m1, W)
        # transform_w_basis_runtime: (m2, W)
        # Result: (B, C, m1, m2)
        transformed_freq_hw = torch.einsum("bcmw,nw->bcmn", transformed_freq_h, transform_w_basis_runtime)

        # --- Step 2: Multiplication by learned weights in the transformed space ---
        # self.learned_weights_freq: (C, m1, m2)
        # transformed_freq_hw: (B, C, m1, m2)
        # processed_freq_domain = transformed_freq_hw * self.learned_weights_freq.unsqueeze(0)

        processed_freq_domain = torch.einsum("bixy,oixy->boxy", transformed_freq_hw, self.learned_weights_freq)

        # --- Step 3: Inverse transform (projection back) ---
        # f_approx = (W'_M)^H * F_truncated * (W'_N)^H

        # Inverse transform along columns (m2 modes -> W)
        # processed_freq_domain: (B, C, m1, m2)
        # transform_w_basis_runtime.H: (W, m2) (conjugate transpose of (m2, W))
        # Result: (B, C, m1, W)
        reconstructed_spatial_w = torch.einsum("bcmn,wn->bcmw", processed_freq_domain, transform_w_basis_runtime.H)

        # Inverse transform along rows (m1 modes -> H)
        # reconstructed_spatial_w: (B, C, m1, W)
        # transform_h_basis_runtime.H: (H, m1) (conjugate transpose of (m1, H))
        # Result: (B, C, H, W)
        reconstructed_spatial_hw = torch.einsum("bcmw,hm->bchw", reconstructed_spatial_w, transform_h_basis_runtime.H)

        # Normalization
        normalisation_factor = 1.0 / (H * W)
        reconstructed_spatial_hw = reconstructed_spatial_hw.real * normalisation_factor

        # Mixing channels
        output = self.mixer(reconstructed_spatial_hw)
        output = self.norm(output) if self.norm is not None else output
        output = self.activation(output)

        # Shortcut connection
        output = output + self.shortcut(x)
        output = self.activation(output)
        return output


class NLITBlock(nn.Module):
    """NLITBlock (Non Linear Integral Transform Block)
    A PyTorch module implementing a learned, resolution-invariant 2D non-linear integral transform.
    The block learns continuous basis functions via MLPs, conditioned on both spatial coordinates and input values,
    to perform data-dependent transforms along spatial dimensions. Inspired by IAE-NET, this block enables
    discretization-invariant learning for image-like data.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        m1 (int): Number of modes for the height (H) dimension.
        m2 (int): Number of modes for the width (W) dimension.
        mlp_hidden_dim (int, optional): Hidden dimension for the MLPs learning the bases. Default: 64.
        mlp_num_layers (int, optional): Number of hidden layers in the MLPs. Default: 2.
        activation (callable, optional): Activation function for MLPs and output. Default: nn.GELU.
        norm (callable, optional): Normalization layer for output. Default: LayerNorm2d.
    Forward Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W), where H and W may vary.
        torch.Tensor: Output tensor of shape (B, out_channels, H, W)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        m1,
        m2,
        mlp_hidden_dim=64,
        mlp_num_layers=2,
        activation=nn.GELU,
        norm=LayerNorm2d,
    ):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.m1 = m1  # Number of modes kept in height (u)
        self.m2 = m2  # Number of modes kept in width (v)
        self.coords_dim = 1 + in_channels

        # The bases are MLPs that generate the basis values.
        # These MLPs are the learnable parameters.
        self.basis_h_fn = MLPBlock(
            out_ch=self.m1,
            in_ch=self.coords_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=activation,
        )
        self.basis_w_fn = MLPBlock(
            out_ch=self.m2,
            in_ch=self.coords_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=activation,
        )

        # Learned parameters for multiplication in the transformed space (still per channel and mode).
        self.learned_weights_freq = nn.Parameter(
            torch.randn(self.in_channels, out_channels, self.m1, self.m2, dtype=torch.cfloat)
        )
        # Optional: Initialization of learned_weights_freq for a good starting point (e.g., all to 1.0)
        nn.init.constant_(self.learned_weights_freq, 1.0)  # Initialize to 1+0j
        self.mixer = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        # Activation & normalization
        self.activation = activation()
        self.norm = norm(out_channels) if norm is not None else None
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Performs the forward pass of the module for variable resolution input.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
                                H and W may vary.

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W), real part.
        """
        batch_size, channels, H, W = x.shape

        # Convert input to complex numbers if it is real
        x_complex = torch.complex(x, torch.zeros_like(x)) if not x.is_complex() else x

        # STEP 1.1 Process coordinates along W dimension (width)
        # Generate normalized coordinates for the current width.
        w_coords = torch.linspace(0, 1, W, device=x.device, dtype=torch.float).unsqueeze(1)

        # Add function values to condition our kernel to the input
        x_w = x_complex.real.permute(0, 2, 3, 1)  # (B, H, W, Cin)
        # Repeat the coordinates to match batch and height dimensions: (B, H, W, 1)
        w_coords = w_coords.unsqueeze(0).unsqueeze(0).repeat(batch_size, H, 1, 1)
        # Concatenate channel values and W coordinates: (B, H, W, Cin + 1)
        w_input = torch.cat([x_w, w_coords], dim=-1)
        # Flatten for MLP input: (B*H*W, Cin + 1)
        # w_input = w_input.view(-1, channels + 1) not needed
        # Generate basis values for W: (B*H*W, m2)
        w_basis_runtime = self.basis_w_fn(w_input)
        # Reshape to (B, H, W, m2)
        # w_basis_runtime = w_basis_runtime.view(batch_size, H, W, self.m2)
        # Perform matrix multiplication for the W transformation
        # x_complex: (B, Cin, H, W) -> permute to (B, H, Cin, W) for matmul with (B, H, W, m2)
        x_w = x_complex.permute(0, 2, 1, 3)  # (B, H, Cin, W)
        # Multiplication: (B, H, Cin, W) @ (B, H, W, m2) -> (B, H, Cin, m2)
        transformed_freq_w = torch.matmul(x_w, w_basis_runtime)
        # Permute to (B, Cin, H, m2) for the next step
        transformed_freq_w = transformed_freq_w.permute(0, 2, 1, 3)  # (B, Cin, H, m2)

        # STEP 1.2: Process coordinates along H dimension (height)
        # Permute to (B, m2, H, Cin) for concatenation
        x_h = transformed_freq_w.real.permute(0, 3, 2, 1)  # (B, m2, H, Cin)

        h_coords = torch.linspace(0, 1, H, device=x.device, dtype=torch.float).unsqueeze(1)  # (H, 1)
        # Repeat the coordinates to match batch and m2 dimensions: (B, m2, H, 1)
        h_coords = h_coords.unsqueeze(0).unsqueeze(0).repeat(batch_size, self.m2, 1, 1)

        # Concatenate channel values and H coordinates: (B, m2, H, Cin + 1)
        h_input = torch.cat([x_h, h_coords], dim=-1)

        # Flatten for MLP input: (B*m2*H, Cin + 1)
        # h_input = h_input.view(-1, channels + 1)

        # Generate basis values for H: (B*m2*H, m1)
        h_basis_runtime = self.basis_h_fn(h_input)

        # Reshape to (B, m2, H, m1) - this is the "kernel" of IAE-Net for the second layer
        h_basis_runtime = h_basis_runtime.view(batch_size, self.m2, H, self.m1)

        # Perform matrix multiplication for the H transformation
        # transformed_freq_w_permuted: (B, Cin, H, m2) -> permute to (B, m2, Cin, H) for matmul with (B, m2, H, m1)
        x_for_h_transform = transformed_freq_w.permute(0, 3, 1, 2)  # (B, m2, Cin, H)

        # Multiplication: (B, m2, Cin, H) @ (B, m2, H, m1) -> (B, m2, Cin, m1)
        transformed_freq_hw = torch.matmul(x_for_h_transform, h_basis_runtime)

        # Permute to (B, Cin, m2, m1) - our final format for the frequency domain
        # Note: The order of modes (m1, m2) or (m2, m1) depends on convention.
        # Here, we transformed W (m2) then H (m1), so (B, Cin, m2, m1) makes sense.
        # However, for compatibility with learned_weights_freq (Cin, m1, m2), we permute.
        transformed_freq_hw = transformed_freq_hw.permute(
            0, 2, 3, 1
        )  # (B, Cin, m1, m2) if m1 is axis 2 and m2 is axis 3

        # --- Step 2: Multiplication by learned weights ---
        # processed_freq_domain = transformed_freq_hw * self.learned_weights_freq.unsqueeze(0)
        # transformed_freq_hw: (B, C_in, m1, m2)
        # self.learned_weights_freq: (C_out, C_in, m1, m2)
        # Résultat: (B, C_out, m1, m2)
        processed_freq_domain = torch.einsum("bixy,oixy->boxy", transformed_freq_hw, self.learned_weights_freq)

        # --- Step 3: Inverse transform with data-dependent bases ---
        # We reverse the order of transformations: first H, then W.

        # 1. Inverse transform along H dimension (Height)
        # Input: processed_freq_domain (B, C_out, m1, m2)
        # Permute to (B, m2, C, m1) for matmul with (B, m2, m1, H)
        x_for_h_inv_transform = processed_freq_domain.permute(0, 3, 1, 2)  # (B, m2, C_out, m1)

        # Conjugate transpose of the H basis: (B, m2, m1, H).H -> (B, m2, H, m1)
        # Multiplication: (B, m2, C_out, m1) @ (B, m2, m1, H) -> (B, m2, C_out, H)
        reconstructed_h = torch.matmul(x_for_h_inv_transform, h_basis_runtime.mH)

        # Permute to (B, C_out, m2, H)
        reconstructed_h_permuted = reconstructed_h.permute(0, 2, 3, 1)  # (B, C_out, H, m2)

        # 2. Inverse transform along W dimension (Width)
        # Input: reconstructed_h_permuted (B, C_out, H, m2)
        # Permute to (B, H, C_out, m2) for matmul with (B, H, m2, W)
        x_for_w_inv_transform = reconstructed_h_permuted.permute(0, 2, 1, 3)  # (B, H, C_out, m2)

        # Conjugate transpose of the W basis: (B, H, m2, W).H -> (B, H, W, m2)
        # Multiplication: (B, H, C_out, m2) @ (B, H, m2, W) -> (B, H, C_out, W)
        reconstructed_spatial_hw = torch.matmul(x_for_w_inv_transform, w_basis_runtime.mH)

        # Permute to (B, C_out, H, W)
        reconstructed_spatial_hw = reconstructed_spatial_hw.permute(0, 2, 1, 3)

        # Dynamic normalization
        normalisation_factor = 1.0 / (H * W)
        reconstructed_spatial_hw = reconstructed_spatial_hw.real * normalisation_factor

        # Mixing channels
        output = self.mixer(reconstructed_spatial_hw)
        output = self.norm(output) if self.norm is not None else output
        output = self.activation(output)

        # Shortcut connection
        output = output + self.shortcut(x)
        output = self.activation(output)
        return output


class NRNLITBlock(nn.Module):
    """NRNLITBlock (Non Reversible Non Linear Integral Transform Block)
    A PyTorch module implementing a learned, resolution-invariant 2D non-linear integral transform.
    The transform and its inverse are parameterized by MLPs that generate continuous basis functions,
    conditioned on spatial coordinates and input features. The transformation is not strictly reversible,
    as both the forward and inverse kernels are learned independently and conditioned differently.
    Main features:
    - Learns spatially-adaptive, non-linear basis functions for both axes via MLPs.
    - Applies a learned frequency-domain weighting in the transformed space.
    - Supports arbitrary input resolutions.
    - Includes a shortcut connection for residual learning.
    - Inspired by IAE-NET: Integral Autoencoders for Discretization-Invariant Learning.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        m1 (int): Number of modes for the height dimension.
        m2 (int): Number of modes for the width dimension.
        mlp_hidden_dim (int, optional): Hidden dimension for the basis MLPs. Default: 64.
        mlp_num_layers (int, optional): Number of layers in the basis MLPs. Default: 2.
        activation (callable, optional): Activation function for MLPs and output. Default: nn.GELU.
        norm (callable, optional): Normalization layer for output. Default: LayerNorm2d.
    Input:
    Output:
        torch.Tensor: Output tensor of shape (B, out_channels, H, W)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        m1,
        m2,
        mlp_hidden_dim=64,
        mlp_num_layers=2,
        activation=nn.GELU,
        norm=LayerNorm2d,
    ):
        """
        Initializes the module.

        Args:
            in_channels (int): Number of input channels (C).
            m1 (int): Number of modes to keep for the height dimension (u).
            m2 (int): Number of modes to keep for the width dimension (v).
            output_shape: (tuple(H,W)): shape of the reconstructed image
            mlp_hidden_dim (int): Hidden dimension of the MLPs learning the bases.
            mlp_num_layers (int): Number of hidden layers in the MLPs learning the bases.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.m1 = m1  # Number of modes kept in height (u)
        self.m2 = m2  # Number of modes kept in width (v)
        self.coords_dim = 1 + in_channels

        # The bases are MLPs that generate the basis values.
        # These MLPs are the learnable parameters.
        self.basis_h_fn = MLPBlock(
            out_ch=self.m1,
            in_ch=self.coords_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=activation,
        )
        self.basis_w_fn = MLPBlock(
            out_ch=self.m2,
            in_ch=self.coords_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=activation,
        )
        # Decoder MLPs: bases conditionnées par les valeurs des modes spectraux (fréquentiel)
        # Pour la transformation inverse H (m1 -> H), conditionné par (C_out, m2) modes
        self.mlp_h_inv = MLPBlock(
            in_ch=1 + 2 * out_channels,
            out_ch=m1,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=activation,
        )
        self.mlp_w_inv = MLPBlock(
            in_ch=1 + 2 * out_channels,
            out_ch=m2,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            activation=activation,
        )

        self.compressor_h_inv = nn.Linear(m1, 1)
        self.compressor_w_inv = nn.Linear(m2, 1)

        # Learned parameters for multiplication in the transformed space (still per channel and mode).
        self.learned_weights_freq = nn.Parameter(
            torch.randn(self.in_channels, out_channels, self.m1, self.m2, dtype=torch.cfloat)
        )
        # Optional: Initialization of learned_weights_freq for a good starting point (e.g., all to 1.0)
        nn.init.constant_(self.learned_weights_freq, 1.0)  # Initialize to 1+0j

        self.mixer = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        # Activation & normalization
        self.activation = activation()
        self.norm = norm(out_channels) if norm is not None else None

        # Shortcut Branch Layer
        # 1x1 Conv to potentially change input channels to output channels
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Performs the forward pass of the module

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
                                H and W may vary.

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W), real part.
        """
        b, c, h, w = x.shape
        # Convert input to complex if it is real
        xc = torch.complex(x, torch.zeros_like(x)) if not x.is_complex() else x.float()

        # --- Encoder (Direct transform) ---
        # 1. W (width) transformation
        # We predict m2 kernels for each spatial coordinates
        # Input MLP: (B, H, W, C+1)
        w_coords = torch.linspace(0, 1, w, device=x.device).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(b, h, 1, 1)
        mlp_w_in = torch.cat([xc.real.permute(0, 2, 3, 1), w_coords], dim=-1)
        b_w = self.basis_w_fn(mlp_w_in)  # (B, H, W, m2)

        # Matmul: (B, H, C, W) @ (B, H, W, m2) -> (B, H, C, m2)
        x_w = torch.matmul(xc.permute(0, 2, 1, 3), b_w)
        x_w = x_w.permute(0, 2, 1, 3)  # (B, C, H, m2)

        # 2. H (height) transformation
        # Input MLP: (B, m2, H, C+1)
        h_coords = (
            torch.linspace(0, 1, h, device=x.device).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(b, self.m2, 1, 1)
        )
        mlp_h_in = torch.cat([x_w.real.permute(0, 3, 2, 1), h_coords], dim=-1)
        # predict m1 kernels for eachcoordinates (m2, H)
        b_h = self.basis_h_fn(mlp_h_in)  # (B, m2, H, m1)

        # Matmul: (B, m2, C, H) @ (B, m2, H, m1) -> (B, m2, C, m1)
        x_hw = torch.matmul(x_w.permute(0, 3, 1, 2), b_h)
        x_hw = x_hw.permute(0, 2, 3, 1)  # (B, C, m1, m2)

        # --- Frequency multiplication ---
        xf = torch.einsum("bixy,oixy->boxy", x_hw, self.learned_weights_freq)  # (B, C_out, m1, m2)

        # --- Decoder (Inverse transformation) ---
        # 1. Inverse H transformation (height reconstruction)
        # Output spatial coordinates for H
        h_rec_coords = (
            torch.linspace(0, 1, h, device=x.device).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(b, self.m2, 1, 1)
        )  # B, m2, H, 1

        # Prepare conditioning for the MLP (concatenate real and imaginary parts and compress)
        # xf is (B, C_out, m1, m2)
        # Concatenate real and imaginary: (B, 2*C_out, m1, m2)
        xf_combined_real_imag = torch.cat([xf.real, xf.imag], dim=1)

        # Permute so that m1 is the last dimension for the compressor: (B, 2*C_out, m2, m1)
        cond_h_val_for_comp = xf_combined_real_imag.permute(0, 1, 3, 2)
        # Apply the compressor: (B, 2*C_out, m2, 1)
        cond_h_val_compressed = self.compressor_h_inv(cond_h_val_for_comp)
        # Repeat the compressed value along the output spatial dimension H: (B, 2*C_out, m2, h)
        cond_h_val_repeated = cond_h_val_compressed.repeat(1, 1, 1, h)
        # Permute for the MLP: (B, m2, h, 2*C_out)
        cond_h_val_mlp_in = cond_h_val_repeated.permute(0, 2, 3, 1)

        mlp_h_inv_in = torch.cat([h_rec_coords, cond_h_val_mlp_in], dim=-1)  # (B, m2, h, 1 + 2*C_out)
        b_h_inv = self.mlp_h_inv(mlp_h_inv_in)  # (B, m2, h, m1) - Basis generated by MLP

        # Matmul: (B, m2, C_out, m1) @ (B, m2, m1, h).mH -> (B, m2, C_out, h)
        x_h_rec = torch.matmul(xf.permute(0, 3, 1, 2), b_h_inv.mH)  # Still complex
        x_h_rec = x_h_rec.permute(0, 2, 3, 1)  # (B, C_out, h, m2) - Still complex

        # 2. Inverse W transformation (width reconstruction)
        # Output spatial coordinates for W
        w_rec_coords = (
            torch.linspace(0, 1, w, device=x.device).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(b, h, 1, 1)
        )

        # Prepare conditioning for the MLP (concatenate real and imaginary parts and compress)
        # x_h_rec is (B, C_out, h, m2)
        # Concatenate real and imaginary: (B, 2*C_out, h, m2)
        x_h_rec_combined_real_imag = torch.cat([x_h_rec.real, x_h_rec.imag], dim=1)

        # Permute so that m2 is the last dimension for the compressor: (B, 2*C_out, h, m2)
        cond_w_val_for_comp = x_h_rec_combined_real_imag
        # Apply the compressor: (B, 2*C_out, h, 1)
        cond_w_val_compressed = self.compressor_w_inv(cond_w_val_for_comp)
        # Repeat the compressed value along the output spatial dimension W: (B, 2*C_out, h, w)
        cond_w_val_repeated = cond_w_val_compressed.repeat(1, 1, 1, w)
        # Permute for the MLP: (B, h, w, 2*C_out)
        cond_w_val_mlp_in = cond_w_val_repeated.permute(0, 2, 3, 1)

        mlp_w_inv_in = torch.cat([w_rec_coords, cond_w_val_mlp_in], dim=-1)  # (B, h, w, 1 + 2*C_out)
        b_w_inv = self.mlp_w_inv(mlp_w_inv_in)  # (B, h, w, m2) - Basis generated by MLP

        # Matmul: (B, h, C_out, m2) @ (B, h, m2, w).mH -> (B, h, C_out, w)
        x_rec = torch.matmul(x_h_rec.permute(0, 2, 1, 3), b_w_inv.mH).real
        x_rec = x_rec.permute(0, 2, 1, 3)  # (B, C_out, h, w)

        # Dynamic normalization (based on output resolution)
        x_rec = x_rec * (1.0 / (h * w))

        # Mixing channels
        output = self.mixer(x_rec)
        output = self.norm(output) if self.norm is not None else output
        output = self.activation(output)

        # Shortcut connection
        output = self.activation(output + self.shortcut(x))

        return output
