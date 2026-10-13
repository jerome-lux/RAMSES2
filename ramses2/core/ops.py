import math
import torch
from torch import nn
import torch.nn.functional as F

from ..utils import utils


class ConvLayer(nn.Module):
    """
    2D Convolutional layer with optional same padding, normalization, activation, and dropout.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel. Default: 3.
        stride (int): Stride of the convolution. Default: 1.
        dilation (int): Dilation factor. Default: 1.
        groups (int): Number of groups for grouped convolution. Default: 1.
        use_bias (bool): Whether to use bias in the convolution. Default: False.
        dropout (float): Dropout probability. Default: 0.
        norm (callable): Normalization layer constructor. Default: None.
        activation (callable): Activation function constructor. Default: None.
        padding (str): Padding mode ('same' or other). Default: 'same'.
        boundary_confitions (str): Padding type for the boundary. Default: 'zero'.
    Input shape:
        (B, Cin, H, W)
    Output shape:
        (B, Cout, H_out, W_out)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm=None,
        activation=None,
        padding="same",
        boundary_confitions="zero",
    ):
        """Conv2D Layer with same padding, norm and activation"""

        super(ConvLayer, self).__init__()
        self.ops = nn.ModuleList(())

        if padding == "same":
            padding = utils.get_same_padding(kernel_size, stride)
            self.ops.append(nn.ZeroPad2d(padding))

        if dropout > 0:
            self.ops.append(nn.Dropout2d(dropout))

        self.ops.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=(dilation, dilation),
                groups=groups,
                bias=use_bias,
                padding=0,
            )
        )

        if norm is not None:
            self.ops.append(norm(out_channels))
        if activation is not None:
            self.ops.append(activation())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.ops:
            x = op(x)
        return x


class StridedConvDownsamplingLayer(nn.Module):
    """
    Downsampling layer using a strided convolution.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        factor (int): Stride factor for downsampling.
        kernel_size (int): Size of the convolutional kernel. Default: 1.
        groups (int): Number of groups for grouped convolution. Default: 1.
    Input shape:
        (B, Cin, H*factor, W*factor)
    Output shape:
        (B, Cout, H, W)
    """

    def __init__(self, in_channels, out_channels, factor, kernel_size=1, groups=1):
        super().__init__()
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=factor,
            groups=groups,
            use_bias=True,
            norm=None,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class MaxPoolConv(nn.Module):
    """
    Downsampling layer using max pooling followed by a 1x1 convolution.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the max pooling kernel.
        factor (int): Stride for max pooling.
    Input shape:
        (B, Cin, H*factor, W*factor)
    Output shape:
        (B, Cout, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
    ):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride=factor, padding=math.floor((kernel_size - 1) / 2))
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_bias=True,
            norm=None,
            activation=None,
        )

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class ConvPixelUnshuffleDownSampleLayer(nn.Module):
    """
    A PyTorch module that performs downsampling by first applying a convolution to reduce the number of channels,
    followed by pixel unshuffling. The convolution reduces the input channels, and pixel unshuffling rearranges
    spatial data into the channel dimension, effectively downsampling the spatial resolution by the given factor.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels after pixel unshuffling. Must be divisible by factor**2.
        kernel_size (int): Size of the convolutional kernel.
        factor (int): Downsampling factor. The spatial dimensions will be reduced by this factor.

    Shape:
        Input:  (N, in_channels, H * factor, W * factor)
        Output: (N, out_channels, H, W)

    Example:
        layer = ConvPixelUnshuffleDownSampleLayer(16, 64, 3, 2)
        output = layer(input_tensor)

    (*,Cin,H*r,W*r) -> (*,Cout/r**2,H*r,W*r) -> (*, Cout, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**2
        assert out_channels % out_ratio == 0
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels // out_ratio,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.pixel_unshuffle(x, self.factor)
        return x


class PixelUnshuffleChannelAveragingDownSampleLayer(nn.Module):
    """
    A PyTorch layer that downsamples an input tensor by first applying pixel unshuffle and then averaging groups of channels.
    This layer reduces the spatial resolution of the input tensor by a given factor using pixel unshuffle, which rearranges spatial data into the channel dimension. The resulting channels are then grouped and averaged to produce the desired number of output channels.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels after downsampling and channel averaging.
        factor (int): Downsampling factor for spatial dimensions.
    Raises:
        AssertionError: If `in_channels * factor**2` is not divisible by `out_channels`.
    Shape:
        - Input: (B, in_channels, H, W)
        - Output: (B, out_channels, H // factor, W // factor)
    Example:
        >>> layer = PixelUnshuffleChannelAveragingDownSampleLayer(8, 4, 2)
        >>> x = torch.randn(1, 8, 32, 32)
        >>> y = layer(x)
        >>> y.shape
        torch.Size([1, 4, 16, 16])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert in_channels * factor**2 % out_channels == 0
        self.group_size = in_channels * factor**2 // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pixel_unshuffle(x, self.factor)
        B, C, H, W = x.shape
        x = x.view(B, self.out_channels, self.group_size, H, W)
        x = x.mean(dim=2)
        return x


class ConvPixelShuffleUpSampleLayer(nn.Module):
    """A PyTorch module that upsamples an input tensor using a convolution followed by pixel shuffling.
    (*, Cin, H, W) -> (*, Cout, H, W) -> (*, Cout // r**2, H*r, W*r)
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels after upsampling.
        kernel_size (int): Size of the convolutional kernel.
        factor (int): Upsampling factor (spatial resolution is increased by this factor).
    Input shape:
        (*, in_channels, H, W)
    Output shape:
        (*, out_channels, H * factor, W * factor)
    Example:
        layer = ConvPixelShuffleUpSampleLayer(16, 32, 3, 2)
        out = layer(torch.randn(1, 16, 32, 32))
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**2
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels * out_ratio,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.pixel_shuffle(x, self.factor)
        return x


class InterpolateConvUpSampleLayer(nn.Module):
    """
    A PyTorch module that performs upsampling using interpolation followed by a convolution.
    (*, Cin, H, W) -> (*, Cin, H*r, W*r) -> (*, Cout, H*r, W*r)
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolution kernel. Default is 1.
        factor (int, optional): Upsampling scale factor. Default is 2.
        mode (str, optional): Interpolation mode (e.g., 'bilinear', 'nearest'). Default is 'bilinear'.
    Shape:
        Input:  (*, in_channels, H, W)
        Output: (*, out_channels, H * factor, W * factor)
    Forward:
        1. Upsamples the input tensor using the specified interpolation mode and scale factor.
        2. Applies a convolution to the upsampled tensor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        factor: int = 2,
        mode: str = "bilinear",
    ) -> None:
        super().__init__()
        self.factor = factor
        self.mode = mode
        self.conv = ConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=self.factor, mode=self.mode)
        x = self.conv(x)
        return x


class ChannelDuplicatingPixelUnshuffleUpSampleLayer(nn.Module):
    """
    Upsampling layer that first duplicates input channels, then applies pixel shuffle.
    (*, Cin, H, W) -> (*, Cin, H*r, W*r) -> (*, Cout//r**2, H*r, W*r)
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels after upsampling.
        factor (int): Upsampling factor for height and width.
    Shape:
        Input:  (*, in_channels, H, W)
        Output: (*, out_channels, H * factor, W * factor)
    Process:
        1. Duplicates channels to match required output channels and pixel shuffle requirements.
        2. Applies pixel shuffle to rearrange channels into spatial dimensions.
    Raises:
        AssertionError: If out_channels * factor**2 is not divisible by in_channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert out_channels * factor**2 % in_channels == 0
        self.repeats = out_channels * factor**2 // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = F.pixel_shuffle(x, self.factor)
        return x


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """
    Drop paths (Stochastic Depth) per sample.
    Randomly drops entire residual paths during training for regularization.
    Args:
        x (torch.Tensor): Input tensor of any shape (B, ...).
        drop_prob (float): Probability of dropping a path.
        training (bool): Whether in training mode.
        scale_by_keep (bool): Whether to scale outputs by keep probability.
    Returns:
        torch.Tensor: Output tensor after drop path, same shape as input.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample as a module.
    Args:
        drop_prob (float): Probability of dropping a path.
        scale_by_keep (bool): Whether to scale outputs by keep probability.
    Input shape:
        (B, ...)
    Output shape:
        (B, ...)
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class GRN(nn.Module):
    """
    Global Response Normalization (GRN) layer.
    Applies normalization based on the global L2 norm of the input.
    Args:
        dim (int): Number of channels (last dimension).
    Input shape:
        (B, H, W, dim)
    Output shape:
        (B, H, W, dim)
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

