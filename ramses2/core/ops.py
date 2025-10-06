import math
from typing import Union
import torch
from torch import nn
import torch.nn.functional as F
import torch.fft

from ..utils import utils


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


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


class LinearLayer(nn.Module):
    """
    Linear (fully connected) layer with optional dropout, normalization, and activation.
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        use_bias (bool): Whether to use bias in the linear layer. Default: True.
        dropout (float): Dropout probability. Default: 0.
        norm (callable): Normalization layer constructor. Default: None.
        activation (callable): Activation function constructor. Default: None.
    Input shape:
        (B, *, in_features) or (B, in_features, ...)
    Output shape:
        (B, out_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias=True,
        dropout=0,
        norm=None,
        activation=None,
    ):
        super(LinearLayer, self).__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.norm = norm(out_features) if norm is not None else None
        self.act = activation() if activation is not None else None

    def _try_squeeze(self, x: torch.Tensor) -> torch.Tensor:
        """
        Flattens all dimensions except batch to (B, N) if input has more than 2 dims.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for LinearLayer.
        Args:
            x (torch.Tensor): Input tensor of shape (B, *, in_features) or (B, in_features, ...)
        Returns:
            torch.Tensor: Output tensor of shape (B, out_features)
        """
        x = self._try_squeeze(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
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


class TuckerSpectralConv2d(nn.Module):
    """TuckerSpectralConv2d implements a 2D spectral convolution layer with Tucker decomposition using complex-valued parameters.
    This layer factorizes the spectral convolution weights into four complex tensors (U, V, S, G) according to the Tucker decomposition,
    significantly reducing the number of parameters compared to a standard spectral convolution.

    The number of parameters is equal to (Cin * r1 + Cout * r2 + (modesx * modesy) * r3 + r1 * r2 * r3)
    vs (Cin * Cout * modes_x * modes_y without factorization).
    Typical choices:
    r1 = Cin / k
    r2 = Cout / k
    r3 = modes_x * modes_y / k'
    where k & k' are the desired reduction factors (2, 4, etc).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        modes (tuple): (modes_x, modes_y), number of low-frequency modes to keep along height and width.
        ranks (tuple): (r1, r2, r3), Tucker decomposition ranks.
        scaling (int or float, optional): Output spatial scaling factor. Default is 1.
    Shape:
        Input: (batch_size, in_channels, H, W)
        Output: (batch_size, out_channels, H * scaling, W * scaling)
    Notes:
        - The number of spectral modes kept is limited by the input spatial size.
        - Parameters are initialized with adapted Xavier initialization for complex values.
        - The layer operates in the Fourier domain and reconstructs the output via inverse FFT."""

    def __init__(self, in_channels, out_channels, modes, ranks, scaling: Union[int, float] = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x, self.modes_y = modes
        self.scaling = scaling

        self.r1, self.r2, self.r3 = ranks

        # Facteurs de la décomposition de Tucker
        self.U = nn.Parameter(torch.empty(in_channels, self.r1, dtype=torch.cfloat))  # (C_in, r1)
        self.V = nn.Parameter(torch.empty(out_channels, self.r2, dtype=torch.cfloat))  # (C_out, r2)
        self.S = nn.Parameter(torch.empty(self.modes_x, self.modes_y, self.r3, dtype=torch.cfloat))  # (mx, my, r3)
        self.G = nn.Parameter(torch.empty(self.r1, self.r2, self.r3, dtype=torch.cfloat))  # (r1, r2, r3)

        # Initialisation des paramètres complexes
        self._initialize_parameters_complex()

    def _initialize_parameters_complex(self):
        with torch.no_grad():
            # Option 1: Initialisation Réelle/Imaginaire séparée avec Kaiming (adapté)
            nn.init.xavier_normal_(self.U.real)
            nn.init.xavier_normal_(self.V.real)
            nn.init.xavier_normal_(self.S.real)
            nn.init.xavier_normal_(self.G.real)

            self.U.imag.copy_(torch.randn_like(self.U.imag) * 0.01)
            self.V.imag.copy_(torch.randn_like(self.V.imag) * 0.01)
            self.S.imag.copy_(torch.randn_like(self.S.imag) * 0.01)
            self.G.imag.copy_(torch.randn_like(self.G.imag) * 0.01)

    def forward(self, x):
        batchsize, C_in, H, W = x.shape
        device = x.device

        assert C_in == self.in_channels, f"Les canaux d'entrée attendus sont {self.in_channels}, mais reçus {C_in}"

        # Les dimensions spectrales après rfft2(H, W) sont (H, W//2 + 1)
        max_spectral_modes_x = H
        max_spectral_modes_y = W // 2 + 1

        if self.modes_x > max_spectral_modes_x:
            raise ValueError(
                f"Number of modes in x ({self.modes_x}) exceeds available spectral dimension in height ({max_spectral_modes_x}) "
                f"for input spatial size ({H}, {W}). modes_x must be <= H."
            )
        if self.modes_y > max_spectral_modes_y:
            raise ValueError(
                f"Number of modes in y ({self.modes_y}) exceeds available spectral dimension in width ({max_spectral_modes_y}) "
                f"for input spatial size ({H}, {W}). modes_y must be <= W//2 + 1."
            )

        # 1. FFT -> domaine spectral (x_ft est complexe)
        x_ft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")  # (B, C_in, H, W//2 + 1), cfloat

        # 2. Sélection des modes bas (garde les dimensions mx, my)
        x_ft_low_modes = x_ft[:, :, : self.modes_x, : self.modes_y]  # (B, C_in, mx, my), cfloat

        # 3. Reconstruire les poids spectraux W_hat (maintenant directement complexe)
        G1 = torch.einsum("ij,jkl->ikl", self.U, self.G)  # (C_in, r2, r3), cfloat
        G2 = torch.einsum("ok,ikl->iol", self.V, G1)  # (C_in, C_out, r3), cfloat
        # Correction: einsum adaptée pour S(mx, my, r3) et G2(i, o, r3)
        W_hat = torch.einsum("iol,xyl->ioxy", G2, self.S)  # (C_in, C_out, modes_x, modes_y), cfloat

        # Plus besoin de caster W_hat
        # W_hat = W_hat.to(dtype=torch.cfloat)

        # 4. Appliquer la convolution spectrale (multiplication de tenseurs complexes)
        # y_ft_low_modes[b, o, mx, my] = sum_i (x_ft_low_modes[b, i, mx, my] * W_hat[i, o, mx, my])
        y_ft_low_modes = torch.einsum("bixy,ioxy->boxy", x_ft_low_modes, W_hat)  # (B, C_out, mx, my), cfloat

        # 5. Zero-padding
        out_ft = torch.zeros(batchsize, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=device)
        out_ft[:, :, : self.modes_x, : self.modes_y] = y_ft_low_modes  # Copie de tenseurs complexes

        # 6. Retour au domaine spatial
        y = torch.fft.irfft2(
            out_ft, s=(int(H * self.scaling), int(W * self.scaling)), dim=(-2, -1), norm="ortho"
        )  # (B, C_out, H, W), float

        return y


class SpectralConv2d(nn.Module):
    """SpectralConv2d implements a 2D spectral convolution layer using the Fourier domain.
        modes (tuple[int, int]): Number of low-frequency Fourier modes to retain in (height, width) dimensions.
        scaling (int | float, optional): Scaling factor for output spatial size. Default is 1 (no scaling).
    Attributes:
        spectral_weights (torch.nn.Parameter): Learnable complex weights for spectral convolution.
    Forward Args:
        x (torch.Tensor): Input tensor of shape (batch, in_channels, height, width).
    Returns:
        torch.Tensor: Output tensor of shape (batch, out_channels, scaled_height, scaled_width).
    Raises:
        ValueError: If the specified number of modes exceeds the available spectral dimensions.
    Notes:
        - Performs FFT on input, applies learned spectral weights to selected low-frequency modes, zero-pads the rest, and returns to spatial domain via inverse FFT.
        - Only the specified number of low-frequency modes are convolved; higher frequencies are set to zero."""

    def __init__(self, in_channels: int, out_channels: int, modes: tuple[int, int], scaling: Union[int, float] = 1):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x, self.modes_y = modes
        self.scaling = scaling

        # Spectral convolution kernel (learned parameter)
        # Its shape is (out_channels, in_channels, modes_x, modes_y)
        self.spectral_weights = nn.Parameter(
            torch.empty(out_channels, in_channels, self.modes_x, self.modes_y, dtype=torch.cfloat)
        )

        # Initialize complex parameters (Xavier Uniform)
        self._initialize_parameters_complex_xavier()

    def _initialize_parameters_complex_xavier(self):
        # Apply Xavier uniform separately to the real and imaginary parts
        with torch.no_grad():
            # Initialize the real part
            nn.init.xavier_uniform_(self.spectral_weights.real)
            # Initialize the imaginary part
            nn.init.xavier_uniform_(self.spectral_weights.imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize, C_in, H, W = x.shape
        device = x.device

        assert C_in == self.in_channels, f"Expected input channels are {self.in_channels}, but received {C_in}"

        # Spectral dimensions after rfft2(H, W) are (H, W//2 + 1)
        max_spectral_modes_x = H
        max_spectral_modes_y = W // 2 + 1

        if self.modes_x > max_spectral_modes_x:
            raise ValueError(
                f"Number of modes in x ({self.modes_x}) exceeds available spectral dimension in height ({max_spectral_modes_x}) "
                f"for input spatial size ({H}, {W}). modes_x must be <= H."
            )
        if self.modes_y > max_spectral_modes_y:
            raise ValueError(
                f"Number of modes in y ({self.modes_y}) exceeds available spectral dimension in width ({max_spectral_modes_y}) "
                f"for input spatial size ({H}, {W}). modes_y must be <= W//2 + 1."
            )

        # 1. FFT -> spectral domain (x_ft is complex)
        # (B, C_in, H, W//2 + 1) after rfft2 of a (B, C_in, H, W) input
        x_ft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")

        # 2. Select low modes (keep mx, my dimensions)
        # (B, C_in, modes_x, modes_y)
        x_ft_low_modes = x_ft[:, :, : self.modes_x, : self.modes_y]

        # 3. Apply spectral convolution (complex tensor multiplication)
        # This is element-wise multiplication per mode (mx, my)
        # followed by a summation over input channels (C_in) to get output channels (C_out).
        # y_ft_low_modes_out[b, o, mx, my] = sum_i (x_ft_low_modes[b, i, mx, my] * self.spectral_weights[o, i, mx, my])
        # einsum('bixy, oixy -> boxy')
        # b: batch, i: in_channels, o: out_channels, x: modes_x, y: modes_y
        y_ft_low_modes_out = torch.einsum(
            "bixy, oixy -> boxy", x_ft_low_modes, self.spectral_weights
        )  # (B, C_out, modes_x, modes_y)

        # 4. Zero-padding of ignored frequencies
        # Create a tensor of zeros with the full spectral size after rfft2
        out_ft = torch.zeros(batchsize, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=device)
        # Copy the calculated modes to the low-frequency positions
        out_ft[:, :, : self.modes_x, : self.modes_y] = y_ft_low_modes_out

        # 5. Return to spatial domain
        # irfft2 to get a real output of size (H, W)
        y = torch.fft.irfft2(
            out_ft, s=(int(H * self.scaling), int(W * self.scaling)), dim=(-2, -1), norm="ortho"
        )  # (B, C_out, H, W), float

        return y


class CartesianEmbedding(nn.Module):
    """
    Generates and concatenates normalized Cartesian coordinates (x, y) as additional channels.
    Coordinates are normalized to the range [-1, 1] using torch.meshgrid.
    """

    def __init__(self):
        super().__init__()
        # No learnable parameters needed for this embedding layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates normalized (x, y) coordinates using meshgrid and concatenates them to the input.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            torch.Tensor: Input tensor concatenated with normalized x and y coordinates.
                          Shape (B, C + 2, H, W).
        """
        batch_size, _, height, width = x.shape
        device = x.device

        # Generate 1D coordinate vectors normalized to [-1, 1]
        x_lin = torch.linspace(-1, 1, steps=width, device=device)  # Shape (W,)
        y_lin = torch.linspace(-1, 1, steps=height, device=device)  # Shape (H,)

        # Use meshgrid to create 2D grids of coordinates
        # grid_y will have shape (H, W), grid_x will have shape (H, W)
        # Use indexing='ij' to match the (H, W) spatial dimensions ordering
        grid_y, grid_x = torch.meshgrid(y_lin, x_lin, indexing="ij")

        coords = torch.stack([grid_x, grid_y], dim=-1)  # Shape (H, W, 2)

        # Permute dimensions to get shape (2, H, W) and add batch dimension (1, 2, H, W)
        # Permute moves the coordinate dimension (originally last) to the first position
        # unsqueeze(0) adds the batch dimension at the start
        coords = coords.permute(2, 0, 1).unsqueeze(0)  # Shape (1, 2, H, W)

        # Expand for the batch size to match the input tensor batch size
        coords = coords.expand(batch_size, -1, -1, -1)  # Shape (B, 2, H, W)

        # Concatenate with the input tensor along the channel dimension
        # Output shape (B, C_in + 2, H, W)
        output = torch.cat([x, coords], dim=1)

        return output


class SinusoidalEmbedding(nn.Module):
    """
    Generates and concatenates sinusoidal positional embeddings (x, y) as additional channels.
    Uses sine and cosine function pairs at multiple frequencies.
    Coordinates are first normalized to the range [0, 1].
    frequencies are multiple of 2 * math.pi * (2 ** d) where d=0 to num_frequency-1
    The embeddings are periodic in the image (useful for PDEs with periodic BCs!)
    **Note**: This is NOT the same as the positionnal embeddings found in vision transformer, where the
    frequencies are given by 1 / (10000 ** ((2 * i // 2) / num_freq)), where i goes from 0 to num_freq-1
    """

    def __init__(self, num_frequencies: int = 10):
        """
        Args:
            num_frequencies (int): The number of sinusoidal frequency pairs (sin/cos)
                                   per spatial dimension (x and y).
                                   Total added channels = num_frequencies * 2 (sin/cos) * 2 (x/y).
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        # Define frequencies. Commonly powers of 2 multiplied by 2*pi.
        # E.g., frequencies = [2*pi*2^0, 2*pi*2^1, ..., 2*pi*2^(num_frequencies-1)]
        self.frequencies = 2 * math.pi * (2 ** torch.arange(num_frequencies))
        # Frequencies are not learnable parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates sinusoidal positional embeddings and concatenates them to the input.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            torch.Tensor: Input tensor concatenated with sinusoidal positional embeddings.
                          Shape (B, C + num_frequencies * 4, H, W).
        """
        batch_size, _, height, width = x.shape
        device = x.device

        # Move frequencies to the correct device
        frequencies = self.frequencies.to(device)  # Shape (num_frequencies,)

        # Generate x and y coordinates normalized to [0, 1]
        # Used to calculate the arguments for the sinusoidal functions
        x_coords_normalized = torch.linspace(0, 1, steps=width, device=device)  # Shape (W,)
        y_coords_normalized = torch.linspace(0, 1, steps=height, device=device)  # Shape (H,)

        # Create a full grid of normalized coordinates
        # grid_y (H, W), grid_x (H, W)
        grid_y, grid_x = torch.meshgrid(y_coords_normalized, x_coords_normalized, indexing="ij")

        # Stack coordinates to get shape (H, W, 2)
        grid_coords = torch.stack([grid_x, grid_y], dim=-1)  # Shape (H, W, 2)

        # Apply frequencies. Broadcast frequencies (1, 1, 1, num_frequencies)
        # over coordinates (H, W, 2, 1) -> (H, W, 2, num_frequencies)
        grid_frequencies = frequencies.view(1, 1, 1, self.num_frequencies)
        grid_coords_freq = grid_coords.unsqueeze(-1) * grid_frequencies  # Shape (H, W, 2, num_frequencies)

        # Apply sin and cos. Along the last dimension, we have [x_freq1, y_freq1, x_freq2, y_freq2, ...]
        # We want [sin(x_freq1), cos(x_freq1), sin(y_freq1), cos(y_freq1), ...]
        # We can concatenate sin and cos applied separately.
        sin_vals = torch.sin(grid_coords_freq)  # (H, W, 2, num_frequencies)
        cos_vals = torch.cos(grid_coords_freq)  # (H, W, 2, num_frequencies)

        # Concatenate sin and cos for each coordinate and each frequency
        # Resulting shape (H, W, 2, num_frequencies * 2)
        grid_embeddings = torch.cat([sin_vals, cos_vals], dim=-1)

        # Rearrange dimensions to have embedding channels after the batch
        # From (H, W, 2, num_frequencies * 2) to (2 * num_frequencies * 2, H, W)
        # then (B, 2 * num_frequencies * 2, H, W)
        # permute(2, 3, 0, 1) -> (2, num_frequencies * 2, H, W)
        # reshape(1, -1, height, width) -> (1, total_embedding_channels, H, W)
        grid_embeddings = grid_embeddings.permute(2, 3, 0, 1).reshape(
            1, -1, height, width
        )  # Shape (1, num_frequencies * 4, H, W)

        # Expand over the batch dimension
        grid_embeddings = grid_embeddings.expand(batch_size, -1, -1, -1)  # Shape (B, num_frequencies * 4, H, W)

        # Concatenate with the input tensor
        # The output will have shape (B, C_in + num_frequencies * 4, H, W)
        output = torch.cat([x, grid_embeddings], dim=1)

        return output


class FiniteDifferenceConvolution(nn.Module):
    """Finite Difference Convolution Layer introduced in [1]_.
    "Neural Operators with Localized Integral and Differential Kernels" (ICML 2024)
        https://arxiv.org/abs/2402.16845

    Computes a finite difference convolution on a regular grid,
    which converges to a directional derivative as the grid is refined.

    Parameters
    ----------
    in_channels : int
        number of in_channels
    out_channels : int
        number of out_channels
    n_dim : int
        number of dimensions in the input domain
    kernel_size : int
        odd kernel size used for convolutional finite difference stencil
    groups : int
        splitting number of channels
    padding : literal {'periodic', 'replicate', 'reflect', 'zeros'}
        mode of padding to use on input.
        See `torch.nn.functional.padding`.

    References
    ----------
    .. [1] : Liu-Schiaffini, M., et al. (2024). "Neural Operators with
        Localized Integral and Differential Kernels".
        ICML 2024, https://arxiv.org/abs/2402.16845.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_dim=2,
        kernel_size=3,
        groups=1,
        stride=1,
        padding="same",
    ):

        super().__init__()

        self.conv_function = getattr(F, f"conv{n_dim}d")

        assert kernel_size % 2 == 1, "Kernel size should be odd"
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.groups = groups
        self.n_dim = n_dim
        self.padding = padding
        self.stride = stride

        # init kernel weigths
        self.weights = torch.rand((out_channels, in_channels // groups, kernel_size, kernel_size))
        k = torch.sqrt(torch.tensor(groups / (in_channels * kernel_size**2)))
        self.weights = self.weights * 2 * k - k

    def forward(self, x, grid_width: float = 1.0) -> torch.Tensor:
        """FiniteDifferenceConvolution's forward pass.

        Parameters
        ----------
        x : torch.tensor
            input tensor, shape (batch, in_channels, d_1, d_2, ...d_n)
        grid_width : float
            discretization size of input grid
        """

        self.weights = self.weights.to(x.device)
        x = (
            self.conv_function(
                x,
                (self.weights - torch.mean(self.weights)),
                groups=self.groups,
                stride=self.stride,
                padding=self.padding,
            )
            / grid_width
        )
        return x


class FiniteDifferenceLayer(nn.Module):
    """Finite Difference Layer introduced in [1]_.
    "Neural Operators with Localized Integral and Differential Kernels" (ICML 2024)
        https://arxiv.org/abs/2402.16845

    Computes a finite difference convolution on a regular grid,
    which converges to a directional derivative as the grid is refined.

    Parameters
    ----------
    in_channels : int
        number of in_channels
    out_channels : int
        number of out_channels
    n_dim : int
        number of dimensions in the input domain
    kernel_size : int
        odd kernel size used for convolutional finite difference stencil
    groups : int
        splitting number of channels
    padding : literal {'periodic', 'replicate', 'reflect', 'zeros'}
        mode of padding to use on input.
        See `torch.nn.functional.padding`.

    References
    ----------
    .. [1] : Liu-Schiaffini, M., et al. (2024). "Neural Operators with
        Localized Integral and Differential Kernels".
        ICML 2024, https://arxiv.org/abs/2402.16845.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_dim=2,
        kernel_size=3,
        groups=1,
        stride=1,
        padding="same",
        norm=None,
        activation=None,
        grid_width: float = 1.0,
    ):
        super().__init__()
        self.fdc = FiniteDifferenceConvolution(
            in_channels,
            out_channels,
            n_dim=n_dim,
            kernel_size=kernel_size,
            groups=groups,
            stride=stride,
            padding=padding,
        )
        self.normalization = norm(out_channels) if norm is not None else None
        self.activation = activation() if activation is not None else None
        self.grid_width = grid_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fdc(x, self.grid_width)
        if self.normalization is not None:
            x = self.normalization(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
