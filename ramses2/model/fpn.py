import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.ops import ConvLayer


class FPN(nn.Module):
    """
    PyTorch Feature Pyramid Network with normalization and activation using ConvLayer.
    Args:
        num_layers (int): Number of input feature maps (e.g. 3 or 4 for C5, C4, C3, (C2)).
        in_channels_list (list[int]): List of input channels for each feature map.
        pyramid_filters (int): Number of output channels for each FPN level.
        activation (callable): Activation function constructor (e.g. nn.GELU).
        normalization (callable): Normalization layer constructor (e.g. nn.GroupNorm).
        extra_layers (int): Number of extra FPN layers to add (stride 2 downsampling).
        interpolation (str): Interpolation mode for upsampling (default: 'bilinear').
    Input:
        features (list[Tensor]): List of feature maps, ordered from high to low resolution (C5, C4, C3, (C2)).
    Output:
        List of FPN feature maps (highest to lowest resolution).
    """

    def __init__(
        self,
        num_layers,
        in_channels_list,
        pyramid_filters=256,
        activation=None,
        normalization=None,
        extra_layers=0,
        interpolation="bilinear",
    ):
        super().__init__()
        assert len(in_channels_list) == num_layers
        self.num_layers = num_layers
        self.pyramid_filters = pyramid_filters
        self.extra_layers = extra_layers
        self.interpolation = interpolation
        self.activation = activation
        self.normalization = normalization

        # 1x1 convs to reduce channels for each input
        self.lateral_convs = nn.ModuleList(
            [
                ConvLayer(
                    in_channels=in_channels_list[i],
                    out_channels=pyramid_filters,
                    kernel_size=1,
                    norm=normalization,
                    activation=activation,
                )
                for i in range(num_layers)
            ]
        )
        # 3x3 convs for output smoothing
        self.output_convs = nn.ModuleList(
            [
                ConvLayer(
                    in_channels=pyramid_filters,
                    out_channels=pyramid_filters,
                    kernel_size=3,
                    norm=normalization,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        # Extra layers if needed
        self.extra_convs = nn.ModuleList(
            [
                ConvLayer(
                    in_channels=pyramid_filters,
                    out_channels=pyramid_filters,
                    kernel_size=3,
                    stride=2,
                    norm=normalization,
                    activation=activation,
                )
                for _ in range(extra_layers)
            ]
        )

    def forward(self, features):
        # features: list of tensors [C5, C4, C3, (C2)]
        assert len(features) == self.num_layers
        # Apply lateral 1x1 convs
        feats = [self.lateral_convs[i](features[i]) for i in range(self.num_layers)]
        # Build top-down path
        results = [feats[0]]  # Start from top (lowest spatial resolution)
        for i in range(1, self.num_layers):
            prev_shape = feats[i].shape[-2:]
            upsampled = F.interpolate(results[-1], size=prev_shape, mode=self.interpolation, align_corners=False)
            fused = feats[i] + upsampled
            results.append(fused)
        # Reverse to get high to low resolution order e.g. [C2, C3, etc.]
        results = results[::-1]
        # Apply 3x3 output convs
        results = [self.output_convs[i](results[i]) for i in range(self.num_layers)]
        # Add extra layers if needed
        for i in range(self.extra_layers):
            results.append(self.extra_convs[i](results[-1]))
        # returns features from highest resolution to lowest (low level to high level)
        return results
