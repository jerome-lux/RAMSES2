import math
import torch.nn as nn
from ..core.ops import ConvLayer
from ..core.blocks import ResBlock, BottleneckBlock, SepResBlock
from ..core.norm import LayerNorm2d
from ..utils.utils import make_tuple


class Resnet(nn.Module):
    """
    Resnet with classic convolutional residual blocks.

    Args:
        in_channels (int): Number of input channels. Default is 3.
        num_classes (int): Number of output classes. Default is 1000.
        depths (list of int): Number of residual blocks at each stage. Default is [3, 4, 6, 3].
        dims (list of int): Output channels for each stage. Default is [128, 256, 512, 1024].
        stem_filters (int): Number of filters in the stem convolution. Default is 64.
        stem_kernel_size (int): Kernel size for the stem convolution. Default is 7.
        stem_stride (int): Stride for the stem convolution. Default is 2.
        stem_act (nn.Module): Activation function for the stem. Default is nn.SiLU.
        activation (nn.Module or tuple): Activation function(s) for residual blocks. Default is nn.SiLU.
        use_bias (bool or tuple): Whether to use bias in convolutions. Default is False.
        norm (nn.Module or tuple): Normalization layer(s) for residual blocks. Default is nn.BatchNorm2d.
        fc_layers (list of int): Optional fully connected layers before the output. Default is [].
        head_activation (nn.Module or None): Activation function for the head. Default is None.
        groups (int): Number of groups for group convolutions. Default is 1.
        build_head (bool): Whether to build the classification head. Default is False.
        dropout (float): Dropout rate for the head. Default is 0.5.
        downsampling_method (str): Downsampling method in residual blocks. Default is "maxpooling".

    Attributes:
        stem (nn.Module): Initial convolutional stem.
        ops (nn.ModuleList): List of residual blocks.
        head (nn.ModuleList): Classification head layers.
        build_head (bool): Indicates if the head is built.
        depths (list of int): Number of blocks per stage.

    Methods:
        forward_features(x): Extracts features from input x.
        forward_head(x): Passes features through the classification head.
        forward(x): Full forward pass, including optional head.

    """

    def __init__(
        self,
        in_channels=3,
        num_classes=1000,
        depths=[3, 4, 6, 3],
        dims=[128, 256, 512, 1024],
        stem_filters=64,
        stem_kernel_size=7,
        stem_stride=2,
        activation=nn.SiLU,
        use_bias=False,
        norm=nn.BatchNorm2d,
        fc_layers=[],
        head_activation=None,
        groups=1,
        build_head=False,
        dropout=0.5,
        downsampling_method="maxpooling",
        return_intermediate_features=True,
    ):

        use_bias = make_tuple(use_bias, 3)
        norm = make_tuple(norm, 3)
        activation = make_tuple(activation, 3)

        super().__init__()
        self.build_head = build_head
        self.depths = depths
        self.stem = ConvLayer(
            in_channels,
            stem_filters,
            kernel_size=stem_kernel_size,
            stride=stem_stride,
            norm=norm[0],
            activation=activation[0],
            use_bias=use_bias[0],
        )
        self.return_intermediate_features = return_intermediate_features

        self.ops = nn.ModuleList()
        for i, depth in enumerate(depths):
            for j in range(depth):
                stride = 2 if j == 0 else 1
                if i == 0 and j == 0:
                    input_channels = stem_filters
                elif j == 0 and i > 0:
                    input_channels = dims[i - 1]
                else:
                    input_channels = dims[i]
                self.ops.append(
                    ResBlock(
                        input_channels,
                        dims[i],
                        nconv=2,
                        stride=stride,
                        norm=norm[1:],
                        use_bias=use_bias[1:],
                        activation=activation[1:],
                        groups=groups,
                        downsampling_method=downsampling_method,
                    )
                )
        self.head = nn.ModuleList()
        if fc_layers is not None:
            for depth in fc_layers:
                self.head.append(nn.Linear(dims[-1], depth, bias=True))
                if head_activation is not None:
                    self.head.append(head_activation())
                if dropout > 0:
                    self.head.append(nn.Dropout(dropout))
                dims[-1] = depth

        self.head.append(nn.Linear(dims[-1], num_classes))

    def forward_head(self, x):
        for op in self.head:
            x = op(x)
        return x

    def forward(self, x):
        x = self.stem(x)
        counter = 0
        outputs = {}
        for i, depth in enumerate(self.depths):
            for _ in range(depth):
                x = self.ops[counter](x)
                counter += 1
            if self.return_intermediate_features:
                outputs[f"C{i + 2}"] = x  # Store intermediate features

        if self.build_head:
            x = x.mean([-2, -1])  # global average pooling, (N, C, H, W) -> (N, C)
            x = self.forward_head(x)
            if self.return_intermediate_features:
                outputs["head"] = x

        if self.return_intermediate_features:
            return outputs
        else:
            return x


class Resnetb(nn.Module):
    """
    Resnet with bottleneck redidual block
    can be used to make a resnext when groups is > 1
    """

    def __init__(
        self,
        in_channels=3,
        num_classes=1000,
        depths=[3, 4, 6, 3],
        dims=[128, 256, 512, 1024],
        stem_filters=64,
        stem_kernel_size=7,
        stem_stride=2,
        se_module=False,
        bottleneck_ratio=4,
        activation=nn.SiLU,
        use_bias=False,
        norm=nn.BatchNorm2d,
        fc_layers=[],
        groups=1,
        build_head=False,
        head_activation=None,
        dropout=0.5,
        downsampling_method="maxpooling",
        return_intermediate_features=True,
    ):
        # parameters of each conv layer in the bottleneck block in index 1->3, stem in 0
        use_bias = make_tuple(use_bias, 4)
        norm = make_tuple(norm, 4)
        activation = make_tuple(activation, 4)

        super().__init__()
        self.build_head = build_head
        self.depths = depths
        self.return_intermediate_features = return_intermediate_features

        self.stem = ConvLayer(
            in_channels,
            stem_filters,
            kernel_size=stem_kernel_size,
            stride=stem_stride,
            norm=norm[0],
            activation=activation[0],
            use_bias=use_bias[0],
        )

        self.ops = nn.ModuleList()
        for i, depth in enumerate(depths):
            for j in range(depth):
                stride = 2 if j == 0 else 1
                if i == 0 and j == 0:
                    input_channels = stem_filters
                elif j == 0 and i > 0:
                    input_channels = dims[i - 1]
                else:
                    input_channels = dims[i]
                self.ops.append(
                    BottleneckBlock(
                        input_channels,
                        dims[i],
                        stride=stride,
                        bottleneck_ratio=bottleneck_ratio,
                        se=se_module,
                        norm=norm[1:],
                        use_bias=use_bias[1:],
                        activation=activation[1:],
                        groups=groups,
                        downsampling_method=downsampling_method,
                    )
                )

        self.head = nn.ModuleList()

        if fc_layers is not None:
            for depth in fc_layers:
                self.head.append(nn.Linear(dims[-1], depth, bias=True))
                if head_activation is not None:
                    self.head.append(head_activation())
                if dropout > 0:
                    self.head.append(nn.Dropout(dropout))
                dims[-1] = depth

        self.head.append(nn.Linear(dims[-1], num_classes))

    def forward_head(self, x):
        for op in self.head:
            x = op(x)
        return x

    def forward(self, x):
        x = self.stem(x)
        counter = 0
        outputs = {}
        for i, depth in enumerate(self.depths):
            for _ in range(depth):
                x = self.ops[counter](x)
                counter += 1
            if self.return_intermediate_features:
                outputs[f"C{i + 2}"] = x

        if self.build_head:
            x = x.mean([-2, -1])  # global average pooling, (N, C, H, W) -> (N, C)
            x = self.forward_head(x)
            if self.return_intermediate_features:
                outputs["head"] = x
        if self.return_intermediate_features:
            return outputs
        else:
            return x


class SepResnet(nn.Module):
    """
    Resnet with separable convolutional residual blocks.

    Args:
        in_channels (int): Number of input channels. Default is 3.
        num_classes (int): Number of output classes. Default is 1000.
        depths (list of int): Number of residual blocks at each stage. Default is [3, 4, 6, 3].
        dims (list of int): Output channels for each stage. Default is [128, 256, 512, 1024].
        stem_filters (int): Number of filters in the stem convolution. Default is 64.
        stem_kernel_size (int): Kernel size for the stem convolution. Default is 7.
        stem_stride (int): Stride for the stem convolution. Default is 2.
        stem_act (nn.Module): Activation function for the stem. Default is nn.SiLU.
        stem_norm (nn.Module): Normalization layer for the stem. Default is nn.BatchNorm2d.
        activation (nn.Module or tuple): Activation function(s) for residual blocks. Default is nn.SiLU.
        use_bias (bool or tuple): Whether to use bias in convolutions. Default is False.
        norm (nn.Module or tuple): Normalization layer(s) for residual blocks. Default is nn.BatchNorm2d.
        fc_layers (list of int): Optional fully connected layers before the output. Default is [].
        head_activation (nn.Module or None): Activation function for the head. Default is None.
        build_head (bool): Whether to build the classification head. Default is False.
        dropout (float): Dropout rate for the head. Default is 0.5.
        downsampling_method (str): Downsampling method in residual blocks. Default is "maxpooling".

    Attributes:
        stem (nn.Module): Initial convolutional stem.
        ops (nn.ModuleList): List of residual blocks.
        head (nn.ModuleList): Classification head layers.
        build_head (bool): Indicates if the head is built.
        depths (list of int): Number of blocks per stage.

    Methods:
        forward_features(x): Extracts features from input x.
        forward_head(x): Passes features through the classification head.
        forward(x): Full forward pass, including optional head.

    """

    def __init__(
        self,
        in_channels=3,
        num_classes=1000,
        depths=[3, 4, 6, 3],
        dims=[128, 256, 512, 1024],
        stem_filters=64,
        stem_kernel_size=7,
        stem_stride=2,
        activation=nn.SiLU,
        use_bias=False,
        norm=nn.BatchNorm2d,
        fc_layers=[],
        head_activation=None,
        build_head=False,
        dropout=0.5,
        downsampling_method="maxpooling",
        return_intermediate_features=False,
    ):

        use_bias = make_tuple(use_bias, 3)
        norm = make_tuple(norm, 3)
        activation = make_tuple(activation, 3)

        super().__init__()
        self.build_head = build_head
        self.depths = depths
        self.stem = ConvLayer(
            in_channels,
            stem_filters,
            kernel_size=stem_kernel_size,
            stride=stem_stride,
            norm=norm[0],
            activation=activation[0],
            use_bias=use_bias[0],
        )
        self.return_intermediate_features = return_intermediate_features

        self.ops = nn.ModuleList()
        for i, depth in enumerate(depths):
            for j in range(depth):
                stride = 2 if j == 0 else 1
                if i == 0 and j == 0:
                    input_channels = stem_filters
                elif j == 0 and i > 0:
                    input_channels = dims[i - 1]
                else:
                    input_channels = dims[i]
                self.ops.append(
                    SepResBlock(
                        input_channels,
                        dims[i],
                        nconv=2,
                        stride=stride,
                        norm=norm[1:],
                        use_bias=use_bias[1:],
                        activation=activation[1:],
                        downsampling_method=downsampling_method,
                    )
                )
        self.head = nn.ModuleList()
        if fc_layers is not None:
            for depth in fc_layers:
                self.head.append(nn.Linear(dims[-1], depth, bias=True))
                if head_activation is not None:
                    self.head.append(head_activation())
                if dropout > 0:
                    self.head.append(nn.Dropout(dropout))
                dims[-1] = depth

        self.head.append(nn.Linear(dims[-1], num_classes))

    def forward_head(self, x):
        for op in self.head:
            x = op(x)
        return x

    def forward(self, x):
        x = self.stem(x)
        counter = 0
        outputs = {}
        for i, depth in enumerate(self.depths):
            for _ in range(depth):
                x = self.ops[counter](x)
                counter += 1
            if self.return_intermediate_features:
                outputs[f"C{i + 2}"] = x  # Store intermediate features

        if self.build_head:
            x = x.mean([-2, -1])  # global average pooling, (N, C, H, W) -> (N, C)
            x = self.forward_head(x)
            if self.return_intermediate_features:
                outputs["head"] = x

        if self.return_intermediate_features:
            return outputs
        else:
            return x


def resnet18(**kwargs):

    return Resnet(dims=[64, 128, 256, 512], depths=[2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):

    return Resnet(dims=[64, 128, 256, 512], depths=[3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):

    return Resnetb(dims=[256, 512, 1024, 2048], depths=[3, 4, 6, 3], **kwargs)


def resnext50(**kwargs):

    return Resnetb(dims=[256, 512, 1024, 2048], depths=[3, 4, 6, 3], groups=32, bottleneck_ratio=2, **kwargs)
