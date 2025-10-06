import json
from pathlib import Path
import os


class Config:

    def __init__(self, **kwargs):
        """
        Initializes the configuration for the model with default parameters, allowing overrides via keyword arguments.

        Parameters
        ----------
        load_backbone : bool or str
            Whether to load a backbone model (default: False). If str or Path, it's the path of the .pt file
        backbone_params : dict, optional
            Parameters for the backbone model (default: {}).
        backbone : str, optional
            Name of the backbone architecture (default: "Resnext").
        backbone_source : str, optional
            Source library for the backbone (default: "torchvision").
        backbone_feature_nodes : dict, optional
            Mapping of backbone stages to feature node names (default: {"stages.0": "C2", ...}).
        ncls : int, optional
            Number of classes (default: 1).
        imshape : tuple, optional
            Input image shape (default: (4096, 6144, 3)).
        mask_stride : int, optional
            Stride for mask prediction, must match backbone/output_level (default: 4).
        activation : str, optional
            Activation function to use (default: "SiLU").
        normalization : str, optional
            Normalization layer type (default: "GroupNorm").
        normalization_kw : dict, optional
            Keyword arguments for normalization (default: {"num_groups": 32}).
        connection_layers : dict, optional
            Mapping of FPN connection layers to output channels (default: {"C2": 256, ...}).
        FPN_filters : int, optional
            Number of filters in FPN layers (default: 256).
        extra_FPN_layers : int, optional
            Number of extra FPN layers after P5 (default: 1).
        strides : list, optional
            Strides for each FPN level (default: [4, 8, 16, 32, 64]).
        head_layers : int, optional
            Number of repeated convolutional layers in the head (default: 4).
        head_filters : int, optional
            Number of filters in head layers (default: 256).
        kernel_size : int, optional
            Kernel size for head convolutions (default: 1).
        grid_sizes : list, optional
            Grid sizes for each FPN level (default: [64, 36, 24, 16, 12]).
        point_nms : bool, optional
            Whether to use point NMS in mask head (default: False).
        mask_mid_filters : int, optional
            Number of filters in intermediate mask head layers (default: 128).
        mask_output_filters : int, optional
            Number of filters in mask output layer (default: 256).
        geom_feat_convs : int, optional
            Number of convolutional layers in geometry factor branch (default: 2).
        geom_feats_filters : int, optional
            Number of filters in geometry feature convolutions (default: 128).
        mask_output_level : int, optional
            FPN level for unified mask output (default: 0).
        FPN_output_upscaling : bool, optional
            Whether to upscale FPN output (default: False).
        sigma_nms : float, optional
            Sigma value for NMS (default: 0.5).
        min_area : int, optional
            Minimum area for mask filtering (default: 0).
        use_binary_masks : bool, optional
            Whether to use binary masks in training (default: True).
        lossweights : list, optional
            Weights for different loss components (default: [1.0, 1.0, 1.0]).
        max_pos_samples : int, optional
            Maximum number of positive ground truth samples for loss computation (default: 512).
        scale_ranges : list, optional
            Scale ranges for target allocation per FPN level (default: [[1, 96], ...]).
        offset_factor : float, optional
            Offset factor for target assignment (default: 0.25).
        **kwargs
            Additional keyword arguments to override default parameters.
        """

        self.load_backbone = False
        self.backbone_params = {}
        self.backbone = "Resnext"
        self.backbone_source = "local"
        self.backbone_feature_nodes = {
            "ops.2": "C2",
            "ops.6": "C3",
            "ops.12": "C4",
            "ops.15": "C5",
        }

        # Specific params
        self.ncls = 1
        self.imshape = (4096, 6144, 3)
        self.mask_stride = 4  # It must match with the backbone and the param 'output_level'

        # General layers params
        self.activation = "SiLU"
        self.normalization = "GroupNorm"
        self.normalization_kw = {"num_groups": 32}

        # FPN connection_layer smust be a dict with the id of the layer as key and its output channels as value
        self.connection_layers = {"C2": 256, "C3": 512, "C4": 1024, "C5": 2048}  # backbone connection layers,

        self.FPN_filters = 256
        self.extra_FPN_layers = 1  # layers after P5. Strides must correspond to the number of FPN layers !

        # SOLO head
        self.strides = [4, 8, 16, 32, 64]  # strides of FPN levels
        self.head_layers = 4  # Number of repeats of head conv layers
        self.head_filters = 256
        self.kernel_size = 1
        self.grid_sizes = [64, 36, 24, 16, 12]

        # SOLO MASK head
        self.point_nms = False
        self.mask_mid_filters = 128
        self.mask_output_filters = 256
        self.geom_feat_convs = 2  # number rof convs in the geometry factor branch
        self.geom_feats_filters = 128
        self.mask_output_level = 0  # size of the unified mask (in level of the FPN)
        self.FPN_output_upscaling = False
        self.sigma_nms = 0.5
        self.min_area = 0

        # Training Params
        # density head option
        self.use_binary_masks = True

        # loss and training parameters
        self.lossweights = [1.0, 1.0, 1.0]
        self.max_pos_samples = (
            512  # limit the number of positive gt samples when computing loss to limit memory footprint
        )

        # tagets allocation
        self.scale_ranges = [[1, 96], [48, 192], [96, 384], [192, 768], [384, 2048]]  # if P2 level is stride 4
        self.offset_factor = 0.25

        # Update defaults parameters with kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):

        s = ""

        for k, v in self.__dict__.items():
            s += "{}:{}\n".format(k, v)

        return s

    def save(self, filename):

        # data = {k:v for k, v in self.__dict__.items()}

        p = Path(filename).parent.absolute()
        if not os.path.isdir(p):
            os.mkdir(p)

        with open(filename, "w") as f:
            json.dump(self.__dict__, f)
