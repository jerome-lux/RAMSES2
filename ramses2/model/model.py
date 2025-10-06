# coding: utf-8
from functools import partial
import torch
import torch.nn as nn

from ..backbones.build_backbone import build
from ..core.ops import ConvLayer
from .fpn import FPN
from ..utils.utils import pad_with_coords, get_class


class RAMSESModel(nn.Module):
    """
    RAMSESModel is a neural network module for instance segmentation and mass prediction
    It integrates a backbone network, a Feature Pyramid Network (FPN),
    and specialized heads for classification, mass and mask prediction.

    Args:
        config (object): Configuration object containing all necessary parameters for model construction,
            including backbone details, FPN settings, head configurations, activation functions, and normalization.
            See config.py

    Attributes:
        config: Stores the configuration object.
        kernel_depth: Number of output filters for the mask head times the square of the kernel size.
        ncls: Number of classes.
        strides: Stride values for the backbone.
        backbone: Backbone network module.
        activation: Activation function used throughout the model.
        normalization: Normalization layer constructor.
        FPN: Feature Pyramid Network module.
        shared_heads: Head module for classification and kernel prediction.
        mask_head: Head module for mask and geometry feature prediction.

    Methods:
        forward(x, training=False, cls_threshold=0.5, nms_cls_threshold=0.5, mask_threshold=0.5,
                max_detections=768, min_area=0):
            Performs a forward pass through the model.

            Args:
                x (Tensor): Input tensor of shape (B, C, H, W).
                training (bool, optional): If True, returns raw predictions for training. If False, returns
                    post-processed instance segmentation results. Default is False.
                cls_threshold (float, optional): Threshold for class probability to consider a detection. Default is 0.5.
                nms_threshold (float, optional): Threshold during non-maximum suppression. Default is 0 [correspond to iou threshold for soft and greedy NMS and to cls threshold in MatrixNMS]
                mask_threshold (float, optional): Threshold for mask probability to consider a pixel as part of an instance. Default is 0.5.
                max_detections (int, optional): Maximum number of instances to detect per image. Default is 768.
                min_area (int, optional): Minimum area (in pixels) for a detected instance to be kept. Default is 0.

            Returns:
                If training is True:
                    Tuple of raw predictions:
                        - flat_pred_cls (Tensor): Flattened class predictions (all levels).
                        - flat_cls_factor (Tensor): Flattened class factors (all levels).
                        - flat_pred_kernel (Tensor): Flattened kernel predictions (all levels).
                        - seg_preds (Tensor): Segmentation mask predictions (C, H//mask_stride, W//mask_stride).
                        - geom_features (Tensor): Geometry feature predictions (1, H//mask_stride, W//mask_stride).
                If training is False:
                    List[Dict]: List of dictionaries (one per image in the batch) containing:
                        - "masks": Predicted instance masks (N, H//mask_stride, W//mask_stride).
                        - "scores": Confidence scores for each instance (N).
                        - "cls_labels": Predicted class labels for each instance (N).
                        - "masses": Normalized mass for each instance (N).
    """

    def __init__(self, config):
        super(RAMSESModel, self).__init__()
        self.config = config
        self.kernel_depth = config.mask_output_filters * config.kernel_size**2
        self.ncls = config.ncls
        self.strides = config.strides
        self.backbone = config.backbone
        self.activation = getattr(nn, config.activation)
        self.normalization = partial(get_class(config.normalization), **self.config.normalization_kw)

        if config.backbone_params.get("activation", None) is not None:
            config.backbone_params.update({"activation": getattr(nn, config.backbone_params["activation"])})
        if config.backbone_params.get("normalization", None) is not None:
            norm_kwargs = config.backbone_params.pop("normalization_kw", {})
            bb_norm = partial(get_class(config.backbone_params["normalization"]), **norm_kwargs)
            config.backbone_params.update({"normalization": bb_norm})

        self.backbone = build(
            name=config.backbone,
            return_nodes=config.backbone_feature_nodes,
            load=config.load_backbone,
            source=config.backbone_source,
            **config.backbone_params
        )

        self.FPN = FPN(
            num_layers=len(config.connection_layers),
            in_channels_list=[
                config.connection_layers[i] for i in sorted(config.connection_layers.keys(), reverse=True)
            ],
            pyramid_filters=config.FPN_filters,
            activation=None,
            normalization=None,
            extra_layers=config.extra_FPN_layers,
            interpolation="bilinear",
        )

        self.shared_heads = RAMSESHead(
            ncls=config.ncls,
            filters_in=config.FPN_filters + 2,
            head_layers=config.head_layers,
            conv_filters=config.head_filters,
            kernel_filters=config.mask_output_filters * config.kernel_size**2,
            activation=self.activation,
            normalization=self.normalization,
        )

        self.mask_head = RAMSESMaskHead(
            num_levels=len(self.config.connection_layers),  # number of FPN levels expected
            in_ch=config.FPN_filters + 2,
            mid_ch=self.config.mask_mid_filters,
            geom_ch=self.config.geom_feats_filters,
            out_ch=self.config.mask_output_filters,
            nconv=self.config.geom_feat_convs,
            output_level=self.config.mask_output_level,
            upscaling=self.config.FPN_output_upscaling,
            activation=self.activation,
            normalization=self.normalization,
        )

        # initialization of the kernel branch
        # for module in self.shared_heads.kernel_head:
        #     nn.init.uniform_(module.ops[1].weight, 0, 0.001)
        # nn.init.uniform_(self.shared_heads.kernel_out.ops[1].weight, 0, 0.001)

    def forward(
        self,
        x,
        training=False,
        cls_threshold=0.5,
        nms_threshold=0.5,
        mask_threshold=0.5,
        max_detections=768,
        scale_by_mask_scores=True,
        min_area=0,
        nms_mode="greedy",
    ):

        fpn_inputs = self.backbone(x)  # dict of feature maps at different levels
        # keep only the levels we need, sorted from the highest to the lowest level [C5->C3 or C2 for example]
        fpn_inputs = [fpn_inputs[lvl] for lvl in sorted(self.config.connection_layers.keys(), reverse=True)]
        flat_pred_cls = []
        flat_cls_factor = []
        flat_pred_kernel = []
        fpn_outputs = self.FPN(
            fpn_inputs
        )  # list of FPN levels from high res (low level) to low res (high level) features, eg. [(C2), C3, etc..]
        for i, feature in enumerate(fpn_outputs):
            H, W = feature.shape[-2:]
            maxdim = max(H, W)
            g = self.config.grid_sizes[i]
            size = (int(round(g * H / maxdim)), int(round(g * W / maxdim)))
            if size[0] != H or size[1] != W:
                rescaled_features = nn.functional.interpolate(feature, size=size, mode="bilinear", align_corners=False)
            else:
                rescaled_features = feature
            class_probs, class_factor, kernel_features = self.shared_heads(
                pad_with_coords(rescaled_features)
            )  # predictions for each FPN level
            B, C_cls, H, W = class_probs.shape
            _, C_fac, _, _ = class_factor.shape
            _, C_ker, _, _ = kernel_features.shape
            # flatten the predcitions and add to list
            flat_pred_cls.append(class_probs.reshape(B, C_cls, -1))
            flat_cls_factor.append(class_factor.reshape(B, C_fac, -1))
            flat_pred_kernel.append(kernel_features.reshape(B, C_ker, -1))

        flat_pred_cls = torch.cat(flat_pred_cls, dim=2)  # (B, ncls, N)
        flat_cls_factor = torch.cat(flat_cls_factor, dim=2)  # (B, 1, N)
        flat_pred_kernel = torch.cat(flat_pred_kernel, dim=2)  # (B, kernel_depth, N)
        # compute the unified mask representations and the geometry features
        seg_preds, geom_features = self.mask_head(
            fpn_outputs[: len(fpn_outputs) - self.config.extra_FPN_layers]
        )  # mask output, geom features

        if training:
            # return the raw preduictions for training
            return flat_pred_cls, flat_cls_factor, flat_pred_kernel, seg_preds, geom_features
        else:
            # compute the final instance masks, class scores, class labels and normalized mass
            results = compute_masks(
                flat_pred_cls,
                flat_cls_factor,
                flat_pred_kernel,
                seg_preds,
                geom_features,
                cls_threshold=cls_threshold,
                mask_threshold=mask_threshold,
                nms_threshold=nms_threshold,
                kernel_size=1,
                kernel_depth=self.config.mask_output_filters,
                max_detections=max_detections,
                min_area=min_area,
                sigma_nms=self.config.sigma_nms,
                scale_by_mask_confidence=scale_by_mask_scores,
                use_binary_masks=self.config.use_binary_masks,
                nms_mode=nms_mode,
            )

            # return a list of dict (one for each image in the batch)
            # [{"masks": seg_preds, "scores": scores, "cls_labels": cls_labels_pos, "masses": masses}]
            # where seg_preds is a tensor of shape [N, H, W] (N the number of predicted instances)
            # scores, cls_labels_pos and densities are tensors of shape [N]

            return results


class RAMSESHead(nn.Module):
    """RAMSESHead is implementing the shared head for the RAMSES2 model.
    This layer produces three outputs for each level of a Feature Pyramid Network (FPN):
        - Class logits (classification scores)
        - Kernel features (for dynamic convolution)
        - Class factor (auxiliary output)
    Inputs:
        - A 4D tensor of shape [batch, channels, height, width], where the last two channels are reserved for coordinates.
    Arguments:
        ncls (int): Number of output classes.
        filters_in (int): Number of input channels.
        conv_filters (int): Number of filters for intermediate convolutional layers (default: 256).
        kernel_filters (int): Number of output filters for the kernel head (default: 256).
        head_layers (int): Number of convolutional layers in the class and kernel heads (default: 4).
        cls_factor_layers (int): Number of convolutional layers in the class factor head (default: 2).
        activation (Callable or None): Activation function to use.
        name (str): Name for the layer (default: "RAMSES_head").
        normalization (Callable or None): Type of normalization

    Call Arguments:
        inputs (torch.Tensor): Input tensor of shape [B, C, H, W].

    Returns:
        tuple:
            - class_logits (torch.Tensor): Classification logits of shape [B, ncls, H, W, ].
            - class_factor (torch.Tensor): Class factor output of shape [B, 1, H, W].
            - kernel_features (torch.Tensor): Kernel features of shape [B, kernel_filters, H, W].
    """

    def __init__(
        self,
        ncls,
        filters_in,
        conv_filters=256,
        kernel_filters=256,
        head_layers=4,
        cls_factor_layers=2,
        activation=None,
        normalization=None,
    ):
        super().__init__()
        self.ncls = ncls
        self.filters_in = filters_in
        self.conv_filters = conv_filters
        self.kernel_filters = kernel_filters
        self.head_layers = head_layers
        self.cls_factor_layers = cls_factor_layers
        self.activation = activation
        self.normalization = normalization

        # Class head (no coordinates)
        class_head_layers = []
        for i in range(head_layers):
            in_ch = filters_in - 2 if i == 0 else conv_filters
            class_head_layers.append(
                ConvLayer(
                    in_channels=in_ch,
                    out_channels=conv_filters,
                    norm=normalization,
                    activation=activation,
                )
            )
        self.class_head = nn.Sequential(*class_head_layers)
        self.class_logits = ConvLayer(
            in_channels=conv_filters,
            out_channels=ncls,
            kernel_size=3,
            norm=None,
            activation=nn.Sigmoid,
            use_bias=True,
        )

        # Class factor head (no coordinates)
        class_factor_layers = []
        for i in range(cls_factor_layers):
            in_ch = filters_in - 2 if i == 0 else conv_filters
            class_factor_layers.append(
                ConvLayer(
                    in_channels=in_ch,
                    out_channels=conv_filters,
                    norm=normalization,
                    activation=activation,
                )
            )
        self.class_factor_head = nn.Sequential(*class_factor_layers)
        self.class_factor_out = ConvLayer(
            in_channels=conv_filters,
            out_channels=1,
            kernel_size=3,
            norm=None,
            activation=nn.ReLU,
            use_bias=True,
        )

        # Kernel head (with coordinates)
        kernel_head_layers = []
        for i in range(head_layers):
            in_ch = filters_in if i == 0 else conv_filters
            kernel_head_layers.append(
                ConvLayer(
                    in_channels=in_ch,
                    out_channels=conv_filters,
                    norm=normalization,
                    activation=activation,
                )
            )
        self.kernel_head = nn.Sequential(*kernel_head_layers)
        self.kernel_out = ConvLayer(
            in_channels=conv_filters,
            out_channels=kernel_filters,
            kernel_size=3,
            norm=None,
            activation=None,
            use_bias=True,
        )

    def forward(self, x):
        # x: (B, C, H, W), where C = filters_in
        class_head = x[:, :-2, ...]  # remove last 2 channels (coordinates)
        class_factor_head = x[:, :-2, ...]
        kernel_head = x

        class_head = self.class_head(class_head)
        class_logits = self.class_logits(class_head)

        class_factor_head = self.class_factor_head(class_factor_head)
        class_factor = self.class_factor_out(class_factor_head)

        kernel_head = self.kernel_head(kernel_head)
        kernel_features = self.kernel_out(kernel_head)

        return class_logits, class_factor, kernel_features


class RAMSESMaskHead(nn.Module):
    """
    PyTorch implementation of the RAMSES mask head.

    For each FPN level, a sequence of operations (upsample/downsample/conv) is applied to bring the feature map to the output level resolution, adding normalized coordinates at each level.
    The pipelines for each level are declared in __init__ (one nn.Sequential per level).
    The resized features are summed (not stacked) before being passed to the segmentation head (self.mask_out).

    Args:
        num_levels (int): Number of expected FPN levels.
        in_ch (int): Number of input channels for FPN features (before adding coordinates).
        mid_ch (int): Number of intermediate channels for convolutions.
        geom_ch (int): Number of channels for geometric features.
        out_ch (int): Number of output channels for the mask head.
        nconv (int): Number of convolutions for the geometric head.
        output_level (int): Target FPN level for feature fusion.
        upscaling (bool): If True, upsample the final output.
        activation: Activation function to use.
        normalization: Normalization type to use.

    Inputs:
        fpn_features (list[Tensor]): List of tensors (B, C, H, W), one per FPN level.

    Returns:
        seg_outputs (Tensor): Mask feature map, shape (B, out_ch, H, W)
        geom_feats (Tensor): Geometric feature map, shape (B, 1, H, W)
    """

    def __init__(
        self,
        num_levels=4,  # number of FPN levels expected
        in_ch=258,
        mid_ch=128,
        geom_ch=128,
        out_ch=256,
        nconv=2,
        output_level=0,
        upscaling=False,
        activation=None,
        normalization=None,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.geom_ch = geom_ch
        self.out_ch = out_ch
        self.nconv = nconv
        self.output_level = output_level
        self.upscaling = upscaling
        self.activation = activation
        self.normalization = normalization
        self.num_levels = num_levels

        # For each FPN level, create a nn.Sequential containing the full sequence of resizing operations
        self.level_pipelines = nn.ModuleList()
        for level in range(num_levels):
            ops = []
            # Upsampling
            for i in range(max(0, level - output_level)):
                in_channels = in_ch if i == 0 else mid_ch
                ops.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                        ConvLayer(
                            in_channels=in_channels,
                            out_channels=mid_ch,
                            kernel_size=3,
                            norm=normalization,
                            activation=activation,
                        ),
                    )
                )
            # Downsampling
            for i in range(max(0, output_level - level)):
                in_channels = in_ch if i == 0 else mid_ch
                ops.append(
                    ConvLayer(
                        in_channels=in_channels,
                        out_channels=mid_ch,
                        kernel_size=3,
                        stride=2,
                        norm=normalization,
                        activation=activation,
                    )
                )
            # 1x1 conv if not up/down
            if level == output_level:
                ops.append(
                    ConvLayer(
                        in_channels=in_ch,
                        out_channels=mid_ch,
                        kernel_size=1,
                        norm=normalization,
                        activation=activation,
                    )
                )
            self.level_pipelines.append(nn.Sequential(*ops))  # pipeline for this level
        # Geometry feature convs
        self.geom_convs = nn.ModuleList()
        for i in range(nconv):
            in_channels = mid_ch if i == 0 else geom_ch
            self.geom_convs.append(
                ConvLayer(
                    in_channels=in_channels,
                    out_channels=geom_ch,
                    kernel_size=3,
                    norm=normalization,
                    activation=activation,
                )
            )
        self.geom_final = ConvLayer(
            in_channels=geom_ch,
            out_channels=1,
            kernel_size=1,
            norm=None,
            activation=nn.ReLU,
            use_bias=True,
        )
        self.mask_out = ConvLayer(
            in_channels=mid_ch,
            out_channels=out_ch,
            kernel_size=1,
            norm=normalization,
            activation=activation,
            use_bias=True,
        )

    def add_coords(self, x):
        # x: (B, C, H, W)
        B, _, H, W = x.shape
        device = x.device
        y_coords = torch.linspace(-1, 1, steps=H, device=device)
        x_coords = torch.linspace(-1, 1, steps=W, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)  # (B, 2, H, W)
        return torch.cat([x, coords], dim=1)

    def forward(self, fpn_features):
        # fpn_features: list of (B, C, H, W), length == num_levels
        seg_outputs = None
        for level, (feature, pipeline) in enumerate(zip(fpn_features, self.level_pipelines)):
            x = self.add_coords(feature)
            x = pipeline(x)
            if seg_outputs is None:
                seg_outputs = x
            else:
                seg_outputs = seg_outputs + x
        if self.upscaling:
            seg_outputs = nn.functional.interpolate(seg_outputs, scale_factor=2.0, mode="bilinear", align_corners=False)
        # Geometry features
        geom_feats = seg_outputs
        for conv in self.geom_convs:
            geom_feats = conv(geom_feats)
        geom_feats = self.geom_final(geom_feats)
        # Mask output
        seg_outputs = self.mask_out(seg_outputs)
        return seg_outputs, geom_feats


def matrix_nms(
    cls_labels,
    scores,
    cls_factors,
    seg_preds,
    binary_masks,
    mask_sum,
    sigma=0.5,
    pre_nms_k=1536,
    post_nms_k=768,
    score_threshold=0.5,
    mode="gaussian",
):
    """
    PyTorch implementation of Matrix NMS as defined in SOLOv2 paper.
    Args:
        cls_labels (Tensor): [N] class labels
        scores (Tensor): [N] scores for each instance
        cls_factors (Tensor): [N] class factors
        seg_preds (Tensor): [N, H, W] predicted masks for each instance
        binary_masks (Tensor): [N, H, W] binary masks for each instance
        mask_sum (Tensor): [N] area of each instance mask
    Returns:
        seg_preds, scores, cls_labels, cls_factors (all filtered)
    """
    N = scores.shape[0]
    device = scores.device
    # Select only first pre_nms_k instances (sorted by scores)
    if N > pre_nms_k:
        num_selected = pre_nms_k
        post_nms_k = min(post_nms_k, num_selected)
        indices = torch.argsort(scores, descending=True)[:num_selected]
        seg_preds = seg_preds[indices]
        binary_masks = binary_masks[indices]
        cls_labels = cls_labels[indices]
        scores = scores[indices]
        cls_factors = cls_factors[indices]
        mask_sum = mask_sum[indices]
        N = num_selected
    else:
        num_selected = N
        post_nms_k = min(post_nms_k, N)

    # calculate iou between different masks
    binary_masks_flat = binary_masks.view(N, -1).float()  # [N, H*W]
    intersection = torch.matmul(binary_masks_flat, binary_masks_flat.T)  # [N, N]
    mask_sum_tile = mask_sum.view(1, -1) + mask_sum.view(
        1, -1
    )  # instead of mask_sum.view(1, -1).expand(N,N), it works because of broadcasting!
    union = mask_sum_tile - intersection  # + mask_sum_tile.t() - intersection
    iou = intersection / (union + 1e-6)
    iou = torch.triu(iou, diagonal=1)  # upper triangular, zero diagonal

    # iou decay and compensation - Only compare instances with the same class
    labels_match = (cls_labels.view(1, -1) == cls_labels.view(-1, 1)).float()
    labels_match = torch.triu(labels_match, diagonal=1)
    decay_iou = iou * labels_match
    compensate_iou, _ = torch.max(decay_iou, dim=0)
    compensate_iou = compensate_iou.unsqueeze(0).expand(N, N).T
    # matrix nms
    if mode == "gaussian":
        inv_sigma = 1.0 / sigma
        decay_coefficient = torch.exp(-inv_sigma * (decay_iou**2 - compensate_iou**2))
        decay_coefficient, _ = torch.min(decay_coefficient, dim=0)
    else:
        decay_coefficient = (1 - decay_iou) / (1 - compensate_iou + 1e-6)
        decay_coefficient, _ = torch.min(decay_coefficient, dim=0)
    decayed_scores = scores * decay_coefficient
    keep = decayed_scores >= score_threshold
    scores = scores[keep]
    seg_preds = seg_preds[keep]
    cls_labels = cls_labels[keep]
    cls_factors = cls_factors[keep]
    # Keep the post_nms_k first predictions
    if scores.shape[0] > post_nms_k:
        sorted_indices = torch.argsort(scores, descending=True)[:post_nms_k]
        scores = scores[sorted_indices]
        seg_preds = seg_preds[sorted_indices]
        cls_labels = cls_labels[sorted_indices]
        cls_factors = cls_factors[sorted_indices]
    return seg_preds, scores, cls_labels, cls_factors


def mask_nms(
    cls_labels,
    scores,
    seg_preds,
    binary_masks,
    cls_factors,
    iou_threshold=0.5,
    post_nms_k=768,
):
    """
    Greedy NMS pour masques binaires.

    Args:
        cls_labels (Tensor): [N] étiquettes de classe
        scores (Tensor): [N] scores
        seg_preds (Tensor): [N, H, W] masques prédits
        binary_masks (Tensor): [N, H, W] masques binaires
        iou_threshold (float): seuil d'IoU pour suppression
        post_nms_k (int): nombre max de masques à garder

    Returns:
        seg_preds, scores, cls_labels, cls_factors (filtrés après NMS)
    """
    N = scores.shape[0]
    if N == 0:
        return seg_preds, scores, cls_labels

    # Tri par score décroissant
    indices = torch.argsort(scores, descending=True)
    scores = scores[indices]
    seg_preds = seg_preds[indices]
    binary_masks = binary_masks[indices]
    cls_labels = cls_labels[indices]
    cls_factors = cls_factors[indices]

    keep = []
    suppressed = torch.zeros(N, dtype=torch.bool, device=scores.device)

    binary_masks_flat = binary_masks.view(N, -1).float()
    mask_area = binary_masks_flat.sum(dim=1)

    for i in range(N):
        if suppressed[i]:
            continue
        keep.append(i)
        mask_i = binary_masks_flat[i]
        label_i = cls_labels[i]

        # Comparer à toutes les autres instances restantes
        rest = torch.arange(i + 1, N, device=scores.device)
        mask_rest = binary_masks_flat[rest]
        label_rest = cls_labels[rest]

        # Seulement mêmes classes
        same_class = label_rest == label_i
        if same_class.sum() == 0:
            continue

        inter = (mask_i * mask_rest[same_class]).sum(dim=1)
        area_i = mask_area[i]
        area_rest = mask_area[rest][same_class]
        union = area_i + area_rest - inter
        iou = inter / (union + 1e-6)

        # Marque comme supprimé si IoU > seuil
        suppressed_idx = rest[same_class][iou > iou_threshold]
        suppressed[suppressed_idx] = True

    keep = torch.tensor(keep, device=scores.device)
    if keep.numel() > post_nms_k:
        keep = keep[:post_nms_k]

    return seg_preds[keep], scores[keep], cls_labels[keep], cls_factors[keep]


def soft_mask_nms(
    cls_labels,
    scores,
    seg_preds,
    binary_masks,
    cls_factors,
    method="gaussian",  # ou "linear"
    sigma=0.5,
    iou_threshold=0.5,
    min_score=0.001,
    post_nms_k=768,
):
    """
    Soft-NMS pour des masques binaires.

    Args:
        cls_labels (Tensor): [N] étiquettes de classe
        scores (Tensor): [N] scores
        seg_preds (Tensor): [N, H, W] masques prédits
        binary_masks (Tensor): [N, H, W] masques binaires
        method (str): 'linear' ou 'gaussian' pour la décroissance
        sigma (float): paramètre du mode 'gaussian'
        iou_threshold (float): seuil utilisé en mode 'linear'
        min_score (float): score minimal en dessous duquel on ignore l'instance
        post_nms_k (int): nombre maximal d'instances à conserver

    Returns:
        seg_preds, scores, cls_labels, cls_factors (après Soft-NMS)
    """
    N = scores.shape[0]
    if N == 0:
        return seg_preds, scores, cls_labels

    # Tri initial par score
    indices = torch.argsort(scores, descending=True)
    scores = scores[indices]
    seg_preds = seg_preds[indices]
    binary_masks = binary_masks[indices]
    cls_labels = cls_labels[indices]
    cls_factors = cls_factors[indices]

    binary_masks_flat = binary_masks.view(N, -1).float()
    mask_area = binary_masks_flat.sum(dim=1)

    keep = []

    for i in range(N):
        if scores[i] < min_score:
            continue

        # Conserver l'instance i
        keep.append(i)
        mask_i = binary_masks_flat[i]
        area_i = mask_area[i]
        label_i = cls_labels[i]

        for j in range(i + 1, N):
            if cls_labels[j] != label_i or scores[j] < min_score:
                continue

            mask_j = binary_masks_flat[j]
            area_j = mask_area[j]

            inter = (mask_i * mask_j).sum()
            union = area_i + area_j - inter
            iou = inter / (union + 1e-6)

            # Appliquer la décroissance
            if method == "linear":
                if iou > iou_threshold:
                    decay = 1.0 - iou
                else:
                    decay = 1.0
            elif method == "gaussian":
                decay = torch.exp(-(iou**2) / sigma)
            else:
                raise ValueError("Méthode inconnue pour Soft-NMS : {}".format(method))

            scores[j] *= decay

    keep = torch.tensor(keep, device=scores.device)
    if keep.numel() > post_nms_k:
        keep = keep[:post_nms_k]

    return seg_preds[keep], scores[keep], cls_labels[keep], cls_factors[keep]


def compute_masks(
    flat_pred_cls,  # [B, ncls, N]
    flat_cls_factor_pred,  # [B, 1, N]
    flat_pred_kernel,  # [B, kernel_depth*kernel_size**2, N]
    masks_head_output,  # [B, mask_ch, H, W]
    geom_feats,  # [B, 1, H, W]
    cls_threshold=0.5,
    mask_threshold=0.5,
    nms_threshold=0.5,
    kernel_size=1,
    kernel_depth=256,
    max_detections=768,
    min_area=0,
    sigma_nms=0.5,
    scale_by_mask_confidence=True,
    use_binary_masks=False,
    nms_mode="greedy",
):
    """
    Compute instance masks, scores, class labels, and densities for each image in a batch using PyTorch tensors.

    This function takes the flattened predictions from the RAMSES head and mask head, applies thresholding, dynamic convolution,
    mask scoring, area filtering, and Matrix NMS to produce the final instance segmentation outputs for each image in the batch.

    Args:
        flat_pred_cls (Tensor): Class predictions, shape [B, ncls, N], where B is batch size, ncls is number of classes, N is number of locations.
        flat_cls_factor_pred (Tensor): Class factor predictions, shape [B, 1, N].
        flat_pred_kernel (Tensor): Kernel predictions, shape [B, kernel_depth*kernel_size**2, N].
        masks_head_output (Tensor): Mask head output features, shape [B, mask_ch, H, W].
        geom_feats (Tensor): Geometric features, shape [B, 1, H, W].
        cls_threshold (float): Threshold for class score to consider a location as a positive instance.
        mask_threshold (float): Threshold for mask binarization.
        nms_threshold (float): Score threshold for Matrix NMS post-processing.
        kernel_size (int): Size of the dynamic convolution kernel.
        kernel_depth (int): Number of channels for the dynamic kernel.
        max_detections (int): Maximum number of instances to keep after NMS.
        min_area (int): Minimum area (in pixels) for a mask to be kept.
        sigma_nms (float): Sigma parameter for Matrix NMS.
        scale_by_mask_confidence (bool): Whether to scale class scores by mask confidence.
        use_binary_masks (bool): Whether to use binary masks for density computation.

    Returns:
        results (list of dict): For each image in the batch, a dict with keys:
            - 'masks': Tensor of instance masks, shape [N_inst, H, W]
            - 'scores': Tensor of instance scores, shape [N_inst]
            - 'cls_labels': Tensor of class labels, shape [N_inst]
            - 'densities': Tensor of instance densities, shape [N_inst]
    """
    B, ncls, N = flat_pred_cls.shape
    device = flat_pred_cls.device
    results = []
    for b in range(B):
        # [ncls, N] -> [N, ncls]
        pred_cls = flat_pred_cls[b].transpose(0, 1)  # [N, ncls]
        pred_cls_factor = flat_cls_factor_pred[b].squeeze(0)  # [N]
        kernel_preds = flat_pred_kernel[b].transpose(0, 1)  # [N, kernel_depth*kernel_size**2]
        # Only one prediction by pixel
        cls_labels = torch.argmax(pred_cls, dim=1)  # [N]
        cls_scores, _ = torch.max(pred_cls, dim=1)  # [N]
        positive_idx = torch.where(cls_scores >= cls_threshold)[0]
        if positive_idx.numel() == 0:
            # No positive, return empty tensors
            Hmask, Wmask = masks_head_output.shape[2:]
            results.append(
                {
                    "masks": torch.empty((0, Hmask, Wmask), device=device),
                    "scores": torch.empty((0,), device=device),
                    "cls_labels": torch.empty((0,), dtype=torch.long, device=device),
                    "masses": torch.empty((0,), device=device),
                },
            )
            continue
        cls_scores_pos = cls_scores[positive_idx]  # [P]
        cls_labels_pos = cls_labels[positive_idx]  # [P]
        cls_factors_pos = pred_cls_factor[positive_idx]  # [P]
        kernel_preds = kernel_preds[positive_idx]  # [P, kernel_depth]
        npos = positive_idx.numel()  # Number of positive predictions
        if npos == 1:  # if npos=1, we must make unsqueeze the tensor to keep the leading dim
            kernel_preds = kernel_preds.unsqueeze(0)
        # kernel_preds: [P, kernel_depth] -> [P, kernel_depth, ks, ks]
        kernel_preds = kernel_preds.reshape(npos, kernel_depth, kernel_size, kernel_size)
        # masks_head_output: [B, mask_ch, H, W] (mask_ch == kernel_depth)
        mask_feat = masks_head_output[b]  # [mask_ch, H, W]
        # Dynamic conv2d: [mask_ch, H, W] * [P, kernel_depth, ks, ks]
        assert mask_feat.shape[0] == kernel_depth, "Mask feature channels must match kernel depth"
        seg_preds = torch.nn.functional.conv2d(mask_feat, kernel_preds, stride=1, padding="same")
        seg_preds = torch.sigmoid(seg_preds)
        if npos > 1:
            seg_preds = seg_preds.squeeze(0)  # [P, H, W]
        binary_masks = (seg_preds >= mask_threshold).float()  # [P, H, W]
        mask_sum = binary_masks.view(binary_masks.size(0), -1).sum(dim=1)  # [P]
        # scale the category score by mask confidence
        if scale_by_mask_confidence:
            mask_scores = (seg_preds * binary_masks).view(seg_preds.size(0), -1).sum(dim=1) / (mask_sum + 1e-6)  # [P]
            scores = cls_scores_pos * mask_scores  # [P]
        else:
            scores = cls_scores_pos
        # Filter by min_area
        if min_area > 0:
            keep = mask_sum > min_area
            if keep.sum() == 0:
                results.append(
                    {
                        "masks": torch.empty((0, *seg_preds.shape[1:]), device=device),
                        "scores": torch.empty((0,), device=device),
                        "cls_labels": torch.empty((0,), dtype=torch.long, device=device),
                        "masses": torch.empty((0,), device=device),
                    },
                )
                continue

            cls_labels_pos = cls_labels_pos[keep]
            scores = scores[keep]
            cls_factors_pos = cls_factors_pos[keep]
            seg_preds = seg_preds[keep]
            binary_masks = binary_masks[keep]
            mask_sum = mask_sum[keep]

        if nms_mode == "greedy":
            seg_preds, scores, cls_labels_pos, cls_factors_pos = mask_nms(
                cls_labels_pos,
                scores,
                seg_preds,
                binary_masks,
                cls_factors_pos,
                iou_threshold=nms_threshold,
                post_nms_k=max_detections,
            )
        elif nms_mode == "soft":
            seg_preds, scores, cls_labels_pos, cls_factors_pos = soft_mask_nms(
                cls_labels_pos,
                scores,
                seg_preds,
                binary_masks,
                cls_factors_pos,
                method="gaussian",  # ou "linear"
                sigma=sigma_nms,
                iou_threshold=nms_threshold,
                min_score=cls_threshold,
                post_nms_k=max_detections,
            )
        else:
            # Matrix NMS
            seg_preds, scores, cls_labels_pos, cls_factors_pos = matrix_nms(
                cls_labels_pos,
                scores,
                cls_factors_pos,
                seg_preds,
                binary_masks,
                mask_sum,
                post_nms_k=max_detections,
                score_threshold=nms_threshold,
                sigma=sigma_nms,
            )
        # Mass: each masks (one mask per slice in axis 0) is multiplied by the geometry features
        # Then we sum each slice and multiply by the corresponding class factor to get the predicted normalized mass
        geom = geom_feats[b]  # [1, H, W]

        if seg_preds.shape[0] > 0:
            if use_binary_masks:
                binary_masks = (seg_preds >= mask_threshold).float()
                masses = (binary_masks * geom).view(seg_preds.size(0), -1).sum(dim=1)
            else:
                masses = (seg_preds * geom).view(seg_preds.size(0), -1).sum(dim=1)
            masses = masses * cls_factors_pos
        else:
            masses = torch.empty((0,), device=device)
        results.append({"masks": seg_preds, "scores": scores, "cls_labels": cls_labels_pos, "masses": masses})

    return results
