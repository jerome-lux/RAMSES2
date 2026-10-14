from typing import Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from ..core import norm


def get_class(name: str, extra_modules=None):
    """
    Look up a class by name in torch.nn, then in the custom core ``norm`` module,
    then in any additional provided modules.

    Args:
        name: Class name (e.g., "GroupNorm", "LayerNorm2d", "RMSNorm2d").
        extra_modules: Optional list of additional modules to search after
            torch.nn and the core ``norm`` module.

    Returns:
        type: The matching Python class.

    Raises:
        ValueError: If the class is not found in any module.
    """
    if extra_modules is None:
        extra_modules = []

    # torch.nn first, then the custom core ``norm`` module, then any extras.
    for module in (nn, norm, *extra_modules):
        if hasattr(module, name):
            return getattr(module, name)

    raise ValueError(f"Class '{name}' not found in torch.nn, core.norm, or the provided modules {extra_modules}.")


def make_tuple(value, n, fill_value=None):
    """
    Returns a tuple of length n.
    If value is a sequence (excluding str), pads with fill_value if needed.
    Otherwise, returns (value,) * n.
    """
    if isinstance(value, Sequence) and not isinstance(value, str):
        value = list(value)
        if len(value) < n:
            value = value + [fill_value] * (n - len(value))
        return tuple(value[:n])
    else:
        return tuple([value] * n)


def point_nms(x, kernel_size=2):
    """
    Apply local non-maximum suppression to a dense score map.

    Args:
        x: Dense score map of shape (N, C, H, W) or (1, 1, H, W).
        kernel_size: Size of the local window for max-pooling (default: 2).

    Returns:
        Tensor: Score map with non-local-maximum points suppressed to zero.
    """
    # Local max-pooling to detect maxima
    max_score = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    # Keep only local maxima
    keep = (x == max_score).float()

    return x * keep


def get_same_padding(kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]]):
    """
    Compute the (left, right, top, bottom) padding for a "same" convolution,
    i.e., output spatial size equals ceil(input_size / stride).

    Args:
        kernel_size: Convolution kernel size (int or (h, w) tuple).
        stride: Convolution stride (int or (h, w) tuple).

    Returns:
        Tuple[int, int, int, int]: (padding_left, padding_right, padding_top, padding_bottom).
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    # Total padding per dimension = kernel_size - 1, split as evenly as possible.
    pad_h = kernel_size[0] - 1
    pad_w = kernel_size[1] - 1

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return (pad_left, pad_right, pad_top, pad_bottom)


def pad_with_coords(data: torch.Tensor) -> torch.Tensor:
    """
    Append normalized coordinate channels (x, y in [-1, 1]) to a feature map.

    Args:
        data: Input tensor of shape (B, C, H, W).

    Returns:
        Tensor of shape (B, C+2, H, W) with the last two channels being
        the x and y spatial coordinates normalized to [-1, 1].
    """
    # data: (batch_size, channels, height, width)
    batch_size, channels, height, width = data.shape
    device = data.device
    dtype = data.dtype

    x = torch.linspace(-1, 1, steps=width, device=device, dtype=dtype)
    x = x.unsqueeze(0).unsqueeze(2).expand(batch_size, width, height).permute(0, 2, 1)
    x = x.unsqueeze(1)  # (batch_size, 1, height, width)

    y = torch.linspace(-1, 1, steps=height, device=device, dtype=dtype)
    y = y.unsqueeze(0).unsqueeze(2).expand(batch_size, height, width)
    y = y.unsqueeze(1)  # (batch_size, 1, height, width)

    data = torch.cat([data, x, y], dim=1)
    return data


def decode_predictions(seg_preds, scores, threshold=0.5, by_mask_scores=True):
    """
    Compute the labeled mask array from segmentation predictions.
    If two masks overlap, the one with either the higher score or the higher seg value is chosen.
    Returns labeled array.

    Args:
        seg_preds: (N, H, W) torch.Tensor, one predicted mask per instance (sigmoid activation).
        scores: (N,) torch.Tensor, score of each predicted instance.
        threshold: float, threshold to compute binary masks.
        by_mask_scores: bool, if True, rank masks by score; otherwise rank each pixel by its seg_pred value.

    Returns:
        labeled_masks: (H, W) torch.LongTensor, labeled mask image
    """
    N, H, W = seg_preds.shape

    if by_mask_scores:
        binary_masks = (seg_preds >= threshold).long()  # (N, H, W)
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        sorted_masks = binary_masks[sorted_indices]  # (N, H, W)
        # Add background slice
        bg_slice = torch.zeros((1, H, W), dtype=torch.long, device=seg_preds.device)
        labeled_masks = torch.cat([bg_slice, sorted_masks], dim=0)  # (N+1, H, W)
        # Take argmax (mask with higher score wins when overlap)
        labeled_masks = torch.argmax(labeled_masks, dim=0)  # (H, W)
    else:
        filt_seg = torch.where(seg_preds >= threshold, seg_preds, torch.zeros_like(seg_preds))
        bg_slice = torch.zeros((1, H, W), dtype=seg_preds.dtype, device=seg_preds.device)
        labeled_masks = torch.cat([bg_slice, filt_seg], dim=0)  # (N+1, H, W)
        labeled_masks = torch.argmax(labeled_masks, dim=0)  # (H, W)

    return labeled_masks


def compute_cls_targets(
    classes,
    masks,
    labels,
    densities,
    shape,
    strides=[4, 8, 16, 32, 64],
    grid_sizes=[64, 36, 24, 16, 12],
    scale_ranges=[[1, 96], [48, 192], [96, 384], [192, 768], [384, 2048]],
    mode="diag",
    offset_factor=0.5,
):
    """
    Generates classification, label, and density targets for multi-scale object detection grids.
    Args:
        classes: Tensor [N]
        masks: Tensor [H, W] with labeled object (0: bg)
        densities: Tensor [N]
        shape: Shape of the input image (height, width).
        strides (list, optional): List of stride values for each feature level. Default is [4, 8, 16, 32, 64].
        grid_sizes (list, optional): List of grid sizes for each feature level. Default is [64, 36, 24, 16, 12].
        scale_ranges (list, optional): List of [min, max] object scale ranges for each level. Default is [[1, 96], [48, 192], [96, 384], [192, 768], [384, 2048]].
        mode (str, optional): Mode for object scale calculation ('min', 'max', or 'diag'). Default is "diag".
        offset_factor (float, optional): Factor to adjust the bounding box region for target assignment. Default is 0.5.
    Returns:
        tuple: (class_targets, label_targets, density_targets), each as a concatenated tensor for all levels.
    """

    offset_factor = offset_factor * 0.5
    nx, ny = map(float, shape)
    maxdim = max(nx, ny)
    device = classes.device

    # relabel masks and filter other tensors to ensure labels are consecutive and to discard missing labels
    masks, labels, filtered_tensors = relabel_and_filter(masks, labels, classes, densities)
    classes, densities = filtered_tensors
    boxes, _ = boxes_from_masks(masks)

    dx = boxes[:, 2] * (nx - 1)
    dy = boxes[:, 3] * (ny - 1)
    dmin = torch.minimum(dx, dy)

    if mode == "min":
        object_scale = dmin
    elif mode == "max":
        object_scale = torch.maximum(dx, dy)
    else:
        object_scale = torch.sqrt(dx * dy)

    idx_per_lvl = []
    bboxes_per_lvl = []
    labels_per_lvl = []
    cls_per_lvl = []
    densities_by_lvl = []

    for lvl, (minsize, maxsize) in enumerate(scale_ranges):
        if lvl == 0:
            filtered_idx = torch.where(object_scale <= maxsize)[0]
        elif lvl + 1 < len(grid_sizes) - 1:
            cond1 = (object_scale >= minsize) & (object_scale <= maxsize) & (dmin >= strides[lvl])
            cond2 = (object_scale > maxsize) & (dmin < strides[lvl + 1]) & (dmin >= strides[lvl])
            filtered_idx = torch.where(cond1 | cond2)[0]
        else:
            filtered_idx = torch.where(object_scale >= minsize)[0]

        filtered_idx = filtered_idx.to(device)
        idx_per_lvl.append(filtered_idx)
        bboxes_per_lvl.append(boxes[filtered_idx])
        labels_per_lvl.append(labels[filtered_idx])
        cls_per_lvl.append(classes[filtered_idx].to(torch.long))
        densities_by_lvl.append(densities[filtered_idx].float())

    class_targets = []
    label_targets = []
    density_targets = []

    for lvl, gridsize in enumerate(grid_sizes):
        lvl_imshape = (int(round(gridsize * nx / maxdim)), int(round(gridsize * ny / maxdim)))
        # print(lvl_imshape)

        cls_img = torch.zeros(lvl_imshape, dtype=torch.long, device=device)
        labels_img = torch.zeros(lvl_imshape, dtype=torch.long, device=device)
        densities_img = torch.zeros(lvl_imshape, dtype=torch.float32, device=device)

        if cls_per_lvl[lvl].numel() > 0:
            bboxes = bboxes_per_lvl[lvl]
            areas = bboxes[:, 2] * bboxes[:, 3]
            order = torch.argsort(areas, descending=True)

            ordered_labels = labels_per_lvl[lvl][order].to(torch.long)
            ordered_cls = cls_per_lvl[lvl][order].to(torch.long)
            ordered_densities = densities_by_lvl[lvl][order]

            lvl_nx, lvl_ny = map(float, lvl_imshape)
            locations_lvl = compute_locations(1, lvl_imshape).float().to(device)  # Version PyTorch

            for i, _ in enumerate(ordered_cls):
                lab = ordered_labels[i]
                cx, cy, dx, dy = boxes[lab - 1]

                left = (lvl_nx - 1) * (cx - dx * offset_factor)
                right = (lvl_nx - 1) * (cx + dx * offset_factor)
                top = (lvl_ny - 1) * (cy - dy * offset_factor)
                bottom = (lvl_ny - 1) * (cy + dy * offset_factor)

                inside_mask = (
                    (locations_lvl[:, 0] >= left)
                    & (locations_lvl[:, 0] <= right)
                    & (locations_lvl[:, 1] >= top)
                    & (locations_lvl[:, 1] <= bottom)
                )
                inside_indices = torch.where(inside_mask)[0]

                # Add the box center if no grid point falls inside
                cx_pix = min(int(round((lvl_nx - 1) * cx.item())), lvl_imshape[0] - 1)
                cy_pix = min(int(round((lvl_ny - 1) * cy.item())), lvl_imshape[1] - 1)
                center_indices = torch.where(
                    (torch.round(locations_lvl[:, 0]).to(torch.long) == cx_pix)
                    & (torch.round(locations_lvl[:, 1]).to(torch.long) == cy_pix)
                )[0]

                if center_indices.numel() > 0:
                    inside_indices = torch.cat([center_indices, inside_indices])

                coords = locations_lvl[inside_indices].to(torch.long)
                coords = coords.clamp(torch.zeros(2, device=device), torch.tensor(lvl_imshape, device=device) - 1)
                coords = coords.to(torch.long)

                if coords.numel() > 0:
                    cls_img[coords[:, 0], coords[:, 1]] = ordered_cls[i]
                    labels_img[coords[:, 0], coords[:, 1]] = ordered_labels[i]
                    densities_img[coords[:, 0], coords[:, 1]] = ordered_densities[i]

        class_targets.append(cls_img.reshape(-1))
        label_targets.append(labels_img.reshape(-1))
        density_targets.append(densities_img.reshape(-1))

    class_targets = torch.cat(class_targets, dim=0)
    label_targets = torch.cat(label_targets, dim=0)
    density_targets = torch.cat(density_targets, dim=0)

    return class_targets, label_targets, density_targets


def boxes_from_masks(masks: torch.Tensor):
    """
    Compute bounding boxes from a labeled mask image.

    Args:
        masks: Labeled mask tensor of shape (H, W), where 0 is background
            and values >= 1 are instance labels.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - boxes: (N, 4) tensor of [cx, cy, dx, dy] normalized to [0, 1].
            - unique_labels: (N,) tensor of the non-zero labels present in the mask.
    """
    device = masks.device
    unique_labels = torch.unique(masks)
    unique_labels = unique_labels[unique_labels != 0]
    unique_labels = unique_labels.sort().values
    nx = masks.shape[0] - 1.0
    ny = masks.shape[1] - 1.0
    boxes = []
    for label in unique_labels:
        coords = (masks == label).nonzero(as_tuple=False)
        if coords.numel() == 0:
            continue
        cx = coords[:, 0].float().mean()
        cy = coords[:, 1].float().mean()
        dx = coords[:, 0].max() + 1 - coords[:, 0].min()
        dy = coords[:, 1].max() + 1 - coords[:, 1].min()
        dx = dx / nx
        dy = dy / ny
        cx = cx / nx
        cy = cy / ny
        boxes.append(torch.tensor([cx, cy, dx, dy], device=device))
    if boxes:
        boxes = torch.stack(boxes, dim=0)
    else:
        boxes = torch.zeros((0, 4), device=device)
    return boxes, unique_labels





def compute_locations(stride: int, shape: Tuple[int, int], shift: str = "r") -> torch.Tensor:
    """
    Compute a grid of anchor point locations for a given feature map size and stride.

    Args:
        stride: Spatial stride (sampling step) between anchor points.
        shape: Feature map spatial dimensions (height, width).
        shift: Grid alignment — "r"/"right" centers points at stride//2,
            "l"/"left" shifts them to stride//2 - 1.

    Returns:
        Tensor of shape (num_points, 2) with (x, y) coordinates for each location.
    """
    if shift.lower() in ["r", "right"]:
        begin = stride // 2
    else:
        begin = stride // 2 - 1
    xc = torch.arange(begin, shape[0], stride, dtype=torch.long)
    yc = torch.arange(begin, shape[1], stride, dtype=torch.long)
    xc, yc = torch.meshgrid(xc, yc, indexing="ij")
    xc = xc.reshape(-1)
    yc = yc.reshape(-1)
    locations = torch.stack([xc, yc], -1)
    return locations


def normalize_bboxes(bboxes: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """
    Normalize bounding box coordinates from pixel space to [0, 1].

    Args:
        bboxes: Array of shape (..., 4) with [x_min, y_min, x_max, y_max] in pixels.
        nx: Image width in pixels.
        ny: Image height in pixels.

    Returns:
        np.ndarray of shape (..., 4) with normalized coordinates in [0, 1].
    """
    normalized_bboxes = bboxes.astype(np.float32)
    normalized_bboxes[..., 0] /= nx - 1
    normalized_bboxes[..., 1] /= ny - 1
    normalized_bboxes[..., 2] /= nx - 1
    normalized_bboxes[..., 3] /= ny - 1
    return normalized_bboxes


def denormalize_bboxes(norm_bboxes: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """
    Convert normalized bounding box coordinates back to pixel space.

    Args:
        norm_bboxes: Array of shape (..., 4) with normalized [x_min, y_min, x_max, y_max] in [0, 1].
        nx: Image width in pixels.
        ny: Image height in pixels.

    Returns:
        np.ndarray of shape (..., 4) with integer pixel coordinates, clamped to [0, nx/ny].
    """
    bboxes = np.zeros(norm_bboxes.shape).astype(np.int32)
    bboxes[..., 0] = np.maximum(np.around(norm_bboxes[..., 0] * (nx - 1)), 0)
    bboxes[..., 1] = np.maximum(np.around(norm_bboxes[..., 1] * (ny - 1)), 0)
    bboxes[..., 2] = np.minimum(np.around(norm_bboxes[..., 2] * (nx - 1)), nx)
    bboxes[..., 3] = np.minimum(np.around(norm_bboxes[..., 3] * (ny - 1)), ny)
    return bboxes


def crop_to_aspect_ratio(target_shape, image):
    """Crop an image so that its aspect ratio matches target_shape.
    Note: this does NOT resize the image to target_shape."""

    ratio = target_shape[0] / target_shape[1]

    nx, ny = image.shape[:2]

    target_nx = min(int(np.around(ny * ratio)), nx)
    target_ny = int(np.around(target_nx / ratio))

    cropx = nx - target_nx
    cropy = ny - target_ny
    cx = cropx // 2
    cy = cropy // 2
    rx = np.abs(cropx) % 2
    ry = np.abs(cropy) % 2

    if image.ndim not in (2, 3, 4):
        raise ValueError(f"crop_to_aspect_ratio only supports 2D, 3D, or 4D images (got ndim={image.ndim})")

    if cropx > 0:
        if image.ndim == 4:
            image = image[:, cx : -rx - cx, :]
        elif image.ndim == 3:
            image = image[cx : -rx - cx, :]
        elif image.ndim == 2:
            image = image[cx : -rx - cx]

    if cropy > 0:
        if image.ndim == 4:
            image = image[:, :, cy : -cy - ry, :]
        elif image.ndim == 3:
            image = image[:, cy : -cy - ry, :]
        elif image.ndim == 2:
            image = image[:, cy : -cy - ry]

    return image, ((cx, rx + cx), (cy, cy + ry))


def pad_to_aspect_ratio(target_shape, image):
    """Pad an image so that its aspect ratio is the same as target_shape
    It DOES NOT resize the image to target_shape!"""

    target_ratio = target_shape[0] / target_shape[1]

    nx, ny = image.shape[:2]

    target_nx = np.around(ny * target_ratio)
    target_ny = np.around(target_nx / target_ratio)

    if target_nx < nx:
        target_ny = target_ny * nx / target_nx
        target_nx = nx

    if target_ny < ny:
        target_nx = target_nx * ny / target_ny
        target_ny = ny

    target_nx = int(np.around(target_nx))
    target_ny = int(np.around(target_ny))

    padx = target_nx - nx
    pady = target_ny - ny
    px = padx // 2
    py = pady // 2
    rx = np.abs(padx) % 2
    ry = np.abs(pady) % 2

    if padx > 0 or pady > 0:
        if image.ndim == 4:
            image = np.pad(image, ((0, 0), (px, px + rx), (py, py + ry), (0, 0)))
        elif image.ndim == 3:
            image = np.pad(image, ((px, px + rx), (py, py + ry), (0, 0)))
        elif image.ndim == 2:
            image = np.pad(image, ((px, px + rx), (py, py + ry)))
        else:
            print("pad_to_aspect_ratio only support gray or RGB 2D images")

    return image, ((px, px + rx), (py, py + ry))





def relabel_and_filter(masks: torch.Tensor, original_labels: torch.Tensor, *tensors_to_filter):
    """
    Relabel a mask image with consecutive integers (1..N) and filter associated tensors.

    Uses a look-up table (LUT) for O(1) per-pixel relabeling.

    Args:
        masks: Labeled mask tensor (0 = background, >=1 = instance labels).
        original_labels: 1D tensor of original labels corresponding to each instance.
        *tensors_to_filter: Additional 1D tensors (e.g., scores, masses) to filter
            according to the new label ordering.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
            - relabeled_image: Masks with consecutive labels 1..N (background remains 0).
            - new_labels: (N,) tensor of the new consecutive labels.
            - filtered_tensors: List of filtered input tensors reordered to match new labels.

    Raises:
        ValueError: If a label present in masks is not found in original_labels.
    """
    device = masks.device

    # 1. Extract unique non-zero labels present in the mask
    unique_labels = torch.unique(masks)
    unique_labels = unique_labels[unique_labels != 0]

    if unique_labels.numel() == 0:
        return masks, torch.tensor([], device=device), [t[:0] for t in tensors_to_filter]

    # 2. Find the index mapping to filter associated tensors (scores, masses, etc.)
    sorted_orig, sorted_idx = torch.sort(original_labels)
    pos = torch.searchsorted(sorted_orig, unique_labels)

    # Validate: all labels in the image must exist in original_labels
    pos_clipped = torch.clamp(pos, max=sorted_orig.numel() - 1)
    mask_valid = (pos < sorted_orig.numel()) & (sorted_orig[pos_clipped] == unique_labels)
    if not torch.all(mask_valid):
        missing_label = unique_labels[~mask_valid][0].item()
        raise ValueError(f"Label {missing_label} present in masks but missing from original_labels")

    mapping_indices = sorted_idx[pos]

    # 3. Relabel via LUT
    max_label = int(unique_labels.max().item())
    lut = torch.zeros(max_label + 1, dtype=masks.dtype, device=device)
    new_labels = torch.arange(1, len(unique_labels) + 1, device=device, dtype=masks.dtype)
    lut[unique_labels.long()] = new_labels
    relabeled_image = lut[masks.long()]

    # 4. Filter additional tensors
    filtered_tensors = [t[mapping_indices] for t in tensors_to_filter]

    return relabeled_image, new_labels, filtered_tensors