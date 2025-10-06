from typing import Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from ..core import norm


def get_class(name: str, extra_modules=[norm]):
    """
    Recherche une classe par nom dans torch.nn puis dans une liste de modules supplémentaires.

    Args:
        name (str): nom de la classe (ex: "ReLU" ou "GroupNormCustom")
        extra_modules (list[module]): modules dans lesquels chercher si non trouvé dans torch.nn

    Returns:
        type: la classe Python correspondante

    Raises:
        ValueError: si la classe n'est trouvée dans aucun module
    """
    if extra_modules is None:
        extra_modules = []

    # Essayer d'abord dans torch.nn
    if hasattr(nn, name):
        return getattr(nn, name)

    # Sinon chercher dans les modules fournis
    for module in extra_modules:
        if hasattr(module, name):
            return getattr(module, name)

    raise ValueError(f"Classe '{name}' non trouvée dans torch.nn ni dans les modules fournis {extra_modules}.")


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
    Args:
        x (Tensor): [N, H, W] ou [C, H, W], la carte de score dense
        kernel_size (int): taille de la fenêtre locale (souvent 2)
    Returns:
        Tensor: carte de score avec suppression des non-maxima locaux
    """
    # Appliquer un max pooling local pour détecter les maxima
    max_score = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    # Garder uniquement les points qui sont des maxima
    keep = (x == max_score).float()

    return x * keep


def get_same_padding(kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]]):
    """
    Calcule le padding (gauche, droite, haut, bas) nécessaire pour avoir une sortie
    de taille in_dim // stride avec un kernel donné.

    Args:
        kernel_size: taille du noyau (int ou tuple de 2)
        stride: facteur de stride (int ou tuple de 2)

    Returns:
        Tuple[int, int, int, int]: (padding_left, padding_right, padding_top, padding_bottom)
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)

    pad_h = max(stride[0] - 1 + kernel_size[0] - stride[0], 0)
    pad_w = max(stride[1] - 1 + kernel_size[1] - stride[1], 0)

    # Répartir le padding à gauche/droite et haut/bas
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return (pad_left, pad_right, pad_top, pad_bottom)


def pad_with_coords(data):
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
        seg_preds: (N, H, W) torch.Tensor, one predicted mask per slice (sigmoid activation)
        scores: (N,) torch.Tensor, score of each predicted instance
        threshold: float, threshold to compute binary masks
        by_scores: bool, if True, rank the masks by score, else, rank each pixel by their seg_pred value.

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
        shape (tuple): Shape of the input image (width, height).
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

                # Ajouter le centre du box si rien à l’intérieur
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


def boxes_from_masks(masks):
    """
    Returns:
    boxes (center x, center y, H, W) of labelled objects
    an array with the corresponding labels in the image.
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


# def compute_mask_targets(gt_masks, gt_labels):
#     """
#     gt_masks: torch.Tensor of shape (H, W, num_objects), one-hot encoded masks (last dim is object index)
#     gt_labels: torch.Tensor of shape (N,), flattened vector of labels (corresponding to mask indices)
#     Returns:
#         mask_targets: torch.Tensor of shape (H, W, npos), where npos = number of positive labels (labels > 0)
#                       Each slice corresponds to a mask target, arranged in the same order as the class/labels targets.
#     """
#     pos_indices = (gt_labels > 0).nonzero(as_tuple=True)[0]
#     pos_labels = gt_labels[pos_indices]

#     # Gather the corresponding mask slices for each positive label
#     mask_targets = []
#     for label in pos_labels:
#         mask_targets.append(gt_masks[..., label.item()].unsqueeze(-1))
#     if mask_targets:
#         mask_targets = torch.cat(mask_targets, dim=-1)
#     else:
#         # No positive labels, return empty tensor with correct shape
#         H, W, _ = gt_masks.shape
#         mask_targets = torch.empty((H, W, 0), dtype=gt_masks.dtype, device=gt_masks.device)
#     return mask_targets


def compute_locations(stride, shape, shift="r"):
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


def normalize_bboxes(bboxes, nx, ny):

    normalized_bboxes = bboxes.astype(np.float32)
    normalized_bboxes[..., 0] /= nx - 1
    normalized_bboxes[..., 1] /= ny - 1
    normalized_bboxes[..., 2] /= nx - 1
    normalized_bboxes[..., 3] /= ny - 1
    return normalized_bboxes


def denormalize_bboxes(norm_bboxes, nx, ny):

    bboxes = np.zeros(norm_bboxes.shape).astype(np.int32)
    bboxes[..., 0] = np.maximum(np.around(norm_bboxes[..., 0] * (nx - 1)), 0)
    bboxes[..., 1] = np.maximum(np.around(norm_bboxes[..., 1] * (ny - 1)), 0)
    bboxes[..., 2] = np.minimum(np.around(norm_bboxes[..., 2] * (nx - 1)), nx)
    bboxes[..., 3] = np.minimum(np.around(norm_bboxes[..., 3] * (ny - 1)), ny)
    return bboxes


def crop_to_aspect_ratio(target_shape, image):
    """Crop an image so that its aspect ratio is the same as target_shape
    Note: it does not resize the image to target_shape !"""

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

    if image.ndim > 4:
        print("pad_to_aspect_ratio only support gray or RGB 2D images")

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
    """Pad an image so that its aspect ratio is the same as target_shape"""

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

    return image, ((px, px + rx), (py), py + ry)


def relabel_and_filter(masks: torch.Tensor, original_labels: torch.Tensor, *tensors_to_filter):
    """
    Relabels the labeled mask image with consecutive integers starting from 1,
    and filters associated tensors based on the mapping from original labels.

    Args:
        masks (torch.Tensor): Labeled image after rotation, with integer labels (0 for background).
        original_labels (torch.Tensor): 1D tensor of original labels.
        *tensors_to_filter: Additional tensors to be filtered according to the new label mapping.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
            - relabeled_image: Image with relabeled components (background remains 0).
            - mapping_indices: Indices mapping new labels to original labels.
            - filtered_tensors: List of filtered tensors corresponding to the new labels.
    """

    # Extraire les labels uniques, en excluant 0 (le fond)
    device = masks.device
    unique_labels = torch.unique(masks)
    unique_labels = unique_labels[unique_labels != 0]
    unique_labels = unique_labels.sort().values

    expected_consecutive_labels = torch.arange(1, unique_labels.numel() + 1, device=device)

    if (
        torch.all(unique_labels == expected_consecutive_labels)
        and unique_labels.numel() == original_labels.numel()
        and torch.all(unique_labels == original_labels.sort().values)
    ):
        return masks, original_labels, list(tensors_to_filter)

    # Créer le mapping: ancien label -> nouveau label (entiers consécutifs à partir de 1)
    new_labels = torch.arange(1, len(unique_labels) + 1)
    label_to_new = dict(zip(unique_labels.tolist(), new_labels.tolist()))

    # Relabellisation de l'image (le fond reste 0)
    relabeled_image = masks.clone()
    for old_label, new_label in label_to_new.items():
        relabeled_image[masks == old_label] = new_label

    # Créer le mapping des indices dans original_labels
    mapping_indices = []
    for label in unique_labels:
        matches = (original_labels == label).nonzero(as_tuple=True)[0]
        if len(matches) == 0:
            # print(unique_labels)
            # print(original_labels)
            # print(ValueError(f"Label {label.item()} not found in original_labels"))
            raise ValueError(f"Label {label.item()} not found in original_labels")
        if len(matches) > 1:
            raise ValueError(
                f"Multiple matches found for label {label.item()} in original_labels: {matches.tolist()}"
                f", unique labels: {unique_labels.tolist()}, original labels: {original_labels.tolist()}"
                f"{tensors_to_filter}"
            )
        mapping_indices.append(matches.item())  # Un seul match attendu

    mapping_indices = torch.tensor(mapping_indices, dtype=torch.long)

    # Filtrage des tenseurs d'entrée
    filtered_tensors = [t[mapping_indices].to(device) for t in tensors_to_filter]

    return relabeled_image.to(device), new_labels.to(device), filtered_tensors
