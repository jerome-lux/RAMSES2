import torch
import torch.nn.functional as F
import sys

# TODO: add counting loss ?


def compute_image_loss(
    inputs,
    weights=[1.0, 1.0, 1.0],
    kernel_size=1,
    max_pos=512,
    compute_seg_loss=True,
    compute_cls_loss=True,
    compute_density_loss=True,
    label_smoothing=0.1,
    seg_loss_func="dice",
    mask_quality_weighting=True,
    beta=2.0,
):
    """Compute the loss for one image
    Note that the inputs must not have a batch dimension.
    Args:
        inputs: tuple of tensors (see docstring below)
        weights: list of weights for [cls, seg, density]
        kernel_size: mask kernel params
        max_pos: max positive GT mask (to limit memory footprint)
        compute_seg_loss, compute_cls_loss, compute_density_loss: bools
        label_smoothing: float
        seg_loss_func: 'focal', 'dice', or 'both'
    Returns:
        cls_loss, seg_loss, density_loss (weighted)

    inputs:
        - cls_targets: [nloc, n_classes] one_hot encoded class targets (without the background slice). The targets are flattened over the spatial dimension and over all the FPN levels
        - label_targets: [nloc] tensor of class labels (1-indexed)
        - density_targets: [nloc] tensor of target densities
        - gt_masks: [H, W] tensor of ground truth masks (instance IDs)
        - cls_pred: [nloc, n_classes] tensor of class predictions (after sigmoid!)
        - cls_factor_pred: [nloc] tensor of class factors
        - kernel_pred: [nloc, k*k*in_ch] tensor of predicted kernels
        - mask_head_pred: [in_ch, H, W] tensor of mask head predictions
        - geom_factor_pred: [1, H, W] or [H, W, 1] tensor of geometric factors
    """
    (
        cls_targets,
        label_targets,
        density_targets,
        gt_masks,
        cls_pred,
        cls_factor_pred,
        kernel_pred,
        mask_head_pred,
        geom_factor_pred,
    ) = inputs

    seg_loss = 0.0
    density_loss = 0.0
    cls_loss = 0.0
    nloc = cls_pred.shape[0]

    mask_head_channels = mask_head_pred.shape[0]

    if compute_seg_loss or compute_density_loss:
        # labels = torch.unique(label_targets)
        max_label = gt_masks.max()

        pos_idx = torch.where(label_targets > 0)[0]

        if (max_pos is not None and max_pos > 0) and pos_idx.numel() > max_pos:
            perm = torch.randperm(pos_idx.numel(), device=cls_targets.device)
            pos_idx = pos_idx[perm[:max_pos]]
        # Ensure gt_masks is (H, W) and ohe_masks is (H, W, n_inst)
        if gt_masks.dim() == 3 and gt_masks.shape[0] == 1:
            gt_masks = gt_masks.squeeze(0)  # (H, W)

        ohe_masks = F.one_hot(gt_masks.long(), num_classes=int(max_label.item()) + 1)[..., 1:]  # [H, W, n_inst]

        if compute_density_loss:
            cls_factor_pos = cls_factor_pred[pos_idx]
            for i in range(pos_idx.shape[0]):
                # First Compute the element-wise product between the binary mask and the geom feature map
                mask_slice = (ohe_masks[..., label_targets[pos_idx[i]] - 1] > 0).float()
                geom_slice = geom_factor_pred[0]
                # The predicted mass is the product between the sum of the geom features and the predicted class factor (gt locations)
                pred_density = torch.sum(mask_slice * geom_slice) * cls_factor_pos[i]
                density_loss = density_loss + MAPEIgnoringNaN(density_targets[pos_idx[i]], pred_density)
            density_loss = density_loss / pos_idx.shape[0]
        if compute_seg_loss:
            # Note that we use the kernels predicted at the ground truth positive locations (and not at the positive locations of the cls_pred)
            kernel_pred_pos = kernel_pred[pos_idx]  # -> [npos, k*k*in_ch]
            # Reshape to [in_ch, npos, k, k]
            kernel_pred_pos = kernel_pred_pos.view(
                -1, mask_head_channels, kernel_size, kernel_size
            ).contiguous()  # [npos, in_ch, k, k]
            seg_preds_logits = F.conv2d(mask_head_pred.unsqueeze(0), kernel_pred_pos, stride=1, padding="same").squeeze(
                0
            )  # -> [npos, H, W]
            # because each slice is a binary mask, we use a sigmoid activation
            # reg_loss = torch.mean(seg_preds_logits.abs())
            # TODO: dynamically set new gt locations using best (dice) top-k masks?
            seg_preds = torch.sigmoid(seg_preds_logits)
            mask_quality = torch.zeros([nloc], device=cls_targets.device)
            for i in range(pos_idx.shape[0]):
                # if label_targets[pos_idx[i]] > ohe_masks.shape[-1]:
                #     print(label_targets, ohe_masks.shape, pos_idx[i])
                #     continue
                target_mask = ohe_masks[..., label_targets[pos_idx[i]] - 1]  # (H, W)

                if seg_loss_func == "dice":
                    mask_loss = dice_loss(seg_preds[i, ...], target_mask, label_smoothing=label_smoothing)
                elif seg_loss_func == "focal":
                    mask_loss = focal_loss(seg_preds[i, ...], target_mask, label_smoothing=label_smoothing)
                elif seg_loss_func == "both":
                    mask_loss = 0.5 * focal_loss(
                        seg_preds[i, ...], target_mask, label_smoothing=label_smoothing
                    ) + 0.5 * dice_loss(seg_preds[i, ...], target_mask, label_smoothing=label_smoothing)
                else:
                    print(f"Unknown seg_loss_func: {seg_loss_func}", file=sys.stderr)
                    sys.exit(1)
                seg_loss = seg_loss + mask_loss
                if mask_quality_weighting:
                    mask_quality[pos_idx[i]] = 1.0 - mask_loss.detach()

            seg_loss = seg_loss / pos_idx.shape[0]

            if compute_cls_loss and mask_quality_weighting:

                cls_loss = focal_loss_label_smoothing(
                    cls_pred, cls_targets, label_smoothing=label_smoothing, reduction="none"
                ).view(cls_pred.shape) + cls_loss * (mask_quality**beta).unsqueeze(-1)
                cls_loss = cls_loss.sum()
                compute_cls_loss = False

    if compute_cls_loss:
        cls_loss = focal_loss_label_smoothing(cls_pred, cls_targets, label_smoothing=label_smoothing)

    return cls_loss * weights[0], seg_loss * weights[1], density_loss * weights[2]


def focal_loss_label_smoothing(pred, gt, alpha=0.25, gamma=2.0, label_smoothing: float = 0.0, reduction="sum"):
    """Focal loss variant that applies label smoothing only on spatial positions
    where the last-dimension argmax equals 1.

    - If `gt` is one-hot (last dim > 1), a position is considered positive when
      `gt.argmax(dim=-1) == 1`. Label smoothing is applied only
      on those positions; other positions remain unchanged (0).
    - If `gt` is binary (no channel dim), smoothing is applied on entries equal to 1.

    The weighting and normalization use the count of positive positions (anchor_obj_count).
    """
    orig_gt = gt.clone()

    is_one_hot = orig_gt.dim() > 1 and orig_gt.shape[-1] > 1

    # Determine positive positions (argmax == 1)
    if is_one_hot:
        pos_positions = orig_gt.argmax(dim=-1) == 1
        anchor_obj_count = pos_positions.sum().float()
        n_class = orig_gt.shape[-1]
    else:
        pos_positions = orig_gt == 1
        anchor_obj_count = pos_positions.sum().float()
        n_class = 1

    # Build smoothed gt: only apply smoothing on the selected positions
    if label_smoothing and label_smoothing > 0.0:
        if is_one_hot:
            smooth = torch.where(orig_gt == 1, 1.0 - label_smoothing, label_smoothing / (n_class - 1))
            gt = orig_gt.float()
            mask_expand = pos_positions.unsqueeze(-1).expand_as(orig_gt)
            gt = torch.where(mask_expand, smooth, gt)
        else:
            gt = torch.where(orig_gt == 1, 1.0 - label_smoothing, label_smoothing)
    else:
        gt = orig_gt.float()

    # Flatten inputs for BCE and weighting
    pred = pred.reshape(1, -1)
    gt = gt.reshape(1, -1)
    pos_mask = None
    if is_one_hot:
        pos_mask = pos_positions.reshape(1, -1)
    else:
        pos_mask = (orig_gt == 1).reshape(1, -1)

    alpha_factor = torch.ones_like(gt) * alpha
    alpha_factor = torch.where(pos_mask, alpha_factor, 1 - alpha_factor)
    focal_weight = torch.where(pos_mask, 1 - pred, pred)
    focal_weight = alpha_factor * (focal_weight**gamma) / (anchor_obj_count + 1)
    focal_weight = focal_weight.float().detach()

    return F.binary_cross_entropy(pred, gt.float(), reduction=reduction, weight=focal_weight)


def focal_loss(pred, gt, alpha=0.25, gamma=2.0, label_smoothing: float = 0.0, reduction="sum"):

    pred = pred.reshape(1, -1)
    gt = gt.reshape(1, -1)
    anchor_obj_count = (gt != 0).sum().float()
    alpha_factor = torch.ones_like(gt) * alpha
    alpha_factor = torch.where(gt == 1, alpha_factor, 1 - alpha_factor)
    focal_weight = torch.where(gt == 1, 1 - pred, pred)
    focal_weight = alpha_factor * focal_weight**gamma / (anchor_obj_count + 1)
    focal_weight = focal_weight.float().detach()
    bce = F.binary_cross_entropy(pred, gt.float(), reduction=reduction, weight=focal_weight)
    return bce


def dice_loss_with_logits(logits, targets, eps=1e-6, label_smoothing=0.0, reduction="mean"):
    """
    Binary Dice Loss calcul partir de logits en utilisant logsigmoid

    Args:
        logits (Tensor): pr�dictions non activ�es, shape (N, *) ou (N, 1, H, W)
        targets (Tensor): valeurs binaires (0 ou 1), m�me shape que logits
        eps (float): terme pour �viter la division par z�ro
        reduction (str): 'mean', 'sum' ou 'none'

    Returns:
        Tensor: Dice loss
    """
    if label_smoothing > 0:
        targets = torch.where(targets == 1, 1.0 - label_smoothing, label_smoothing)

    probs = torch.exp(F.logsigmoid(logits))  # sigmoid(logits)

    # Dice coefficient
    intersection = torch.sum(probs * targets, dim=(1, 2, 3) if logits.dim() == 4 else 1)
    union = torch.sum(probs + targets, dim=(1, 2, 3) if logits.dim() == 4 else 1)

    dice = (2 * intersection + eps) / (union + eps)
    loss = 1 - dice

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def dice_loss(pred, gt, label_smoothing: float = 0.0):
    if label_smoothing > 0:
        gt = torch.where(gt == 1, 1.0 - label_smoothing, label_smoothing)
    a = torch.sum(pred * gt)
    b = torch.sum(pred * pred)
    c = torch.sum(gt * gt)
    dice = (2 * a) / (b + c + 1e-6)  # if (b + c) > 0 else torch.tensor(0.0, device=pred.device)
    return 1.0 - dice


def masked_MSE(masked_y_true, y_pred, mask):
    return torch.sum((masked_y_true - y_pred) ** 2) / torch.clamp(mask.float().sum(), min=1.0)


def masked_MAPE(masked_y_true, y_pred, mask, eps=1e-7):
    num = torch.sum(torch.abs(y_pred - masked_y_true) / torch.clamp(torch.abs(masked_y_true), min=eps))
    denom = torch.clamp(mask.float().sum(), min=1.0)
    return num / denom


def masked_MAE(masked_y_true, y_pred, mask):
    return torch.sum(torch.abs(y_pred - masked_y_true)) / torch.clamp(mask.float().sum(), min=1.0)


def MSEIgnoringNaN(y_true, y_pred):
    mask = torch.isfinite(y_true)
    masked_y_true = torch.where(mask, y_true, y_pred)
    loss = masked_MSE(masked_y_true, y_pred, mask)
    return loss


def MAPEIgnoringNaN(y_true, y_pred):
    mask = torch.isfinite(y_true)
    masked_y_true = torch.where(mask, y_true, y_pred)
    loss = masked_MAPE(masked_y_true, y_pred, mask)
    return loss


def MAEIgnoringNaN(y_true, y_pred):
    mask = torch.isfinite(y_true)
    masked_y_true = torch.where(mask, y_true, y_pred)
    loss = masked_MAE(masked_y_true, y_pred, mask)
    return loss
