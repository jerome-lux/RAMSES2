import torch
import torch.nn.functional as F
import torch.nn as nn
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
    use_geom=True,
    beta=2.0,
):
    """Compute the combined loss for a single image (no batch dimension).

    Args:
        inputs: tuple of tensors:
            - cls_targets: [nloc, n_classes] one-hot class targets (background excluded),
              flattened over spatial positions and all FPN levels.
            - label_targets: [nloc] instance class labels (1-indexed).
            - density_targets: [nloc] target densities.
            - gt_masks: [H, W] ground-truth instance masks (instance IDs).
            - cls_pred: [nloc, n_classes] class predictions (post-sigmoid).
            - cls_factor_pred: [nloc] per-location class factors.
            - kernel_pred: [nloc, k*k*in_ch] predicted convolution kernels.
            - mask_head_pred: [in_ch, H, W] mask head feature map.
            - geom_factor_pred: [1, H, W] or [H, W, 1] geometric factor map.
        weights: list of [cls, seg, density] loss weights.
        kernel_size: spatial kernel size for the mask convolution.
        max_pos: max number of positive positions to use (memory cap).
        compute_seg_loss / compute_cls_loss / compute_density_loss: enable flags.
        label_smoothing: label smoothing rate (0 = off).
        seg_loss_func: 'focal', 'dice', or 'both'.
        use_geom: if True, multiply mask by geometric factors for density.
        beta: reserved (mask-quality weighting, currently unused).

    Returns:
        Tuple (cls_loss, seg_loss, density_loss), each already scaled by its weight.
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
                if use_geom:
                    pred_density = torch.sum(mask_slice * geom_slice) * cls_factor_pos[i]
                else: # Only for ablation study
                    pred_density = torch.sum(mask_slice) * cls_factor_pos[i]
                    # pred_density =  cls_factor_pos[i]
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
            # mask_quality = torch.zeros([nloc], device=cls_targets.device)
            for i in range(pos_idx.shape[0]):
                # if label_targets[pos_idx[i]] > ohe_masks.shape[-1]:
                #     print(label_targets, ohe_masks.shape, pos_idx[i])
                #     continue
                target_mask = ohe_masks[..., label_targets[pos_idx[i]] - 1]  # (H, W)

                if seg_loss_func == "dice":
                    mask_loss = dice_loss(seg_preds[i, ...], target_mask)
                elif seg_loss_func == "focal":
                    mask_loss = focal_loss(seg_preds[i, ...], target_mask)
                elif seg_loss_func == "both":
                    mask_loss = 0.5 * focal_loss(
                        seg_preds[i, ...], target_mask
                    ) + 0.5 * dice_loss(seg_preds[i, ...], target_mask)
                else:
                    print(f"Unknown seg_loss_func: {seg_loss_func}", file=sys.stderr)
                    sys.exit(1)
                seg_loss = seg_loss + mask_loss
                # if mask_quality_weighting:
                #     mask_quality[pos_idx[i]] = 1.0 - mask_loss.detach()

            seg_loss = seg_loss / pos_idx.shape[0]

            # if compute_cls_loss and mask_quality_weighting:

            #     cls_loss = focal_loss_label_smoothing(
            #         cls_pred, cls_targets, label_smoothing=label_smoothing, reduction="none"
            #     ).view(cls_pred.shape)
            #     cls_loss = cls_loss * (1 + (mask_quality**beta).unsqueeze(-1))
            #     cls_loss = cls_loss.sum()
            #     compute_cls_loss = False

    if compute_cls_loss:

        if label_smoothing>0:
            cls_loss = focal_loss_label_smoothing(cls_pred, cls_targets, label_smoothing=label_smoothing)
        else:
            cls_loss = focal_loss(cls_pred, cls_targets)

    return cls_loss * weights[0], seg_loss * weights[1], density_loss * weights[2]


def focal_loss_label_smoothing(pred, gt, alpha=0.25, gamma=2.0, label_smoothing: float = 0.0, reduction="sum"):
    """Binary focal loss with label smoothing.

    Label smoothing is applied only to positive positions:
    - one-hot targets (last dim > 1): positive = argmax == 1 (multi-class case);
    - binary targets: positive = entry == 1.

    Normalization uses the count of positive positions (anchor_obj_count).

    Args:
        pred: predictions (post-sigmoid), shape [nloc, n_classes] or [nloc].
        gt: one-hot or binary targets, same shape as pred.
        alpha: positive-class weight (default 0.25).
        gamma: focusing parameter (default 2.0).
        label_smoothing: smoothing rate (0 = disabled).
        reduction: 'sum' (default) or 'mean'.

    Returns:
        Focal loss (scalar if reduction != 'none').
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

    # Clamp predictions to [0, 1] to avoid numerical instability in BCE
    pred = torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)

    # Flatten inputs for BCE and weighting
    pred = pred.reshape(1, -1)
    gt = gt.reshape(1, -1)

    # If label_smoothing = 0, this is equivalent to (gt >= 1.0)
    pos_mask = (gt >= (1.0 - label_smoothing))

    # Apply alpha to positives, (1-alpha) to negatives
    alpha_tensor = torch.tensor(alpha, device=gt.device)
    alpha_factor = torch.where(pos_mask, alpha_tensor, 1.0 - alpha_tensor)

    # Calculate focal weight
    focal_weight = torch.where(pos_mask, 1.0 - pred, pred)
    focal_weight = alpha_factor * (focal_weight**gamma) / (anchor_obj_count + 1)
    focal_weight = focal_weight.detach()

    return F.binary_cross_entropy(pred, gt.float(), reduction=reduction, weight=focal_weight)


def focal_loss(pred, gt, alpha=0.25, gamma=2.0, reduction="sum"):
    """Standard binary focal loss (no label smoothing).

    Args:
        pred: predictions (post-sigmoid), any shape.
        gt: binary targets (0/1), same shape as pred.
        alpha: positive-class weight (default 0.25).
        gamma: focusing parameter (default 2.0).
        reduction: 'sum' (default) or 'mean'.

    Returns:
        Focal loss (scalar if reduction != 'none').
    """
    pred = pred.reshape(1, -1)
    gt = gt.reshape(1, -1)
    # Clamp to [0, 1] to avoid numerical instability in BCE
    pred = torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)

    if not torch.isfinite(pred).all():
        raise RuntimeError(f"focal_loss: non-finite values detected in predictions. min={pred.min().item()}, max={pred.max().item()}")
    anchor_obj_count = (gt != 0).sum().float()
    alpha_factor = torch.ones_like(gt) * alpha
    alpha_factor = torch.where(gt == 1, alpha_factor, 1 - alpha_factor)
    focal_weight = torch.where(gt == 1, 1 - pred, pred)
    focal_weight = alpha_factor * focal_weight**gamma / (anchor_obj_count + 1)
    focal_weight = focal_weight.float().detach()
    bce = F.binary_cross_entropy(pred, gt.float(), reduction=reduction, weight=focal_weight)
    return bce


class FocalLossMulti(nn.Module):
    """Multi-class focal loss (softmax) for one-hot targets.

    Computes L = -alpha_y * (1 - p_y)^gamma * log(p_y) per sample,
    where p_y is the softmax probability of the true class.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: [ncls] per-class weights tensor (None = unweighted).
            gamma: focusing parameter (default 2.0).
            reduction: 'mean' (default), 'sum', or 'none'.
        """
        super(FocalLossMulti, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [N, ncls] (logits), targets: [N, ncls] (one-hot)

        # 1. Compute softmax probabilities
        p = torch.softmax(inputs, dim=-1)

        # 2. Log-probabilities for the cross-entropy part
        log_p = F.log_softmax(inputs, dim=-1)

        # 3. Calculate Focal Term: (1 - p_target)^gamma
        # We extract the probability of the 'true' class using the mask
        pt = (p * targets).sum(dim=1)
        log_pt = (log_p * targets).sum(dim=1)

        focal_term = (1 - pt) ** self.gamma

        # 4. Combine into loss
        loss = -focal_term * log_pt

        # 5. Apply Alpha weights if provided
        if self.alpha is not None:
            # Broadcast alpha to match targets shape or apply per-sample
            at = (self.alpha.to(inputs.device) * targets).sum(dim=1)
            loss = at * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def dice_loss(pred, gt, label_smoothing: float = 0.0):
    """Binary Dice loss.

    Args:
        pred: predictions (post-sigmoid), shape [H, W] or [1, H, W].
        gt: binary masks (0/1), same shape as pred.
        label_smoothing: target smoothing rate (0 = disabled).

    Returns:
        Dice loss (scalar).
    """
    if label_smoothing > 0:
        gt = torch.where(gt == 1, 1.0 - label_smoothing, label_smoothing)
    a = torch.sum(pred * gt)
    b = torch.sum(pred * pred)
    c = torch.sum(gt * gt)
    dice = (2 * a) / (b + c + 1e-6)
    return 1.0 - dice


def masked_MSE(masked_y_true, y_pred, mask):
    """Mean squared error over the finite (True) entries of mask."""
    return torch.sum((masked_y_true - y_pred) ** 2) / torch.clamp(mask.float().sum(), min=1.0)


def masked_MAPE(masked_y_true, y_pred, mask, eps=1e-7):
    """Mean absolute percentage error over the finite (True) entries of mask."""
    num = torch.sum(torch.abs(y_pred - masked_y_true) / torch.clamp(torch.abs(masked_y_true), min=eps))
    denom = torch.clamp(mask.float().sum(), min=1.0)
    return num / denom


def masked_MAE(masked_y_true, y_pred, mask):
    """Mean absolute error over the finite (True) entries of mask."""
    return torch.sum(torch.abs(y_pred - masked_y_true)) / torch.clamp(mask.float().sum(), min=1.0)


def MSEIgnoringNaN(y_true, y_pred):
    """MSE that ignores NaN entries in y_true."""
    mask = torch.isfinite(y_true)
    masked_y_true = torch.where(mask, y_true, y_pred)
    loss = masked_MSE(masked_y_true, y_pred, mask)
    return loss


def MAPEIgnoringNaN(y_true, y_pred):
    """MAPE that ignores NaN entries in y_true."""
    mask = torch.isfinite(y_true)
    masked_y_true = torch.where(mask, y_true, y_pred)
    loss = masked_MAPE(masked_y_true, y_pred, mask)
    return loss


def MAEIgnoringNaN(y_true, y_pred):
    """MAE that ignores NaN entries in y_true."""
    mask = torch.isfinite(y_true)
    masked_y_true = torch.where(mask, y_true, y_pred)
    loss = masked_MAE(masked_y_true, y_pred, mask)
    return loss
