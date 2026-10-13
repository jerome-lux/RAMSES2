import torch


def Recall(y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float = 0, eps: float = 1e-8) -> float:
    """
    Compute recall for one-hot tensors [N, ncls] WITHOUT background column.
    Dynamically adds a background (class 0) column on the left.
    Args:
        y_true (torch.Tensor): one-hot, shape [N, ncls] (no background)
        y_pred (torch.Tensor): probabilities, shape [N, ncls] (no background)
        threshold (float): predictions below this value are zeroed out before classification (default 0 = no thresholding)
        eps (float): small value to avoid division by zero
    Returns:
        float: recall over all classes
    """
    ncls = y_true.shape[1]
    # Clamp y_pred to [0, 1] to avoid negative or > 1 values
    y_pred = y_pred.clamp(0, 1)
    if threshold > 0:
        y_pred = torch.where(y_pred > threshold, y_pred, 0)
    y_true_bg = torch.cat([1 - y_true.sum(dim=1, keepdim=True), y_true], dim=1).clamp(0, 1)
    y_pred_bg = torch.cat([1 - y_pred.sum(dim=1, keepdim=True), y_pred], dim=1).clamp(0, 1)
    true_labels = y_true_bg.argmax(dim=1)
    pred_labels = y_pred_bg.argmax(dim=1)
    mask_fg = true_labels != 0  # Only foreground
    tp = ((pred_labels == true_labels) & mask_fg).sum()
    npos = mask_fg.sum()
    recall = tp.float() / (npos + eps)
    return recall.item()


def Precision(y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float = 0, eps: float = 1e-8) -> float:
    """
    Compute precision for one-hot tensors [N, ncls] WITHOUT background column.
    Dynamically adds a background (class 0) column on the left.
    Args:
        y_true (torch.Tensor): one-hot, shape [N, ncls] (no background)
        y_pred (torch.Tensor): probabilities, shape [N, ncls] (no background)
        threshold (float): predictions below this value are zeroed out before classification (default 0 = no thresholding)
        eps (float): small value to avoid division by zero
    Returns:
        float: precision over all classes
    """
    ncls = y_true.shape[1]
    # Clamp y_pred to [0, 1] to avoid negative or > 1 values
    y_pred = y_pred.clamp(0, 1)
    if threshold > 0:
        y_pred = torch.where(y_pred > threshold, y_pred, 0)
    y_true_bg = torch.cat([1 - y_true.sum(dim=1, keepdim=True), y_true], dim=1).clamp(0, 1)
    y_pred_bg = torch.cat([1 - y_pred.sum(dim=1, keepdim=True), y_pred], dim=1).clamp(0, 1)
    true_labels = y_true_bg.argmax(dim=1)
    pred_labels = y_pred_bg.argmax(dim=1)
    mask_fg = pred_labels != 0  # Only foreground predictions
    tp = ((pred_labels == true_labels) & mask_fg).sum()
    n_pred = mask_fg.sum()
    precision = tp.float() / (n_pred + eps)
    return precision.item()
