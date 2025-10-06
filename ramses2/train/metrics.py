import torch


def Recall(y_true, y_pred, threshold=0, eps=1e-8):
    """
    Compute recall for one-hot tensors [N, ncls] WITHOUT background column.
    Dynamically adds a background (class 0) column on the left.
    Args:
        y_true (torch.Tensor): one-hot, shape [N, ncls] (no background)
        y_pred (torch.Tensor): logits or probabilities, shape [N, ncls] (no background)
        eps (float): small value to avoid division by zero
    Returns:
        float: recall over all classes
    """
    N, ncls = y_true.shape
    if threshold > 0:
        y_pred = torch.where(y_pred > threshold, y_pred, 0)
    y_true_bg = torch.cat([1 - y_true.sum(dim=1, keepdim=True), y_true], dim=1)
    y_pred_bg = torch.cat([1 - y_pred.sum(dim=1, keepdim=True), y_pred], dim=1)
    true_labels = y_true_bg.argmax(dim=1)
    pred_labels = y_pred_bg.argmax(dim=1)
    mask_fg = true_labels != 0  # Only foreground
    tp = ((pred_labels == true_labels) & mask_fg).sum()
    npos = mask_fg.sum()
    recall = tp.float() / (npos + eps)
    return recall.item()


def Precision(y_true, y_pred, threshold=0, eps=1e-8):
    """
    Compute precision for one-hot tensors [N, ncls] WITHOUT background column.
    Dynamically adds a background (class 0) column on the left.
    Args:
        y_true (torch.Tensor): one-hot, shape [N, ncls] (no background)
        y_pred (torch.Tensor): logits or probabilities, shape [N, ncls] (no background)
        eps (float): small value to avoid division by zero
    Returns:
        float: precision over all classes
    """
    N, ncls = y_true.shape
    if threshold > 0:
        y_pred = torch.where(y_pred > threshold, y_pred, 0)
    y_true_bg = torch.cat([1 - y_true.sum(dim=1, keepdim=True), y_true], dim=1)
    y_pred_bg = torch.cat([1 - y_pred.sum(dim=1, keepdim=True), y_pred], dim=1)
    true_labels = y_true_bg.argmax(dim=1)
    pred_labels = y_pred_bg.argmax(dim=1)
    mask_fg = pred_labels != 0  # Only foreground predictions
    tp = ((pred_labels == true_labels) & mask_fg).sum()
    n_pred = mask_fg.sum()
    precision = tp.float() / (n_pred + eps)
    return precision.item()
