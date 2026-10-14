# coding: utf-8

import logging

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc

from ..train import collate_fn

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    model,
    dataset,
    annotations,
    maxiter,
    ids_to_cls,
    iou_threshold=0.75,
    mask_stride=4,
    cls_threshold=0.45,
    nms_threshold=0.5,
    mask_threshold=0.5,
    max_detections=768,
    scale_by_mask_scores=False,
    min_area=128,
    nms_mode="greedy",
    exclude=None,
    verbose=False,
    remove_wiggles=True,
    device="cuda:0",
    shuffle=False,
    **kwargs,
):
    """Compute AP, mass-error metrics, and per-image statistics.

    Args:
        model: RAMSES2 model instance.
        dataset: A ``torch.utils.data.Dataset`` yielding annotated images.
        annotations: A pandas DataFrame with per-instance metadata
            (columns: ``baseimg``, ``res``, ``height``, ``width``, …).
        maxiter: Maximum number of images to evaluate.
        ids_to_cls: Mapping from integer class ID to class name.
        iou_threshold: Minimum IoU for a prediction to count as a true positive.
        mask_stride: Down-sampling factor of the mask feature map relative to input.
        cls_threshold: Classification confidence threshold.
        nms_threshold: IoU threshold for greedy/soft NMS, or class threshold for MatrixNMS.
        mask_threshold: Binarisation threshold for predicted masks.
        max_detections: Maximum number of instances to retain per image.
        scale_by_mask_scores: Whether to scale mask scores by class factors.
        min_area: Minimum mask area (pixels) to keep.
        nms_mode: One of ``"greedy"``, ``"soft"``, or ``"matrix"``.
        exclude: Optional list of class names to exclude from metrics.
        verbose: If True, print per-image details.
        remove_wiggles: Enforce monotonic precision before AUC computation.
        device: Torch device string.
        shuffle: Whether to shuffle the dataloader.

    Returns:
        tuple: ``(results, total_gt_mass_per_cls, per_image_results)`` where
        *results* is a dict keyed by class name (plus ``"all"``) containing
        AP, precision/recall curves, mass-error metrics, and counts.
    """
    model = model.to(device)
    model.eval()

    # Normalise exclude list (avoids mutable-default pitfall and in-place mutation)
    exclude = list(exclude) if exclude else []
    exclude.append("UNKNOWN")

    # Extract per-image metadata in a single vectorised pass
    first_per_img = annotations.drop_duplicates(subset=["baseimg"], keep="first")
    resolutions = dict(zip(first_per_img["baseimg"], first_per_img["res"]))
    heights = dict(zip(first_per_img["baseimg"], first_per_img["height"]))

    total_raw_pred_mass_per_cls = {name: 0.0 for name in ids_to_cls.values()}

    gt_accumulator = 0
    total_gt_mass_per_cls = {}

    TPFP_per_pred = []
    gt_cls_labels_per_pred = []
    iou_per_pred = []
    pred_cls_labels_list = []
    pred_scores_list = []
    pred_masses_list = []
    gt_cls_labels_list = []
    gt_mass_per_pred = []
    all_pred_score_list = []
    all_pred_cls_labels_list = []

    per_image_results = {}

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=shuffle, num_workers=2, collate_fn=collate_fn, pin_memory=True
    )

    for i, inputs in enumerate(dataloader):
        if i >= maxiter:
            break

        imgname = inputs["filename"][0]
        gt_img = inputs["image"].to(device)
        gt_mask_img = inputs["masks"][0].to(device)
        gt_cls_ids = inputs["category_id"][0]
        gt_mass = inputs["mass"][0]
        mask_res = inputs["res"][0][0]
        nx, _ = gt_img.shape[-2:]

        ratio = (nx // mask_stride) / heights[imgname]
        scaling = (10 * resolutions[imgname] * ratio) ** 2

        # Accumulate total GT mass per class (skip non-finite masses)
        cls_ids_np = gt_cls_ids.numpy()
        finite_mask = torch.isfinite(gt_mass)
        for j in range(cls_ids_np.shape[-1]):
            if finite_mask[j]:
                total_gt_mass_per_cls[cls_ids_np[j]] = (
                    total_gt_mass_per_cls.get(cls_ids_np[j], 0.0) + gt_mass[j].item() / scaling
                )

        ngt = gt_cls_ids.shape[0]
        gt_accumulator += ngt

        results = model(
            gt_img,
            training=False,
            cls_threshold=cls_threshold,
            nms_threshold=nms_threshold,
            mask_threshold=mask_threshold,
            max_detections=max_detections,
            scale_by_mask_scores=scale_by_mask_scores,
            min_area=min_area,
            nms_mode=nms_mode,
        )[0]

        pred_masks = results["masks"].detach()
        pred_masses = results["masses"].detach()
        pred_cls_ids = results["cls_labels"].detach() + 1
        pred_scores = results["scores"].detach()
        npred = pred_cls_ids.shape[0]

        # --- Per-image results (populated regardless of prediction count) ---
        finite_gt_mass_sum = gt_mass[finite_mask].sum().item() / scaling if finite_mask.any() else 0.0
        pred_mass_sum = pred_masses.sum().item() / scaling if npred > 0 else 0.0

        per_image_results[imgname] = {
            "gt_masses": {name: 0.0 for name in ids_to_cls.values()},
            "pred_masses": {name: 0.0 for name in ids_to_cls.values()},
            "total_pred_mass": pred_mass_sum,
            "total_gt_mass": finite_gt_mass_sum,
            "ngt": ngt,
            "npred": npred,
            "RE": (pred_mass_sum - finite_gt_mass_sum) / (finite_gt_mass_sum + 1e-10),
        }

        # Per-class GT mass for this image
        for j in range(ngt):
            cls_name = ids_to_cls.get(gt_cls_ids[j].item(), "UNKNOWN")
            if cls_name in per_image_results[imgname]["gt_masses"]:
                per_image_results[imgname]["gt_masses"][cls_name] += gt_mass[j].item() / scaling

        # Per-class predicted mass for this image
        for j in range(npred):
            cls_name = ids_to_cls.get(pred_cls_ids[j].item(), "UNKNOWN")
            if cls_name not in per_image_results[imgname]["pred_masses"]:
                per_image_results[imgname]["pred_masses"][cls_name] = 0.0
            per_image_results[imgname]["pred_masses"][cls_name] += pred_masses[j].item() / scaling

        # Accumulate raw predicted mass per class
        for j in range(npred):
            cls_name = ids_to_cls.get(pred_cls_ids[j].item(), "UNKNOWN")
            if cls_name != "UNKNOWN":
                total_raw_pred_mass_per_cls[cls_name] = (
                    total_raw_pred_mass_per_cls.get(cls_name, 0.0) + pred_masses[j].item() / scaling
                )

        # Record GT labels for this image (used for n_gt_instances and class counts)
        gt_cls_labels_list.append(gt_cls_ids)

        # --- No predictions: nothing to match ---
        if npred == 0:
            if verbose:
                print(f"{i + 1}/{maxiter} Image {imgname}: {ngt} GT, 0 predictions.")
            else:
                print(f"Iteration {i + 1:>5d}/{maxiter} ", end="\r")
            continue

        # --- No GT objects: all predictions are false positives ---
        if ngt == 0:
            TPFP_per_pred.append(torch.zeros(npred, dtype=torch.float32))
            iou_per_pred.append(torch.zeros(npred, dtype=torch.float32))
            pred_cls_labels_list.append(pred_cls_ids.cpu())
            pred_scores_list.append(pred_scores.cpu())
            pred_masses_list.append(pred_masses.cpu() / scaling)
            gt_cls_labels_per_pred.append(torch.empty(0, dtype=gt_cls_ids.dtype))
            gt_mass_per_pred.append(torch.empty(0, dtype=gt_mass.dtype))
            all_pred_score_list.append(pred_scores.cpu())
            all_pred_cls_labels_list.append(pred_cls_ids)
            if verbose:
                print(f"{i + 1}/{maxiter} Image {imgname}: 0 GT, {npred} predictions (all FP).")
            else:
                print(f"Iteration {i + 1:>5d}/{maxiter} ", end="\r")
            continue

        # --- IoU computation ---
        gt_masks = F.one_hot(gt_mask_img, ngt + 1)[..., 1:]
        gt_masks = gt_masks.reshape(-1, ngt).transpose(0, 1).float()  # [Ngt, H*W]

        pred_masks_bin = (pred_masks > mask_threshold).to(pred_masks.dtype)
        pred_masks_bin = pred_masks_bin.reshape(npred, -1)  # [Npred, H*W]

        intersections = torch.matmul(gt_masks, pred_masks_bin.T)
        gt_sums = gt_masks.sum(dim=1)
        pred_sums = pred_masks_bin.sum(dim=1)
        unions = gt_sums.unsqueeze(1) + pred_sums.unsqueeze(0) - intersections
        iou = intersections / (unions + 1e-6)

        # --- Greedy one-to-one matching (sorted by IoU descending) ---
        max_iou_per_pred, gt_indices_for_each_pred = torch.max(iou, dim=0)
        final_gt_matches = -torch.ones(npred, dtype=torch.long)
        final_iou_values = torch.zeros(npred, dtype=torch.float32)

        candidate_matches = [
            (max_iou_per_pred[p].item(), p, gt_indices_for_each_pred[p].item())
            for p in range(npred)
        ]
        candidate_matches.sort(key=lambda x: x[0], reverse=True)

        assigned_gt = torch.zeros(ngt, dtype=torch.bool)
        for iou_val, pred_idx, gt_idx in candidate_matches:
            if not assigned_gt[gt_idx]:
                final_gt_matches[pred_idx] = gt_idx
                final_iou_values[pred_idx] = iou_val
                assigned_gt[gt_idx] = True

        valid_pred_indices = torch.where(final_gt_matches != -1)[0]
        if valid_pred_indices.numel() == 0:
            if verbose:
                print(f"{i + 1}/{maxiter} Image {imgname}: no matching predictions.")
            else:
                print(f"Iteration {i + 1:>5d}/{maxiter} ", end="\r")
            continue

        valid_gt_indices = final_gt_matches[valid_pred_indices]

        TPFP_per_pred.append(torch.where(final_iou_values > iou_threshold, 1, 0))
        iou_per_pred.append(final_iou_values)
        gt_cls_labels_per_pred.append(torch.gather(gt_cls_ids, 0, valid_gt_indices))
        pred_cls_labels_list.append(torch.gather(pred_cls_ids.cpu(), 0, valid_pred_indices))
        pred_scores_list.append(torch.gather(pred_scores.cpu(), 0, valid_pred_indices))
        pred_masses_list.append(torch.gather(pred_masses.cpu(), 0, valid_pred_indices) / scaling)
        gt_mass_per_pred.append(torch.gather(gt_mass, 0, valid_gt_indices) / scaling)
        all_pred_score_list.append(pred_scores.cpu())
        all_pred_cls_labels_list.append(pred_cls_ids)

        if verbose:
            cls_unique = [ids_to_cls.get(int(c), "UNKNOWN") for c in np.unique(cls_ids_np)]
            pred_cls_unique = [ids_to_cls.get(int(c), "UNKNOWN") for c in np.unique(pred_cls_ids.cpu().numpy())]
            print(
                f"{i + 1}/{maxiter} Image {imgname}: {ngt} GT, {npred} pred. "
                f"GT mass: {finite_gt_mass_sum:.2f}, pred mass: {pred_mass_sum:.2f} "
                f"| classes: {cls_unique}, pred classes: {pred_cls_unique} "
                f"| res: {resolutions[imgname]} pix/mm, "
                f"mask res: {resolutions[imgname] * ratio:.2f} ({mask_res:.2f})",
                flush=True,
            )
        else:
            print(f"Iteration {i + 1:>5d}/{maxiter} ", end="\r")

    # --- Concatenate per-prediction tensors ---
    if not TPFP_per_pred:
        logger.warning("No valid predictions across the entire dataset. Returning empty results.")
        print("")
        return {}, total_gt_mass_per_cls, per_image_results

    TPFP_per_pred = torch.cat(TPFP_per_pred, dim=0)
    pred_cls_labels_list = torch.cat(pred_cls_labels_list, dim=0)
    pred_scores_list = torch.cat(pred_scores_list, dim=0)
    pred_masses_list = torch.cat(pred_masses_list, dim=0)
    gt_cls_labels_per_pred = torch.cat(gt_cls_labels_per_pred, dim=0)
    gt_cls_labels_list = torch.cat(gt_cls_labels_list, dim=0)
    gt_mass_per_pred = torch.cat(gt_mass_per_pred, dim=0)
    iou_per_pred = torch.cat(iou_per_pred, dim=0)
    all_pred_score_list = torch.cat(all_pred_score_list, dim=0)
    all_pred_cls_labels_list = torch.cat(all_pred_cls_labels_list, dim=0)
    print("")

    # Sort predictions by score (descending)
    sorted_scores, sorted_pred_indices = torch.sort(pred_scores_list, descending=True)
    sorted_iou_per_pred = iou_per_pred[sorted_pred_indices]
    sorted_TPFP_per_pred = TPFP_per_pred[sorted_pred_indices]
    sorted_pred_cls_labels = pred_cls_labels_list[sorted_pred_indices]
    sorted_gt_cls_labels_per_pred = gt_cls_labels_per_pred[sorted_pred_indices]
    sorted_pred_masses = pred_masses_list[sorted_pred_indices]
    sorted_gt_masses_per_pred = gt_mass_per_pred[sorted_pred_indices]

    # Zero out TP where the matched GT class differs from the predicted class
    sorted_TPFP_per_pred = torch.where(
        sorted_gt_cls_labels_per_pred == sorted_pred_cls_labels, sorted_TPFP_per_pred, 0
    )

    # Build class list from GT
    cls_ids_tensor, gt_count_per_cls = torch.unique(gt_cls_labels_list, return_counts=True)
    cls_ids = cls_ids_tensor.cpu().numpy().tolist()
    cls_ids.append("all")

    results = {}
    total_pred_mass_per_cls = []

    # AP per class
    for i, cls_id in enumerate(cls_ids):
        if cls_id != "all" and ids_to_cls.get(cls_id, "UNKNOWN") in exclude:
            logger.debug("Skipping class %s (in exclude list)", ids_to_cls.get(cls_id, "UNKNOWN"))
            continue

        if cls_id == "all":
            # Exclude excluded classes from the denominator
            n_gt_instances = sum(
                c for cid, c in zip(cls_ids[:-1], gt_count_per_cls.tolist())
                if ids_to_cls.get(cid, "UNKNOWN") not in exclude
            )
            iou_per_cls = sorted_iou_per_pred
            sorted_TPFP_per_pred_per_cls = sorted_TPFP_per_pred
            scores_cls = sorted_scores
        else:
            n_gt_instances = gt_count_per_cls[i].item()
            idx = (sorted_pred_cls_labels == cls_id).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                logger.debug("No predictions found for class %s", cls_id)
                continue
            sorted_TPFP_per_pred_per_cls = sorted_TPFP_per_pred[idx]
            iou_per_cls = sorted_iou_per_pred[idx]
            scores_cls = sorted_scores[idx]

        # Precision / Recall curve
        TP = torch.cumsum(sorted_TPFP_per_pred_per_cls, dim=0)
        FP = torch.cumsum(1.0 - sorted_TPFP_per_pred_per_cls, dim=0)
        precision = TP / (TP + FP + 1e-10)
        recall = TP / (n_gt_instances + 1e-10)

        # Prepend (recall=0, precision=1) anchor point
        recall = torch.cat([torch.zeros(1), recall], dim=0).numpy()
        precision = torch.cat([torch.ones(1), precision], dim=0).numpy()
        if remove_wiggles:
            precision = np.maximum.accumulate(precision[::-1])[::-1]
        AP = auc(recall, precision)

        # Mass-error metrics (only for predictions with finite GT mass and IoU above threshold)
        if cls_id == "all":
            key = "all"
            filtered_total_gt_mass_per_cls = np.sum(
                [m for c, m in total_gt_mass_per_cls.items() if ids_to_cls.get(c, "UNKNOWN") not in exclude]
            )
            mass_mask = torch.isfinite(sorted_gt_masses_per_pred) & (sorted_iou_per_pred > iou_threshold)
            filtered_raw_pred_mass = np.sum(
                [m for c, m in total_raw_pred_mass_per_cls.items() if c not in exclude]
            )
        else:
            key = ids_to_cls.get(cls_id, "UNKNOWN")
            filtered_total_gt_mass_per_cls = total_gt_mass_per_cls.get(cls_id, np.nan)
            filtered_raw_pred_mass = total_raw_pred_mass_per_cls.get(key, np.nan)
            mass_mask = (
                (sorted_gt_cls_labels_per_pred == cls_id)
                & torch.isfinite(sorted_gt_masses_per_pred)
                & (sorted_iou_per_pred > iou_threshold)
            )
            if mass_mask.sum() == 0:
                continue

        sorted_pred_masses_per_cls = sorted_pred_masses[mass_mask]
        sorted_gt_masses_per_cls = sorted_gt_masses_per_pred[mass_mask]

        instance_errors = torch.abs(
            sorted_pred_masses_per_cls - sorted_gt_masses_per_cls
        ) / (sorted_gt_masses_per_cls + 1e-10)
        outlier_rate_50 = (instance_errors > 0.5).float().sum() / (instance_errors.numel() + 1e-10)

        MAE = torch.abs(sorted_pred_masses_per_cls - sorted_gt_masses_per_cls).mean()
        MMAPE = torch.where(
            sorted_gt_masses_per_cls != 0,
            torch.abs(sorted_pred_masses_per_cls - sorted_gt_masses_per_cls) / sorted_gt_masses_per_cls,
            torch.zeros_like(MAE),
        ).mean()

        if cls_id != "all":
            total_pred_mass_per_cls.append(sorted_pred_masses_per_cls.sum().item())
            total_pred_masses_pos = total_pred_mass_per_cls[-1]
        else:
            total_pred_masses_pos = float(np.sum(total_pred_mass_per_cls))

        TMAPE = (total_pred_masses_pos - filtered_total_gt_mass_per_cls) / (filtered_total_gt_mass_per_cls + 1e-10)
        TME = total_pred_masses_pos - filtered_total_gt_mass_per_cls

        results[key] = {
            "AP": AP,
            "precision": precision,
            "recall": recall,
            "count": n_gt_instances,
            "scores": scores_cls.numpy(),
            "iou": iou_per_cls.numpy(),
            "TPFP": sorted_TPFP_per_pred_per_cls.numpy(),
            "MassMAPE": MMAPE.item(),
            "MassMAE": MAE.item(),
            "TotalMassMAPE": TMAPE,
            "TotalMassError": TME,
            "GTMass": filtered_total_gt_mass_per_cls,
            "RawPREDMass": filtered_raw_pred_mass,
            "PosPREDMass": total_pred_masses_pos,
            "RawREMass": (filtered_raw_pred_mass - filtered_total_gt_mass_per_cls) / (filtered_total_gt_mass_per_cls + 1e-10),
            "OutlierRate50": outlier_rate_50.item(),
        }

    # False-positive classes (predicted but absent from GT)
    gt_cls_set = set(cls_ids[:-1])  # exclude "all"
    fp_cls_ids = [
        cid for cid in all_pred_cls_labels_list.cpu().numpy().tolist()
        if cid not in gt_cls_set
    ]
    for cls_id in fp_cls_ids:
        idx = (all_pred_cls_labels_list == cls_id).nonzero(as_tuple=True)[0]
        key = ids_to_cls.get(cls_id, "UNKNOWN")
        scores = all_pred_score_list[idx]

        results[key] = {
            "AP": np.nan,
            "precision": [np.nan],
            "recall": [np.nan],
            "count": idx.numel(),
            "scores": scores.numpy(),
            "iou": [0],
            "TPFP": [np.nan],
            "MassMAPE": np.nan,
            "MassMAE": np.nan,
            "TotalMassMAPE": np.nan,
            "TotalMassError": np.nan,
            "GTMass": 0.0,
            "RawPREDMass": total_raw_pred_mass_per_cls.get(key, np.nan),
            "PosPREDMass": 0.0,
            "RawREMass": np.nan,
            "OutlierRate50": np.nan,
        }

    # Summary table
    print(
        "class |  AP  | count | Mass MAPE | Mass MAE | TotalMassMAPE | GT Mass | Pred Mass | Raw Pred Mass | Raw RE | OutlierRate"
    )
    for cls_name, val in results.items():
        print(
            f"{cls_name:5s} | {val['AP']:.3f} | {val['count']:5d} |  "
            f"{val['MassMAPE'] * 100: ^5.2f}%  | {val['MassMAE']: ^8.2f} | {val['TotalMassMAPE'] * 100: >12.2f}% "
            f"| {val['GTMass']: ^8.2f} | {val['PosPREDMass']: ^10.2f} | {val['RawPREDMass']: ^10.2f} "
            f"| {100 * val['RawREMass']: ^10.2f}% | {val['OutlierRate50'] * 100: >12.2f}"
        )

    return results, total_gt_mass_per_cls, per_image_results
