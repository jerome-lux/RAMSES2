# coding: utf-8

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import auc
from ..train import collate_fn


@torch.no_grad()
def evaluate(
    model,
    dataset,
    annotations,
    maxiter,
    ids_to_cls,
    iou_threshold=0.75,  # iou threshold for matching GT and predictions
    mask_stride=4,
    cls_threshold=0.45,
    nms_threshold=0.5,  # iou threshold for greedy and soft NMS, or class threshold in MatrixNMX
    mask_threshold=0.5,
    max_detections=768,
    scale_by_mask_scores=False,
    min_area=128,
    nms_mode="greedy",
    exclude=[],
    verbose=False,
    remove_wiggles=True,
    device="cuda:0",
    shuffle=False,
    **kwargs,
):
    """Compute AP score and mAP
    inputs:
    model: RAMSES2 model
    dataset:  an instance of torch.Dataset
    annotations: a pandas DataFrame containing the data for each instance and image
    scaling: image downsampling ratio (input_shape / original_imshape)
    maxiter: number of images used
    threshold: iou threshold
    remove_wiggles: remove wiggles in the PR curve before computing AUC

    outputs:
    AP
    sorted scores
    precision
    interpolated precision
    recall

    """
    model = model.to(device)
    model.eval()

    # Get the resolutions for each image
    unique_indexes = annotations.reset_index().groupby(["baseimg"])["index"].min().to_list()
    resolutions = {annotations.iloc[i]["baseimg"]: annotations.iloc[i]["res"] for i in unique_indexes}
    heights = {annotations.iloc[i]["baseimg"]: annotations.iloc[i]["height"] for i in unique_indexes}
    widths = {annotations.iloc[i]["baseimg"]: annotations.iloc[i]["width"] for i in unique_indexes}

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
    # list to keep the unmatched prediction
    all_pred_score_list = []
    all_pred_cls_labels_list = []

    # To save results per image
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
        gt_labels = inputs["label"][0]
        gt_mass = inputs["mass"][0]
        mask_res = inputs["res"][0][0]
        gt_cls_labels = [ids_to_cls[id] for id in list(gt_cls_ids.numpy())]
        nx, ny = gt_img.shape[-2:]

        ratio = (nx // mask_stride) / heights[imgname]
        scaling = (10 * resolutions[imgname] * ratio) ** 2

        # Get the total gt mass of each class in the image and update total mass per class dict
        cls_ids_np = gt_cls_ids.numpy()
        isfinite_gt_mass = torch.isfinite(gt_mass)
        for j in range(cls_ids_np.shape[-1]):
            try:
                if isfinite_gt_mass[j]:
                    total_gt_mass_per_cls[cls_ids_np[j]] = (
                        total_gt_mass_per_cls.get(cls_ids_np[j], 0.0) + gt_mass[j].item() / scaling
                    )
            except Exception as e:
                print("Error", e)

        ngt = gt_cls_ids.shape[0]
        gt_accumulator += ngt

        results = model(
            gt_img,
            training=False,
            cls_threshold=cls_threshold,
            nms_threshold=nms_threshold,  # iou threshold or cls threshold in MatrixNMX
            mask_threshold=mask_threshold,
            max_detections=max_detections,
            scale_by_mask_scores=scale_by_mask_scores,
            min_area=min_area,
            nms_mode=nms_mode,
        )[0]

        # processed_masks = ramses2.decode_predictions(results['masks'], results["scores"], threshold=0.5, by_mask_scores=False)

        pred_masks = results["masks"].detach()  # [Npred, H, W]
        pred_masses = results["masses"].detach()
        pred_cls_ids = results["cls_labels"].detach() + 1
        pred_scores = results["scores"].detach()

        # No predcitions
        if pred_cls_ids.shape[0] == 0:
            print(f"{i+1}/{maxiter} Processing image {imgname} containing {ngt} objets. " f"No predictions!")
            gt_cls_labels_list.append(gt_cls_ids)
            continue

        # Raw predictions (for all positive/negative detections) vs gt mass

        for j in range(pred_masses.shape[0]):
            cls_id = pred_cls_ids[j].item()
            cls_name = ids_to_cls.get(cls_id, "UNKNOWN")
            if cls_name != "UNKNOWN":
                total_raw_pred_mass_per_cls[cls_name] = (
                    total_raw_pred_mass_per_cls.get(cls_name, 0.0) + (pred_masses[j].item() / scaling)
                )


        gt_masks = F.one_hot(gt_mask_img, ngt + 1)[..., 1:]
        gt_masks = torch.reshape(gt_masks, (-1, ngt))
        gt_masks = torch.transpose(gt_masks, 1, 0).float()  # [Ngt, H*W]

        pred_masks = torch.where(pred_masks > mask_threshold, 1, 0)
        pred_masks = torch.reshape(pred_masks, (pred_masks.shape[0], -1)).float()  # [Npred, H*W]

        total_pred_mass_per_cls_np = pred_masses / scaling

        per_image_results[imgname] = {
            "gt_masses": {name: 0.0 for name in ids_to_cls.values()},
            "pred_masses": {name: 0.0 for name in ids_to_cls.values()},
            "total_pred_mass": pred_masses.sum().item() / scaling,
            "total_gt_mass": gt_mass.sum().item() / scaling,
            "ngt": gt_cls_ids.size(0),
            "npred": pred_cls_ids.size(0),
            "RE": (pred_masses.sum().item() - gt_mass.sum().item()) / gt_mass.sum().item()
        }

        for j in range(len(gt_cls_ids)):
            cls_name = ids_to_cls.get(gt_cls_ids[j].item(), "UNKNOWN")
            if cls_name in per_image_results[imgname]["gt_masses"]:
                per_image_results[imgname]["gt_masses"][cls_name] += (gt_mass[j].item() / scaling)

        for j in range(len(pred_cls_ids)):
            cls_name = ids_to_cls.get(pred_cls_ids[j].item(), "UNKNOWN")
            # On ajoute même si la classe est "UNKNOWN" (ou une classe prédite non présente en GT)
            if cls_name not in per_image_results[imgname]["pred_masses"]:
                per_image_results[imgname]["pred_masses"][cls_name] = 0.0
            per_image_results[imgname]["pred_masses"][cls_name] += (pred_masses[j].item() / scaling)

        cls_unique = np.unique(cls_ids_np)
        cls_unique = [ids_to_cls.get(c, "UNKNOWN") for c in cls_unique]
        pred_cls_unique = np.unique(pred_cls_ids.cpu().numpy())
        pred_cls_unique = [ids_to_cls.get(c, "UNKNOWN") for c in pred_cls_unique]


        if verbose:
            # print(" "*500, end="\r")
            print(
                f"{i+1}/{maxiter} Processing image {imgname} containing {ngt} objets. "
                f"Detected : {pred_masks.shape[0]}, gt mass: {gt_mass.sum()/scaling:5.2f}, pred mass: {total_pred_mass_per_cls_np.sum().item():5.2f} "
                f"in classes {cls_unique}, pred classes {pred_cls_unique}. res: {resolutions[imgname]} pix/mm,"
                f"mask res: {resolutions[imgname] * ratio:5.2f} ({mask_res:5.2f})",
                flush=True,
            )
        else:
            print(f"Iteration {i+1:>5d}/{maxiter} ", end="\r")

        # calcul de la matrice des ious entre les gt et les pred masks
        intersections = torch.matmul(gt_masks, pred_masks.T)
        gt_sums = gt_masks.sum(dim=1)  # Mask area
        pred_sums = pred_masks.sum(dim=1)
        # Here the output is broadcasted to [Ngt, Npred]
        unions = gt_sums.unsqueeze(1) + pred_sums.unsqueeze(0) - intersections
        # iou représente la matrice des ious entre GT et PRED -> [Ngt, Npred]
        iou = intersections / (unions + 1e-6)

        max_iou_per_pred, gt_indices_for_each_pred = torch.max(iou, dim=0)
        final_gt_matches = -torch.ones(pred_masks.shape[0], dtype=torch.long)
        final_iou_values = torch.zeros(pred_masks.shape[0], dtype=torch.float32)

        # Gestion les doublons : Il faut s'assurer qu'un GT n'est attribué qu'une seule fois.
        candidate_matches = []
        for pred_idx in range(pred_masks.shape[0]):
            candidate_matches.append(
                (max_iou_per_pred[pred_idx].item(), pred_idx, gt_indices_for_each_pred[pred_idx].item())
            )

        # Trie les appariements potentiels par IoU décroissant
        # Le premier élément de chaque tuple est l'IoU
        candidate_matches.sort(key=lambda x: x[0], reverse=True)

        # Suivre quels GT ont déjà été attribués
        assigned_gt_masks = torch.zeros(gt_masks.shape[0], dtype=torch.bool)  # False si non assigné, True si assigné

        # Parcourir les appariements triés et attribuer les GT
        for iou_val, pred_idx, gt_idx in candidate_matches:
            # Si le masque GT n'a pas encore été attribué
            if not assigned_gt_masks[gt_idx]:
                final_gt_matches[pred_idx] = gt_idx
                final_iou_values[pred_idx] = iou_val
                assigned_gt_masks[gt_idx] = True  # Marquer ce GT comme assigné

        # Filtrer le tenseur des prédictions pour ne garder que celles qui ont un match valide
        valid_pred_indices = torch.where(final_gt_matches != -1)[0]
        if valid_pred_indices.numel() == 0:
            gt_cls_labels_list.append(gt_cls_ids)
            print("No matching predictions for this image.")
            continue
        # Obtenir les indices des GT correspondants pour ces prédictions valides
        valid_gt_indices_to_gather = final_gt_matches[valid_pred_indices]

        TPFP_per_pred.append(
            torch.where(final_iou_values > iou_threshold, 1, 0)
        )  # matching masks (class may not match)
        iou_per_pred.append(final_iou_values)
        # On doit filtrer les tenseurs pour ne garder que les prédictions valides (matching masks)
        gt_cls_labels_per_pred.append(torch.gather(gt_cls_ids, 0, valid_gt_indices_to_gather))
        pred_cls_labels_list.append(torch.gather(pred_cls_ids.cpu(), 0, valid_pred_indices))
        pred_scores_list.append(torch.gather(pred_scores.cpu(), 0, valid_pred_indices))
        pred_masses_list.append(torch.gather(pred_masses.cpu(), 0, valid_pred_indices) / scaling)
        gt_cls_labels_list.append(gt_cls_ids)
        gt_mass_per_pred.append(torch.gather(gt_mass, 0, valid_gt_indices_to_gather) / scaling)
        all_pred_score_list.append(pred_scores.cpu())
        all_pred_cls_labels_list.append(pred_cls_ids)

    TPFP_per_pred = torch.concat(TPFP_per_pred, dim=0)
    pred_cls_labels_list = torch.concat(pred_cls_labels_list, dim=0)
    pred_scores_list = torch.concat(pred_scores_list, dim=0)
    pred_masses_list = torch.concat(pred_masses_list, dim=0)
    gt_cls_labels_per_pred = torch.concat(gt_cls_labels_per_pred, dim=0)
    gt_cls_labels_list = torch.concat(gt_cls_labels_list, dim=0)
    gt_mass_per_pred = torch.concat(gt_mass_per_pred, dim=0)
    iou_per_pred = torch.concat(iou_per_pred, dim=0)
    all_pred_score_list = torch.concat(all_pred_score_list, dim=0)
    all_pred_cls_labels_list = torch.concat(all_pred_cls_labels_list, dim=0)
    print("")

    # Sort predictions by score
    sorted_scores, sorted_pred_indices = torch.sort(pred_scores_list.to("cpu"), descending=True)
    sorted_iou_per_pred = iou_per_pred[sorted_pred_indices]
    sorted_TPFP_per_pred = TPFP_per_pred[sorted_pred_indices]
    sorted_pred_cls_labels = pred_cls_labels_list[sorted_pred_indices]
    sorted_gt_cls_labels_per_pred = gt_cls_labels_per_pred[sorted_pred_indices]
    sorted_pred_masses = pred_masses_list[sorted_pred_indices]
    sorted_gt_masses_per_pred = gt_mass_per_pred[sorted_pred_indices]

    # We check if matched masks have the same class to get the True Positives and False Positives
    # cls_matching_idx = torch.where(sorted_gt_cls_labels_per_pred == sorted_pred_cls_labels)
    sorted_TPFP_per_pred = torch.where(sorted_gt_cls_labels_per_pred == sorted_pred_cls_labels, sorted_TPFP_per_pred, 0)

    # Total number of predictions over samples
    npred = sorted_scores.shape[0]

    # list of all cls ids in gt dataset
    cls_ids, gt_count_per_cls = torch.unique(gt_cls_labels_list, return_counts=True)
    cls_ids = cls_ids.cpu().numpy().tolist()
    cls_ids.append("all")  # used to compute the overall AP
    results = {}
    exclude.append("UNKNOWN")  # we do not process the UNKNOWN class
    total_pred_mass_per_cls = []

    # AP per class
    for i, cls_id in enumerate(cls_ids):

        if ids_to_cls.get(cls_id, "UNKNOWN") in exclude and cls_id != "all":
            print(f"Skipping class {ids_to_cls.get(cls_id, 'UNKNOWN')} in exclude list")
            continue

        # Filter by class
        if cls_id == "all":
            n_gt_instances = gt_cls_labels_list.numel()
            iou_per_cls = sorted_iou_per_pred
            sorted_TPFP_per_pred_per_cls = sorted_TPFP_per_pred
            scores_cls = sorted_scores
        else:
            n_gt_instances = gt_count_per_cls[i].item()
            indexes = torch.where(sorted_pred_cls_labels == cls_id)
            if indexes[0].numel() == 0:
                print("No instance of class", cls_id, "found in predictions")
                continue
            sorted_TPFP_per_pred_per_cls = sorted_TPFP_per_pred[indexes]
            iou_per_cls = sorted_iou_per_pred[indexes]
            scores_cls = sorted_scores[indexes]

        TP = torch.cumsum(sorted_TPFP_per_pred_per_cls, dim=0)
        FP = torch.cumsum(1.0 - sorted_TPFP_per_pred_per_cls, dim=0)
        precision = TP / (TP + FP + 1e-10)
        recall = TP / (n_gt_instances + 1e-10)

        # Add 1 and 0 to precision and 0 and 1 to recall... is it really needed?
        recall = torch.cat([torch.zeros(1, device=recall.device), recall], dim=0).cpu().numpy()
        precision = torch.cat([torch.ones(1, device=precision.device), precision], dim=0).cpu().numpy()
        if remove_wiggles:
            precision = np.maximum.accumulate(precision[::-1])[::-1]
        AP = auc(recall, precision)

        # Mass error MAPE and MAE
        # Here, we discard objects without gt mass
        if cls_id == "all":
            key = "all"
            filtered_total_gt_mass_per_cls = np.sum(
                [m for c, m in total_gt_mass_per_cls.items() if ids_to_cls.get(c, "UNKNOWN") not in exclude]
            )
            # mass_indexes = torch.where(torch.isfinite(sorted_gt_masses_per_pred))
            mass_mask = torch.isfinite(sorted_gt_masses_per_pred) & (sorted_iou_per_pred > iou_threshold)
            filtered_raw_pred_mass = np.sum(
                [m for c, m in total_raw_pred_mass_per_cls.items() if c not in exclude]
            )
            #
        else:
            key = ids_to_cls.get(cls_id, "UNKNOWN")
            filtered_total_gt_mass_per_cls = total_gt_mass_per_cls.get(cls_id, np.nan)
            # mass_indexes = torch.where(
            #     (sorted_gt_cls_labels_per_pred == cls_id) & (torch.isfinite(sorted_gt_masses_per_pred))
            # )
            filtered_raw_pred_mass = total_raw_pred_mass_per_cls.get(key, np.nan)
            mass_mask = (
                (sorted_gt_cls_labels_per_pred == cls_id)
                & torch.isfinite(sorted_gt_masses_per_pred)
                & (sorted_iou_per_pred > iou_threshold)
            )
            if mass_mask.sum() == 0:
                continue
            # if mass_indexes[0].numel() == 0:
            #     continue

        sorted_pred_masses_per_cls = sorted_pred_masses[mass_mask]
        sorted_gt_masses_per_pred_per_cls = sorted_gt_masses_per_pred[mass_mask]

        instance_errors = torch.abs(sorted_pred_masses_per_cls - sorted_gt_masses_per_pred_per_cls) / (sorted_gt_masses_per_pred_per_cls + 1e-10)
        outliers_50 = (instance_errors > 0.5).float().sum()
        outlier_rate_50 = outliers_50 / (instance_errors.numel() + 1e-10)

        MAE = torch.abs(sorted_pred_masses_per_cls - sorted_gt_masses_per_pred_per_cls)
        MMAPE = torch.where(
            sorted_gt_masses_per_pred_per_cls != 0, MAE / sorted_gt_masses_per_pred_per_cls, torch.zeros_like(MAE)
        )
        MMAPE = MMAPE.mean()
        MAE = MAE.mean()

        if cls_id != "all":
            total_pred_mass_per_cls.append(sorted_pred_masses_per_cls.sum().item())
            total_pred_masses_pos = total_pred_mass_per_cls[-1]
        else:
            total_pred_masses_pos = np.sum(total_pred_mass_per_cls)

        TMAPE = (total_pred_masses_pos - filtered_total_gt_mass_per_cls) / (filtered_total_gt_mass_per_cls + 1e-10)
        TME = total_pred_masses_pos - filtered_total_gt_mass_per_cls

        results[key] = {
            "AP": AP,
            "precision": precision,
            "recall": recall,
            "count": n_gt_instances,
            "scores": scores_cls.cpu().numpy(),
            "iou": iou_per_cls.cpu().numpy(),
            "TPFP": sorted_TPFP_per_pred_per_cls.cpu().numpy(),
            "MassMAPE": MMAPE.item(),
            "MassMAE": MAE.item(),
            "TotalMassMAPE": TMAPE,
            "TotalMassError": TME,
            "GTMass": filtered_total_gt_mass_per_cls,
            "RawPREDMass": filtered_raw_pred_mass,
            "PosPREDMass": total_pred_masses_pos,
            "RawREMass": (filtered_raw_pred_mass - filtered_total_gt_mass_per_cls) / (filtered_total_gt_mass_per_cls + 1e-10),
            "OutlierRate50": outlier_rate_50.item()
        }

    # Now process the predictions where the class is not in the GT classes
    fp_cls_ids = [cls_id for cls_id in all_pred_cls_labels_list.detach().cpu().numpy().tolist() if cls_id not in cls_ids]
    for cls_id in fp_cls_ids:
        indexes = torch.where(all_pred_cls_labels_list.cpu() == cls_id)
        key = ids_to_cls.get(cls_id, "UNKNOWN")
        scores = all_pred_score_list[indexes]

        results[key] = {
            "AP": np.nan,
            "precision": [np.nan],
            "recall": [np.nan],
            "count": indexes[0].numel(),
            "scores": scores.cpu().numpy(),
            "iou": [0],
            "TPFP": [np.nan],
            "MassMAPE": np.nan,
            "MassMAE": np.nan,
            "TotalMassMAPE": np.nan,
            "TotalMassError": np.nan,
            "GTMass": 0.,
            "RawPREDMass": total_raw_pred_mass_per_cls.get(key, np.nan),
            "PosPREDMass": 0.,
            "RawREMass": np.nan,
            "OutlierRate50": np.nan
        }

    print("class |  AP  | count | Mass MAPE | Mass MAE | TotalMassMAPE | GT Mass | Pred Mass | Raw Pred Mass | Raw RE | OutlierRate")
    for cls_id, val in results.items():
        print(
            f"{cls_id:5s} | {val['AP']:.3f} | {val['count']:5d} |  "
            f"{val['MassMAPE']*100: ^5.2f}%  | {val['MassMAE']: ^8.2f} | {val['TotalMassMAPE']*100: >12.2f}% "
            f"| {val['GTMass']: ^8.2f} | {val['PosPREDMass']: ^10.2f} | {val['RawPREDMass']: ^10.2f} "
            f"| {100*val['RawREMass']: ^10.2f}% | {val["OutlierRate50"]*100: >12.2f}"
        )

    return results, total_gt_mass_per_cls, per_image_results
