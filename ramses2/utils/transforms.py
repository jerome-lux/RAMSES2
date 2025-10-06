from typing import Union, Sequence
import torch
import torchvision.transforms.functional as TF
import numpy as np
from . import utils
from .utils import make_tuple


class TorchAugmentations:
    """
    Augmentations for use in torchDataset and DataLoader pipelines.

    This class applies random brightness, gaussian noise, and random rotation to images and masks.
    After rotation, it relabels and filters the mask and associated label tensors to ensure consistency.

    Args:
        probability (float or sequence of float): Probability for each augmentation (brightness, noise, rotation).
            Can be a single float or a sequence of 3 floats. If a single float, the same probability is used for all.
        seed (int, optional): Random seed for reproducibility.
        factor (float): Maximum rotation factor (as a fraction of 180 degrees).

    Example:
        >>> transform = TorchAugmentations(probability=[0.5, 0.2, 0.3], seed=42)
        >>> sample = transform(sample)
    """

    def __init__(self, probability: Union[Sequence, float] = 0.5, seed: Union[int, None] = None, factor: float = 1.0):
        self.probability = make_tuple(probability, 3, fill_value=0)
        self.factor = factor
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, sample):
        # sample: dict with keys 'image', 'masks', 'category_id', 'label', 'mass', 'res', 'filename'
        img = sample["image"]  # [C, H, W], float32
        masks = sample["masks"]  # [H, W], int64
        cat_ids = sample["category_id"]
        labels = sample["label"]
        mass = sample["mass"]
        res = sample["res"]
        basename = sample.get("filename", None)

        # Random brightness
        if self.rng.uniform() < self.probability[0]:
            img = TF.adjust_brightness(img, 1.0 + self.rng.uniform(-0.05, 0.05))
            img = torch.clamp(img, 0.0, 1.0)
        # Gaussian noise
        if self.rng.uniform() < self.probability[1]:
            noise = torch.randn_like(img) * 0.05
            img = torch.clamp(img + noise, 0.0, 1.0)
        # Random rotation (image + mask + labels)
        if self.rng.uniform() < self.probability[2]:
            angle = float(self.rng.uniform(-self.factor * 180, self.factor * 180))
            # Rotate image (bilinear)
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            # Rotate mask (nearest)
            masks_rot = (
                TF.rotate(masks.unsqueeze(0).float(), angle, interpolation=TF.InterpolationMode.NEAREST, fill=0)
                .squeeze(0)
                .long()
            )
            # Relabel and filter tensors
            masks_rot, new_labels, filtered_tensors = utils.relabel_and_filter(masks_rot, labels, cat_ids, mass, res)
            if new_labels.numel() == 0:
                # No objects left after rotation
                return {**sample, "image": img}
            # Update tensors after relabeling
            cat_ids, mass, res = filtered_tensors
            labels = new_labels
            masks = masks_rot
        # Return updated sample
        return {
            "filename": basename,
            "image": img,
            "masks": masks,
            "category_id": cat_ids,
            "label": labels,
            "mass": mass,
            "res": res,
        }
