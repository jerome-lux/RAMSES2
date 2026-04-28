from typing import Union, Sequence
import torch
import torchvision.transforms.functional as TF
import numpy as np
from . import utils
from .utils import make_tuple

class TorchAugmentations:
    """
    Augmentations for torch pipelines (brightness, hue, saturation, noise, rotation).
    """

    def __init__(self, probability: Union[Sequence, float] = 0.5,
                 seed: Union[int, None] = None,
                 brightness_factor=(-.05, 0.05),
                 hue_factor = (-0.1, 0.1),
                 saturation_factor = (0.9, 1.1),
                 rotation_factor: float = 1.0):
        # Increased to 5 to include hue and saturation
        self.probability = make_tuple(probability, 5, fill_value=0)
        self.rotation_factor = rotation_factor
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.saturation_factor = saturation_factor
        self.hue_factor = hue_factor
        self.brightness_factor = brightness_factor

    def __call__(self, sample):
        img = sample["image"]
        masks = sample["masks"]
        cat_ids = sample["category_id"]
        labels = sample["label"]
        mass = sample["mass"]
        res = sample["res"]
        basename = sample.get("filename", None)

        # 1. Random brightness
        if self.rng.uniform() < self.probability[0]:
            img = TF.adjust_brightness(img, 1.0 + self.rng.uniform(*self.brightness_factor))
            img = torch.clamp(img, 0.0, 1.0)

        # 2. Random hue variation
        if self.rng.uniform() < self.probability[1]:
            hue_factor_values = float(self.rng.uniform(*self.hue_factor))
            img = TF.adjust_hue(img, hue_factor_values)
            img = torch.clamp(img, 0.0, 1.0)

        # 3. Random light saturation variation
        if self.rng.uniform() < self.probability[2]:
            # saturation change (0.9 to 1.1)
            saturation_factor_values = float(self.rng.uniform(*self.saturation_factor))
            img = TF.adjust_saturation(img, saturation_factor_values)
            img = torch.clamp(img, 0.0, 1.0)

        # 4. Gaussian noise
        if self.rng.uniform() < self.probability[3]:
            noise = torch.randn_like(img) * 0.05
            img = torch.clamp(img, 0.0, 1.0)

        img = torch.clamp(img, 0.0, 1.0)

        # 5. Random rotation
        if self.rng.uniform() < self.probability[4]:
            angle = float(self.rng.uniform(-self.rotation_factor * 180, self.rotation_factor * 180))
            # Image rotation
            new_img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
            new_img = torch.clamp(img, 0.0, 1.0)
            # Mask rotation
            masks_rot = (
                TF.rotate(masks.unsqueeze(0).float(), angle, interpolation=TF.InterpolationMode.NEAREST, fill=0)
                .squeeze(0)
                .long()
            )
            # Consistency filtering
            masks_rot, new_labels, filtered_tensors = utils.relabel_and_filter(masks_rot, labels, cat_ids, mass, res)
            # if no instance in the rotated image then return the non rotated image
            if new_labels.numel() == 0:
                sample["image"] = torch.nan_to_num(img, nan=0.0)
                return sample

            cat_ids, mass, res = filtered_tensors
            labels = new_labels
            masks = masks_rot

        else:
            new_img = img

        return {
            "filename": basename, "image": torch.nan_to_num(new_img, nan=0.0), "masks": masks,
            "category_id": cat_ids, "label": labels, "mass": mass, "res": res,
        }