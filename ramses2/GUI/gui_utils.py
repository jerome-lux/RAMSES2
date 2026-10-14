"""Utilities shared by the RAMSES2 GUI.

This module provides the non-visual helpers used by :mod:`ramses2.GUI.gui`:

    * :func:`load_model`           -- load a RAMSES2 checkpoint into a model.
    * :func:`get_img_list`        -- list the displayable images in a folder.
    * :func:`compute_granulometry`-- compute a cumulative mass granulometric curve.
    * :func:`process_predictions` -- aggregate per-instance predictions into summaries.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas
import torch

from ramses2 import Config, RAMSESModel

#: File extensions considered displayable images.
VALID_IMAGE_FORMATS: list[str] = [".jpg", ".png", ".tif", ".bmp", ".jpeg"]

#: Mapping from fine-grained model class names to EN 933-11 super-classes.
CLASS_CONVERSION: dict[str, str] = {
    "Ra": "Ra",
    "Rb01": "Rb",
    "Rb02": "Rb",
    "Rc": "Rc",
    "Rcu01": "Rc",
    "Ru01": "Ru",
    "Ru02": "Ru",
    "Ru03": "Ru",
    "Ru04": "Ru",
    "Ru05": "Ru",
    "Ru06": "Ru",
    "Rg": "Rg",
    "X01": "X",
    "X02": "X",
    "X03": "X",
    "X04": "X",
    "UNKNOWN": "X",
    "Coin": "X",
}

#: Sieve openings (mm) used for the granulometric curve.
SIEVES: list[float] = [0, 0.125, 0.25, 0.5, 1, 2, 4, 8, 10, 12.5, 16, 20, 31.5, 63, 100]


def compute_granulometry(dataframe: pandas.DataFrame, column: str, resolution: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute the cumulative mass granulometric curve.

    Args:
        dataframe: Per-instance results with a ``mass`` column and a diameter column.
        column: Name of the diameter column (in pixels) used to size the aggregates.
        resolution: Pixel resolution of the images (pixels per millimetre).

    Returns:
        A tuple ``(curve, bins)`` where ``curve`` is the cumulative mass fraction
        (length ``len(SIEVES) - 1``) and ``bins`` are the sieve bin edges (mm).
    """
    mass = dataframe["mass"].to_numpy(dtype=float)
    total = mass.sum()
    if total <= 0:
        return np.zeros(len(SIEVES) - 1), np.asarray(SIEVES)

    diameters = dataframe[column].to_numpy(dtype=float) / resolution
    hist, bins = np.histogram(diameters, bins=SIEVES, weights=mass)
    hist = np.cumsum(hist / total)
    return hist, bins


def load_model(checkpoint: str | Path, device: str = "cpu") -> tuple[RAMSESModel, dict[int, str]]:
    """Load a RAMSES2 checkpoint (PyTorch or Lightning) into a model.

    The model configuration and class mapping are read from the ``config.json``
    and ``classes.json`` files located next to the checkpoint.

    Args:
        checkpoint: Path to the model checkpoint file (``.pt``, ``.pth`` or ``.ckpt``).
        device: Device on which to map the loaded tensors (default ``"cpu"``).

    Returns:
        A tuple ``(model, idx_to_cls)`` where ``model`` is a RAMSESModel in eval
        mode and ``idx_to_cls`` maps a 1-based class index to its name.
    """
    print(f"Opening {checkpoint}")
    folder = os.path.dirname(str(checkpoint))

    with open(os.path.join(folder, "config.json"), "r", encoding="utf-8") as configfile:
        config = json.load(configfile)
    model = RAMSESModel(Config(**config))

    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=True)
    # Support both Lightning checkpoints ({"state_dict": ...}) and plain state dicts.
    if isinstance(checkpoint_data, dict) and "state_dict" in checkpoint_data:
        state_dict_raw = checkpoint_data["state_dict"]
    elif isinstance(checkpoint_data, dict):
        state_dict_raw = checkpoint_data
    else:
        state_dict_raw = checkpoint_data

    # Remove the "model." prefix added by the Lightning module wrapper.
    prefix = "model."
    state_dict = {
        (key[len(prefix):] if key.startswith(prefix) else key): value
        for key, value in state_dict_raw.items()
    }

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    with open(os.path.join(folder, "classes.json"), "r", encoding="utf-8") as classfile:
        cls_to_idx = json.load(classfile)
    idx_to_cls = {value: key for key, value in cls_to_idx.items()}

    return model, idx_to_cls


def get_img_list(input_dir: str) -> list[str]:
    """Return the sorted list of displayable image filenames in a folder.

    Args:
        input_dir: Directory to scan.

    Returns:
        A lexicographically sorted list of image filenames, or an empty list if
        the directory does not exist or contains no valid image.
    """
    if not os.path.isdir(input_dir):
        return []
    names = [
        entry.name
        for entry in os.scandir(input_dir)
        if entry.is_file() and os.path.splitext(entry.name)[-1].lower() in VALID_IMAGE_FORMATS
    ]
    names.sort()
    return names


def _summarize_by_class(frame: pandas.DataFrame) -> dict:
    """Aggregate a predictions frame into per-class mass / fraction / count.

    Args:
        frame: A DataFrame with at least a ``class`` and a ``mass`` column.

    Returns:
        A dictionary with keys ``class``, ``mass(g)``, ``mass_fraction`` and
        ``number of elts``.
    """
    grouped = frame.groupby("class")["mass"]
    pred_masses = grouped.sum()
    total = pred_masses.sum()
    fractions = (pred_masses / total).to_numpy() if total else np.zeros(len(pred_masses))
    return {
        "class": list(pred_masses.keys()),
        "mass(g)": pred_masses.to_numpy(),
        "mass_fraction": fractions,
        "number of elts": grouped.count().to_numpy(),
    }


def process_predictions(results: dict) -> tuple[pandas.DataFrame, dict, dict]:
    """Aggregate raw per-instance predictions into two summaries.

    Args:
        results: The ``data`` dictionary returned by :func:`ramses2.inference.predict`.

    Returns:
        A tuple ``(annotations, all_summary, en933_summary)`` where
        ``annotations`` is the per-instance DataFrame, ``all_summary`` keeps the
        fine-grained classes and ``en933_summary`` groups them into EN 933-11
        super-classes.
    """
    annotations = pandas.DataFrame().from_dict(results)

    all_summary = _summarize_by_class(annotations)

    en933 = annotations.copy()
    en933["class"] = en933["class"].map(lambda name: CLASS_CONVERSION.get(name, name))
    en933_summary = _summarize_by_class(en933)

    return annotations, all_summary, en933_summary
