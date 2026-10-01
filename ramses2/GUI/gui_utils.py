import os
import json
from pathlib import Path
from PIL import Image
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import TextArea, AnnotationBbox
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.color import label2rgb, rgb2gray
from skimage.segmentation import find_boundaries
import pandas
import numpy as np
import torch
from ramses2 import RAMSESModel, Config

VALID_IMAGE_FORMATS = [".jpg", ".png", ".tif", ".bmp", ".jpeg"]
EN93311_CLASSES = ("Ra", "Rb", "Rc", "Rg", "Ru", "X")
CLASS_CONVERSION = {
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

# SIEVES_ALL = [0, 0.063, 0.125, 0.25, 0.50, 1, 2, 4, 8, 10, 12.5, 16, 20, 31.5, 63]
SIEVES = [0, 0.125, 0.25, 0.5, 1, 2, 4, 8, 10, 12.5, 16, 20, 31.5, 63, 100]


def compute_granulometry(dataframe, column, resolution):
    mtot = np.sum(dataframe["mass"])
    diameters = dataframe[column].to_numpy() / resolution
    h, b = np.histogram(diameters, bins=SIEVES, weights=dataframe["mass"])
    h /= mtot
    h = np.cumsum(h)
    return h, b


def load_model(checkpoint: str, device: str = "cpu"):
    """Load a PyTorch or PyTorch Lightning checkpoint into a RAMSESModel.

    Parameters
    ----------
    checkpoint : str
        Path to the model checkpoint file (.pt, .pth, or .ckpt).
    device : str, optional
        Device on which to map the loaded tensors (default is "cpu").

    Returns
    -------
    tuple
        A tuple containing:
        - model (RAMSESModel): The model with loaded weights in eval mode.
        - idx_to_cls (dict): Mapping from class index to class name.
    """
    print(f"Opening {checkpoint}")
    folder = os.path.dirname(checkpoint)

    # 1. Load model configuration
    with open(
        os.path.join(folder, "config.json"), "r", encoding="utf-8"
    ) as configfile:
        config = json.load(configfile)

    model = RAMSESModel(Config(**config))

    # 2. Load state dict safely
    checkpoint_data = torch.load(
        checkpoint, map_location=device, weights_only=True
    )

    # Detect if checkpoint is from PyTorch Lightning
    if isinstance(checkpoint_data, dict) and "state_dict" in checkpoint_data:
        state_dict_raw = checkpoint_data["state_dict"]
    elif isinstance(checkpoint_data, dict):
        state_dict_raw = checkpoint_data
    else:
        state_dict_raw = checkpoint_data

    # Remove 'model.' prefix added by LightningModule
    prefix = "model."
    state_dict_cleaned = {
        (k[len(prefix) :] if k.startswith(prefix) else k): v
        for k, v in state_dict_raw.items()
    }

    model.load_state_dict(state_dict_cleaned, strict=False)
    model.to(device)
    model.eval()

    # 3. Load class mapping
    with open(
        os.path.join(folder, "classes.json"), "r", encoding="utf-8"
    ) as classfile:
        cls_to_idx = json.load(classfile)

    idx_to_cls = {v: k for k, v in cls_to_idx.items()}

    return model, idx_to_cls

# def load_model(checkpoint):
#     print(f"Opening {checkpoint}")
#     folder = os.path.dirname(checkpoint)
#     with open(os.path.join(folder, "config.json"), "r", encoding="utf-8") as configfile:
#         config = json.load(configfile)

#     model = RAMSESModel(Config(**config))
#     state_dict = torch.load(checkpoint)
#     model.load_state_dict(state_dict, strict=False)

#     with open(os.path.join(folder, "classes.json"), "r", encoding="utf-8") as classfile:
#         cls_to_idx = json.load(classfile)
#     idx_to_cls = {v: k for k, v in cls_to_idx.items()}
#     return model, idx_to_cls


def get_img_list(input_dir):
    img_list = []
    if not os.path.exists(input_dir):
        return []
    for entry in os.scandir(input_dir):
        f = entry.name
        if entry.is_file() and os.path.splitext(f)[-1].lower() in VALID_IMAGE_FORMATS:
            img_list.append(f)

    sorted(img_list)
    return img_list


def process_predictions(results):

    results = pandas.DataFrame().from_dict(results)

    group = results.groupby("class")["mass"]
    pred_masses = group.sum()
    pred_classes = pred_masses.keys().to_list()
    ALL_summary = {
        "class": pred_classes,
        "mass(g)": pred_masses.to_numpy(),
        "mass_fraction": np.array(pred_masses / pred_masses.sum()),
        "number of elts": group.count().to_numpy(),
    }

    def map_func(x):
        return CLASS_CONVERSION.get(x, x)

    EN93311_summary = results.map(map_func)
    group = EN93311_summary.groupby("class")["mass"]
    pred_masses = group.sum()
    pred_classes = pred_masses.keys().to_list()
    EN93311_summary = {
        "class": pred_classes,
        "mass(g)": pred_masses.to_numpy(),
        "mass_fraction": np.array(pred_masses / pred_masses.sum()),
        "number of elts": group.count().to_numpy(),
    }
    return results, ALL_summary, EN93311_summary
