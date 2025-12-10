# coding=utf-8

import os
import argparse
import numpy as np
import pandas
import json


VALID_IMAGE_FORMATS = [".jpg", ".png", ".tif", ".bmp"]

parser = argparse.ArgumentParser(
    description=(
        "Compute mass of individual aggregates using metadata.json and update annotations.csv"
        "See https://doi.org/10.1016/j.compind.2023.103889 and https://doi.org/10.1016/j.autcon.2020.103204"
    ),
    fromfile_prefix_chars="@",
)
parser.add_argument(
    "-i",
    "--input",
    dest="input_dir",
    default=os.getcwd(),
    help="input directory, root dir where the json files are stored. Default: current working directory",
)
parser.add_argument(
    "--ann",
    dest="ann",
    default="annotations.csv",
    help="name of the annotation file in dataset folder which will be updated",
)
parser.add_argument(
    "--data",
    dest="data",
    default="metadata.json",
    help="name of the file in dataset folder which contains mass and resolution data",
)

args = parser.parse_args()


def update_mass(**kwargs):
    """
    Compute mass of aggregates using metadata.json and update annotations.csv file
    metadata.json contains mass and resolutions of each class in series of photos:
    {"images":[{
                "file_name": "OVERLAP_P20241007_00008-P20241007_00009.jpg",
                "res": 28.7,
                "height": 4096,
                "width": 6144,
                "camera": "JAI SW-8000Q",
                "date": "2024-10-09",
                "tag": "CONVEYOR"
            },],
    "batches":[{"image_names":[file1.jpg,file2.jpg,...],
                "mass":{"XX":111,"YY":222}}]
    }
    Total mass is dispatched based on the radius and area of each aggregate
    """
    db_path = kwargs.get("input_dir", os.getcwd())

    # Load annotations file
    ann_fn = os.path.join(db_path, kwargs.get("ann", "annotations.csv"))
    annotations = pandas.read_csv(ann_fn, sep=None, engine="python")
    annotations.to_csv(os.path.join(db_path, "backup-annotations.csv"), index=False)
    # Create mass column if it does not exists and set values to "nan"
    if "mass" not in annotations:
        annotations["mass"] = "nan"
        annotations.to_csv(os.path.join(db_path, "annotations.csv"), index=False)

    # if "F" not in annotations:
    annotations["F"] = ""

    if "gt_mass" in annotations:
        # If gt_mass is True, then the mass should be already in the "mass" column and should not be recomputed
        annotations.loc[annotations["gt_mass"] != True, "mass"] = "nan"

    elif "gt_mass" not in annotations:
        print("No individual mass data. Masses will be estimated using batch data")
        annotations["gt_mass"] = False
        annotations["mass"] = "nan"

    # Load mass and resolutions data
    with open(os.path.join(db_path, kwargs.get("data", "metadata.json")), "r") as metadatafile:
        metadata = json.load(metadatafile)

    # Get resolutions
    # res_dict = {}
    # for imdata in metadata["images"]:
    #     res_dict[imdata["file_name"]] = imdata["res"]
    nbatches = len(metadata["batches"])
    # Iterate batches
    for i, batch in enumerate(metadata["batches"]):
        print("Processing batch {}/{} ".format(i + 1, nbatches), end="")
        # Get subset of annotations
        imlist = [os.path.splitext(os.path.basename(imname))[0] for imname in batch["image_names"]]
        # On ne calcule que les masses qui ne sont pas mesurées individuellement
        ann_subset = annotations.loc[(annotations["baseimg"].isin(imlist)) & (annotations["gt_mass"] is not True)]
        if ann_subset.empty:
            print(
                f"containing {len(imlist)} image(s) ({imlist[0]}->{imlist[-1]}). No need to recompute mass, as it is GT mass"
            )
            annotations.loc[ann_subset.index, "mass"] = "nan"
            res = np.array(ann_subset["res"])
            annotations.loc[ann_subset.index, "res"] = res
            annotations.loc[ann_subset.index, "F"] = "nan"
            continue
        mass_per_cls = batch["mass"]
        if len(mass_per_cls) == 0:
            print(f"containing {len(imlist)} image(s) ({imlist[0]}->{imlist[-1]}) with no mass data.")
        if len(imlist) == 1:
            print(f"containing {len(imlist)} image ({imlist[0]}). Mass:{mass_per_cls}")
        else:
            print(f"containing {len(imlist)} images ({imlist[0]}->{imlist[-1]}). Mass:{mass_per_cls}")

        # iterate classes with non zero mass (if zero but there are objects in this class, set their masses to "nan")
        masses = 0
        F = 0
        for cls, mass in mass_per_cls.items():
            cls_subset = ann_subset[ann_subset["class"] == cls]
            if len(cls_subset) <= 0:
                continue
            areas = np.array(cls_subset["area"])
            radii = np.array(cls_subset["radius"])
            res = np.array(cls_subset["res"])
            if mass > 0 and np.isfinite(mass):
                # Compute the density - shape factor in batch
                F = mass / np.sum(areas * 2 * radii / res**3)
                # Compute mass for each object
                masses = F * areas * 2 * radii / res**3
                masses = np.where(np.isfinite(masses), masses, np.nan)
            else:
                masses = np.zeros((areas.size)) + np.nan
                F = np.nan
            # update dataframe
            # print("masses", masses.dtype)

            annotations.loc[cls_subset.index, "mass"] = masses
            annotations.loc[cls_subset.index, "res"] = res
            annotations.loc[cls_subset.index, "F"] = F

    # If the class is 'Coin', then put the mass of a 5 euro cents coin
    annotations.loc[annotations["class"] == "Coin", "mass"] = 3.92

    # annotations.to_csv(os.path.join(db_path, kwargs.get("data", "updated_annotations.csv")), index=False)
    annotations.to_csv(os.path.join(db_path, "annotations.csv"), index=False, na_rep="nan")


if __name__ == "__main__":
    update_mass(**vars(args))
