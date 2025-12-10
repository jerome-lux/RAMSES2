import os
import random
from PIL import Image
import numpy as np
import json
import datetime
import glob
import pandas
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Resize
from ..utils import utils

VALID_IMAGE_FORMATS = [".jpg", ".png", ".tif", ".bmp"]


class DatasetManager:
    """
    DatasetManager is a utility class for creating and managing training and validation image sets from annotated data.
    It operates on a pandas DataFrame containing instance-level annotations and provides methods to filter, split, and save datasets
    according to various constraints (class exclusion, minimum/maximum instances, resolution, etc.).

    The DataFrame must contain the following columns:
        label, folder, baseimg, area, x0, y0, x1, y1, class, res, mass

    Main features:
        - Filtering annotations/images based on constraints (resolution, class, number of instances, etc.)
        - Creating train/validation splits with oversampling and class balancing
        - Saving/loading dataset splits and annotation states
        - Tracking per-class instance counts in splits
    """

    def __init__(
        self,
        annotations,
        input_shape,
        crop_to_aspect_ratio=True,
        mask_stride=4,
        cls_to_idx=None,
        mininst=1,
        maxinst=400,
        exclude=[],
        minres=20,
        seed=None,
        filter_annotations=True,
        **kwargs,
    ):
        """
        Initialize the DatasetGenerator.

        Args:
            annotations (pd.DataFrame): DataFrame with instance-level annotations.
            input_shape (tuple): Target (height, width) for images.
            crop_to_aspect_ratio (bool): Whether to crop images to the target aspect ratio.
            mask_stride (int): Stride for mask downsampling (model-dependent).
            cls_to_idx (dict or None): Mapping from class names to integer indices. If None, generated automatically.
            mininst (int): Minimum number of instances per image to keep.
            maxinst (int): Maximum number of instances per image to keep.
            exclude (list): List of class names to exclude from the dataset.
            minres (float): Minimum image resolution (pix/mm).
            seed (int or None): Random seed for reproducibility.
            filter_annotations (bool): Whether to filter annotations/images on init.
            **kwargs: Additional arguments (ignored).
        """

        self.annotations = annotations.reset_index(drop=True)  # store current filtered annotations
        self.input_shape = input_shape
        self.cls_to_idx = cls_to_idx
        self.exclude = exclude
        self.mininst = mininst
        self.maxinst = maxinst
        self.mask_stride = mask_stride
        self.crop_to_aspect_ratio = crop_to_aspect_ratio
        self.ratio = input_shape[0] / input_shape[1]
        self.minres = minres
        self.seed = seed

        self.train_basenames = []
        self.valid_basenames = []

        self.train_class_counts = {}
        self.valid_class_counts = {}

        self.rng = np.random.default_rng(self.seed)

        if filter_annotations:
            self.filter_annotations(
                minres=self.minres, mininst=self.mininst, maxinst=self.maxinst, exclude=self.exclude
            )

        self.classes = self.annotations["class"].unique().tolist()
        self.classes.sort()

        # define class index if not given
        if self.cls_to_idx is None:
            self.cls_to_idx = {c: i + 1 for i, c in enumerate(self.classes)}

    @classmethod
    def from_file(cls, annfile, filename):
        """
        Load annotations and dataset split from files and create a DatasetGenerator instance.

        Args:
            annfile (str): Path to the CSV file with annotations.
            filename (str): Path to the JSON file with dataset split info.

        Returns:
            DatasetGenerator: An initialized instance with loaded splits and annotations.
        """

        with open(filename, "r", encoding="utf-8") as jsonfile:
            data = json.load(jsonfile)

        anns = pandas.read_csv(annfile, sep=None, engine="python")

        instance = cls(
            annotations=anns,
            filter_annotations=False,
            **data,
        )

        instance.train_class_counts = data.get("train_class_counts", {})
        instance.valid_class_counts = data.get("valid_class_counts", {})

        instance.train_basenames = data.get("train", [])
        instance.valid_basenames = data.get("valid", [])
        # instance.train_dataset = instance.build(names=instance.train_basenames, augment=True)
        # instance.valid_dataset = instance.build(names=instance.valid_basenames, augment=False)

        instance.basenames = data.get("train", []) + data.get("valid", [])

        return instance

    def save(self, filename, id=None):
        """
        Save the current dataset split and annotation state to disk.

        Args:
            filename (str): Base filename (without extension) for saving JSON and CSV files.
            id (str, optional): Dataset ID. If None, uses current timestamp.
        """

        if id is None:
            id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        data = {"dataset_id": id}
        data["exclude"] = self.exclude
        data["ratio"] = self.ratio
        data["crop_to_ar"] = self.crop_to_aspect_ratio
        data["mask_stride"] = self.mask_stride
        data["minres"] = self.minres
        data["input_shape"] = list(self.input_shape)
        data["cls_to_idx"] = self.cls_to_idx
        data["mininst"] = self.mininst
        data["maxinst"] = self.maxinst
        data["seed"] = self.seed
        data["train_class_counts"] = {str(k): int(v) for k, v in self.train_class_counts.items()}
        data["valid_class_counts"] = {str(k): int(v) for k, v in self.valid_class_counts.items()}

        if self.train_basenames is not None:
            data["train"] = self.train_basenames
        if self.valid_basenames is not None:
            data["valid"] = self.valid_basenames

        with open(filename + ".json", "w", encoding="utf-8") as jsonfile:
            # json.dump(data, jsonfile, indent=4)
            jsonfile.write(json.dumps(data, indent=4))

        self.annotations.to_csv(
            filename + ".csv", index=False, index_label="baseimg", na_rep="nan", mode="w", header=True
        )

    def update_info(self):
        """
        Recompute the number of instances per class in the train and validation sets.
        Updates self.train_class_counts and self.valid_class_counts.
        """

        self.train_class_counts = {c: 0 for c in self.classes}

        for basename in self.train_basenames:
            inst_dataframe = self.annotations.loc[self.annotations["baseimg"] == basename]
            inst_cls_value_count = {
                c: n + self.train_class_counts.get(c, 0) for c, n in inst_dataframe["class"].value_counts().items()
            }
            self.train_class_counts.update(inst_cls_value_count)

        self.valid_class_counts = {c: 0 for c in self.classes}

        for basename in self.valid_basenames:
            inst_dataframe = self.annotations.loc[self.annotations["baseimg"] == basename]
            inst_cls_value_count = {
                c: n + self.valid_class_counts.get(c, 0) for c, n in inst_dataframe["class"].value_counts().items()
            }
            self.valid_class_counts.update(inst_cls_value_count)

    def _populate_set(self, basenames, annotations, nmax, exclude=[], not_counting=[], max_reuse=3, shuffle=True):
        """
        Generate a set of image basenames meeting class and reuse constraints.

        Args:
            basenames (list): List of image basenames to consider.
            annotations (pd.DataFrame): DataFrame with instance data.
            nmax (dict): Max number of images per class.
            exclude (list): Classes to exclude.
            not_counting (list): Classes not counted for balancing.
            max_reuse (int): Max times an image can be reused.
            shuffle (bool): Whether to shuffle basenames before processing.

        Returns:
            tuple: (list of selected basenames, dict of per-class instance counts)
        """

        # Bug: classes that are not counted are not in the instance counter

        basenames = list(basenames)
        cls_inst_counter = {c: 0 for c in self.classes}
        img_reuse_counter = {b: 0 for b in basenames}

        rng = np.random.default_rng(self.seed)

        results_basenames = []

        add_instances = True
        it_counter = 0
        while add_instances:
            it_counter += 1
            add_instances = False
            img_left_per_class = {c: 0 for c in self.classes}

            if shuffle:
                rng.shuffle(basenames)

            print(f"Adding images, iteration {it_counter}     ")  # , end="\r")

            for basename in basenames:

                if img_reuse_counter[basename] <= max_reuse:

                    inst_dataframe = annotations.loc[annotations["baseimg"] == basename]
                    inst_cls_value_count = inst_dataframe["class"].value_counts().to_dict()
                    temp_dict = {}
                    skipit = False

                    for c, n in inst_cls_value_count.items():
                        if c in exclude:
                            skipit = True
                            break
                        temp_dict[c] = n + cls_inst_counter.get(c, 0)
                        if temp_dict[c] > nmax.get(c, 0):
                            skipit = True
                            break

                    if not skipit:
                        results_basenames.append(basename)
                        cls_inst_counter.update(temp_dict)
                        # print(f"Adding image {basename} {temp_dict}")
                        img_reuse_counter[basename] += 1
                        if img_reuse_counter[basename] < max_reuse:
                            for c, n in cls_inst_counter.items():
                                img_left_per_class[c] += 1

            # s'il reste des images on vérifie si on doit encore ajouter des objets pour atteindre l'objectif
            for c, n in cls_inst_counter.items():
                if c not in not_counting and n < nmax.get(c, 0) and img_left_per_class[c] > 0:
                    add_instances = True
                    break

        if len(results_basenames) > 0:
            print(
                f"Ended in {it_counter} iterations.\n Added {len(results_basenames)} images. Instances per class: {cls_inst_counter} \n"
            )

        return results_basenames, cls_inst_counter

    def filter_set(self, ds_id=None, hasmass=True, method="notin"):
        """
        Filter the current train/valid split based on dataset ID and mass availability.

        Args:
            ds_id (str, optional): Dataset ID to filter on.
            hasmass (bool): If True, keep only instances with finite, positive mass.
            method (str): 'notin' to exclude ds_id, otherwise include only ds_id.
        """
        # Filter annotation dataframe
        filtered_anns = self.annotations[self.annotations["VALID"]]

        if hasmass:
            filtered_anns = filtered_anns.loc[np.isfinite(filtered_anns["mass"]) & filtered_anns["mass"] > 0]
        else:
            filtered_anns = filtered_anns

        if ds_id is not None:
            if method == "notin":
                filtered_anns = filtered_anns.loc[filtered_anns["id"] != ds_id]
            else:
                filtered_anns = filtered_anns.loc[filtered_anns["id"] == ds_id]

        filtered_basenames = filtered_anns["baseimg"].unique().tolist()

        new_train_basenames = [b for b in self.train_basenames if b in filtered_basenames]
        new_valid_basenames = [b for b in self.valid_basenames if b in filtered_basenames]

        self.train_basenames = new_train_basenames
        self.valid_basenames = new_valid_basenames
        self.update_info()
        # self.train_dataset = self.build(self.train_basenames, augment=True)
        # self.valid_dataset = self.build(self.valid_basenames, augment=False)

    def create_set(
        self,
        n=100,
        subset="valid",
        exclude=[],
        not_counting=[],
        max_reuse=3,
        seed=None,
        append=False,
        dataset_name=None,
        constraint="in",
        mass=False,
        shuffle=True,
    ):
        """
        Create a training or validation set with up to n instances per class, oversampling as needed.

        Args:
            n (int): Max number of instances per class.
            subset (str): 'train' or 'valid'.
            exclude (list): Classes to exclude.
            not_counting (list): Classes not counted for balancing.
            max_reuse (int): Max times an image can be reused.
            seed (int or None): Random seed.
            append (bool): If True, append to existing split; otherwise, overwrite.
            dataset_name (tuple or None): (column name, list of values) to filter images.
            constraint (str): 'in' or 'not_in' for dataset_name filtering.
            mass (bool): If True, keep only instances with defined mass.
            shuffle (bool): Whether to shuffle images before selection.
        """

        if constraint.lower() not in ["in", "not_in"]:
            print("constraint must be either 'in' or 'not_in'. Set it to default 'in'")
            constraint = "in"

        if seed is not None:
            self.seed = seed

        filtered_annotations = self.annotations[self.annotations["VALID"]]

        skip = filtered_annotations[filtered_annotations["class"].isin(exclude)]["baseimg"].unique().tolist()
        filtered_annotations = filtered_annotations[~filtered_annotations["baseimg"].isin(skip)]

        if dataset_name is not None:
            text = f"{dataset_name[1]}"
            if constraint == "in":
                filtered_annotations = filtered_annotations.loc[
                    filtered_annotations[dataset_name[0]].isin(dataset_name[1])
                ]
            else:
                filtered_annotations = filtered_annotations.loc[
                    ~filtered_annotations[dataset_name[0]].isin(dataset_name[1])
                ]
        else:
            text = "whole dataset"

        # Note: the following filter deletes images where there is no mass data.
        # Beware that it keeps an image if there is at least one objet with mass data.
        if mass:
            filtered_annotations = filtered_annotations[np.isfinite(filtered_annotations["mass"])]

        filtered_basenames = filtered_annotations["baseimg"].unique().tolist()

        if shuffle:
            self.rng.shuffle(filtered_basenames)

        # To avoid using images already used in other set or already present in current set
        if subset == "train":
            temp_basenames = [f for f in filtered_basenames if f not in self.valid_basenames]
            if append:
                temp_basenames = [f for f in temp_basenames if f not in self.train_basenames]
        else:
            temp_basenames = [f for f in filtered_basenames if f not in self.train_basenames]
            if append:
                temp_basenames = [f for f in temp_basenames if f not in self.valid_basenames]

        max_inst_per_cls = {c: n for c in self.classes}

        print(f"Creating {subset} set with an objective of {n} training intances {constraint} {text}")
        print(f"Using {len(temp_basenames)} images")

        basenames, class_counts = self._populate_set(
            temp_basenames,
            annotations=filtered_annotations,
            nmax=max_inst_per_cls,
            exclude=exclude,
            not_counting=not_counting,
            max_reuse=max_reuse,
            shuffle=shuffle,
        )

        if append:
            print("Appending images and instances to train/valid sets")
            if subset == "train":
                self.train_basenames.extend(basenames)
                if shuffle:
                    self.rng.shuffle(self.train_basenames)
                self.train_class_counts = {c: self.train_class_counts.get(c, 0) + n for c, n in class_counts.items()}
            else:
                self.valid_basenames.extend(basenames)
                self.rng.shuffle(self.valid_basenames)
                self.valid_class_counts = {c: self.valid_class_counts.get(c, 0) + n for c, n in class_counts.items()}

        else:

            if subset == "train":
                self.train_basenames = basenames
                if shuffle:
                    self.rng.shuffle(self.train_basenames)
                self.train_class_counts = class_counts
                # self.train_dataset = self.build(self.train_basenames, augment=True)
            else:
                self.valid_basenames = basenames
                if shuffle:
                    self.rng.shuffle(self.valid_basenames)
                self.valid_class_counts = class_counts
                # self.valid_dataset = self.build(self.valid_basenames, augment=False)

    def create_sets(
        self,
        ntrain=100,
        nval=0,
        exclude=[],
        not_counting=[],
        max_reuse=3,
        seed=None,
        append=False,
        dataset_name=None,
        constraint="in",
        mass=False,
        shuffle=True,
    ):
        """
        Create both training and validation sets with up to ntrain/nval instances per class, oversampling as needed.

        Args:
            ntrain (int): Max number of instances per class in train set.
            nval (int): Max number of instances per class in valid set.
            exclude (list): Classes to exclude.
            not_counting (list): Classes not counted for balancing.
            max_reuse (int): Max times an image can be reused in train set.
            seed (int or None): Random seed.
            append (bool): If True, append to existing splits; otherwise, overwrite.
            dataset_name (tuple or None): (column, list of values) to filter images.
            constraint (str): 'in' or 'not_in' for dataset_name filtering.
            mass (bool): If True, keep only instances with defined mass.
            shuffle (bool): Whether to shuffle images before selection.
        """

        if constraint.lower() not in ["in", "not_in"]:
            print("constraint must be either 'in' or 'not_in'. Set it to default 'in'")
            constraint = "in"

        self.seed = seed

        filtered_annotations = self.annotations[self.annotations["VALID"]]
        skip = filtered_annotations[filtered_annotations["class"].isin(exclude)]["baseimg"].unique().tolist()
        filtered_annotations = filtered_annotations[~filtered_annotations["baseimg"].isin(skip)]

        if dataset_name is not None:
            text = f"{dataset_name[1]}"
            if constraint == "in":
                filtered_annotations = filtered_annotations.loc[
                    filtered_annotations[dataset_name[0]].isin(dataset_name[1])
                ]
            else:
                filtered_annotations = filtered_annotations.loc[
                    ~filtered_annotations[dataset_name[0]].isin(dataset_name[1])
                ]

        else:
            text = "whole dataset"

        # Note: the following filter deletes images where there is no mass data.
        # Beware that it keeps an image if there is at least one objet with mass data.
        if mass:
            filtered_annotations = filtered_annotations[np.isfinite(filtered_annotations["mass"])]

        filtered_basenames = filtered_annotations["baseimg"].unique().tolist()

        # temp_train_basenames = filtered_basenames[: int(np.around(train_frac * len(filtered_basenames)))]
        # temp_valid_basenames = filtered_basenames[int(np.around(train_frac * len(filtered_basenames))) :]

        if shuffle:
            self.rng.shuffle(filtered_basenames)

        inst_per_cls = filtered_annotations["class"].value_counts().to_dict()

        # We start by generating the validation set.
        # However, in the case where the number of instance reamining in the train set is too small
        # we limit the number of validation instance to 20% of the total number of instances
        max_inst_per_cls = {c: int(min(nval, 0.2 * v)) for c, v in inst_per_cls.items()}

        # Note reuse=0 in valid set
        print(f"Creating test set with an objective of {max_inst_per_cls} valid intances in {text}")
        print(f"Using {len(filtered_basenames)} images")

        valid_basenames, valid_class_counts = self._populate_set(
            filtered_basenames,
            annotations=filtered_annotations,
            nmax=max_inst_per_cls,
            exclude=exclude,
            not_counting=not_counting,
            max_reuse=0,
            shuffle=shuffle,
        )

        # taking the remaining images to generate the train set
        max_inst_per_cls = {c: ntrain for c in self.classes}
        temp_train_basenames = set(filtered_basenames) - set(valid_basenames)

        print(f"Creating training set with an objective of {ntrain} training intances in {text}")
        print(f"Using {len(temp_train_basenames)} images")
        train_basenames, train_class_counts = self._populate_set(
            temp_train_basenames,
            annotations=filtered_annotations,
            nmax=max_inst_per_cls,
            exclude=exclude,
            not_counting=not_counting,
            max_reuse=max_reuse,
            shuffle=shuffle,
        )

        if append:
            print("Appending images and instances to train/valid sets")
            self.train_basenames.extend(train_basenames)
            self.train_class_counts = {c: self.train_class_counts.get(c, 0) + n for c, n in train_class_counts.items()}
            self.valid_basenames.extend(valid_basenames)
            self.valid_class_counts = {c: self.valid_class_counts.get(c, 0) + n for c, n in valid_class_counts.items()}

        else:
            print("creating new train/valid sets")
            self.train_basenames = train_basenames
            self.train_class_counts = train_class_counts
            self.valid_basenames = valid_basenames
            self.valid_class_counts = valid_class_counts

        if shuffle:
            self.rng.shuffle(self.train_basenames)
            self.rng.shuffle(self.valid_basenames)

        # self.train_dataset = self.build(self.train_basenames, augment=True)
        # self.valid_dataset = self.build(self.valid_basenames, augment=False)

    def filter_annotations(self, minres, mininst, maxinst, exclude=[]):
        """
        Mark images as valid/invalid in the annotations DataFrame based on constraints.
        Checks for minimum resolution, excluded classes, file existence, and instance count.
        Updates the 'VALID' column in self.annotations.

        Args:
            minres (float): Minimum image resolution (pix/mm).
            mininst (int): Minimum number of instances per image.
            maxinst (int): Maximum number of instances per image.
            exclude (list): Classes to exclude.
        """

        # Reset annotations
        self.annotations.reset_index(drop=True, inplace=True)
        self.annotations["VALID"] = True

        self.skip = []
        del_indexes = []

        # Resolution
        self.skip = self.annotations[self.annotations["res"] < minres]["baseimg"].unique().tolist()

        # classes
        self.skip.extend(self.annotations[self.annotations["class"].isin(exclude)]["baseimg"].unique().tolist())
        self.annotations.loc[self.annotations["baseimg"].isin(self.skip), "VALID"] = False

        print(f"Skipping images {self.skip}")

        baseimgnames = self.annotations[self.annotations["VALID"]]["baseimg"].unique().tolist()

        for basename in baseimgnames:
            # Ensure that the image and labels exists !
            inst_dataframe = self.annotations.loc[self.annotations["baseimg"] == basename]
            indexes = self.annotations[self.annotations["baseimg"] == basename].index

            if len(inst_dataframe) == 0:
                print(f"Cannot find annotations for image {basename}")
                continue

            folder = inst_dataframe["folder"].to_numpy()[0]
            imgname = glob.glob(os.path.join(folder, "images", inst_dataframe["baseimg"].to_numpy()[0] + ".*"))
            labelname = glob.glob(os.path.join(folder, "labels", inst_dataframe["baseimg"].to_numpy()[0] + ".*"))
            # imgname = next((folder / Path("images")).glob(inst_dataframe["baseimg"][0] + ".*"))
            # labelname = next((folder / Path("images")).glob(inst_dataframe["baseimg"][0] + ".*"))

            if len(imgname) == 0:
                print(
                    f"Warning: image {basename} in folder {inst_dataframe['folder'].values[0] + '/images'} not found, while its annotations exists ! Skipping"
                )
                del_indexes.extend(indexes)
                self.skip.append(basename)
                continue

            if len(labelname) == 0:
                print(
                    f"Warning: image {basename} in folder {inst_dataframe['folder'].values[0] + '/labels'} not found, while its annotations exists ! Skipping"
                )
                del_indexes.extend(indexes)
                self.skip.append(basename)
                continue

            if inst_dataframe["label"].to_numpy().size < mininst:
                print(f"Skipping image {basename}: too few objects ({inst_dataframe['label'].to_numpy().size})")
                del_indexes.extend(indexes)
                self.skip.append(basename)
                continue

            if inst_dataframe["label"].to_numpy().size > maxinst:
                print(f"Skipping image {basename}: too much objects ({inst_dataframe['label'].to_numpy().size})")
                del_indexes.extend(indexes)
                self.skip.append(basename)
                continue

            # Check if a bbox fall into the cropped zone. If this is the case, the image is dismissed
            im = Image.open(imgname[0])  # Just need dimensions
            ny = im.width
            nx = im.height
            im.close()
            target_nx = min(int(np.around(ny * self.ratio)), nx)
            target_ny = int(np.around(target_nx / self.ratio))

            if self.crop_to_aspect_ratio:
                if target_nx != nx or target_ny != ny:

                    cropy = ny - target_ny
                    cropx = nx - target_nx
                    cx = cropx // 2
                    rx = cropx % 2
                    cy = cropy // 2
                    ry = cropy % 2

                    # if nx or ny is < 0, we just need to pad the image after opening it whenbuilding the dataset, so no pb
                    # if not, we must verify that there are no instances in the crop zone
                    # we crop like this: [cropx//2 + cropx%2:-cropx//2, cropy//2 + cropy%2:-cropy//2,:]

                    x0 = inst_dataframe["x0"]
                    y0 = inst_dataframe["y0"]
                    x1 = inst_dataframe["x1"]
                    y1 = inst_dataframe["y1"]

                    y0min = y0.min()
                    y0max = y0.max()
                    y1min = y1.min()
                    y1max = y1.max()

                    x0min = x0.min()
                    x0max = x0.max()
                    x1min = x1.min()
                    x1max = x1.max()

                    if cropy > 0 and not (
                        y0min > ry
                        and y0min < ny - cy - ry
                        and y0max > cy
                        and y0max < ny - cy - ry
                        and y1min > cy
                        and y1min < ny - cy - ry
                        and y1max > cy
                        and y1max < ny - cy - ry
                    ):
                        print(f"skipping image {basename} containing box outside the cropped area")
                        # del_indexes.extend(indexes)
                        del_indexes.extend(indexes)
                        self.skip.append(basename)
                        continue

                    elif cropx > 0 and not (
                        x0min > cx
                        and x0min < nx - cx - rx
                        and x0max > cx
                        and x0max < nx - cx - rx
                        and x1min > cx
                        and x1min < nx - cx - rx
                        and x1max > cx
                        and x1max < nx - cx - rx
                    ):
                        print(f"skipping image {basename} containing box outside the cropped area")
                        del_indexes.extend(indexes)
                        self.skip.append(basename)
                        continue

        self.annotations.loc[del_indexes, "VALID"] = False
        self.basenames = self.annotations[self.annotations["VALID"]]["baseimg"].unique().tolist()


class torchDataset(Dataset):

    def __init__(
        self,
        annotations,
        filenames,
        input_shape,
        mask_stride,
        cls_to_idx,
        transform=None,
        crop_to_aspect_ratio=True,
        random_resize_method=True,
        seed=None,
    ):

        self.annotations = annotations
        self.cls_to_idx = cls_to_idx
        self.filenames = filenames
        self.annotations = self.annotations.loc[self.annotations["baseimg"].isin(self.filenames)]
        # use a dict because dataframe is not safe with num_workers>1
        self.annotations_dict = (
            self.annotations.groupby("baseimg")
            .apply(lambda x: x.to_dict(orient="records"), include_groups=False)
            .to_dict()
        )
        self.input_shape = input_shape
        self.mask_stride = mask_stride
        self.crop_to_aspect_ratio = crop_to_aspect_ratio
        self.random_resize_method = random_resize_method
        self.transform = transform
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        basename = self.filenames[idx]
        # anns = self.annotations[self.annotations["baseimg"] == basename]
        instance_data = self.annotations_dict[basename]  # list of dict
        folder = instance_data[0]["folder"]
        resolution = instance_data[0]["res"]
        imgpath = os.path.join(folder, "images", basename + ".jpg")
        maskpath = os.path.join(folder, "labels", basename + ".png")
        # Use cv2 to read images and masks
        image = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image file not found: {imgpath}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        masks = cv2.imread(maskpath, cv2.IMREAD_UNCHANGED)
        if masks is None:
            raise FileNotFoundError(f"Mask file not found: {maskpath}")
        masks = masks.astype(np.int32)

        if self.crop_to_aspect_ratio:
            image, _ = utils.crop_to_aspect_ratio(self.input_shape, image)
            masks, _ = utils.crop_to_aspect_ratio(self.input_shape, masks)
        else:
            image, _ = utils.pad_to_aspect_ratio(self.input_shape, image)
            masks, _ = utils.pad_to_aspect_ratio(self.input_shape, masks)

        new_nx, new_ny = image.shape[:2]

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        masks = torch.tensor(masks, dtype=torch.int64).unsqueeze(0)

        # resize to input_shape
        if self.random_resize_method:
            method = self.rng.choice(np.array([InterpolationMode.BILINEAR, InterpolationMode.BICUBIC]))
        else:
            method = InterpolationMode.BILINEAR

        if self.input_shape[0] != new_nx or self.input_shape[1] != new_ny:
            image = Resize(size=self.input_shape, interpolation=method, antialias=True)(image)

        masks = Resize(
            (self.input_shape[0] // self.mask_stride, self.input_shape[1] // self.mask_stride),
            interpolation=InterpolationMode.NEAREST_EXACT,
        )(masks).squeeze(0)

        # reduction ratio between original image size and mask output, where the mass is computed
        # input_shape is the shape of the image AFTER downsampling/upsampling. new_nx is dimension after padding/cropping but BEFORE downsampling/upsampling

        ratio = (self.input_shape[0] // self.mask_stride) / new_nx
        masses = np.array([i["mass"] for i in instance_data])
        norm_density = torch.tensor(100 * ((resolution * ratio) ** 2) * masses, dtype=torch.float32)
        cats = [i["class"] for i in instance_data]
        category_id = torch.tensor(
            [self.cls_to_idx[c] for c in cats],
            dtype=torch.int64,
        )
        res = torch.tensor(np.array([i["res"] for i in instance_data]) * ratio, dtype=torch.float32)
        labels = torch.tensor(np.array([i["label"] for i in instance_data]), dtype=torch.int64)

        x = {
            "filename": basename,
            "image": image,  # convert to CxHxW
            "masks": masks,
            "mass": norm_density,
            "res": res,
            "category_id": category_id,
            "label": labels,
        }

        if self.transform:
            x = self.transform(x)

        return x


def collate_fn(batch):
    """
    Custom collate function to handle variable length masks and images in a batch.
    """
    return {
        "filename": [item["filename"] for item in batch],
        "image": torch.stack([item["image"] for item in batch]),
        "masks": torch.stack([item["masks"] for item in batch]),
        "mass": torch.nested.nested_tensor([item["mass"] for item in batch], layout=torch.jagged),
        "res": torch.nested.nested_tensor([item["res"] for item in batch], layout=torch.jagged),
        "category_id": torch.nested.nested_tensor([item["category_id"] for item in batch], layout=torch.jagged),
        "label": torch.nested.nested_tensor([item["label"] for item in batch], layout=torch.jagged),
    }


def collate_fn_cutmix(
    batch,
    p=0.5,
    mask_stride=4,
    max_patches=3,
    min_patch_ratio=0.05,
    max_patch_ratio=0.5,
    allow_self_mix=False,
    max_resample_attempts=5,
):
    """
    Collate function CutMix multi-patch robuste avec:
      - Remapping d'IDs sans collision (vérifié via labels[i] + mapping persistant)
      - Réutilisation du même dst_id si (j, src_id) recollé plusieurs fois
      - Pas de double ajout de métadonnées pour un dst_id déjà présent
      - Gestion batch=1, patchs vides, nettoyage final
    """
    B = len(batch)
    if B == 0:
        return {
            "filename": [],
            "image": torch.empty(0),
            "masks": torch.empty(0),
            "mass": torch.nested.nested_tensor([]),
            "res": torch.nested.nested_tensor([]),
            "category_id": torch.nested.nested_tensor([]),
            "label": torch.nested.nested_tensor([]),
        }

    # Empilement tenseurs
    images = torch.stack([item["image"] for item in batch])  # (B, H, W, 3)
    masks = torch.stack([item["masks"] for item in batch])  # (B, Hm, Wm)
    filenames = [item["filename"] for item in batch]

    # Copie / normalisation métadonnées en listes de scalaires Python
    def to_py_list(x):
        # x est déjà une liste dans ton dataset; on force les scalaires
        out = []
        for v in x:
            if torch.is_tensor(v):
                out.append(int(v.item()))
            else:
                # int() marche pour numpy scalars et ints Python
                out.append(int(v))
        return out

    masses = [list(item["mass"]) for item in batch]
    ress = [list(item["res"]) for item in batch]
    category_ids = [list(item["category_id"]) for item in batch]
    labels = [to_py_list(item["label"]) for item in batch]

    if random.random() < p:
        _, H, W, _ = images.shape
        Hm, Wm = masks.shape[1:]

        # Pour chaque image cible i : ensemble des labels déjà utilisés et max courant
        used_labels = [set(lbls) for lbls in labels]
        max_label_val = [max(lbls) if len(lbls) > 0 else 0 for lbls in labels]

        # Attribution des partenaires
        indices = list(range(B))
        random.shuffle(indices)

        for i in range(B):
            # Partenaire j
            if B == 1:
                if not allow_self_mix:
                    continue  # pas de CutMix possible
                j = 0
            else:
                if i == indices[i] and not allow_self_mix:
                    # choisir un autre que i
                    candidates = [k for k in range(B) if k != i]
                    j = random.choice(candidates)
                else:
                    j = indices[i]

            # Mapping persistant pour cette cible i (clé = (j, src_id) → dst_id)
            # Il est spécifique à l'image i pour toute la durée de ses collages
            reuse_map = {}

            num_patches = random.randint(1, max_patches)

            for _ in range(num_patches):
                # ==== rééchantillonnage d'un patch non vide (au moins une instance) ====
                patch_valid = False
                attempts = 0
                while not patch_valid and attempts < max_resample_attempts:
                    attempts += 1

                    # Taille patch (min clampé à mask_stride pour garantir >= 1 cellule masque)
                    pw = max(int(W * random.uniform(min_patch_ratio, max_patch_ratio)), mask_stride)
                    ph = max(int(H * random.uniform(min_patch_ratio, max_patch_ratio)), mask_stride)
                    pw = min(pw, W)
                    ph = min(ph, H)

                    # Position
                    x1 = random.randint(0, W - pw)
                    y1 = random.randint(0, H - ph)
                    x2, y2 = x1 + pw, y1 + ph

                    # Coordonnées masque
                    mx1, my1 = x1 // mask_stride, y1 // mask_stride
                    mx2 = min(max(mx1 + 1, x2 // mask_stride), Wm)
                    my2 = min(max(my1 + 1, y2 // mask_stride), Hm)
                    if mx2 <= mx1 or my2 <= my1:
                        continue  # trop petit côté masque

                    # Patch source
                    patch_img = images[j, y1:y2, x1:x2, :].clone()
                    patch_mask = masks[j, my1:my2, mx1:mx2].clone()

                    # Vérifie présence instance
                    if torch.unique(patch_mask).numel() > 1:  # > {0}
                        patch_valid = True

                if not patch_valid:
                    continue  # on passe ce patch si rien trouvé

                # ==== remapping des IDs avec réutilisation pour (j, src_id) ====
                unique_src = torch.unique(patch_mask).tolist()

                for src_id in unique_src:
                    if src_id == 0:
                        continue
                    src_id = int(src_id)
                    key = (j, src_id)

                    if key in reuse_map:
                        dst_id = reuse_map[key]
                    else:
                        # Propose src_id, sinon alloue un nouvel ID si collision avec labels DEJA PRÉSENTS
                        if src_id in used_labels[i]:
                            max_label_val[i] += 1
                            dst_id = max_label_val[i]
                        else:
                            dst_id = src_id
                        reuse_map[key] = dst_id

                        # Ajoute la métadonnée UNE SEULE FOIS par dst_id (si pas déjà présent)
                        if dst_id not in used_labels[i]:
                            # Cherche l'entrée correspondante dans la source j
                            # (labels[j] est 1:1 avec masses[j], ress[j], category_ids[j])
                            for k, lbl_j in enumerate(labels[j]):
                                if int(lbl_j) == src_id:
                                    labels[i].append(dst_id)
                                    category_ids[i].append(category_ids[j][k])
                                    masses[i].append(masses[j][k])
                                    ress[i].append(ress[j][k])
                                    used_labels[i].add(dst_id)
                                    break

                    # Applique le mapping dans le patch (peut être identité)
                    if dst_id != src_id:
                        patch_mask[patch_mask == src_id] = dst_id

                # Collage image + masque
                images[i, y1:y2, x1:x2, :] = patch_img
                masks[i, my1:my2, mx1:mx2] = patch_mask

            # ==== Nettoyage final pour i : ne garder que les objets visibles ====
            visible = set(int(x) for x in torch.unique(masks[i]).tolist() if int(x) != 0)
            keep_idx = [k for k, lbl in enumerate(labels[i]) if int(lbl) in visible]

            labels[i] = [labels[i][k] for k in keep_idx]
            category_ids[i] = [category_ids[i][k] for k in keep_idx]
            masses[i] = [masses[i][k] for k in keep_idx]
            ress[i] = [ress[i][k] for k in keep_idx]

    # Sortie
    return {
        "filename": filenames,
        "image": images,
        "masks": masks,
        "mass": torch.nested.nested_tensor(masses, layout=torch.jagged),
        "res": torch.nested.nested_tensor(ress, layout=torch.jagged),
        "category_id": torch.nested.nested_tensor(category_ids, layout=torch.jagged),
        "label": torch.nested.nested_tensor(labels, layout=torch.jagged),
    }
