import os
import json
from pathlib import Path
import datetime
from copy import deepcopy
from PIL import Image
import pandas
import numpy as np
import scipy
import skimage as sk
from skimage.color import label2rgb
from skimage.measure import find_contours, approximate_polygon, regionprops
from ..utils.visualization import _COLORS
from ..utils import crop_to_aspect_ratio, pad_to_aspect_ratio, decode_predictions
from ..model import Config, RAMSESModel
from scipy.ndimage import distance_transform_edt
import torch
import torchvision.transforms.functional as F


now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

VALID_IMAGE_FORMATS = [".jpg", ".png", ".tif", ".bmp", ".jpeg"]
MINAREA = 16
BGCOLOR = (90, 140, 200)
deltaL = 10


def box_to_coco(boxes):
    cocoboxes = np.zeros_like(boxes)
    cocoboxes[..., 0:2] = boxes[..., 0:2]
    cocoboxes[..., 2:4] = boxes[..., 2:4] - boxes[..., 0:2]
    return cocoboxes


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_polygon(binary_mask, level=0.5, tolerance=0, x_offset=0, y_offset=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode="constant", constant_values=0)
    contours = find_contours(padded_binary_mask, 0.5, fully_connected="high")
    contours = [c - 1 for c in contours]
    for contour in contours:
        contour = close_contour(contour)
        contour = approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            print("Polyshape must have at least 2 points. Skipping")
            continue
        contour = np.flip(contour, axis=1)
        contour[..., 0] += y_offset
        contour[..., 1] += x_offset
        seg = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        seg = [0 if i < 0 else i for i in seg]
        polygons.append(seg)

    return polygons


def predict(
    output_dir,
    input_size,
    resolution,
    input_dir,
    model,
    idx_to_cls,
    thresholds=(0.5, 0.5, 0.5),
    crop_to_ar=True,
    max_detections=400,
    minarea=64,
    subdirs=False,
    save_imgs=True,
    device="cuda:0",
):
    """Instance segmentation + mass estimation on an image or a series of images.

    Inputs:
    input_dir: all the images in input_dir will be processed
    input_size: input size of the network. image will be padded/cropped and resized to input_shape
    output_dir: where to put teh results
    model: pytorch model
    resolution: res of the input images
    thresholds: a list of thresolds: (t1, t2, t3)
        {"score_threshold": 0.5, "seg_threshold": 0.5, "nms_threshold": 0.6}
    crop_to_ar: wether to crop before resizing the image to input_size. If false, the image is padded.
    subdirs [False]: if True, all images in subfolders are processed
    max_detections: lmax number of detected instances (beware the masks tensor shape is [max_instances, Nx, Ny]
    minarea and minsize are the min area of instances (min area of each connected part of an instance) and minsize of boxes. Useful for filtering noise.

    Outputs:
    -COCO object (dict -> saved as json)
    -data dict (dict -> saved as csv)

    Creates on disk:
    - A coco file (json)
    - A csv file containing instance boxes resolution, mass, area, class or other information (if available)
    - label images  in output_dir/labels folder
    - image superimposed with colored labels for vizualisation in output_dir/vizu folder (low-res images)
    - individual instances in output_dir/crops folder

    """
    categories = [{"id": k, "name": v, "supercategory": "RA"} for k, v in idx_to_cls.items()]
    coco = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "contributor": "",
            "date_created": "",
            "description": str(input_dir),
            "url": "",
            "version": "",
            "year": "",
        },
        "categories": categories,
        "images": [],
        "annotations": [],
    }

    # Retrieve images (either in the input dir only or also in all the subdirectiories)
    img_dict = {}

    if not subdirs:
        for entry in os.scandir(input_dir):
            f = entry.name
            if entry.is_file() and os.path.splitext(f)[-1].lower() in VALID_IMAGE_FORMATS:
                img_dict[f] = os.path.join(input_dir, f)

    else:
        for root, _, files in os.walk(input_dir):
            for f in files:
                if os.path.isfile(os.path.join(root, f)) and os.path.splitext(f)[-1].lower() in VALID_IMAGE_FORMATS:
                    img_dict[f] = os.path.join(root, f)

    print("Found", len(img_dict), "images in ", input_dir)

    img_counter = -1

    print("Resolution of original images:", resolution, "pixels/mm")

    OUTPUT_DIR = Path(output_dir)
    VIZU_DIR = OUTPUT_DIR / Path("vizu")
    LABELS_DIR = OUTPUT_DIR / Path("labels")

    os.makedirs(VIZU_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)

    data = {
        "baseimg": [],
        "label": [],
        "res": [],
        "class": [],
        "x0": [],
        "x1": [],
        "y0": [],
        "y1": [],
        "area": [],
        "mass": [],
        "axis_major_length": [],
        "axis_minor_length": [],
        "feret_diameter_max": [],
        "max_inscribed_radius": [],
    }

    model = model.to(device)
    model.eval()

    # network input size
    nx, ny = input_size[:2]

    keys = sorted(list(img_dict.keys()))
    # np.random.shuffle(keys)

    for counter, imgname in enumerate(keys):

        impath = img_dict[imgname]
        PILimg = Image.open(impath)

        image = np.array(PILimg) / 255.0
        ini_nx, ini_ny = image.shape[0:2]

        print(
            "Processing image {} ({}/{}), size {}x{}".format(imgname, counter + 1, len(keys), ini_nx, ini_ny),
            end="",
        )

        #  crop the image to the prescribed size before resizing
        padding = ((0, 0), (0, 0))
        if crop_to_ar:
            image, _ = crop_to_aspect_ratio(input_size, image)
            if image.shape[0:2] != (ini_nx, ini_ny):
                print(f". Cropping to shape {image.shape[0:2]}", end="")
        else:
            image, padding = pad_to_aspect_ratio((nx, ny), image)

        fullsize_nx, fullsize_ny = image.shape[:2]

        resized_image = sk.transform.resize(
            image, (nx, ny), anti_aliasing=True, order=1, mode="reflect", preserve_range=True
        )
        ratio = resized_image.shape[0] / fullsize_nx  # < 1
        inv_ratio = fullsize_nx / resized_image.shape[0]  # > 1

        img_counter += 1
        coco["images"].append(
            {
                "file_name": imgname,
                "coco_url": "",
                "height": PILimg.height,
                "width": PILimg.width,
                "date_captured": "",
                "id": img_counter,
            }
        )

        resized_image = torch.from_numpy(resized_image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).float()
        resized_image = resized_image.to(device)

        # Get network predictions
        results = model(
            resized_image,
            training=False,
            cls_threshold=thresholds[0],
            nms_threshold=thresholds[2],  # iou threshold or cls threshold in MatrixNMX
            mask_threshold=thresholds[1],  # threshold to apply to the masks
            max_detections=max_detections,
            scale_by_mask_scores=False,
            min_area=minarea,
            nms_mode="greedy",
        )[
            0
        ]  # take the first element of the batch (because batch size = 1 here)

        pred_masks = results["masks"].detach()  # [Npred, H, W]
        pred_masses = results["masses"].detach()  # [Npred]
        pred_cls_ids = results["cls_labels"].detach() + 1  # [Npred]
        pred_scores = results["scores"].detach()  # [Npred]
        # get labeled image: [H, W]
        labeled_image = decode_predictions(results["masks"], results["scores"], threshold=0.5, by_mask_scores=False)
        final_labels = torch.unique(labeled_image)[1:] - 1  # skipping 0
        labeled_image = labeled_image.cpu().numpy().astype(int)

        if pred_scores.numel() <= 0:  # No detection !
            print("...OK. No instance detected !")
            continue

        # some slices may be empty -> all pixel values are < seg_threshold
        if torch.numel(final_labels) != torch.numel(pred_cls_ids):
            pred_scores = pred_scores[final_labels]
            pred_cls_ids = pred_cls_ids[final_labels]
            pred_masses = pred_masses[final_labels]

        # mask is usually smaller than input_size.
        # We compute the actual downsampling from the input image size to get the resolution and extract the unpadded mask
        mask_stride = nx // labeled_image.shape[0]
        res_ratio = (nx // mask_stride) / fullsize_nx
        scaling = (10 * resolution * res_ratio) ** 2
        pred_masses = pred_masses.cpu().numpy() / scaling

        # Resize to input size (not fullsize because it's too big!)
        image_for_vizualization = resized_image[0].permute(1, 2, 0).cpu().numpy()
        resized_labeled_image = sk.transform.resize(labeled_image, (nx, ny), order=0)

        # Extract regions on full size image
        region_properties = regionprops(resized_labeled_image, extra_properties=(max_inscribed_radius_func,))
        labels = np.array([prop["label"] for prop in region_properties])

        if labels.size != pred_cls_ids.numel():
            print(labels.size, pred_cls_ids.size)
            print("WARNING: number of detected regions is not the same as number of predicted instances")

        boxes = np.array([prop["bbox"] for prop in region_properties])

        # Saving props (scaled to original image size in pixels)
        area = [prop["area"] * inv_ratio**2 for prop in region_properties]
        axis_major_length = [int(np.around(prop["axis_major_length"] * inv_ratio)) for prop in region_properties]
        axis_minor_length = [int(np.around(prop["axis_minor_length"] * inv_ratio)) for prop in region_properties]
        feret_diameter_max = [int(np.around(prop["feret_diameter_max"] * inv_ratio)) for prop in region_properties]
        max_inscribed_radius = [
            int(np.around(prop["max_inscribed_radius_func"] * inv_ratio)) for prop in region_properties
        ]

        data["area"].extend(area)
        data["axis_major_length"].extend(axis_major_length)
        data["axis_minor_length"].extend(axis_minor_length)
        data["feret_diameter_max"].extend(feret_diameter_max)
        data["max_inscribed_radius"].extend(max_inscribed_radius)
        data["mass"].extend(pred_masses.tolist())
        data["class"].extend(pred_cls_ids.tolist())

        if save_imgs:
            # resize masks and image for vizualisation
            vizuname = "VIZU-{}.jpg".format(os.path.splitext(imgname)[0])
            bd = sk.segmentation.find_boundaries(resized_labeled_image, connectivity=2, mode="inner", background=0)
            vizu = label2rgb(
                resized_labeled_image, image_for_vizualization, alpha=0.25, bg_label=0, colors=_COLORS, saturation=1
            )
            vizu = np.where(bd[..., np.newaxis], (0, 0, 0), vizu)
            vizu = np.around(255 * vizu).astype(np.uint8)
            Image.fromarray(vizu).save(os.path.join(VIZU_DIR, vizuname))

        print(f"...OK. Found {len(labels)} instances. ")

        cocoboxes = box_to_coco(boxes)

        # saving coco instance data
        for i, prop in enumerate(region_properties):

            box = prop["bbox"]

            data["baseimg"].append(imgname)
            data["label"].append(labels[i])
            data["res"].append(resolution)
            data["x0"].append(box[0])
            data["x1"].append(box[2])
            data["y0"].append(box[1])
            data["y1"].append(box[3])

            # Create COCO annotation
            polys = binary_mask_to_polygon(
                prop["image"],
                level=0.5,
                x_offset=cocoboxes[i, 0],
                y_offset=cocoboxes[i, 1],
            )

            coco["annotations"].append(
                {
                    "segmentation": polys,
                    "area": int(data["area"][i]),
                    "iscrowd": 0,
                    "image_id": img_counter,
                    "bbox": [int(b) for b in cocoboxes[i]],
                    "category_id": int(pred_cls_ids[i]),
                    "id": i,
                }
            )

        # Save labels
        labelname = "{}.png".format(os.path.splitext(imgname)[0])
        Image.fromarray(resized_labeled_image.astype(np.uint16)).save(os.path.join(LABELS_DIR, labelname))

    #    info_filepath = os.path.join(OUTPUT_DIR, "info.json")
    #    with open(info_filepath, "w", encoding="utf-8") as jsonconfig:
    #        json.dump(model.config, jsonconfig)

    for k, v in data.items():
        print(k, len(v))

    print("Saving COCO in ", OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "coco_annotations.json"), "w", encoding="utf-8") as jsonfile:
        json.dump(coco, jsonfile)

    df = pandas.DataFrame().from_dict(data)
    df.to_csv(os.path.join(OUTPUT_DIR, "annotations.csv"), na_rep="nan", header=True)

    # return COCO and data dict
    return coco, data


def single_image_prediction(
    input_image,
    model,
    thresholds=(0.5, 0.5, 0.5),
    max_detections=400,
    minarea=MINAREA,
    weight_by_scores=False,
    device="cuda:0",
):
    nx, ny = input_image.shape[:2]  # Note: image can be padded or cropped
    input_image = torch.from_numpy(input_image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).float()
    input_image = input_image.to(device)

    results = model(
        input_image,
        training=False,
        cls_threshold=thresholds[0],
        nms_threshold=thresholds[2],  # iou threshold or cls threshold in MatrixNMX
        mask_threshold=thresholds[1],  # threshold to apply to the masks
        max_detections=max_detections,
        scale_by_mask_scores=weight_by_scores,
        min_area=minarea,
        nms_mode="greedy",
    )[
        0
    ]  # take the first element of the batch (because batch size = 1 here)

    pred_masks = results["masks"].detach()  # [Npred, H, W]
    pred_masses = results["masses"].detach()  # [Npred]
    pred_cls_ids = results["cls_labels"].detach() + 1  # [Npred]
    pred_scores = results["scores"].detach()  # [Npred]

    # get labeled image: [H, W]
    labeled_image = decode_predictions(pred_masks, pred_scores, threshold=0.5, by_mask_scores=False)
    labeled_image = labeled_image.cpu().numpy().astype(int)
    mask_stride = nx // labeled_image.shape[0]
    # Resize to input size (not fullsize because it's too big!) and delete padding
    resized_labeled_image = sk.transform.resize(labeled_image, (nx, ny), order=0)
    region_properties = regionprops(resized_labeled_image, extra_properties=(max_inscribed_radius_func,))

    final_labels = np.array([prop["label"] for prop in region_properties]) - 1
    # some slices may be empty -> all pixel values are < seg_threshold
    if final_labels.size != torch.numel(pred_cls_ids):
        pred_scores = pred_scores[final_labels]
        pred_cls_ids = pred_cls_ids[final_labels]
        pred_masses = pred_masses[final_labels]

    if torch.numel(pred_scores) <= 0:  # No detection !
        return None, None, None, None, None, None

    return (
        region_properties,
        resized_labeled_image,
        pred_cls_ids.cpu(),
        pred_masses.cpu(),
        pred_scores.cpu(),
        mask_stride,
    )


def stream_predict(
    input_size,
    input_dir,
    output_dir,
    resolution,
    model,
    idx_to_cls=None,
    crop_to_ar=True,
    deltaL=5,
    thresholds=(0.5, 0.5, 0.5),
    weight_by_scores=False,
    max_detections=400,
    bgcolor=BGCOLOR,
    minarea=MINAREA,
    subdirs=True,
    save_imgs=True,
    device="cuda:0",
):
    """
    Performs streaming prediction on a directory of images using a segmentation/detection model,
    processes overlapping regions between consecutive images, and saves results and annotations.

    Args:
        input_size (tuple): Size (height, width) to which images are resized for model input.
        input_dir (str or Path): Directory containing input images.
        output_dir (str or Path): Directory to save output images, labels, and annotations.
        resolution (float): Resolution of original images in pixels per mm.
        model (torch.nn.Module): Pre-trained segmentation/detection model.
        crop_to_ar (bool, optional): Whether to crop images to match input aspect ratio. Defaults to True.
        deltaL (int, optional): Margin in pixels to consider objects touching image edges. Defaults to 5.
        thresholds (tuple, optional): Detection thresholds class, segmentation masks and NMS algorithm (IoU). Defaults to (0.5, 0.5, 0.5).
        weight_by_scores (bool, optional): Whether to weight results by detection scores. Defaults to False.
        max_detections (int, optional): Maximum number of detections per image. Defaults to 400.
        bgcolor (tuple or int, optional): Background color value for padding and filtering. Defaults to BGCOLOR.
        minarea (int, optional): Minimum area for detected objects. Defaults to MINAREA.
        subdirs (bool, optional): Whether to search for images recursively in subdirectories. Defaults to True.
        save_imgs (bool, optional): Whether to save visualization images and labels. Defaults to True.
        device (str, optional): Device for model inference (e.g., "cuda:0" or "cpu"). Defaults to "cuda:0".

    Returns:
        dict: Dictionary containing properties and annotations for all detected objects, including:
            - baseimg: List of image filenames.
            - label: List of object labels.
            - res: List of image resolutions.
            - class: List of predicted class IDs.
            - x0, x1, y0, y1: Bounding box coordinates.
            - area: List of object areas.
            - mass: List of predicted masses.
            - axis_major_length, axis_minor_length: Major/minor axis lengths.
            - feret_diameter_max: Maximum Feret diameter.
            - max_inscribed_radius: Maximum inscribed radius.

    Side Effects:
        - Saves processed images, labeled masks, and visualizations to output directories.
        - Writes a CSV file with object annotations and a JSON file with configuration info.

    """

    # Retrieve images
    img_dict = {}

    if not subdirs:
        for entry in os.scandir(input_dir):
            f = entry.name
            if entry.is_file() and os.path.splitext(f)[-1].lower() in VALID_IMAGE_FORMATS:
                img_dict[f] = os.path.join(input_dir, f)

    else:
        for root, _, files in os.walk(input_dir):
            for f in files:
                if os.path.isfile(os.path.join(root, f)) and os.path.splitext(f)[-1].lower() in VALID_IMAGE_FORMATS:
                    img_dict[f] = os.path.join(root, f)

    print("Found", len(img_dict), "images in ", input_dir)

    img_counter = -1

    print("Resolution of original images:", resolution, "pixels/mm")

    OUTPUT_DIR = Path(output_dir)
    VIZU_DIR = OUTPUT_DIR / Path("vizu")
    LABELS_DIR = OUTPUT_DIR / Path("labels")
    OVERLAPPING_IMGS_DIR = OUTPUT_DIR / Path("overlap")

    os.makedirs(VIZU_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)
    os.makedirs(OVERLAPPING_IMGS_DIR, exist_ok=True)

    if np.array(bgcolor).max() > 1:
        bgcolor = np.array(bgcolor) / 255.0

    data = {
        "baseimg": [],
        "label": [],
        "res": [],
        "class": [],
        "x0": [],
        "x1": [],
        "y0": [],
        "y1": [],
        "area": [],
        "mass": [],
        "axis_major_length": [],
        "axis_minor_length": [],
        "feret_diameter_max": [],
        "max_inscribed_radius": [],
    }

    # network input size nx ny or H W
    nx, ny = input_size[:2]

    keys = sorted(list(img_dict.keys()))
    # np.random.shuffle(keys)

    model = model.to(device)
    model.eval()

    for counter, imgname in enumerate(keys):

        impath = img_dict[imgname]
        PILimg = Image.open(impath)

        image = np.array(PILimg) / 255.0
        ini_nx, ini_ny = image.shape[0:2]

        print(
            "Processing image {} ({}/{}), size {}x{}".format(imgname, counter + 1, len(keys), ini_nx, ini_ny),
            end="",
        )
        padding = ((0, 0), (0, 0))
        # Image preprocessing: crop/pad and resize to input size
        if crop_to_ar:
            image, _ = crop_to_aspect_ratio(input_size, image)
            if image.shape[0:2] != (ini_nx, ini_ny):
                print(f". Cropping to shape {image.shape[0:2]}", end="")
        else:
            image, padding = pad_to_aspect_ratio((nx, ny), image)

        fullsize_nx, fullsize_ny = image.shape[:2]

        resized_image = sk.transform.resize(
            image, (nx, ny), anti_aliasing=True, order=1, mode="reflect", preserve_range=True
        )
        ratio = resized_image.shape[0] / fullsize_nx  # < 1
        inv_ratio = fullsize_nx / resized_image.shape[0]  # > 1

        img_counter += 1

        # Get prediction and region properties computed on the labeled image resized to the input image size (nx, ny)
        region_properties, resized_labeled_image, pred_cls_ids, pred_masses, pred_scores, mask_stride = (
            single_image_prediction(
                resized_image,
                model,
                thresholds=thresholds,
                max_detections=max_detections,
                minarea=minarea,
                weight_by_scores=weight_by_scores,
                device=device,
            )
        )

        if region_properties is None:
            continue

        labels = np.array([prop["label"] for prop in region_properties])
        pred_boxes = np.array([prop["bbox"] for prop in region_properties])

        # middle indexes:  indexes of boxes not touching edges,
        # up indexes: boxes touching upper edge (i.e. x->0)
        # TODO: check if an object touches both edges (normalement non !)

        if counter == 0:
            # on extrait toutes les boites sauf celle touchant le bas de l'image (x=nx-padx1)
            middle_indexes = np.where(pred_boxes[:, 2] < nx - deltaL)
        elif counter == len(img_dict) - 1:
            # On extrait toutes les boites sauf celles touchant le haut (x=0+padx0)
            middle_indexes = np.where(pred_boxes[:, 0] > deltaL)
        else:
            # On extrait toutes les boites sauf celles touchant le haut(x=0) ET le bas (x=nx)
            middle_indexes = np.where((pred_boxes[:, 2] < nx - deltaL) & (pred_boxes[:, 0] > deltaL))

        middle_labels = []
        if middle_indexes[0].size > 0:
            middle_labels = labels[middle_indexes]

        # le "haut de l'image" correspond à nx=0, le bas à nx-1
        up_indexes = np.where(pred_boxes[:, 0] <= deltaL)
        bottom_indexes = np.where(pred_boxes[:, 2] >= nx - deltaL)

        print(
            f"...OK. Found {torch.numel(pred_scores)} instances. ",
            f"{up_indexes[0].size} objects touching the upper edge and {bottom_indexes[0].size} touching the bottom.",
        )

        if middle_indexes[0].size > 0:

            # delete labels of objects touching the edges
            middle_masks = np.where(np.isin(resized_labeled_image, middle_labels), resized_labeled_image, 0)

            # Save labels without objects touching the edges
            labelname = "{}.png".format(os.path.splitext(imgname)[0])
            Image.fromarray(middle_masks.astype(np.uint16)).save(os.path.join(LABELS_DIR, labelname))

            # Delete objects *not* touching the edges in the input image
            filtered_image = np.where(middle_masks[..., np.newaxis] == 0, resized_image, bgcolor).astype(
                resized_image.dtype
            )

            middle_pred_boxes = pred_boxes[middle_indexes]

            # Saving properties of objects not touching the edges
            area = [prop["area"] * inv_ratio**2 for prop in region_properties if prop["label"] in middle_labels]
            axis_major_length = [
                int(np.around(prop["axis_major_length"] * inv_ratio))
                for prop in region_properties
                if prop["label"] in middle_labels
            ]
            axis_minor_length = [
                int(np.around(prop["axis_minor_length"] * inv_ratio))
                for prop in region_properties
                if prop["label"] in middle_labels
            ]
            feret_diameter_max = [
                int(np.around(prop["feret_diameter_max"] * inv_ratio))
                for prop in region_properties
                if prop["label"] in middle_labels
            ]
            max_inscribed_radius = [
                int(np.around(prop["max_inscribed_radius_func"] * inv_ratio))
                for prop in region_properties
                if prop["label"] in middle_labels
            ]

            res_ratio = (nx // mask_stride) / fullsize_nx
            scaling = (10 * resolution * res_ratio) ** 2
            masses = pred_masses[middle_indexes] / scaling
            classes = pred_cls_ids[middle_indexes].tolist()

            data["baseimg"].extend([imgname] * len(middle_labels))
            data["label"].extend(middle_labels)
            data["res"].extend([resolution] * len(middle_labels))
            data["x0"].extend(middle_pred_boxes[:, 0].tolist())
            data["x1"].extend(middle_pred_boxes[:, 2].tolist())
            data["y0"].extend(middle_pred_boxes[:, 1].tolist())
            data["y1"].extend(middle_pred_boxes[:, 3].tolist())
            data["area"].extend(area)
            data["mass"].extend(masses.tolist())
            if idx_to_cls is not None:
                classes = [idx_to_cls[x] for x in classes]
            data["class"].extend(classes)
            data["axis_major_length"].extend(axis_major_length)
            data["axis_minor_length"].extend(axis_minor_length)
            data["feret_diameter_max"].extend(feret_diameter_max)
            data["max_inscribed_radius"].extend(max_inscribed_radius)

            if save_imgs:
                # save image and labels without objects touching the edges. No unpadding or upscaling to simplify.
                vizuname = "VIZU-{}.jpg".format(os.path.splitext(imgname)[0])
                bd = sk.segmentation.find_boundaries(middle_masks, connectivity=2, mode="inner", background=0)
                vizu = label2rgb(middle_masks, resized_image, alpha=0.25, bg_label=0, colors=_COLORS, saturation=0.5)
                vizu = np.where(bd[..., np.newaxis], (0, 0, 0), vizu)
                vizu = np.around(255 * vizu).astype(np.uint8)
                Image.fromarray(vizu).save(os.path.join(VIZU_DIR, vizuname))

        else:
            # print(", all detected instances touch the edges")
            filtered_image = resized_image

        # Create overlapping image using previous bottom image and current up image
        # note that we create an image with the same resolution as the input shape of the network (and not the original image in the folders!)

        if counter > 0:
            # hauteur bande de recouvrement haute et basse
            # ici bottom_boxes correspond aux objets détectés sur l'image précédente
            boxes_up = pred_boxes[up_indexes]

            # S'il n'y a pas d'objets touchant les bords en haut de l'image actuelle ou en bas de la précédente alors on passe
            if not (boxes_up.size == 0 and prev_bottom_boxes.size == 0):
                # Up correspond au haut de l'image cad coords à partir de 0
                if boxes_up.size > 0:
                    lxup = boxes_up[:, 2].max()
                else:
                    lxup = 0

                # bottom correspond au "bas" de l'image (vers les coords croissantes)
                if prev_bottom_boxes.size > 0:
                    lxdown = prev_bottom_boxes[:, 0].min()
                else:
                    lxdown = nx

                if lxup == 0 and lxdown < nx:
                    overlap_image = prev_filtered_image[lxdown:, ...]

                elif lxup > 0 and lxdown == nx:
                    overlap_image = filtered_image[0:lxup, ...]
                else:
                    overlap_image = np.concatenate(
                        [
                            prev_filtered_image[lxdown:, ...],
                            filtered_image[0:lxup, ...],
                        ],
                        0,
                    )

                # resize the image to the network's input size using padding. New shape should be equal to (nx, ny)
                overlap_image, o_padding = pad_to_aspect_ratio((nx, ny), overlap_image)
                if overlap_image.shape[0] != nx or overlap_image.shape[1] != ny:
                    print("Warning, overlap image has not the right shape after padding:", overlap_image.shape)

                o_imgname = "OVERLAP_{}-{}.jpg".format(os.path.splitext(prev_imgname)[0], os.path.splitext(imgname)[0])
                print("Processing overlapping image {}...".format(o_imgname), end="")
                # save the overlap image (maybe we should upscale to fullsize here)
                o_PILimg = Image.fromarray(np.around(overlap_image * 255).astype(np.uint8))
                o_PILimg.save(os.path.join(OVERLAPPING_IMGS_DIR, o_imgname), quality=95)

                PILimg = Image.open(os.path.join(OVERLAPPING_IMGS_DIR, o_imgname))
                overlap_image = np.array(PILimg) / 255.0

                # Get predictions on the overlap image
                (
                    o_region_properties,
                    o_resized_labeled_image,
                    o_pred_cls_ids,
                    o_pred_masses,
                    o_pred_scores,
                    mask_stride,
                ) = single_image_prediction(
                    overlap_image,
                    model,
                    thresholds=thresholds,
                    max_detections=max_detections,
                    minarea=minarea,
                    weight_by_scores=weight_by_scores,
                    device=device,
                )

                if o_pred_cls_ids is None:
                    print("No instance detected !")
                    continue

                print(f"{torch.numel(o_pred_cls_ids)} instances detected")

                # Save labeled image
                o_labelname = "{}.png".format(os.path.splitext(o_imgname)[0])
                Image.fromarray(o_resized_labeled_image.astype(np.uint16)).save(os.path.join(LABELS_DIR, o_labelname))

                # Note: the overlap images are saved without resizing, so the properties are not scaled
                # The resolution is therefore the one of the current image
                o_pred_boxes = np.array([prop["bbox"] for prop in o_region_properties])
                o_labels = np.array([prop["label"] for prop in o_region_properties])
                area = np.array([prop["area"] for prop in o_region_properties])
                axis_major_length = np.array([prop["axis_major_length"] for prop in o_region_properties])
                axis_minor_length = np.array([prop["axis_minor_length"] for prop in o_region_properties])
                feret_diameter_max = np.array([prop["feret_diameter_max"] for prop in o_region_properties])
                max_inscribed_radius = np.array([prop["max_inscribed_radius_func"] for prop in o_region_properties])

                res_ratio = (nx // mask_stride) / fullsize_nx
                scaling = (10 * resolution * res_ratio) ** 2
                masses = o_pred_masses / scaling
                classes = o_pred_cls_ids.tolist()

                data["baseimg"].extend([o_imgname] * o_labels.size)
                data["label"].extend(o_labels.tolist())
                data["res"].extend([resolution * nx / fullsize_nx] * o_labels.size)
                data["x0"].extend(o_pred_boxes[:, 0].tolist())
                data["x1"].extend(o_pred_boxes[:, 2].tolist())
                data["y0"].extend(o_pred_boxes[:, 1].tolist())
                data["y1"].extend(o_pred_boxes[:, 3].tolist())
                data["area"].extend(area.tolist())
                data["mass"].extend(masses.tolist())
                if idx_to_cls is not None:
                    classes = [idx_to_cls[x] for x in classes]
                data["class"].extend(classes)
                data["axis_major_length"].extend(axis_major_length.tolist())
                data["axis_minor_length"].extend(axis_minor_length.tolist())
                data["feret_diameter_max"].extend(feret_diameter_max.tolist())
                data["max_inscribed_radius"].extend(max_inscribed_radius.tolist())

                if save_imgs:
                    vizuname = "VIZU-{}.jpg".format(os.path.splitext(o_imgname)[0])
                    bd = sk.segmentation.find_boundaries(
                        o_resized_labeled_image, connectivity=2, mode="inner", background=0
                    )
                    vizu = label2rgb(
                        o_resized_labeled_image,
                        overlap_image,
                        alpha=0.25,
                        bg_label=0,
                        colors=_COLORS,
                        saturation=0.5,
                    )
                    vizu = np.where(bd[..., np.newaxis], (0, 0, 0), vizu)
                    vizu = np.around(255 * vizu).astype(np.uint8)
                    Image.fromarray(vizu).save(os.path.join(VIZU_DIR, vizuname))

        # paramètres utilisés dans l'itération suivante
        # bottom_labels = labels[bottom_indexes]
        prev_bottom_boxes = pred_boxes[bottom_indexes]
        # print("bottom boxes", pred_boxes[bottom_indexes])
        prev_filtered_image = filtered_image
        prev_imgname = imgname

    info_filepath = os.path.join(OUTPUT_DIR, "info.json")
    config = {"DATA_DIR": str(input_dir)}
    with open(info_filepath, "w", encoding="utf-8") as jsonconfig:
        json.dump(config, jsonconfig)

    df = pandas.DataFrame().from_dict(data)
    # df = df.set_index("filename")
    df.to_csv(os.path.join(OUTPUT_DIR, "annotations.csv"), na_rep="nan", header=True, index=False)

    # return data dict
    return data


def max_inscribed_radius_func(mask):

    return distance_transform_edt(np.pad(mask, 1)).max()
