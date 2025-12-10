import os
import json
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import datetime
import pandas
import multiprocessing
from skimage.measure import find_contours, approximate_polygon
from functools import partial

# PARAMS
TOL = 0.5
LVL = 0.5

VALID_IMAGE_FORMATS = [".jpg", ".png", ".tif", ".bmp", ".jpeg"]

parser = argparse.ArgumentParser(
    description="Create a coco json file from images, masks ans annotations.csv", fromfile_prefix_chars="@"
)
parser.add_argument(
    "-i", "--input", dest="input_dir", default=os.getcwd(), help="input directory. Default: current working directory"
)
parser.add_argument(
    "--base",
    dest="json_base",
    help="base json file where is defined the list of classes and the dataset description, etc.",
)
parser.add_argument("--tol", dest="tol", default=TOL, help="tolerance when converting bitmap masks to polygons")
parser.parse_args()
args = parser.parse_args()


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


def create_anns(img_id, imlist, annotations, cls_to_ids, tol):
    ann = annotations.loc[annotations["baseimg"] == imlist[img_id][0]]
    print("{}/{} Processing image {} containing {} instances".format(img_id, len(imlist), imlist[img_id][0], len(ann)))
    labels_img = np.array(Image.open(imlist[img_id][1]))
    res = []
    for index in ann.index:
        x0, y0, x1, y1 = ann.loc[index]["x0"], ann.loc[index]["y0"], ann.loc[index]["x1"], ann.loc[index]["y1"]
        binary_mask = labels_img[x0:x1, y0:y1]
        binary_mask = np.where(binary_mask == int(ann.loc[index]["label"]), 1, 0).astype(np.uint8)
        polys = binary_mask_to_polygon(binary_mask, level=LVL, tolerance=tol, x_offset=x0, y_offset=y0)
        area = binary_mask.sum()
        # print(label, area, pred_boxes_i[i,...])

        res.append(
            {
                "segmentation": polys,
                "area": int(area),
                "iscrowd": 0,
                "image_id": int(img_id),
                "bbox": [int(y0), int(x0), int(y1 - y0), int(x1 - x0)],
                "category_id": cls_to_ids[ann.loc[index]["class"]],
                "id": 0,
            }
        )

    return res


def convert_to_COCO(parserobj):

    print(parserobj)

    BASE_FOLDER = Path(parserobj.input_dir)
    IM_FOLDER = BASE_FOLDER / Path("images")
    LABEL_FOLDER = BASE_FOLDER / Path("labels")

    # base coco file with the definition of categories (e.g. classes) and other misc informations
    with open(parserobj.json_base) as json_file:
        coco = json.load(json_file)

    coco["info"]["date_created"] = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Dict of class names to class ids
    cls_to_ids = {}
    for cat in coco["categories"]:
        cls_to_ids[cat["name"]] = cat["id"]

    print("Classes:", cls_to_ids)

    # Création de la liste des images/labels/annotations
    imlist = {}
    annotations = {}
    for root, dirs, files in os.walk(IM_FOLDER):
        for f in files:
            basename = os.path.splitext(f)[0]
            if os.path.splitext(f)[-1].lower() in VALID_IMAGE_FORMATS:
                imlist[basename] = {"img": os.path.join(root, f)}
    print(f"Found {len(imlist)} images in {IM_FOLDER}")

    # Récupéraiton des labels
    for root, dirs, files in os.walk(LABEL_FOLDER):
        for f in files:
            basename = os.path.splitext(f)[0]
            if os.path.splitext(f)[-1].lower() in VALID_IMAGE_FORMATS:
                if imlist.get(basename, None) is not None:
                    imlist[basename].update({"labels": os.path.join(root, f)})

    annotations = pandas.read_csv(BASE_FOLDER / Path("annotations.csv"), engine="python")


    # Now verify that there is an image + a labels image + an anns file for each item
    imlist = {
        key: item for key, item in imlist.items() if None not in [item.get("img", None), item.get("labels", None)]
    }

    # Create the coco image list
    coco["images"] = []
    for id, (key, item) in enumerate(imlist.items()):
        PILimg = Image.open(item["img"])
        coco["images"].append(
            {
                "file_name": os.path.basename(item["img"]),
                "coco_url": "",
                "height": PILimg.height,
                "width": PILimg.width,
                "date_captured": "",
                "id": id,
            }
        )


    # Create the coco annotations
    coco["annotations"] = []
    imlist = [(k, v["labels"]) for k, v in imlist.items()]
    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 4)
    create_anns_func = partial(
        create_anns, imlist=imlist, annotations=annotations, cls_to_ids=cls_to_ids, tol=parserobj.tol
    )
    coco_anns = pool.map(create_anns_func, range(len(imlist)))
    pool.close()
    # flatten
    coco["annotations"] = [inst_data for img_data in coco_anns for inst_data in img_data]
    # Update the instance id
    for i, k in enumerate(coco["annotations"]):
        coco["annotations"][i]["id"] = i

    with open(os.path.join(BASE_FOLDER, "coco-annotations.json"), "w") as jsonfile:
        json.dump(coco, jsonfile)


if __name__ == "__main__":
    convert_to_COCO(args)
