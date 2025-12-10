import os
import json
from pathlib import Path
import pandas
import argparse
import numpy as np
import shutil
from functools import partial
import multiprocessing
from PIL import Image
from pycocotools.coco import COCO
from skimage.measure import regionprops
from scipy import ndimage as ndi
from utils import polys_to_bitmap

# Local
import sys

"""Example fo data.json file:"
        "{
            "images": [
                {"id": 1, "file_name": "IMG_0001.jpg", "res": 28.7},
                {"id": 2, "file_name": "IMG_0002.jpg", "res": 28.7}
            ],
            "batches": [
                {"image_names": ["IMG_0001.jpg","IMG_0002.jpg"], "mass": {"Ra": 12.3, "Rc": "NaN", "UNKNOWN": "NaN"}}
            ]
        }"""

# PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(PARENT_DIR)

VALID_IMAGE_FORMATS = [".jpg", ".png", ".tif", ".bmp", ".jpeg"]

default_metadata = {"tag": "CONVEYOR", "camera": "JAI SW-8000Q"}

parser = argparse.ArgumentParser(
    description=(
        "Convert COCO annotation JSON(s) into a dataset composed of: "
        "1) labeled mask images saved in /labels (one 16-bit PNG per original image), "
        "2) image CSV annotation files saved in output folder, "
        "3) an optional instances/ folder with per-instance crops, and "
        "4) a metadata.json (merged or created from data.json). "
        "The script can also copy original images to an output images/ folder. "
        "Supports filtering by minimum instance area, using provided resolution (res) when no data.json is present, "
        "and setting per-image classes from data.json (--setcats). "
        "Options: --copy to copy images, --masks to generate mask images, --overwrite to replace the output CSV, "
        "--extract_crops to save instance crops, --min_area to filter small instances, --res to set default resolution, "
        "and --setcats to populate instance classes from metadata. "
    ),
    fromfile_prefix_chars="@",
)
parser.add_argument(
    "-i", "--input", dest="input_dir", default=os.getcwd(), help="input directory. Default: current working directory"
)
parser.add_argument(
    "-o",
    "--output",
    dest="output_dir",
    default=os.path.join(os.getcwd(), "new_dataset"),
    help="output directory. Default: 'current working dir/new_dataset'",
)
parser.add_argument("-c", "--copy", dest="copyfiles", action="store_true", help="copy images in output folder")
parser.add_argument(
    "-m",
    "--masks",
    dest="masks",
    action="store_true",
    help="Create mask images from polygon annotations in /label folder",
)
parser.add_argument(
    "--overwrite", dest="overwrite", action="store_true", help="overwrite instance csv file. Default is append"
)
parser.add_argument(
    "--extract_crops",
    dest="extract_crops",
    action="store_true",
    help="Extract crops of each instance and put them in /instances folder",
)
parser.add_argument("--min_area", dest="min_area", default=0.0, type=float, help="min area of generated instances")
parser.add_argument("--res", dest="res", default=28.7, type=float, help="resolution of the image (if no data.json)")
parser.add_argument(
    "--setcats",
    dest="setcats",
    action="store_true",
    help="use data.json to set instance categories" "It works only when there is exactly one class per image",
)


parser.parse_args()

args = parser.parse_args()


def crop(image, box):
    """Extract aand return a crop defined in box (x0,y0,x1,y1) from image"""
    box = np.around(np.array(box)).astype(int)
    return image[box[0] : box[2], box[1] : box[3], ...]


def merge_json(file1, file2):
    with open(file1) as jsonfile1:
        json1 = json.load(jsonfile1)
    with open(file2) as jsonfile2:
        json2 = json.load(jsonfile2)

    # Max id of the first json file
    ids1 = [imdata["id"] for imdata in json1["images"]]
    id1max = np.max(ids1)
    for imdata in json2["images"]:
        # Need to convert to int, because json does not support numpy int64 (returned by max function)
        imdata["id"] = int(imdata["id"] + id1max)

    fnames = [item["file_name"] for item in json1["images"]]

    for item in json2["images"]:
        if item["file_name"] not in fnames:
            json1["images"].append(item)
            fnames.append(item["file_name"])

    for item in json2["batches"]:
        if item not in json1["batches"]:
            json1["batches"].append(item)

    # json1["images"].extend(json2["images"])
    # json1["batches"].extend(json2["batches"])

    return json1


def import_COCO(parserobj):

    print(parserobj)

    BASE_FOLDER = Path(parserobj.input_dir)
    IM_FOLDER = BASE_FOLDER / Path("images")
    ANN_FOLDER = BASE_FOLDER / Path("annotations")
    OUTPUT_FOLDER = parserobj.output_dir

    # if parserobj.imdata == "data.json":
    #     datapath = BASE_FOLDER / Path("data.json")
    # else:
    datapath = BASE_FOLDER / Path("data.json")

    nodata = False

    if os.path.exists(datapath):
        with open(datapath) as json_file:
            imdatafile = json.load(json_file)
        print(f"Using provided masses in  {datapath}")
    else:
        nodata = True

    # stockage instance data 1 fichier par image
    instance_data = {}

    # Création de la correspondance nom d'image / localisation
    img_name_to_loc = {}
    for root, dirs, files in os.walk(IM_FOLDER):
        for f in files:
            if os.path.splitext(f)[-1].lower() in VALID_IMAGE_FORMATS:
                img_name_to_loc[f] = os.path.join(root, f)

    images_info = []
    batch_info = []

    inst_cats = {}
    # Loading COCO annotation files
    ann_files = {}
    for root, dirs, files in os.walk(ANN_FOLDER):
        for f in files:
            if f.endswith(".json"):
                ann_files[f] = COCO(os.path.join(root, f))
                # If no data is provided, then we create a file with given resolution and no mass info
                if nodata:
                    # print([i for i in ann_files[f].imgs.items()])
                    images_info.extend(
                        [
                            {
                                "id": i["id"],
                                "file_name": i["file_name"],
                                "res": parserobj.res,
                                "height": i["height"],
                                "width": i["width"],
                                "camera": default_metadata["camera"],
                                "tag": default_metadata["tag"],
                            }
                            for i in ann_files[f].imgs.values()
                        ]
                    )
                    imnames = [i["file_name"] for i in ann_files[f].imgs.values()]
                    batch_info.append({"image_names": imnames, "mass": {"UNKNOWN": "NaN"}})

    print(f"Annotations files to process: {ann_files}")

    # Save data if file does not exist
    if nodata:
        imdatafile = {"images": images_info, "batches": batch_info}
        with open(datapath, "w+", encoding="utf-8") as json_file:
            json.dump(imdatafile, json_file, indent=4)

    # Beware ! We create a dict storing the class for each image -> it works when there is exactly one class per image (where mass is not nan)
    elif args.setcats:
        for batch in imdatafile["batches"]:
            cat = [k for k, v in batch["mass"].items() if not np.isnan(v)][0]
            for img in batch["image_names"]:
                inst_cats[img] = cat

    resolutions = {elem["file_name"]: elem["res"] for elem in imdatafile["images"]}

    os.makedirs(os.path.join(OUTPUT_FOLDER, "labels"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "annotations"), exist_ok=True)

    if parserobj.extract_crops:
        os.makedirs(os.path.join(OUTPUT_FOLDER, "instances"), exist_ok=True)

    if parserobj.copyfiles:
        print("Copying images to {}".format(OUTPUT_FOLDER / Path("images/")))
        pool = multiprocessing.Pool()
        os.makedirs(OUTPUT_FOLDER / Path("images/"), exist_ok=True)
        copyfunc = partial(shutil.copy, dst=OUTPUT_FOLDER / Path("images/"))
        pool.map(copyfunc, img_name_to_loc.values())
        pool.close()

    if os.path.exists(OUTPUT_FOLDER / Path("metadata.json")):
        mergeddata = merge_json(OUTPUT_FOLDER / Path("metadata.json"), datapath)
        with open(OUTPUT_FOLDER / Path("metadata.json"), "w") as jsonfile:
            json.dump(mergeddata, jsonfile, indent=4)
    else:
        shutil.copy(src=datapath, dst=OUTPUT_FOLDER / Path("metadata.json"))

    df_list = []

    # On parcours les objets coco
    for f, coco in ann_files.items():
        # Pour chaque objet coco, on parcours les images
        print(f"Processing annotation file {f}")
        ids_to_cats = {c["id"]: c["name"] for c in coco.cats.values()}
        cats_to_ids = {c["name"]: c["id"] for c in coco.cats.values()}

        for i, imgdata in coco.imgs.items():

            if not os.path.exists(os.path.join(IM_FOLDER, imgdata["file_name"])):
                print(f"Warning: Image {imgdata['file_name']} not found.")
                continue

            shape = (imgdata["height"], imgdata["width"])
            print(
                "Processing image",
                imgdata["file_name"],
                shape,
                "res",
                resolutions.get(imgdata["file_name"], parserobj.res),
            )
            # On charge les annotations de l'image
            annIds = coco.getAnnIds(imgIds=imgdata["id"])
            anns = coco.loadAnns(annIds)
            # On trie par ordre décroissant de surface
            anns = sorted(anns, key=lambda d: d["area"], reverse=True)
            total_anns = len(anns)
            anns = [ann for ann in anns if ann["area"] > parserobj.min_area]
            filtered_anns = len(anns)
            if total_anns - filtered_anns > 0:
                print(total_anns - filtered_anns, "small instances deleted")

            if len(anns) == 0:
                print("annotations are empty")
                continue

            # 1 fichier par image
            instance_data[imgdata["file_name"]] = []
            # Note: il peut y avoir plusieurs contours pour le même id.
            # On récupère les polygones de chaque objet
            # Il faut modifier les ids des objets afin qu'ils commencent à 1 sur chaque image
            polys = {}
            counter = 0
            for i, ann in enumerate(anns):
                for contour in ann["segmentation"]:
                    counter += 1
                    polys[counter] = {
                        "points": (list(zip(np.array(contour[::2]), np.array(contour[1::2])))),
                        "label": i + 1,
                    }
                instance_data[imgdata["file_name"]].append({"label": i + 1})

            # Création de l'image des masques
            label_img = polys_to_bitmap(polys, shape, target_shape=None, dtype=np.uint16)
            maskname = os.path.splitext(imgdata["file_name"])[0] + ".png"
            Image.fromarray(label_img).save(os.path.join(OUTPUT_FOLDER, "labels", maskname), bits=16)
            properties = regionprops(label_img)

            # Création de la base de donnée des objets
            if parserobj.extract_crops:
                image = np.array(Image.open(os.path.join(IM_FOLDER, imgdata["file_name"])))

            for i, ann in enumerate(anns):
                prop = properties[i]
                R = ndi.distance_transform_edt(prop["image"]).max()
                if args.setcats:
                    cls = inst_cats.get(imgdata["file_name"], "UNKNOWN")
                    # cls = cats_to_ids[cls]
                else:
                    cls = ids_to_cats[ann["category_id"]]
                instance_data[imgdata["file_name"]][i].update(
                    {
                        "baseimg": os.path.splitext(imgdata["file_name"])[0],
                        "area": prop["area"],
                        "x0": prop["bbox"][0],
                        "y0": prop["bbox"][1],
                        "x1": prop["bbox"][2],
                        "y1": prop["bbox"][3],
                        "feret": prop["feret_diameter_max"],
                        "radius": R,
                        "class": cls,
                        "res": resolutions.get(imgdata["file_name"], parserobj.res),
                        "mass": "NaN",
                        "gt_mass": False,
                        "height": imgdata["height"],
                        "width": imgdata["width"],
                    }
                )
                # Create a crop for each instance
                if parserobj.extract_crops:
                    fname = "{}-{:06d}.{}".format(os.path.splitext(imgdata["file_name"])[0], prop["label"], "jpg")
                    bbox = prop["bbox"]
                    img_roi = crop(image, bbox)
                    mask_roi = crop(label_img, bbox)
                    mask_roi = mask_roi[..., np.newaxis]
                    # R, G, B = np.random.randint(0, 256, size=3)
                    # BLue color
                    R, G, B = np.array([0, 0, 255]).astype(np.uint8)
                    roi = np.where((mask_roi == prop["label"]) | (mask_roi == 0), img_roi, [R, G, B]).astype(np.uint8)
                    Image.fromarray(roi.astype("uint8")).save(
                        os.path.join(OUTPUT_FOLDER, "instances", fname), quality=100
                    )

            # 1 fichier csv annotation par image avec plusieurs instances
            df_list.append(pandas.json_normalize(instance_data[imgdata["file_name"]]))
            csv_fname = os.path.splitext(imgdata["file_name"])[0] + ".csv"
            df_list[-1].to_csv(
                os.path.join(OUTPUT_FOLDER, "annotations", csv_fname), index=False, index_label="baseimg", na_rep="nan"
            )
            # pandas.json_normalize(
            #     instance_data[imgdata["file_name"]]).to_csv(
            #     os.path.join(OUTPUT_FOLDER, "annotations", csv_fname),
            #     index=False, index_label='label', na_rep='nan')
    if len(df_list) > 1:
        concat_df = pandas.concat(df_list, ignore_index=True)
    else:
        concat_df = df_list[0]
    concat_df = concat_df.sort_values(by=["baseimg", "label"])

    output_csv_file = os.path.join(OUTPUT_FOLDER, "annotations.csv")

    if not args.overwrite:
        if os.path.exists(output_csv_file):
            concat_df.to_csv(output_csv_file, index=False, index_label="baseimg", na_rep="nan", mode="a", header=False)
        else:
            concat_df.to_csv(output_csv_file, index=False, index_label="baseimg", na_rep="nan", mode="a", header=True)
    else:
        concat_df.to_csv(output_csv_file, index=False, index_label="baseimg", na_rep="nan", mode="w", header=True)


if __name__ == "__main__":
    import_COCO(args)
