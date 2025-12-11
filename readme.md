# Recycled Aggregates Mass Estimation and Segmentation (RAMSES)

RAMSES is a neural network model for instance segmentation and mass estimation of recycled aggregates in construction and demolition waste (CDW) images.

It is based on the SOLOv2 instance segmentation model (Segmenting Objects by LOcations — https://arxiv.org/pdf/2003.10152.pdf).

<div style="text-align: center;">
    <img src="./images/model.jpg" alt="RAMSES architecture" width="75%">
</div>

<img src="./images/SEG_P20220504_00119.png" alt="example segmentation 1" width="49%">
<img src="./images/SEG_P20240507_00019.png" alt="example segmentation 2" width="49%">

## Recycled Aggregates Dataset

The link to the Recycled Aggregate Database will be available soon.  
It contains **90,000 instances** of labeled aggregates, as well as corresponding mass data.

It features 18 classes of recycled aggregates:

| Class ID | Instances | Description |
|----------|-----------|-------------|
| Pl       | 55        | Plaster     |
| Ra       | 7,309     | Bituminous grains |
| Rb01     | 7,795     | Terracotta material, such as clay bricks or roof tiles |
| Rb02     | 3,742     | Ceramic tiles, earthenware tiles, etc. |
| Rc       | 26,910    | Concrete grains |
| Rcu01    | 547       | Lime mortar |
| Rg       | 145       | Glass       |
| Ru01     | 14,245    | White stones such as limestone |
| Ru02     | 10,614    | Grey stones such as basalt and similar grainy stones |
| Ru04     | 5,683     | Siliceous and rather angular stones |
| Ru05     | 5,993     | Rounded alluvial stones |
| Ru06     | 2,891     | Slate       |
| X01      | 1,797     | Wood        |
| X02      | 586       | Plastics    |
| X03      | 769       | Steel       |
| X04      | 133       | Paper and cardboard |
| SHELLS   | 48        | Shells      |
| UNKNOWN   | 28        | Does not belong to other categories |
| **TOTAL** | **89,600** |             |

## Quick Start

### Load an Existing Model

A trained model checkpoint is available in `/checkpoints/2048x3072`.

```python
with open(Path("../checkpoints/2048x3072/config.json"), 'r') as jsonfile:
    params = json.load(jsonfile)
config = ramses2.Config(**params)
model = ramses2.RAMSESModel(config)
```

### Inference

A forward pass with a [1, 3, H, W] image returns a dictionary:
```python
{
  "masks": seg_preds,   # [N, H_mask, W_mask]
  "scores": scores,     # [N]
  "cls_labels": cls_ids,# [N]
  "masses": masses      # [N]
}
```

Model call signature:

```python
results = model(
    x,                  # [B, 3, H, W]
    training=False,
    cls_threshold=0.5,
    nms_threshold=0.3,
    mask_threshold=0.5,
    max_detections=768,
    scale_by_mask_scores=False,
    min_area=32,
    nms_mode="greedy"   # "greedy", "soft", or "matrix"
)
```

Predicted masks may overlap. To obtain a labeled image (non-overlapping instance map):

```python
processed_masks = ramses2.decode_predictions(results["masks"], 
                                             results["scores"], 
                                             threshold=0.5, 
                                             by_mask_scores=False)
```

Two functions can be used to automate predictions on a folder of images: use `predict()` to run inference on each image and `stream_predict()` to process image streams where objects may be split across consecutive frames. The function identifies truncated objects, reconstructs complete instances using overlap regions, and saves overlap images to the OUTPUT_FOLDER.

Example:

```python
results = ramses2.predict(
    output_dir=OUTPUT_DIR,
    input_size=imshape,
    resolution=28.7,
    input_dir=INPUT_DIR,
    model=model,
    idx_to_cls=idx_to_cls,
    thresholds=(0.5, 0.5, 0.25),
    crop_to_ar=True,
    max_detections=768,
    minarea=8,
    subdirs=False,
    save_imgs="class",
    device="cuda:0",
)
```

### Visualization

```python
fig = ramses2.plot_instances(
    image,                          # [H, W, 3]
    processed_masks.cpu().numpy(),
    cls_ids=pred_cls_labels,
    cls_scores=pred_scores,
    alpha=0.4,
    fontsize=3,
    fontcolor="black",
    draw_boundaries=True,
    boundary_mode="inner",
    dpi=200,
    show=False,
    x_offset=20,
    y_offset=10,
)
```

## Advanced Configuration

Create a config object:

```python
config = RAMSES.Config()  # default config
```

Or customize parameters (see `model/config.py` for all options). Example parameters:
```python
    params = {
        # backbone
        "load_backbone": "checkpoints/convnextv2_femto_1k_224_ema.pt",
        "backbone_params": {"build_head": True},
        "backbone": "convnextv2_femto",
        "backbone_source": "local",
        # backbone layers to connect to FPN
        "backbone_feature_nodes": {"stages.0": "C2", "stages.1": "C3", "stages.2": "C4", "stages.3": "C5"},
        "ncls": 15,
        "imshape": (2048, 3072, 3),
        "mask_stride": 8,  # must match backbone and 'output_level'
        # general layers
        "activation": nn.GELU,
        "normalization": nn.LayerNorm2d,
        "normalization_kw": {"eps":1e-6},
        # FPN connections: dict[layer_name] = out_channels
        "connection_layers": {"C3": 96, "C4": 192, "C5": 384},
        "FPN_filters": 256,
        "extra_FPN_layers": 1,  # layers after P5
        # SOLO head
        "strides": [8, 16, 32, 64],  # FPN head strides (P3->P6)
        "head_layers": 4,
        "head_filters": 256,
        "kernel_size": 1,
        "grid_sizes": [144, 72, 36, 18],
        # mask head
        "point_nms": False,
        "mask_mid_filters": 128,
        "mask_output_filters": 256,
        "geom_feat_convs": 4,
        "geom_feats_filters": 128,
        "mask_output_level": 0, # Level of the FPN where the mask is built from
        "FPN_output_upscaling": False, # Upscale the mask_output_level
        # inference
        "use_binary_masks": True,
        "sigma_nms": 0.5,# only for MarixNMS
        "min_area": 0,
        # target allocation
        "scale_ranges": [[1, 128], [64, 256], [128, 512], [256, 3072]],
        "offset_factor": 0.75,
    }

config = ramses2.Config(**params)
```

### Model Creation

Instantiate the model with the config:

```python
model = ramses2.model.RAMSESModel(config)
```

If using a custom backbone, set `connection_layers` to map backbone layer names to FPN channels.

## Dataset Layout

Expected dataset folder structure:

```
Dataset/
├─ annotations.csv
├─ images/
│  ├─ IM0001.jpg
│  └─ ...
└─ labels/
   ├─ IM0001.png   # 1-channel uint16, uncompressed PNG
   └─ ...
```

`annotations.csv` must include at least:

- label: mask object ID
- baseimg: image basename (e.g. "IM0001")
- x0, y0, x1, y1: bounding box coordinates (pixels)
- class: object class
- res: resolution (pixels/mm)
- mass: mass in grams
- gt_mass: True if mass is a ground-truth individual measurement
- height, width: image dimensions

`metadata.json` (optional) can store batch-level measures used to compute per-instance mass (see https://doi.org/10.1016/j.compind.2023.103889 and https://doi.org/10.1016/j.autcon.2020.103204).

To create train/validation splits, use the `DatasetManager` class. See `dataset.ipynb` for more details.

## Utilities

In the `scripts/` folder, there are three Python scripts to recompute the mass of the dataset and to convert `annotations.csv` to or from the COCO format.

## Training

See `training.ipynb` for examples of model creation and training workflows.

## GUI

A simple GUI is included for EN 933-11 composition estimation and granulometric curve approximation. Just run `python ramses2/GUI/gui.py`.
