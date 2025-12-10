---
lang: ENGLISH
---

# Recycled Aggregates Mass Estimation and Segmentation RAMSES

Add mass estimation to SOLOv2  model (Segmenting Objects by LOcations, https://arxiv.org/pdf/2003.10152.pdf)).
Implemented using tensorflow (tf version must be <2.16, because of changes introduced in keras 3+)

## Creating the model

### Config

First, create a config object

    config = RAMSES.Config() # default config

You can also customize the config: see the model/config.py file for all available parameters.

    params = {
    # backbone
    load_backbone:checkpoints/convnextv2_femto_1k_224_ema.pt
    backbone_params:{'build_head': True}
    backbone:convnextv2_femto
    backbone_source:local
    backbone_feature_nodes:{'stages.0': 'C2', 'stages.1': 'C3', 'stages.2': 'C4', 'stages.3': 'C5'}
    ncls:15
    imshape:(2048, 3072, 3)
    mask_stride:8 # It must match the backbone and the param 'output_level'
    # General layers params
    "activation": nn.GELU,
    "normalization": nn.LayerNorm2d,
    "normalization_kw": normalization_kw,
    # FPN connection_layer smust be a dict with the id of the layer as key and its output channels as value
    "connection_layers": connection_layers,  # backbone connection layers,
    "FPN_filters": 256,
    "extra_FPN_layers": 1,  # layers after P5. Strides must correspond to the number of FPN layers !
    # SOLO head
    "strides": [8, 16, 32, 64],  # strides of FPN levels used in the heads [used to compute targets] here P3->P6
    "head_layers": 4,  # Number of repeats of head conv layers
    "head_filters": 256,
    "kernel_size": 1,
    "grid_sizes": [144, 72, 36, 18], # grid size for each head (from high-res to low-res, here P3->P6)
    # SOLO MASK head
    "point_nms": False,
    "mask_mid_filters": 128,
    "mask_output_filters": 256,
    "geom_feat_convs": 4,  # number of convs in the geometry factor branch
    "geom_feats_filters": 128,
    "mask_output_level": 0,  # size of the unified mask (in level of the FPN)
    "FPN_output_upscaling": False, #if True, upscale the FPN fusion
    # For inference
    "use_binary_masks": True,
    "sigma_nms": 0.5,
    "min_area": 0,
    # target allocation
    "scale_ranges":[[1, 128], [64, 256], [128, 512], [256, 3072]],
    "offset_factor":0.75 # full bbox if offset_factor=1, half-box if 0.5
}

     config = ramses2.Config(**params)


### Model creation
just pass the config object to the constructor:

    myRAMSESmodel = ramses2.model.RAMSESModel(config)

When using a custom backbone, you have to put the name of the layers that will be connected to the FPN in the dict `connection_layers`

## Dataset format
The data must be stored in one or (multiple) folders with the following structure
```
├── Dataset1
    ├── annotations.csv
    ├── images
        ├── IM0001.jpg
        ├── ...
    ├── labels
        ├── IM0001.png
        ├── ...
    ├── metadata.json
```

Images are stored in the images folder. 
Masks are stored in labels\  folder (should be 1-channel uint16 uncompressed png). They must have the name basename as their corresponding image.
the `annotations.csv` file must contain at least the following columns:

``` 
label: label of the object in the mask image 
baseimg: base image name e.g. "IM0001", 
x0,y0,x1,y1: coordinates of the bounding box corners (in pixels, format x0y0x1y1)
class: class of the object
res: resolution in pixels/mm
mass: mass in g
gt_mass: True if ground truth mass (individual measure), False if estimated mass
height, width: image dims
```
The file `metadata.json` can be used to compute the mass of each aggregates using the total measured mass of a batch of aggregates, possibly on a series of images (see https://www.sciencedirect.com/science/article/pii/S0166361523000398). Its format is quite explicit. 

To control the creation of train and valid sets, you must create a DatasetManager

A `DatasetGenerator` can be created using the `annotations.csv` file. The `DatasetGenerator` object is useful to create train and test sets from multiple data folders and to balance classes.

The attributes `train_basenames` and `valid_basenames` are lists containing the names of the images in the sets. Once it is done the dataset should be saved as a TFRecord file to be used for training.

See `dataset.ipynb` for an example.

## Training with custom dataset <br>

See `training.ipynb` for an example of model creation and training.

## Inference
A call to the model with a [1, 3, H, W] image returns return a dictionnary [{"masks": seg_preds, "scores": scores, "cls_labels": cls_labels_pos, "masses": masses}] where:
- seg_preds is a tensor of shape [N, H_mask, W_mask] (N the number of predicted instances) 
- scores, cls_labels_pos and densities are tensors of shape [N]

```
Args:
x: input image [B, 3, H, W],
training=False,
cls_threshold=0.5, # threshold for positive detection
nms_threshold=0.3, # iou threshold or cls threshold in MatrixNMX
mask_threshold=0.5,
max_detections=768,
scale_by_mask_scores=False, #scale the class score by mask scores
min_area=32,
nms_mode="greedy"   # can be "greedy" (default), "soft" or "matrix"
```

The predicted masks can sometimes overlap. To get a labeled image use:
'''python
processed_masks = ramses2.decode_predictions(results['masks'], results["scores"], threshold=0.5, by_mask_scores=False)
'''

To visualize the segmented image:
```python
fig = ramses2.plot_instances(
        image,
        processed_masks.cpu().detach().numpy(),,
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
where image must be [H, W, 3]

To run inference on a series of images use the function

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

The function ```ramses2.stream_predict()``` is designed to process an image stream in which aggregates may be truncated or partially visible along the stream direction (the height axis). The function identifies aggregates that are truncated (or split) between two consecutive images. It then reconstructs a single image containing only the complete aggregates by utilizing the overlap region. Ovelap images are created in OUTPUT_FOLDER.
    
## GUI
A small GUI is provided. It predits the composition following the EN 933-11 standard, as well as an estimate of the granulometric curve.