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

You can also customize the config:
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

Images are stored in the images folder (must be jpg). 
Masks are stored in labels\  folder (should be 1-channel uint16 uncompressed png). They must have the name basename as their corresponding image.
the `annotations.csv` file must contain at least the following columns:

``` 
label: label of the object in the mask image 
baseimg: base image name e.g. "IM0001", 
x0,y0,x1,y1: coordinates of the bounding box corners (in pixels, format x0y0x1y1)
class: class o fthe object
res: resolution in pixels/mm
mass: mass in g
gt_mass: True if ground truth mass (individual measure), False if estimated mass
height, width: image dims
```
The file `metadata.json` can be used to compute the mass of each aggregates using the total measured mass of a batch of aggregates, possibly on a series of images (see https://www.sciencedirect.com/science/article/pii/S0166361523000398). Its format is quite explicit. 

To control the creation of train and valid sets, you must create a DatasetManager

A `DatasetGenerator` can be created using the `annotations.csv` file. The `DatasetGenerator` object is useful to create train and test sets from multiple data folders and to balance classes.

The attributes `train_basenames` and `valid_basenames` are lists containing the names of the images in the sets. Once it is done the dataset should be saved as a TFRecord file to be used for training.

See `datasetpynb` for an example.

## Training with custom dataset <br>

See `training.ipynb` for an example of model creation and training.

## Inference
A call to the model with a [1, H, W, 3] image returns the N masks tensor (one slice per instance [1, N, H/2, W/2]) and corresponding classes [1, N] and scores [1, N] and normalized masses [1, N]. <br>
The model **ALWAYS** return ragged tensors, and should work with batchsize > 1.
The final labeled prediction can be obtained by the RAMSES.utils.decode_predictions function

    default_kwargs = {
        "score_threshold": 0.5,
        "seg_threshold": 0.5,
        "nms_threshold": 0.5,
        "max_detections": 400,
        "point_nms": False,
        "use_binary_masks":True,
        "min_area": 0,
    }

    seg_peds, scores, cls_labels, norm_masses = myRAMSESmodel(input, **default_kwargs)
    labeled_masks = RAMSES.utils.decode_predictions function(seg_preds, scores, score_threshold=0.5, by_scores=False, )

Results can be vizualised using the RAMSES.visualization.draw_instances function:

    
    img = RAMSES.visualization.draw_instances(input, 
            labeled_masks.numpy(), 
            cls_ids=cls_labels[0,...].numpy() + 1, 
            cls_scores=scores[0,...].numpy(), 
            class_ids_to_name=id_to_cls, 
            show=True, 
            fontscale=0., 
            fontcolor=(0,0,0),
            alpha=0.5, 
            thickness=0)

or using matplotlib:

    cls_ids = [idx_to_cls[id + 1] for id in cls_labels|0, ...].numpy()]
    fig = ISMENet.plot_instances(
                    input,
                    labeled_masks.numpy()[0, ...],
                    cls_ids=cls_ids,
                    cls_scores=scores.numpy()[0,...],
                    alpha=0.2,
                    fontsize=2.5,
                    fontcolor="black",
                    draw_boundaries=True,
                    dpi=300,
                    show=False,
    )
    plt.show()


Note that all inputs to this function must have a batch dimension and should be converted to numpy arrays.
    
