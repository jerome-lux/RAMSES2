<!-- filepath: c:\Users\jerom\Nextcloud\Codes\python3\RAMSES2\datasets\readme.md -->
This folder contains the two lists of images used to train the model.

The `15CLS_20250723-173206` dataset consists of a JSON file listing the images for training and validation sets, and a CSV file containing all instance information (e.g., classes, labels, bounding box, mass). You must update `/PATH/TO/THE/DATA2/ONDISK` to match your specific local path.

The images originate from:
1.  The Recycled Aggregate Database (https://doi.org/10.57745/KC4EA2), referred to as "RASET" in the annotations.
2.  Synthetic images generated from existing instances ("SYNTH15CLS"). The synthetic image dataset (images + masks) weighs 1.2 GB and can be made available upon request.

The `15CLS_20250723-173206_MASS_ONLY` dataset is a subset of `15CLS_20250723-173206`, excluding instances without mass information. It was used to train specifically the new mass heads in RAMSES.

To create a `DatasetManager` object with these files, use:
```python
datamanager = ramses2.DatasetManager.from_file(
    annfile="15CLS_20250723-173206.csv", filename="15CLS_20250723-173206.json")
```
The DatasetManager class is useful to generate new train/valid splits, to control oversampling, etc.

To create a torch.Dataset with augmentations:

```python
train_dataset = ramses2.torchDataset(
    datamanager.annotations,
    datamanager.train_basenames,
    input_shape=target_shape,
    mask_stride=mask_stride,
    cls_to_idx=cls_to_idx,
    transform=ramses2.TorchAugmentations(probability=[0.33, 0.33, 0.33], factor=0.3, seed=0),
    crop_to_aspect_ratio=True,
    random_resize_method=True,
    seed=1,
)
```
where `cls_to_idx` is the dictionary mapping the class names to their value (beginning at 1)

You can use the torchDataset object to generate a torch.Dataloader. Use either

```python
collate_fn = partial(ramses2.collate_fn_cutmix, p=0.5, mask_stride=config.mask_stride, max_patches=3, min_patch_ratio=0.1, max_patch_ratio=0.25)
```
to use a CutMix augmentation or: 

```python
collate_fn =  ramses2.collate_fn
```
Then you can create the torch.Dataloader:

```python
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn, pin_memory=True
)
```