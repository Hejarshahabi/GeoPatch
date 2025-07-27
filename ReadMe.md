# GeoPatch

## *GeoPatch* is a package for generating patches from remote sensing data [![PyPI version](https://img.shields.io/badge/PyPi%20Package-1.2-green)](https://pypi.org/project/GeoPatch/) [![Downloads](https://pepy.tech/badge/geopatch)](https://pepy.tech/project/geopatch) [![Github](https://img.shields.io/badge/Github-GeoPatch-blueviolet)](https://github.com/Hejarshahabi/GeoPatch) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Hejar%20Shahabi-blue)](https://www.linkedin.com/in/hejarshahabi/) [![Twitter URL](https://img.shields.io/twitter/url?color=blue&label=Hejar%20Shahabi&style=social&url=https%3A%2F%2Ftwitter.com%2Fhejarshahabi)](https://twitter.com/hejarshahabi)

*GeoPatch* enables users to read, process, and export GeoTIFFs in various patch sizes. Built on the Rasterio library, it simplifies reading and exporting GeoTIFF patches for training deep learning models. The package supports both semantic segmentation and object detection, generating patches in GeoTIFF or NumPy formats with optional YOLO-format bounding box annotations.

Users can feed satellite imagery and corresponding label data to export patches, with support for data augmentation (vertical flip, horizontal flip, and 90/180/270-degree rotations) for NumPy output. The package ensures complete image coverage, including edge pixels, and supports visualization of patches.

Any feedback is welcome! Contact me at hejarshahabi@gmail.com for contributions or suggestions.

<img src="https://github.com/Hejarshahabi/GeoPatch/blob/main/Patch_logo.jpg?raw=true" width="880" height="325">

## Quick Tutorial on How to Use GeoPatch

### 1. Installation

```bash
pip install GeoPatch
```

This automatically installs dependencies, including GDAL, for handling GeoTIFF files.

### 2. Calling the Package

```bash
from GeoPatch import TrainPatch, PredictionPatch
```

### 3. Feeding Data

For both `image` and `label` variables, you can pass either a string (file path to `.tif` or `.npy`) or a NumPy array. The package automatically processes and reads the dataset.

```bash
patch = TrainPatch(image="xxx/image.tif", label="xxx/label.tif", patch_size=128, stride=64, channel_first=True)
```

### 4. Input Data Specifications

Display the shape and size of the input data:

```bash
patch.data_dimension()
```

### 5. Patch Details

Show the number of original image patches generated based on the given patch size and stride:

```bash
patch.patch_info()
```

### 6. Saving Image Patches as GeoTIFF Files (Segmentation)

Save image and label patches as GeoTIFF files in the specified `folder_name` under the current working directory. If `only_label=True`, only patches with non-zero labels are saved.

```bash
patch.generate_segmentation(format="tif", folder_name="seg_tif", only_label=True)
```

### 7. Saving Image Patches as NumPy Arrays (Segmentation)

Generate image and label patches in NumPy format with optional data augmentation (vertical flip, horizontal flip, 90/180/270-degree rotations). Augmentations are applied only for `format="npy"`.

```bash
patch.generate_segmentation(
    format="npy",
    folder_name="seg_npy",
    only_label=False,
    return_stacked=False,
    save_stack=False,
    V_flip=True,
    H_flip=True,
    Rotation=True
)

# To return stacked NumPy patches:
patch_stacked, label_stacked = patch.generate_segmentation(
    format="npy",
    folder_name="seg_npy",
    only_label=False,
    return_stacked=True,
    save_stack=False,
    V_flip=True,
    H_flip=True,
    Rotation=True
)
```

### 8. Saving Patches for Object Detection (YOLO Format)

Generate patches for object detection, producing image patches, optional segmentation masks, and YOLO-format bounding box annotations (`.txt` files). Augmentations are applied only for `format="npy"`.

```bash
patch.generate_detection(
    format="npy",
    folder_name="det_npy",
    only_label=True,
    return_stacked=False,
    save_stack=False,
    V_flip=True,
    H_flip=True,
    Rotation=True,
    segmentation=True
)
```

### 9. Patch Visualization

Display patches with their corresponding labels or bounding boxes. Specify the exact `folder_name` where patches are saved. Use `show_bboxes=True` to overlay YOLO bounding boxes.

```bash
patch.visualize(
    folder_name="seg_npy",
    patches_to_show=2,
    band_num=1,
    fig_size=(10, 20),
    dpi=96,
    show_bboxes=False
)
```

### 10. Generating Prediction Patches

Generate patches for prediction using the `PredictionPatch` class:

```bash
prediction = PredictionPatch(image="xxx/test_image.tif", patch_size=128, stride=128, channel_first=True)

```

### 11. Saving Prediction Patches

Save prediction patches as GeoTIFF or NumPy arrays. Edge pixels are included to ensure complete image coverage.

```bash
# Save as GeoTIFF
prediction.save_Geotif(folder_name="pred_tif")

# Save as NumPy arrays
prediction.save_numpy(folder_name="pred_npy")
```

## Change Log

### 1.0 (04/07/2022)
- First Release

### 1.1 (03/08/2022)
- Fixed issues with loading NumPy arrays
- Fixed random visualization of samples

### 1.1.1 (15/12/2022)
- Fixed visualization issues in Linux environments
- Added `PredictionPatch` class for generating prediction patches

### 1.1.1 (22/11/2023)
- Fixed edge pixel issue in prediction patch generation to ensure entire image is patched
- Added GDAL to automatically installed packages

### 1.2 (27/07/2025)
- Added `generate_detection` method to `TrainPatch` for object detection with YOLO-format bounding box annotations
- Modified `generate_segmentation` and `generate_detection` to apply augmentations (V_flip, H_flip, Rotation) only for `format="npy"`, not for `format="tif"`, to prevent shape mismatch errors