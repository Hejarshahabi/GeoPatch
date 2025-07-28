# -*- coding: utf-8 -*-
"""
Script for patch generation from satellite imagery with preprocessing of DEM and shapefile.
Includes a new function to preprocess and rasterize shapefile for use as label in TrainPatch.
Supports GeoTIFFs with any number of bands and any valid CRS.
"""

import numpy as np
import rasterio as rs
import geopandas as gpd
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from shapely.geometry import box
from rasterio import features
import os
import tqdm
from skimage.measure import label as skimage_label, regionprops

def preprocess_and_rasterize(img_path, shapefile_path, label_field, output_label_path="rasterized_label.tif"):
    """
    Preprocess DEM and shapefile to create a rasterized label aligned with the DEM.
    
    Parameters:
    - img_path (str): Path to the input DEM GeoTIFF.
    - shapefile_path (str): Path to the input shapefile.
    - label_field (str): Field name in shapefile containing integer class IDs.
    - output_label_path (str): Path to save the rasterized label GeoTIFF.
    
    Returns:
    - str: Path to the saved rasterized label GeoTIFF.
    """
    # Step 1: Load DEM and convert extent to WGS84
    with rs.open(img_path) as dem:
        if not dem.crs:
            raise ValueError("DEM CRS is undefined.")
        dem_crs = dem.crs
        dem_bounds = dem.bounds
        dem_width, dem_height = dem.width, dem.height
        dem_transform = dem.transform
        dem_band_count = dem.count

        wgs84_crs = CRS.from_epsg(4326)
        try:
            wgs84_bounds = transform_bounds(dem_crs, wgs84_crs, *dem_bounds)
        except Exception as e:
            raise ValueError(f"Failed to transform DEM bounds to WGS84: {str(e)}")

    # Step 2: Load shapefile and reproject to WGS84
    gdf = gpd.read_file(shapefile_path)
    if not gdf.crs:
        raise ValueError("Shapefile CRS is undefined.")
    if label_field not in gdf.columns:
        raise ValueError(f"Label field '{label_field}' not found in shapefile.")

    # Validate shapefile geometries
    invalid_geoms = gdf[~gdf.geometry.is_valid]
    if not invalid_geoms.empty:
        gdf.geometry = gdf.geometry.buffer(0)

    try:
        gdf_wgs84 = gdf.to_crs(wgs84_crs)
    except Exception as e:
        raise ValueError(f"Failed to reproject shapefile to WGS84: {str(e)}")

    # Step 3: Clip shapefile to DEM extent in WGS84
    wgs84_extent = box(wgs84_bounds[0], wgs84_bounds[1], wgs84_bounds[2], wgs84_bounds[3])
    try:
        gdf_clipped_wgs84 = gdf_wgs84[gdf_wgs84.geometry.intersects(wgs84_extent)].copy()
        if gdf_clipped_wgs84.empty:
            raise ValueError("No polygons intersect with DEM extent in WGS84. Check shapefile extent and CRS.")
    except Exception as e:
        raise ValueError(f"Failed to clip shapefile in WGS84: {str(e)}")

    # Step 4: Reproject clipped shapefile to DEM's original CRS
    try:
        gdf_clipped = gdf_clipped_wgs84.to_crs(dem_crs)
    except Exception as e:
        raise ValueError(f"Failed to reproject clipped shapefile to {dem_crs}: {str(e)}")

    # Step 5: Rasterize clipped shapefile
    label_array = np.zeros((dem_height, dem_width), dtype=np.int32)
    try:
        shapes = [(geom, int(value)) for geom, value in zip(gdf_clipped.geometry, gdf_clipped[label_field])]
        if not shapes:
            raise ValueError("No valid shapes to rasterize. Check label_field values and geometry validity.")
        label_array = features.rasterize(
            shapes,
            out_shape=(dem_height, dem_width),
            transform=dem_transform,
            fill=0,
            dtype=np.int32
        )
    except Exception as e:
        raise ValueError(f"Failed to rasterize shapefile: {str(e)}")

    # Step 6: Save rasterized label as GeoTIFF
    with rs.open(
        output_label_path,
        "w",
        driver="GTiff",
        count=1,
        dtype=np.int32,
        width=dem_width,
        height=dem_height,
        transform=dem_transform,
        crs=dem_crs
    ) as dst:
        dst.write(label_array, 1)

    return output_label_path

class PatchBase:
    """Base class for patch generation from satellite imagery."""
    def __init__(self, image, patch_size, stride, channel_first=True):
        if image is None:
            raise ValueError("Image must be provided.")
        if not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError("Patch size must be a positive integer.")
        if not isinstance(stride, int) or stride <= 0:
            raise ValueError("Stride must be a positive integer.")
        self.image = image
        self.patch_size = patch_size
        self.stride = stride
        self.channel_first = channel_first
        self.img = None
        self.imgarr = None
        self.shape = None

    def readData(self):
        """Read image data and return as NumPy array."""
        if isinstance(self.image, str):
            if self.image.endswith(".tif"):
                if not os.path.exists(self.image):
                    raise FileNotFoundError(f"Image file {self.image} not found.")
                self.img = rs.open(self.image)
                self.shape = (self.img.count, self.img.width, self.img.height)
                self.imgarr = self.img.read()  # Handles any number of bands
            elif self.image.endswith(".npy"):
                if not os.path.exists(self.image):
                    raise FileNotFoundError(f"Image file {self.image} not found.")
                self.imgarr = np.load(self.image)
                self.shape = self.imgarr.shape
            else:
                raise ValueError("Image file must be .tif or .npy.")
        else:
            self.imgarr = self.image
            self.shape = self.image.shape
        if not self.channel_first:
            self.imgarr = np.transpose(self.imgarr, (1, 2, 0))
        return self.imgarr

    def data_dimension(self):
        """Display input data dimensions."""
        self.readData()
        print("############################################")
        print(f"The shape of image is: {self.shape}")
        print("############################################")

    def patch_info(self):
        """Display patch generation details."""
        self.readData()
        x_dim = self.imgarr.shape[1]
        y_dim = self.imgarr.shape[2]
        x_steps = (x_dim - self.patch_size) // self.stride + 1
        y_steps = (y_dim - self.patch_size) // self.stride + 1
        total_patches = x_steps * y_steps
        print("#######################################################################################")
        print(f"The effective X-dimension for patch generation is from 0 to {x_dim}")
        print(f"The effective Y-dimension for patch generation is from 0 to {y_dim}")
        print(f"Total non-augmented patches with patch size {self.patch_size}x{self.patch_size} "
              f"and stride {self.stride} is {total_patches}")
        print("#######################################################################################")
        return total_patches, x_steps, y_steps

    def _extract_patch(self, i, j):
        """Extract a single patch from image."""
        img_patch = self.imgarr[:, i:i+self.patch_size, j:j+self.patch_size]
        if img_patch.shape[1] != self.patch_size or img_patch.shape[2] != self.patch_size:
            temp = np.zeros((self.imgarr.shape[0], self.patch_size, self.patch_size), dtype=self.imgarr.dtype)
            temp[:, :img_patch.shape[1], :img_patch.shape[2]] = img_patch
            img_patch = temp
        return img_patch

class TrainPatch(PatchBase):
    """Generate training patches for semantic segmentation and/or object detection."""
    def __init__(self, image, label, patch_size, stride, channel_first=True, shapefile_path=None, label_field=None):
        """Initialize with image, label, and optional shapefile for preprocessing."""
        super().__init__(image, patch_size, stride, channel_first)
        if label is None and shapefile_path is None:
            raise ValueError("Either label or shapefile_path must be provided for training.")
        if shapefile_path and not label_field:
            raise ValueError("label_field must be provided when shapefile_path is specified.")
        self.label_data = label
        self.shapefile_path = shapefile_path
        self.label_field = label_field
        self.inv = None

    def readData(self):
        """Read image and label data, preprocess shapefile if provided."""
        self.imgarr = super().readData()
        if self.shapefile_path:
            if not os.path.exists(self.shapefile_path):
                raise FileNotFoundError(f"Shapefile {self.shapefile_path} not found.")
            # Preprocess shapefile to create rasterized label
            self.label_data = preprocess_and_rasterize(
                img_path=self.image if isinstance(self.image, str) and self.image.endswith(".tif") else None,
                shapefile_path=self.shapefile_path,
                label_field=self.label_field,
                output_label_path="rasterized_label.tif"
            )
        if isinstance(self.label_data, str):
            if self.label_data.endswith(".tif"):
                if not os.path.exists(self.label_data):
                    raise FileNotFoundError(f"Label file {self.label_data} not found.")
                self.inv = rs.open(self.label_data).read(1)
            elif self.label_data.endswith(".npy"):
                if not os.path.exists(self.label_data):
                    raise FileNotFoundError(f"Label file {self.label_data} not found.")
                self.inv = np.load(self.label_data)
            else:
                raise ValueError("Label file must be .tif or .npy.")
        else:
            self.inv = self.label_data
        if self.imgarr.shape[1:] != self.inv.shape:
            raise ValueError(f"Image {self.imgarr.shape[1:]} and label {self.inv.shape} dimensions must match.")
        return self.imgarr, self.inv

    def data_dimension(self):
        """Display image and label dimensions."""
        self.readData()
        print("############################################")
        print(f"The shape of image is: {self.shape}")
        print(f"The shape of label is: {self.inv.shape}")
        print("############################################")

    def _extract_patch(self, i, j):
        """Extract a single patch from image and label."""
        img_patch = super()._extract_patch(i, j)
        lbl_patch = self.inv[i:i+self.patch_size, j:j+self.patch_size]
        if lbl_patch.shape[0] != self.patch_size or lbl_patch.shape[1] != self.patch_size:
            temp = np.zeros((self.patch_size, self.patch_size), dtype=self.inv.dtype)
            temp[:lbl_patch.shape[0], :lbl_patch.shape[1]] = lbl_patch
            lbl_patch = temp
        return img_patch, lbl_patch

    def _generate_bounding_boxes(self, lbl_patch):
        """Generate YOLO-format bounding box annotations from label patch."""
        bboxes = []
        labeled = skimage_label(lbl_patch > 0)
        for region in regionprops(labeled, intensity_image=lbl_patch):
            class_id = int(np.max(lbl_patch[region.coords[:, 0], region.coords[:, 1]])) - 1
            if class_id < 0:
                continue
            min_row, min_col, max_row, max_col = region.bbox
            x_center = (min_col + max_col) / 2 / self.patch_size
            y_center = (min_row + max_row) / 2 / self.patch_size
            width = (max_col - min_col) / self.patch_size
            height = (max_row - min_row) / self.patch_size
            bboxes.append((class_id, x_center, y_center, width, height))
        return bboxes

    def _save_bounding_boxes(self, bboxes, filepath):
        """Save bounding box annotations in YOLO format."""
        with open(filepath, 'w') as f:
            for bbox in bboxes:
                f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")

    def _apply_segmentation_augmentation(self, img_patch, lbl_patch, aug):
        """Apply augmentation to image and label patches for segmentation."""
        if aug == "V":
            aug_img = np.flip(img_patch, axis=0)
            aug_lbl = np.flip(lbl_patch, axis=0)
        elif aug == "H":
            aug_img = np.flip(img_patch, axis=1)
            aug_lbl = np.flip(lbl_patch, axis=1)
        elif aug in ["90", "180", "270"]:
            k = {"90": 1, "180": 2, "270": 3}[aug]
            aug_img = np.rot90(img_patch, k, axes=(0, 1))
            aug_lbl = np.rot90(lbl_patch, k, axes=(0, 1))
        else:
            aug_img, aug_lbl = img_patch, lbl_patch
        return aug_img, aug_lbl

    def _apply_detection_augmentation(self, img_patch, lbl_patch, aug):
        """Apply augmentation to image, label, and bounding boxes for detection."""
        aug_bboxes = []
        if aug == "V":
            aug_img = np.flip(img_patch, axis=0)
            aug_lbl = np.flip(lbl_patch, axis=0)
            aug_bboxes = [(b[0], b[1], 1 - b[2], b[3], b[4]) for b in self._generate_bounding_boxes(lbl_patch)]
        elif aug == "H":
            aug_img = np.flip(img_patch, axis=1)
            aug_lbl = np.flip(lbl_patch, axis=1)
            aug_bboxes = [(b[0], 1 - b[1], b[2], b[3], b[4]) for b in self._generate_bounding_boxes(lbl_patch)]
        elif aug in ["90", "180", "270"]:
            k = {"90": 1, "180": 2, "270": 3}[aug]
            aug_img = np.rot90(img_patch, k, axes=(0, 1))
            aug_lbl = np.rot90(lbl_patch, k, axes=(0, 1))
            orig_bboxes = self._generate_bounding_boxes(lbl_patch)
            for b in orig_bboxes:
                class_id, x_c, y_c, w, h = b
                if aug == "90":
                    x_new, y_new = y_c, 1 - x_c
                    w_new, h_new = h, w
                elif aug == "180":
                    x_new, y_new = 1 - x_c, 1 - y_c
                    w_new, h_new = w, h
                elif aug == "270":
                    x_new, y_new = 1 - y_c, x_c
                    w_new, h_new = h, w
                aug_bboxes.append((class_id, x_new, y_new, w_new, h_new))
        else:
            aug_img, aug_lbl = img_patch, lbl_patch
            aug_bboxes = self._generate_bounding_boxes(lbl_patch)
        return aug_img, aug_lbl, aug_bboxes

    def generate_segmentation(self, format="npy", folder_name="seg_data", only_label=True,
                             return_stacked=True, save_stack=True, V_flip=True, H_flip=True, Rotation=True):
        """Generate patches for semantic segmentation."""
        self.readData()
        total_patches, x_steps, y_steps = self.patch_info()
        os.makedirs(f"{folder_name}/patch", exist_ok=True)
        os.makedirs(f"{folder_name}/label", exist_ok=True)
        augmentations = [0]
        if format == "npy" and V_flip:
            augmentations.append("V")
        if format == "npy" and H_flip:
            augmentations.append("H")
        if format == "npy" and Rotation:
            augmentations.extend(["90", "180", "270"])
        patch_counter = 0
        x, y = [], []
        valid_patches = total_patches if not only_label else sum(
            1 for i in range(0, x_steps * self.stride, self.stride)
            for j in range(0, y_steps * self.stride, self.stride)
            if self.inv[i:i+self.patch_size, j:j+self.patch_size].any()
        )
        with tqdm.tqdm(total=valid_patches * len(augmentations), desc="Patch Counter", unit="Patch") as pbar:
            index = 1
            for i in range(0, x_steps * self.stride, self.stride):
                for j in range(0, y_steps * self.stride, self.stride):
                    img_patch, lbl_patch = self._extract_patch(i, j)
                    if only_label and not lbl_patch.any():
                        continue
                    img_patch_save = np.transpose(img_patch, (1, 2, 0))
                    if format == "tif":
                        x_cord = j * self.img.transform[0] + self.img.transform[2]
                        y_cord = self.img.transform[5] + i * self.img.transform[4]
                        transform = [self.img.transform[0], 0, x_cord, 0, self.img.transform[4], y_cord]
                        with rs.open(
                            f"{folder_name}/patch/{index}_img.tif", "w", driver="GTiff",
                            count=self.imgarr.shape[0], dtype=self.imgarr.dtype,
                            width=self.patch_size, height=self.patch_size, transform=transform, crs=self.img.crs
                        ) as raschip:
                            raschip.write(np.transpose(img_patch_save, (2, 0, 1)))
                        with rs.open(
                            f"{folder_name}/label/{index}_lbl.tif", "w", driver="GTiff",
                            count=1, dtype=self.inv.dtype, width=self.patch_size, height=self.patch_size,
                            transform=transform, crs=self.img.crs
                        ) as lblchip:
                            lblchip.write(lbl_patch, 1)
                    else:
                        np.save(f"{folder_name}/patch/{index}_img.npy", img_patch_save)
                        np.save(f"{folder_name}/label/{index}_lbl.npy", lbl_patch)
                    patch_counter += 1
                    if return_stacked:
                        x.append(img_patch_save)
                        y.append(lbl_patch)
                    pbar.update(1)
                    if format == "npy":
                        for aug in augmentations[1:]:
                            aug_img, aug_lbl = self._apply_segmentation_augmentation(img_patch_save, lbl_patch, aug)
                            np.save(f"{folder_name}/patch/{index}_img_{aug}.npy", aug_img)
                            np.save(f"{folder_name}/label/{index}_lbl_{aug}.npy", aug_lbl)
                            patch_counter += 1
                            if return_stacked:
                                x.append(aug_img)
                                y.append(aug_lbl)
                            pbar.update(1)
                    index += 1
        percentage = int((patch_counter / (total_patches * len(augmentations))) * 100)
        print(f"{patch_counter} patches ({percentage}% of total) saved as .{format} in {os.getcwd()}\\{folder_name}")
        if return_stacked:
            patch_stacked = np.array(x, dtype="float32")
            label_stacked = np.array(y, dtype="int")
            if save_stack:
                np.save(f"{folder_name}/Patch_stacked_{self.patch_size}.npy", patch_stacked)
                np.save(f"{folder_name}/label_stacked_{self.patch_size}.npy", label_stacked)
            print(f"Stacked patches and labels saved with shapes: {patch_stacked.shape, label_stacked.shape}")
            return patch_stacked, label_stacked

    def generate_detection(self, format="npy", folder_name="det_data", only_label=True,
                          return_stacked=True, save_stack=True, V_flip=True, H_flip=True,
                          Rotation=True, segmentation=True):
        """Generate patches for object detection with optional segmentation masks."""
        self.readData()
        total_patches, x_steps, y_steps = self.patch_info()
        os.makedirs(f"{folder_name}/patch", exist_ok=True)
        os.makedirs(f"{folder_name}/bbox", exist_ok=True)
        if segmentation:
            os.makedirs(f"{folder_name}/label", exist_ok=True)
        augmentations = [0]
        if format == "npy" and V_flip:
            augmentations.append("V")
        if format == "npy" and H_flip:
            augmentations.append("H")
        if format == "npy" and Rotation:
            augmentations.extend(["90", "180", "270"])
        patch_counter = 0
        x, y = [], []
        valid_patches = total_patches if not only_label else sum(
            1 for i in range(0, x_steps * self.stride, self.stride)
            for j in range(0, y_steps * self.stride, self.stride)
            if self.inv[i:i+self.patch_size, j:j+self.patch_size].any()
        )
        with tqdm.tqdm(total=valid_patches * len(augmentations), desc="Patch Counter", unit="Patch") as pbar:
            index = 1
            for i in range(0, x_steps * self.stride, self.stride):
                for j in range(0, y_steps * self.stride, self.stride):
                    img_patch, lbl_patch = self._extract_patch(i, j)
                    if only_label and not lbl_patch.any():
                        continue
                    img_patch_save = np.transpose(img_patch, (1, 2, 0))
                    bboxes = self._generate_bounding_boxes(lbl_patch)
                    if format == "tif":
                        x_cord = j * self.img.transform[0] + self.img.transform[2]
                        y_cord = self.img.transform[5] + i * self.img.transform[4]
                        transform = [self.img.transform[0], 0, x_cord, 0, self.img.transform[4], y_cord]
                        with rs.open(
                            f"{folder_name}/patch/{index}_img.tif", "w", driver="GTiff",
                            count=self.imgarr.shape[0], dtype=self.imgarr.dtype,
                            width=self.patch_size, height=self.patch_size, transform=transform, crs=self.img.crs
                        ) as raschip:
                            raschip.write(np.transpose(img_patch_save, (2, 0, 1)))
                        if segmentation:
                            with rs.open(
                                f"{folder_name}/label/{index}_lbl.tif", "w", driver="GTiff",
                                count=1, dtype=self.inv.dtype, width=self.patch_size, height=self.patch_size,
                                transform=transform, crs=self.img.crs
                            ) as lblchip:
                                lblchip.write(lbl_patch, 1)
                    else:
                        np.save(f"{folder_name}/patch/{index}_img.npy", img_patch_save)
                        if segmentation:
                            np.save(f"{folder_name}/label/{index}_lbl.npy", lbl_patch)
                    self._save_bounding_boxes(bboxes, f"{folder_name}/bbox/{index}_bbox.txt")
                    patch_counter += 1
                    if return_stacked:
                        x.append(img_patch_save)
                        if segmentation:
                            y.append(lbl_patch)
                    pbar.update(1)
                    if format == "npy":
                        for aug in augmentations[1:]:
                            aug_img, aug_lbl, aug_bboxes = self._apply_detection_augmentation(img_patch_save, lbl_patch, aug)
                            np.save(f"{folder_name}/patch/{index}_img_{aug}.npy", aug_img)
                            if segmentation:
                                np.save(f"{folder_name}/label/{index}_lbl_{aug}.npy", aug_lbl)
                            self._save_bounding_boxes(aug_bboxes, f"{folder_name}/bbox/{index}_bbox_{aug}.txt")
                            patch_counter += 1
                            if return_stacked:
                                x.append(aug_img)
                                if segmentation:
                                    y.append(aug_lbl)
                            pbar.update(1)
                    index += 1
        percentage = int((patch_counter / (total_patches * len(augmentations))) * 100)
        print(f"{patch_counter} patches ({percentage}% of total) saved as .{format} in {os.getcwd()}\\{folder_name}")
        if return_stacked:
            patch_stacked = np.array(x, dtype="float32")
            label_stacked = np.array(y, dtype="int") if segmentation and y else None
            if save_stack:
                np.save(f"{folder_name}/Patch_stacked_{self.patch_size}.npy", patch_stacked)
                if segmentation and label_stacked is not None:
                    np.save(f"{folder_name}/label_stacked_{self.patch_size}.npy", label_stacked)
            print(f"Stacked patches and labels saved with shapes: {patch_stacked.shape, label_stacked.shape}")
            return patch_stacked, label_stacked if segmentation else patch_stacked

    def visualize(self, folder_name="seg_data", patches_to_show=1, band_num=1, fig_size=(10, 20), dpi=96, show_bboxes=False):
        """Visualize random patches with optional bounding boxes."""
        import matplotlib.pyplot as plt
        import glob
        patch_dir = sorted(glob.glob(os.path.join(os.getcwd(), folder_name, "patch/*")))
        label_dir = sorted(glob.glob(os.path.join(os.getcwd(), folder_name, "label/*")))
        bbox_dir = sorted(glob.glob(os.path.join(os.getcwd(), folder_name, "bbox/*"))) if show_bboxes else []
        if not patch_dir:
            raise ValueError(f"No patches found in {folder_name}.")
        if label_dir and len(patch_dir) != len(label_dir):
            raise ValueError(f"Mismatched patch/label counts in {folder_name}.")
        if show_bboxes and len(patch_dir) != len(bbox_dir):
            raise ValueError(f"Mismatched patch/bbox counts in {folder_name}.")
        patches_to_show = min(patches_to_show, len(patch_dir))
        idx = np.random.choice(len(patch_dir), patches_to_show, replace=False)
        print("Displaying patches:")
        fig, ax = plt.subplots(patches_to_show, 2 if label_dir else 1, figsize=fig_size, dpi=dpi)
        if patches_to_show == 1:
            ax = [ax] if label_dir else [[ax]]
        elif not label_dir:
            ax = [[a] for a in ax]
        for i, index in enumerate(idx):
            file = patch_dir[index]
            print(file)
            if file.endswith(".tif"):
                img = rs.open(file).read()
                img_display = img[band_num - 1 if band_num > 0 else -1]
            else:
                img = np.load(file)
                img_display = img[:, :, band_num - 1 if band_num > 0 else -1]
            ax[i][0].imshow(img_display)
            ax[i][0].set_title("Image Patch")
            if label_dir:
                file_ = label_dir[index]
                print(file_)
                if file_.endswith(".tif"):
                    lbl = rs.open(file_).read(1)
                else:
                    lbl = np.load(file_)
                ax[i][1].imshow(lbl)
                ax[i][1].set_title("Label Patch")
                if show_bboxes and bbox_dir:
                    bbox_file = bbox_dir[index]
                    print(bbox_file)
                    with open(bbox_file, 'r') as f:
                        bboxes = [list(map(float, line.strip().split())) for line in f]
                    for bbox in bboxes:
                        class_id, x_c, y_c, w, h = map(float, bbox)
                        x_min = (x_c - w / 2) * self.patch_size
                        y_min = (y_c - h / 2) * self.patch_size
                        w_px = w * self.patch_size
                        h_px = h * self.patch_size
                        rect = plt.Rectangle((x_min, y_min), w_px, h_px, edgecolor='red', facecolor='none', linewidth=2)
                        ax[i][0].add_patch(rect)
                        ax[i][0].text(x_min, y_min, f"Class {int(class_id)}", color='red', fontsize=8)
        plt.tight_layout()
        plt.show()

class PredictionPatch(PatchBase):
    """Generate patches for prediction from satellite imagery."""
    def save_Geotif(self, folder_name="tif"):
        """Save prediction patches as GeoTIFF files."""
        self.readData()
        total_patches, x_steps, y_steps = self.patch_info()
        os.makedirs(f"{folder_name}/Prediction_patch", exist_ok=True)
        patch_counter = 0
        index = 1
        with tqdm.tqdm(total=total_patches, desc="Patch Counter", unit="Patch") as pbar:
            for i in range(0, x_steps * self.stride, self.stride):
                for j in range(0, y_steps * self.stride, self.stride):
                    img_patch = self._extract_patch(i, j)
                    x_cord = j * self.img.transform[0] + self.img.transform[2]
                    y_cord = self.img.transform[5] + i * self.img.transform[4]
                    transform = [self.img.transform[0], 0, x_cord, 0, self.img.transform[4], y_cord]
                    with rs.open(
                        f"{folder_name}/Prediction_patch/{index}_img.tif", "w", driver="GTiff",
                        count=self.imgarr.shape[0], dtype=self.imgarr.dtype,
                        width=self.patch_size, height=self.patch_size, transform=transform, crs=self.img.crs
                    ) as raschip:
                        raschip.write(img_patch)
                    patch_counter += 1
                    index += 1
                    pbar.update(1)
        percentage = int((patch_counter / total_patches) * 100)
        print(f"{patch_counter} patches ({percentage}% of total) saved as .tif in {os.getcwd()}\\{folder_name}")

    def save_numpy(self, folder_name="npy", return_stacked=True, save_stack=True):
        """Save prediction patches as NumPy arrays."""
        self.readData()
        total_patches, x_steps, y_steps = self.patch_info()
        os.makedirs(f"{folder_name}/Prediction_patch", exist_ok=True)
        patch_counter = 0
        x = []
        with tqdm.tqdm(total=total_patches, desc="Patch Counter", unit="Patch") as pbar:
            index = 1
            for i in range(0, x_steps * self.stride, self.stride):
                for j in range(0, y_steps * self.stride, self.stride):
                    img_patch = self._extract_patch(i, j)
                    img_patch = np.transpose(img_patch, (1, 2, 0))
                    np.save(f"{folder_name}/Prediction_patch/{index}_img.npy", img_patch)
                    patch_counter += 1
                    if return_stacked:
                        x.append(img_patch)
                    pbar.update(1)
                    index += 1
        percentage = int((patch_counter / total_patches) * 100)
        print(f"{patch_counter} patches ({percentage}% of total) saved as .npy in {os.getcwd()}\\{folder_name}")
        if return_stacked:
            patch_stacked = np.array(x, dtype="float32")
            if save_stack:
                np.save(f"{folder_name}/Patch_stacked_{self.patch_size}.npy", patch_stacked)
            print(f"Stacked patches saved with shape: {patch_stacked.shape}")
            return patch_stacked
