
# -*- coding: utf-8 -*-
"""
Created on Dec 15, 2022
Updated on Jul 28, 2025

@author: Hejar Shahabi
@email: hejarshahabi@gmail.com

Optimized patch generation for satellite imagery with support for semantic segmentation,
object detection, or both. Outputs GeoTIFF or NumPy patches with segmentation masks
and/or YOLO-format bounding box annotations. Includes support for generating labels from shapefiles.
"""

import numpy as np
import rasterio as rs
import geopandas as gpd
import glob
import os
import math
import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from rasterio import features
from shapely.geometry import box
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from skimage.measure import label as skimage_label, regionprops

class PatchBase:
    """Base class for patch generation from satellite imagery."""
    
    def __init__(self, image, patch_size, stride, channel_first=True):
        """Initialize patch generator with image and parameters.

        Args:
            image: NumPy array or string path to satellite imagery (.tif or .npy).
            patch_size: Integer, size of square patches (e.g., 128 for 128x128).
            stride: Integer, step size for sliding window (overlap if < patch_size).
            channel_first: Boolean, True if channels are first dimension (default: True).
        """
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
        """Read image data and return as NumPy array.

        Returns:
            NumPy array of image data.
        """
        if isinstance(self.image, str):
            if self.image.endswith(".tif"):
                if not os.path.exists(self.image):
                    raise FileNotFoundError(f"Image file {self.image} not found.")
                self.img = rs.open(self.image)
                self.shape = (self.img.count, self.img.width, self.img.height)
                self.imgarr = self.img.read()
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
    
    def __init__(self, image, label, patch_size, stride, channel_first=True):
        """Initialize with image, label, and patch parameters.

        Args:
            image: NumPy array or string path to satellite imagery (.tif or .npy).
            label: NumPy array or string path to label raster (.tif or .npy) or shapefile (.shp).
            patch_size: Integer, size of square patches.
            stride: Integer, step size for sliding window.
            channel_first: Boolean, True if channels are first dimension.
        """
        super().__init__(image, patch_size, stride, channel_first)
        if label is None:
            raise ValueError("Label must be provided for training.")
        self.label_data = label
        self.inv = None

    def readData(self):
        """Read image and label data, validate dimensions.

        Returns:
            Tuple of (image array, label array).
        """
        self.imgarr = super().readData()
        
        if isinstance(self.label_data, str):
            if self.label_data.endswith(".tif"):
                if not os.path.exists(self.label_data):
                    raise FileNotFoundError(f"Label file {self.label_data} not found.")
                self.inv = rs.open(self.label_data).read(1)
            elif self.label_data.endswith(".npy"):
                if not os.path.exists(self.label_data):
                    raise FileNotFoundError(f"Label file {self.label_data} not found.")
                self.inv = np.load(self.label_data)
            elif self.label_data.endswith(".shp"):
                self.inv = None  # Shapefile processing deferred to generate_from_shapefile
            else:
                raise ValueError("Label file must be .tif, .npy, or .shp.")
        else:
            self.inv = self.label_data

        if self.inv is not None and self.imgarr.shape[1:] != self.inv.shape:
            raise ValueError(f"Image {self.imgarr.shape[1:]} and label {self.inv.shape} dimensions must match.")
        
        return self.imgarr, self.inv

    def data_dimension(self):
        """Display image and label dimensions."""
        self.readData()
        print("############################################")
        print(f"The shape of image is: {self.shape}")
        if self.inv is not None:
            print(f"The shape of label is: {self.inv.shape}")
        print("############################################")

    def _extract_patch(self, i, j):
        """Extract a single patch from image and label."""
        img_patch = super()._extract_patch(i, j)
        if self.inv is not None:
            lbl_patch = self.inv[i:i+self.patch_size, j:j+self.patch_size]
            if lbl_patch.shape[0] != self.patch_size or lbl_patch.shape[1] != self.patch_size:
                temp = np.zeros((self.patch_size, self.patch_size), dtype=self.inv.dtype)
                temp[:lbl_patch.shape[0], :lbl_patch.shape[1]] = lbl_patch
                lbl_patch = temp
            return img_patch, lbl_patch
        return img_patch, None

    def _generate_bounding_boxes(self, lbl_patch):
        """Generate YOLO-format bounding box annotations from label patch.

        Args:
            lbl_patch: NumPy array, label patch with class IDs.

        Returns:
            List of tuples (class_id, x_center, y_center, width, height) normalized to [0,1].
        """
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
        """Save bounding box annotations in YOLO format.

        Args:
            bboxes: List of tuples (class_id, x_center, y_center, width, height).
            filepath: String, path to save .txt file.
        """
        with open(filepath, 'w') as f:
            for bbox in bboxes:
                f.write(f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n")

    def _apply_segmentation_augmentation(self, img_patch, lbl_patch, aug):
        """Apply augmentation to image and label patches for segmentation.

        Args:
            img_patch: NumPy array, image patch (channel-last, height, width, channels).
            lbl_patch: NumPy array, label patch (height, width).
            aug: String, augmentation type ("V", "H", "90", "180", "270").

        Returns:
            Tuple of (augmented image, augmented label).
        """
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
        """Apply augmentation to image, label, and bounding boxes for detection.

        Args:
            img_patch: NumPy array, image patch (channel-last, height, width, channels).
            lbl_patch: NumPy array, label patch (height, width).
            aug: String, augmentation type ("V", "H", "90", "180", "270").

        Returns:
            Tuple of (augmented image, augmented label, augmented bboxes).
        """
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
        """Generate patches for semantic segmentation.

        Args:
            format: String, output format ("tif" for GeoTIFF, "npy" for NumPy).
            folder_name: String, directory to save patches (subfolders: patch, label).
            only_label: Boolean, save only patches with non-zero labels if True.
            return_stacked: Boolean, return stacked arrays if True.
            save_stack: Boolean, save stacked arrays to disk if True.
            V_flip: Boolean, apply vertical flip augmentation (only for NumPy output).
            H_flip: Boolean, apply horizontal flip augmentation (only for NumPy output).
            Rotation: Boolean, apply 90/180/270-degree rotation augmentation (only for NumPy output).
        """
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
        """Generate patches for object detection with optional segmentation masks.

        Args:
            format: String, output format ("tif" for GeoTIFF, "npy" for NumPy).
            folder_name: String, directory to save patches (subfolders: patch, bbox, label if segmentation).
            only_label: Boolean, save only patches with non-zero labels if True.
            return_stacked: Boolean, return stacked arrays if True.
            save_stack: Boolean, save stacked arrays to disk if True.
            V_flip: Boolean, apply vertical flip augmentation (only for NumPy output).
            H_flip: Boolean, apply horizontal flip augmentation (only for NumPy output).
            Rotation: Boolean, apply 90/180/270-degree rotation augmentation (only for NumPy output).
            segmentation: Boolean, generate segmentation masks if True.
        """
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
                    self._save_bounding_boxes(boxes, f"{folder_name}/bbox/{index}_bbox.txt")
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
                else:
                    print(f"Stacked patches saved with shape: {patch_stacked.shape}")
            return patch_stacked, label_stacked if segmentation else patch_stacked

    def generate_from_shapefile(self, label_field, format="npy", folder_name="shp_data", 
                               only_label=True, return_stacked=True, save_stack=True, 
                               V_flip=True, H_flip=True, Rotation=True, segmentation=True):
        """Generate patches for segmentation and/or object detection using a polygon shapefile.

        Args:
            label_field: String, name of the field in the shapefile containing label values.
            format: String, output format ("tif" for GeoTIFF, "npy" for NumPy).
            folder_name: String, directory to save patches (subfolders: patch, bbox, label if segmentation).
            only_label: Boolean, save only patches with non-zero labels if True.
            return_stacked: Boolean, return stacked arrays if True.
            save_stack: Boolean, save stacked arrays to disk if True.
            V_flip: Boolean, apply vertical flip augmentation (only for NumPy output).
            H_flip: Boolean, apply horizontal flip augmentation (only for NumPy output).
            Rotation: Boolean, apply 90/180/270-degree rotation augmentation (only for NumPy output).
            segmentation: Boolean, generate segmentation masks if True.
        """
        self.readData()
        if not isinstance(self.label_data, str) or not self.label_data.endswith(".shp"):
            raise ValueError("Label must be a shapefile (.shp) for generate_from_shapefile.")
        
        total_patches, x_steps, y_steps = self.patch_info()
        
        # Read shapefile and reproject to WGS84 (EPSG:4326)
        if not os.path.exists(self.label_data):
            raise FileNotFoundError(f"Shapefile {self.label_data} not found.")
        gdf = gpd.read_file(self.label_data)
        if not gdf.crs:
            raise ValueError("Shapefile must have a defined CRS.")
        if label_field not in gdf.columns:
            raise ValueError(f"Label field '{label_field}' not found in shapefile.")
        
        # Reproject shapefile to WGS84
        wgs84_crs = CRS.from_epsg(4326)
        gdf = gdf.to_crs(wgs84_crs)
        
        # Get raster bounds and transform to WGS84
        if not self.img.crs:
            raise ValueError("Image must have a defined CRS.")
        src_bounds = self.img.bounds
        dst_bounds = transform_bounds(self.img.crs, wgs84_crs, *src_bounds)
        raster_extent = box(dst_bounds[0], dst_bounds[1], dst_bounds[2], dst_bounds[3])
        
        # Clip polygons to raster extent in WGS84
        gdf_clipped = gdf[gdf.geometry.intersects(raster_extent)].copy()
        if gdf_clipped.empty:
            raise ValueError("No polygons intersect with the raster extent in WGS84.")
        
        # Reproject image transform to WGS84
        # Calculate new transform for WGS84-aligned raster
        transform = from_bounds(dst_bounds[0], dst_bounds[1], dst_bounds[2], dst_bounds[3], 
                               self.img.width, self.img.height)
        
        # Rasterize the clipped shapefile in WGS84
        label_array = np.zeros((self.img.height, self.img.width), dtype=np.int32)
        shapes = [(geom, int(value)) for geom, value in zip(gdf_clipped.geometry, gdf_clipped[label_field])]
        if shapes:
            label_array = features.rasterize(
                shapes,
                out_shape=(self.img.height, self.img.width),
                transform=transform,
                fill=0,
                dtype=np.int32
            )
        
        # Set label array for patch extraction
        self.inv = label_array
        
        # Create output directories
        os.makedirs(f"{folder_name}/patch", exist_ok=True)
        os.makedirs(f"{folder_name}/bbox", exist_ok=True)
        if segmentation:
            os.makedirs(f"{folder_name}/label", exist_ok=True)

        # Define augmentations only for NumPy output
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
            if label_array[i:i+self.patch_size, j:j+self.patch_size].any()
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
                        x_cord = j * transform[0] + transform[2]
                        y_cord = transform[5] + i * transform[4]
                        patch_transform = [transform[0], 0, x_cord, 0, transform[4], y_cord]
                        with rs.open(
                            f"{folder_name}/patch/{index}_img.tif", "w", driver="GTiff",
                            count=self.imgarr.shape[0], dtype=self.imgarr.dtype,
                            width=self.patch_size, height=self.patch_size, transform=patch_transform, crs=wgs84_crs
                        ) as raschip:
                            raschip.write(np.transpose(img_patch_save, (2, 0, 1)))
                        if segmentation:
                            with rs.open(
                                f"{folder_name}/label/{index}_lbl.tif", "w", driver="GTiff",
                                count=1, dtype=label_array.dtype, width=self.patch_size, height=self.patch_size,
                                transform=patch_transform, crs=wgs84_crs
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
                else:
                    print(f"Stacked patches saved with shape: {patch_stacked.shape}")
            return patch_stacked, label_stacked if segmentation else patch_stacked

    def visualize(self, folder_name="seg_data", patches_to_show=1, band_num=1, fig_size=(10, 20), dpi=96, show_bboxes=False):
        """Visualize random patches with optional bounding boxes.

        Args:
            folder_name: String, directory containing saved patches.
            patches_to_show: Integer, number of patches to display.
            band_num: Integer, image band to display (1-based; 0 for last band).
            fig_size: Tuple, figure size for plotting.
            dpi: Integer, dots per inch for figure resolution.
            show_bboxes: Boolean, overlay bounding boxes if True.
        """
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
        """Save prediction patches as GeoTIFF files.

        Args:
            folder_name: String, directory to save patches (subfolder: Prediction_patch).
        """
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
        """Save prediction patches as NumPy arrays.

        Args:
            folder_name: String, directory to save patches (subfolder: Prediction_patch).
            return_stacked: Boolean, return stacked array if True.
            save_stack: Boolean, save stacked array to disk if True.
        """
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
