import numpy as np
from terratiff import TerraTiff
import os
import shutil
from GeoPatch import TrainPatch, PredictionPatch

def create_dummy_tif(filename, bands, rows, cols, nodata=None):
    array = np.random.randint(0, 255, (bands, rows, cols)).astype(np.uint8)
    if bands == 1:
        array = array[0]
    # some dummy transform values
    origin_x = 100.0
    origin_y = 200.0
    pixel_width = 1.0
    pixel_height = -1.0
    crs = "WGS84"
    tif = TerraTiff.from_array(array, origin_x, origin_y, pixel_width, pixel_height, crs, nodata=nodata)
    tif.save(filename)
    return filename

def test():
    if os.path.exists("test_data"):
        shutil.rmtree("test_data")
    os.makedirs("test_data")
    
    img_path = create_dummy_tif("test_data/image.tif", 3, 256, 256)
    lbl_path = create_dummy_tif("test_data/label.tif", 1, 256, 256)
    
    # Test TrainPatch segmentation
    print("Testing TrainPatch segmentation...")
    train_patch = TrainPatch(img_path, lbl_path, patch_size=64, stride=64, channel_first=True)
    train_patch.generate_segmentation(format="tif", folder_name="test_data/seg_output", only_label=False)
    
    # Test TrainPatch detection
    print("Testing TrainPatch detection...")
    train_patch.generate_detection(format="npy", folder_name="test_data/det_output", only_label=False, segmentation=True)
    
    # Test PredictionPatch
    print("Testing PredictionPatch...")
    pred_patch = PredictionPatch(img_path, patch_size=64, stride=64, channel_first=True)
    pred_patch.save_Geotif(folder_name="test_data/pred_output")
    
    print("All tests passed!")

if __name__ == "__main__":
    test()
