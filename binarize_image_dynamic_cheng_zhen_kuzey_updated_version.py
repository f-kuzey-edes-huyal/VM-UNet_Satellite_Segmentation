import os
import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
from natsort import natsorted

"""
This code is prepared by adjusting the code provided by Cheng Zhen!
"""

def normalize_by_layer(image_array):
    """
    Normalize each layer (band) of the image to the range [0, 1].
    """
    image_array = image_array.astype(np.float64)
    for i in range(image_array.shape[2]):
        layer_min = np.min(image_array[:, :, i])
        layer_max = np.max(image_array[:, :, i])
        try:
            image_array[:, :, i] = (image_array[:, :, i] - layer_min) / (layer_max - layer_min)
        except ZeroDivisionError:
            print(f"Band {i} has zero variation (min = max = {layer_min}). Skipping normalization.")
            image_array[:, :, i] = 0
    return image_array

def convert_binary_mask(mask_array, multiclass=True, threshold=None):
    """
    Convert fractional masks into binary masks.
    If multiclass is True, the mask remains unchanged.
    If threshold is None, a dynamic threshold (mean of the mask) is used.
    """
    if not multiclass:
        if threshold is None:
            threshold = np.mean(mask_array)
        mask_array = (mask_array >= threshold).astype(np.float64)
    return mask_array

def center_crop(image, crop_size):
    """Center crop the image to the desired size."""
    h, w, _ = image.shape
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return image[top: top + crop_size, left: left + crop_size]

def read_file(data_dir, image_dir, multiclass=True, threshold=None, subset_size=None):
    """
    Read and preprocess images and masks from specified directories.
    Applies center cropping and normalization.
    """
    image_dataset = []
    mask_dataset = []
    mask_dir = f"{image_dir}_masks"
    image_files = natsorted([
        f for f in os.listdir(os.path.join(data_dir, image_dir)) if f.endswith('_1_GeoTIFF.tif')
    ])

    if subset_size:
        image_files = image_files[:subset_size]

    print(f"Reading images and masks from {image_dir} and {mask_dir}")
    for image_file in image_files:
        mask_file = image_file.replace('.tif', '_fractional_mask.tif')
        image_path = os.path.join(data_dir, image_dir, image_file)
        mask_path = os.path.join(data_dir, mask_dir, mask_file)

        # Read image and convert from (C, H, W) to (H, W, C)
        with rasterio.open(image_path) as img:
            image_array = img.read()
        image_array = np.transpose(image_array, [1, 2, 0])

        # Read mask and convert from (C, H, W) to (H, W, C)
        with rasterio.open(mask_path) as msk:
            mask_array = msk.read()
        mask_array = np.transpose(mask_array, [1, 2, 0])

        # Center crop to a fixed size (e.g., 128x128)
        image_array = center_crop(image_array, 128)
        mask_array = center_crop(mask_array, 128)

        # Preprocess image and mask
        image_array[np.isnan(image_array)] = 0
        # Select relevant bands (e.g., bands 1, 2, 3) and normalize each band
        image_array = normalize_by_layer(image_array[:, :, (1, 2, 3)])

        mask_array[np.isnan(mask_array)] = 0
        mask_array = convert_binary_mask(mask_array, multiclass, threshold)

        image_dataset.append(image_array)
        mask_dataset.append(mask_array)

    print("Finished reading images.")
    return np.array(image_dataset), np.array(mask_dataset).astype(np.float64)

def split_dataset(image_dataset, mask_dataset, test_size=0.2, random_state=42):
    """
    Split dataset into train and test sets.
    Train: 80%
    Test: 20%
    """
    X_train, X_test, y_train, y_test = train_test_split(
        image_dataset, mask_dataset, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def save_data_to_folders(image_dataset, mask_dataset, data_dir, image_dir, split_folders=["train", "test"]):
    """
    Save images and masks into specified folders for each split (train, test).
    The folder structure will include separate 'images' and 'masks' directories.
    """
    for split, image_data, mask_data in zip(split_folders, image_dataset, mask_dataset):
        split_dir = os.path.join(data_dir, split)
        images_dir = os.path.join(split_dir, "images")
        masks_dir = os.path.join(split_dir, "masks")

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        for idx, (image, mask) in enumerate(zip(image_data, mask_data)):
            image_path = os.path.join(images_dir, f"{split}_{idx}_image.tif")
            mask_path = os.path.join(masks_dir, f"{split}_{idx}_mask.tif")

            # Save image
            with rasterio.open(image_path, 'w', driver='GTiff', count=image.shape[2],
                               width=image.shape[1], height=image.shape[0], dtype=image.dtype) as img_out:
                for band in range(image.shape[2]):
                    img_out.write(image[:, :, band], band + 1)

            # Save mask
            with rasterio.open(mask_path, 'w', driver='GTiff', count=mask.shape[2],
                               width=mask.shape[1], height=mask.shape[0], dtype=mask.dtype) as mask_out:
                for band in range(mask.shape[2]):
                    mask_out.write(mask[:, :, band], band + 1)

    print(f"Finished saving images and masks to {data_dir}.")

# Set paths and load dataset
data_dir = os.path.expanduser("data_dir")  # Update to your actual data directory path
image_dir = 'VBWVA_8R_and_SLMO_8R_1'

# Read and preprocess the data with dynamic thresholding (threshold=None triggers dynamic calculation)
image_dataset, mask_dataset = read_file(
    data_dir,
    image_dir,
    multiclass=False,
    threshold=None
)

# Split the dataset into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = split_dataset(image_dataset, mask_dataset, test_size=0.2)

# Save the dataset into 'train' and 'test' directories
save_data_to_folders(
    [X_train, X_test],  # Image data for train and test splits
    [y_train, y_test],  # Mask data for train and test splits
    data_dir,
    image_dir
)
