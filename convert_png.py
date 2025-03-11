import os
import cv2
import numpy as np
from tifffile import imread

# Define your paths (update these)
input_base = "data_dir"  # Main dataset folder
output_base = "data/omdena"  # Folder where PNG images will be saved

# Subfolders for train, val, test
sets = ["train", "val"]

# Process both images and masks
for dataset_type in sets:
    image_folder = os.path.join(input_base, dataset_type, "images")
    mask_folder = os.path.join(input_base, dataset_type, "masks")

    output_image_folder = os.path.join(output_base, dataset_type, "images")
    output_mask_folder = os.path.join(output_base, dataset_type, "masks")

    # Ensure output directories exist
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)

    # Convert images (3-channel TIFF → PNG)
    for filename in os.listdir(image_folder):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            img_path = os.path.join(image_folder, filename)
            img = imread(img_path).astype(np.float32)  # Read TIFF image as float

            # Normalize to 0-255 for PNG format
            img = (255 * (img - img.min()) / (img.max() - img.min() + 1e-8)).astype(np.uint8)

            # Convert to 3-channel if needed
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # Convert filename to PNG
            new_filename = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_image_folder, new_filename)

            # Save as PNG
            cv2.imwrite(output_path, img)
            print(f"Converted {filename} -> {new_filename} (Images)")

    # Convert masks (1-channel TIFF → PNG)
    for filename in os.listdir(mask_folder):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            mask_path = os.path.join(mask_folder, filename)
            mask = imread(mask_path).astype(np.float32)  # Read mask as float

            # Normalize mask
            mask = (255 * (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)).astype(np.uint8)

            # Ensure mask is single-channel (OpenCV requires 1, 3, or 4 channels)
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]  # Take the first channel if extra dimensions exist

            # Convert filename to PNG
            new_filename = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_mask_folder, new_filename)

            # Save as PNG
            cv2.imwrite(output_path, mask)
            print(f"Converted {filename} -> {new_filename} (Masks)")

print("✅ All images and masks successfully converted to PNG format!")
