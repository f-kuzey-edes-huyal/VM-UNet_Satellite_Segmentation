import os
import cv2
import numpy as np

# Path to the folder containing PNG images
image_folder = "data/omdena/train/images"

# Initialize list to store all pixel values
pixel_values = []

# Loop through all images in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".png"):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Read image in unchanged mode
        
        if img is not None:
            pixel_values.append(img.flatten())  # Flatten image and store pixel values

# Concatenate all pixel values from all images
all_pixels = np.concatenate(pixel_values, axis=0)

# Compute the overall mean and standard deviation
overall_mean = np.mean(all_pixels)
overall_std = np.std(all_pixels)

print(f"Overall Mean: {overall_mean:.4f}")
print(f"Overall Standard Deviation: {overall_std:.4f}")
