import os
import numpy as np
import cv2  # or use PIL
from glob import glob

def compute_class_weights(mask_folder):
    """
    Computes class weights for binary segmentation based on pixel counts.
    
    Args:
        mask_folder (str): Path to folder containing binary mask images.
    
    Returns:
        dict: Class weights {0: weight_for_background, 1: weight_for_foreground}
    """
    total_pixels = 0
    class_counts = {0: 0, 1: 0}

    mask_paths = glob(os.path.join(mask_folder, "*.png"))

    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale (single channel)
        if mask is None:
            continue  # Skip unreadable files
        
        mask = mask // 255  # Normalize to {0,1} (assuming masks are stored in 0-255 format)
        unique, counts = np.unique(mask, return_counts=True)
        pixel_counts = dict(zip(unique, counts))
        
        for class_label in [0, 1]:
            class_counts[class_label] += pixel_counts.get(class_label, 0)
        
        total_pixels += mask.size

    # Compute class weights
    weights = {cls: total_pixels / (2.0 * count) if count > 0 else 0 for cls, count in class_counts.items()}
    
    return weights

# Example usage:
mask_folder = "data/isic2018/train/masks"  # Change this to your mask folder path
class_weights = compute_class_weights(mask_folder)
print("Computed class weights:", class_weights)
