import os
import random
import numpy as np
import rasterio
import matplotlib.pyplot as plt

def load_image_and_mask(image_path, mask_path):
    """Loads an image and its corresponding mask from given paths."""
    with rasterio.open(image_path) as img:
        image = img.read([1, 2, 3])  # RGB bands
        image = np.transpose(image, [1, 2, 0])  # Reorder to (height, width, channels)

    with rasterio.open(mask_path) as msk:
        mask = msk.read(1)  # Assuming mask is single band
        mask = np.expand_dims(mask, axis=-1)  # Convert to (height, width, 1)

    return image, mask

def plot_images_and_masks(images, masks, output_path):
    """Plots the images and their corresponding masks side by side."""
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    for i in range(3):
        axs[i, 0].imshow(images[i])
        axs[i, 0].set_title(f"Image {i+1}")
        axs[i, 0].axis('off')

        axs[i, 1].imshow(masks[i], cmap='gray')
        axs[i, 1].set_title(f"Mask {i+1}")
        axs[i, 1].axis('off')

    # Save the figure to the specified output path
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main(images_folder, masks_folder, output_image_path):
    """Main function to load random images and masks, plot them, and save the result."""
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.tif')]
    mask_files = [f for f in os.listdir(masks_folder) if f.endswith('.tif')]

    # Ensure we have the same number of image and mask files
    if len(image_files) != len(mask_files):
        raise ValueError("The number of images and masks do not match.")

    # Randomly select 3 images and corresponding masks
    selected_images = random.sample(image_files, 3)
    selected_masks = [f.replace('_image.tif', '_mask.tif') for f in selected_images]

    images = []
    masks = []

    for image_file, mask_file in zip(selected_images, selected_masks):
        image_path = os.path.join(images_folder, image_file)
        mask_path = os.path.join(masks_folder, mask_file)

        image, mask = load_image_and_mask(image_path, mask_path)

        images.append(image)
        masks.append(mask)

    # Plot and save the images and masks side by side
    plot_images_and_masks(images, masks, output_image_path)

    print(f"Images and masks saved to: {output_image_path}")

# Specify the folder paths
images_folder = 'data/omdena_last/train/images'  # Folder containing images
masks_folder = 'data/omdena_last/train/masks'  # Folder containing masks
output_image_path = 'output_images_and_masks.png'  # Path to save the final output image

# Run the main function
main(images_folder, masks_folder, output_image_path)
