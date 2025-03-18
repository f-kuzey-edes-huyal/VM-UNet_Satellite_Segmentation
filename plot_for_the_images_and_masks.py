import os
import random
import matplotlib.pyplot as plt
import cv2

def select_and_plot_images(image_folder, mask_folder, output_path):
    # Get list of image files
    image_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]
    
    # Randomly select two images
    selected_images = random.sample(image_files, 2)
    
    # Create a figure
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    
    for i, image_file in enumerate(selected_images):
        # Construct mask file name
        mask_file = image_file.replace("_image", "_mask")
        
        # Read image and mask
        image_path = os.path.join(image_folder, image_file)
        mask_path = os.path.join(mask_folder, mask_file)
        
        if not os.path.exists(mask_path):
            print(f"Warning: Mask file not found for {image_file}, skipping.")
            continue
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to read image {image_file}, skipping.")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read mask as grayscale
        if mask is None:
            print(f"Warning: Failed to read mask {mask_file}, skipping.")
            continue
        
        # Plot image
        axes[i, 0].imshow(image)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")
        
        # Plot mask
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title("Mask")
        axes[i, 1].axis("off")
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

# Example usage
image_folder = "data/omdena/train/images"
mask_folder = "data/omdena/train/masks"
output_path = "sample_image_mask_new.png"
select_and_plot_images(image_folder, mask_folder, output_path)
