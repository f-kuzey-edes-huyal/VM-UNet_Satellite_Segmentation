import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot_tall_images(image_paths, save_path="segmentation_results.png", ):
    """
    Loads and plots 5 tall images side by side and saves the figure.

    Parameters:
    - image_paths: List of file paths to 5 images
    - save_path: Path to save the output image
    """
    assert len(image_paths) == 5, "You must provide exactly 5 image file paths."

    fig, axes = plt.subplots(1, 5, figsize=(10, 20))  # Wide figure for tall images

    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path)  # Load image
        img = np.array(img)  # Convert to NumPy array

        cmap = "gray" if len(img.shape) == 2 else None  # Use grayscale colormap if needed
        axes[i].imshow(img, cmap=cmap)
        axes[i].set_title(f"Image {i+1}", fontsize=12)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")  # Save figure
    plt.show()

# Example usage with actual PNG image file paths
image_paths = ["0.png", "10.png", "20.png", "30.png", "40.png"]
plot_tall_images(image_paths)
