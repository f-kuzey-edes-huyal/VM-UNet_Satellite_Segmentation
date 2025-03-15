import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from torch.utils.data import DataLoader
from datasets.dataset import NPY_datasets, NPY_datasets2, NPY_datasets3, NPY_datasets4, NPY_datasets5
from torchvision import transforms
import torch
from tensorboardX import SummaryWriter
from configs.config_setting_new import setting_config
import torch
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Function to load and plot 4 images and their corresponding masks
def plot_images_and_masks():
    # Define the path to your dataset
    path_data = "data/omdena/"
    
    # Create a config object
    config = setting_config
    
    # Create instances of the datasets with the config object
    train_dataset_1 = NPY_datasets(path_data, config, train=True)
    train_dataset_2 = NPY_datasets2(path_data, config, train=True)
    train_dataset_3 = NPY_datasets3(path_data, config, train=True)
    train_dataset_4 = NPY_datasets5(path_data, config, train=True)
    
    # Create data loaders to fetch data from the datasets
    # Create data loaders with fixed seed and sequential order
    loader_1 = DataLoader(train_dataset_1, batch_size=1, shuffle=False, num_workers=0)
    loader_2 = DataLoader(train_dataset_2, batch_size=1, shuffle=False, num_workers=0)
    loader_3 = DataLoader(train_dataset_3, batch_size=1, shuffle=False, num_workers=0)
    loader_4 = DataLoader(train_dataset_4, batch_size=1, shuffle=False, num_workers=0)

    # Plotting 4 images and their corresponding masks
    fig, axes = plt.subplots(4, 2, figsize=(10, 12))

    # List of data loaders and dataset names for easy iteration
    data_loaders = [(loader_1, "Transform 1"), (loader_2, "Transform 2"), (loader_3, "Transform 3"), (loader_4, "Transform 4")]

    for i, (loader, name) in enumerate(data_loaders):
        # Fetch one image and mask from the dataset
        image, mask = next(iter(loader))

        # Convert the image and mask to numpy arrays for plotting
        image = image[0].permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
        mask = mask[0].numpy().squeeze()  # Remove the singleton dimension for the mask

        # Ensure the image is in the valid range [0, 1] if it's a float
        if image.max() > 1:
            image = image / 255.0  # Scale to [0, 1] for imshow

        # Plot the image and mask
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'{name} - Image {i+1}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f'{name} - Mask {i+1}')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig("augmentation_samples.png")
    plt.show()

# Call the function to plot the images and masks
plot_images_and_masks()
