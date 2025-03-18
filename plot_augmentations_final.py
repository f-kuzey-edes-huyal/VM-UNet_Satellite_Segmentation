import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from datasets.dataset import NPY_datasets, NPY_datasets2, NPY_datasets3
from configs.config_setting_new import setting_config
import torch
import random

# Function to set a fixed seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Function to load and plot 3 images and their corresponding masks
def plot_images_and_masks():
    path_data = "data/omdena/"  # Define dataset path
    config = setting_config  # Load config settings
    
    # Load only the first three datasets
    train_dataset_1 = NPY_datasets(path_data, config, train=True)
    train_dataset_2 = NPY_datasets2(path_data, config, train=True)
    train_dataset_3 = NPY_datasets3(path_data, config, train=True)
    
    # Create data loaders (Fixed seed, no shuffle)
    loader_1 = DataLoader(train_dataset_1, batch_size=1, shuffle=False, num_workers=0)
    loader_2 = DataLoader(train_dataset_2, batch_size=1, shuffle=False, num_workers=0)
    loader_3 = DataLoader(train_dataset_3, batch_size=3, shuffle=False, num_workers=0)

    # Plot 3 images and their corresponding masks
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))

    # Data loaders and transformation names
    data_loaders = [(loader_1, "Transform 1"), (loader_2, "Transform 2"), (loader_3, "Transform 3")]

    for i, (loader, name) in enumerate(data_loaders):
        image, mask = next(iter(loader))  # Fetch one sample

        # Convert image & mask to numpy format for plotting
        image = image[0].permute(1, 2, 0).numpy()  
        mask = mask[0].numpy().squeeze()  

        # Normalize image if necessary
        if image.max() > 1:
            image = image / 255.0  

        # Plot image & mask
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'{name} - Image {i+1}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f'{name} - Mask {i+1}')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig("augmentation_samples_first3.png")  # Save figure
    plt.show()

# Call the function to plot images and masks
plot_images_and_masks()
