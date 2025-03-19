# VM-UNet_Satellite_Segmentation
A repository for segmenting satellite images using [VM-UNet](https://github.com/JCruan519/VM-UNet), a variant of the Mamba architecture originally designed for medical image segmentation. This project adapts VM-UNet for geospatial applications, optimizing it for multi-channel satellite imagery.

 <p float="left">
  <img src="https://github.com/f-kuzey-edes-huyal/VM-UNet_Satellite_Segmentation/blob/main/images/segmentation_results.png" width="100%" />
  


## Model Performance Comparison (With & Without Augmentations)

| Metric         | No Augmentation | With Augmentation | Change |
|---------------|---------------|----------------|--------|
| **Loss**       | 0.1854        | 0.1842         | 🔽 (Better) |
| **mIoU**       | 0.6917        | 0.6949         | 🔼 (Better) |
| **F1/DSC**     | 0.8178        | 0.8199         | 🔼 (Better) |
| **Accuracy**   | 0.7683        | 0.7743         | 🔼 (Better) |
| **Specificity**| 0.7439        | 0.7794         | 🔼 (Much Better) |
| **Sensitivity**| 0.7805        | 0.7717         | 🔽 (Slight Drop) |

### 🔍 **Observations**
- Augmentations **improved performance** across most metrics, especially specificity (+0.035).
- Sensitivity dropped slightly, meaning **more false negatives**.
- The model appears **more conservative**, preferring **fewer false positives**.

 


# Setting Up the VM-UNet Environment

Follow these steps to set up the vmunet environment.

1️⃣ Create & Activate Environment

```conda create -n vmunet python=3.8 -y && conda activate vmunet```

2️⃣ Install CUDA & PyTorch

```conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit```  

```pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117```

3️⃣ Install Dependencies

```pip install packaging timm==0.4.12 pytest chardet yacs termcolor submitit tensorboardX```  

```pip install triton==2.0.0 causal_conv1d==1.0.0 mamba_ssm==1.0.1```  

```pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs```

4️⃣ (Optional) Update Dependencies

```conda env update --file environment.yaml --update-deps```

🔹 This setup is based on the [VM-UNet](https://github.com/JCruan519/VM-UNet) repository, initially attempted via .yaml but later manually refined. Let me know if you run into issues! 🚀

# 🏗️ Training the VM-UNet Architecture

```
# Clone the repository  
git clone https://github.com/f-kuzey-edes-huyal/VM-UNet_Satellite_Segmentation.git  

# Open WSL  
conda init  
exec $SHELL  

# Navigate to the project directory  
cd /mnt/c/VM-UNet_Satellite_Segmentation  

# Set library path  
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH  

# Train the model with augmented data  
python train_kuzey_new_data_dropout_last_with_aug.py
```
