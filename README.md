# VM-UNet_Satellite_Segmentation
A repository for segmenting satellite images using [VM-UNet](https://github.com/JCruan519/VM-UNet), a variant of the Mamba architecture originally designed for medical image segmentation. This project adapts VM-UNet for geospatial applications, optimizing it for multi-channel satellite imagery.

 <p float="left">
  <img src="https://github.com/f-kuzey-edes-huyal/VM-UNet_Satellite_Segmentation/blob/main/images/segmentation_results.png" width="100%" />
  





last 50 epoch no  aug (```criterion = FocalDiceLoss(alpha=0.6, gamma=3.0, smooth=1e-6, dice_weight=0.5)```)
test of best model, loss: 0.1854,miou: 0.6917083442261295, f1_or_dsc: 0.8177631168953663, accuracy: 0.768285918654057,                 specificity: 0.7439478659382351, sensitivity: 0.7804851975601071, confusion_matrix: [[463943 159680]
 [273110 971043]]

 train_kuzey_new_data_dropout_last_with_aug

last 50 epoch __aug__ (```criterion = FocalDiceLoss(alpha=0.6, gamma=3.0, smooth=1e-6, dice_weight=0.5)```)
 100%|█████████████████████████████████████████████████████████████████████████████████| 114/114 [00:16<00:00,  6.95it/s]
test of best model, loss: 0.1842,miou: 0.6948788965900563, f1_or_dsc: 0.8199746872630134, accuracy: 0.7742759303042763,                 specificity: 0.7793538724517858, sensitivity: 0.7717306472756968, confusion_matrix: [[486023 137600]
 [284002 960151]]

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

  ```conda env export > environment.yaml ```
  
  ```conda activate vmunet  # Replace 'myenv' with the name from the YAML file```

git clone https://github.com/f-kuzey-edes-huyal/VM-UNet_Satellite_Segmentation.git

conda init

exec $SHELL

cd /mnt/d/VM-UNet_Satellite_Segmentation

conda env create -f environment2.yaml

pip install --upgrade pip



 
```
conda init

exec $SHELL

conda activate vmunet

cd /mnt/c/Users/Kuzey/VM-Unet

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

__Setting Up the VM-UNet Environment__

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

🔹 This setup is based on the VM-UNet repository, initially attempted via .yaml but later manually refined. Let me know if you run into issues! 🚀


