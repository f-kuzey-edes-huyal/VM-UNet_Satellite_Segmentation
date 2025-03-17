# VM-UNet_Satellite_Segmentation
A repository for segmenting satellite images using [VM-UNet](https://github.com/JCruan519/VM-UNet), a variant of the Mamba architecture originally designed for medical image segmentation. This project adapts VM-UNet for geospatial applications, optimizing it for multi-channel satellite imagery.droput =0.3
 49/49 [00:07<00:00,  6.82it/s]test of best model, loss: 1.0475,miou: 0.43483443953679546, f1_or_dsc: 0.6061109596409913, accuracy: 0.7336699816645408,                 specificity: 0.7730487800436336, sensitivity: 0.6484363298094585, confusion_matrix: [[424495 124623]
 [ 89191 164507]]

 Computed class weights: {0: 0.6707589561896, 1: 1.9640520507891586}

 test of best model, loss: 0.4941,miou: 0.47655385239102427, f1_or_dsc: 0.6454947127317131, accuracy: 0.7555454799107143,                 specificity: 0.7792368853324786, sensitivity: 0.7042664900787551, confusion_matrix: [[427893 121225]
 [ 75027 178671]]

 <p float="left">
  <img src="https://github.com/f-kuzey-edes-huyal/VM-UNet_Satellite_Segmentation/blob/main/images/0.png" width="22%" />
  <img src="https://github.com/f-kuzey-edes-huyal/VM-UNet_Satellite_Segmentation/blob/main/images/10.png" width="22%" />
  <img src="https://github.com/f-kuzey-edes-huyal/VM-UNet_Satellite_Segmentation/blob/main/images/20.png" width="22%" />
  <img src="https://github.com/f-kuzey-edes-huyal/VM-UNet_Satellite_Segmentation/blob/main/images/30.png" width="22%" />
</p>

new data 20 epoch:
est of best model, loss: 0.4123,miou: 0.5885590061840256, f1_or_dsc: 0.7409973490350088, accuracy: 0.7984538831208882,                 specificity: 0.7818074555922686, sensitivity: 0.8297137345476902, confusion_matrix: [[952838 265925]
 [110518 538495]]

 <p align="center">
  <img src="https://github.com/f-kuzey-edes-huyal/VM-UNet_Satellite_Segmentation/blob/main/images/newdata_20epochs/0.png" width="250" />
  <img src="https://github.com/f-kuzey-edes-huyal/VM-UNet_Satellite_Segmentation/blob/main/images/newdata_20epochs/10.png" width="250" />
  <img src="https://github.com/f-kuzey-edes-huyal/VM-UNet_Satellite_Segmentation/blob/main/images/newdata_20epochs/20.png" width="250" />
  <img src="https://github.com/f-kuzey-edes-huyal/VM-UNet_Satellite_Segmentation/blob/main/images/newdata_20epochs/30.png" width="250" />
  <img src="https://github.com/f-kuzey-edes-huyal/VM-UNet_Satellite_Segmentation/blob/main/images/newdata_20epochs/40.png" width="250" />
  <!-- Add more images here up to image50.png -->
</p>

new data 20 epoch( rotation angle decrease)
test of best model, loss: 0.4240,miou: 0.5838969359361299, f1_or_dsc: 0.7372915783703181, accuracy: 0.7919188382332785,                 specificity: 0.7661497764536666, sensitivity: 0.8403098243024408,  confusion_matrix: [[933755 285008]
 [103641 545372]]

three augnentation: converge around 7th epoch: 20 epoch [training](https://github.com/f-kuzey-edes-huyal/VM-UNet_Satellite_Segmentation/blob/main/config_setting_new.py)
test of best model, loss: 0.4155,miou: 0.5871034451754057, f1_or_dsc: 0.7398426951439443, accuracy: 0.7934752347176535,                 specificity: 0.7659750090870826, sensitivity: 0.8451171240021387, confusion_matrix: [[933542 285221]
 [100521 548492]]
equal weights bcedice loss 
test of best model, loss: 0.8601,miou: 0.5784939799474642, f1_or_dsc: 0.7329695105542534, accuracy: 0.7949282997532895,                 specificity: 0.7869150934184908, sensitivity: 0.8099760713575845, confusion_matrix: [[959063 259700]
 [123328 525685]]

 FocalDiceLoss(alpha=0.25, gamma=2.0, dice_weight=0.5)

  0.5611284580874188, f1_or_dsc: 0.7188754457463067, accuracy: 0.7636343972724781,                 specificity: 0.7071407648574826, sensitivity: 0.8697221781381883, confusion_matrix: [[861837 356926]
 [ 84552 564461]]
  criterion= TverskyBCELoss(alpha=0.7, beta=0.3, bce_weight=0.5)
100%|█████████████████████████████████████████████████████████████████████████████████| 114/114 [00:17<00:00,  6.64it/s]
test of best model, loss: 0.6537,miou: 0.5434632827446988, f1_or_dsc: 0.7042127776156395, accuracy: 0.7520221910978618,                 specificity: 0.7000975579337411, sensitivity: 0.8495299785982715, confusion_matrix: [[853253 365510]
 [ 97657 551356]]

 test of best model, loss: 0.4004,miou: 0.5757549165556687, f1_or_dsc: 0.7307670888492872, accuracy: 0.7825767115542763,                 specificity: 0.7471108000489021, sensitivity: 0.8491771351267232, confusion_matrix: [[910551 308212]
 [ 97886 551127]]

 100 epch 

 test of best model, loss: 0.4107,miou: 0.5808558283188433, f1_or_dsc: 0.7348624939904265, accuracy: 0.7962667900219298,                 specificity: 0.7876075988522789, sensitivity: 0.8125276381212703, confusion_matrix: [[959907 258856]
 [121672 527341]]

vmunet_omdena_Sunday_16_March_2025_00h_06m_22s
 test of best model, loss: 0.4040,miou: 0.613971324373178, f1_or_dsc: 0.7608206107523349, accuracy: 0.7006600363212719,                 specificity: 0.4652992943489118, sensitivity: 0.9202132998875135, confusion_matrix: [[419438 481999]
 [ 77101 889238]]

 #----------Testing----------#
100%|█████████████████████████████████████████████████████████████████████████████████| 114/114 [00:27<00:00,  4.19it/s]
test of best model, loss: 0.2386,miou: 0.6143667783966488, f1_or_dsc: 0.7611241591663865, accuracy: 0.735626220703125,                 specificity: 0.6701124559577925, sensitivity: 0.7935459336012525, confusion_matrix: [[587308 289124]
 [204667 786677]]

  criterion = BceDiceLoss(wb=0.5, wd=0.5)

 val epoch: 1, loss: 0.5451, miou: 0.5708965506705284, f1_or_dsc: 0.7268416884954574, accuracy: 0.6090125368352521,                 specificity: 0.18930048195410482, sensitivity: 0.9800735163575913, confusion_matrix: [[165909 710523]
 [ 19754 971590]]

```
conda init

exec $SHELL

conda activate vmunet

cd /mnt/d//VM-Unet-main

cd /mnt/c/Users/Kuzey/VM-Unet
```
