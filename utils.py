import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt

from scipy.ndimage import zoom
import SimpleITK as sitk
from medpy import metric



def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)



def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr = config.lr,
            rho = config.rho,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr = config.lr,
            lr_decay = config.lr_decay,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr = config.lr,
            lambd = config.lambd,
            alpha  = config.alpha,
            t0 = config.t0,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            alpha = config.alpha,
            eps = config.eps,
            centered = config.centered,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr = config.lr,
            etas = config.etas,
            step_sizes = config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            weight_decay = config.weight_decay,
            dampening = config.dampening,
            nesterov = config.nesterov
        )
    else: # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr = 0.01,
            momentum = 0.9,
            weight_decay = 0.05,
        )


def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                        'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = config.step_size,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones = config.milestones,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = config.T_max,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode = config.mode, 
            factor = config.factor, 
            patience = config.patience, 
            threshold = config.threshold, 
            threshold_mode = config.threshold_mode, 
            cooldown = config.cooldown, 
            min_lr = config.min_lr, 
            eps = config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = config.T_0,
            T_mult = config.T_mult,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma**len(
                [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler



def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img
    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0) 

    plt.figure(figsize=(7,15))

    plt.subplot(3,1,1)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(3,1,2)
    plt.imshow(msk, cmap= 'gray')
    plt.axis('off')

    plt.subplot(3,1,3)
    plt.imshow(msk_pred, cmap = 'gray')
    plt.axis('off')

    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    plt.savefig(save_path + str(i) +'.png')
    plt.close()

import torch
import torch.nn as nn

class WeightedBCELoss(nn.Module):
    def __init__(self, weight_0= 0.6707589561896, weight_1=  1.9640520507891586):
        """
        Binary Cross-Entropy Loss with class weights for imbalance handling.

        Args:
            weight_0 (float): Weight for class 0 (background).
            weight_1 (float): Weight for class 1 (foreground).
        """
        super(WeightedBCELoss, self).__init__()
        self.weight_0 = weight_0
        self.weight_1 = weight_1

    def forward(self, pred, target):
        """
        Compute weighted BCE loss.

        Args:
            pred (torch.Tensor): Model predictions (logits).
            target (torch.Tensor): Ground truth binary mask (0 or 1).

        Returns:
            torch.Tensor: Weighted BCE loss.
        """
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        # Compute weights for each pixel
        weights = target_ * self.weight_1 + (1 - target_) * self.weight_0

        # BCE Loss computation with manual weighting
        loss = nn.functional.binary_cross_entropy(pred_, target_, weight=weights, reduction='mean')

        return loss

    
class BceDiceLossw(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLossw, self).__init__()
        self.bce = WeightedBCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss
    



class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss
    

class nDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(nDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class CeDiceLoss(nn.Module):
    def __init__(self, num_classes, loss_weight=[0.4, 0.6]):
        super(CeDiceLoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss()
        self.diceloss = nDiceLoss(num_classes)
        self.loss_weight = loss_weight
    
    def forward(self, pred, target):
        loss_ce = self.celoss(pred, target[:].long())
        loss_dice = self.diceloss(pred, target, softmax=True)
        loss = self.loss_weight[0] * loss_ce + self.loss_weight[1] * loss_dice
        return loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss
    

class GT_BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(GT_BceDiceLoss, self).__init__()
        self.bcedice = BceDiceLoss(wb, wd)

    def forward(self, gt_pre, out, target):
        bcediceloss = self.bcedice(out, target)
        gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
        gt_loss = self.bcedice(gt_pre5, target) * 0.1 + self.bcedice(gt_pre4, target) * 0.2 + self.bcedice(gt_pre3, target) * 0.3 + self.bcedice(gt_pre2, target) * 0.4 + self.bcedice(gt_pre1, target) * 0.5
        return bcediceloss + gt_loss



class myToTensor:
    def __init__(self):
        pass
    def __call__(self, data):
        image, mask = data
        return torch.tensor(image).permute(2,0,1), torch.tensor(mask).permute(2,0,1)
       

class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w
    def __call__(self, data):
        image, mask = data
        return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w])
       

class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.hflip(image), TF.hflip(mask)
        else: return image, mask
            

class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.vflip(image), TF.vflip(mask)
        else: return image, mask
import random
import torchvision.transforms.functional as TF

class myRandomBrightness:
    def __init__(self, factor_range=(0.8, 1.2), p=0.5):
        self.factor_range = factor_range  # Defines brightness adjustment range
        self.p = p  # Probability of applying the transformation

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            factor = random.uniform(*self.factor_range)  # Pick a random factor
            image = TF.adjust_brightness(image, factor)  # Adjust brightness
        return image, mask  # Return image and mask unchanged

import torchvision.transforms.functional as TF
import random

class myAdjustBrightness:
    def __init__(self, factor_range=(0.8, 1.2)):
        self.factor_range = factor_range

    def __call__(self, data):
        image, mask = data
        factor = random.uniform(*self.factor_range)  # Random factor from the range
        image = TF.adjust_brightness(image, factor)  # Adjust brightness of the image
        return image, mask  # Return unchanged mask



class myRandomRotation:
    def __init__(self, p=0.5, degree=[0,15]):
        self.angle = random.uniform(degree[0], degree[1])
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.rotate(image,self.angle), TF.rotate(mask,self.angle)
        else: return image, mask 

import random
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

import random
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

import random
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

import random
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms

class myRandomAffine:
    def __init__(self, p=0.5, degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5):
        self.p = p
        self.degrees = (-degrees, degrees)  # Ensure degrees are symmetric
        self.translate = translate
        self.scale = scale
        self.shear = (-shear, shear)  # Ensure shear is symmetric

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            # Generate affine transformation parameters
            params = transforms.RandomAffine.get_params(
                degrees=self.degrees,
                translate=self.translate,
                scales=self.scale,
                shears=self.shear,
                img_size=image.shape[-2:]  # (H, W)
            )

            # Apply affine transformation
            image = TF.affine(image, *params, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.affine(mask, *params, interpolation=TF.InterpolationMode.NEAREST)  # Avoid interpolation issues

        return image, mask





class myNormalize:
    def __init__(self, data_name, train=True):
        
        if data_name == 'omdena':
            if train:
                self.mean = 30.1974
                self.std = 30.9493
            else:
                self.mean = 30.1974
                self.std = 30.9493
        if data_name == 'omdena_pca':
            if train:
                self.mean = 85.9958
                self.std = 42.5926
            else:
                self.mean = 85.9958
                self.std = 42.5926
        elif data_name == 'isic17':
            if train:
                self.mean = 159.922
                self.std = 28.871
            else:
                self.mean = 148.429
                self.std = 25.748
        elif data_name == 'isic18_82':
            if train:
                self.mean = 156.2899
                self.std = 26.5457
            else:
                self.mean = 149.8485
                self.std = 35.3346
            
    def __call__(self, data):
        img, msk = data
        img_normalized = (img-self.mean)/self.std
        img_normalized = ((img_normalized - np.min(img_normalized)) 
                            / (np.max(img_normalized)-np.min(img_normalized))) * 255.
        return img_normalized, msk
    


from thop import profile		 ## 导入thop模块
def cal_params_flops(model, size, logger):
    input = torch.randn(1, 3, size, size).cuda()
    flops, params = profile(model, inputs=(input,))
    print('flops',flops/1e9)			## 打印计算量
    print('params',params/1e6)			## 打印参数量

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total/1e6))
    logger.info(f'flops: {flops/1e9}, params: {params/1e6}, Total params: : {total/1e6:.4f}')






def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0



def test_single_volume(image, label, net, classes, patch_size=[256, 256], 
                    test_save_path=None, case=None, z_spacing=1, val_or_test=False):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None and val_or_test is True:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
        # cv2.imwrite(test_save_path + '/'+case + '.png', prediction*255)
    return metric_list

    import torch
import torch.nn as nn

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, y_pred, y_true):
        # Flatten the tensors to 1D vectors
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # True Positives, False Positives, False Negatives
        TP = torch.sum(y_true * y_pred)
        FP = torch.sum((1 - y_true) * y_pred)
        FN = torch.sum(y_true * (1 - y_pred))

        # Compute the Tversky index
        tversky_index = TP / (TP + self.alpha * FP + self.beta * FN)

        # Compute the Tversky loss
        tversky_loss = 1 - tversky_index
        
        return tversky_loss


import torch
import torch.nn as nn

class BCEWithLogitsLossCustom(nn.Module):
    def __init__(self, pos_weight=None, reduction='mean', label_smoothing=0.0):
        """
        Custom wrapper for BCEWithLogitsLoss.

        Args:
            pos_weight (torch.Tensor, optional): Weight for the positive class to handle class imbalance.
            reduction (str): Specifies the reduction to apply ('mean', 'sum', 'none').
            label_smoothing (float): Smooths the labels to prevent overconfidence (0 means no smoothing).
        """
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        """
        Compute the loss between logits and targets.

        Args:
            logits (torch.Tensor): Model outputs before sigmoid activation.
            targets (torch.Tensor): Ground truth binary labels (0 or 1).

        Returns:
            torch.Tensor: Computed loss.
        """
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing  # Smooth labels

        return self.loss_fn(logits, targets)

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, y_pred, y_true):
        # Flatten the tensors to 1D vectors
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # True Positives, False Positives, False Negatives
        TP = torch.sum(y_true * y_pred)
        FP = torch.sum((1 - y_true) * y_pred)
        FN = torch.sum(y_true * (1 - y_pred))

        # Compute the Tversky index
        tversky_index = TP / (TP + self.alpha * FP + self.beta * FN)

        # Compute the Tversky loss
        tversky_loss = 1 - tversky_index
        
        return tversky_loss

import torch
import numpy as np
import random

import torch
import numpy as np
import random

class MixUpSegmentation:
    def __init__(self, dataset, alpha=0.4, p=0.5):
        """
        MixUp augmentation for segmentation tasks.

        Parameters:
        - dataset: The dataset used for selecting random samples.
        - alpha: MixUp Beta distribution parameter.
        - p: Probability of applying MixUp.
        """
        self.dataset = dataset
        self.alpha = alpha
        self.p = p

    def __call__(self, data):
        """Applies MixUp augmentation."""
        if random.random() > self.p:
            return data  # Skip MixUp with probability (1 - p)

        if not isinstance(data, tuple) or len(data) != 2:
            raise ValueError(f"Expected (image, mask) tuple, got {type(data)} with length {len(data)}")

        image1, mask1 = data

        # Ensure dataset has enough samples
        if len(self.dataset) < 2:
            return image1, mask1  # No MixUp if dataset is too small

        # Select a random sample using dataset's __getitem__
        index = random.randint(0, len(self.dataset) - 1)
        sample = self.dataset.__getitem__(index)  # Ensure __getitem__ is used

        if not isinstance(sample, tuple) or len(sample) != 2:
            raise ValueError(f"Dataset sample should be (image, mask), but got {type(sample)} with length {len(sample)}")

        image2, mask2 = sample

        # Ensure both images and masks have the same shape
        if image1.shape != image2.shape or mask1.shape != mask2.shape:
            raise ValueError("Image or mask shapes do not match for MixUp!")

        # Compute the MixUp coefficient
        lam = np.random.beta(self.alpha, self.alpha)

        # Apply MixUp
        mixed_image = lam * image1 + (1 - lam) * image2
        mixed_mask = lam * mask1 + (1 - lam) * mask2

        return mixed_image, mixed_mask
