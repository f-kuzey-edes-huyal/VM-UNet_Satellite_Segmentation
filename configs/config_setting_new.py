from torchvision import transforms
from utils import *

from datetime import datetime

class setting_config:
    """
    the config of training setting.
    """

    network = 'vmunet'
    model_config = {
        'num_classes': 1, 
        'input_channels': 3, 
        # ----- VM-UNet ----- #
        'depths': [2,2,2,2],
        'depths_decoder': [2,2,2,1],
        'drop_path_rate': 0.2,
        'load_ckpt_path': './pre_trained_weights/vmamba_small_e238_ema.pth',
    }

    datasets = 'omdena' 
    if datasets == 'omdena':
        data_path = './data/omdena/'
    elif datasets == 'isic17':
        data_path = './data/isic2017/'
    else:
        raise Exception('datasets in not right!')

    #criterion = BceDiceLoss(wb=0.5, wd=0.5)

    #criterion = BceDiceLossw(wb=0.3, wd=0.7)

   # criterion = BceDiceLossw(wb=0.6, wd=0.4)
   
    #criterion = BceDiceLoss(wb=0.6, wd=0.4)
    
    #criterion =  FocalDiceLoss()
    #criterion =   FocalDiceLoss(alpha=0.15, gamma=3.0, smooth=1e-6, dice_weight=0.7)
    #criterion = FocalDiceLoss(alpha=0.5, gamma=2.0, smooth=1e-6, dice_weight=0.6)

    criterion = FocalDiceLoss(alpha=0.6, gamma=3.0, smooth=1e-6, dice_weight=0.5)

    #criterion = AsymmetricFocalDiceLoss(alpha=0.6, gamma=3.0, smooth=1e-6, dice_weight=0.5, beta=1.0)

    criterion = AsymmetricFocalDiceLoss(
        alpha=0.5,  # Balanced between positive and negative class
        gamma=2.0,  # Slightly less focus on hard examples
        smooth=1e-6,  # Keep smooth term to avoid zero division
        dice_weight=0.4,  # Give slightly less weight to Dice loss
        beta=1.0  # Keep equal balance between Dice and Focal loss
    )
   
    criterion = BceDiceLoss(wb=1, wd=1)

    criterion = FocalDiceLoss(alpha=0.6, gamma=3.0, smooth=1e-6, dice_weight=0.5)

    #criterion = TverskyLoss(alpha=0.7, beta=0.3)  # More penalty on false positives


    #criterion = BCEWithLogitsLossCustom()

    #criterion = BceDiceLoss(wb=0.5, wd=0.5)
    #criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=2.0)

    pretrained_path = './pre_trained/'
    num_classes = 1
    input_size_h = 128
    input_size_w = 128
    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = 0
    seed = 42
    world_size = None
    rank = None
    amp = False
    gpu_id = '0'
    batch_size = 32
    epochs = 50

    work_dir = 'results/' + network + '_' + datasets + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    print_interval = 20
    val_interval = 1
    save_interval = 10
    threshold = 0.5

    train_transformer = transforms.Compose([
        myNormalize(datasets, train=True),
        myToTensor(),
        #MedianBlur(blur_limit=3, p=0.1)
        #myRandomHorizontalFlip(p=0.5),
        #myRandomVerticalFlip(p=0.5),
        #myRandomRotation(p=0.5, degree=[0, 360]),
        #myResize(input_size_h, input_size_w)
    ])

    train_transformer2 = transforms.Compose([
        myNormalize(datasets, train=True),
        myToTensor(),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 10]),
        #myMedianBlur(blur_limit=3, p=0.1)
        #myResize(input_size_h, input_size_w)
    ])

    train_transformer3 = transforms.Compose([
        myNormalize(datasets, train=True),  # Normalization first
        myToTensor(),  # Convert to tensor after augmentations
       
        MixUpSegmentation(datasets,alpha=0.4),  # Adding MixUp augmentation
        #myMedianBlur(blur_limit=3, p=0.1),
        
       
    # myRandomHorizontalFlip(p=0.5),
    # myRandomVerticalFlip(p=0.5),
    # myRandomRotation(p=0.5, degree=[0, 360]),
    # myResize(input_size_h, input_size_w)
    ])
    
    train_transformer4 = transforms.Compose([
        myNormalize(datasets, train=True),  # Normalize last,
        RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
        myToTensor(),  # Convert to tensor first
       
        #myMedianBlur(blur_limit=3, p=0.1)
        
    
    ])
    

    train_transformer5 = transforms.Compose([
        #myToTensor(),  # Convert to tensor first
       
        myNormalize(datasets, train=True),  # Normalize last,
        RandomBrightnessContrast(brightness_limit=1, contrast_limit=1, p=1),
        myToTensor(),  # Convert to tensor first
        
        #myMedianBlur(blur_limit=3, p=0.1)
        
    
    ])



    test_transformer = transforms.Compose([
        myNormalize(datasets, train=False),
        myToTensor(),
        #myResize(input_size_h, input_size_w)
    ])

    opt = 'AdamW'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'
    if opt == 'Adadelta':
        lr = 0.01 # default: 1.0 – coefficient that scale delta before it is applied to the parameters
        rho = 0.9 # default: 0.9 – coefficient used for computing a running average of squared gradients
        eps = 1e-6 # default: 1e-6 – term added to the denominator to improve numerical stability 
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty) 
    elif opt == 'Adagrad':
        lr = 0.01 # default: 0.01 – learning rate
        lr_decay = 0 # default: 0 – learning rate decay
        eps = 1e-10 # default: 1e-10 – term added to the denominator to improve numerical stability
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty)
    elif opt == 'Adam':
        lr = 0.001 # default: 1e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability 
        weight_decay = 0.0001 # default: 0 – weight decay (L2 penalty) 
        amsgrad = False # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond
    elif opt == 'AdamW':
        lr = 0.001 # default: 1e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 1e-2 # default: 1e-2 – weight decay coefficient
        amsgrad = False # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond 
    elif opt == 'Adamax':
        lr = 2e-3 # default: 2e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 0 # default: 0 – weight decay (L2 penalty) 
    elif opt == 'ASGD':
        lr = 0.01 # default: 1e-2 – learning rate 
        lambd = 1e-4 # default: 1e-4 – decay term
        alpha = 0.75 # default: 0.75 – power for eta update
        t0 = 1e6 # default: 1e6 – point at which to start averaging
        weight_decay = 0 # default: 0 – weight decay
    elif opt == 'RMSprop':
        lr = 1e-2 # default: 1e-2 – learning rate
        momentum = 0 # default: 0 – momentum factor
        alpha = 0.99 # default: 0.99 – smoothing constant
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        centered = False # default: False – if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
        weight_decay = 0 # default: 0 – weight decay (L2 penalty)
    elif opt == 'Rprop':
        lr = 1e-2 # default: 1e-2 – learning rate
        etas = (0.5, 1.2) # default: (0.5, 1.2) – pair of (etaminus, etaplis), that are multiplicative increase and decrease factors
        step_sizes = (1e-6, 50) # default: (1e-6, 50) – a pair of minimal and maximal allowed step sizes 
    elif opt == 'SGD':
        lr = 0.01 # – learning rate
        momentum = 0.9 # default: 0 – momentum factor 
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty) 
        dampening = 0 # default: 0 – dampening for momentum
        nesterov = False # default: False – enables Nesterov momentum 
    
    sch = 'CosineAnnealingLR'
    if sch == 'StepLR':
        step_size = epochs // 5 # – Period of learning rate decay.
        gamma = 0.5 # – Multiplicative factor of learning rate decay. Default: 0.1
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'MultiStepLR':
        milestones = [60, 120, 150] # – List of epoch indices. Must be increasing.
        gamma = 0.1 # – Multiplicative factor of learning rate decay. Default: 0.1.
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'ExponentialLR':
        gamma = 0.99 #  – Multiplicative factor of learning rate decay.
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'CosineAnnealingLR':
        T_max = 50 # – Maximum number of iterations. Cosine function period.
        eta_min = 0.00001 # – Minimum learning rate. Default: 0.
        last_epoch = -1 # – The index of last epoch. Default: -1.  
    elif sch == 'ReduceLROnPlateau':
        mode = 'min' # – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
        factor = 0.1 # – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
        patience = 10 # – Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.
        threshold = 0.0001 # – Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
        threshold_mode = 'rel' # – One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: ‘rel’.
        cooldown = 0 # – Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
        min_lr = 0 # – A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
        eps = 1e-08 # – Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.
    elif sch == 'CosineAnnealingWarmRestarts':
        T_0 = 50 # – Number of iterations for the first restart.
        T_mult = 2 # – A factor increases T_{i} after a restart. Default: 1.
        eta_min = 1e-6 # – Minimum learning rate. Default: 0.
        last_epoch = -1 # – The index of last epoch. Default: -1. 
    elif sch == 'WP_MultiStepLR':
        warm_up_epochs = 10
        gamma = 0.1
        milestones = [125, 225]
    elif sch == 'WP_CosineLR':
        warm_up_epochs = 20
