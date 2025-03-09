import torch
from torch.utils.data import DataLoader, ConcatDataset
import timm
import os
import sys
import warnings

from datasets.dataset_omdena import NPY_datasets, NPY_datasets2
from tensorboardX import SummaryWriter
from models.vmunet.vmunet import VMUNet

#from engine_meww_binary import train_one_epoch, val_one_epoch, test_one_epoch
#from engine_meww_binary_argmax_newdata import train_one_epoch, val_one_epoch, test_one_epoch
#from engine_meww_binary_argmax import train_one_epoch, val_one_epoch, test_one_epoch

from engine_binarized import train_one_epoch, val_one_epoch, test_one_epoch

from utils import get_logger, set_seed, get_optimizer, get_scheduler, cal_params_flops
from configs.config_setting_omdena_binary import setting_config
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

warnings.filterwarnings("ignore")

def main(config):
    """Main training loop for VMUNet."""
    print('#----------Initializing Logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    outputs_dir = os.path.join(config.work_dir, 'outputs')

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    logger = get_logger('train', log_dir)
    writer = SummaryWriter(os.path.join(config.work_dir, 'summary'))

    print('#----------Setting GPU & Seed----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing Dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    #train_dataset2 = NPY_datasets2(config.data_path, config, train=True)
    #train_dataset_new = ConcatDataset([train_dataset, train_dataset2])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers, pin_memory=True, drop_last=True)

    print('#----------Initializing Model----------#')
    model_cfg = config.model_config
    if config.network.lower() == 'vmunet':
        model = VMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate']
        )
        model.load_from()
    else:
        raise ValueError(f"Invalid network: {config.network}")

    model = model.cuda()
    cal_params_flops(model, 256, logger)

    print('#----------Initializing Loss, Optimizer, Scheduler----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    min_loss = float('inf')
    start_epoch = 1
    val_loss_history = []

    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    if os.path.exists(resume_model):
        print('#----------Resuming from Checkpoint----------#')
        checkpoint = torch.load(resume_model, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        min_loss = checkpoint['min_loss']

    step = 0
    print('#----------Starting Training----------#')

    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()
        model.train()

        step = train_one_epoch(
            train_loader, model, criterion, optimizer, scheduler, 
            epoch, step, logger, config, writer
        )

        # Evaluate model
        val_loss = val_one_epoch(val_loader, model, criterion, epoch, logger, config, scheduler)
        val_loss_history.append(val_loss)

        # Keep only the last 5 validation losses
        if len(val_loss_history) > 5:
            val_loss_history.pop(0)

        # Early Stopping: Stop if last 5 losses show a clear increasing trend
        if len(val_loss_history) == 5 and all(val_loss_history[i] > val_loss_history[i - 1] for i in range(1, 5)):
            print(f"Early stopping triggered at epoch {epoch} (5 consecutive validation loss increases)")
            break

        # Save best model
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))

        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'min_loss': min_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(checkpoint_dir, 'latest.pth'))

    print('#----------Testing Best Model----------#')
    best_model_path = os.path.join(checkpoint_dir, 'best.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location="cpu"))
        test_loss = test_one_epoch(val_loader, model, criterion, logger, config)
        final_model_name = f"best-epoch{start_epoch}-loss{min_loss:.4f}.pth"
        os.rename(best_model_path, os.path.join(checkpoint_dir, final_model_name))

if __name__ == '__main__':
    config = setting_config
    main(config)
