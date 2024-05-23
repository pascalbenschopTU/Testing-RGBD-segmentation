import os
import sys
import time
import json
import random
import argparse
from tqdm import tqdm
import importlib
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import numpy as np

sys.path.append('../UsefullnessOfDepth')

# Model
from model_DFormer.builder import EncoderDecoder as segmodel
from models_CMX.builder import EncoderDecoder as cmxmodel
from model_pytorch_deeplab_xception.deeplab import DeepLab
from models_segformer import SegFormer

# Utils
from utils.dataloader.dataloader import get_train_loader,get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.init_func import init_weight, group_weight, configure_optimizers
from utils.lr_policy import WarmUpPolyLR
from utils.evaluate_models import evaluate_torch
from utils.hyperparameter_tuning import tune_hyperparameters
from utils.update_config import update_config

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.directory))
        
        return self.writer
    
    def decode_seg_map_sequence(self, label_masks, config=None):
        rgb_masks = []
        for label_mask in label_masks:
            rgb_mask = self.decode_segmap(label_mask, config=config)
            rgb_masks.append(rgb_mask)
        rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
        return rgb_masks
    
    def decode_segmap(self, label_mask, config=None):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
            the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
            in a figure.
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        n_classes = config.num_classes
        np.random.seed(0)  # Set a fixed seed value
        label_colours = np.random.randint(0, 256, size=(config.num_classes, 3))

        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
 
        return rgb

    def visualize_image(self, writer, images, depth, target, output, global_step, config=None):
        grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        grid_image = make_grid(depth[:3].clone().cpu().data, 3, normalize=True)
        writer.add_image('Depth', grid_image, global_step)
        grid_image = make_grid(self.decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                    config=config), 3, normalize=False, value_range=(0, 255))
        writer.add_image('Predicted_label', grid_image, global_step)
        grid_image = make_grid(self.decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                    config=config), 3, normalize=False, value_range=(0, 255))
        writer.add_image('Groundtruth_label', grid_image, global_step)

def setup_dirs(config):
    config.log_dir = config.log_dir+'/run_'+time.strftime('%Y%m%d-%H%M%S', time.localtime()).replace(' ','_')
    if not os.path.isdir(config.log_dir):
        os.makedirs(config.log_dir)
    tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
    if not os.path.isdir(tb_dir):
        os.makedirs(tb_dir)
    tb_summary = TensorboardSummary(tb_dir)
    tb = tb_summary.create_summary()

    return tb, tb_summary

def set_seed(seed):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = (
        True  # train speed is slower after enabling this opts.
    )

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True, warn_only=True)

def train_model_from_config(config):
    import torch
    torch.set_float32_matmul_precision("high")
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True

    # set_seed(config.seed)

    tb, tb_summary = setup_dirs(config)

    train_loader_args = {
        # 'random_noise_rgb': True,
        # 'random_noise_rgb_prob': 0.1,
        # 'random_noise_rgb_amount': 1.0,
        # 'random_black': True,
        # 'random_black_prob': 0.1,
        # 'random_mirror': True,
        # 'random_crop_and_scale': True,
        # 'random_crop': True,
    }

    train_loader, _ = get_train_loader(None, RGBXDataset, config, **train_loader_args)
    val_loader, _ = get_val_loader(None, RGBXDataset, config, 1)

    # Dont ignore the background class
    criterion = nn.CrossEntropyLoss(reduction='mean')
    if train_loader_args.get('random_crop_and_scale', False):
        criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=config.background)
    BatchNorm2d = nn.BatchNorm2d
    
    if "DFormer" in config.backbone:
        model = segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
    if config.backbone == "mit_b2":
        model = cmxmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
    if config.backbone == "xception":
        model = DeepLab(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
    if config.backbone == "segformer":
        model = SegFormer(cfg=config, criterion=criterion)
    
    base_lr = config.lr
    
    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)
    
    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError

    config.niters_per_epoch = config.num_train_imgs // config.batch_size + 1
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ',device, 'model name: ',config.backbone, 'batch size: ',config.batch_size)
    model.to(device)

    start_epoch = 1

    if config.pretrained_model is not None:
        checkpoint = torch.load(config.pretrained_model, map_location=device)
        if 'model' in checkpoint:
            print('loading pretrained model from %s' % config.pretrained_model)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            current_idx = checkpoint['iteration']
            print('starting from epoch: ', start_epoch, 'iteration: ', current_idx)

    optimizer.zero_grad()
    best_miou=0.0

    for epoch in range(start_epoch, config.nepochs+1):
        model.train()
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                        bar_format=bar_format)
        dataloader = iter(train_loader)

        sum_loss = 0
        i=0
        for idx in pbar:
            minibatch = next(dataloader)
            imgs = minibatch['data']
            gts = minibatch['label']
            modal_xs = minibatch['modal_x']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            modal_xs = modal_xs.cuda(non_blocking=True)
    
            # loss = model(imgs, modal_xs, gts)
            output = model(imgs, modal_xs)
            if train_loader_args.get('random_crop_and_scale', False):
                loss = criterion(output, gts.long())[gts.long() != config.background].mean()
            else:
                loss = criterion(output, gts.long())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = (epoch- 1) * config.niters_per_epoch + idx 
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            sum_loss += loss.item()

            if (epoch % config.checkpoint_step == 0 and epoch > int(config.checkpoint_start_epoch) or epoch == 1) and idx == 0:
                tb_summary.visualize_image(tb, imgs, modal_xs, gts, model(imgs, modal_xs), epoch, config=config)

            print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                    + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                    + ' lr=%.4e' % lr \
                    + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))
            pbar.set_description(print_str)

            del imgs, gts, modal_xs, loss
        
        tb.add_scalar('train/loss', sum_loss / len(pbar), epoch)
        tb.add_scalar('train/lr', lr, epoch)

        if (epoch % config.checkpoint_step == 0 and epoch > int(config.checkpoint_start_epoch)):
            torch.cuda.empty_cache()

            with torch.no_grad():
                model.eval()
                device = torch.device('cuda')
                metric = evaluate_torch(model, val_loader, config, device)
                ious, miou = metric.compute_iou()
                acc, macc = metric.compute_pixel_acc()
                f1, mf1 = metric.compute_f1()

                if miou > best_miou:
                    best_miou = miou
                    print('saving model...')
                    save_checkpoint(model, optimizer, epoch, current_idx, os.path.join(config.log_dir, f"epoch_{epoch}_miou_{miou}.pth"))
                
                print('macc: ', macc, 'mf1: ', mf1, 'miou: ',miou,'best: ',best_miou)
                result_line = f'acc: {acc}, macc: {macc}, f1: {f1}, mf1: {mf1}, ious: {ious}, miou: {miou}\n'
                with open(config.log_dir + '/results.txt', 'a') as file:
                    file.write(result_line)
                
                tb.add_scalar('val/macc', macc, epoch)
                tb.add_scalar('val/mf1', mf1, epoch)
                tb.add_scalar('val/miou', miou, epoch)

                del metric, ious, acc, macc, f1, mf1

    return best_miou

def save_checkpoint(model, optimizer, epoch, iteration, path):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        key = k
        if k.split('.')[0] == 'module':
            key = k[7:]
        new_state_dict[key] = v

    state_dict = {}
    state_dict['model'] = new_state_dict
    state_dict['optimizer'] = optimizer.state_dict()
    state_dict['epoch'] = epoch
    state_dict['iteration'] = iteration

    torch.save(state_dict, path)

def prepare_SynthDet_config(config_location, model, dataset_classes, x_channels, x_e_channels, num_epochs, dataset_name=None):
    config_path = config_location
    model_name = model.split('-')[0]
    if dataset_name is None:
        dataset_name = config_path.split('SynthDet.')[-1].split(f"_{model_name}")[0]
        if "Dformer" in dataset_name:
            dataset_name = dataset_name.split(f"_{model_name}")[0]
    
    new_config = update_config(
        config_location, 
        {
            "dataset_name": dataset_name, 
            "classes": dataset_classes, 
            "x_channels": x_channels, 
            "x_e_channels": x_e_channels,
            "nepochs": num_epochs,
        }
    )

    return new_config

def prepare_SUNRGBD_config(config_location, x_channels, x_e_channels, num_epochs):
    new_config = update_config(
        config_location, 
        { 
            "x_channels": x_channels, 
            "x_e_channels": x_e_channels,
            "nepochs": num_epochs,
        }
    )

    return new_config

def run_hyperparameters(config_location, file_path, num_samples, num_hyperparameter_epochs):
    config = importlib.import_module(config_location).config

    best_config_dict = tune_hyperparameters(
        config, 
        num_samples=num_samples, 
        max_num_epochs=num_hyperparameter_epochs, 
        cpus_per_trial=8, 
        gpus_per_trial=1
    )
    
    # Write best hyperparameters to file
    try:
        if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir, exist_ok=True)
        with open(file_path, 'a') as file:
            # Append time to best hyperparameters config
            best_config_dict["date_time"] = time.strftime('%Y%m%d-%H%M%S', time.localtime()).replace(' ','_')
            file.write(str(best_config_dict) + "\n")
    except Exception as e:
        print(f"Error occurred while writing to file: {e}")

    # Update the training config with the best hyperparameters
    config = update_config(
        config_location, 
        best_config_dict
    )

    return config

def train_model(config_location: str, 
                checkpoint_dir: str="checkpoints", 
                model: str="DFormer-Tiny", 
                dataset_classes: str="random", 
                num_hyperparameter_samples: int=20, 
                num_hyperparameter_epochs: int=5, 
                num_epochs: int=60, 
                x_channels: int=3, 
                x_e_channels: int=1,
                dataset_name: str=None, 
                gpus: int=1,
    ):

    if "SynthDet" in config_location:
        config = prepare_SynthDet_config(config_location, model, dataset_classes, x_channels, x_e_channels, num_epochs, dataset_name)
    elif "SUNRGBD" in config_location:
        config = prepare_SUNRGBD_config(config_location, x_channels, x_e_channels, num_epochs)

    if model == "DFormer-Tiny":
        config.decoder = "ham"
        config.backbone = "DFormer-Tiny"
    if model == "DFormer-Large":
        config.decoder = "ham"
        config.backbone = "DFormer-Large"
        config.drop_path_rate = 0.2
    if model == "CMX-B2":
        config.decoder = "MLPDecoder"
        config.backbone = "mit_b2"
    if model == "DeepLab":
        config.backbone = "xception"
    if model == "segformer":
        config.backbone = "segformer"

    config = update_config(
        config_location, 
        {
            "backbone": config.backbone,
            "decoder": config.decoder,
            "checkpoint_dir": checkpoint_dir,
        }
    )

    file_path = os.path.join(config.log_dir, f"x_{x_channels}_x_e_{x_e_channels}_best_hyperparameters.txt")

    if num_hyperparameter_epochs == -1:
        try:
            with open(file_path, 'r') as file:
                # Read the file as a string
                data = file.readlines()[-1]

            # Replace single quotes with double quotes
            data = data.replace("'", '"')

            # Load the string as JSON
            hyperparameters = json.loads(data)

            config = update_config(
                config_location,
                hyperparameters
            )
            print(f"Using hyperparameters from file: {hyperparameters}")
        except Exception as e:
            print(f"Error while processing hyperparameters: {e}")
            print("Getting new hyperparameters")
            config = run_hyperparameters(config_location, file_path, num_hyperparameter_samples, num_hyperparameter_epochs)
    if num_hyperparameter_epochs > 0:
        config = run_hyperparameters(config_location, file_path, num_hyperparameter_samples, num_hyperparameter_epochs)

    best_miou = train_model_from_config(config)

    return best_miou, config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="DFormer.local_configs.SynthDet.SynthDet_black_back_default_2_Dformer_Tiny",
        help="The config to use for training the model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="DFormer",
        help="The model to use for training the model, choose DFormer, CMX or DeepLab",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="The directory to save the model checkpoints",
    )
    parser.add_argument(
        "--dataset_classes",
        type=str,
        default="groceries",
        help="The type of dataset to use for training",
    )
    parser.add_argument(
        "--num_hyperparameter_samples",
        type=int,
        default=10,
        help="The number of samples to use for hyperparameter tuning",
    )
    parser.add_argument(
        "--num_hyperparameter_epochs",
        type=int,
        default=5,
        help="The number of epochs to use for hyperparameter tuning",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=60,
        help="The number of epochs to use for hyperparameter tuning",
    )
    parser.add_argument(
        "--x_channels",
        type=int,
        default=3,
        help="The number of channels in the x modalities",
    )
    parser.add_argument(
        "--x_e_channels",
        type=int,
        default=1,
        help="The number of channels in the x_e modalities",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="The number of GPUs to use for training",
    )
    args = parser.parse_args()

    best_miou, config = train_model(
        config_location=args.config,
        checkpoint_dir=args.checkpoint_dir,
        model=args.model,
        dataset_classes=args.dataset_classes,
        num_hyperparameter_samples=args.num_hyperparameter_samples,
        num_hyperparameter_epochs=args.num_hyperparameter_epochs,
        num_epochs=args.num_epochs,
        x_channels=args.x_channels,
        x_e_channels=args.x_e_channels,
        gpus=args.gpus
    )

    print(f"Best mIoU: {best_miou}")