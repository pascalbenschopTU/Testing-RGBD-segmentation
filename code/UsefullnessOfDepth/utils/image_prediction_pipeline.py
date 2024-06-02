import argparse
import importlib
import torch
import torch.functional as F
from torch.utils import data
import torch.nn as nn
import numpy as np
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from torchvision.models.segmentation import deeplabv3_resnet50
import sys
sys.path.append('../UsefullnessOfDepth')

from model_DFormer.builder import EncoderDecoder as segmodel
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.dataloader.dataloader import get_train_loader,get_val_loader
from utils.dataloader.dataloader import ValPre

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_val_loader(engine, dataset,config,gpus=1, setting="val"):
    data_setting = {'rgb_root': config.rgb_root_folder,
                    'rgb_format': config.rgb_format,
                    'gt_root': config.gt_root_folder,
                    'gt_format': config.gt_format,
                    'transform_gt': config.gt_transform,
                    'x_root':config.x_root_folder,
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'class_names': config.class_names,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source,
                    'class_names': config.class_names}
    val_preprocess = ValPre(config.norm_mean, config.norm_std,config.x_is_single_channel,config)

    val_dataset = dataset(data_setting, setting, val_preprocess, config.num_eval_imgs)


    val_sampler = None
    is_shuffle = False
    batch_size = 1

    if engine and engine.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        batch_size = 1
        is_shuffle = False

    val_loader = data.DataLoader(val_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=False,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=val_sampler)

    return val_loader, val_sampler

def set_config_if_dataset_specified(config, dataset_location):
    config.dataset_path = dataset_location
    config.rgb_root_folder = os.path.join(config.dataset_path, 'RGB')
    config.gt_root_folder = os.path.join(config.dataset_path, 'labels')
    config.x_root_folder = os.path.join(config.dataset_path, 'Depth')
    config.train_source = os.path.join(config.dataset_path, "train.txt")
    config.eval_source = os.path.join(config.dataset_path, "test.txt")
    return config

def normalize_image(image):
    image = (image - image.min()) / (image.max() - image.min())
    return image

def make_predictions(
        model_weights_dir,
        dataset_dir,
        config,
        new_folder="predicted",
        x_channels=1,
        x_e_channels=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        test_only=False
):
    config = set_config_if_dataset_specified(config, dataset_dir)
    config.x_channels = x_channels
    config.x_e_channels = x_e_channels
    model = segmodel(cfg=config, criterion=nn.CrossEntropyLoss(reduction='mean'), norm_layer=nn.BatchNorm2d)
    model.load_state_dict(torch.load(model_weights_dir)["model"])
    print(f"Model loaded from {model_weights_dir}")
    model = model.eval()
    model = model.to(device)

    new_folder = os.path.join(dataset_dir, new_folder)
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    file_name_prefix = ["test", "train"]
    dataloader_setting = ["val", "train"]
    if test_only:
        dataloader_setting = ["val"]

    for idx, setting in enumerate(dataloader_setting):
        dataloader, _ = get_val_loader(None, RGBXDataset, config, gpus=1, setting=setting)
        for i, minibatch in enumerate(tqdm(dataloader, dynamic_ncols=True)):
            image_tensor = minibatch["data"].to(device)
            depth_tensor = minibatch["modal_x"].to(device)
            target_tensor = minibatch["label"].to(device)
            prediction = model(image_tensor, depth_tensor)
            prediction_np = prediction[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
            
            image = Image.fromarray((prediction_np * 255).astype(np.uint8))
            image.save(os.path.join(new_folder, f"{file_name_prefix[idx]}_{i}.png"))

    

def remove_background(
        model_weights_dir, 
        dataset_dir, 
        config, 
        new_folder="background_removed",
        x_channels=1,
        x_e_channels=1,
        test_only=False
    ):
    config = set_config_if_dataset_specified(config, dataset_dir)
    config.x_channels = x_channels
    config.x_e_channels = x_e_channels
    model = segmodel(cfg=config, criterion=nn.CrossEntropyLoss(reduction='mean'), norm_layer=nn.BatchNorm2d)
    model.load_state_dict(torch.load(model_weights_dir)["model"])
    print(f"Model loaded from {model_weights_dir}")
    model = model.eval()
    model = model.to(device)

    new_rgb_folder = os.path.join(dataset_dir, new_folder)
    if not os.path.exists(new_rgb_folder):
        os.makedirs(new_rgb_folder)

    file_name_prefix = ["test", "train"]
    dataloader_setting = ["val", "train"]
    if test_only:
        dataloader_setting = ["val"]

    for idx, setting in enumerate(dataloader_setting):
        dataloader, _ = get_val_loader(None, RGBXDataset, config, gpus=1, setting=setting)
        for i, minibatch in enumerate(tqdm(dataloader, dynamic_ncols=True)):
            image_tensor = minibatch["data"].to(device)
            depth_tensor = minibatch["modal_x"].to(device)
            target_tensor = minibatch["label"].to(device)
            prediction = model(image_tensor, depth_tensor)
            prediction_np = prediction[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
            unique_classes = np.unique(prediction_np)
            category = 0
            if category in unique_classes:
                prediction_mask = np.where(prediction_np == category, 0, 1)
                
                image = image_tensor[0, :, :, :].detach().cpu().numpy().transpose(1, 2, 0)
                image = prediction_mask[:, :, np.newaxis] * image
                image = normalize_image(image)

                # Save image
                Image.fromarray((image * 255).astype(np.uint8)).save(os.path.join(new_rgb_folder, f"{file_name_prefix[idx]}_{i}.png"))
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remove background from images')
    parser.add_argument('-mw', '--model_weights_dir', type=str, help='Path to the model weights')
    parser.add_argument('-c', '--config', type=str, help='Path to the config file')
    parser.add_argument('-d', '--dataset_dir', type=str, help='Path to the dataset')
    parser.add_argument('-nf', '--new_folder', type=str, default="background_removed", help='Name of the new folder to save the images')
    args = parser.parse_args()
    
    module_name = args.config
    if ".py" in module_name:
        module_name = module_name.replace(".py", "")
        module_name = module_name.replace("\\", ".")
        while module_name.startswith("."):
            module_name = module_name[1:]

    args.config = module_name

    config_module = importlib.import_module(args.config)
    config = config_module.config
    remove_background(args.model_weights_dir, args.dataset_dir, config, new_folder=args.new_folder)