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

import sys
sys.path.append('../UsefullnessOfDepth')

from utils.model_wrapper import ModelWrapper
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
        model_weights,
        dataset_dir,
        config,
        new_folder="predicted",
        x_channels=3,
        x_e_channels=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        test_only=True,
        classes_of_interest=[],
        model_weights2=None,
        config2=None
):
    config.x_channels = x_channels
    config.x_e_channels = x_e_channels
    model = ModelWrapper(config)
    try:
        weight = torch.load(model_weights)['model']
        print(f'load model {config.backbone} weights : {model_weights}')
        model.load_state_dict(weight, strict=False)
    except Exception as e:
        print(f"Invalid model weights file: {model_weights}. Error: {e}")
    # model.load_state_dict(torch.load(model_weights_dir)["model"])
    print(f"Model loaded from {model_weights}")
    model = model.eval()
    model = model.to(device)

    if model_weights2:
        model2 = ModelWrapper(config2)
        try:
            weight = torch.load(model_weights2)['model']
            print(f'load model {config2.backbone} weights : {model_weights2}')
            model2.load_state_dict(weight, strict=False)
        except Exception as e:
            print(f"Invalid model weights file: {model_weights2}. Error: {e}")
        print(f"Model loaded from {model_weights2}")
        model2 = model2.eval()
        model2 = model2.to(device)

    new_folder = os.path.join(dataset_dir, new_folder)
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    dataloader_setting = ["val", "train"]
    if test_only:
        dataloader_setting = ["val"]

    for idx, setting in enumerate(dataloader_setting):
        dataloader, _ = get_val_loader(None, RGBXDataset, config, gpus=1, setting=setting)
        for i, minibatch in enumerate(tqdm(dataloader, dynamic_ncols=True)):
            image_tensor = minibatch["data"].to(device)
            depth_tensor = minibatch["modal_x"].to(device)
            target_tensor = minibatch["label"].to(device)

            if len(classes_of_interest) > 0:
                # print(target_tensor.shape)
                target_classes = target_tensor[0].detach().cpu().numpy()
                unique_classes = np.unique(target_classes)
               
                contains_classes = [c for c in classes_of_interest if c in unique_classes]
                class_pixels = [np.sum(np.where(target_classes == c, 1, 0)) for c in classes_of_interest]
                # print(f"Unique classes: {unique_classes}, Contains classes: {contains_classes}, Class pixels: {class_pixels}")
                if all([p < 1000 for p in class_pixels]):
                    continue
                # print(unique_classes, classes_of_interest, contains_classes)
                if len(contains_classes) == 0:
                    continue

            class_names = ['bg', 'wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds',
                'desk','shelves','curtain','dresser','pillow','mirror','floor mat','clothes','ceiling','books','refridgerator',
                'television','paper','towel','shower curtain','box','whiteboard','person','night stand','toilet',
                'sink','lamp','bathtub','bag','otherstructure','otherfurniture','otherprop']
                        

            prediction = model(image_tensor, depth_tensor)
            prediction_np = prediction[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
            prediction_image = Image.fromarray((prediction_np).astype(np.uint8))

            image = image_tensor[0, :, :, :].detach().cpu().numpy().transpose(1, 2, 0)
            image = normalize_image(image)

            fig, ax = plt.subplots(2, 5, figsize=(20, 8))
            ax[0, 3].imshow(prediction_image, cmap="viridis")
            ax[0, 3].set_title("Predicted from model: " + config.model)
            ax[0, 3].axis("off")
            ax[0, 1].imshow(image)
            ax[0, 1].set_title("RGB")
            ax[0, 1].axis("off")  
            ax[0, 2].imshow(depth_tensor[0, :, :, :].detach().cpu().numpy().transpose(1, 2, 0)[:, :, 0], cmap="gray")
            ax[0, 2].set_title("Depth")
            ax[0, 2].axis("off")
            ax[0, 0].imshow(target_tensor[0].detach().cpu().numpy(), cmap="viridis")
            ax[0, 0].set_title("Ground Truth")
            ax[0, 0].axis("off")

            if model_weights2:
                prediction2 = model2(image_tensor, depth_tensor)
                prediction_np2 = prediction2[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
                prediction_image2 = Image.fromarray((prediction_np2).astype(np.uint8))
                ax[0, 4].imshow(prediction_image2, cmap="viridis")
                ax[0, 4].set_title("Predicted from model: " + config2.model)
                ax[0, 4].axis("off")

            # Show classes of interest in the label / target_tensor
            for i, class_of_interest in enumerate(classes_of_interest):
                mask = np.where(target_tensor[0].detach().cpu().numpy() == class_of_interest, 1, 0)
                if np.sum(mask) < 1000:
                    continue
                print(f"Category {class_of_interest} with num_pixels: {np.sum(mask)}")
                ax[1, i].imshow(mask, cmap="gray")
                ax[1, i].set_title(f"Class {class_names[class_of_interest]}")
                ax[1, i].axis("off")

            plt.tight_layout()
            plt.show()

            # image.save(os.path.join(new_folder, f"{file_name_prefix[idx]}_{i}.png"))

    

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
    model = ModelWrapper(config)
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
    parser = argparse.ArgumentParser(description="Make predictions using a trained model")
    parser.add_argument("-mw", "--model_weights_dir", type=str, help="Path to the model weights file")
    parser.add_argument("-d", "--dataset_dir", type=str, help="Path to the dataset directory")
    parser.add_argument("-c", "--config", type=str, help="Path to the config file")
    parser.add_argument("-n", "--new_folder", type=str, help="Name of the new folder to store the predictions")
    parser.add_argument("-x", "--x_channels", type=int, help="Number of channels of the X input")
    parser.add_argument("-xe", "--x_e_channels", type=int, help="Number of channels of the X_e input")
    parser.add_argument("-t", "--test_only", action="store_true", help="Only test the model", default=True)
    parser.add_argument("-b", "--background", action="store_true", help="Remove background")
    parser.add_argument("-r", "--remove_background", action="store_true", help="Remove background")
    parser.add_argument("-ci", "--classes_of_interest", nargs='+', help="Classes of interest")
    parser.add_argument("-mw2", "--model_weights_dir2", type=str, help="Path to the model weights file")
    parser.add_argument("-c2", "--config2", type=str, help="Path to the config file")
    args = parser.parse_args()

    config_location = args.config
    if ".py" in config_location:
        config_location = config_location.replace(".py", "")
        config_location = config_location.replace("\\", ".")
        while config_location.startswith("."):
            config_location = config_location[1:]

    config = importlib.import_module(config_location).config
    classes_of_interest = []
    if args.classes_of_interest:
        classes_of_interest = [int(c) for c in args.classes_of_interest]

    if args.config2:
        config_location2 = args.config2
        if ".py" in config_location2:
            config_location2 = config_location2.replace(".py", "")
            config_location2 = config_location2.replace("\\", ".")
            while config_location2.startswith("."):
                config_location2 = config_location2[1:]

        config2 = importlib.import_module(config_location2).config
    else:
        config2 = None

    if args.background:
        remove_background(args.model_weights_dir, args.dataset_dir, config, args.new_folder, args.x_channels, args.x_e_channels, args.test_only)

    else:
        make_predictions(
            args.model_weights_dir,
            args.dataset_dir,
            config,
            args.new_folder,
            args.x_channels,
            args.x_e_channels,
            device,
            args.test_only,
            classes_of_interest,
            args.model_weights_dir2,
            config2
        )