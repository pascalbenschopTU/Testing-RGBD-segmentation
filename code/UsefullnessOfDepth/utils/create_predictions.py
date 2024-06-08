import argparse
import importlib
import os
import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch
import torch.nn as nn
from torchmetrics import JaccardIndex

sys.path.append('../UsefullnessOfDepth')

# Model
from utils.model_wrapper import ModelWrapper
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.dataloader.dataloader import get_val_loader
from utils.engine.logger import get_logger
from utils.metrics_new import Metrics
from utils.evaluate_models import Evaluator

def set_config_if_dataset_specified(config, dataset_location):
    config.dataset_path = dataset_location
    config.rgb_root_folder = os.path.join(config.dataset_path, 'RGB')
    config.gt_root_folder = os.path.join(config.dataset_path, 'labels')
    config.x_root_folder = os.path.join(config.dataset_path, 'Depth')
    config.train_source = os.path.join(config.dataset_path, "train.txt")
    config.eval_source = os.path.join(config.dataset_path, "test.txt")
    return config
    
logger = get_logger()
def load_config(config, dataset=None, x_channels=-1, x_e_channels=-1):
    module_name = config.replace(".py", "").replace("\\", ".").lstrip(".")
    config_module = importlib.import_module(module_name)
    config = config_module.config
    
    if x_channels != -1 and x_e_channels != -1:
        config.x_channels = x_channels
        config.x_e_channels = x_e_channels

    if dataset is not None:
        config = set_config_if_dataset_specified(config, dataset)
    
    return config

def initialize_model(config, model_weights, device):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    BatchNorm2d = nn.BatchNorm2d

    model = ModelWrapper(config, criterion=criterion, norm_layer=BatchNorm2d, pretrained=True)
    
    try:
        weight = torch.load(model_weights)['model']
        logger.info(f'load model {config.backbone} weights : {model_weights}')
        model.load_state_dict(weight, strict=False)
    except Exception as e:
        logger.error(f"Invalid model weights file: {model_weights}. Error: {e}")
    
    model.to(device)
    return model

def get_scores_for_model(
        model, 
        config, 
        model_weights, 
        dataset=None, 
        bin_size=1000, 
        ignore_background=False, 
        x_channels=3,
        x_e_channels=1, 
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        create_confusion_matrix=True,
        prediction_folder=None
    ):
    print("device: ", device)

    # Load and configure model
    config = load_config(config, dataset, x_channels, x_e_channels)

    if not ignore_background:
        config.background = -1

    # Initialize model
    model = initialize_model(config, model_weights, device)
    
    # Get validation loader
    val_loader, val_sampler = get_val_loader(None, RGBXDataset, config, 1)

    print("Val loader length: ", len(val_loader))

    create_predictions(
        model=model,
        dataloader=val_loader,
        config=config,
        device=device,
        bin_size=bin_size,
        ignore_background=ignore_background,
        create_confusion_matrix=create_confusion_matrix,
        prediction_folder=prediction_folder,
    )


def create_predictions(
        model,
        dataloader,
        config,
        device,
        bin_size=1000,
        ignore_background=False,
        create_confusion_matrix=True,
        prediction_folder=None
    ):
    model.eval()
    n_classes = config.num_classes

    evaluator = Evaluator(n_classes, bin_size=bin_size, ignore_index=config.background)

    rgb_evaluator = None
    depth_evaluator = None

    mious = []
    iou_stds = []

    os.makedirs(prediction_folder, exist_ok=True)

    for i, minibatch in enumerate(tqdm(dataloader, dynamic_ncols=True)):
        images = minibatch["data"][0]
        labels = minibatch["label"][0]
        modal_xs = minibatch["modal_x"][0]

        if len(labels.shape) == 2:
            labels = labels.unsqueeze(0)

        images = [images.to(device), modal_xs.to(device)]
        labels = labels.to(device)
        preds = model(images[0], images[1])
        if model.is_token_fusion:
            preds, masks = preds[0], preds[1]
            preds_rgb = preds[0].softmax(dim=1)
            preds_depth = preds[1].softmax(dim=1)
            rgb_evaluator.add_batch(labels.cpu().numpy(), preds_rgb.argmax(1).cpu().numpy())
            depth_evaluator.add_batch(labels.cpu().numpy(), preds_depth.argmax(1).cpu().numpy())

            preds = preds[2] # Ensemble

        preds = preds.softmax(dim=1)
        preds = preds.argmax(1)

        evaluator.add_batch(labels.cpu().numpy(), preds.cpu().numpy())

        if prediction_folder is not None and i < 100:
            fig, ax = plt.subplots(1, 4, figsize=(16, 5))
            img = images[0].cpu().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())

            label = labels.cpu().numpy()[0]
            prediction = preds[0].cpu().numpy()
            prediction = np.where(label == 0, 0, prediction)

            ax[0].imshow(img)
            ax[0].axis('off')
            ax[0].set_title("RGB Image")
            ax[1].imshow(images[1].cpu().numpy().transpose(1, 2, 0), cmap='gray')
            ax[1].axis('off')
            ax[1].set_title("Depth Image")
            ax[2].imshow(prediction, cmap='viridis')
            ax[2].axis('off')
            ax[2].set_title("Prediction")
            ax[3].imshow(label, cmap='viridis')
            ax[3].axis('off')
            ax[3].set_title("Ground Truth")

            plt.tight_layout()
            plt.savefig(os.path.join(prediction_folder, f"prediction_{i}.png"))
            plt.close()


    evaluator.calculate_results()
    print(evaluator.miou)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DFormer")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_weights", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--bin_size", type=int, default=1000)
    parser.add_argument("--ignore_background", action="store_true")
    parser.add_argument("--x_channels", type=int, default=3)
    parser.add_argument("--x_e_channels", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--create_confusion_matrix", action="store_true")
    parser.add_argument("--prediction_folder", type=str, default=None)
    args = parser.parse_args()

    print("hekkie")

    get_scores_for_model(
        model=args.model,
        config=args.config,
        model_weights=args.model_weights,
        dataset=args.dataset,
        bin_size=args.bin_size,
        ignore_background=args.ignore_background,
        x_channels=args.x_channels,
        x_e_channels=args.x_e_channels,
        device=torch.device(args.device),
        create_confusion_matrix=args.create_confusion_matrix,
        prediction_folder=args.prediction_folder
    )