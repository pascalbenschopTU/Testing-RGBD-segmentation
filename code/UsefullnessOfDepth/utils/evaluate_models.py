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
from utils.dataloader.dataloader import get_val_loader, get_train_loader
from utils.engine.logger import get_logger
from utils.update_config import update_config


class Evaluator(object):
    def __init__(self, num_class, bin_size=1000, ignore_index=255):
        self.num_class = num_class
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((self.num_class,)*2)
        self.bin_size = bin_size
        self.bin_count = 0
        self.bin_confusion_matrix = np.zeros((self.num_class,)*2)
        self.bin_mious = []
        self.bin_stdious = []

    def compute_pixel_acc_class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc * 100
    
    def compute_f1(self):
        TP = np.diag(self.confusion_matrix)
        FP = self.confusion_matrix.sum(axis=0) - TP
        FN = self.confusion_matrix.sum(axis=1) - TP
        F1 = 2 * TP / (2 * TP + FP + FN)
        mF1 = np.nanmean(F1)
        f1 = F1[~np.isnan(F1)]
        return np.round(f1 * 100, 2), np.round(mF1 * 100, 2)
    
    def compute_iou(self):
        ious = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - 
                    np.diag(self.confusion_matrix))
        if np.all(np.isnan(ious)):
            print("All ious are NaN", ious)
        miou = np.nanmean(ious)
        ious = ious[~np.isnan(ious)]
        iou_std = np.std(ious)
        return np.round(ious * 100, 3), np.round(iou_std * 100, 3), np.round(miou * 100, 3)
    
    def compute_bin_iou(self):
        ious = np.diag(self.bin_confusion_matrix) / (
                    np.sum(self.bin_confusion_matrix, axis=1) + np.sum(self.bin_confusion_matrix, axis=0) - 
                    np.diag(self.bin_confusion_matrix))
        if np.all(np.isnan(ious)):
            print("All ious are NaN", ious)
        miou = np.nanmean(ious)
        ious = ious[~np.isnan(ious)]
        iou_std = np.std(ious)
        return np.round(ious * 100, 3), np.round(iou_std * 100, 3), np.round(miou * 100, 3)
    
    def compute_pixel_acc(self):
        acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        macc = np.nanmean(acc)
        acc = acc[~np.isnan(acc)]
        return np.round(acc * 100, 2), np.round(macc * 100, 2)
    
    def Mean_Intersection_over_Union_non_zero(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = MIoU[MIoU != 0.]
        MIoU = np.nanmean(MIoU)
        return MIoU * 100

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU * 100

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class) & (gt_image != self.ignore_index)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        self.bin_confusion_matrix += self._generate_matrix(gt_image, pre_image)
        if self.bin_count == self.bin_size:
            _, bin_stdiou, bin_miou = self.compute_bin_iou()
            self.bin_mious.append(bin_miou)
            self.bin_stdious.append(bin_stdiou)
            self.bin_confusion_matrix = np.zeros((self.num_class,) * 2)
            self.bin_count = 0
        else:
            self.bin_count += 1

    def calculate_results(self, ignore_class=-1, ignore_zero=False):
        # if zeros in the confusion matrix add a very small value to avoid division by zero
        if np.any(np.sum(self.confusion_matrix, axis=1) == 0):
            self.confusion_matrix += np.eye(self.num_class) * 1e-32

        if self.bin_count > 0:
            _, bin_stdiou, bin_miou = self.compute_bin_iou()
            self.bin_mious.append(bin_miou)
            self.bin_stdious.append(bin_stdiou)
        if ignore_class != -1:
            self.confusion_matrix = np.delete(self.confusion_matrix, ignore_class, axis=0)
            self.confusion_matrix = np.delete(self.confusion_matrix, ignore_class, axis=1)

        if ignore_zero:
            # calculate iou, pixel acc, f1, etc. without classes that have no occurrences
            non_zero_classes = np.any(self.confusion_matrix != 0, axis=1)
            self.confusion_matrix = self.confusion_matrix[non_zero_classes, :]
            self.confusion_matrix = self.confusion_matrix[:, non_zero_classes]

        self.ious, self.iou_std, self.miou = self.compute_iou()
        self.acc, self.macc = self.compute_pixel_acc()
        self.f1, self.mf1 = self.compute_f1()
        self.pixel_acc_class = self.compute_pixel_acc_class()
        self.fwiou = self.Frequency_Weighted_Intersection_over_Union()


    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def set_config_if_dataset_specified(config_location, dataset_name):
    config = update_config(
        config_location,
        {
            "dataset_name": dataset_name,
        }
    )
    return config
    
logger = get_logger()
def load_config(config, dataset=None, x_channels=-1, x_e_channels=-1):
    module_name = config.replace(".py", "").replace("\\", ".").lstrip(".")
    
    if dataset is not None:
        config = set_config_if_dataset_specified(module_name, dataset)
    
    config_module = importlib.import_module(module_name)
    config = config_module.config
    
    if x_channels != -1 and x_e_channels != -1:
        config.x_channels = x_channels
        config.x_e_channels = x_e_channels

    
    
    return config

def initialize_model(config, model_weights, device):
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
    BatchNorm2d = nn.BatchNorm2d

    model = ModelWrapper(config, criterion=criterion, norm_layer=BatchNorm2d, pretrained=True)
    
    try:
        weight = torch.load(model_weights)['model']
        model.load_state_dict(weight, strict=False)
        logger.info(f'load model {config.backbone} weights : {model_weights}')
    except Exception as e:
        logger.error(f"Invalid model weights file: {model_weights}. Error: {e}")
    
    model.to(device)
    return model

def plot_confusion_matrix(confusion_matrix, class_names, model_weights_dir, miou):
    epsilon = 1e-7  # Small constant
    confusion_matrix = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1)[:, np.newaxis] + epsilon)
    
    default_figsize = (10, 10)
    if len(class_names) > 15:
        default_figsize = (20, 20)
    if len(class_names) < 5:
        default_figsize = (5, 5)

    plt.figure(figsize=default_figsize)
    sns.heatmap(confusion_matrix, annot=True, fmt=".3f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix, mIoU: {miou:.2f}")
    plt.tight_layout()
    
    result_file_name = os.path.join(model_weights_dir, 'confusion_matrix.png')
    if os.path.exists(result_file_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file_name = os.path.join(model_weights_dir, f'confusion_matrix_{timestamp}.png')
    
    plt.savefig(result_file_name)

def save_results(model_results_file, metric, num_params, rgb_metric=None, depth_metric=None):
    class_ious = "[" + " ".join([f"{iou:.1f}" for iou in metric.ious]) + "]"


    with open(model_results_file, 'a') as f:
        f.write(f'miou: {metric.miou:.2f}, macc: {metric.macc:.2f}, mf1: {metric.mf1:.2f}, class ious: {class_ious}\n')
        if num_params != 0:
            f.write(f'model parameters: {num_params}\n')
        if len(metric.bin_mious) > 1:
            f.write(f'bin_mious: {metric.bin_mious} ')
            f.write(f'bin_stdious: {metric.bin_stdious}\n')
        if rgb_metric is not None and depth_metric is not None:
            f.write(f'rgb_miou: {rgb_metric.miou:.2f}, depth_miou: {depth_metric.miou:.2f}\n')
            if len(rgb_metric.bin_mious) > 1:
                f.write(f'rgb_bin_mious: {rgb_metric.bin_mious} ')
                f.write(f'rgb_bin_stdious: {rgb_metric.bin_stdious}\n')
            if len(depth_metric.bin_mious) > 1:
                f.write(f'depth_bin_mious: {depth_metric.bin_mious} ')
                f.write(f'depth_bin_stdious: {depth_metric.bin_stdious}\n') 

def get_scores_for_model(
        model, 
        config, 
        model_weights, 
        dataset=None, 
        bin_size=1000,
        x_channels=-1,
        x_e_channels=-1, 
        results_file="results.txt",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        create_confusion_matrix=True,
        ignore_background=False,
        return_metrics=False
    ):
    print("device: ", device)

    # Load and configure model
    config = load_config(config, dataset, x_channels, x_e_channels)

    if ignore_background:
        config.background = 0

    # Initialize model
    model = initialize_model(config, model_weights, device)
    
    # Get validation loader
    val_loader, val_sampler = get_val_loader(None, RGBXDataset, config, 1)

    metric, rgb_metric, depth_metric = evaluate(
        model=model,
        dataloader=val_loader,
        config=config,
        device=device,
        bin_size=bin_size,
    )

    miou = metric.miou

    # Plot confusion matrix
    if create_confusion_matrix:
        plot_confusion_matrix(metric.confusion_matrix, config.class_names, os.path.dirname(model_weights), miou)

    # Save results
    model_weights_dir = os.path.dirname(model_weights)
    model_results_file = os.path.join(model_weights_dir, results_file)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    extra_metrics = [metric.miou for metric in (rgb_metric, depth_metric) if metric is not None]
    if len(extra_metrics) > 0:
        miou = [miou] + extra_metrics

    save_results(model_results_file, metric, num_params, rgb_metric, depth_metric)

    if return_metrics:
        return metric, rgb_metric, depth_metric
    return miou
   
            
@torch.no_grad()
def evaluate(model, dataloader, config, device, bin_size=1):
    model.eval()
    n_classes = config.num_classes
    evaluator = Evaluator(n_classes, bin_size=bin_size, ignore_index=config.background)

    rgb_evaluator = None
    depth_evaluator = None

    if model.is_token_fusion or config.get('use_aux', False) or model.model_name == "HIDANet":
        rgb_evaluator = Evaluator(n_classes, bin_size=bin_size, ignore_index=config.background)
        depth_evaluator = Evaluator(n_classes, bin_size=bin_size, ignore_index=config.background)

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

        if model.model_name == "HIDANet":
            labels = labels.squeeze()
            preds_rgb = preds[0].squeeze().sigmoid()
            preds_depth = preds[1].squeeze().sigmoid()
            # Convert sigmoid outputs to integers based on a threshold
            preds_rgb_int = (preds_rgb > 0.5).int()
            preds_depth_int = (preds_depth > 0.5).int()
            rgb_evaluator.add_batch(labels.cpu().numpy(), preds_rgb_int.cpu().numpy())
            depth_evaluator.add_batch(labels.cpu().numpy(), preds_depth_int.cpu().numpy())
            preds = preds[2].squeeze().sigmoid() # Ensemble
            # Convert ensemble predictions to integers
            preds_int = (preds > 0.5).int()

            evaluator.add_batch(labels.cpu().numpy(), preds_int.cpu().numpy())

        if config.get('use_aux', False):
            preds, aux_preds = preds[0], preds[1]
            binary_labels = (labels > 0).long()
            rgb_evaluator.add_batch(binary_labels.cpu().numpy(), aux_preds.softmax(1).argmax(1).cpu().numpy())

        if not model.model_name == "HIDANet":
            preds = preds.softmax(dim=1)

            evaluator.add_batch(labels.cpu().numpy(), preds.argmax(1).cpu().numpy())

    evaluator.calculate_results(ignore_class=config.background)
    if rgb_evaluator is not None and depth_evaluator is not None:
        rgb_evaluator.calculate_results(ignore_class=config.background)
        depth_evaluator.calculate_results(ignore_class=config.background)
        
    return evaluator, rgb_evaluator, depth_evaluator


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config', help='train config file path')
    argparser.add_argument('-mw', '--model_weights', help='File of model weights')
    argparser.add_argument('-d', '--dataset', default=None, help='Dataset dir')

    argparser.add_argument('-m', '--model', help='Model name', default='DFormer-Tiny')
    argparser.add_argument('-b', '--bin_size', help='Bin size for testing', default=1000, type=int)
    # argparser.add_argument('-ib', '--ignore_background', action='store_true', help='Ignore background class')

    argparser.add_argument('-xc', '--x_channels', help='Number of channels in X', default=-1, type=int)
    argparser.add_argument('-xec', '--x_e_channels', help='Number of channels in X_e', default=-1, type=int)

    args = argparser.parse_args()

    get_scores_for_model(
        model_weights=args.model_weights,
        config=args.config,
        model=args.model,
        dataset=args.dataset,
        bin_size=args.bin_size,
        x_channels=args.x_channels,
        x_e_channels=args.x_e_channels
    )
