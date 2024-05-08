import argparse
import importlib
import os
import sys

sys.path.append('../UsefullnessOfDepth')

import torch
import torch.nn as nn
from torchmetrics import JaccardIndex
# Model
from model_DFormer.builder import EncoderDecoder as segmodel
from models_CMX.builder import EncoderDecoder as cmxmodel
from model_pytorch_deeplab_xception.deeplab import DeepLab
from models_segformer import SegFormer

from utils.dataloader.RGBXDataset import RGBXDataset
from utils.dataloader.dataloader import get_val_loader, get_train_loader
from utils.engine.logger import get_logger

from metrics_new import Metrics
from tqdm import tqdm
import numpy as np
import seaborn as sns
from datetime import datetime

class Evaluator(object):
    def __init__(self, num_class, ignore_index=255):
        self.num_class = num_class
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc * 100

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc * 100
    
    def F1_Score(self):
        TP = np.diag(self.confusion_matrix)
        FP = self.confusion_matrix.sum(axis=0) - TP
        FN = self.confusion_matrix.sum(axis=1) - TP
        F1 = 2 * TP / (2 * TP + FP + FN)
        F1 = np.nanmean(F1)
        return F1 * 100 
    
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
        miou = np.nanmean(ious)
        ious = ious[~np.isnan(ious)]
        iou_std = np.std(ious)
        return np.round(ious * 100, 3), np.round(iou_std * 100, 3), np.round(miou * 100, 3)
    
    def compute_pixel_acc(self):
        acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        macc = np.nanmean(acc)
        acc = acc[~np.isnan(acc)]
        return np.round(acc * 100, 2), np.round(macc * 100, 2)

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU * 100
    
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
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def set_config_if_dataset_specified(config, dataset_location):
    config.dataset_path = dataset_location
    config.rgb_root_folder = os.path.join(config.dataset_path, 'RGB')
    config.gt_root_folder = os.path.join(config.dataset_path, 'labels')
    config.x_root_folder = os.path.join(config.dataset_path, 'Depth')
    config.train_source = os.path.join(config.dataset_path, "train.txt")
    config.eval_source = os.path.join(config.dataset_path, "test.txt")
    return config

def set_model_weights_if_not_specified(config, args):
    try:
        config_dir = "checkpoints"
        dataset_name = config.dataset_name
        model_name = config.backbone
        checkpoint_dir = os.path.join(config_dir, f"{dataset_name}_{model_name}")
        last_run_dir = ""
        for run_dir in os.listdir(checkpoint_dir):
            if "run" in run_dir:
                last_run_dir = run_dir
        last_run_dir = os.path.join(checkpoint_dir, last_run_dir)
        last_model_weights_file = ""
        for file in os.listdir(last_run_dir):
            if "pth" in file:
                last_model_weights_file = file
        args.model_weights = os.path.join(last_run_dir, last_model_weights_file)
        logger.info(f"Using model weights file: {args.model_weights}")
    except Exception as e:
        logger.error(f"Could not find model weights file. Please specify it using --model_weights. Error: {e}")
        return args
    return args


logger = get_logger()
def get_scores_for_model(args, results_file="results.txt"):
    module_name = args.config
    if ".py" in module_name:
        module_name = module_name.replace(".py", "")
        module_name = module_name.replace("\\", ".")
        while module_name.startswith("."):
            module_name = module_name[1:]

    config_module = importlib.import_module(module_name)
    config = config_module.config

    if args.model is not None:
        if args.model == "DFormer-Tiny":
            config.backbone = "DFormer-Tiny"
        if args.model == "DFormer-Large":
            config.backbone = "DFormer-Large"
        if args.model == "CMX_B2":
            config.backbone = "mit_b2"
        if args.model == "Xception":
            config.backbone = "xception"
        if args.model == "segformer":
            config.backbone = "segformer"

    if args.model_weights is None or not ".pth" in args.model_weights or not config.backbone in args.model_weights:
        args = set_model_weights_if_not_specified(config, args)

    model_weights_dir = os.path.dirname(args.model_weights)
    model_results_file = os.path.join(model_weights_dir, results_file)

    if hasattr(args, "dataset") and args.dataset is not None:
        config = set_config_if_dataset_specified(config, args.dataset)

    val_loader, val_sampler = get_val_loader(None, RGBXDataset,config,1)

    criterion = nn.CrossEntropyLoss(reduction='none')
    BatchNorm2d = nn.BatchNorm2d

    if "DFormer" in config.backbone:
        model = segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
    if config.backbone == "mit_b2":
        model = cmxmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
    if config.backbone == "xception":
        model = DeepLab(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
    if config.backbone == "segformer":
        model = SegFormer(cfg=config, criterion=criterion)

    weight = torch.load(args.model_weights)['model']
    # print('load model: ', args.model_weights)
    logger.info(f'load model {config.backbone} weights : {args.model_weights}')
    model.load_state_dict(weight, strict=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get #parameters of model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info('begin testing:')

    # Get class names from config
    class_names = config.class_names
    
    with torch.no_grad():
        model.eval()
        device = torch.device('cuda')
        metric, mious, iou_stds = evaluate(model, val_loader,config, device, bin_size=args.bin_size)
        if hasattr(args, 'ignore_background') and args.ignore_background:
            metric.confusion_matrix = metric.confusion_matrix[1:, 1:]
            class_names = class_names[1:]
        miou = metric.Mean_Intersection_over_Union()
        acc, macc = metric.compute_pixel_acc()
        mf1 = metric.F1_Score()
        pixel_acc_class = metric.Pixel_Accuracy_Class()
        fwiou = metric.Frequency_Weighted_Intersection_over_Union()
        print('miou, macc, mf1, pixel_acc_class, fwiou: ',miou, macc, mf1, pixel_acc_class, fwiou)

        confusion_matrix = metric.confusion_matrix
        import matplotlib.pyplot as plt

        # Normalize confusion matrix
        # confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        epsilon = 1e-7  # Small constant
        confusion_matrix = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1)[:, np.newaxis] + epsilon)

        default_figsize = (10, 10)
        if len(class_names) > 15:
            default_figsize = (20, 20)
        if len(class_names) < 5:
            default_figsize = (5, 5)

        # Plot confusion matrix
        plt.figure(figsize=default_figsize)
        sns.heatmap(confusion_matrix, annot=True, fmt=".3f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix, mIoU: {:.2f}".format(miou))
        plt.tight_layout()
        result_file_name = os.path.join(model_weights_dir, 'confusion_matrix.png')
        if os.path.exists(result_file_name):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file_name = os.path.join(model_weights_dir, f'confusion_matrix_{timestamp}.png')
        plt.savefig(result_file_name)


        with open(model_results_file, 'a') as f:
            f.write(f'miou: {miou:.2f}, macc: {macc:.2f}, mf1: {mf1:.2f}, mious: [{", ".join([f"{iou:.2f}" for iou in mious])}], iou_stds: [{", ".join([f"{iou_std:.2f}" for iou_std in iou_stds])}]\n')
            f.write(f'model parameters: {num_params}\n')

            
            
@torch.no_grad()
def evaluate(model, dataloader, config, device, bin_size=1):
    model.eval()
    n_classes = config.num_classes
    # metrics = Metrics(n_classes, config.background - 100, device)
    evaluator = Evaluator(n_classes, ignore_index=config.background)
    bin_evaluator = Evaluator(n_classes, ignore_index=config.background)
    jaccard = JaccardIndex(task='multiclass', num_classes=n_classes).to(device)

    mious = []
    iou_stds = []

    for i, minibatch in enumerate(tqdm(dataloader, dynamic_ncols=True)):
        images = minibatch["data"][0]
        labels = minibatch["label"][0]
        modal_xs = minibatch["modal_x"][0]
        # print(images.shape,labels.shape)
        images = [images.to(device), modal_xs.to(device)]
        labels = labels.to(device)
        preds = model(images[0], images[1]).softmax(dim=1)

        if len(labels.shape) == 2:
            labels = labels.unsqueeze(0)
        # print(preds.shape,labels.shape)
        # metrics.update(preds, labels)
        evaluator.add_batch(labels.cpu().numpy(), preds.argmax(1).cpu().numpy())
        bin_evaluator.add_batch(labels.cpu().numpy(), preds.argmax(1).cpu().numpy())
        
        if i > 0 and (i % bin_size == 0 or i == len(dataloader) - 1):
            # miou = bin_evaluator.Mean_Intersection_over_Union()
            bin_evaluator.confusion_matrix = bin_evaluator.confusion_matrix[1:, 1:]
            ious, iou_std, miou = bin_evaluator.compute_iou()
            mious.append(miou)
            iou_stds.append(iou_std)
            bin_evaluator.reset()
        
    return evaluator, mious, iou_stds

@torch.no_grad()
def evaluate_torch(model, dataloader, config, device):
    print("Evaluating...")
    model.eval()
    n_classes = config.num_classes
    metrics = Metrics(n_classes, config.background, device)

    for minibatch in tqdm(dataloader, dynamic_ncols=True):
        images = minibatch["data"][0]
        labels = minibatch["label"][0]
        modal_xs = minibatch["modal_x"][0]
        # print(images.shape,labels.shape)
        images = [images.to(device), modal_xs.to(device)]
        labels = labels.to(device)
        preds = model(images[0], images[1]).softmax(dim=1)
        if len(labels.shape) == 2:
            labels = labels.unsqueeze(0)
        # print(preds.shape,labels.shape)
        metrics.update(preds, labels)

        
    return metrics



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', help='train config file path')
    argparser.add_argument('--model_weights', help='File of model weights', default=None)
    argparser.add_argument('--model', help='Model name', default='DFormer-Tiny')
    argparser.add_argument('--bin_size', help='Bin size for testing', default=1, type=int)
    argparser.add_argument('--dataset', help='Dataset dir')
    argparser.add_argument('--ignore_background', action='store_true', help='Ignore background class')

    args = argparser.parse_args()

    get_scores_for_model(args)
