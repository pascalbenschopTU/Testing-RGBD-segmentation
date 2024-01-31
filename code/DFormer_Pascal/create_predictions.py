import torch
import importlib
import os
from tqdm import tqdm
import numpy as np

from simple_dataloader import get_val_loader
from eval import create_predictions
from segmentation_model import SmallUNet

import sys
sys.path.append("../../DFormer/utils/dataloader/")
from RGBXDataset import RGBXDataset

N_CLASSES = 64

# TODO make it so that it only saves predictions that contain interesting classes

@torch.no_grad()
def create_predictions(location, model, dataloader, config, device):
    model.eval()

    n_classes = config.num_classes

    for i, minibatch in enumerate(tqdm(dataloader)):
        images = minibatch["data"]
        labels = minibatch["label"]
        modal_xs = minibatch["modal_x"]
        # print(images.shape,labels.shape)
        images = images.to(device)
        modal_xs = modal_xs.to(device)
        labels = labels.to(device)
        B, H, W = labels.shape

        logits = model(images, modal_xs)
        predictions = logits.softmax(dim=1)

        for j in range(B):
            current_index = i * config.batch_size + j
            np.save(os.path.join(location, f"pred_test_{current_index}.npy"), predictions[j].unsqueeze(0).cpu().numpy())

def main(args):
    config_module = importlib.import_module(args.config)
    config = config_module.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallUNet(args.channels, N_CLASSES, criterion=torch.nn.CrossEntropyLoss(reduction='mean'))
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    val_loader = get_val_loader(RGBXDataset, config)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
  
    create_predictions(args.output_path, model, val_loader, config, device)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--config', type=str, default='config.SynthDet_default_Segmodel')
    parser.add_argument('--model_path', type=str, default='checkpoints/pretrained/SynthDet_pretrained_rgbd.pth')
    parser.add_argument('--output_path', type=str, default='../../DFormer/datasets/SynthDet/predictions')
    args = parser.parse_args()

    main(args)