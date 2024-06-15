import os
import sys
import time
import json
import random
import torch
import argparse
import importlib

sys.path.append('../UsefullnessOfDepth')

from utils.update_config import update_config
from utils.train import train_model
from utils.image_prediction_pipeline import make_predictions
from utils.evaluate_models import get_scores_for_model
from utils.adjust_background_dataset import merge_datasets

device = "cuda" if torch.cuda.is_available() else "cpu"

x_channels_map = {"rgbd": 3, "rgbd_aux": 3, "rgb": 3, "depth": 1}
x_e_channels_map = {"rgbd": 1, "rgbd_aux":1, "rgb": 3, "depth": 1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="configs.SynthDet.SynthDet_template_DFormer_Tiny",
        help="The config file to use for training the model",
    )
    parser.add_argument(
        "-tc", "--test_config",
        type=str,
        default="configs.SynthDet.SynthDet_template_DFormer_Tiny",
        help="The config file to use for testing the model",
    )
    parser.add_argument(
        "-mw", "--model_weights",
        type=str,
        default=None,
        help="The model weights to use for testing the model",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="DFormer",
        help="The model to use for training the model, choose DFormer, CMX or DeepLab",
    )
    parser.add_argument(
        "-chdir", "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="The directory to save the model checkpoints",
    )
    parser.add_argument(
        "-l", "--log_file",
        type=str,
        default=None,
        help="The log file to write the results to",
    )
    parser.add_argument(
        "-bgdp", "--background_dataset_path",
        type=str,
        help="Path to the dataset where the background images are stored",
    )
    parser.add_argument(
        "-redp", "--result_dataset_path",
        type=str,
        help="Path to the dataset where the crops will be pasted",
    )
    args = parser.parse_args()
    date_time = time.strftime("%Y%m%d_%H%M%S")

    nyudv2_config_module = args.config.replace(".py", "").replace("\\", ".").lstrip(".")

    test_config_location = args.test_config.replace(".py", "").replace("\\", ".").lstrip(".")

    # Load the config file
    config_module = importlib.import_module(nyudv2_config_module)
    config = config_module.config

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    
    if args.log_file is not None:
        log_file = args.log_file
    else:
        log_file = os.path.join(args.checkpoint_dir, f"log_{date_time}.txt")
        with open(log_file, "w") as f:
            f.write("Log file for foreground background separation on NYUDV2\n\n")
            f.write(f"Arguments: {args}\n\n")

    for class_id in range(1, config.num_classes):
        with open(log_file, "a") as f:
            f.write(
                f"\nClass: {class_id}\n"
            )
        
        merge_datasets(config, [class_id], args.background_dataset_path, args.result_dataset_path, original=False)

        metric, _, _ = get_scores_for_model(
            model=args.model,
            config=test_config_location,
            model_weights=args.model_weights,
            dataset=args.result_dataset_path,
            x_channels=3,
            x_e_channels=1,
            device=device,
            create_confusion_matrix=False,
            return_metrics=True,
        )

        miou_class = metric.ious[class_id - 1]

        with open(log_file, "a") as f:
            f.write(
                f"Background adjusted mIoU: {miou_class}\n"
            )

        # Test the model on the dataset without background removed
        merge_datasets(config, [class_id], args.background_dataset_path, args.result_dataset_path, original=True)

        metric, _, _ = get_scores_for_model(
            model=args.model,
            config=test_config_location,
            model_weights=args.model_weights,
            dataset=args.result_dataset_path,
            x_channels=3,
            x_e_channels=1,
            device=device,
            create_confusion_matrix=False,
            return_metrics=True,
        )

        miou_class = metric.ious[class_id - 1]

        with open(log_file, "a") as f:
            f.write(
                f"Original mIoU: {miou_class}\n"
            )
