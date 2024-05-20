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
from utils.background_remover import remove_background
from utils.evaluate_models import get_scores_for_model

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="configs.SynthDet.SynthDet_template_DFormer_Tiny",
        help="The config file to use for training the model",
    )
    parser.add_argument(
        "-d", "--dataset_dir",
        type=str,
        default="datasets",
        help="The directory containing the datasets to use for training",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="DFormer-Tiny",
        help="The model to use for training the model, choose DFormer, CMX or DeepLab",
    )
    parser.add_argument(
        "-chdir", "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="The directory to save the model checkpoints",
    )
    parser.add_argument(
        "-dc", "--dataset_classes",
        type=str,
        default="groceries",
        help="The type of dataset to use for training",
    )
    parser.add_argument(
        "-hs", "--num_hyperparameter_samples",
        type=int,
        default=20,
        help="The number of samples to use for hyperparameter tuning",
    )
    parser.add_argument(
        "-he", "--num_hyperparameter_epochs",
        type=int,
        default=5,
        help="The number of epochs to use for hyperparameter tuning",
    )
    parser.add_argument(
        "-e", "--num_epochs",
        type=int,
        default=60,
        help="The number of epochs to use for hyperparameter tuning",
    )
    args = parser.parse_args()
    date_time = time.strftime("%Y%m%d_%H%M%S")

    config_location = args.config
    if ".py" in config_location:
        config_location = config_location.replace(".py", "")
        config_location = config_location.replace("\\", ".")
        while config_location.startswith("."):
            config_location = config_location[1:]

    # Load the config file
    config_module = importlib.import_module(config_location)
    config = config_module.config

    # For dataset in args.dataset_dir:
    # Train the model with the dataset on RGB-D (model A) and Depth
    # Remove background with model trained on depth
    # Train the model on RGB with the background removed (model B)
    # Evaluate the models (A and B) on this and other datasets in dataset_dir and report the difference with RGB-D

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    
    log_file = os.path.join(args.checkpoint_dir, f"log_{date_time}.txt")
    with open(log_file, "w") as f:
        f.write("Log file for foreground background separation\n\n")

    for dataset_name in os.listdir(args.dataset_dir):
        # Update the config with the dataset_name details

        with open(log_file, "a") as f:
            f.write(f"\nModel trained on dataset: {dataset_name}\n\n")

        rgbd_best_miou, config = train_model(
            config_location,
            checkpoint_dir=args.checkpoint_dir,
            dataset_classes=args.dataset_classes,
            num_hyperparameter_samples=args.num_hyperparameter_samples,
            num_hyperparameter_epochs=args.num_hyperparameter_epochs,
            num_epochs=args.num_epochs,
            x_channels=3,
            x_e_channels=1,
            dataset_name=dataset_name,
        )

        rgbd_model_weights_file = None
        rgbd_model_weights_dir = config.log_dir
        for root, dirs, files in os.walk(rgbd_model_weights_dir):
            for file in files:
                if file.startswith("epoch"):
                    rgbd_model_weights_file = os.path.join(root, file)

        with open(log_file, "a") as f:
            f.write(f"RGB-D mIoU: {rgbd_best_miou}\nModel best weights: {rgbd_model_weights_file}\n")


        depth_best_miou, config = train_model(
            config_location,
            checkpoint_dir=args.checkpoint_dir,
            dataset_classes=args.dataset_classes,
            num_hyperparameter_samples=args.num_hyperparameter_samples,
            num_hyperparameter_epochs=args.num_hyperparameter_epochs,
            num_epochs=args.num_epochs,
            x_channels=1,
            x_e_channels=1,
            dataset_name=dataset_name,
        )

        # Get the last run folder in the checkpoint directory

        depth_model_weights_file = None
        depth_model_weights_dir = config.log_dir
        for root, dirs, files in os.walk(depth_model_weights_dir):
            for file in files:
                if file.startswith("epoch"):
                    depth_model_weights_file = os.path.join(root, file)

        with open(log_file, "a") as f:
            f.write(f"Depth mIoU: {depth_best_miou}\nModel best weights: {depth_model_weights_file}\n")
        
        
        remove_background(
            model_weights_dir=depth_model_weights_file,
            dataset_dir=os.path.join(args.dataset_dir, dataset_name),
            config=config,
            new_folder="background_removed",
            x_channels=1,
            x_e_channels=1,
            test_only=False,
        )

        # Move RGB folder to RGB_original
        # Move background_removed folder to RGB
        # Train model on RGB with background removed

        # Move RGB folder to RGB_original
        rgb_folder = os.path.join(args.dataset_dir, dataset_name, "RGB")
        rgb_original_folder = os.path.join(args.dataset_dir, dataset_name, "RGB_original")
        os.rename(rgb_folder, rgb_original_folder)

        # Move background_removed folder to RGB
        background_removed_folder = os.path.join(args.dataset_dir, dataset_name, "background_removed")
        os.rename(background_removed_folder, rgb_folder)

        # Train model on RGB with background removed
        rgb_background_removed_best_miou, config = train_model(
            config_location,
            checkpoint_dir=args.checkpoint_dir,
            dataset_classes=args.dataset_classes,
            num_hyperparameter_samples=args.num_hyperparameter_samples,
            num_hyperparameter_epochs=args.num_hyperparameter_epochs,
            num_epochs=args.num_epochs,
            x_channels=3,
            x_e_channels=3,
            dataset_name=dataset_name,
        )

        rgb_background_removed_weights_file = None
        rgb_background_removed_weights_dir = config.log_dir
        for root, dirs, files in os.walk(rgb_background_removed_weights_dir):
            for file in files:
                if file.startswith("epoch"):
                    rgb_background_removed_weights_file = os.path.join(root, file)

        with open(log_file, "a") as f:
            f.write(f"RGB background removed mIoU: {rgb_background_removed_best_miou}\nModel best weights: {rgb_background_removed_weights_file}\n\n")

        # Move RGB_original folder to RGB
        os.rename(rgb_folder, background_removed_folder)
        os.rename(rgb_original_folder, rgb_folder)

        for other_dataset_name in os.listdir(args.dataset_dir):
            rgbd_miou = get_scores_for_model(
                model=args.model,
                config=config_location,
                model_weights=rgbd_model_weights_file,
                dataset=os.path.join(args.dataset_dir, other_dataset_name),
                x_channels=3,
                x_e_channels=1,
                ignore_background=True,
                device=device,
            )

            remove_background(
                model_weights_dir=depth_model_weights_file,
                dataset_dir=os.path.join(args.dataset_dir, other_dataset_name),
                config=config,
                new_folder="background_removed",
                x_channels=1,
                x_e_channels=1,
                test_only=True,
            )

            # Move RGB folder to RGB_original
            # Move background_removed folder to RGB
            # Train model on RGB with background removed

            rgb_folder = os.path.join(args.dataset_dir, other_dataset_name, "RGB")
            rgb_original_folder = os.path.join(args.dataset_dir, other_dataset_name, "RGB_original")
            os.rename(rgb_folder, rgb_original_folder)

            background_removed_folder = os.path.join(args.dataset_dir, other_dataset_name, "background_removed")
            os.rename(background_removed_folder, rgb_folder)


            rgb_background_removed_miou = get_scores_for_model(
                model=args.model,
                config=config_location,
                model_weights=rgb_background_removed_weights_file,
                dataset=os.path.join(args.dataset_dir, other_dataset_name),
                x_channels=3,
                x_e_channels=3,
                ignore_background=True,
                device=device,
            )

            # Move RGB_original folder to RGB
            os.rename(rgb_folder, background_removed_folder)
            os.rename(rgb_original_folder, rgb_folder)

            with open(log_file, "a") as f:
                f.write(
                    f"Dataset: {other_dataset_name} RGB-D mIoU: {rgbd_miou} RGB background removed mIoU: {rgb_background_removed_miou}\n"
                )





