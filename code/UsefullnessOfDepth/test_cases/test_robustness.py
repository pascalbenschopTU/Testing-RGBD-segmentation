import os
import sys
import time
import json
import random
import shutil
import torch
import argparse
import importlib
import numpy as np

sys.path.append('../UsefullnessOfDepth')

from utils.train import train_model
from utils.adapt_dataset import adapt_property
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
    # Adapt the dataset to induce variations in saturation, brightness and hue
    # Evaluate the models (A and B) on the adapted dataset and the test dataset with an even wider range of variations

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    
    log_file = os.path.join(args.checkpoint_dir, f"log_{date_time}.txt")
    with open(log_file, "w") as f:
        f.write("Log file for robustness experiment\n\n")

    dataset_name = args.dataset_dir

    RGB_folder = os.path.join(args.dataset_dir, "RGB")
    Depth_folder = os.path.join(args.dataset_dir, "Depth")
    labels_folder = os.path.join(args.dataset_dir, "labels")
    if not os.path.exists(RGB_folder) or not os.path.exists(Depth_folder) or not os.path.exists(labels_folder):
        raise FileNotFoundError(f"Dataset {dataset_name} not found in datasets folder")
    
    RGB_original_folder = os.path.join(args.dataset_dir, "RGB_original")
    if not os.path.exists(RGB_original_folder):
        # Copy RGB to RGB_original with python
        shutil.copytree(RGB_folder, RGB_original_folder)


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

    rgb_best_miou, _ = train_model(
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

    rgb_model_weights_file = None
    rgb_model_weights_dir = config.log_dir
    for root, dirs, files in os.walk(rgb_model_weights_dir):
        for file in files:
            if file.startswith("epoch"):
                rgb_model_weights_file = os.path.join(root, file)


    with open(log_file, "a") as f:
        f.write(f"RGB mIoU: {rgb_best_miou}\nModel best weights: {rgb_model_weights_file}\n")

    experiments = ["saturation", "brightness", "hue"]
    property_values_train = [
        (0.7, 1.3),
        (0.7, 1.3),
        (-0.1, 0.1),

    ]
    property_values_test = [
        np.linspace(0.001, 2.0, 11),
        np.linspace(0.001, 2.0, 11),
        np.linspace(-0.5, 0.5, 11),
    ]

    for i, experiment in enumerate(experiments):
        adapt_property(
            origin_directory_path=RGB_original_folder,
            destination_directory_path=RGB_folder,
            property_value=property_values_train[i],
            property_name=experiment,
            split="train",
        )

        adapt_property(
            origin_directory_path=RGB_original_folder,
            destination_directory_path=RGB_folder,
            property_value=property_values_train[i],
            property_name=experiment,
            split="test",
        )
        rgbd_variation_best_miou, config = train_model(
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

        rgbd_variation_model_weights_file = None
        rgbd_variation_model_weights_dir = config.log_dir
        for root, dirs, files in os.walk(rgbd_variation_model_weights_dir):
            for file in files:
                if file.startswith("epoch"):
                    rgbd_variation_model_weights_file = os.path.join(root, file)

        with open(log_file, "a") as f:
            f.write(f"RGB-D variation {experiment} mIoU: {rgbd_variation_best_miou}\nModel best weights: {rgbd_variation_model_weights_file}\n")

        rgb_variation_best_miou, config = train_model(
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

        rgb_variation_model_weights_file = None
        rgb_variation_model_weights_dir = config.log_dir
        for root, dirs, files in os.walk(rgb_variation_model_weights_dir):
            for file in files:
                if file.startswith("epoch"):
                    rgb_variation_model_weights_file = os.path.join(root, file)

        with open(log_file, "a") as f:
            f.write(f"RGB variation {experiment} mIoU: {rgb_variation_best_miou}\nModel best weights: {rgb_variation_model_weights_file}\n\n")
        
        property_values = property_values_test[i]
        for property_value in property_values:
            adapt_property(
                origin_directory_path=RGB_original_folder,
                destination_directory_path=RGB_folder,
                property_value=property_value,
                property_name=experiment,
                split="test",
            )

            miou_values_rgbd_variation = get_scores_for_model(
                model=args.model,
                config=config_location,
                model_weights=rgbd_variation_model_weights_file,
                dataset=args.dataset_dir,
                x_channels=3,
                x_e_channels=1,
                device=device,
                create_confusion_matrix=False,
            )

            miou_values_rgb_variation = get_scores_for_model(
                model=args.model,
                config=config_location,
                model_weights=rgb_variation_model_weights_file,
                dataset=args.dataset_dir,
                x_channels=3,
                x_e_channels=3,
                device=device,
                create_confusion_matrix=False,
            )

            miou_values_rgbd = get_scores_for_model(
                model=args.model,
                config=config_location,
                model_weights=rgbd_model_weights_file,
                dataset=args.dataset_dir,
                x_channels=3,
                x_e_channels=1,
                device=device,
                create_confusion_matrix=False,
            )

            miou_values_rgb = get_scores_for_model(
                model=args.model,
                config=config_location,
                model_weights=rgb_model_weights_file,
                dataset=args.dataset_dir,
                x_channels=3,
                x_e_channels=3,
                device=device,
                create_confusion_matrix=False,
            )

            with open(log_file, "a") as f:
                f.write(
                    f"Property: {experiment} Property value: {property_value}\n"
                    f"RGB-D mIoU: {miou_values_rgbd} RGB-D variation mIoU: {miou_values_rgbd_variation}\n"
                    f"RGB mIoU: {miou_values_rgb} RGB variation mIoU: {miou_values_rgb_variation}\n"
                )

        # Reset the RGB folder to the original state
        shutil.rmtree(RGB_folder)
        shutil.copytree(RGB_original_folder, RGB_folder)





