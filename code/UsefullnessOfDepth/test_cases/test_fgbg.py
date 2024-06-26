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

device = "cuda" if torch.cuda.is_available() else "cpu"

x_channels_map = {"rgbd": 3, "rgbd_aux": 3, "rgb": 3, "depth": 1}
x_e_channels_map = {"rgbd": 1, "rgbd_aux":1, "rgb": 3, "depth": 1}

def train_model_on_dataset(args, dataset_name, config_location, x_channels, x_e_channels, max_train_images=500):
    best_miou, config = train_model(
        config_location,
        checkpoint_dir=args.checkpoint_dir,
        model=args.model,
        dataset_classes=args.dataset_classes,
        num_hyperparameter_samples=args.num_hyperparameter_samples,
        num_hyperparameter_epochs=args.num_hyperparameter_epochs,
        num_epochs=args.num_epochs,
        x_channels=x_channels,
        x_e_channels=x_e_channels,
        dataset_name=dataset_name,
        max_train_images=max_train_images,
    )

    best_model_weights_dir = config.log_dir
    for root, dirs, files in os.walk(best_model_weights_dir):
        for file in files:
            if file.startswith("epoch"):
                best_model_weights_file = os.path.join(root, file)

    return best_miou, best_model_weights_file


def train_models(log_file, args, config_location, dataset_name, model_file_names, max_train_images=500):
    model_files = {name: None for name in model_file_names}
    model_file_index = 0

    with open(log_file, "r") as f:
        log_file_contents = f.read()
        has_seen_line = False
        for line in log_file_contents.split("\n"):
            if f"Model trained on dataset: {dataset_name}" in line:
                has_seen_line = True
            if has_seen_line:
                if "Model best weights" in line:
                    model_files[model_file_names[model_file_index]] = line.split(": ")[1]
                    model_file_index += 1
                    if model_file_index >= len(model_file_names):
                        break

    print(f"Model files: {model_files}")
    # Determine which models need to be trained
    models_to_train = [name for name, file in model_files.items() if file is None]
    
    if len(models_to_train) == 0:
        print("Models already trained on dataset, skipping training")
        return model_files
    
    if len(models_to_train) == len(model_file_names):
        with open(log_file, "a") as f:
            f.write(f"\nModel trained on dataset: {dataset_name}\n\n")

    for model_name in models_to_train:
        x_channels = x_channels_map.get(model_name, 3)
        x_e_channels = x_e_channels_map.get(model_name, 1)
        if "aux" in model_name:
            update_config(
                config_location, 
                {
                    "use_aux": True,
                    "aux_rate": 0.5,
                }   
            )

        best_miou, model_weights_file = train_model_on_dataset(
            args,
            dataset_name,
            config_location,
            x_channels=x_channels,
            x_e_channels=x_e_channels,
            max_train_images=max_train_images,
        )

        model_files[model_name] = model_weights_file

        with open(log_file, "a") as f:
            f.write(f"{model_name.upper()} mIoU: {best_miou}\nModel best weights: {model_weights_file}\n")

        if "aux" in model_name:
            update_config(
                config_location, 
                {
                    "use_aux": False,
                    "aux_rate": 0.0,
                }   
            )

    return model_files


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
        default=3,
        help="The number of epochs to use for hyperparameter tuning",
    )
    parser.add_argument(
        "-e", "--num_epochs",
        type=int,
        default=40,
        help="The number of epochs to use for hyperparameter tuning",
    )
    parser.add_argument(
        "-l", "--log_file",
        type=str,
        default=None,
        help="The log file to write the results to",
    )
    parser.add_argument(
        "-t", "--test_only",
        type=bool,
        default=False,
        help="Whether to only test the model",
    )
    parser.add_argument(
        "-mti", "--max_train_images",
        type=int,
        default=200,
        help="The maximum number of images to use for training",
    )
    parser.add_argument(
        "-mn", "--model_names",
        type=str,
        nargs="+",
        default=["rgbd_aux", "rgbd"],
        help="The names of the models to train",
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
    
    if args.log_file is not None:
        log_file = args.log_file
    else:
        log_file = os.path.join(args.checkpoint_dir, f"log_{date_time}.txt")
        with open(log_file, "w") as f:
            f.write("Log file for foreground background separation\n\n")
            f.write(f"Arguments: {args}\n\n")

    for dataset_name in os.listdir(args.dataset_dir):
        dataset_train_file = os.path.join(args.dataset_dir, dataset_name, "train.txt")
        # If the train.txt file is empty, skip the dataset
        if not os.path.exists(dataset_train_file) or os.path.getsize(dataset_train_file) == 0:
            continue

        # If there is a directory containing the dataset name in the checkpoint dir
        # Check if there are three directories starting with run
        # If there are, skip training the model
        # If there are not, train the model
        # model_file_names = ["rgbd_aux", "rgbd", "rgb", "depth", "depth_preprocessed"]
        model_file_names = args.model_names
        model_files = train_models(
            log_file=log_file,
            args=args,
            dataset_name=dataset_name,
            config_location=config_location,
            model_file_names=model_file_names,
            max_train_images=args.max_train_images,
        )


        # Train the models on the dataset with variations
        update_config(
            config_location, 
            {
                "use_aux": False,
                "aux_rate": 0.0,
            }   
        )

        for other_dataset_name in os.listdir(args.dataset_dir):
            with open(log_file, "a") as f:
                    f.write(
                        f"\nDataset: {other_dataset_name}\n"
                    )

            for model_name in model_file_names:
                model_weights_file = model_files[model_name]
                x_channels = x_channels_map.get(model_name, 3)
                x_e_channels = x_e_channels_map.get(model_name, 1)

                miou_value = get_scores_for_model(
                    model=args.model,
                    config=config_location,
                    model_weights=model_weights_file,
                    dataset=os.path.join(args.dataset_dir, other_dataset_name),
                    x_channels=x_channels,
                    x_e_channels=x_e_channels,
                    device=device,
                    create_confusion_matrix=False,
                    ignore_background=True,
                )

                with open(log_file, "a") as f:
                    f.write(
                        f"{model_name.upper()} mIoU: {miou_value}\n"
                    )
