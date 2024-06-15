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
from utils.update_config import update_config


device = "cuda" if torch.cuda.is_available() else "cpu"

x_channels_map = {"rgbd": 3, "rgbd_variation": 3, "rgb": 3, "rgb_variation": 3, "depth": 1}
x_e_channels_map = {"rgbd": 1, "rgbd_variation":1, "rgb": 3, "rgb_variation": 3, "depth": 1}

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
        if f"Model trained on dataset: {dataset_name}" in log_file_contents:
            for line in log_file_contents.split("\n"):
                if "Model best weights" in line and model_file_index < len(model_file_names):
                    model_files[model_file_names[model_file_index]] = line.split(": ")[1]
                    model_file_index += 1

    print(f"Model files: {model_files}")
    # Determine which models need to be trained
    models_to_train = [name for name, file in model_files.items() if file is None]
    
    if len(models_to_train) == 0:
        print("Models already trained on dataset, skipping training")
        return model_files
    
    if len(models_to_train) == len(model_file_names):
        with open(log_file, "a") as f:
            f.write(f"\nModel trained on dataset: {dataset_name}\n\n")


    # First train the models on the dataset without any variations
    update_config(config_location, {"random_color_jitter": False})

    for model_name in models_to_train:
        x_channels = x_channels_map.get(model_name, 3)
        x_e_channels = x_e_channels_map.get(model_name, 1)

        if "variation" in model_name:
            update_config(
                config_location, 
                {
                    "random_color_jitter": True,
                    "min_color_jitter": 0.7,
                    "max_color_jitter": 1.3,
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

        if "variation" in model_name:
            update_config(config_location, {"random_color_jitter": False})

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
        default=15,
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
        help="The log file to save the results of the experiments",
    )
    parser.add_argument(
        "-exp", "--experiments",
        nargs="+",
        type=str,
        default=["saturation", "brightness", "hue"],
        help="The experiments to run on the dataset",
    )
    parser.add_argument(
        "-mti", "--max_train_images",
        type=int,
        default=200,
        help="The maximum number of training images to use for training",
    )
    parser.add_argument(
        "-mn", "--model_names",
        nargs="+",
        type=str,
        default=["rgbd", "rgbd_variation"],
        help="The names of the models to train",
    )
    args = parser.parse_args()
    date_time = time.strftime("%Y%m%d_%H%M%S")

    config_location = args.config.replace(".py", "").replace("\\", ".").lstrip(".")

    # Load the config file
    config_module = importlib.import_module(config_location)
    config = config_module.config

    # For dataset in args.dataset_dir:
    # Train the model with the dataset on RGB-D (model A) and Depth
    # Adapt the dataset to induce variations in saturation, brightness and hue
    # Evaluate the models (A and B) on the adapted dataset and the test dataset with an even wider range of variations

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    
    if args.log_file is not None:
        log_file = args.log_file
    else:
        log_file = os.path.join(args.checkpoint_dir, f"log_{date_time}.txt")
        with open(log_file, "w") as f:
            f.write("Log file for robustness experiment\n\n")
            f.write(f"Arguments: {args}\n\n")

    dataset_name = args.dataset_dir

    RGB_folder = os.path.join(args.dataset_dir, "RGB")
    Depth_folder = os.path.join(args.dataset_dir, "Depth")
    labels_folder = os.path.join(args.dataset_dir, "labels")
    if not os.path.exists(RGB_folder) or not os.path.exists(Depth_folder) or not os.path.exists(labels_folder):
        raise FileNotFoundError(f"Dataset {dataset_name} not found in datasets folder")
    
    dataset_length = len([f for f in os.listdir(RGB_folder) if f.startswith("test")])
    
    RGB_original_folder = os.path.join(args.dataset_dir, "RGB_original")
    if not os.path.exists(RGB_original_folder):
        # Copy RGB to RGB_original with python
        shutil.copytree(RGB_folder, RGB_original_folder)

    model_file_names = args.model_names
    model_files = train_models(
        log_file=log_file,
        args=args,
        config_location=config_location,
        dataset_name=dataset_name,
        model_file_names=model_file_names,
        max_train_images=args.max_train_images,
    )

    experiments = ["saturation", "brightness", "hue"]

    property_values_test = [
        np.linspace(0.001, 2.0, 11),
        np.linspace(0.001, 2.0, 11),
        np.linspace(-0.5, 0.5, 11),
    ]

    experiment_indexes = [i for i in range(len(experiments)) if experiments[i] in args.experiments]
    print(f"The experiments to run are: {[experiments[i] for i in experiment_indexes]}")

    for i in experiment_indexes:
        with open(log_file, "a") as f:
            f.write(f"\nExperiment: {experiments[i]}\n\n")

        property_values = property_values_test[i]
        for property_value in property_values:
            
            adapt_property(
                origin_directory_path=RGB_original_folder,
                destination_directory_path=RGB_folder,
                property_value=property_value,
                property_name=experiments[i],
                split="test",
            )

            with open(log_file, "a") as f:
                f.write(
                    f"Property: {experiments[i]} Property value: {property_value}\n"
                )

            for model_name in model_file_names:
                model_weights_file = model_files[model_name]
                x_channels = x_channels_map.get(model_name, 3)
                x_e_channels = x_e_channels_map.get(model_name, 1)

                print(f"Model: {model_name}, x_channels: {x_channels}, x_e_channels: {x_e_channels}")

                miou_value = get_scores_for_model(
                    model=args.model,
                    config=config_location,
                    model_weights=model_weights_file,
                    dataset=dataset_name,
                    x_channels=x_channels,
                    x_e_channels=x_e_channels,
                    device=device,
                    create_confusion_matrix=False,
                )

                with open(log_file, "a") as f:
                    f.write(
                        f"{model_name.upper()} mIoU: {miou_value}\n"
                    )

        # Reset the RGB folder to the original state
        shutil.rmtree(RGB_folder)
        shutil.copytree(RGB_original_folder, RGB_folder)

    # Evaluate the models on the test dataset with lighting changes
    # if args.dataset_dir ends with a slash, remove it
    if args.dataset_dir[-1] == "\\":
        args.dataset_dir = args.dataset_dir[:-1]
    dataset_root = os.path.dirname(args.dataset_dir)
    dataset_test_root_dir = os.path.join(dataset_root, "test")
    if not os.path.exists(dataset_test_root_dir):
        print(f"Test dataset not found in {args.dataset_dir}")
    else:
        for dataset_name in os.listdir(dataset_test_root_dir):
            dataset_test_dir = os.path.join(dataset_test_root_dir, dataset_name)
            # if not dataset_name in args.dataset_dir:
            #     continue
            
            update_config(
                config_location, 
                {
                    "dataset_name": dataset_test_dir,
                }   
            )

            with open(log_file, "a") as f:
                f.write(
                    f"\n\nExperiment: Light angle\n"
                    f"\nTest dataset: {dataset_name}\n"
                )
            
            for model_name in model_file_names:
                model_weights_file = model_files[model_name]
                x_channels = x_channels_map.get(model_name, 3)
                x_e_channels = x_e_channels_map.get(model_name, 1)

                metric, _, _ = get_scores_for_model(
                    model=args.model,
                    config=config_location,
                    model_weights=model_weights_file,
                    dataset=dataset_test_dir,
                    x_channels=x_channels,
                    x_e_channels=x_e_channels,
                    device=device,
                    create_confusion_matrix=False,
                    bin_size=dataset_length,
                    return_bins=True,
                )

                light_angle = np.linspace(-110, 110, 11)

                with open(log_file, "a") as f:
                    for bin_index, bin_value in enumerate(metric.bin_mious):
                        f.write(
                            f"Property: Light angle Property value: {light_angle[bin_index]}\n"
                            f"{model_name.upper()} mIoU: {bin_value}\n"
                        )


    # Revert the config file to the original state
    update_config(
        config_location, 
        {
            "dataset_name": args.dataset_dir,
        }   
    )

    







