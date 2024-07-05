import os
import sys
import time
import json
import random
import numpy as np
import torch
import argparse
import importlib
import shutil

sys.path.append('../UsefullnessOfDepth')

from utils.update_config import update_config
from utils.train import train_model
from utils.adapt_dataset_and_test import test_property_shift

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
        default="DFormer",
        help="The model to use for training the model, choose DFormer, CMX or DeepLab",
    )
    parser.add_argument(
        "-mw", "--model_weights",
        type=str,
        default=None,
        help="The model weights to use for training the model",
    )
    parser.add_argument(
        "-dc", "--dataset_classes",
        type=str,
        default="groceries",
        help="The type of dataset to use for training",
    )
    parser.add_argument(
        "-chdir", "--checkpoint_dir",
        type=str,
        default="checkpoints_spatial_NYUDepthV2",
        help="The directory to save the model checkpoints",
    )
    parser.add_argument(
        "-l", "--log_file",
        type=str,
        default=None,
        help="The log file to save the results to",
    )
    args = parser.parse_args()
    date_time = time.strftime("%Y%m%d_%H%M%S")

    config_location = args.config.replace(".py", "").replace("\\", ".").lstrip(".")

    # Load the config file
    config_module = importlib.import_module(config_location)
    config = config_module.config

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if args.log_file is not None:
        log_file = args.log_file
    else:
        log_file = os.path.join(args.checkpoint_dir, f"log_{date_time}.txt")
        with open(log_file, "w") as f:
            f.write("Log file for spatial tests on NYUDepthV2\n\n")
            f.write(f"Arguments: {args}\n\n")
    
    if not os.path.exists(args.dataset_dir):
        raise ValueError("The dataset directory does not exist")
    else:
        dataset_dir = args.dataset_dir
        if not os.path.exists(os.path.join(dataset_dir, "Depth_original")):
            # Copy dataset_dir/Depth to dataset_dir/Depth_original
            shutil.copytree(os.path.join(dataset_dir, "Depth"), os.path.join(dataset_dir, "Depth_original"))

    args.origin_directory_path = os.path.join(dataset_dir, "Depth_original")
    args.destination_directory_path = os.path.join(dataset_dir, "Depth")
    

    # depth_ranges = np.linspace(0.1, 0.9, 9)
    # for depth_range in depth_ranges:
    #     args.property_name = "depth_level"
    #     args.depth_range = depth_range

    #     property_values, miou_values = test_property_shift(
    #         config=args.config, 
    #         property_values=[0], 
    #         model_weights=args.model_weights, 
    #         property_name=args.property_name, 
    #         origin_directory_path=args.origin_directory_path, 
    #         destination_directory_path=args.destination_directory_path, 
    #         model=args.model, 
    #         split="", 
    #         device="cuda",
    #         args=args
    #     )

    #     with open(log_file, "a") as result_file:
    #         result_file.write(f"Depth range: {depth_range}\n")
    #         result_file.write(f"mIoU values: {miou_values}\n")
    #         result_file.write("\n")

    property_values = np.linspace(0, 0.9, 10)
    args.property_name = "depth_level"
    args.depth_range = 0.1

    test_property_shift(
        config=args.config, 
        property_values=property_values, 
        model_weights=args.model_weights, 
        property_name=args.property_name, 
        origin_directory_path=args.origin_directory_path, 
        destination_directory_path=args.destination_directory_path, 
        model=args.model, 
        split="", 
        device="cuda",
        args=args
    )

    property_values = np.linspace(0, 0.8, 5)
    args.property_name = "depth_level"
    args.depth_range = 0.2


    test_property_shift(
        config=args.config, 
        property_values=property_values, 
        model_weights=args.model_weights, 
        property_name=args.property_name, 
        origin_directory_path=args.origin_directory_path, 
        destination_directory_path=args.destination_directory_path, 
        model=args.model, 
        split="",
        device="cuda",
        args=args
    )

    property_values = np.linspace(0, 0.66666, 3)
    args.property_name = "depth_level"
    args.depth_range = 0.33333


    test_property_shift(
        config=args.config, 
        property_values=property_values, 
        model_weights=args.model_weights, 
        property_name=args.property_name, 
        origin_directory_path=args.origin_directory_path, 
        destination_directory_path=args.destination_directory_path, 
        model=args.model, 
        split="",
        device="cuda",
        args=args
    )



        
       
