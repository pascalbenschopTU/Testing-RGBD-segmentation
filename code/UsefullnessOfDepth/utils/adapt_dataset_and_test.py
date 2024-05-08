import torchvision.transforms.functional as F
from PIL import Image
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import shutil
import importlib
from evaluate_models import get_scores_for_model
from update_config import update_config
from scipy.ndimage import gaussian_filter

def adapt_dataset(origin_directory_path, destination_directory_path, property_value, adaptation_method, split):
    paths = [os.path.join(origin_directory_path, file) for file in os.listdir(origin_directory_path) if file.startswith(split)]
    destination_paths = [os.path.join(destination_directory_path, file) for file in os.listdir(origin_directory_path) if file.startswith(split)]
    paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    destination_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                           
    # Initialize the progress bar
    progress_bar = tqdm(total=len(paths), desc=f'Processing files in {origin_directory_path}')
    print("Property value: ", property_value)

    for path, destination_directory_path in zip(paths, destination_paths):
        # new_saturation = adjust_saturation(path, destination_directory_path, saturation_value)
        new_saturation = adaptation_method(path, destination_directory_path, property_value)
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()


def adjust_saturation(image_path, destination_directory_path, saturation_value):
    image = Image.open(image_path)
    saturated_image = F.adjust_saturation(image, saturation_value)

    # Save the image
    saturated_image.save(destination_directory_path)

    return saturation_value

def adjust_hue(image_path, destination_directory_path, hue):
    image = Image.open(image_path)
    color_image = F.adjust_hue(image, hue)

    # Save the image
    color_image.save(destination_directory_path)

    return hue

def adjust_shadow(image_path, destination_directory_path, shadow_percentage):
    # Create random blobs of shadows (black patches) on the image
    # Where the shadow_percentage is the percentage of the image that is shadowed
    assert 0 < shadow_percentage < 1
    image = Image.open(image_path)
    image = np.array(image)

    blob_size = 15 + int(10 * shadow_percentage)

    # Create a binary mask with Gaussian blobs
    mask = np.random.rand(*image.shape[:2]) > shadow_percentage
    mask = gaussian_filter(mask.astype(float), sigma=blob_size)

    # Normalize the mask to ensure values are between 0 and 1
    mask = (mask - mask.min()) / (mask.max() - mask.min())

    # Apply a nonlinear filter so that the mask < 0.5 is 0 and mask > 0.5 is mask
    mask = np.where(mask < shadow_percentage, 0, mask)

    # Create a 3D mask by stacking the 2D mask along the color dimension
    mask = np.stack([mask]*3, axis=-1)

    # Apply the mask to the image, reducing the pixel values where the mask is 1
    shadowed_image = image * mask

    # Convert the image back to its original data type
    shadowed_image = shadowed_image.astype(image.dtype)

    # Save the image
    shadowed_image = Image.fromarray(shadowed_image)
    shadowed_image.save(destination_directory_path)

    return shadow_percentage

def adjust_brightness(image_path, destination_directory_path, brightness_factor):
    image = Image.open(image_path)
    brightened_image = F.adjust_brightness(image, brightness_factor)

    # Save the image
    brightened_image.save(destination_directory_path)

    return brightness_factor



def adapt_property(origin_directory_path, destination_directory_path, property_value, property_name, split):
    if property_name == "saturation":
        adapt_dataset(origin_directory_path, destination_directory_path, property_value, adjust_saturation, split)
    if property_name == "hue":
        adapt_dataset(origin_directory_path, destination_directory_path, property_value, adjust_hue, split)
    if property_name == "shadow":
        adapt_dataset(origin_directory_path, destination_directory_path, property_value, adjust_shadow, split)
    if property_name == "brightness":
        adapt_dataset(origin_directory_path, destination_directory_path, property_value, adjust_brightness, split)


def test_property_shift(args, property_values):
    for property_value in property_values:
        adapt_property(args.origin_directory_path, args.destination_directory_path, property_value, args.property_name, args.split)
        config_module = importlib.import_module(args.config)
        config = config_module.config

        if config.x_e_channels == 3:
            dataset_dir = os.path.dirname(os.path.dirname(args.origin_directory_path))
            depth_dir = os.path.join(dataset_dir, 'Depth')
            if not os.path.exists(depth_dir):
                os.makedirs(depth_dir)

            # Check for the first file if it single channel
            if len(os.listdir(depth_dir)) > 0:
                depth_image = Image.open(os.path.join(depth_dir, os.listdir(depth_dir)[0]))
                if len(np.array(depth_image).shape) == 2:
                    # Move Depth folder to Depth_original
                    os.rename(depth_dir, os.path.join(dataset_dir, 'Depth_original'))

            for file in os.listdir(args.destination_directory_path):
                if file.startswith(args.split):
                    shutil.copy(os.path.join(args.destination_directory_path, file), os.path.join(depth_dir, file))

        get_scores_for_model(args, f"{args.property_name}_tests.txt")

    
def update_config_with_model(args):
    config_module = importlib.reload(importlib.import_module(args.config))
    config = config_module.config

    if args.model == "DFormer-Tiny":
        config.decoder = "ham"
        config.backbone = "DFormer-Tiny"
    if args.model == "DFormer-Large":
        config.decoder = "ham"
        config.backbone = "DFormer-Large"
        config.drop_path_rate = 0.2
    if args.model == "CMX_B2":
        config.decoder = "MLPDecoder"
        config.backbone = "mit_b2"
    if args.model == "DeepLab":
        config.backbone = "xception"

    update_config(
        args.config, 
        {
            "backbone": config.backbone,
            "decoder": config.decoder,
        }
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-op", "--origin_directory_path", type=str, default="datasets/SUNRGBD/RGB_original", help="The path to the SUNRGBD dataset")
    argparser.add_argument("-dp", "--destination_directory_path", type=str, default="datasets/SUNRGBD/RGB", help="The path to save the adapted dataset")
    argparser.add_argument("-s", "--split", type=str, default="test", help="The split to consider", choices=["train", "test"])
    argparser.add_argument("-cfg", '--config', help='train config file path')
    argparser.add_argument("-mw", '--model_weights', help='File of model weights')
    argparser.add_argument("-m", '--model', help='Model name', default='DFormer-Tiny')
    argparser.add_argument("-bs", '--bin_size', help='Bin size for testing', default=1, type=int)
    argparser.add_argument("-pname", '--property_name', help='Property name', default='saturation')
    argparser.add_argument("-minpv", '--min_property_value', help='Minimum property value', default=-1.0, type=float)
    argparser.add_argument("-maxpv", '--max_property_value', help='Maximum property value', default=-1.0, type=float)
    argparser.add_argument("-pvr", '--property_value_range', help='Property value range', default=10, type=int)

    args = argparser.parse_args()

    module_name = args.config
    if ".py" in module_name:
        module_name = module_name.replace(".py", "")
        module_name = module_name.replace("\\", ".")
        while module_name.startswith("."):
            module_name = module_name[1:]

    args.config = module_name

    update_config_with_model(args)

    if args.config is not None:
        if args.property_name == "saturation":
            saturation_values = np.linspace(0.001, 5.0, 10)
            if args.min_property_value != -1.0 and args.max_property_value != -1.0:
                saturation_values = np.linspace(args.min_property_value, args.max_property_value, args.property_value_range)
            test_property_shift(args, saturation_values)

        if args.property_name == "hue":
            hue_values = np.linspace(-0.5, 0.5, 36)
            if args.min_property_value != -1.0 and args.max_property_value != -1.0:
                hue_values = np.linspace(args.min_property_value, args.max_property_value, args.property_value_range)
            test_property_shift(args, hue_values)

        if args.property_name == "shadow":
            shadow_values = np.linspace(0.1, 0.5, 10)
            if args.min_property_value != -1.0 and args.max_property_value != -1.0:
                shadow_values = np.linspace(args.min_property_value, args.max_property_value, args.property_value_range)
            test_property_shift(args, shadow_values)

        if args.property_name == "brightness":
            brightness_values = np.linspace(0.0, 2.0, 21)
            if args.min_property_value != -1.0 and args.max_property_value != -1.0:
                brightness_values = np.linspace(args.min_property_value, args.max_property_value, args.property_value_range)
            test_property_shift(args, brightness_values)
        
    