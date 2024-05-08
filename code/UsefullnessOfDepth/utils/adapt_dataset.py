import torchvision.transforms.functional as F
from PIL import Image
import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse
from skimage.filters import threshold_multiotsu

def adapt_dataset(origin_directory_path, destination_directory_path, property_value, adaptation_method, split):
    paths = [os.path.join(origin_directory_path, file) for file in os.listdir(origin_directory_path) if file.startswith(split)]
    destination_paths = [os.path.join(destination_directory_path, file) for file in os.listdir(origin_directory_path) if file.startswith(split)]
    paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    destination_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                           
    # Initialize the progress bar
    progress_bar = tqdm(total=len(paths), desc=f'Processing files in {origin_directory_path}')
    print("Property value: ", property_value)

    for path, destination_directory_path in zip(paths, destination_paths):
        if isinstance(property_value, tuple) and len(property_value) == 2:
            # Get a random value between the two values
            new_saturation = adaptation_method(path, destination_directory_path, np.random.uniform(property_value[0], property_value[1]))
        else:
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

def adjust_brightness(image_path, destination_directory_path, brightness_factor):
    image = Image.open(image_path)
    brightened_image = F.adjust_brightness(image, brightness_factor)

    # Save the image
    brightened_image.save(destination_directory_path)

    return brightness_factor

def adjust_focus_with_multi_otsu_thresholding(image_path, destination_directory_path, _):
    rgb_image = np.array(Image.open(image_path))
    
    image_directory_path = os.path.dirname(image_path)
    dataset_directory_path = os.path.dirname(image_directory_path)
    depth_directory_path = os.path.join(dataset_directory_path, "Depth")
    if not os.path.exists(depth_directory_path):
        raise FileNotFoundError("The depth directory does not exist")
    
    depth_image = np.array(Image.open(os.path.join(depth_directory_path, os.path.basename(image_path))))
    
    num_thresholds = 3
    thresholds = threshold_multiotsu(depth_image, classes=num_thresholds)
    # Generate three classes from the original image
    regions = np.digitize(depth_image, bins=thresholds)

    most_important_region = 0
    second_most_important_region = 1
    third_most_important_region = 2

    rgb_cars_only = rgb_image.copy()
    rgb_cars_only[regions == most_important_region] = rgb_cars_only[regions == most_important_region] * 1.0
    rgb_cars_only[regions == second_most_important_region] = rgb_cars_only[regions == second_most_important_region] * 0.3
    rgb_cars_only[regions == third_most_important_region] = rgb_cars_only[regions == third_most_important_region] * 0.0

    # Save the image
    Image.fromarray(rgb_cars_only).save(destination_directory_path)


def adapt_property(origin_directory_path, destination_directory_path, property_value, property_name, split):
    if property_name == "saturation":
        adapt_dataset(origin_directory_path, destination_directory_path, property_value, adjust_saturation, split)
    if property_name == "hue":
        adapt_dataset(origin_directory_path, destination_directory_path, property_value, adjust_hue, split)
    if property_name == "brightness":
        adapt_dataset(origin_directory_path, destination_directory_path, property_value, adjust_brightness, split)
    if property_name == "focus":
        adapt_dataset(origin_directory_path, destination_directory_path, property_value, adjust_focus_with_multi_otsu_thresholding, split)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-op", "--origin_directory_path", type=str, default="datasets/SUNRGBD/RGB_original", help="The path to the SUNRGBD dataset")
    argparser.add_argument("-dp", "--destination_directory_path", type=str, default="datasets/SUNRGBD/RGB", help="The path to save the adapted dataset")
    argparser.add_argument("-s", "--split", type=str, default="test", help="The split to consider", choices=["train", "test"])
    argparser.add_argument("-pname", '--property_name', help='Property name', default='saturation')
    argparser.add_argument("-minpv", '--min_property_value', help='Minimum property value', default=0.0, type=float)
    argparser.add_argument("-maxpv", '--max_property_value', help='Maximum property value', default=1.0, type=float)

    args = argparser.parse_args()

    if args.min_property_value != args.max_property_value:
        property_value = (args.min_property_value, args.max_property_value)
    else:
        property_value = args.min_property_value

    adapt_property(args.origin_directory_path, args.destination_directory_path, property_value, args.property_name, args.split)
    