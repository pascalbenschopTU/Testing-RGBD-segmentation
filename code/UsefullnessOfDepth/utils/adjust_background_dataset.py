import os
import os.path as osp
import sys
import time
import argparse
import importlib
import cv2
import numpy as np
import random
from tqdm import tqdm
import re

sys.path.append('../UsefullnessOfDepth')

from utils.dataloader.dataloader import get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset


# Given a dataset path like "code/UsefullnessOfDepth/datasets/NYUDepthv2"
# And a set of classes that we would like to paste on another dataset
# Create crops from the classes in the dataset from RGB, Depth and Label folders -> from config
# Paste the crops on the background dataset
def get_large_components(mask, min_area=500):
    # Use morphological operations to remove small noise
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
    
    large_components = []

    # Find components larger than the minimum area
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            component_mask = np.zeros_like(mask, dtype=np.uint8)
            component_mask[labels == i] = 1
            large_components.append(component_mask)

    return large_components

def sort_key(file_name):
    # Extract the number from the filename
    number = re.findall(r'\d+', file_name)
    number = int(number[0]) if number else -1
    return number

def merge_datasets(config, classes, background_dataset_path, result_dataset_path, original=False):
    rgb_folder = os.path.basename(os.path.normpath(config.rgb_root_folder))
    x_folder = os.path.basename(os.path.normpath(config.x_root_folder))
    # gt_folder = os.path.basename(os.path.normpath(config.gt_root_folder))
    gt_folder = "labels"

    os.makedirs(osp.join(result_dataset_path, rgb_folder), exist_ok=True)
    os.makedirs(osp.join(result_dataset_path, x_folder), exist_ok=True)
    os.makedirs(osp.join(result_dataset_path, gt_folder), exist_ok=True)

    # Create train and test.txt files
    with open(osp.join(result_dataset_path, "train.txt"), "w") as f:
        f.write("")
    with open(osp.join(result_dataset_path, "test.txt"), "w") as f:
        f.write("")

    background_rgb_files = [f for f in os.listdir(osp.join(background_dataset_path, rgb_folder))]
    background_depth_files = [f for f in os.listdir(osp.join(background_dataset_path, x_folder))]
    background_label_files = [f for f in os.listdir(osp.join(background_dataset_path, gt_folder))]

    background_rgb_files = sorted(background_rgb_files, key=sort_key)
    background_depth_files = sorted(background_depth_files, key=sort_key)
    background_label_files = sorted(background_label_files, key=sort_key)

    val_loader, _ = get_val_loader(None, RGBXDataset, config, 1)

    for i, data in tqdm(enumerate(val_loader)):
        rgb = data['data'].numpy().squeeze().transpose(1, 2, 0)
        depth = data['modal_x'].numpy().squeeze()
        label = data['label'].numpy().squeeze()

        # Convert rgb and depth to uint8, but first normalize them back to 0, 1, and scale by 255
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255

        rgb = rgb.astype(np.uint8)
        depth = depth.astype(np.uint8)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Get the classes in the image
        classes_in_image = set(label.flatten().tolist())

        # Get the classes that we want to crop
        classes_of_interest = classes_in_image.intersection(classes)

        max_foreground_depth = 0
        mean_foreground_depth = 0
        class_in_image = False
        # Get the bounding boxes for the classes
        for foreground_class in classes_of_interest:
            mask = (label == foreground_class).astype(np.uint8)
            if np.count_nonzero(mask) < 500:
                continue

            large_components = get_large_components(mask)

            for component in large_components:
                # Get the bounding box for the component
                x, y, w, h = cv2.boundingRect(component)

                # Get the depth of the component
                component_depth = np.where(component == 1, depth, 0)
                component_depth = component_depth[component_depth > 0]

                if len(component_depth) == 0:
                    continue

                max_foreground_depth = max(max_foreground_depth, component_depth.max())
                mean_foreground_depth = max(mean_foreground_depth, component_depth.mean())
                class_in_image = True

        mask = depth > max_foreground_depth
        mask = mask.astype(np.uint8)

        background_rgb_file = background_rgb_files[i]
        background_depth_file = background_depth_files[i]
        background_label_file = background_label_files[i]

        if not class_in_image:
            new_rgb_file = osp.join(result_dataset_path, rgb_folder, background_rgb_file)
            new_depth_file = osp.join(result_dataset_path, x_folder, background_depth_file)
            new_label_file = osp.join(result_dataset_path, gt_folder, background_label_file)

            label[~np.isin(label, list(classes_of_interest))] = 0

            cv2.imwrite(new_rgb_file, rgb)
            cv2.imwrite(new_depth_file, depth)
            cv2.imwrite(new_label_file, label)

            with open(osp.join(result_dataset_path, "test.txt"), "a") as f:
                # RGB/test_0.png labels/test_0.png
                f.write(f"RGB/{background_rgb_file} labels/{background_label_file}\n")

            continue

        # Get the image
        background_rgb = cv2.imread(osp.join(background_dataset_path, rgb_folder, background_rgb_file), cv2.IMREAD_COLOR)
        background_depth = cv2.imread(osp.join(background_dataset_path, x_folder, background_depth_file), cv2.IMREAD_GRAYSCALE)
        background_label = cv2.imread(osp.join(background_dataset_path, gt_folder, background_label_file), cv2.IMREAD_GRAYSCALE)

        # Merge the image with the background based on the mask
        if original:
            new_rgb = np.array(rgb)
            new_rgb[mask == 0] = rgb[mask == 0]
            new_depth = np.array(depth)
            new_depth[mask == 0] = depth[mask == 0]
        else:
            new_rgb = np.array(background_rgb)
            new_rgb[mask == 0] = rgb[mask == 0] # Keep the original image where the depth is lower than the foreground
            # Scale background depth to (max_foreground_depth, 255) and keep the original depth where the foreground is
            min_depth = background_depth.min()
            max_depth = background_depth.max()
            new_depth = (background_depth - min_depth) / (max_depth - min_depth) * (255 - max_foreground_depth) + max_foreground_depth
            new_depth = np.array(new_depth)
            new_depth[mask == 0] = depth[mask == 0]
        new_label = np.zeros_like(background_label)
        new_label[mask == 0] = label[mask == 0]
        # Remove all classes in the label that are not in the classes of interest
        new_label[~np.isin(new_label, list(classes_of_interest))] = 0

        # Save the new image, depth and label to the result dataset
        new_rgb_file = osp.join(result_dataset_path, rgb_folder, background_rgb_file)
        new_depth_file = osp.join(result_dataset_path, x_folder, background_depth_file)
        new_label_file = osp.join(result_dataset_path, gt_folder, background_label_file)

        cv2.imwrite(new_rgb_file, new_rgb)
        cv2.imwrite(new_depth_file, new_depth)
        cv2.imwrite(new_label_file, new_label)

        with open(osp.join(result_dataset_path, "test.txt"), "a") as f:
            # RGB/test_0.png labels/test_0.png
            f.write(f"RGB/{background_rgb_file} labels/{background_label_file}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-bgdp", "--background_dataset_path", type=str, help="Path to the dataset where the background images are stored")
    parser.add_argument("-redp", "--result_dataset_path", type=str, help="Path to the dataset where the crops will be pasted")
    parser.add_argument("-c", "--config", type=str, help="Config file from original dataset")
    parser.add_argument("-cl", "--classes", nargs="+", type=str, help="Classes to crop")
    parser.add_argument("-o", "--original", action="store_true", help="Use the original image instead of the cropped image")
    args = parser.parse_args()

    classes = args.classes
    # Parse the classes to integers
    classes = [int(c) for c in args.classes]

    config_location = args.config
    if ".py" in config_location:
        config_location = config_location.replace(".py", "")
        config_location = config_location.replace("\\", ".")
        while config_location.startswith("."):
            config_location = config_location[1:]

    # Load the config
    config_module = importlib.import_module(config_location)
    config = config_module.config

    if classes[0] == -1:
        classes = list(range(config.num_classes))

    merge_datasets(config, classes, args.background_dataset_path, args.result_dataset_path, original=args.original)


if __name__ == "__main__":
    main()

            
