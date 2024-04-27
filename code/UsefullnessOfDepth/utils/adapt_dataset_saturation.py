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

def get_saturation(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Compute the saturation
    saturation = np.mean(hsv[:, :, 1])

    return saturation

def plot_saturation(dataset_path, split='test'):
    paths = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if file.startswith(split)]
    paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    saturation_values = [get_saturation(path) for path in paths]


    # plt.hist(saturation_values, bins=50)
    plt.plot(saturation_values, 'o', markersize=1, color='green', label='Saturation values')
    plt.xlabel('Saturation')
    plt.ylabel('Frequency')
    plt.title('Saturation distribution of SUNRGBD dataset')
    plt.legend()
    plt.show()

def adapt_dataset(dataset_path, destination_path, min_saturation, max_saturation, split='test'):
    paths = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if file.startswith(split)]
    destination_paths = [os.path.join(destination_path, file) for file in os.listdir(dataset_path) if file.startswith(split)]
    paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    destination_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Initialize the progress bar
    progress_bar = tqdm(total=len(paths), desc=f'Processing files in {dataset_path}')
    print("Min saturation: ", min_saturation, " Max saturation: ", max_saturation)
    # Process each image in parallel
    saturation_range = np.linspace(min_saturation, max_saturation, len(paths))
    # saturation_range = np.logspace(np.log10(min_saturation), np.log10(max_saturation), len(paths))

    new_saturation_values = []
   
    for path, destination_path, saturation_value in zip(paths, destination_paths, saturation_range):
        new_saturation = adjust_saturation(path, destination_path, saturation_value)
        new_saturation_values.append(new_saturation)
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

def adapt_dataset_single_value(dataset_path, destination_path, saturation_value, split='test'):
    paths = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if file.startswith(split)]
    destination_paths = [os.path.join(destination_path, file) for file in os.listdir(dataset_path) if file.startswith(split)]
    paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    destination_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                           
    # Initialize the progress bar
    progress_bar = tqdm(total=len(paths), desc=f'Processing files in {dataset_path}')
    print("Saturation value: ", saturation_value)
    # Process each image in parallel
    new_saturation_values = []

    for path, destination_path in zip(paths, destination_paths):
        new_saturation = adjust_saturation(path, destination_path, saturation_value)
        new_saturation_values.append(new_saturation)
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()


def adjust_saturation(image_path, destination_path, saturation_value):
    image = Image.open(image_path)
    saturated_image = F.adjust_saturation(image, saturation_value)

    # Save the image
    saturated_image.save(destination_path)

    return saturation_value


def test_saturation_shift(argparser, saturation_values, dataset_path, destination_path):
    for saturation in saturation_values:
        adapt_dataset_single_value(dataset_path, destination_path, saturation)
        args = argparser.parse_args()
        config_module = importlib.import_module(args.config)
        config = config_module.config

        if config.x_e_channels == 3:
            dataset_dir = os.path.dirname(os.path.dirname(dataset_path))
            depth_dir = os.path.join(dataset_dir, 'Depth')
            if not os.path.exists(depth_dir):
                os.makedirs(depth_dir)

            # Check for the first file if it single channel
            if len(os.listdir(depth_dir)) > 0:
                depth_image = Image.open(os.path.join(depth_dir, os.listdir(depth_dir)[0]))
                if len(np.array(depth_image).shape) == 2:
                    # Move Depth folder to Depth_original
                    os.rename(depth_dir, os.path.join(dataset_dir, 'Depth_original'))

            for file in os.listdir(destination_path):
                if file.startswith(args.split):
                    shutil.copy(os.path.join(destination_path, file), os.path.join(depth_dir, file))

        get_scores_for_model(argparser, "saturation_tests.txt")
    


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/SUNRGBD",
        help="The path to the SUNRGBD dataset",
    )
    argparser.add_argument(
        "--destination_path",
        type=str,
        default="datasets/SUNRGBD",
        help="The path to save the adapted dataset",
    )
    argparser.add_argument(
        "--min_saturation",
        type=float,
        default=0.1,
        help="The minimum saturation value to consider",
    )
    argparser.add_argument(
        "--max_saturation",
        type=float,
        default=3.0,
        help="The maximum saturation value to consider",
    )
    argparser.add_argument(
        "--split",
        type=str,
        default="test",
        help="The split to consider",
    )
    argparser.add_argument(
        "--analyze",
        default=False,
        action="store_true",
        help="Analyze the saturation distribution of the dataset",
    )
    argparser.add_argument('--config', help='train config file path')
    argparser.add_argument('--model_weights', help='File of model weights')
    argparser.add_argument('--model', help='Model name', default='DFormer-Tiny')
    argparser.add_argument('--bin_size', help='Bin size for testing', default=1, type=int)
    argparser.add_argument('--dataset', help='Dataset dir')

    args = argparser.parse_args()

    if args.config is not None:
        # saturation_values = np.linspace(args.min_saturation, args.max_saturation, 10)
        saturation_values = np.logspace(np.log10(args.min_saturation), np.log10(args.max_saturation), 20)
        test_saturation_shift(argparser, saturation_values, args.dataset_path, args.destination_path)
        exit()

    if args.analyze:
        plot_saturation(args.dataset_path)
        exit()

    adapt_dataset(args.dataset_path, args.destination_path, args.min_saturation, args.max_saturation, split=args.split)