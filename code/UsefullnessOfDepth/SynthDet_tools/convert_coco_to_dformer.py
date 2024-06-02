import argparse
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from CocoDepthDataset import CocoDepthDataset
from PIL import Image
import multiprocessing
from tqdm import tqdm

class AdaptiveDatasetCreator:
    def __init__(self, args, image_size=(400, 400), dataset_split=(0.5, 0.5)):
        self.dataset_root = args.dataset_root
        self.save_location = args.save_location
        self.image_size = image_size
        if hasattr(args, 'train_split'):
            # self.dataset_split = args.dataset_split
            self.dataset_split = (args.train_split, 1 - args.train_split)
        else:
            self.dataset_split = dataset_split

        self.dataset_gems = args.dataset_type == 'gems' or args.dataset_type == 'gems_cars'
        self.dataset_cars = args.dataset_type == 'cars'
        self.dataset_depth_tests = args.depth_tests

        # Create the save location if it does not exist
        if not os.path.exists(args.save_location):
            os.makedirs(args.save_location, exist_ok=True)

        # Create train.txt and test.txt files
        with open(os.path.join(args.save_location, 'train.txt'), 'w'):
            pass

        with open(os.path.join(args.save_location, 'test.txt'), 'w'):
            pass

    def convert_and_save_RGB(self, rgb_data, file_name):
        rgb_image = rgb_data.numpy()
        rgb_image = (rgb_image * 255).astype('uint8')
        rgb_image = Image.fromarray(rgb_image.transpose(1, 2, 0))
        if not os.path.exists(os.path.join(self.save_location, f"RGB")):
            os.makedirs(os.path.join(self.save_location, f"RGB"), exist_ok=True)
        rgb_image.save(os.path.join(self.save_location, f"RGB/{file_name}.png"))
    
    def convert_and_save_label(self, label, file_name):
        label = label.squeeze(0)
        label_image = label.numpy()
        label_image = label_image.astype('uint8')
        if self.dataset_gems:
            label_image[label_image != 0] = label_image[label_image != 0] - 63
        if self.dataset_cars:
            label_image[label_image != 0] = label_image[label_image != 0] - 79
        label_image = Image.fromarray(label_image)
        if not os.path.exists(os.path.join(self.save_location, f"labels")):
            os.makedirs(os.path.join(self.save_location, f"labels"), exist_ok=True)
        label_image.save(os.path.join(self.save_location, f"labels/{file_name}.png"))

    def convert_and_save_depth(self, depth_data, file_name):
        depth_data = depth_data.squeeze(0)
        depth_data = depth_data.numpy()
        depth_data = (depth_data * 255).astype('uint8')
        depth_image = Image.fromarray(depth_data)
        if not os.path.exists(os.path.join(self.save_location, f"Depth")):
            os.makedirs(os.path.join(self.save_location, f"Depth"), exist_ok=True)
        depth_image.save(os.path.join(self.save_location, f"Depth/{file_name}.png"))


    def convert_and_save_grayscale(self, rgb_data, file_name):
        rgb_image = rgb_data.numpy()
        rgb_image = (rgb_image * 255).astype('uint8')
        rgb_image = Image.fromarray(rgb_image.transpose(1, 2, 0))
        rgb_image = rgb_image.convert('L')
        if not os.path.exists(os.path.join(self.save_location, f"Grayscale")):
            os.makedirs(os.path.join(self.save_location, f"Grayscale"), exist_ok=True)
        rgb_image.save(os.path.join(self.save_location, f"Grayscale/{file_name}.png"))

    def convert_and_save_dataset(self):
        # Define data transformations
        transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=InterpolationMode.NEAREST),  # Adjust the size based on your model's input size
            transforms.ToTensor(),
        ])

        dataset = CocoDepthDataset(os.path.join(self.dataset_root, 'images'), os.path.join(self.dataset_root, 'semantic.json'), os.path.join(self.dataset_root, 'depth'), transform=transform)
        
        train_size = int(len(dataset) * self.dataset_split[0])
        test_size = len(dataset) - train_size
        train_dataset = torch.utils.data.Subset(dataset, list(range(train_size)))
        test_dataset = torch.utils.data.Subset(dataset, list(range(train_size, train_size + test_size)))

        self.process_dataset(train_dataset, dataset_split="train")
        self.process_dataset(test_dataset, dataset_split="test")


    def process_dataset(self, dataset, dataset_split="train", single_process=True):
        progress_bar = tqdm(total=len(dataset), desc='Processing')

        if single_process:
            pool = multiprocessing.Pool(processes=1)
        else:
            pool = multiprocessing.Pool()

        for idx, (rgb_data, depth_data, label) in enumerate(dataset):
            pool.apply(self.process_sequence, args=(idx, rgb_data, depth_data, label, dataset_split))
            progress_bar.update(1)
            # pool.apply_async(self.process_sequence, args=(idx, rgb_data, depth_data, label, dataset_split), callback=lambda _: progress_bar.update(1))


        pool.close()
        pool.join()

        progress_bar.close()

    def process_sequence(self, idx, rgb_data, depth_data, label, dataset_split):
        self.convert_and_save_RGB(rgb_data, f"{dataset_split}_{idx}")
        self.convert_and_save_label(label, f"{dataset_split}_{idx}")
        self.convert_and_save_depth(depth_data, f"{dataset_split}_{idx}")

        with open(os.path.join(self.save_location, f'{dataset_split}.txt'), 'a') as file:
            file.write(f"RGB/{dataset_split}_{idx}.png labels/{dataset_split}_{idx}.png\n")
        

def main():
    parser = argparse.ArgumentParser(description='Convert COCO dataset to DFormer dataset')
    parser.add_argument('dataset_root', help='Path to COCO dataset')
    parser.add_argument('save_location', help='Path to save the converted dataset')
    parser.add_argument('dataset_type', default="groceries", help='Type of dataset to convert')
    parser.add_argument('--depth_tests', action='store_true', help='Add extra depth test datasets')
    parser.add_argument('--train_split', type=float, default=0.5, help='Train and test split')
    args = parser.parse_args()
    
    dataset_creator = AdaptiveDatasetCreator(args, image_size=(400, 400))
    dataset_creator.convert_and_save_dataset()


if __name__ == '__main__':
    main()