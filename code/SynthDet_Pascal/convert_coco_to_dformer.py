import argparse
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from CocoDepthDataset import CocoDepthDataset
from PIL import Image

def convert_dataset(dataset_root, save_location, image_size=(480, 480), dataset_split=(0.7, 0.3)):

    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=Image.NEAREST),  # Adjust the size based on your model's input size
        transforms.ToTensor(),
    ])

    dataset = CocoDepthDataset(os.path.join(dataset_root, 'images'), os.path.join(dataset_root, 'semantic.json'), os.path.join(dataset_root, 'depth'), transform=transform)

    # Split dataset into training and validation sets
    train_size = int(len(dataset) * dataset_split[0])
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)

    # Create directories to save the converted dataset
    os.makedirs(os.path.join(save_location, 'RGB'), exist_ok=True)
    os.makedirs(os.path.join(save_location, 'Depth'), exist_ok=True)
    os.makedirs(os.path.join(save_location, 'labels'), exist_ok=True)

    # Create train.txt and test.txt files
    with open(os.path.join(save_location, 'train.txt'), 'w') as train_file:
        pass

    with open(os.path.join(save_location, 'test.txt'), 'w') as test_file:
        pass

    # Iterate through the dataset and save to another location
    for batch_idx, (rgb_data, depth_data, label) in enumerate(train_loader):
        # Process the data here
        # Save the processed data to the save_location
        # Save depth to  Depth/train_{batch_idx}.png
        depth_data = depth_data.squeeze(1)  # Remove the channel dimension
        # Take the first item, as the batch size is 1
        depth_image = depth_data[0].numpy()
        depth_image = (depth_image * 255).astype('uint8')  # Convert depth image to 0-255 range
        depth_image = Image.fromarray(depth_image)
        depth_image.save(os.path.join(save_location, f"Depth/train_{batch_idx}.png"))

        # Save RGB to  and RGB/train_{batch_idx}.png
        rgb_image = rgb_data[0].numpy()
        rgb_image = (rgb_image * 255).astype('uint8')
        rgb_image = Image.fromarray(rgb_image.transpose(1, 2, 0))
        rgb_image.save(os.path.join(save_location, f"RGB/train_{batch_idx}.png"))

        # Save the label to labels/train_{batch_idx}.png
        label = label.squeeze(1)  # Remove the channel dimension
        label_image = label[0].numpy()
        label_image = label_image.astype('uint8')
        label_image = Image.fromarray(label_image)
        label_image.save(os.path.join(save_location, f"labels/train_{batch_idx}.png"))

        # add a line to train.txt with the path to the RGB image and the path to the depth image
        # Example: RGB/train_0.jpg labels/train_0.png
        with open(os.path.join(save_location, 'train.txt'), 'a') as train_file:
            train_file.write(f"RGB/train_{batch_idx}.png labels/train_{batch_idx}.png\n")


    for batch_idx, (rgb_data, depth_data, label) in enumerate(test_loader):
        # Process the data here
        # Save the processed data to the save_location
        # Save depth to  Depth/test_{batch_idx}.png
        depth_data = depth_data.squeeze(1)
        depth_image = depth_data[0].numpy()
        depth_image = (depth_image * 255).astype('uint8')
        depth_image = Image.fromarray(depth_image)
        depth_image.save(os.path.join(save_location, f"Depth/test_{batch_idx}.png"))

        # Save RGB to RGB/test_{batch_idx}.png
        rgb_image = rgb_data[0].numpy()
        rgb_image = (rgb_image * 255).astype('uint8')
        rgb_image = Image.fromarray(rgb_image.transpose(1, 2, 0))
        rgb_image.save(os.path.join(save_location, f"RGB/test_{batch_idx}.png"))

        # Save the label to labels/test_{batch_idx}.png
        label = label.squeeze(1)
        label_image = label[0].numpy()
        label_image = label_image.astype('uint8')
        label_image = Image.fromarray(label_image)
        label_image.save(os.path.join(save_location, f"labels/test_{batch_idx}.png"))

        # add a line to test.txt with the path to the RGB image and the path to the depth image
        # Example: RGB/test_0.jpg labels/test_0.png
        with open(os.path.join(save_location, 'test.txt'), 'a') as test_file:
            test_file.write(f"RGB/test_{batch_idx}.png labels/test_{batch_idx}.png\n")


def main():
    parser = argparse.ArgumentParser(description='Convert COCO dataset to DFormer dataset')
    parser.add_argument('dataset_root', help='Path to COCO dataset')
    parser.add_argument('save_location', help='Path to save the converted dataset')
    parser.add_argument('--dataset_split', nargs=2, type=float, default=(0.7, 0.3), help='Train and test split')
    args = parser.parse_args()

    convert_dataset(args.dataset_root, args.save_location)


if __name__ == '__main__':
    main()