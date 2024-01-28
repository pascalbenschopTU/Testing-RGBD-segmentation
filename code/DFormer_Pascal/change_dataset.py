import argparse
import os
from PIL import Image


def change_dataset(dataset_location, src_dir='Depth_original', dst_dir='Depth'):
    # Define the source and destination directories
    src_dir = os.path.join(dataset_location, src_dir)
    dst_dir = os.path.join(dataset_location, dst_dir)

    # Create the destination directory if it doesn't exist
    os.makedirs(dst_dir, exist_ok=True)

    # Iterate over the PNG files in the source directory
    for filename in os.listdir(src_dir):
        if filename.endswith('.png'):
            # Create a new black image with the same dimensions and mode as the source image
            src_image = Image.open(os.path.join(src_dir, filename))
            black_image = Image.new(src_image.mode, src_image.size)

            # Save the black image to the destination directory with the same name as the source image
            black_image.save(os.path.join(dst_dir, filename))

            # # Create a grayscale image from the image in the img_dir directory and save it to the destination directory
            # img = Image.open(os.path.join(img_dir, filename))
            # img = img.convert('L')

            # # Save the grayscale image to the destination directory with the same name as the source image
            # img.save(os.path.join(dst_dir, filename.replace('.jpg', '.png')))
        

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Change the dataset')
    arg_parser.add_argument('--src', type=str, help='Path to the source directory of the dataset')
    arg_parser.add_argument('--src_dir', type=str, help='Path to the source directory of the dataset', default='Depth_original')
    arg_parser.add_argument('--dst_dir', type=str, help='Path to the destination directory of the dataset', default='Depth')
    args = arg_parser.parse_args()

    change_dataset(args.src, args.src_dir, args.dst_dir)