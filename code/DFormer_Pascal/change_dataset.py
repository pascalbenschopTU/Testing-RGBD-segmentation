import argparse
import os
from PIL import Image
import numpy as np
import cv2
from multiprocessing import Pool

from create_noise import create_noisy_depth, create_depth_noise_2


def process_depth_kinect_noise(filename, src_dir, dst_dir, image_dir):
    if filename.endswith('.png'):
        noisy_depth = create_noisy_depth(os.path.join(src_dir, filename))
        noisy_depth = np.stack([noisy_depth] * 3, axis=-1)
        noisy_depth = Image.fromarray(noisy_depth.astype(np.uint8))
        noisy_depth.save(os.path.join(dst_dir, filename))

def process_depth_black(filename, src_dir, dst_dir, image_dir):
    if filename.endswith('.png'):
        original_depth = cv2.imread(os.path.join(src_dir, filename), cv2.IMREAD_UNCHANGED)
        original_depth = np.array(original_depth)
        depth_black = np.zeros((original_depth.shape[0], original_depth.shape[1], 3), dtype=np.uint8)  # Convert to np.uint8
        depth_black = Image.fromarray(depth_black)
        depth_black.save(os.path.join(dst_dir, filename))


def change_dataset(dataset_location, src_dir='Depth_original', dst_dir='Depth', type='black'):
    src_dir = os.path.join(dataset_location, src_dir)
    dst_dir = os.path.join(dataset_location, dst_dir)
    image_dir = os.path.join(dataset_location, 'RGB')
    os.makedirs(dst_dir, exist_ok=True)

    filenames = [filename for filename in os.listdir(src_dir) if filename.endswith('.png')]

    # Create a pool of worker processes
    pool = Pool()

    # Process the images in parallel
    if type == 'kinect':
        pool.starmap(process_depth_kinect_noise, [(filename, src_dir, dst_dir, image_dir) for filename in filenames])
    elif type == 'black':
        pool.starmap(process_depth_black, [(filename, src_dir, dst_dir, image_dir) for filename in filenames])

    # Close the pool of worker processes
    pool.close()
    pool.join()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Change the dataset')
    arg_parser.add_argument('--src', type=str, help='Path to the source directory of the dataset')
    arg_parser.add_argument('--src_dir', type=str, help='Path to the source directory of the dataset', default='Depth_original')
    arg_parser.add_argument('--dst_dir', type=str, help='Path to the destination directory of the dataset', default='Depth')
    arg_parser.add_argument('--type', type=str, help='Apply transformation to depth: black or kinect', default='black')
    args = arg_parser.parse_args()

    change_dataset(args.src, args.src_dir, args.dst_dir, args.type)