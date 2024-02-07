import argparse
import os
from PIL import Image
import numpy as np
import cv2


def change_dataset(dataset_location, src_dir='Depth_original', dst_dir='Depth'):
    # Define the source and destination directories
    src_dir = os.path.join(dataset_location, src_dir)
    dst_dir = os.path.join(dataset_location, dst_dir)

    image_dir = os.path.join(dataset_location, 'RGB')

    # Create the destination directory if it doesn't exist
    os.makedirs(dst_dir, exist_ok=True)

    # Iterate over the PNG files in the source directory
    for filename in os.listdir(src_dir):
        if filename.endswith('.png'):
            # Create a new black image with the same dimensions and mode as the source image
            depth_image = Image.open(os.path.join(src_dir, filename))
            src_image = Image.open(os.path.join(image_dir, filename))
            # black_image = Image.new(src_image.mode, src_image.size)

            # # Save the black image to the destination directory with the same name as the source image
            # black_image.save(os.path.join(dst_dir, filename))

            noisy_depth = add_depth_noise(np.array(depth_image), np.array(src_image))
            noisy_depth_image = Image.fromarray(noisy_depth.astype(np.uint8))
            noisy_depth_image.save(os.path.join(dst_dir, filename))

            # # Create a grayscale image from the image in the img_dir directory and save it to the destination directory
            # img = Image.open(os.path.join(img_dir, filename))
            # img = img.convert('L')

            # # Save the grayscale image to the destination directory with the same name as the source image
            # img.save(os.path.join(dst_dir, filename.replace('.jpg', '.png')))

            # # Create a grayscale image from the image in the img_dir directory and save it to the destination directory
            # img = Image.open(os.path.join(img_dir, filename))
            # img = img.convert('L')

            # # Save the grayscale image to the destination directory with the same name as the source image
            # img.save(os.path.join(dst_dir, filename.replace('.jpg', '.png')))


def add_depth_noise(depth_map, rgb_image, noise_level=0.2):
    """
    Simulate noise in the depth map based on object properties, scene layout, motion blur, and scene illumination.

    Parameters:
        depth_map (numpy.ndarray): Perfect depth map.
        rgb_image (numpy.ndarray): RGB image.
        noise_level (float): Noise level.

    Returns:
        numpy.ndarray: Noisy depth map.
    """
    # Initialize noisy depth map
    noisy_depth_map = depth_map.copy().astype(np.float64)

    # Simulate edges where the depth is not well defined, i.e. black
    # Compute gradients using Sobel operator
    grad_x = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
    # Compute gradient magnitude and threshold for edge detection
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    edge_mask = (grad_mag < 50).astype(np.uint8)  # Adjust threshold as needed
    # Invert the edge mask
    inverted_edges = 1.0 - edge_mask
    random_edges = np.zeros_like(inverted_edges)
    # Give the edges a chance to disappear
    random_edges[inverted_edges > 0] = np.random.choice([0, 1], size=inverted_edges.shape, p=[1.0-noise_level, noise_level])[inverted_edges > 0]
    # Make the edges wider
    kernel_size = (5, 5)
    kernel = np.ones(kernel_size, np.uint8)
    dilated_edges = cv2.dilate(random_edges.astype(np.uint8), kernel, iterations=1)
    # Convert depth_map to float64
    depth_map = depth_map.astype(np.float64)
    # Subtract the edge mask from the depth map
    noisy_depth_map[dilated_edges > 0] = dilated_edges[dilated_edges > 0]

    # Simulate noise in bright regions of the RGB image
    # Calculate brightness of the RGB image
    brightness = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    bright_regions = brightness > 200  # Define threshold for brightness
    bright_noise = noise_level * np.random.normal(128, 50, depth_map.shape)
    noisy_depth_map += bright_noise * bright_regions

    # Simulate freckles
    # Where noise looks like random blobs of black
    noisy_region_mask = (depth_map > 50) & (brightness < 100)
    # Create a random noise mask with the same shape as the depth map
    noise_mask = np.random.rand(*depth_map.shape) < 0.005
    # Combine the noisy region mask and the noise mask
    noisy_regions = noisy_region_mask & noise_mask
    dilated_regions = cv2.dilate(noisy_regions.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    noisy_depth_map[dilated_regions == 1] = dilated_regions[dilated_regions == 1]

    # Clip the noisy depth map to the range [0, 255]
    noisy_depth_map = np.clip(noisy_depth_map, 0, 255)

    # Blur the noisy depth map neurest neighbors
    noisy_depth_map = cv2.medianBlur(noisy_depth_map.astype(np.uint8), 3)

    return noisy_depth_map
        

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Change the dataset')
    arg_parser.add_argument('--src', type=str, help='Path to the source directory of the dataset')
    arg_parser.add_argument('--src_dir', type=str, help='Path to the source directory of the dataset', default='Depth_original')
    arg_parser.add_argument('--dst_dir', type=str, help='Path to the destination directory of the dataset', default='Depth')
    args = arg_parser.parse_args()

    change_dataset(args.src, args.src_dir, args.dst_dir)