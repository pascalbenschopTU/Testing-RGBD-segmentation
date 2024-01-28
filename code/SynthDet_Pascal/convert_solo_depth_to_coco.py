import argparse
import os
import cv2
import OpenEXR
import Imath
import numpy as np
from tqdm import tqdm

def extract_depth_from_exr(exr_path):
    # Read the EXR file
    exr_file = OpenEXR.InputFile(exr_path)

    # Get the data window (region of interest)
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Read the R channel directly as a float, this channel contains the depth
    # From the camera in unity meters
    redstr = exr_file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
    depth_image = np.frombuffer(redstr, dtype=np.float32)
    
    # Reshape the flattened array to the original image shape
    depth_image = depth_image.reshape((height, width, 1))

    # Get min and max depth of all objects in scene
    min_depth = np.min(depth_image[depth_image > 0])
    max_depth = np.max(depth_image[depth_image > 0])

     # Map the depth values to a suitable visualization range
    normalized_depth = np.clip((depth_image - min_depth) / (max_depth - min_depth), 0.0, 1.0)

    return normalized_depth

def save_depth_image(depth_image, output_path):
    # Convert depth image to 0-255 range
    depth_image = (depth_image * 255).astype(np.uint8)  # Convert depth image to 0-255 range
    cv2.imwrite(output_path, depth_image)  # Save only one channel for simplicity

def process_solo_dataset(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    directory_list = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]

    # Initialize the progress bar
    progress_bar = tqdm(total=len(directory_list), desc='Processing')

    for sequence_folder in directory_list:
        sequence_path = os.path.join(input_folder, sequence_folder)

        # Find the current iteration, which is the last digit in the sequence folder name
        sequence_digit = int(sequence_folder.split("sequence.")[-1])

        # Find the depth EXR file
        exr_file_path = os.path.join(sequence_path, f"step0.camera.Depth.exr")

        if os.path.isfile(exr_file_path):
            # Extract depth from EXR file
            depth_image = extract_depth_from_exr(exr_file_path)

            # Save depth image as PNG in the specified output folder
            output_path = os.path.join(output_folder, f"depth_{sequence_digit}.png")
            save_depth_image(depth_image, output_path)

        # Update the progress bar
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()


def main():
    parser = argparse.ArgumentParser(description='Process Unity SOLO dataset and extract depth images.')
    parser.add_argument('input_folder', help='Path to the SOLO dataset folder')
    parser.add_argument('output_folder', help='Path to the output folder for depth images')
    args = parser.parse_args()

    process_solo_dataset(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()