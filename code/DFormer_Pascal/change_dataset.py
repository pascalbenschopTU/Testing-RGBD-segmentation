import os
from PIL import Image

# Define the source and destination directories
img_dir = "datasets/SUNRGBD/RGB"
src_dir = "datasets/SUNRGBD/Depth_original"
dst_dir = "datasets/SUNRGBD/Depth"

# Create the destination directory if it doesn't exist
os.makedirs(dst_dir, exist_ok=True)

# Iterate over the PNG files in the source directory
for filename in os.listdir(img_dir):
    if filename.endswith('.jpg'):
        # # Create a new black image with the same dimensions and mode as the source image
        # src_image = Image.open(os.path.join(src_dir, filename))
        # black_image = Image.new(src_image.mode, src_image.size)

        # # Save the black image to the destination directory with the same name as the source image
        # black_image.save(os.path.join(dst_dir, filename))

        # Create a grayscale image from the image in the img_dir directory and save it to the destination directory
        img = Image.open(os.path.join(img_dir, filename))
        img = img.convert('L')

        # Save the grayscale image to the destination directory with the same name as the source image
        img.save(os.path.join(dst_dir, filename.replace('.jpg', '.png')))
        