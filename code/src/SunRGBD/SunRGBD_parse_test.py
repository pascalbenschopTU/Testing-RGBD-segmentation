import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os

DIR_ = '../../data/SunRGBD/kinect2data/'

data_sample = DIR_ + '000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize/'

def load_image_depth_segmentation(data_directory):
    data_file_number = '0' + data_directory.split('rgbf')[1].split('-')[0]

    # Load the image, depth map, and segmentation mask
    image = cv2.imread(data_directory + 'image/' + data_file_number + '.jpg')
    depth = cv2.imread(data_directory + 'depth_bfx/' + data_file_number + '.png', cv2.IMREAD_UNCHANGED)
    fixed_depth = fix_depth(depth)

    # Load the segmentation mask
    seg_mat = scipy.io.loadmat(data_directory + 'seg.mat')
    seg_mask = seg_mat['seglabel']
    seg_labels = seg_mat['names'][0]

    return image, fixed_depth, seg_mask, seg_labels

def fix_depth(depth):
    # Convert depth to numpy array
    depth = np.array(depth, dtype=np.float32)

    # Get a mask for the region of artifacts
    artifacts_region = (depth == depth.min()).astype(np.uint8)

    # Inpaint holes in the segmentation mask
    inpainted_depth = cv2.inpaint(depth, artifacts_region, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    return inpainted_depth

# # Load the image, depth map, and segmentation mask
# image, depth, seg_mask, seg_labels = load_image_depth_segmentation(data_sample)
# print(seg_labels)


# # Plot the image, depth map, and segmentation mask
# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# axs[0].set_title('Image')
# axs[1].imshow(depth, cmap='gray')
# axs[1].set_title('Depth Map')
# axs[2].imshow(seg_mask, cmap='gray')
# axs[2].set_title('Segmentation Mask')
# plt.show()


for directory in os.listdir(DIR_):
    if os.path.isdir(os.path.join(DIR_, directory)):
        data_sample = os.path.join(DIR_, directory) + "/"
        image, depth, seg_mask, seg_labels = load_image_depth_segmentation(data_sample)

        # Plot the image, depth map, fixed depth map, and segmentation mask
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[0].set_title('Image')
        axs[1].imshow(depth, cmap='gray')
        axs[1].set_title('Depth Map')
        axs[2].imshow(seg_mask, cmap='gray')
        axs[2].set_title('Segmentation Mask')
        plt.show()