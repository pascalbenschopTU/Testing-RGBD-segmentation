import torchvision.transforms.functional as F
from PIL import Image
import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
import sys
import re
sys.path.append('../UsefullnessOfDepth')


# This class is used to add noise to the depth images
# Taken from https://github.com/ankurhanda/simkinect/tree/master 
class KinectNoise:
    def add_gaussian_shifts(self, depth, std=1/2.0):
        rows, cols = depth.shape 
        gaussian_shifts = np.random.normal(0, std, size=(rows, cols, 2))
        gaussian_shifts = gaussian_shifts.astype(np.float32)

        # creating evenly spaced coordinates  
        xx = np.linspace(0, cols-1, cols)
        yy = np.linspace(0, rows-1, rows)

        # get xpixels and ypixels 
        xp, yp = np.meshgrid(xx, yy)

        xp = xp.astype(np.float32)
        yp = yp.astype(np.float32)

        xp_interp = np.minimum(np.maximum(xp + gaussian_shifts[:, :, 0], 0.0), cols)
        yp_interp = np.minimum(np.maximum(yp + gaussian_shifts[:, :, 1], 0.0), rows)

        depth_interp = cv2.remap(depth, xp_interp, yp_interp, cv2.INTER_LINEAR)

        return depth_interp
        

    def filterDisp(self, disp, dot_pattern_, invalid_disp_):

        size_filt_ = 9

        xx = np.linspace(0, size_filt_-1, size_filt_)
        yy = np.linspace(0, size_filt_-1, size_filt_)

        xf, yf = np.meshgrid(xx, yy)

        xf = xf - int(size_filt_ / 2.0)
        yf = yf - int(size_filt_ / 2.0)

        sqr_radius = (xf**2 + yf**2)
        vals = sqr_radius * 1.2**2 

        vals[vals==0] = 1 
        weights_ = 1 /vals  

        fill_weights = 1 / ( 1 + sqr_radius)
        fill_weights[sqr_radius > 9] = -1.0 

        disp_rows, disp_cols = disp.shape 
        dot_pattern_rows, dot_pattern_cols = dot_pattern_.shape

        lim_rows = np.minimum(disp_rows - size_filt_, dot_pattern_rows - size_filt_)
        lim_cols = np.minimum(disp_cols - size_filt_, dot_pattern_cols - size_filt_)

        center = int(size_filt_ / 2.0)

        window_inlier_distance_ = 0.1

        out_disp = np.ones_like(disp) * invalid_disp_

        interpolation_map = np.zeros_like(disp)

        for r in range(0, lim_rows):

            for c in range(0, lim_cols):

                if dot_pattern_[r+center, c+center] > 0:
                                    
                    # c and r are the top left corner 
                    window  = disp[r:r+size_filt_, c:c+size_filt_] 
                    dot_win = dot_pattern_[r:r+size_filt_, c:c+size_filt_] 
    
                    valid_dots = dot_win[window < invalid_disp_]

                    n_valids = np.sum(valid_dots) / 255.0 
                    n_thresh = np.sum(dot_win) / 255.0 

                    if n_valids > n_thresh / 1.2: 

                        mean = np.mean(window[window < invalid_disp_])

                        diffs = np.abs(window - mean)
                        diffs = np.multiply(diffs, weights_)

                        cur_valid_dots = np.multiply(np.where(window<invalid_disp_, dot_win, 0), 
                                                    np.where(diffs < window_inlier_distance_, 1, 0))

                        n_valids = np.sum(cur_valid_dots) / 255.0

                        if n_valids > n_thresh / 1.2: 
                        
                            accu = window[center, center] 

                            assert(accu < invalid_disp_)

                            out_disp[r+center, c + center] = round((accu)*8.0) / 8.0

                            interpolation_window = interpolation_map[r:r+size_filt_, c:c+size_filt_]
                            disp_data_window     = out_disp[r:r+size_filt_, c:c+size_filt_]

                            substitutes = np.where(interpolation_window < fill_weights, 1, 0)
                            interpolation_window[substitutes==1] = fill_weights[substitutes ==1 ]

                            disp_data_window[substitutes==1] = out_disp[r+center, c+center]

        return out_disp


    def create_noisy_depth(self, input_depth):
        # reading the image directly in gray with 0 as input 
        dot_pattern_ = Image.open("utils/data/kinect-pattern_3x3.png")
        dot_pattern_ = dot_pattern_.convert('L')
        dot_pattern_ = np.array(dot_pattern_)

        # various variables to handle the noise modelling
        scale_factor  = 100     # converting depth from m to cm 
        focal_length  = 480.0   # focal length of the camera used 
        baseline_m    = 0.075   # baseline in m 
        invalid_disp_ = 99999999.9

        h, w = input_depth.shape 

        # Our depth images were scaled by 5000 to store in png format so dividing to get 
        # depth in meters 
        depth = input_depth.astype('float')
        # Normalize depth to 0, 6
        depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth)) * 6

        depth_interp = self.add_gaussian_shifts(depth)

        disp_= focal_length * baseline_m / (depth_interp + 1e-10)
        depth_f = np.round(disp_ * 8.0)/8.0

        out_disp = self.filterDisp(depth_f, dot_pattern_, invalid_disp_)

        depth = focal_length * baseline_m / out_disp
        depth[out_disp == invalid_disp_] = 0 
        
        # The depth here needs to converted to cms so scale factor is introduced 
        # though often this can be tuned from [100, 200] to get the desired banding / quantisation effects 
        noisy_depth = (35130/(np.round((35130/(np.round(depth*scale_factor) + 1e-16)) + np.random.normal(size=(h, w))*(1.0/6.0) + 0.5)+1e-16))/ scale_factor 

        # Normalize depth back to 0, 255
        noisy_depth = (noisy_depth - np.min(noisy_depth)) / (np.max(noisy_depth) - np.min(noisy_depth)) * 255
        noisy_depth = noisy_depth.astype('uint16')

        return noisy_depth

def adapt_dataset(origin_directory_path, destination_directory_path, property_value, adaptation_method, split):
    if split == "empty":
        split = ""
    paths = [os.path.join(origin_directory_path, file) for file in os.listdir(origin_directory_path) if file.startswith(split)]
    destination_paths = [os.path.join(destination_directory_path, file) for file in os.listdir(origin_directory_path) if file.startswith(split)]
    paths.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    destination_paths.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    print("Property value: ", property_value)                       
    # Initialize the progress bar
    progress_bar = tqdm(total=len(paths), desc=f'Processing files in {origin_directory_path}')
    

    for path, destination_directory_path in zip(paths, destination_paths):
        if isinstance(property_value, tuple) and len(property_value) == 2:
            # Get a random value between the two values
            new_saturation = adaptation_method(path, destination_directory_path, np.random.uniform(property_value[0], property_value[1]))
        else:
            new_saturation = adaptation_method(path, destination_directory_path, property_value)
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()


def adjust_saturation(image_path, destination_directory_path, saturation_value):
    image = Image.open(image_path)
    saturated_image = F.adjust_saturation(image, saturation_value)

    # Save the image
    saturated_image.save(destination_directory_path)

    return saturation_value

def adjust_hue(image_path, destination_directory_path, hue):
    image = Image.open(image_path)
    color_image = F.adjust_hue(image, hue)

    # Save the image
    color_image.save(destination_directory_path)

    return hue

def adjust_brightness(image_path, destination_directory_path, brightness_factor):
    image = Image.open(image_path)
    brightened_image = F.adjust_brightness(image, brightness_factor)

    # Save the image
    brightened_image.save(destination_directory_path)

    return brightness_factor

def adjust_focus_with_multi_otsu_thresholding(image_path, destination_directory_path, _):
    rgb_image = np.array(Image.open(image_path))
    
    image_directory_path = os.path.dirname(image_path)
    dataset_directory_path = os.path.dirname(image_directory_path)
    depth_directory_path = os.path.join(dataset_directory_path, "Depth")
    if not os.path.exists(depth_directory_path):
        raise FileNotFoundError("The depth directory does not exist")
    
    depth_image = np.array(Image.open(os.path.join(depth_directory_path, os.path.basename(image_path))))
    
    num_thresholds = 3
    thresholds = threshold_multiotsu(depth_image, classes=num_thresholds)
    # Generate three classes from the original image
    regions = np.digitize(depth_image, bins=thresholds)

    most_important_region = 0
    second_most_important_region = 1
    third_most_important_region = 2

    rgb_cars_only = rgb_image.copy()
    rgb_cars_only[regions == most_important_region] = rgb_cars_only[regions == most_important_region] * 1.0
    rgb_cars_only[regions == second_most_important_region] = rgb_cars_only[regions == second_most_important_region] * 0.3
    rgb_cars_only[regions == third_most_important_region] = rgb_cars_only[regions == third_most_important_region] * 0.0

    # Save the image
    Image.fromarray(rgb_cars_only).save(destination_directory_path)

def adjust_depth_kinect(image_path, destination_directory_path, _):
    depth_image = np.array(Image.open(image_path))
    kinect_noise = KinectNoise()
    if len(depth_image.shape) == 3:
        red_noise = kinect_noise.create_noisy_depth(depth_image[:, :, 0])
        green_noise = kinect_noise.create_noisy_depth(depth_image[:, :, 1])
        blue_noise = kinect_noise.create_noisy_depth(depth_image[:, :, 2])
        depth_image = np.stack([red_noise, green_noise, blue_noise], axis=-1)
    else:
        depth_image =kinect_noise.create_noisy_depth(depth_image)
    depth_image = (depth_image * 255).astype('uint8')
    depth_image = Image.fromarray(depth_image)
    depth_image.save(destination_directory_path)

    return None

def random_shift_and_mirror_noise(image, shift_range=15, mirror_probability=0.0):
    W, H = image.shape[:2]

    # Randomly shift the image
    shift_x = np.random.randint(-W // shift_range, W // shift_range)
    shift_y = np.random.randint(-H // shift_range, H // shift_range)
    image = np.roll(image, shift_x, axis=0)
    image = np.roll(image, shift_y, axis=1)

    if np.random.rand() < mirror_probability:
        image = np.flip(image, axis=0)
    if np.random.rand() < mirror_probability:
        image = np.flip(image, axis=1)

    return image

def adjust_random_shift_and_mirror_noise(image_path, destination_directory_path, _):
    image = np.array(Image.open(image_path))
    noisy_image = random_shift_and_mirror_noise(image)
    noisy_image = Image.fromarray(noisy_image)
    noisy_image.save(destination_directory_path)

    return None

def adjust_white(image_path, destination_directory_path, _):
    image = np.array(Image.open(image_path))
    image = np.ones_like(image) * 255
    image = Image.fromarray(image)
    image.save(destination_directory_path)

    return None

def adjust_black(image_path, destination_directory_path, _):
    image = np.array(Image.open(image_path))
    image = np.zeros_like(image)
    image = Image.fromarray(image)
    image.save(destination_directory_path)

    return None

def adjust_depth_color(image_path, destination_directory_path, _):
    depth_image = np.array(Image.open(image_path))
    depth_image = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image))
    depth_image = (depth_image * 255).astype('uint8')
    depth_image = plt.get_cmap('jet')(depth_image)[:, :, :3]
    depth_image = (depth_image * 255).astype('uint8')
    depth_image = Image.fromarray(depth_image)
    depth_image.save(destination_directory_path)

    return None

def adjust_random_noise(image_path, destination_directory_path, _):
    image = np.array(Image.open(image_path))
    noise = np.random.normal(0, 1, image.shape)
    noisy_image = np.clip(noise * 255, 0, 255).astype(np.uint8)
    noisy_image = Image.fromarray(noisy_image)
    noisy_image.save(destination_directory_path)

    return None

def copy_image(image_path, destination_directory_path, _):
    image = Image.open(image_path)
    image.save(destination_directory_path)

    return None

def adjust_blur(image_path, destination_directory_path, property_value):
    image = Image.open(image_path)
    property_value = int(property_value)
    image = np.array(image)
    if isinstance(property_value, tuple) and len(property_value) == 2:
        property_value = np.random.randint(property_value[0], property_value[1])
    image += np.min(image)
    image = (image / np.max(image) * 255).astype(np.uint8)
    # image = image.astype(np.uint8)
    blurred_image = cv2.GaussianBlur(image, (property_value, property_value), 0)

    # Save the image
    Image.fromarray(blurred_image).save(destination_directory_path)

    return None

def adjust_label_fgbg(image_path, destination_directory_path, _):
    label_image = np.array(Image.open(image_path))
    # Check that only integer values in the range 0, 255 are present
    assert label_image.dtype == np.uint8 and label_image.min() >= 0 and label_image.max() <= 255 
    label_image = np.where(label_image > 0, 1, 0)
    label_image = Image.fromarray(label_image)
    label_image.save(destination_directory_path)

def adapt_property(origin_directory_path, destination_directory_path, property_value, property_name, split):
    if property_name == "saturation":
        adapt_dataset(origin_directory_path, destination_directory_path, property_value, adjust_saturation, split)
    if property_name == "hue":
        adapt_dataset(origin_directory_path, destination_directory_path, property_value, adjust_hue, split)
    if property_name == "brightness":
        adapt_dataset(origin_directory_path, destination_directory_path, property_value, adjust_brightness, split)
    if property_name == "focus":
        adapt_dataset(origin_directory_path, destination_directory_path, property_value, adjust_focus_with_multi_otsu_thresholding, split)
    if property_name == "depth_kinect_noise":
        adapt_dataset(origin_directory_path, destination_directory_path, property_value, adjust_depth_kinect, split)
    if property_name == "random_shift_and_mirror":
        adapt_dataset(origin_directory_path, destination_directory_path, property_value, adjust_random_shift_and_mirror_noise, split)
    if property_name == "white":
        adapt_dataset(origin_directory_path, destination_directory_path, property_value, adjust_white, split)
    if property_name == "black":
        adapt_dataset(origin_directory_path, destination_directory_path, property_value, adjust_black, split)
    if property_name == "depth_color":
        adapt_dataset(origin_directory_path, destination_directory_path, property_value, adjust_depth_color, split)
    if property_name == "random_noise":
        adapt_dataset(origin_directory_path, destination_directory_path, property_value, adjust_random_noise, split)
    if property_name == "copy":
        adapt_dataset(origin_directory_path, destination_directory_path, property_value, copy_image, split)
    if property_name == "blur":
        adapt_dataset(origin_directory_path, destination_directory_path, property_value, adjust_blur, split)
    if property_name == "label_fgbg":
        adapt_dataset(origin_directory_path, destination_directory_path, property_value, adjust_label_fgbg, split)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-op", "--origin_directory_path", type=str, default="datasets/SUNRGBD/RGB_original", help="The path to the SUNRGBD dataset")
    argparser.add_argument("-dp", "--destination_directory_path", type=str, default="datasets/SUNRGBD/RGB", help="The path to save the adapted dataset")
    argparser.add_argument("-s", "--split", type=str, default="test", help="The split to consider")
    argparser.add_argument("-pname", '--property_name', help='Property name', default='saturation')
    argparser.add_argument("-pmin", '--min_property_value', help='Minimum property value', default=0.0, type=float)
    argparser.add_argument("-pmax", '--max_property_value', help='Maximum property value', default=1.0, type=float)

    args = argparser.parse_args()

    if args.min_property_value != args.max_property_value:
        property_value = (args.min_property_value, args.max_property_value)
    else:
        property_value = args.min_property_value

    adapt_property(args.origin_directory_path, args.destination_directory_path, property_value, args.property_name, args.split)
    