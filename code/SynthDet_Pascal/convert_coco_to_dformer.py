import argparse
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from CocoDepthDataset import CocoDepthDataset
from PIL import Image
import multiprocessing

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
        dot_pattern_ = Image.open("./data/kinect-pattern_3x3.png")
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

class AdaptiveDatasetCreator:
    def __init__(self, dataset_root, save_location, image_size=(480, 480), dataset_split=(0.7, 0.3)):
        self.dataset_root = dataset_root
        self.save_location = save_location
        self.image_size = image_size
        self.dataset_split = dataset_split

        # Create directories to save the converted dataset
        os.makedirs(os.path.join(save_location, 'RGB'), exist_ok=True)
        os.makedirs(os.path.join(save_location, 'Depth'), exist_ok=True)
        os.makedirs(os.path.join(save_location, 'labels'), exist_ok=True)

        # Create directories for the different types of depth images
        # os.makedirs(os.path.join(save_location, 'Depth_black'), exist_ok=True)
        os.makedirs(os.path.join(save_location, 'Depth_kinect_noise'), exist_ok=True)
        os.makedirs(os.path.join(save_location, 'Depth_noise'), exist_ok=True)
        os.makedirs(os.path.join(save_location, 'Depth_compressed'), exist_ok=True)

        # Create train.txt and test.txt files
        with open(os.path.join(save_location, 'train.txt'), 'w'):
            pass

        with open(os.path.join(save_location, 'test.txt'), 'w'):
            pass

    def convert_and_save_RGB(self, rgb_data, file_name):
        rgb_image = rgb_data[0].numpy()
        rgb_image = (rgb_image * 255).astype('uint8')
        rgb_image = Image.fromarray(rgb_image.transpose(1, 2, 0))
        rgb_image.save(os.path.join(self.save_location, f"RGB/{file_name}.png"))
    
    def convert_and_save_label(self, label, file_name):
        label = label.squeeze(1)
        label_image = label[0].numpy()
        label_image = label_image.astype('uint8')
        label_image = Image.fromarray(label_image)
        label_image.save(os.path.join(self.save_location, f"labels/{file_name}.png"))

    def convert_and_save_depth(self, depth_data, file_name):
        depth_data = depth_data.squeeze(1)
        depth_data = depth_data[0].numpy()
        depth_data = (depth_data * 255).astype('uint8')
        depth_image = Image.fromarray(depth_data)
        depth_image.save(os.path.join(self.save_location, f"Depth/{file_name}.png"))

    def convert_and_save_depth_compressed(self, depth_data, file_name, compression_factor=0.5):
        depth_data = depth_data.squeeze(1)
        depth_data = depth_data[0].numpy()
        compressed_depth_data = np.power(depth_data / 255.0, compression_factor)
        compressed_depth_image = (compressed_depth_data * 255).astype('uint8')
        compressed_depth_image = Image.fromarray(compressed_depth_image)
        compressed_depth_image.save(os.path.join(self.save_location, f"Depth_compressed/{file_name}.png"))

    def convert_and_save_depth_black(self, file_name):
        depth_black_image = Image.new('L', self.image_size, 0)
        depth_black_image.save(os.path.join(self.save_location, f"Depth_black/{file_name}.png"))

    def convert_and_save_depth_kinect_noise(self, depth_data, file_name):
        depth_data = depth_data.squeeze(1)
        depth_data = depth_data[0].numpy()
        depth_data = KinectNoise().create_noisy_depth(depth_data)
        depth_data = (depth_data * 255).astype('uint8')
        depth_image = Image.fromarray(depth_data)
        depth_image.save(os.path.join(self.save_location, f"Depth_kinect_noise/{file_name}.png"))


    def convert_and_save_depth_noise(self, depth_data, file_name, noise_factor=0.1):
        depth_data = depth_data.squeeze(1)
        depth_data = depth_data[0].numpy()
        noisy_depth_data = np.clip(depth_data + np.random.normal(scale=noise_factor, size=depth_data.shape), 0.0, 1.0)
        noisy_depth_data = (noisy_depth_data * 255).astype('uint8')
        noisy_depth_image = Image.fromarray(noisy_depth_data)
        noisy_depth_image.save(os.path.join(self.save_location, f"Depth_noise/{file_name}.png"))


    def convert_and_save_dataset(self):
        # Define data transformations
        transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=Image.NEAREST),  # Adjust the size based on your model's input size
            transforms.ToTensor(),
        ])

        dataset = CocoDepthDataset(os.path.join(self.dataset_root, 'images'), os.path.join(self.dataset_root, 'semantic.json'), os.path.join(self.dataset_root, 'depth'), transform=transform)

        # Split dataset into training and validation sets
        train_size = int(len(dataset) * self.dataset_split[0])
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        # Define the number of workers for parallel data loading
        num_workers = multiprocessing.cpu_count()

        # Define data loaders
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

        # Iterate through the dataset and save to another location
        for batch_idx, (rgb_data, depth_data, label) in enumerate(train_loader):
            self.convert_and_save_RGB(rgb_data, f"train_{batch_idx}")
            self.convert_and_save_label(label, f"train_{batch_idx}")
            self.convert_and_save_depth(depth_data, f"train_{batch_idx}")
            self.convert_and_save_depth_compressed(depth_data, f"train_{batch_idx}")
            self.convert_and_save_depth_kinect_noise(depth_data, f"train_{batch_idx}")
            self.convert_and_save_depth_noise(depth_data, f"train_{batch_idx}", noise_factor=10)

            # add a line to train.txt with the path to the RGB image and the path to the depth image
            # Example: RGB/train_0.jpg labels/train_0.png
            with open(os.path.join(self.save_location, 'train.txt'), 'a') as train_file:
                train_file.write(f"RGB/train_{batch_idx}.png labels/train_{batch_idx}.png\n")

        for batch_idx, (rgb_data, depth_data, label) in enumerate(test_loader):
            self.convert_and_save_RGB(rgb_data, f"test_{batch_idx}")
            self.convert_and_save_label(label, f"test_{batch_idx}")
            self.convert_and_save_depth(depth_data, f"test_{batch_idx}")
            self.convert_and_save_depth_compressed(depth_data, f"test_{batch_idx}")
            self.convert_and_save_depth_kinect_noise(depth_data, f"test_{batch_idx}")
            self.convert_and_save_depth_noise(depth_data, f"test_{batch_idx}", noise_factor=10)

            # add a line to test.txt with the path to the RGB image and the path to the depth image
            # Example: RGB/test_0.jpg labels/test_0.png
            with open(os.path.join(self.save_location, 'test.txt'), 'a') as test_file:
                test_file.write(f"RGB/test_{batch_idx}.png labels/test_{batch_idx}.png\n")


def main():
    parser = argparse.ArgumentParser(description='Convert COCO dataset to DFormer dataset')
    parser.add_argument('dataset_root', help='Path to COCO dataset')
    parser.add_argument('save_location', help='Path to save the converted dataset')
    parser.add_argument('--dataset_split', nargs=2, type=float, default=(0.7, 0.3), help='Train and test split')
    args = parser.parse_args()
    
    dataset_creator = AdaptiveDatasetCreator(args.dataset_root, args.save_location, dataset_split=args.dataset_split)
    dataset_creator.convert_and_save_dataset()


if __name__ == '__main__':
    main()