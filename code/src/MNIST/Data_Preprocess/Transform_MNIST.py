import matplotlib.pyplot as plt
import numpy as np
import os

class MNIST_Transformer:
    # Create a set of parameters for the transformer
    def __init__(self, train_images, train_labels, test_images, test_labels):
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels

    def create_rgbd_MNIST_with_background(self, save_folder, train_transforms=[], test_transforms=[]):
        # Create save folder if not exists
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Create train and test folders if not exists
        if not os.path.exists(save_folder + "train_images/"):
            os.makedirs(save_folder + "train_images/")
        if not os.path.exists(save_folder + "test_images/"):
            os.makedirs(save_folder + "test_images/")

        # Set a random colored background to the digits
        train_images = []
        test_images = []

        samples = 25

        for i, img in enumerate(self.train_images):
            H, W = img.shape
            new_img = self.normalize_image(np.repeat(img, 3, axis=-1).reshape((H, W, 3)))
            new_depth = self.normalize_image(np.array(255.0 - img))

            for transform in train_transforms:
                new_img, new_depth = transform(new_img, new_depth)

            train_images.append(np.concatenate((new_img, new_depth.reshape((H, W, 1))), axis=-1))

            if i < samples:
                plt.imsave(save_folder + "train_images/" + str(i) + ".png", new_img)
                plt.imsave(save_folder + "train_images/" + str(i) + "_depth.png", new_depth, cmap="gray")

        for i, img in enumerate(self.test_images):
            new_img = self.normalize_image(np.repeat(img, 3, axis=-1).reshape((H, W, 3)))
            new_depth = self.normalize_image(np.array(255.0 - img))

            for transform in test_transforms:
                new_img, new_depth = transform(new_img, new_depth)

            test_images.append(np.concatenate((new_img, new_depth.reshape((H, W, 1))), axis=-1))

            if i < samples:
                plt.imsave(save_folder + "test_images/" + str(i) + ".png", new_img)
                plt.imsave(save_folder + "test_images/" + str(i) + "_depth.png", new_depth, cmap="gray")

        
        # Save the dataset as a numpy array
        np.save(save_folder + "train_images.npy", train_images)
        np.save(save_folder + "train_labels.npy", self.train_labels)
        np.save(save_folder + "test_images.npy", test_images)
        np.save(save_folder + "test_labels.npy", self.test_labels)

    def normalize_image(self, img):
        # Make sure img is a numpy array
        img = np.array(img)
        if (np.max(img) - np.min(img)) == 0:
            if np.max(img) == 0:
                return img
            else:
                return img / np.max(img)
        return (img - np.min(img)) / (np.max(img) - np.min(img))


    def transform_add_background(self, img, depth, color_range=(255, 255, 255)):
        new_img = np.zeros_like(img)
        # Add a random background to the image
        background_color = [
            np.random.randint(0, max(1, color_range[0])), 
            np.random.randint(0, max(1, color_range[1])), 
            np.random.randint(0, max(1, color_range[2]))
        ]
        background_img = np.ones_like(img[:, :, :3]) * background_color
        new_img = np.where(img == 0, self.normalize_image(background_img), img)
        # Keep depth the same
        return new_img, depth
    
    def transform_add_noise(self, img, depth, img_noise_range=(0, 255), depth_noise_range=(0, 255)):
        new_img = np.zeros_like(img)
        # Add noise to the image
        img_noise = np.random.randint(img_noise_range[0], np.max([1, img_noise_range[1]]), new_img.shape)
        new_img =  self.normalize_image(np.array(img + img_noise))
        # Add noise to the depth
        depth_noise = np.random.randint(depth_noise_range[0], np.max([1, depth_noise_range[1]]), depth.shape)
        new_depth = self.normalize_image(np.array(depth + depth_noise))
        
        return new_img, new_depth
    
    def transform_add_background_noise(self, img, depth, img_noise_range=(0, 255)):
        new_img = np.zeros_like(img)
        mask = depth == 1.0
        mask_rgb = np.repeat(mask, 3, axis=-1).reshape(img.shape)
        # Add a random background to the image
        img_noise = np.random.randint(img_noise_range[0], np.max([1, img_noise_range[1]]), new_img.shape)
        # Add noise to the image based on depth
        new_img = np.where(mask_rgb, self.normalize_image(img + img_noise), img)
        new_depth = np.where(mask, self.normalize_image(depth + np.mean(img_noise, axis=-1)), depth)

        return new_img, new_depth
