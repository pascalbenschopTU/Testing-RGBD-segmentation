import matplotlib.pyplot as plt
import numpy as np
import os

class MNIST_Transformer:
    # Create a set of parameters for the transformer
    def __init__(self, location, train_images, train_labels, test_images, test_labels):
        self.location = location
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels

    def create_rgbd_MNIST_with_background(self, save_folder, train_color=(0, 255, 0), test_color=(255, 0, 0)):
        # Create save folder if not exists
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Create train and test folders if not exists
        if not os.path.exists(save_folder + "train_images/"):
            os.makedirs(save_folder + "train_images/")
        if not os.path.exists(save_folder + "test_images/"):
            os.makedirs(save_folder + "test_images/")

        # Set a random colored background to the digits
        train_images_with_background = []
        test_images_with_background = []

        samples = 256

        for i, img in enumerate(self.train_images):
            new_img = np.zeros((28, 28, 4))
            new_img[:, :, :3][np.where(img == 0)] = [np.random.randint(0, max(1, train_color[0])), np.random.randint(0, max(1, train_color[1])), np.random.randint(0, max(1, train_color[2]))]
            new_img[:, :, :3][np.where(img != 0)] = np.repeat(img[np.where(img != 0)], 3, axis=-1).reshape((-1, 3))
            new_img[:, :, 3] = 255 - img # Invert original mnist digit to create sense of depth
            train_images_with_background.append(new_img)
            if i < samples:
                plt.imsave(save_folder + "train_images/" + str(i) + ".png", new_img[:, :, :3] / 255.0)
                plt.imsave(save_folder + "train_images/" + str(i) + "_depth.png", new_img[:, :, 3] / 255.0, cmap="gray")

        for i, img in enumerate(self.test_images):
            new_img = np.zeros((28, 28, 4))
            new_img[:, :, :3][np.where(img == 0)] = [np.random.randint(0, max(1, test_color[0])), np.random.randint(0, max(1, test_color[1])), np.random.randint(0, max(1, test_color[2]))]
            new_img[:, :, :3][np.where(img != 0)] = np.repeat(img[np.where(img != 0)], 3, axis=-1).reshape((-1, 3))
            new_img[:, :, 3] = 1.0 - img # Invert original mnist digit to create sense of depth
            test_images_with_background.append(new_img)
            if i < samples:
                plt.imsave(save_folder + "test_images/" + str(i) + ".png", new_img[:, :, :3] / 255.0)
                plt.imsave(save_folder + "test_images/" + str(i) + "_depth.png", new_img[:, :, 3] / 255.0, cmap="gray")

        
        # Save the dataset as a numpy array
        np.save(save_folder + "train_images.npy", train_images_with_background)
        np.save(save_folder + "train_labels.npy", self.train_labels)
        np.save(save_folder + "test_images.npy", test_images_with_background)
        np.save(save_folder + "test_labels.npy", self.test_labels)