import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import datasets, transforms
import torch

class MNIST_Transformer:
    # Create a set of parameters for the transformer
    def __init__(self, location, train_images, train_labels, test_images, test_labels):
        self.location = location
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels

    def create_mnist_with_background(self, save_folder):
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

        for i, img in enumerate(self.train_images):
            new_img = np.zeros((28, 28, 3))
            new_img[np.where(img == 0)] = np.random.randint(0, 255, size=3)
            new_img[np.where(img != 0)] = np.repeat(img[np.where(img != 0)], 3, axis=-1).reshape((-1, 3))
            train_images_with_background.append(new_img)
            plt.imsave(save_folder + "train_images/" + str(i) + ".png", new_img / 255.0)

        for i, img in enumerate(self.test_images):
            new_img = np.zeros((28, 28, 3))
            new_img[np.where(img == 0)] = np.random.randint(0, 255, size=3)
            new_img[np.where(img != 0)] = np.repeat(img[np.where(img != 0)], 3, axis=-1).reshape((-1, 3))
            test_images_with_background.append(new_img)
            plt.imsave(save_folder + "test_images/" + str(i) + ".png", new_img / 255.0)

        
        # Save the dataset as a numpy array
        np.save(save_folder + "train_images.npy", train_images_with_background)
        np.save(save_folder + "train_labels.npy", self.train_labels)
        np.save(save_folder + "test_images.npy", test_images_with_background)
        np.save(save_folder + "test_labels.npy", self.test_labels)

    def get_dataloader(self, dataset_location):
        # Define the transform to apply to the data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        # Load the dataset
        dataset = datasets.ImageFolder(dataset_location, transform=transform)

        # Create a data loader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        return dataloader