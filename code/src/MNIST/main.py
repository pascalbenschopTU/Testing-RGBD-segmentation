from Download_MNIST import *
from Transform_MNIST import MNIST_Transformer

# Specify the folder where you want to save the dataset
location = "../../data/"

train_images, train_labels, test_images, test_labels = get_MNIST_data(location)

# Create a transformer object
transformer = MNIST_Transformer(location, train_images, train_labels, test_images, test_labels)

# Create a dataset with a random colored background
save_folder = location + "MNIST_with_background/"

# Create the dataset
transformer.create_mnist_with_background(save_folder)