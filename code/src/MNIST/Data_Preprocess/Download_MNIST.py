from torchvision import datasets, transforms

def download_MNIST_data(location):
    # Define the transform to apply to the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Download the MNIST dataset
    train_dataset = datasets.MNIST(root=location, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=location, train=False, download=True, transform=transform)

    # Access the images and labels
    train_images = train_dataset.data
    train_labels = train_dataset.targets
    test_images = test_dataset.data
    test_labels = test_dataset.targets

    return train_images, train_labels, test_images, test_labels
