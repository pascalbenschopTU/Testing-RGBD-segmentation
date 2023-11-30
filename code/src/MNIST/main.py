from Download_MNIST import *
from Transform_MNIST import MNIST_Transformer
from RGBD_classification_network import RGBDClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Specify the folder where you want to save the dataset
location = "../../data/"

data_creation = False
plotting = True

if data_creation:
    train_images, train_labels, test_images, test_labels = get_MNIST_data(location)

    # Create a transformer object
    transformer = MNIST_Transformer(location, train_images, train_labels, test_images, test_labels)

    # Create a dataset with a random colored background
    save_folder = location + "MNIST_with_background/"

    # Create the dataset
    transformer.create_rgbd_MNIST_with_background(save_folder)


# Usage example
location_MNIST_background = location + "MNIST_with_background/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################## RGBD example ##########################
# Load the dataset
train_image_file = location_MNIST_background + "train_images.npy"
train_label_file = location_MNIST_background + "train_labels.npy"
test_image_file = location_MNIST_background + "test_images.npy"
test_label_file = location_MNIST_background + "test_labels.npy"

train_dataset = TensorDataset(
    torch.tensor(np.load(train_image_file), dtype=torch.float32).to(device), 
    torch.tensor(np.load(train_label_file), dtype=torch.long).to(device)  # Change dtype to torch.long
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(
    torch.tensor(np.load(test_image_file), dtype=torch.float32).to(device), 
    torch.tensor(np.load(test_label_file), dtype=torch.long).to(device)  # Change dtype to torch.long
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Create the neural network model
model = RGBDClassifier().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 1
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

predictions = []

# Evaluate the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        predictions.append({
            "image": data.cpu().numpy(),
            "label": targets.cpu().numpy(),
            "prediction": predicted.cpu().numpy()
        })

        

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')


if plotting:
    # Plot the incorrect predictions
    for x in range(10):
        fig, ax = plt.subplots(5, 5, figsize=(10, 10))
        fig.suptitle('Predictions', fontsize=20)
        for i in range(5):
            for j in range(5):
                image = predictions[x]["image"][i+5*j][:, :, :3]
                image = (image - image.min()) / (image.max() - image.min())  # Normalize image
                ax[i, j].imshow(image)
                ax[i, j].set_title(f"Label: {predictions[x]['label'][i+5*j]}\nPrediction: {predictions[x]['prediction'][i+5*j]}")
                ax[i, j].axis('off')



        plt.show()