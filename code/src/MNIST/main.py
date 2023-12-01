import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Network.RGBD_classification_network import RGBDClassifier
from Data_Preprocess.Transform_MNIST import MNIST_Transformer
from Data_Preprocess.Download_MNIST import download_MNIST_data

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Specify the folder where you want to save the dataset
location = "../../data/"

##################### Constants ######################
data_creation = True
plotting = True

##################### Data creation ######################

if data_creation:
    train_images, train_labels, test_images, test_labels = download_MNIST_data(location)

    # Create a transformer object
    transformer = MNIST_Transformer(location, train_images, train_labels, test_images, test_labels)

    # Create a dataset with a random colored background
    save_folder = location + "MNIST_with_background_select/"

    # Create the dataset
    transformer.create_rgbd_MNIST_with_background(save_folder, train_color=(0, 255, 0), test_color=(255, 0, 0))


##################### MNIST with background ######################
location_MNIST_background = location + "MNIST_with_background_select/"

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

# Specify parameters
image_width = 28
image_height = 28
camera_params = {}
camera_params['intrinsic'] = {}
camera_params['intrinsic']['fx'] = None
camera_params['intrinsic']['fy'] = None
camera_params['intrinsic']['cx'] = None
camera_params['intrinsic']['cy'] = None

for k1 in camera_params:
    for k2 in camera_params[k1]:
        camera_params[k1][k2] = torch.from_numpy(np.array(camera_params[k1][k2], dtype=np.float32)).float()

# Create the neural network model
rgbd_model = RGBDClassifier(image_width=image_width, image_height=image_height).to(device)
rgb_model = RGBDClassifier(image_width=image_width, image_height=image_height).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()

rgbd_optimizer = optim.Adam(rgbd_model.parameters(), lr=0.001)
rgb_optimizer = optim.Adam(rgb_model.parameters(), lr=0.001)

rgb_accuracy_list = []
rgbd_accuracy_list = []

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        rgbd_optimizer.zero_grad()
        rgb_optimizer.zero_grad()

        x = data[:, :, :, :3].permute(0, 3, 1, 2) / 255.0
        depth = data[:, :, :, 3:4].permute(0, 3, 1, 2) / 255.0
        fake_depth = torch.zeros_like(depth)

        camera_params['intrinsic']['fx'] = torch.ones((data.shape[0], 1, 1)).to(device) * 500
        
        rgbd_outputs = rgbd_model(x, depth, camera_params)
        rgb_outputs = rgb_model(x, fake_depth, camera_params)
        
        rgbd_loss = criterion(rgbd_outputs, targets)
        rgbd_loss.backward()
        rgbd_optimizer.step()

        rgb_loss = criterion(rgb_outputs, targets)
        rgb_loss.backward()
        rgb_optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], rgb loss: {rgb_loss.item():.4f}, rgbd loss: {rgbd_loss.item():.4f}')


    # Evaluate the model
    predictions = []

    rgbd_model.eval()
    rgb_model.eval()
    # with torch.no_grad():
    rgbd_correct = 0
    rgb_correct = 0
    total = 0
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        x = data[:, :, :, :3].permute(0, 3, 1, 2) / 255.0
        depth = data[:, :, :, 3:4].permute(0, 3, 1, 2) / 255.0
        fake_depth = torch.zeros_like(depth)

        camera_params['intrinsic']['fx'] = torch.ones((data.shape[0], 1, 1)).to(device) * 500

        rgbd_outputs = rgbd_model(x, depth, camera_params)
        rgb_outputs = rgb_model(x, fake_depth, camera_params)

        _, rgbd_predicted = torch.max(rgbd_outputs.data, 1)
        _, rgb_predicted = torch.max(rgb_outputs.data, 1)

        total += targets.size(0)
        rgbd_correct += (rgbd_predicted == targets).sum().item()
        rgb_correct += (rgb_predicted == targets).sum().item()

        predictions.append({
            "image": data.cpu().numpy(),
            "label": targets.cpu().numpy(),
            "rgbd_prediction": rgbd_predicted.cpu().numpy(),
            "rgb_prediction": rgb_predicted.cpu().numpy()
        })

    rgbd_accuracy = 100 * rgbd_correct / total
    rgb_accuracy = 100 * rgb_correct / total
    print(f'Test RGBD Accuracy: {rgbd_accuracy:.2f}%')
    print(f'Test RGB Accuracy: {rgb_accuracy:.2f}%')

    rgbd_accuracy_list.append(rgbd_accuracy)
    rgb_accuracy_list.append(rgb_accuracy)


# Plot the accuracy over epochs
if plotting:
    plt.plot(rgbd_accuracy_list, label="RGBD")
    plt.plot(rgb_accuracy_list, label="RGB")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of a RGBD and RGB classifier on MNIST with random background colors")
    plt.legend()
    plt.show()

# Plot the predictions
if plotting:
    for x in range(10):
        fig, ax = plt.subplots(5, 5, figsize=(20, 20))
        fig.suptitle('Predictions', fontsize=20)
        for i in range(5):
            for j in range(5):
                image = predictions[x]["image"][i+5*j][:, :, :3]
                image = (image - image.min()) / (image.max() - image.min())  # Normalize image
                ax[i, j].imshow(image)
                ax[i, j].set_title(f"Label: {predictions[x]['label'][i+5*j]}\nRGBD Prediction: {predictions[x]['rgbd_prediction'][i+5*j]}\nRGB Prediction: {predictions[x]['rgb_prediction'][i+5*j]}")
                ax[i, j].axis('off')

        plt.show()