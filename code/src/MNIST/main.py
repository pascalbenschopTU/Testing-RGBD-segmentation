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
from torch.utils.data.sampler import SubsetRandomSampler

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Specify the folder where you want to save the dataset
data_location = "../../data/"
experiment_name = "MNIST_occlusion_multiple"
test_location = data_location + experiment_name + "/"

##################### Constants ######################
data_creation = True
plotting = True

##################### Data creation ######################

if data_creation:
    train_images, train_labels, test_images, test_labels = download_MNIST_data(data_location)

    # Create a transformer object
    transformer = MNIST_Transformer(train_images, train_labels, test_images, test_labels)

    # Create the dataset
    transformer.create_rgbd_MNIST_with_background(
        test_location, 
        train_transforms=[
            lambda img, depth: transformer.add_background(img, depth, color_range=(255, 255, 255)),
            # lambda img, depth: transformer.add_background_noise(img, depth, img_noise_range=(0, 255)),
            lambda img, depth: transformer.add_occlusion(img, depth, occlusion_size=(5, 10), occlusion_color_range=(255, 255, 255)),
            lambda img, depth: transformer.add_occlusion(img, depth, occlusion_size=(5, 15), occlusion_color_range=(255, 255, 255)),
        ],
        test_transforms=[
            lambda img, depth: transformer.add_background(img, depth, color_range=(255, 255, 255)),
            # lambda img, depth: transformer.add_background_noise(img, depth, img_noise_range=(150, 255)),
            lambda img, depth: transformer.add_occlusion(img, depth, occlusion_size=(5, 10), occlusion_color_range=(255, 255, 255)),
            lambda img, depth: transformer.add_occlusion(img, depth, occlusion_size=(5, 15), occlusion_color_range=(255, 255, 255)),
        ]
    )

########################## RGBD example ##########################

# Load the dataset
train_image_file = test_location + "train_images.npy"
train_label_file = test_location + "train_labels.npy"
test_image_file = test_location + "test_images.npy"
test_label_file = test_location + "test_labels.npy"

train_dataset = TensorDataset(
    torch.tensor(np.load(train_image_file), dtype=torch.float32).to(device), 
    torch.tensor(np.load(train_label_file), dtype=torch.long).to(device)  # Change dtype to torch.long
)

test_dataset = TensorDataset(
    torch.tensor(np.load(test_image_file), dtype=torch.float32).to(device), 
    torch.tensor(np.load(test_label_file), dtype=torch.long).to(device)  # Change dtype to torch.long
)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

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
def reset_model():
    # Create the neural network model
    rgbd_model = RGBDClassifier(image_width=image_width, image_height=image_height).to(device)
    rgb_model = RGBDClassifier(image_width=image_width, image_height=image_height).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    rgbd_optimizer = optim.Adam(rgbd_model.parameters(), lr=0.001)
    rgb_optimizer = optim.Adam(rgb_model.parameters(), lr=0.001)

    return rgbd_model, rgb_model, criterion, rgbd_optimizer, rgb_optimizer

# Evaluate the model
predictions = []

max_data_size = 100

# Reset the model before each data size iteration
for data_size in range(1, max_data_size + 1):
    rgbd_model, rgb_model, criterion, rgbd_optimizer, rgb_optimizer = reset_model()

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, 
                              sampler=SubsetRandomSampler(range(int((data_size/max_data_size) * len(train_dataset)))))

    # Continue with the rest of the code...
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        rgbd_optimizer.zero_grad()
        rgb_optimizer.zero_grad()

        x = data[:, :, :, :3].permute(0, 3, 1, 2)
        depth = data[:, :, :, 3:4].permute(0, 3, 1, 2)
        fake_depth = torch.ones_like(depth)

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
            print(f'Step [{batch_idx+1}/{len(train_loader)}], rgb loss: {rgb_loss.item():.4f}, rgbd loss: {rgbd_loss.item():.4f}')

    rgbd_model.eval()
    rgb_model.eval()
    with torch.no_grad():
        rgbd_correct = 0
        rgb_correct = 0
        total = 0
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)

            x = data[:, :, :, :3].permute(0, 3, 1, 2)
            depth = data[:, :, :, 3:4].permute(0, 3, 1, 2)
            fake_depth = torch.ones_like(depth)

            camera_params['intrinsic']['fx'] = torch.ones((data.shape[0], 1, 1)).to(device) * 500

            rgbd_outputs = rgbd_model(x, depth, camera_params)
            rgb_outputs = rgb_model(x, fake_depth, camera_params)

            _, rgbd_predicted = torch.max(rgbd_outputs.data, 1)
            _, rgb_predicted = torch.max(rgb_outputs.data, 1)

            total += targets.size(0)
            rgbd_correct += (rgbd_predicted == targets).sum().item()
            rgb_correct += (rgb_predicted == targets).sum().item()

            if data_size / max_data_size > 0.9:
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
        print(f"Data size: {data_size/max_data_size * 100}%")

        rgbd_accuracy_list.append(rgbd_accuracy)
        rgb_accuracy_list.append(rgb_accuracy)


# Plot the accuracy over epochs
if plotting:
    x_values = [i*(100 // max_data_size) for i in range(len(rgbd_accuracy_list))]
    plt.plot(x_values, rgbd_accuracy_list, label="RGBD")
    plt.plot(x_values, rgb_accuracy_list, label="RGB")
    plt.ylabel("Accuracy")
    plt.xlabel("Data size (percentage)")
    plt.title("Accuracy of a RGBD and RGB classifier on different data sizes")
    plt.legend()
    plt.xlim(0, 100)
    plt.savefig(test_location + "results.png")
    plt.show()
    

# Plot the predictions
if plotting:
    prediction_index = 0
    fig, ax = plt.subplots(5, 5, figsize=(20, 20))
    fig.suptitle('Predictions', fontsize=20)
    for i in range(5):
        for j in range(5):
            image = predictions[prediction_index]["image"][i+5*j][:, :, :3]
            image = (image - image.min()) / (image.max() - image.min())  # Normalize image
            ax[i, j].imshow(image)

            label = predictions[prediction_index]['label'][i+5*j]
            rgbd_prediction = predictions[prediction_index]['rgbd_prediction'][i+5*j]
            rgb_prediction = predictions[prediction_index]['rgb_prediction'][i+5*j]
            ax[i, j].set_title(f"Label: {label}\nRGBD Prediction: {rgbd_prediction}\nRGB Prediction: {rgb_prediction}")
            ax[i, j].axis('off')

    plt.show()