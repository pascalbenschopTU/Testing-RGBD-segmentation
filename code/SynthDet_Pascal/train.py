import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50
from torch.utils.tensorboard import SummaryWriter
import os
from CocoDepthDataset import CocoDepthDataset
from model.RGBDModel import RGBDModel
from PIL import Image


def train_model(dataset_root, log_directory, saved_model_path):
    # Define a device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Define the model, criterion, and optimizer
    num_classes = 64  # Adjust based on the number of classes in your segmentation task
    # model = RGBDModel(num_classes)

    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 3 classes
    model= model.to(device)


    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.NEAREST),  # Adjust the size based on your model's input size
        transforms.ToTensor(),
    ])

    # Define paths to your COCO dataset
    dataset = CocoDepthDataset(os.path.join(dataset_root, 'images'), os.path.join(dataset_root, 'semantic.json'), os.path.join(dataset_root, 'depth'), transform=transform)

    # Split dataset into training and validation sets
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

    # # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Tensorboard for visualization
    writer = SummaryWriter(log_directory)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (rgb_data, depth_data, target) in enumerate(train_loader):
            rgb_data, depth_data, target = rgb_data.to(device), depth_data.to(device), target.to(device)

            # Forward pass
            output = model(rgb_data)['out']

            target = target.squeeze(1).long()

            loss = criterion(output, target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Average training loss for the epoch
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {avg_loss:.4f}')

        # Log training loss to Tensorboard
        writer.add_scalar('Train/Loss', avg_loss, epoch)

        # Validation loop
        model.eval()
        with torch.no_grad():
            total_val_loss = 0.0

            for batch_idx, (rgb_data, depth_data, target) in enumerate(test_loader):
                rgb_data, depth_data, target = rgb_data.to(device), depth_data.to(device), target.to(device)

                # Forward pass
                output = model(rgb_data)['out']

                target = target.squeeze(1).long()

                loss = criterion(output, target)

                total_val_loss += loss.item()

            # Average validation loss for the epoch
            avg_val_loss = total_val_loss / len(test_loader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Average Validation Loss: {avg_val_loss:.4f}')

            # Log validation loss to Tensorboard
            writer.add_scalar('Val/Loss', avg_val_loss, epoch)

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(saved_model_path, 'model.pth'))

    # Close Tensorboard writer
    writer.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train a model for RGB-D semantic segmentation')
    arg_parser.add_argument('--dataset_root', type=str, help='Path to the root directory of the dataset')
    arg_parser.add_argument('--log_directory', type=str, help='Path to the directory for logging')
    arg_parser.add_argument('--saved_model_path', type=str, help='Path to save the trained model')
    args = arg_parser.parse_args()

    train_model(args.dataset_root, args.log_directory, args.saved_model_path)