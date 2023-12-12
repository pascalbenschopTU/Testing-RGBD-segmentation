import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from Network.RGBD_classification_network import RGBDClassifier
from Network.Smart_RGBD_classification_network import SmartRGBDClassifier

class Evaluator:
    def __init__(self, device, test_location, image_width=28, image_height=28):
        self.device = device
        self.test_location = test_location
        self.image_width = image_width
        self.image_height = image_height

    def load_dataset(self):
        # Load the dataset
        train_image_file = self.test_location + "train_images.npz"
        train_label_file = self.test_location + "train_labels.npz"
        test_image_file = self.test_location + "test_images.npz"
        test_label_file = self.test_location + "test_labels.npz"

        train_images = np.load(train_image_file, allow_pickle=True)['arr_0']
        train_labels = np.load(train_label_file, allow_pickle=True)['arr_0']
        test_images = np.load(test_image_file, allow_pickle=True)['arr_0']
        test_labels = np.load(test_label_file, allow_pickle=True)['arr_0']

        self.train_dataset = TensorDataset(
            torch.tensor(train_images, dtype=torch.float32).to(self.device),
            torch.tensor(train_labels, dtype=torch.long).to(self.device)
        )

        self.test_dataset = TensorDataset(
            torch.tensor(test_images, dtype=torch.float32).to(self.device),
            torch.tensor(test_labels, dtype=torch.long).to(self.device)
        )
        self.test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=True)


    def make_camera_params(self):
        camera_params = {}
        camera_params['intrinsic'] = {}
        camera_params['intrinsic']['fx'] = None
        camera_params['intrinsic']['fy'] = None
        camera_params['intrinsic']['cx'] = None
        camera_params['intrinsic']['cy'] = None

        for k1 in camera_params:
            for k2 in camera_params[k1]:
                camera_params[k1][k2] = torch.from_numpy(np.array(camera_params[k1][k2], dtype=np.float32)).float()

        return camera_params

    # Train the model
    def reset_model(self):
        # Create the neural network model
        rgbd_model = RGBDClassifier(image_width=self.image_width, image_height=self.image_height).to(self.device)
        smart_rgbd_model = SmartRGBDClassifier(image_width=self.image_width, image_height=self.image_height).to(self.device)
        rgb_model = RGBDClassifier(image_width=self.image_width, image_height=self.image_height).to(self.device)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()

        rgbd_optimizer = optim.Adam(rgbd_model.parameters(), lr=0.001)
        smart_rgbd_optimizer = optim.Adam(smart_rgbd_model.parameters(), lr=0.001)
        rgb_optimizer = optim.Adam(rgb_model.parameters(), lr=0.001)

        return rgbd_model, smart_rgbd_model, rgb_model, criterion, rgbd_optimizer, smart_rgbd_optimizer, rgb_optimizer

    def evaluate(self, max_data_size=100, verbose=True):
        camera_params = self.make_camera_params()

        rgb_accuracy_list = []
        rgbd_accuracy_list = []
        smart_rgbd_accuracy_list = []

        # Evaluate the model
        predictions = []

        max_data_size = 100

        # Reset the model before each data size iteration
        for data_size in range(1, max_data_size + 1):
            rgbd_model, smart_rgbd_model, rgb_model, criterion, rgbd_optimizer, smart_rgbd_optimizer, rgb_optimizer = self.reset_model()

            train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=False, 
                                    sampler=SubsetRandomSampler(range(int((data_size/max_data_size) * len(self.train_dataset)))))

            # Continue with the rest of the code...
            for batch_idx, (data, targets) in enumerate(train_loader):
                data = data.to(self.device)
                targets = targets.to(self.device)
                rgbd_optimizer.zero_grad()
                # smart_rgbd_optimizer.zero_grad()
                rgb_optimizer.zero_grad()

                x = data[:, :, :, :3].permute(0, 3, 1, 2)
                depth = data[:, :, :, 3:4].permute(0, 3, 1, 2)
                fake_depth = torch.ones_like(depth)

                camera_params['intrinsic']['fx'] = torch.ones((data.shape[0], 1, 1)).to(self.device) * 500
                
                rgbd_outputs = rgbd_model(x, depth, camera_params)
                # smart_rgbd_outputs = smart_rgbd_model(x, depth, camera_params)
                rgb_outputs = rgb_model(x, fake_depth, camera_params)
                
                rgbd_loss = criterion(rgbd_outputs, targets)
                rgbd_loss.backward()
                rgbd_optimizer.step()

                # smart_rgbd_loss = criterion(smart_rgbd_outputs, targets)
                # smart_rgbd_loss.backward()
                # smart_rgbd_optimizer.step()

                rgb_loss = criterion(rgb_outputs, targets)
                rgb_loss.backward()
                rgb_optimizer.step()

                if (batch_idx + 1) % 100 == 0:
                    print(f'Step [{batch_idx+1}/{len(train_loader)}], rgb loss: {rgb_loss.item():.4f}, rgbd loss: {rgbd_loss.item():.4f}')

            rgbd_model.eval()
            # smart_rgbd_model.eval()
            rgb_model.eval()
            with torch.no_grad():
                rgbd_correct = 0
                # smart_rgbd_correct = 0
                rgb_correct = 0
                total = 0
                for data, targets in self.test_loader:
                    data = data.to(self.device)
                    targets = targets.to(self.device)

                    x = data[:, :, :, :3].permute(0, 3, 1, 2)
                    depth = data[:, :, :, 3:4].permute(0, 3, 1, 2)
                    fake_depth = torch.ones_like(depth)

                    camera_params['intrinsic']['fx'] = torch.ones((data.shape[0], 1, 1)).to(self.device) * 500

                    rgbd_outputs = rgbd_model(x, depth, camera_params)
                    # smart_rgbd_outputs = smart_rgbd_model(x, depth, camera_params)
                    rgb_outputs = rgb_model(x, fake_depth, camera_params)

                    _, rgbd_predicted = torch.max(rgbd_outputs.data, 1)
                    # _, smart_rgbd_predicted = torch.max(smart_rgbd_outputs.data, 1)
                    _, rgb_predicted = torch.max(rgb_outputs.data, 1)

                    total += targets.size(0)
                    rgbd_correct += (rgbd_predicted == targets).sum().item()
                    # smart_rgbd_correct += (smart_rgbd_predicted == targets).sum().item()
                    rgb_correct += (rgb_predicted == targets).sum().item()

                    if data_size / max_data_size > 0.9:
                        predictions.append({
                            "image": data.cpu().numpy(),
                            "label": targets.cpu().numpy(),
                            "rgbd_prediction": rgbd_predicted.cpu().numpy(),
                            # "smart_rgbd_prediction": smart_rgbd_predicted.cpu().numpy(),
                            "rgb_prediction": rgb_predicted.cpu().numpy()
                        })

                rgbd_accuracy = 100 * rgbd_correct / total
                # smart_rgbd_accuracy = 100 * smart_rgbd_correct / total
                rgb_accuracy = 100 * rgb_correct / total
                
                if verbose:
                    print(f'Test RGBD Accuracy: {rgbd_accuracy:.2f}%')
                    # print(f'Test Smart RGBD Accuracy: {smart_rgbd_accuracy:.2f}%')
                    print(f'Test RGB Accuracy: {rgb_accuracy:.2f}%')
                    print(f"Data size: {data_size/max_data_size * 100}%")

                rgbd_accuracy_list.append(rgbd_accuracy)
                # smart_rgbd_accuracy_list.append(smart_rgbd_accuracy)
                rgb_accuracy_list.append(rgb_accuracy)

        return rgbd_accuracy_list, smart_rgbd_accuracy_list, rgb_accuracy_list, predictions