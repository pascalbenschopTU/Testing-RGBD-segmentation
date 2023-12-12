from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from Network.conv_2_5d import Conv2_5D_Depth, Malleable_Conv2_5D_Depth
        

class SmartRGBDClassifier(nn.Module):
    def __init__(self, image_width=28, image_height=28):
        super(SmartRGBDClassifier, self).__init__()
        self.image_width = image_width
        self.image_height = image_height

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = Malleable_Conv2_5D_Depth(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)  
        nn.init.xavier_uniform_(self.conv1.weight_0)
        nn.init.xavier_uniform_(self.conv1.weight_1)
        nn.init.xavier_uniform_(self.conv1.weight_2)
        nn.init.zeros_(self.conv1.bias)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = Malleable_Conv2_5D_Depth(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv2.weight_0)
        nn.init.xavier_uniform_(self.conv2.weight_1)
        nn.init.xavier_uniform_(self.conv2.weight_2)
        nn.init.zeros_(self.conv2.bias)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = Malleable_Conv2_5D_Depth(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv3.weight_0)
        nn.init.xavier_uniform_(self.conv3.weight_1)
        nn.init.xavier_uniform_(self.conv3.weight_2)
        nn.init.zeros_(self.conv3.bias)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(128 * (image_width // (2**3)) * (image_height // (2**3)), 128)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        
        self.fc2 = nn.Linear(128, 10)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, depth, camera_params):
        x = self.pool1(F.relu(self.conv1(x, depth, camera_params)))
        depth = self.pool1(depth)
        x = self.pool2(F.relu(self.conv2(x, depth, camera_params)))
        depth = self.pool2(depth)
        x = self.pool3(F.relu(self.conv3(x, depth, camera_params)))

        # Register the hook only if x requires gradient
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)

        return x
    
    # Method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # Method for the activation exctraction
    def get_activations(self, x):
        x = x.permute(0, 3, 1, 2)
        x_rgb = x[:, :3, :, :].reshape(-1, 3, self.image_width, self.image_height)
        x_depth = x[:, 3, :, :].reshape(-1, 1, self.image_width, self.image_height)
        return F.relu(self.conv1(x_rgb, x_depth))