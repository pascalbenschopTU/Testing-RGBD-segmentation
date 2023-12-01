from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from Network.conv_2_5d import Conv2_5D_Depth

class DepthAwareConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthAwareConv2d, self).__init__()
        
        # RGB convolution
        self.rgb_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        
        # Depth convolution
        self.depth_conv = nn.Conv2d(1, out_channels, kernel_size, stride=stride, padding=padding)
        
    def forward(self, rgb, depth):
        # RGB convolution
        rgb_output = self.rgb_conv(rgb)
        
        # Depth convolution
        depth_output = self.depth_conv(depth)
        
        # Calculate soft mask based on depth difference from the center
        center_depth = depth[:, :, depth.size(2) // 2, depth.size(3) // 2].unsqueeze(2).unsqueeze(3)
        depth_diff = torch.abs(depth - center_depth)
        depth_mask = torch.exp(-depth_diff)  # Soft mask based on depth difference

        # Combine RGB and depth features using the soft mask
        combined_output = rgb_output * depth_mask + depth_output * (1 - depth_mask)
        
        return combined_output
    

class RGBDClassifier(nn.Module):
    def __init__(self, image_width=28, image_height=28):
        super(RGBDClassifier, self).__init__()
        self.image_width = image_width
        self.image_height = image_height
        
        # Initialize the weights
        self.conv1 = Conv2_5D_Depth(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight_0)
        nn.init.xavier_uniform_(self.conv1.weight_1)
        nn.init.xavier_uniform_(self.conv1.weight_2)
        nn.init.zeros_(self.conv1.bias)

        self.conv2 = Conv2_5D_Depth(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv2.weight_0)
        nn.init.xavier_uniform_(self.conv2.weight_1)
        nn.init.xavier_uniform_(self.conv2.weight_2)
        nn.init.zeros_(self.conv2.bias)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * self.image_width // 2 * self.image_height // 2, 128)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        self.fc2 = nn.Linear(128, 10)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, depth, camera_params):
        x = F.relu(self.conv1(x, depth, camera_params))

        # Re use x_depth and use x as x_rgb
        x = self.pool1(F.relu(self.conv2(x, depth, camera_params)))

        # Register the hook only if x requires gradient
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)

        x = x.reshape(-1, 16 * self.image_width // 2 * self.image_height // 2)
        x = self.fc1(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
    
    # Method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # Method for the activation exctraction
    def get_activations(self, x):
        x = x.permute(0, 3, 1, 2)
        x_rgb = x[:, :3, :, :].reshape(-1, 3, self.image_width, self.image_height)
        x_depth = x[:, 3, :, :].reshape(-1, 1, self.image_width, self.image_height)
        return F.relu(self.conv1(x_rgb, x_depth))