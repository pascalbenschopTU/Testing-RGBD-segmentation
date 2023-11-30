import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self):
        super(RGBDClassifier, self).__init__()
        self.conv1 = DepthAwareConv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x_rgb = x[:, :3, :, :].reshape(-1, 3, 28, 28)
        x_depth = x[:, 3, :, :].reshape(-1, 1, 28, 28)
        x = F.relu(self.conv1(x_rgb, x_depth))
        x_rgb = x[:, :3, :, :].reshape(-1, 3, 28, 28)
        x = self.pool1(F.relu(self.conv1(x_rgb, x_depth)))
        x = x.reshape(-1, 16 * 14 * 14)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)