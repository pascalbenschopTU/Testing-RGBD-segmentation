import torch
import torch.nn as nn
import torchvision.models as models

class RGBDModel(nn.Module):
    def __init__(self, num_classes):
        super(RGBDModel, self).__init__()

        # Backbone for RGB data (you can replace this with a different architecture)
        self.rgb_backbone = models.resnet18(pretrained=True)
        
        # Adjust the input channels of the first convolution layer for depth
        self.rgb_backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Backbone for depth data (you can replace this with a different architecture)
        self.depth_backbone = models.resnet18(pretrained=True)
        
        # Adjust the input channels of the first convolution layer for RGB
        self.depth_backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Fusion layer
        self.fusion_layer = nn.Conv2d(512 * 2, 256, kernel_size=1)

        # Segmentation head
        self.segmentation_head = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, rgb, depth):
        print(rgb.shape, depth.shape)

        # Forward pass for RGB data
        rgb_features = self.rgb_backbone(rgb)
        
        # Forward pass for depth data
        depth_features = self.depth_backbone(depth)

        print(rgb_features.shape, depth_features.shape)

        # Concatenate RGB and depth features
        fused_features = torch.cat((rgb_features, depth_features), dim=1)

        # Apply fusion layer
        fused_features = self.fusion_layer(fused_features)

        # Apply segmentation head
        segmentation_output = self.segmentation_head(fused_features)

        return segmentation_output