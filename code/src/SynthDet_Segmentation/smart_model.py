import cv2
import torch
import torch.nn as nn
from skimage.filters import threshold_multiotsu
import numpy as np
from torchvision.transforms.functional import gaussian_blur

from segmentation_model import DoubleConvolution, DownSample, UpSample, CropAndConcat

class SmallUNet(nn.Module):
    """
    ### Small U-Net

    The U-Net architecture consists of a contracting path and an expansive path.
    The contracting path follows the typical architecture of a convolutional neural network.
    It consists of the repeated application of two $3 \times 3$ convolutions (unpadded convolutions),
    each followed by a rectified linear unit (ReLU) and a $2 \times 2$ max pooling operation with stride 2 for down-sampling.

    Every step in the expansive path consists of an up-convolution of the feature map followed by a concatenation with the correspondingly cropped feature map from the contracting path,
    and two $3 \times 3$ convolutions, each followed by a ReLU.

    At the final layer a $1 \times 1$ convolution is used to map each 64-component feature vector to the desired number of classes.
    """

    def __init__(self, in_channels: int, out_channels: int, criterion=nn.CrossEntropyLoss(reduction='mean')):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        """
        super().__init__()
        self.in_channels = in_channels

        # Double convolution layers for the contracting path.
        # The number of features gets doubled at each step starting from $64$.
        self.down_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in
                                        [(in_channels, 32), (32, 64), (64, 128)]])
        # Batch normalization layers for the contracting path
        self.bn_down = nn.ModuleList([nn.BatchNorm2d(o) for o in [32, 64, 128]])
        # Down sampling layers for the contracting path
        self.down_sample = nn.ModuleList([DownSample() for _ in range(3)])

        self.middle_conv = DoubleConvolution(128, 256)

        # Up sampling layers for the expansive path.
        # The number of features is halved with up-sampling.
        self.up_sample = nn.ModuleList([UpSample(i, o) for i, o in
                                        [(256, 128), (128, 64), (64, 32)]])
        # Double convolution layers for the expansive path.
        # Their input is the concatenation of the current feature map and the feature map from the
        # contracting path. Therefore, the number of input features is double the number of features
        # from up-sampling.
        self.up_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in
                                      [(256, 128), (128, 64), (64, 32)]])
        # Batch normalization layers for the expansive path
        self.bn_up = nn.ModuleList([nn.BatchNorm2d(o) for o in [128, 64, 32]])
        # Crop and concatenate layers for the expansive path.
        self.concat = nn.ModuleList([CropAndConcat() for _ in range(3)])
        # Final $1 \times 1$ convolution layer to produce the output
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

        self.criterion = criterion

    def encode_decode(self, x: torch.Tensor):
        """
        :param x: input image
        """
        # Contracting path
        pass_through = []
        for i in range(len(self.down_conv)):
            x = self.down_conv[i](x)
            x = self.bn_down[i](x)
            pass_through.append(x)
            x = self.down_sample[i](x)

        x = self.middle_conv(x)

        # Expansive path
        for i in range(len(self.up_sample)):
            x = self.up_sample[i](x)
            x = self.concat[i](x, pass_through.pop())
            x = self.up_conv[i](x)
            x = self.bn_up[i](x)

        # Output layer
        return self.final_conv(x)

    def forward(self, x: torch.Tensor, modal_x: torch.Tensor=None, label: torch.Tensor=None):
        """
        :param x: input tensor
        :param depth: depth tensor
        :return: output tensor
        """
        if modal_x is not None:
            # Here just append the depth map to rgb
            depth = modal_x[:, 0, :, :].unsqueeze(1)
            x = torch.cat([x, depth], dim=1)

        output = self.encode_decode(x)

        if label is not None:
            loss = self.criterion(output, label.long())
            return loss
        else:
            return output
    

class SmartPeripheralRGBDModel(nn.Module):
    def __init__(self, in_channels, out_channels, criterion=nn.CrossEntropyLoss(reduction='mean')):
        super(SmartPeripheralRGBDModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.criterion = criterion

        self.model = SmallUNet(in_channels, out_channels, criterion)

        self.depth_layer_weights = nn.Parameter(torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], dtype=torch.float32))

    def create_masks(self, depth, num_classes=4):
        B, C, H, W = depth.shape
        
        # Initialize masks tensor
        masks = torch.zeros((B, num_classes, H, W), dtype=torch.bool)

        for batch in range(B):
            current_depth = depth[batch, 0]
            thresholds = threshold_multiotsu(current_depth.cpu().numpy(), classes=num_classes)

            prev_threshold = torch.min(current_depth) - 0.1
            for i, threshold in enumerate(thresholds):
                mask = (current_depth < threshold) & (current_depth >= prev_threshold)
                prev_threshold = threshold
                masks[batch, i, :, :] = mask

            masks[batch, -1, :, :] = (current_depth >= prev_threshold)

        return masks
        
    def forward(self, rgb, depth, label=None, plot=False):
        B, C, H, W = depth.shape

        with torch.no_grad():
            masks = self.create_masks(depth, num_classes=4)
            masks = masks.to(rgb.device)

        output = torch.zeros(B, self.out_channels , H, W).to(rgb.device)

        individual_losses = []
        for m in range(masks.shape[1]):
            expanded_masks = masks[:, m].unsqueeze(1)  # Expand dimensions to match the number of channels in rgb
            soft_masked_image = (expanded_masks * rgb) + (~expanded_masks * gaussian_blur(rgb, kernel_size=25, sigma=15.0))

            out = self.model(soft_masked_image, depth)
            output += out * self.depth_layer_weights[m]

            if label is not None:
                masked_output = out * expanded_masks
                masked_label = label * expanded_masks.squeeze(1)

                if plot:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(1, 5, figsize=(20, 5))
                    ax[0].imshow(rgb[0].permute(1, 2, 0).cpu().numpy())
                    ax[1].imshow(soft_masked_image[0].permute(1, 2, 0).cpu().numpy())
                    ax[2].imshow(out[0].argmax(0).cpu().numpy())
                    ax[3].imshow(masked_output[0].argmax(0).cpu().numpy())
                    ax[4].imshow(masked_label[0].cpu().numpy())
                    plt.show()

                individual_loss = self.model.criterion(masked_output, masked_label.long())
                individual_losses.append(individual_loss)
        
        if label is not None:
            loss = self.model.criterion(output, label.long())
            individual_losses.append(loss)
            final_loss = sum(individual_losses)
            return final_loss
        else:
            return output
        