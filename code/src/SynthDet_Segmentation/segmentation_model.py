import torch
import torchvision.transforms.functional
from torch import nn

# Adapted from https://nn.labml.ai/unet/index.html


class DoubleConvolution(nn.Module):
    """
    ### Two $3 \times 3$ Convolution Layers

    Each step in the contraction path and expansive path have two $3 \times 3$
    convolutional layers followed by ReLU activations.

    In the U-Net paper they used $0$ padding,
    but we use $1$ padding so that final feature map is not cropped.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        :param in_channels: is the number of input channels
        :param out_channels: is the number of output channels
        """
        super().__init__()

        # First $3 \times 3$ convolutional layer
        padding = 1
        self.first = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        # Second $3 \times 3$ convolutional layer
        self.second = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # Apply the two convolution layers and activations
        x = self.first(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.second(x)
        x = self.bn2(x)
        return self.act2(x)


class DownSample(nn.Module):
    """
    ### Down-sample

    Each step in the contracting path down-samples the feature map with
    a $2 \times 2$ max pooling layer.
    """

    def __init__(self):
        super().__init__()
        # Max pooling layer
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        return self.pool(x)


class UpSample(nn.Module):
    """
    ### Up-sample

    Each step in the expansive path up-samples the feature map with
    a $2 \times 2$ up-convolution.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Up-convolution
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.up(x)


class CropAndConcat(nn.Module):
    """
    ### Crop and Concatenate the feature map

    At every step in the expansive path the corresponding feature map from the contracting path
    concatenated with the current feature map.
    """

    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        """
        :param x: current feature map in the expansive path
        :param contracting_x: corresponding feature map from the contracting path
        """

        # Crop the feature map from the contracting path to the size of the current feature map
        contracting_x = torchvision.transforms.functional.center_crop(
            contracting_x, [x.shape[2], x.shape[3]])
        # Concatenate the feature maps
        x = torch.cat([x, contracting_x], dim=1)
        #
        return x


class SmallUNet(nn.Module):
    """
    ## U-Net
    """

    def __init__(self, in_channels: int, out_channels: int, criterion=nn.CrossEntropyLoss(reduction='mean')):
        """
        :param in_channels: number of channels in the input image
        :param out_channels: number of channels in the result feature map
        """
        super(SmallUNet, self).__init__()

        # Double convolution layers for the contracting path.
        # The number of features gets doubled at each step starting from $64$.
        self.down_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in
                                        [(in_channels, 32), (32, 64), (64, 128), (128, 256), (256, 512)]])
        # Batch normalization layers for the contracting path
        self.bn_down = nn.ModuleList([nn.BatchNorm2d(o) for o in [32, 64, 128, 256, 512]])
        # Down sampling layers for the contracting path
        self.down_sample = nn.ModuleList([DownSample() for _ in range(5)])

        # The two convolution layers at the lowest resolution (the bottom of the U).
        self.middle_conv = DoubleConvolution(512, 1024)

        # Up sampling layers for the expansive path.
        # The number of features is halved with up-sampling.
        self.up_sample = nn.ModuleList([UpSample(i, o) for i, o in
                                        [(1024, 512), (512, 256), (256, 128), (128, 64), (64, 32)]])
        # Double convolution layers for the expansive path.
        # Their input is the concatenation of the current feature map and the feature map from the
        # contracting path. Therefore, the number of input features is double the number of features
        # from up-sampling.
        self.up_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in
                                      [(1024, 512), (512, 256), (256, 128), (128, 64), (64, 32)]])
        # Batch normalization layers for the expansive path
        self.bn_up = nn.ModuleList([nn.BatchNorm2d(o) for o in [512, 256, 128, 64, 32]])
        # Crop and concatenate layers for the expansive path.
        self.concat = nn.ModuleList([CropAndConcat() for _ in range(5)])
        # Final $1 \times 1$ convolution layer to produce the output
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

        self.criterion = criterion


    def encode_decode(self, rgb: torch.Tensor, modal_x: torch.Tensor=None):
        """
        :param x: input image
        """
        if modal_x is not None:
             # In DFormer: x_e=x_e[:,0,:,:].unsqueeze(1)
            depth = modal_x[:, 0, :, :].unsqueeze(1)
            # Here just append the depth map to rgb
            rgb = torch.cat([rgb, depth], dim=1)

        # To collect the outputs of contracting path for later concatenation with the expansive path.
        pass_through = []
        # Contracting path
        for i in range(len(self.down_conv)):
            # Two $3 \times 3$ convolutional layers
            rgb = self.down_conv[i](rgb)
            rgb = self.bn_down[i](rgb)
            # Collect the output
            pass_through.append(rgb)
            # Down-sample
            rgb = self.down_sample[i](rgb)

        # Two $3 \times 3$ convolutional layers at the bottom of the U-Net
        rgb = self.middle_conv(rgb)

        # Expansive path
        for i in range(len(self.up_conv)):
            # Up-sample
            rgb = self.up_sample[i](rgb)
            # Concatenate the output of the contracting path
            rgb = self.concat[i](rgb, pass_through.pop())
            # Two $3 \times 3$ convolutional layers
            rgb = self.up_conv[i](rgb)
            rgb = self.bn_up[i](rgb)

        # Final $1 \times 1$ convolution layer
        out = self.final_conv(rgb)

        return out

    def forward(self, rgb: torch.Tensor, modal_x: torch.Tensor=None, label: torch.Tensor=None):
        """
        :param x: input image
        """
        # To collect the outputs of contracting path for later concatenation with the expansive path.
        out = self.encode_decode(rgb, modal_x)

        if label is not None:
            loss = self.criterion(out, label.long())
            return loss
        
        return out