import cv2
import torch
import torch.nn as nn
from skimage.filters import threshold_multiotsu
import numpy as np
from torchvision.transforms.functional import gaussian_blur
from torchvision.utils import make_grid

from segmentation_model import DoubleConvolution, DownSample, UpSample, CropAndConcat

class TensorboardSummary(object):
    def __init__(self, writer):
        self.writer = writer
    
    def decode_seg_map_sequence(self, label_masks, config=None):
        rgb_masks = []
        for label_mask in label_masks:
            rgb_mask = self.decode_segmap(label_mask, config=config)
            rgb_masks.append(rgb_mask)
        rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
        return rgb_masks
    
    def decode_segmap(self, label_mask, config=None):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
            the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
            in a figure.
        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        n_classes = config.num_classes
        np.random.seed(0)  # Set a fixed seed value
        label_colours = np.random.randint(0, 256, size=(config.num_classes, 3))

        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
 
        return rgb

    def visualize_image(self, images, depth, target, output, global_step, config=None):
        grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
        self.writer.add_image('Image', grid_image, global_step)
        grid_image = make_grid(depth[:3].clone().cpu().data, 3, normalize=True)
        self.writer.add_image('Depth', grid_image, global_step)
        grid_image = make_grid(self.decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                    config=config), 3, normalize=False, value_range=(0, 255))
        self.writer.add_image('Seg_Predicted_label', grid_image, global_step)
        grid_image = make_grid(self.decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                    config=config), 3, normalize=False, value_range=(0, 255))
        self.writer.add_image('Seg_Groundtruth_label', grid_image, global_step)

        grid_image = make_grid((target[:3] == output[:3].argmax(1)).unsqueeze(1).expand(-1, 3, -1, -1), 3, normalize=False, value_range=(0, 1), scale_each=True)
        self.writer.add_image('Correct', grid_image, global_step)

        softmax_output = torch.nn.functional.softmax(output, dim=1)
        grid_image = make_grid(torch.std(softmax_output[:3], dim=1).unsqueeze(1), 3, normalize=True, scale_each=True)
        self.writer.add_image('Softmax_Std', grid_image, global_step)

        grid_image = make_grid(torch.max(softmax_output[:3], dim=1)[0].unsqueeze(1), 3, normalize=True, scale_each=True)
        self.writer.add_image('Softmax_Max', grid_image, global_step)

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
        B, C, H, W = rgb.shape
        num_classes = 3

        with torch.no_grad():
            masks = self.create_masks(depth, num_classes=4)
            masks = masks.to(rgb.device)

        individual_losses = []
        for m in range(num_classes):
            expanded_masks = masks[:, m].unsqueeze(1)  # Expand dimensions to match the number of channels in rgb
            rgb *= expanded_masks
            rgb += (~expanded_masks * gaussian_blur(rgb, kernel_size=25, sigma=15.0))

            out = self.model(rgb, depth)
            mask = masks[:, m].unsqueeze(1)
            if plot:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(2, 5, figsize=(20, 5))
                ax[0].imshow(rgb[0].permute(1, 2, 0).cpu().numpy())
                ax[2].imshow(out[0].argmax(0).cpu().numpy())
                ax[3].imshow((mask[0] * output[0]).argmax(0).cpu().numpy())
                ax[4].imshow(label[0].cpu().numpy())
                plt.show()
            if m == 0:
                output = out * self.depth_layer_weights[m]
                if label is not None:
                    individual_losses.append(self.model.criterion(output * mask, label.long()))
            else:
                output += out * self.depth_layer_weights[m]
                if label is not None:
                    individual_losses.append(self.model.criterion(out * self.depth_layer_weights[m] * mask, label.long()))

        if label is not None:
            combined_loss = self.model.criterion(output, label.long())

            return sum(individual_losses) + combined_loss
        else:
            return output
        

class SmartDepthModel(nn.Module):
    def __init__(self, in_channels, out_channels, criterion=nn.CrossEntropyLoss(reduction='mean'), config=None, writer=None):
        super(SmartDepthModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.criterion = criterion
        self.config=config
        self.tb_summary = TensorboardSummary(writer)

        # self.model = SmallUNet(in_channels, out_channels, criterion)
        self.backbone_depth = DepthEncoder(1, 64)
        self.decoder = Decoder(64, out_channels)

    # Normalize image data to range [0, 1]
    def normalize_image(self, image):
        # Normalize to range [0, 1]
        image_min = np.min(image)
        image_max = np.max(image)
        normalized_image = (image - image_min) / (image_max - image_min)
        return normalized_image

    def forward(self, rgb, modal_x, label=None, plot=False, epoch=0):
        depth = modal_x[:, 0, :, :].unsqueeze(1)
        features = self.backbone_depth(depth)
        output = self.decoder(features)

        if plot:
            with torch.no_grad():
                # tb.add_image('Softmax Outputs', softmax_outputs, epoch)
                self.tb_summary.visualize_image(rgb, depth, label, output, epoch, self.config)

            
        if label is not None:
            loss = self.criterion(output, label.long())
            return loss
        else:
            return output 


class DepthEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.small_encoder = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            InceptionBlock(in_channels, 16),
            nn.MaxPool2d(kernel_size=2),
            InceptionBlock(16, 32),
        )

        self.middle_conv = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        enc = self.small_encoder(x)

        return self.middle_conv(enc)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        assert out_channels % 4 == 0, "out_channels must be divisible by 4"
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, int(out_channels / 4), kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, int(out_channels / 4), kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, int(out_channels / 2), kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.relu(out)
        return out
        

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 64, kernel_size=2, stride=2),
            InceptionBlock(in_channels, 64),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2),
            InceptionBlock(64, out_channels),
        )

    def forward(self, x):
        return self.decoder(x)