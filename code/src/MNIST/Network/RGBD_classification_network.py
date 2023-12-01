from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from Network.conv_2_5d import Conv2_5D_Depth, Malleable_Conv2_5D_Depth

class DepthAwareConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(DepthAwareConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_size_prod = kernel_size * kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        # Initialize the weights
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))


    def forward(self, x, depth, camera_params):
        # Convolve the RGB image using weight
        N, C, H, W = x.shape

        out_H = (H + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        out_W = (W + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

        x_unfold = F.unfold(x, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        x_unfold = x_unfold.view(N, C, self.kernel_size_prod, out_H * out_W)
        depth_unfold = F.unfold(depth, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        depth_unfold = depth_unfold.view(N, 1, self.kernel_size_prod, out_H * out_W)

        output = torch.matmul(self.weight.view(-1, C * self.kernel_size_prod), (x_unfold * depth_unfold).view(N, C * self.kernel_size_prod, out_H * out_W))
        output = output.view(N, self.out_channels, out_H, out_W)

        # # Plot output of all channels in one plot (16)
        # fig, ax = plt.subplots(5, 4, figsize=(20, 20))
        # for i in range(4):
        #     for j in range(4):
        #         ax[i, j].imshow(output[0, i*4+j, :, :].detach().cpu().numpy())
        #         ax[i, j].set_title(f"Channel {i*4+j}")
        #         ax[i, j].axis('off')
                
        # ax[4, 0].imshow(x[0, :3, :, :].permute(1, 2, 0).detach().cpu().numpy())
        # ax[4, 0].set_title(f"RGB")
        # ax[4, 0].axis('off')
        # ax[4, 1].imshow(depth[0, :, :, :].permute(1, 2, 0).detach().cpu().numpy(), cmap="gray")
        # ax[4, 1].set_title(f"Depth")
        # ax[4, 1].axis('off')
        # plt.show()
        
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)

        return output
        

class RGBDClassifier(nn.Module):
    def __init__(self, image_width=28, image_height=28):
        super(RGBDClassifier, self).__init__()
        self.image_width = image_width
        self.image_height = image_height
        
        # Initialize the layers and weights
        self.conv1 = DepthAwareConv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # nn.init.xavier_uniform_(self.conv1.weight_0)
        # nn.init.xavier_uniform_(self.conv1.weight_1)
        # nn.init.xavier_uniform_(self.conv1.weight_2)
        # nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)

        self.conv2 = DepthAwareConv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        # nn.init.xavier_uniform_(self.conv2.weight_0)
        # nn.init.xavier_uniform_(self.conv2.weight_1)
        # nn.init.xavier_uniform_(self.conv2.weight_2)
        nn.init.xavier_uniform_(self.conv2.weight)
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