

from matplotlib import pyplot as plt
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_gradients(model, data_loader, camera_params):
    model.eval()

    correct = 0
    total = 0
    for data, targets in data_loader:
        data = data.to(device)
        targets = targets.to(device)

        x = data[:, :, :, :3].permute(0, 3, 1, 2)
        depth = data[:, :, :, 3:4].permute(0, 3, 1, 2)
        fake_depth = torch.ones_like(depth)

        camera_params['intrinsic']['fx'] = torch.ones((data.shape[0], 1, 1)).to(device) * 500

        outputs = model(x, depth, camera_params)

        _, predicted = torch.max(outputs.data, 1)

        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        ############################ Grad-CAM ############################
        
        # Get the gradient of the output with respect to the parameters of the model
        outputs.backward(gradient=torch.ones_like(outputs))

        # Pull the gradients out of the model
        gradients = model.get_activations_gradient()

        # Pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # Get the activations of the last convolutional layer
        activations = model.get_activations(x).detach()

        # Weight the channels by corresponding gradients
        for i in range(16):
            activations[:, i, :, :] *= pooled_gradients[i]

        # Average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # ReLU on top of the heatmap
        heatmap = np.maximum(heatmap.cpu(), 0)

        # Normalize the heatmap
        heatmap /= torch.max(heatmap)

        # Draw the heatmap
        plt.matshow(heatmap.squeeze())

        # Draw the heatmap on top of the image
        img_sample = data[0].cpu().numpy()
        img_sample = (img_sample - img_sample.min()) / (img_sample.max() - img_sample.min())
        plt.imshow(img_sample[:, :, :3])
        plt.show()