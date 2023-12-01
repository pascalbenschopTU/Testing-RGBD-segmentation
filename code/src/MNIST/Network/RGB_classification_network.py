import torch
import torch.nn as nn

# Define the neural network architecture
class RGBClassifier(nn.Module):
    def __init__(self):
        super(RGBClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = x.reshape(-1, 16 * 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        



