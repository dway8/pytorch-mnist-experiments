import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        # 1 channel, arbitrary 6 channels as output, kernel of 5x5 => images of 6x24x24
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        # divides by 2 => images of 6x12x12
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 6 channel, arbitrary 16 channels as output, kernel of 5x5 => images of 16x8x8
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=24, kernel_size=5)
        # after pooling + flattening: images of 16x4x4
        self.fc1 = nn.Linear(24 * 4 * 4, 320)
        self.fc2 = nn.Linear(320, 60)
        self.fc3 = nn.Linear(60, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
