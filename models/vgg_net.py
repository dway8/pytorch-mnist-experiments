import torchvision
import torch.nn as nn


class VGGNet(nn.Module):
    def __init__(self):
        super().__init__()
        original_vgg = torchvision.models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(original_vgg.features))

        # Modify last layer for MNIST classification (10 classes instead of 1000)
        IN_FEATURES = self.classifier[-1].in_features
        final_layer = nn.Linear(IN_FEATURES, 10)
        self.classifier[-1] = final_layer

    def forward(self, x):
        super().forward(self, x)
