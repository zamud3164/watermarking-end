import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block.
    It learns to recalibrate channel-wise feature responses by using:
    1. Global Average Pooling
    2. Fully Connected Layers with non-linearity
    3. Sigmoid activation
    """
    def __init__(self, channels, reduction_ratio=16):
        super(SEBlock, self).__init__()

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction_ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction_ratio, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        
        # Squeeze operation: Global Average Pooling
        squeezed = self.global_avg_pool(x).view(batch_size, num_channels)

        # Excitation operation: Two fully connected layers with ReLU and Sigmoid
        excitation = self.fc1(squeezed)
        excitation = self.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation)

        # Reshape and apply channel-wise attention
        excitation = excitation.view(batch_size, num_channels, 1, 1)
        return x * excitation  # Scale input feature maps
