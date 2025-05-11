import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu
from model.se_block import SEBlock

class FeatureExtractor(nn.Module):
    def __init__(self, config: HiDDenConfiguration):
        super(FeatureExtractor, self).__init__()
        
        self.conv1 = ConvBNRelu(3, 32)
        self.se1 = SEBlock(32)  # First SE block after initial conv
        
        self.conv2 = ConvBNRelu(32, 64)
        self.se2 = SEBlock(64)  # Second SE block
        
        self.conv3 = ConvBNRelu(64, 30)
        self.se3 = SEBlock(30)  # Third SE block
        
        self.final_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, image):
        x = self.conv1(image)
        x = self.se1(x)  # Apply SE block
        x = self.conv2(x)
        x = self.se2(x)  # Apply SE block
        x = self.conv3(x)
        x = self.se3(x)  # Apply SE block
        
        x = self.final_pool(x).squeeze(3).squeeze(2)
        return x
