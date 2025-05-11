import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu
from model.se_block import SEBlock
import torch


class Decoder(nn.Module):
    def __init__(self, config: HiDDenConfiguration):
        super(Decoder, self).__init__()
        self.channels = config.decoder_channels

        # First Conv-BN-ReLU block
        layers = [ConvBNRelu(3, self.channels)]

        # Three SE-blocks for downsampling
        for _ in range(3):
            layers.append(SEBlock(self.channels))

        # Additional Conv-BN-ReLU blocks
        layers.append(ConvBNRelu(self.channels, self.channels))
        layers.append(ConvBNRelu(self.channels, config.message_length))

        # Final SE-block and Linear layer
        layers.append(SEBlock(config.message_length))
        
        # Adaptive Pooling to reduce spatial dimensions
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        self.layers = nn.Sequential(*layers)

        # CHECK IF IT WORKS
        #self.dropout = nn.Dropout(0.3)  # Drop 30% of activations

        # Fully connected Linear layer for final message extraction
        self.linear = nn.Linear(config.message_length, config.message_length)


    def forward(self, image_with_wm):
        x = self.layers(image_with_wm)

        # Squeeze spatial dimensions (keeping batch and channels)
        x = x.squeeze(3).squeeze(2)  # Shape: [B, C]

        # Flatten spatial dimensions for Linear layer
        #x = x.view(x.shape[0], -1)  # Shape: [B, C * H * W] (no Adaptive Pooling)

        #x = self.dropout(x)

        # Fully connected transformation to final message
        x = self.linear(x)
        return x