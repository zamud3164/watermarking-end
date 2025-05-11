import torch.nn as nn

class UpConvBNRelu(nn.Module):
    """
    Upsampling Convolution Block with:
    - Transposed Convolution (for upsampling)
    - Batch Normalization
    - ReLU Activation
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        super(UpConvBNRelu, self).__init__()

        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, output_padding=output_padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
