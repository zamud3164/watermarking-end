import torch
import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu
from model.adaptor import Adaptor
from model.se_block import SEBlock


class Encoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, config: HiDDenConfiguration, use_adaptor=False):
        super(Encoder, self).__init__()
        self.use_adaptor = config.use_adaptor
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks

        # Adaptor (only enabled if `use_adaptor` is True)
        if self.use_adaptor:
            self.adaptor = Adaptor(config.message_length)

        # First convolutional layers (extracting features from image)
        layers = [ConvBNRelu(3, self.conv_channels)]
        for _ in range(config.encoder_blocks - 1):
            layers.append(ConvBNRelu(self.conv_channels, self.conv_channels))
            layers.append(SEBlock(self.conv_channels)) 
        
        self.conv_layers = nn.Sequential(*layers)

        self.after_concat_layer = ConvBNRelu(
            #self.conv_channels + 3 + 60,
            30 + 30 + 3,  
            self.conv_channels
        )

        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)
        self.alpha = 1

    def forward(self, image, combined_input, message):

        # Optionally use the adaptor to get the strength factor (if `use_adaptor` is True)
        if self.use_adaptor and message is not None:
          strength_factor = self.adaptor(image, message)
          self.alpha = strength_factor.mean().item()  # Use the output from the adaptor
          #self.alpha = 1

        """
        # Pass the cover image through convolution layers
        encoded_image = self.conv_layers(image)  # [batch, encoder_channels, 128, 128]

        combined_input = combined_input.expand(-1, -1, 128, 128)
        
        # Concatenate latent features + message with the encoded image and cover image
        concat = torch.cat([combined_input, encoded_image, image], dim=1)  # [batch, channels, 128, 128]
        """

        B, _, H, W = image.shape

        # Expand latent features
        combined_input = combined_input.expand(-1, -1, H, W)  # [B, latent_channels, 128, 128]

        # Expand message
        message = message.expand(-1, -1, H, W)  # [B, message_channels, 128, 128]

        concat = torch.cat([message, combined_input, image], dim=1)

        # THIS IS BACK TO ORIGINAL CODE
        im_w = self.after_concat_layer(concat)  # Further processing
        
        residual = self.final_layer(im_w)  # Residual output
        
        # Apply strength factor α (Jia et al. used α=1)
        encoded_image = image + self.alpha * residual
        
        return encoded_image
    
    def get_strength_factor(self):
        """
        This method returns the strength factor.
        """
        return self.alpha if hasattr(self, 'alpha') else None