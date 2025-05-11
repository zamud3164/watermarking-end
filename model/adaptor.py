import torch
import torch.nn as nn
import torch.nn.functional as F


class Adaptor(nn.Module):
    def __init__(self, message_length: int, channel_count: int = 64):
        super(Adaptor, self).__init__()

        # Message feature extraction (reshape and project)
        self.message_fc = nn.Linear(message_length, channel_count)
        self.message_conv = nn.Conv2d(channel_count, channel_count, kernel_size=3, stride=1, padding=1)
        
        # Cover image feature extraction (convolution + SE block)
        self.cover_conv = nn.Conv2d(3, channel_count, kernel_size=3, stride=1, padding=1)
        
        # Fusion layer
        self.fusion_conv = nn.Conv2d(channel_count * 2, channel_count, kernel_size=3, stride=1, padding=1)
        
        # Down-sampling to produce the strength factor (final output)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channel_count, 1)

    def forward(self, image, message):

        message = message.view(message.size(0), -1) 

        # Process the message
        message_features = F.relu(self.message_fc(message))
        message_features = message_features.view(message_features.size(0), -1, 1, 1)  # [batch_size, C, 1, 1]
        message_features = self.message_conv(message_features)  # Process the reshaped message

        # Process the cover image
        cover_features = self.cover_conv(image)  # [batch_size, C, H, W]

        # Upscale message features to match cover features' spatial dimensions
        message_features = F.interpolate(message_features, size=cover_features.shape[2:], mode='bilinear', align_corners=False)

        # Fuse the message and cover features
        fused_features = torch.cat([message_features, cover_features], dim=1)  # Concatenate along channels
        fused_features = self.fusion_conv(fused_features)  # Apply fusion conv layer

        # Down-sample to get strength factor
        pooled_features = self.pool(fused_features).view(fused_features.size(0), -1)
        strength_factor = self.fc(pooled_features)  # Output the strength factor
        
        # Output is between 0 and 1 (sigmoid)
        return torch.sigmoid(strength_factor)
