import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import numpy as np
import cv2


class MedianBlur(nn.Module):
    """
    Applies median blur to the noised image using OpenCV. Kernel size must be odd (e.g., 3, 5, 7).
    """
    def __init__(self, kernel_size=3):
        super(MedianBlur, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.kernel_size = kernel_size

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]
        cover_image = noised_and_cover[1]

        # Convert to NumPy for OpenCV medianBlur (since PyTorch doesn't support median filter natively)
        # shape: [B, C, H, W]
        noised_blurred = []

        for img in noised_image:
            channels = []
            for c in img:
                np_img = c.detach().cpu().numpy()
                blurred = cv2.medianBlur(np_img, self.kernel_size)
                channels.append(torch.tensor(blurred, device=noised_image.device))
            noised_blurred.append(torch.stack(channels))
        
        noised_blurred = torch.stack(noised_blurred)

        return [noised_blurred, cover_image]
