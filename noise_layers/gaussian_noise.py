# noise_layers/gaussian_noise.py

import torch
import torch.nn as nn
import numpy as np

class GaussianNoise(nn.Module):
    """
    Adds Gaussian noise with given mean and std to the noised image
    """
    def __init__(self, mean=0.0, std_range=(0.01, 0.05)):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.std_min = std_range[0]
        self.std_max = std_range[1]

    def forward(self, noised_and_cover):
        noised_image = noised_and_cover[0]

        std = np.random.uniform(self.std_min, self.std_max)
        noise = torch.randn_like(noised_image) * std + self.mean
        noised_image = noised_image + noise
        noised_image = torch.clamp(noised_image, 0.0, 1.0)  # Optional: keep image range valid
        return [noised_image, noised_and_cover[1]]
