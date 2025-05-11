import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

import torch
from skimage.metrics import structural_similarity as ssim

def compute_ssim(img1, img2):
    # Detach before converting to numpy (but will restore gradient tracking later)
    img1_np = img1.detach().cpu().numpy().transpose(0, 2, 3, 1)
    img2_np = img2.detach().cpu().numpy().transpose(0, 2, 3, 1)

    batch_size, H, W, C = img1_np.shape
    data_range = img1_np.max() - img1_np.min()
    data_range = max(data_range, 1)

    # Compute SSIM for each image in the batch
    ssim_vals = [ssim(img1_np[i], img2_np[i], win_size=min(H, W, 7), channel_axis=-1, data_range=data_range) 
                 for i in range(batch_size)]

    # Convert SSIM result back to tensor and enable gradient tracking
    return torch.tensor(ssim_vals, device=img1.device, dtype=torch.float32, requires_grad=True).mean()



def compute_psnr(img1, img2, max_psnr=50):
    img1_np = img1.detach().cpu().numpy().transpose(0, 2, 3, 1)
    img2_np = img2.detach().cpu().numpy().transpose(0, 2, 3, 1)

    psnr_vals = [psnr(img1_np[i], img2_np[i]) for i in range(img1.shape[0])]

    # Convert back to tensor with gradients enabled
    psnr_vals = torch.tensor(psnr_vals, device=img1.device, dtype=torch.float32, requires_grad=True).mean()

    return 100 - (psnr_vals / max_psnr)

def compute_ber(original, decoded):
    original_np = original.detach().cpu().numpy()
    decoded_np = decoded.detach().cpu().numpy()

    # Convert back to tensor with gradients enabled
    return torch.tensor(np.mean(np.abs(original_np - decoded_np)), device=original.device, dtype=torch.float32, requires_grad=True)
