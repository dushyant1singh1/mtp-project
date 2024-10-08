import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(original, reconstructed):
    o = original.squeeze().detach().clone().cpu().numpy()
    r = reconstructed.squeeze().detach().clone().cpu().numpy()
    # Compute the Mean Squared Error (MSE)
    mse = np.mean((o - r) ** 2)
    # Avoid division by zero
    if mse == 0:
        return float('inf')
    # Compute PSNR
    max_pixel_value = 1
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    
    return psnr
    

def calculate_ssim(original, reconstructed):
    # Remove batch dimension and reshape to 2D if necessary
    o = original.squeeze().detach().clone().cpu().numpy()  # Shape: (250, 90)
    r = reconstructed.squeeze().detach().clone().cpu().numpy()  # Shape: (250, 90)
    # Determine data_range
    data_range = o.max() - o.min()
    # Compute SSIM
    ssim_value, _ = ssim(o, r, data_range=data_range, full=True)
    return ssim_value