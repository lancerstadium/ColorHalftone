import torch
import numpy as np


# 计算 PSNR
def calculate_psnr(original, generated):
    """
    计算 PSNR（峰值信噪比）。
    Args:
        original: 原图像，形状为 (C, H, W)，范围 [0, 1]
        generated: 生成图像，形状为 (C, H, W)，范围 [0, 1]
    Returns:
        psnr_value: PSNR 值
    """
    mse = torch.mean((original - generated) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # 像素值范围 [0, 1]
    psnr_value = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr_value.item()


def calculate_ssim(original, generated):
    """
    计算 SSIM（结构相似性）。
    Args:
        original: 原图像，形状为 (C, H, W)，范围 [0, 1]
        generated: 生成图像，形状为 (C, H, W)，范围 [0, 1]
    Returns:
        ssim_value: SSIM 值
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # 使用 NumPy 直接计算
    mu_x = np.mean(original, axis=(1, 2), keepdims=True)
    mu_y = np.mean(generated, axis=(1, 2), keepdims=True)
    sigma_x = np.var(original, axis=(1, 2), keepdims=True)
    sigma_y = np.var(generated, axis=(1, 2), keepdims=True)
    sigma_xy = np.mean((original - mu_x) * (generated - mu_y), axis=(1, 2), keepdims=True)

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return np.mean(ssim_map)