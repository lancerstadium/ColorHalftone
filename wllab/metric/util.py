import torch
import numpy as np


def rgb_to_yuv(x: np.array) -> np.array:
    """
    将 RGB 图像转换为 YUV 图像。
    Args:
        x: RGB 图像，形状为 (H, W, 3)，范围 [0, 1]
    Returns:
        yuv: YUV 图像，形状为 (H, W, 3)，范围 [0, 1]
    """
    r, g, b = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.147 * r - 0.289 * g + 0.436 * b
    v = 0.615 * r - 0.515 * g - 0.100 * b
    yuv = np.stack([y, u, v], axis=-1)
    return yuv


def rgb_to_ycbcr(x: np.array) -> np.array:
    """
    将 RGB 图像转换为 YCbCr 图像。
    Args:
        x: RGB 图像，形状为 (H, W, 3)，范围 [0, 1]
    Returns:
        ycbcr: YCbCr 图像，形状为 (H, W, 3)，范围 [0, 1]
    """
    r, g, b = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.169 * r - 0.331 * g + 0.500 * b
    cr = 0.500 * r - 0.419 * g - 0.081 * b
    ycbcr = np.stack([y, cb, cr], axis=-1)
    return ycbcr


def rgb_to_y(x: np.array) -> np.array:
    """
    将 RGB 图像转换为ycbcr中的y。
    Args:
        x: RGB 图像，形状为 (H, W, 3)，范围 [0, 1]
    Returns:
        y: Y 图像，形状为 (H, W)，范围 [0, 1]
    """
    r, g, b = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y