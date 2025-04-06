import torch
import numpy as np



def rgb_to_ycbcr(x: np.array, only_y=False) -> np.array:
    """
    将 RGB 图像转换为 YCbCr 图像。
    Args:
        x: RGB 图像，形状为 (H, W, 3)，范围 [0, 1]
    Returns:
        ycbcr: YCbCr 图像，形状为 (H, W, 3)，范围 [0, 1]
    """
    in_img_type = x.dtype
    x.astype(np.float32)
    if in_img_type != np.uint8:
        x *= 255.
    # convert
    if only_y:
        rlt = np.dot(x, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(x, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def rgb_to_y(x: np.array) -> np.array:
    """
    将 RGB 图像转换为ycbcr中的y。
    Args:
        x: RGB 图像，形状为 (H, W, 3)，范围 [0, 1]
    Returns:
        y: Y 图像，形状为 (H, W)，范围 [0, 1]
    """
    return rgb_to_ycbcr(x, only_y=True)


def rgb_to_y_torch(x: torch.Tensor) -> torch.Tensor:
    device = x.device
    """
    将 RGB 图像转换为ycbcr中的y。
    Args:
        x: RGB 图像，形状为 (N, 3, H, W)，范围 [0, 1]
    Returns:
        y: Y 图像，形状为 (N, 1, H, W)，范围 [0, 1]
    """
    return (torch.sum(x * torch.tensor([65.481, 128.553, 24.966])[None, :, None, None].to(device) / 255.0, dim=1, keepdim=True) + 16.0).to(device) / 255.0