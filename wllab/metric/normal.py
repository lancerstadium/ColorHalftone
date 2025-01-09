import numpy as np
from skimage.metrics import structural_similarity as ssim


# 计算 PSNR
def calculate_psnr(original, generated) -> float:
    """
    计算 PSNR（峰值信噪比）。 numpy 版本
    Args:
        original: 原图像，形状为 (H, W, C)，范围 [0, 1]
        generated: 生成图像，形状为 (H, W, C)，范围 [0, 1]
    Returns:
        psnr_value: PSNR 值
    """
    mse = np.mean((original - generated) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(1.0 / np.sqrt(mse))


def calculate_ssim(image1, image2) -> float:
    """ '
    计算两幅彩色图像的结构相似性指数（SSIM）。 
    参数: 
        image1 (np.ndarray): 第一幅图像，应该是彩色图像 (H, W, C)。 
        image2 (np.ndarray): 第二幅图像，应该是彩色图像 (H, W, C)。 
    返回: 
        float: 两幅图像之间的平均SSIM值。 
    """
    # 确保输入图像具有相同的形状 
    assert image1.shape == image2.shape, "输入图像形状必须相同" 
    
    # 逐通道计算SSIM，并求平均 
    if len(image1.shape) == 3:
        ssim_total = 0.0 
        for i in range(image1.shape[2]):
            ssim_value, _ = ssim(image1[:, :, i], image2[:, :, i], full=True) 
            ssim_total += ssim_value 
        ssim_avg = ssim_total / image1.shape[2] 
    else:
        ssim_avg, _ = ssim(image1[:, :, i], image2[:, :, i], full=True) 
    
    return ssim_avg