import numpy as np



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


def calculate_ssim(original, generated) -> float:
    """
    计算 SSIM（结构相似性）。 numpy 版本
    Args:
        original: 原图像，形状为 (H, W, C)，范围 [0, 1]
        generated: 生成图像，形状为 (H, W, C)，范围 [0, 1]
    Returns:
        ssim_value: SSIM 值
    """
    K1 = 0.01
    K2 = 0.03
    L = 1  # 像素值范围 [0, L]
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    original = original.astype(np.float64)
    generated = generated.astype(np.float64)
    cov = np.cov(original.flatten(), generated.flatten(), bias=True)
    mean_original = np.mean(original)
    mean_generated = np.mean(generated)
    var_original = np.var(original)
    var_generated = np.var(generated)

    numerator = (2 * mean_original * mean_generated + C1) * (2 * cov[0, 1] + C2)
    denominator = (mean_original ** 2 + mean_generated ** 2 + C1) * (var_original + var_generated + C2)
    ssim_value = numerator / denominator
    return ssim_value