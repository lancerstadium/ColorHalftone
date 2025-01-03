import torch
import torch.nn as nn
import torch.nn.functional as F

# from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import vgg16




# 感知损失（Perceptual Loss）
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features[:16]  # 使用 VGG16 到 relu3_3
        # vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:16]  # 使用 VGG16 到 relu3_3
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_features = self.vgg(input)
        target_features = self.vgg(target)
        loss = F.mse_loss(input_features, target_features)
        return loss


# 稀疏性损失函数
def sparsity_loss(templates, sparsity_threshold=0.5):
    """
    稀疏性损失，鼓励模板稀疏性接近目标阈值。
    Args:
        templates: 模板张量 (num_classes, block_size, block_size)
        sparsity_threshold: 稀疏目标阈值（0-1）
    Returns:
        loss: 稀疏性损失值
    """
    sparsity = torch.mean(torch.abs(templates))
    loss = (sparsity - sparsity_threshold) ** 2
    return loss


def gaussian_blur(x, kernel_size=11, sigma=1.0):
    """
    对图像应用高斯模糊
    Args:
        x: 输入图像 [B, C, H, W]
        kernel_size: 高斯核大小
        sigma: 高斯核的标准差
    Returns:
        blurred_image: 模糊后的图像
    """
    # 生成高斯核
    kernel = create_gaussian_kernel(kernel_size, sigma)
    
    # 将高斯核扩展到与输入图像的通道数匹配
    kernel = kernel.expand(x.size(1), 1, kernel_size, kernel_size).to(x.device)
    
    # 对每个图像通道应用卷积来进行模糊
    blurred_image = F.conv2d(x, kernel, padding=kernel_size//2, groups=x.size(1))
    return blurred_image

def create_gaussian_kernel(kernel_size=11, sigma=1.0):
    """
    创建高斯核
    """
    # 生成高斯核的坐标
    coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    grid = torch.meshgrid([coords, coords])  # 获取坐标网格
    grid = torch.stack(grid, dim=-1)  # [kernel_size, kernel_size, 2]
    
    # 计算高斯核
    dist = grid[..., 0]**2 + grid[..., 1]**2  # 计算每个点到中心的平方距离
    kernel = torch.exp(-dist / (2 * sigma**2))  # 计算高斯函数
    kernel = kernel / kernel.sum()  # 归一化，使其和为 1
    return kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, kernel_size, kernel_size]

# 颜色正则化损失
def color_regularization_loss(output, target):
    """
    颜色正则化损失，计算生成图像与目标图像在高斯模糊后的差异
    Args:
        output: 模型生成的图片 [B, 3, H, W]
        target: 原始目标图片 [B, 3, H, W]
    Returns:
        loss: 颜色正则化损失
    """
    # 应用高斯模糊
    blurred_output = gaussian_blur(output)
    blurred_target = gaussian_blur(target)
    
    # 计算模糊图像与原图之间的损失
    color_diff = torch.abs(blurred_output - blurred_target)
    loss = torch.mean(color_diff)
    return loss

def adjacent_difference_penalty(class_indices):
    """
    计算每个通道中相邻元素的差异并应用惩罚。
    Args:
        class_indices: 类索引张量 [B, C, H, W]
    Returns:
        penalty: 相邻差异的惩罚项
    """
    # 确保 class_indices 是浮点类型
    class_indices = class_indices.float()

    # 计算水平方向的差异
    horizontal_diff = torch.abs(class_indices[:, :, :, 1:] - class_indices[:, :, :, :-1])
    # 计算垂直方向的差异
    vertical_diff = torch.abs(class_indices[:, :, 1:, :] - class_indices[:, :, :-1, :])
    
    # 计算惩罚项
    penalty = torch.mean(horizontal_diff) + torch.mean(vertical_diff)
    return penalty