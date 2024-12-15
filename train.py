import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import vgg16
from torchvision.transforms import ToTensor, Resize, Normalize
from torch.utils.data import DataLoader
import os
import tqdm

from data import HalftoneDataset
from model import HalftoneNet


# 感知损失（Perceptual Loss）
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features[:16]  # 使用 VGG16 到 relu3_3
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


def gaussian_blur(x, kernel_size=5, sigma=1.0):
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

def create_gaussian_kernel(kernel_size=5, sigma=1.0):
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



# 模型训练函数
def train(
    model,
    dataloader,
    num_epochs=50,
    lr=1e-4,
    lambda1=2.0,
    lambda2=0.01,
    lambda3=0.8,
    sparsity_threshold=0.5,
    save_path="./checkpoints"
):
    """
    模型训练函数，结合感知损失、稀疏性损失和颜色正则化损失。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    perceptual_loss = PerceptualLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 创建保存路径
    os.makedirs(save_path, exist_ok=True)
    latest_model_path = os.path.join(save_path, "latest_model.pth")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        # tqdm 进度条
        with tqdm.tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for i, batch in enumerate(dataloader):
                batch = batch.to(device)  # 输入: [B, 3, H, W]

                # 前向传播
                output, class_indices, offsets = model(batch)

                # 1. 重建损失（感知损失 + MSE）
                recon_loss = F.mse_loss(output, batch) + perceptual_loss(output, batch)

                # 2. 稀疏性损失
                sparse_loss = sparsity_loss(model.lookup_table.templates, sparsity_threshold)

                # 3. 颜色正则化损失
                color_loss = color_regularization_loss(output, batch)

                # 总损失
                loss = lambda1 * recon_loss + lambda2 * sparse_loss + lambda3 * color_loss

                # 梯度清零，反向传播，优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # 更新 tqdm 显示
                pbar.set_postfix({
                    "recon_loss": f"{recon_loss.item():.4f}",
                    "sparse_loss": f"{sparse_loss.item():.4f}",
                    "color_loss": f"{color_loss.item():.4f}",
                    "total_loss": f"{loss.item():.4f}"
                })
                pbar.update(1)

        # 平均损失
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}")

        # 保存模型
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss
        }
        torch.save(checkpoint, latest_model_path)
        print(f"Model saved: {latest_model_path}")

        # 每 10 个 epoch 额外保存
        if (epoch + 1) % 10 == 0:
            epoch_model_path = os.path.join(save_path, f"model_epoch_{epoch + 1}.pth")
            torch.save(checkpoint, epoch_model_path)
            print(f"Checkpoint saved at epoch {epoch + 1}: {epoch_model_path}")


# 主函数
if __name__ == "__main__":
    from torchvision.transforms import ToTensor, Normalize, RandomCrop
    from model import HalftoneNet
    from data import HalftoneDataset
    from torch.utils.data import DataLoader

    # 数据预处理
    transform = torchvision.transforms.Compose([
        RandomCrop(48),  # 随机裁剪为 48x48
        ToTensor(),  # 转换为Tensor
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用标准 ImageNet 均值和标准差
    ])

    # 数据集
    dataset = HalftoneDataset(
        image_dir="/home/lancer/item/half-tone/dataset/VOC2012/val/raw",
        transform=transform,
        max_images=10000
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 初始化模型
    model = HalftoneNet(in_channels=3, num_classes=256, num_features=64, block_size=3)

    # 开始训练
    train(
        model=model,
        dataloader=dataloader,
        num_epochs=200,
        lr=1e-5,
        lambda1=2.0,
        lambda2=0.05,
        lambda3=0.8,
        sparsity_threshold=0.5,
        save_path="./checkpoints"
    )
