import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Normalize, RandomCrop, CenterCrop, Grayscale
import os
import tqdm


from ..network.lut import SRNet
from ..network.ht import HalftoneNet
from ..data.data import SingleDataset
from ..metric.normal import calculate_psnr, calculate_ssim


# 保存图片函数
def save_image(tensor, file_path, is_in=False):
    """
    将 Tensor 图像保存为文件。
    Args:
        tensor: 形状为 (C, H, W)，值范围 [0, 1]
        file_path: 保存路径
        mean: 反标准化的均值
        std: 反标准化的标准差
    """

    device = tensor.device  # 获取 tensor 的设备
    #print(tensor)
    if is_in:
        # 将 mean 和 std 移动到 tensor 所在的设备
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        mean = torch.tensor(mean).to(device).view(-1, 1, 1)
        std = torch.tensor(std).to(device).view(-1, 1, 1)
        # 反标准化
        tensor = tensor * std + mean

    # 将范围限制在 [0, 1] 之间
    tensor = tensor.clamp(0, 1)
    
    # 转换为 (H, W, C) 并保存
    image = torchvision.transforms.ToPILImage()(tensor)
    image.save(file_path)






# 评估函数
def evaluate_sr(model, idataloader, odataloader, save_dir="./results"):
    """
    模型评估函数，计算 PSNR 和 SSIM。
    Args:
        model: 待评估的模型
        dataloader: 测试集数据加载器
        save_dir: 保存结果的目录
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    num_images = 0

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 获取文件名列表
    file_names = idataloader.dataset.image_files

    # 评估模式下禁用梯度计算
    with torch.no_grad():
        i = 0
        for batch, ref in tqdm.tqdm(zip(idataloader,odataloader), desc="Evaluating"):
            batch = batch.to(device)  # 输入形状: [B, 3, H, W]
            ref = ref.to(device)

            # 前向传播
            output = model(batch)  # 输出: [B, 3, H, W]

            # 保存原图和输出图
            batch_start = i * idataloader.batch_size
            batch_end = min(batch_start + batch.size(0), len(file_names))
            i = i + 1
            for idx, file_name in enumerate(file_names[batch_start:batch_end]):
                original_image = batch[idx].cpu().permute(1, 2, 0).numpy()  # 转换为 (H, W, C)
                output_image = output[idx].cpu().permute(1, 2, 0).numpy()  # 转换为 (H, W, C)\
                ref_image = ref[idx].cpu().permute(1, 2, 0).numpy()  # 转换为 (H, W, C)

                # 保存图片
                original_path = os.path.join(save_dir, f"org_{file_name}")
                output_path = os.path.join(save_dir, f"out_{file_name}")
                save_image(batch[idx], original_path, is_in=True)
                save_image(output[idx], output_path)

                # 计算 PSNR 和 SSIM
                psnr_value = calculate_psnr(torch.tensor(ref_image), torch.tensor(output_image))
                ssim_value = calculate_ssim(ref_image, output_image)

                total_psnr += psnr_value
                total_ssim += ssim_value
                num_images += 1

    # 平均 PSNR 和 SSIM
    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images

    # 输出评估结果
    print(f"Evaluation Results:")
    print(f"  Average PSNR: {avg_psnr:.4f}")
    print(f"  Average SSIM: {avg_ssim:.4f}")


def evaluate_ht(model, dataloader, save_dir="./results"):
    """
    模型评估函数，计算 PSNR 和 SSIM。
    Args:
        model: 待评估的模型
        dataloader: 测试集数据加载器
        save_dir: 保存结果的目录
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    num_images = 0

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 获取文件名列表
    file_names = dataloader.dataset.image_files

    # 评估模式下禁用梯度计算
    with torch.no_grad():
        i = 0
        for batch in tqdm.tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)  # 输入形状: [B, 3, H, W]

            # 前向传播
            output, _, _ = model(batch)  # 输出: [B, 3, H, W]

            # 保存原图和输出图
            batch_start = i * dataloader.batch_size
            batch_end = min(batch_start + batch.size(0), len(file_names))
            i = i + 1
            for idx, file_name in enumerate(file_names[batch_start:batch_end]):
                original_image = batch[idx].cpu().permute(1, 2, 0).numpy()  # 转换为 (H, W, C)
                output_image = output[idx].cpu().permute(1, 2, 0).numpy()  # 转换为 (H, W, C)\

                # 保存图片
                original_path = os.path.join(save_dir, f"org_{file_name}")
                output_path = os.path.join(save_dir, f"out_{file_name}")
                save_image(batch[idx], original_path, is_in=True)
                save_image(output[idx], output_path)

                # 计算 PSNR 和 SSIM
                psnr_value = calculate_psnr(torch.tensor(original_image), torch.tensor(output_image))
                ssim_value = calculate_ssim(original_image, output_image)

                total_psnr += psnr_value
                total_ssim += ssim_value
                num_images += 1

    # 平均 PSNR 和 SSIM
    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images

    # 输出评估结果
    print(f"Evaluation Results:")
    print(f"  Average PSNR: {avg_psnr:.4f}")
    print(f"  Average SSIM: {avg_ssim:.4f}")



