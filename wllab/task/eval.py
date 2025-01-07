import torch
import torch.nn.functional as F
import torchvision
import os
import tqdm


from ..metric.util import rgb_to_y
from ..metric.normal import calculate_psnr, calculate_ssim


# 保存图片函数
def save_image(tensor, file_path, is_norm=False, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    将 Tensor 图像保存为文件。
    Args:
        tensor: 形状为 (C, H, W)，值范围 [0, 1]
        file_path: 保存路径
        mean: 反标准化的均值
        std: 反标准化的标准差
    """

    device = tensor.device  # 获取 tensor 的设备
    # print(tensor)
    if is_norm:
        # 将 mean 和 std 移动到 tensor 所在的设备
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
def evaluate_sr(model, 
    pdataloader, 
    load_path='./checkpoints/latest_model.pth', 
    save_dir="./results",
    pad = 1):
    """
    模型评估函数，计算 PSNR 和 SSIM。
    Args:
        model: 待评估的模型
        pdataloader: 对测试集数据加载器
        save_dir: 保存结果的目录
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if load_path:
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    num_images = 0

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 获取文件名列表
    file_names = pdataloader.dataset.image_files

    # 评估模式下禁用梯度计算
    with torch.no_grad():
        i = 0
        for org, ref in tqdm.tqdm(pdataloader, desc="Evaluating", total=len(pdataloader)):
            org = org.to(device)  # 输入形状: [B, 3, H, W]
            org = F.pad(org, (0, pad, 0, pad), mode='replicate')
            ref = ref.to(device)
            # 前向传播
            out = model(org)  # 输出: [B, 3, H, W]

            # 保存原图和输出图
            org_start = i * pdataloader.batch_size
            org_end = min(org_start + org.size(0), len(file_names))
            i = i + 1
            for idx, file_name in enumerate(file_names[org_start:org_end]):
                org_image = org[idx].cpu().permute(1, 2, 0).numpy()  # 转换为 (H, W, C)
                out_image = out[idx].cpu().permute(1, 2, 0).numpy()  # 转换为 (H, W, C)
                ref_image = ref[idx].cpu().permute(1, 2, 0).numpy()  # 转换为 (H, W, C)

                # 保存图片
                if save_dir:
                    org_path = os.path.join(save_dir, f"org_{file_name}")
                    out_path = os.path.join(save_dir, f"out_{file_name}")
                    ref_path = os.path.join(save_dir, f"ref_{file_name}")
                    save_image(org[idx], org_path, is_norm=False)
                    save_image(out[idx], out_path, is_norm=False)
                    save_image(ref[idx], ref_path, is_norm=False)

                # 计算 PSNR 和 SSIM
                ref_y = rgb_to_y(ref_image)
                out_y = rgb_to_y(out_image)
                psnr_value = calculate_psnr(ref_y, out_y)
                ssim_value = calculate_ssim(ref_y, out_y)

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


# 评估函数
def evaluate_st(model, 
    cdataloader, 
    sdataloader,
    load_path=None, 
    save_dir="./results",
    pad = 1):
    """
    模型评估函数，计算 PSNR 和 SSIM。
    Args:
        model: 待评估的模型
        pdataloader: 对测试集数据加载器
        save_dir: 保存结果的目录
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if load_path:
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    num_images = 0

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 获取文件名列表
    cfile_names = cdataloader.dataset.image_files
    sfile_names = sdataloader.dataset.image_files

    # 评估模式下禁用梯度计算
    with torch.no_grad():
        i = 0
        for sty in tqdm.tqdm(sdataloader, desc="Evaluating", total=len(sdataloader)):
            for org in cdataloader:
                org = org.to(device)  # 输入形状: [B, 3, H, W]
                org = F.pad(org, (0, pad, 0, pad), mode='replicate')
                sty = sty.to(device)
                # 前向传播
                out = model(org,sty)  # 输出: [B, 3, H, W]

                # 保存输出图
                org_start = i * cdataloader.batch_size
                org_end = min(org_start + org.size(0), len(cfile_names))
                style_name = sfile_names[i]
                i = i + 1
                for idx, file_name in enumerate(cfile_names[org_start:org_end]):
                    org_image = org[idx].cpu().permute(1, 2, 0).numpy()  # 转换为 (H, W, C)
                    out_image = out[idx].cpu().permute(1, 2, 0).numpy()  # 转换为 (H, W, C)

                    # 保存图片
                    if save_dir:
                        out_path = os.path.join(save_dir, f"out_{file_name}_{style_name}")
                        save_image(out[idx], out_path, is_norm=True)

                    # 计算 PSNR 和 SSIM
                    org_y = rgb_to_y(org_image)
                    out_y = rgb_to_y(out_image)
                    psnr_value = calculate_psnr(org_y, out_y)
                    ssim_value = calculate_ssim(org_y, out_y)

                    total_psnr += psnr_value
                    total_ssim += ssim_value
                    num_images += 1

            # 平均 PSNR 和 SSIM
            avg_psnr = total_psnr / num_images
            avg_ssim = total_ssim / num_images

            # 输出评估结果
            print(f"Evaluation Results({style_name}):")
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
        for org in tqdm.tqdm(dataloader, desc="Evaluating"):
            org = org.to(device)  # 输入形状: [B, 3, H, W]

            # 前向传播
            out, _, _ = model(org)  # 输出: [B, 3, H, W]

            # 保存原图和输出图
            org_start = i * dataloader.org_size
            org_end = min(org_start + org.size(0), len(file_names))
            i = i + 1
            for idx, file_name in enumerate(file_names[org_start:org_end]):
                org_image = org[idx].cpu().permute(1, 2, 0).numpy()  # 转换为 (H, W, C)
                out_image = out[idx].cpu().permute(1, 2, 0).numpy()  # 转换为 (H, W, C)\

                # 保存图片
                org_path = os.path.join(save_dir, f"org_{file_name}")
                out_path = os.path.join(save_dir, f"out_{file_name}")
                save_image(org[idx], org_path, is_norm=True)
                save_image(out[idx], out_path)

                # 计算 PSNR 和 SSIM
                psnr_value = calculate_psnr(org_image, out_image)
                ssim_value = calculate_ssim(org_image, out_image)

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



