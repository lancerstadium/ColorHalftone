import os
import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from ..metric.loss import PerceptualLoss, sparsity_loss, color_regularization_loss, adjacent_difference_penalty
from ..task.eval import evaluate_sr

def train_ht(
    model,
    dataloader,
    num_epochs=50,
    lr=1e-4,
    lambda1=2.0,
    lambda2=0.02,
    lambda3=0.8,
    lambda4=0.5,
    sparsity_threshold=0.2,
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
            for org in dataloader:
                
                org = org.to(device)  # 将输入移到相同的设备上
                
                # 前向传播
                out, class_indices, offsets = model(org)

                # 确保 out 也在相同的设备上
                out = out.to(device)

                # 1. 重建损失（感知损失 + MSE）
                recon_loss = F.mse_loss(out, org)

                # 2. 稀疏性损失
                sparse_loss = sparsity_loss(model.lookup_table.templates, sparsity_threshold) + sparsity_loss(out, 1 - sparsity_threshold)

                # 3. 颜色正则化损失
                color_loss = color_regularization_loss(out, org)

                # 4. 相邻值惩罚损失
                adj_penalty = adjacent_difference_penalty(class_indices)

                # 总损失
                loss = lambda1 * recon_loss + lambda2 * sparse_loss + lambda3 * color_loss + lambda4 / (1 + adj_penalty)

                # 梯度清零，反向传播，优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # 更新 tqdm 显示
                pbar.set_postfix({
                    "recon": f"{recon_loss.item():.4f}",
                    "sparse": f"{sparse_loss.item():.4f}",
                    "color": f"{color_loss.item():.4f}",
                    "diff": f"{adj_penalty.item():.4f}",
                    "total": f"{loss.item():.4f}"
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


def train_sr(
    model,
    pdataloader=None,
    load_path=None,
    num_epochs=50,
    save_epoch=20,
    lr=1e-4,
    save_path="./checkpoints"
):
    """
    模型训练函数，结合感知损失、稀疏性损失和颜色正则化损失。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if load_path:
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    perceptual_loss = PerceptualLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 创建保存路径
    os.makedirs(save_path, exist_ok=True)
    latest_model_path = os.path.join(save_path, "latest_model.pth")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        # tqdm 进度条
        with tqdm.tqdm(total=len(pdataloader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for org, ref in pdataloader:
                org = org.to(device)  # 将输入移到相同的设备上
                ref = ref.to(device)
                
                # 前向传播
                out = model(org).to(device)

                # 1. 重建损失（感知损失 + MSE）
                recon_loss = F.mse_loss(out, ref) 
                # 总损失
                loss = recon_loss

                # 梯度清零，反向传播，优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # 更新 tqdm 显示
                pbar.set_postfix({
                    "recon": f"{recon_loss.item():.4f}",
                    "total": f"{loss.item():.4f}"
                })
                pbar.update(1)

        # 平均损失
        avg_loss = epoch_loss / len(pdataloader)
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

        # 每 save_epoch 个 epoch 额外保存
        if (epoch + 1) % save_epoch == 0:
            epoch_model_path = os.path.join(save_path, f"model_epoch_{epoch + 1}.pth")
            torch.save(checkpoint, epoch_model_path)
            print(f"Checkpoint saved at epoch {epoch + 1}: {epoch_model_path}")



def finetune_sr_lut(model, 
            pdataloader=None,
            vdataloader=None,
            load_path=None,
            num_epochs=50,
            val_epochs=10,
            modes: list = ['s', 'd', 'y'], 
            stages: int = 2, 
            bitwith: int = 8, 
            interval: int = 4, 
            upscale : int = 4, 
            compressed_dim: str = "xyzt", 
            diagonal_with: int = 2, 
            sampling_interval: int = 5, 
            optim: torch.optim.Optimizer = torch.optim.Adam, 
            lr0: float = 1e-3, 
            lr1: float = 1e-4,
            betas: tuple = (0.9, 0.999), 
            eps: float = 1e-8, 
            weight_decay: float = 0,
            save_path="./lut"):
    # 初始化优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if load_path:
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model_state_dict"])
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim(params, lr=lr0, betas=betas, eps=eps, weight_decay=weight_decay)

    # 微调
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # tqdm 进度条
        with tqdm.tqdm(total=len(pdataloader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for org, ref in pdataloader:
                org = org.to(device)  # 将输入移到相同的设备上
                ref = ref.to(device)
                # 前向传播
                out = model(org).to(device)
                # 1. 重建损失（感知损失 + MSE）
                recon_loss = F.mse_loss(out, ref) 
                # 总损失
                loss = recon_loss

                # 梯度清零，反向传播，优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        
        # 验证
        if (epoch + 1) % val_epochs == 0:
            evaluate_sr(model, vdataloader, load_path=None)
    # 保存模型的 LUT
    for s in stages:
        stage = s + 1
        for m in modes:
            ft_lut_path = os.path.join(save_path, f"x{upscale}_{interval}b_i8_s{stage}_{m}.npy")
            ft_lut_weight = np.round(np.clip(getattr(model, f"weight_s{stage}_{m}").cpu().detach().numpy(), -1, 1) * 127).astype(np.int8)
            np.save(ft_lut_path, ft_lut_weight)
            print(f"Finetuned LUT saved: {ft_lut_path}")
    
    print(f"Finetuning completed. Average Loss: {epoch_loss:.4f}")
