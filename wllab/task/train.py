import os
import tqdm
import math

import torch
import torch.optim as optim
import torch.nn as nn
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
    lr0: float = 1e-3, 
    lr1: float = 1e-4,
    betas: tuple = (0.9, 0.999), 
    eps: float = 1e-8, 
    weight_decay: float = 0,
    save_path="./checkpoints",
    is_self_ensemble = False,
    pad = 1,
    is_rev = False,
    is_acc = False
):
    """
    模型训练函数，结合感知损失、稀疏性损失和颜色正则化损失。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if load_path:
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    # perceptual_loss = PerceptualLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr0, betas=betas, eps=eps, weight_decay=weight_decay)
    # 初始化调度器
    totalIter = num_epochs * pdataloader.batch_size
    if lr1 < 0:
        lf = lambda x: (((1 + math.cos(x * math.pi / totalIter)) / 2) ** 1.0) * 0.8 + 0.2
    else:
        lr_b = lr1 / lr0
        lr_a = 1 - lr_b
        lf = lambda x: (((1 + math.cos(x * math.pi / totalIter)) / 2) ** 1.0) * lr_a + lr_b
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 创建保存路径
    os.makedirs(save_path, exist_ok=True)
    latest_model_path = os.path.join(save_path, "latest_model.pth")

    # 初始化模型时设置内存优化选项
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    if is_rev:
        pad_tuple = (pad, 0, pad, 0)
    else:
        pad_tuple = (0, pad, 0, pad)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        # tqdm 进度条
        with tqdm.tqdm(total=len(pdataloader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for org, ref in pdataloader:
                org = org.to(device)  # 将输入移到相同的设备上
                ref = ref.to(device)
                recon_loss = 0.0
                if is_self_ensemble:
                    # 自集成训练
                    if is_acc:
                        out = torch.zeros_like(ref).to(device)
                    for i in range(4):
                        # 前向传播
                        orx = F.pad(torch.rot90(org, i, [2, 3]), pad_tuple, mode='replicate').to(device)
                        out = model(orx).to(device)
                        if is_acc:
                            out += torch.rot90(out, -i, [2, 3]).to(device)
                        else:
                            out = torch.rot90(out, -i, [2, 3]).to(device)
                            # 重建损失（感知损失 + MSE）
                            recon_loss += F.mse_loss(out, ref)
                    if is_acc:
                        recon_loss = F.mse_loss(out, ref)
                    else:
                        recon_loss /= 4
                else:
                    # 前向传播
                    org = F.pad(org, pad_tuple, mode='replicate')
                    out = model(org).to(device)
                    # 1. 重建损失（感知损失 + MSE）
                    recon_loss = F.mse_loss(out, ref) 

                # 总损失
                loss = recon_loss

                # 梯度清零，反向传播，优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

                # 更新 tqdm 显示
                pbar.set_postfix({
                    "recon": f"{recon_loss.item():.5f}",
                    "total": f"{loss.item():.5f}"
                })
                pbar.update(1)

        # 平均损失
        avg_loss = epoch_loss / len(pdataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Average Loss: {avg_loss:.5f}")

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
            torch.cuda.empty_cache()
            epoch_model_path = os.path.join(save_path, f"model_epoch_{epoch + 1}.pth")
            torch.save(checkpoint, epoch_model_path)
            print(f"Checkpoint saved at epoch {epoch + 1}: {epoch_model_path}")



def finetune_lut_sr(model, 
            pdataloader=None,
            vdataloader=None,
            load_path=None,
            num_epochs=50,
            val_epochs=10,
            modes: list = ['s', 'd', 'y'], 
            stages: int = 2, 
            bitwidth: int = 8, 
            interval: int = 4, 
            upscale : int = 4, 
            optim: torch.optim.Optimizer = torch.optim.Adam, 
            lr0: float = 1e-3, 
            lr1: float = 1e-4,
            betas: tuple = (0.9, 0.999), 
            eps: float = 1e-8, 
            weight_decay: float = 0,
            lut_dir="./lut",
            is_self_ensemble = False,
            pad = 1):
    # 初始化优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if load_path:
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model_state_dict"])
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim(params, lr=lr0, betas=betas, eps=eps, weight_decay=weight_decay)

    # 初始化调度器
    totalIter = num_epochs * pdataloader.batch_size
    if lr1 < 0:
        lf = lambda x: (((1 + math.cos(x * math.pi / totalIter)) / 2) ** 1.0) * 0.8 + 0.2
    else:
        lr_b = lr1 / lr0
        lr_a = 1 - lr_b
        lf = lambda x: (((1 + math.cos(x * math.pi / totalIter)) / 2) ** 1.0) * lr_a + lr_b
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 微调模型
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # tqdm 进度条
        with tqdm.tqdm(total=len(pdataloader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for org, ref in pdataloader:
                org = org.to(device)  # 将输入移到相同的设备上
                ref = ref.to(device)
                recon_loss = 0.0
                if is_self_ensemble:
                    # 自集成训练
                    for i in range(4):
                        # 前向传播
                        orx = F.pad(torch.rot90(org, i, [2, 3]), (0, pad, 0, pad), mode='replicate').to(device)
                        out = model(orx).to(device)
                        out = torch.rot90(out, -i, [2, 3]).to(device)
                        # 1. 重建损失（感知损失 + MSE）
                        recon_loss += F.mse_loss(out, ref)
                    recon_loss /= 4
                else:
                    # 前向传播
                    org = F.pad(org, (0, pad, 0, pad), mode='replicate')
                    out = model(org).to(device)
                    # 1. 重建损失（感知损失 + MSE）
                    recon_loss = F.mse_loss(out, ref) 
                
                # 总损失
                loss = recon_loss
                # 梯度清零，反向传播，优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

                # 更新 tqdm 显示
                pbar.set_postfix({
                    "recon": f"{recon_loss.item():.5f}",
                    "total": f"{loss.item():.5f}"
                })
                pbar.update(1)

        # 平均损失
        avg_loss = epoch_loss / len(pdataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Average Loss: {avg_loss:.5f}")

        # 验证
        if (epoch + 1) % val_epochs == 0:
            evaluate_sr(model, vdataloader, load_path=None, pad=pad)

            # 保存模型的 LUT
            for s in range(stages):
                stage = s + 1
                for mode in modes:
                    ft_lut_path = os.path.join(lut_dir, f"x{upscale}_{interval}b_i{bitwidth}_s{stage}_{mode}.npy")
                    ft_lut_weight = np.round(np.clip(getattr(model, f"weight_s{stage}_{mode}").cpu().detach().numpy(), -1, 1) * 127).astype(np.int8)
                    np.save(ft_lut_path, ft_lut_weight)
                    print(f"Finetuned LUT saved: {ft_lut_path}")
    
    print(f"Finetuning completed. Average Loss: {epoch_loss:.4f}")







# ================= 训练函数（与之前相同接口） =================
def train_cf(model, dataloader, num_epochs=50,save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 损失函数与优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # tqdm 进度条
        with tqdm.tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.float().to(device).view(-1,1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
                pbar.set_postfix({
                    "loss": f"{running_loss / (pbar.n + 1):.5f}",
                    "acc": f"{correct / total:.5f}"
                })
                pbar.update(1)
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Average Loss: {running_loss / len(dataloader):.5f}, Accuracy: {correct / total:.5f}")
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}!")
    return model