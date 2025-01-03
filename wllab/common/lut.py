import torch
import torch.nn as nn
import numpy as np
import os

def lut_genI(bitwith, interval) -> torch.Tensor:
    """
    生成 LUT 网络的输入。
    Args:
        bitwith : 位宽
        interval: 间隔
    """
    # 生成 1D 输入
    # [L]               0..2^bitwith-1 = L
    base = torch.arange(0, 2 ** bitwith + 1 , 2 ** interval)
    base[-1] -= 1
    L = base.size(0)

    # 生成 2D 输入       |   L    |
    # [L * L]           0 0 0... | 1 1 1... | ... | 255 255 255...
    C1 = base.unsqueeze(1).repeat(1, L).view(-1)
    # [L * L]           0 1 2... 255 | 0 1 2... 255 | ... | 0 1 2... 255
    C2 = base.repeat(L)
    # [L * L, 2]
    D2 = torch.stack([C1, C2], dim=1)

    # 生成 3D 输入       | L * L  |
    # [L * L * L]       0 0 0... | 1 1 1... | ... | 255 255 255...
    C1 = base.unsqueeze(1).repeat(1, L * L).view(-1)
    # [L * L * L]       
    C2 = D2.repeat(L, 1)
    # [L * L * L, 3]   
    D3 = torch.cat([C1.unsqueeze(1) ,C2], dim=1)

    # 生成 4D 输入       | L*L*L  |
    # [L * L * L * L]   0 0 0... | 1 1 1... | ... | 255 255 255...
    C1 = base.unsqueeze(1).repeat(1, L * L * L).view(-1)
    # [L * L * L * L]
    C2 = D3.repeat(L, 1)
    # [L * L * L * L, 4]
    D4 = torch.cat([C1.unsqueeze(1) ,C2], dim=1)

    # 生成输入张量 [L * L * L * L, 1, 2, 2]
    I = D4.unsqueeze(1).unsqueeze(1).reshape(-1, 1, 2, 2).float() / 255.0
    return I

def lut_modeI(I : torch.Tensor, mode) -> torch.Tensor:
    """
    根据模式切换 LUT 网络的输入。
    Args:
        I : 输入张量
        mode : 模式
    """
    # 生成输出张量
    if mode == "d":
        O = torch.zeros((I.shape[0], I.shape[1], 3, 3), dtype=I.dtype).to(I.device)
        O[:, :, 0, 0] = I[:, :, 0, 0]
        O[:, :, 0, 2] = I[:, :, 0, 1]
        O[:, :, 2, 0] = I[:, :, 1, 0]
        O[:, :, 2, 2] = I[:, :, 1, 1]
        I = O
    elif mode == "y":
        O = torch.zeros((I.shape[0], I.shape[1], 3, 3), dtype=I.dtype).to(I.device)
        O[:, :, 0, 0] = I[:, :, 0, 0]
        O[:, :, 1, 1] = I[:, :, 0, 1]
        O[:, :, 1, 2] = I[:, :, 1, 0]
        O[:, :, 2, 2] = I[:, :, 1, 1]
        I = O
    else:
        raise ValueError("Mode {} is not supported.".format(mode))
    return I


def lut_save(model, 
            modes: list = ['s', 'd', 'y'], 
            stages: int = 2, 
            bitwith: int = 8, 
            interval: int = 4, 
            upscale : int = 4, 
            batch : int = 64, 
            save_dir: str = "./lut"):
    """
    保存 LUT 网络的输出。
    Args:
        model: LUT 网络
        modes: 模式列表
        stages: 阶段数
        bitwith: 位宽
        interval: 间隔
        batch: 批次大小
        save_dir: 保存目录
    """
    tot_size = 0
    for s in range(stages):
        stage = s + 1
        for mode in modes:
            I = lut_genI(bitwith, interval)
            if mode != "s":
                I = lut_modeI(I, mode)
            # 拆分 input 的批次
            B = I.size(0) // batch
            Os = []
            # 逐批次进行转换
            with torch.no_grad():
                for b in range(batch):
                    if b == batch - 1:
                        BI = I[b * B:]
                    else:
                        BI = I[b * B: (b + 1) * B]
                    BO = model(BI)
                    BO = torch.round(torch.tanh(BO) * 127).cpu().numpy().astype(np.int8)
                    Os.append(BO)
            O = np.concatenate(Os, axis=0)
            # 保存输出
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            path = os.path.join(save_dir, f"x{upscale}_{interval}b_i8_s{stage}_{mode}.npy")
            np.save(path, O)
            size = O.shape[0] * O.shape[1] * O.shape[2] * O.shape[3]
            tot_size += size
            print(f"Saved LUT{O.shape} ({size/1024:.2f} KB): {path}")
    print(f"Total LUT size: {tot_size/1024:.2f} KB")


def lut_load(model,
             modes: list = ['s', 'd', 'y'], 
             stages: int = 2, 
             bitwith: int = 8, 
             interval: int = 4, 
             upscale : int = 4, 
             compressed_dim: str = "xyzt", 
             diagonal_with: int = 2, 
             sampling_interval: int = 5,
             load_dir: str = "./lut"):
    for s in range(stages):
        stage = s + 1
        scale = upscale if stage == stages else 1
        for mode in modes:
            path = os.path.join(load_dir, f"x{upscale}_{interval}b_i8_s{stage}_{mode}.npy")
            lut = np.load(path).reshape(-1, scale * scale).astype(np.float32) / 127.0
            setattr(model, f"weight_s{stage}_{mode}", nn.Parameter(torch.tensor(lut)))
            print(f"Loaded LUT: {path}")







