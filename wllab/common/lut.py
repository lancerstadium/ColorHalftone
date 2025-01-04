import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

lut_mode_pad_dict = {"s": 1, "d": 2, "y": 2, "e": 3, "h": 3, "o": 3}

def lut_round(x: torch.Tensor) -> torch.Tensor:
    """
    BPDA 四舍五入（不可导 -> 可导）
    Args:
        x : 输入张量
    """
    y = torch.round(x).clone()
    z = x.clone()
    z.data = y.data
    return z

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
            path = os.path.join(save_dir, f"x{upscale}_{interval}b_i{bitwith}_s{stage}_{mode}.npy")
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
             load_dir: str = "./lut"):
    """
    从 npy 文件中加载 LUT pytorch tensor。
    """
    for s in range(stages):
        stage = s + 1
        scale = upscale if stage == stages else 1
        for mode in modes:
            path = os.path.join(load_dir, f"x{upscale}_{interval}b_i{bitwith}_s{stage}_{mode}.npy")
            lut = np.load(path).reshape(-1, scale * scale).astype(np.float32) / 127.0
            res = nn.Parameter(torch.tensor(lut))
            setattr(model, f"weight_s{stage}_{mode}", res)
            print(f"Loaded LUT({res.shape}): {path}")


def lut_load_np(model,
                modes: list = ['s', 'd', 'y'], 
                stages: int = 2, 
                bitwith: int = 8, 
                interval: int = 4, 
                upscale : int = 4, 
                load_dir: str = "./lut"):
    """
    从 npy 文件中加载 LUT numpy array。
    """
    for s in range(stages):
        stage = s + 1
        scale = upscale if stage == stages else 1
        for mode in modes:
            path = os.path.join(load_dir, f"x{upscale}_{interval}b_i{bitwith}_s{stage}_{mode}.npy")
            lut = np.load(path).reshape(-1, scale * scale).astype(np.float32) / 127.0
            setattr(model, f"weight_s{stage}_{mode}", lut)
            print(f"Loaded LUT{lut.size}: {path}")

# | LUT | Interpolation | Multiplications | Comparisons |
# | --- | ------------- | --------------- | ----------- |
# |  2D |   Bilinear    |        4        |      0      |
# |  2D |   Triangular  |        3        |      1      |
# |  3D |   Trilinear   |       11        |      0      |
# |  3D |   Tetrahedral |        4        |     2.5     |
# |  4D |   Tetralinear |       26        |      0      |
# |  4D |   4-simplex   |        5        |     4.5     |


def lut_interp_4d_simplex(
        img: torch.Tensor, 
        lut: torch.Tensor, 
        bitwith: int = 8,
        interval: int = 4,
        upscale: int = 4,
        pad: int = 1,
        mode: str = 's') -> torch.Tensor:
    """
    4D 简单插值
    Args:
        img : 输入图像
        lut : LUT
        bitwith : 位宽
        interval : 间隔
        upscale : 放大倍数
        pad : 填充
        mode : 模式
    """
    _, _, h, w = img.shape
    h -= pad
    w -= pad

    border = 2 ** (bitwith - 1) - 1     # 量化边界
    lut = lut_round(lut * border)
    lut = torch.clamp(lut, -border, border)

    Q = 2 ** interval                   # 间隔点数
    L = 2 ** (bitwith - interval) + 1   # 采样点数

    Aimg = torch.floor_divide(img, Q).type(torch.int64)
    Fimg = img % Q

    if mode == "s":
        # 提取 MSBs
        Ma = Aimg[:, :, 0 : 0 + h, 0 : 0 + w]
        Mb = Aimg[:, :, 0 : 0 + h, 1 : 1 + w]
        Mc = Aimg[:, :, 1 : 1 + h, 0 : 0 + w]
        Md = Aimg[:, :, 1 : 1 + h, 1 : 1 + w]

        # 提取 LSBs
        La = Fimg[:, :, 0 : 0 + h, 0 : 0 + w]
        Lb = Fimg[:, :, 0 : 0 + h, 1 : 1 + w]
        Lc = Fimg[:, :, 1 : 1 + h, 0 : 0 + w]
        Ld = Fimg[:, :, 1 : 1 + h, 1 : 1 + w]
    elif mode == "d":
        # 提取 MSBs
        Ma = Aimg[:, :, 0 : 0 + h, 0 : 0 + w]
        Mb = Aimg[:, :, 0 : 0 + h, 2 : 2 + w]
        Mc = Aimg[:, :, 2 : 2 + h, 0 : 0 + w]
        Md = Aimg[:, :, 2 : 2 + h, 2 : 2 + w]

        # 提取 LSBs
        La = Fimg[:, :, 0 : 0 + h, 0 : 0 + w]
        Lb = Fimg[:, :, 0 : 0 + h, 2 : 2 + w]
        Lc = Fimg[:, :, 2 : 2 + h, 0 : 0 + w]
        Ld = Fimg[:, :, 2 : 2 + h, 2 : 2 + w]
    elif mode == "y":
        # 提取 MSBs
        Ma = Aimg[:, :, 0 : 0 + h, 0 : 0 + w]
        Mb = Aimg[:, :, 1 : 1 + h, 1 : 1 + w]
        Mc = Aimg[:, :, 1 : 1 + h, 2 : 2 + w]
        Md = Aimg[:, :, 2 : 2 + h, 1 : 1 + w]

        # 提取 LSBs
        La = Fimg[:, :, 0 : 0 + h, 0 : 0 + w]
        Lb = Fimg[:, :, 1 : 1 + h, 1 : 1 + w]
        Lc = Fimg[:, :, 1 : 1 + h, 2 : 2 + w]
        Ld = Fimg[:, :, 2 : 2 + h, 1 : 1 + w]
    else:
        raise ValueError("Mode {} is not supported.".format(mode))

    # 展平索引
    N, C, H, W = Ma.shape[0], Ma.shape[1], Ma.shape[2], Ma.shape[3]
    sz = N * C * H * W

    Na = Ma + 1
    Nb = Mb + 1
    Nc = Mc + 1
    Nd = Md + 1

    Ma = Ma.flatten()
    Mb = Mb.flatten()
    Mc = Mc.flatten()
    Md = Md.flatten()
    Na = Na.flatten()
    Nb = Nb.flatten()
    Nc = Nc.flatten()
    Nd = Nd.flatten()

    # 查找
    lut_get = lambda x, y, z, w: lut[x * L * L * L + y * L * L + z * L + w]
    p0000 = lut_get(Ma, Mb, Mc, Md).reshape(sz, -1)
    p0001 = lut_get(Ma, Mb, Mc, Nd).reshape(sz, -1)
    p0010 = lut_get(Ma, Mb, Nc, Md).reshape(sz, -1)
    p0011 = lut_get(Ma, Mb, Nc, Nd).reshape(sz, -1)
    p0100 = lut_get(Ma, Na, Mc, Md).reshape(sz, -1)
    p0101 = lut_get(Ma, Na, Mc, Nd).reshape(sz, -1)
    p0110 = lut_get(Ma, Na, Nc, Md).reshape(sz, -1)
    p0111 = lut_get(Ma, Na, Nc, Nd).reshape(sz, -1)
    p1000 = lut_get(Na, Mb, Mc, Md).reshape(sz, -1)
    p1001 = lut_get(Na, Mb, Mc, Nd).reshape(sz, -1)
    p1010 = lut_get(Na, Mb, Nc, Md).reshape(sz, -1)
    p1011 = lut_get(Na, Mb, Nc, Nd).reshape(sz, -1)
    p1100 = lut_get(Na, Na, Mc, Md).reshape(sz, -1)
    p1101 = lut_get(Na, Na, Mc, Nd).reshape(sz, -1)
    p1110 = lut_get(Na, Na, Nc, Md).reshape(sz, -1)
    p1111 = lut_get(Na, Na, Nc, Nd).reshape(sz, -1)

    La = La.reshape(-1, 1)
    Lb = Lb.reshape(-1, 1)
    Lc = Lc.reshape(-1, 1)
    Ld = Ld.reshape(-1, 1)

    Lab = La > Lb
    Lac = La > Lc
    Lad = La > Ld
    Lbc = Lb > Lc
    Lbd = Lb > Ld
    Lcd = Lc > Ld

    # 输出插值
    out_set = lambda idx, l1, l2, l3, l4, p2, p3, p4: (Q - l1[idx]) * p0000[idx] + (l1[idx] - l2[idx]) * p2[idx] + (l2[idx] - l3[idx]) * p3[idx] + (l3[idx] - l4[idx]) * p4[idx] + (l4[idx]) * p1111[idx]
    out = torch.zeros((N, C, H, W, upscale, upscale), dtype=lut.dtype, device=lut.device).reshape(sz, -1)

    idx = id1 = torch.all(torch.cat([Lab, Lbc, Lcd], dim=1), dim=1)
    out[idx] = out_set(idx, La, Lb, Lc, Ld, p1000, p1100, p1110)
    idx = id2 = torch.all(torch.cat([~id1[:,None], Lab, Lbc, Lbd], dim=1), dim=1)
    out[idx] = out_set(idx, La, Lb, Ld, Lc, p1000, p1100, p1101)
    idx = id3 = torch.all(torch.cat([~id1[:,None], ~id2[:,None], Lab, Lbc, Lad], dim=1), dim=1)
    out[idx] = out_set(idx, La, Ld, Lb, Lc, p1000, p1001, p1101)
    idx = id4 = torch.all(torch.cat([~id1[:,None], ~id2[:,None], ~id3[:,None], Lab, Lbc], dim=1), dim=1)
    out[idx] = out_set(idx, Ld, La, Lb, Lc, p0001, p1001, p1101)

    idx = id5 = torch.all(torch.cat([~(Lbc), Lab, Lac, Lbd], dim=1), dim=1)
    out[idx] = out_set(idx, La, Lc, Lb, Ld, p1000, p1010, p1110)
    idx = id6 = torch.all(torch.cat([~(Lbc), ~id5[:,None], Lab, Lac, Lcd], dim=1), dim=1)
    out[idx] = out_set(idx, La, Lc, Ld, Lb, p1000, p1010, p1011)
    idx = id7 = torch.all(torch.cat([~(Lbc), ~id5[:,None], ~id6[:,None], Lab, Lac, Lad], dim=1), dim=1)
    out[idx] = out_set(idx, La, Ld, Lc, Lb, p1000, p1001, p1011)
    idx = id8 = torch.all(torch.cat([~(Lbc), ~id5[:,None], ~id6[:,None], ~id7[:,None], Lab, Lac], dim=1), dim=1)
    out[idx] = out_set(idx, Ld, La, Lc, Lb, p0001, p1001, p1011)

    idx = id9 = torch.all(torch.cat([~(Lbc), ~(Lac), Lab, Lad], dim=1), dim=1)
    out[idx] = out_set(idx, Lc, La, Lb, Ld, p0010, p1010, p1110)
    idx = id10 = torch.all(torch.cat([~(Lbc), ~(Lac), ~id9[:,None], Lab, Lad], dim=1), dim=1)
    out[idx] = out_set(idx, Lc, La, Ld, Lb, p0010, p1010, p1011)
    idx = id11 = torch.all(torch.cat([~(Lbc), ~(Lac), ~id9[:,None], ~id10[:,None], Lab, Lcd], dim=1), dim=1)
    out[idx] = out_set(idx, Lc, Ld, La, Lb, p0010, p0011, p1011)
    idx = id12 = torch.all(torch.cat([~(Lbc), ~(Lac), ~id9[:,None], ~id10[:,None], ~id11[:,None], Lab], dim=1), dim=1)
    out[idx] = out_set(idx, Ld, Lc, La, Lb, p0001, p0011, p1011)

    idx = id13 = torch.all(torch.cat([~(Lab), Lac, Lcd], dim=1), dim=1)
    out[idx] = out_set(idx, Lb, La, Lc, Ld, p0100, p1100, p1110)
    idx = id14 = torch.all(torch.cat([~(Lab), ~id13[:,None], Lac, Lad], dim=1), dim=1)
    out[idx] = out_set(idx, Lb, La, Ld, Lc, p0100, p1100, p1101)
    idx = id15 = torch.all(torch.cat([~(Lab), ~id13[:,None], ~id14[:,None], Lac, Lbd], dim=1), dim=1)
    out[idx] = out_set(idx, Lb, Ld, La, Lc, p0100, p0101, p1101)
    idx = id16 = torch.all(torch.cat([~(Lab), ~id13[:,None], ~id14[:,None], ~id15[:,None], Lac], dim=1), dim=1)
    out[idx] = out_set(idx, Ld, Lb, La, Lc, p0001, p0101, p1101)

    idx = id17 = torch.all(torch.cat([~(Lab), ~(Lac), Lbc, Lad], dim=1), dim=1)
    out[idx] = out_set(idx, Lb, Lc, La, Ld, p0100, p0110, p1110)
    idx = id18 = torch.all(torch.cat([~(Lab), ~(Lac), ~id17[:,None], Lbc, Lcd], dim=1), dim=1)
    out[idx] = out_set(idx, Lb, Lc, Ld, La, p0100, p0110, p0111)
    idx = id19 = torch.all(torch.cat([~(Lab), ~(Lac), ~id17[:,None], ~id18[:,None], Lbc, Lbd], dim=1), dim=1)
    out[idx] = out_set(idx, Lb, Ld, Lc, La, p0100, p0101, p0111)
    idx = id20 = torch.all(torch.cat([~(Lab), ~(Lac), ~id17[:,None], ~id18[:,None], ~id19[:,None], Lbc], dim=1), dim=1)
    out[idx] = out_set(idx, Ld, Lb, Lc, La, p0001, p0101, p0111)

    idx = id21 = torch.all(torch.cat([~(Lab), ~(Lac), ~(Lbc), Lad], dim=1), dim=1)
    out[idx] = out_set(idx, Lc, Lb, La, Ld, p0010, p0110, p1110)
    idx = id22 = torch.all(torch.cat([~(Lab), ~(Lac), ~(Lbc), ~id21[:,None], Lbd], dim=1), dim=1)
    out[idx] = out_set(idx, Lc, Lb, Ld, La, p0010, p0110, p0111)
    idx = id23 = torch.all(torch.cat([~(Lab), ~(Lac), ~(Lbc), ~id21[:,None], ~id22[:,None], Lcd], dim=1), dim=1)
    out[idx] = out_set(idx, Lc, Ld, Lb, La, p0010, p0011, p0111)
    idx = id24 = torch.all(torch.cat([~(Lab), ~(Lac), ~(Lbc), ~id21[:,None], ~id22[:,None], ~id23[:,None]], dim=1), dim=1)
    out[idx] = out_set(idx, Ld, Lc, Lb, La, p0001, p0011, p0111)

    # 输出转换
    out = out.reshape(N, C, H, W, upscale, upscale)
    out = out.permute(0, 1, 2, 4, 3, 5).reshape(N, C, H * upscale, W * upscale)
    return out
    

class BaseLUT(nn.Module):
    """
    LUT 基本推理模块。
    """
    def __init__(self, load_dir='./lut', stages=2, modes=['s', 'd', 'y'], bitwith=8, interval=4, upscale=4, phase='train', lut_interp=lut_interp_4d_simplex):
        super(BaseLUT, self).__init__()
        self.stages = stages
        self.modes = modes
        self.load_dir = load_dir
        self.bitwith = bitwith
        self.interval = interval
        self.upscale = upscale
        self.lut_interp = lut_interp
        self.phase = phase
        if load_dir is not None:
            lut_load(self, modes, stages, bitwith, interval, upscale, load_dir)

    def forward(self, x, phase='train'):
        x = torch.clamp(x, 0, 1) * 255.0
        for s in range(self.stages):
            pred = 0
            stage = s + 1
            if stage == self.stages:
                scale, avg_factor, bias = self.upscale, len(self.modes), 0
            else:
                scale, avg_factor, bias = 1, len(self.modes) * 4, 127

            for mode in self.modes:
                pad = lut_mode_pad_dict[mode]
                lut = getattr(self, f"weight_s{stage}_{mode}")
                for r in [0, 1, 2, 3]:
                    orx = F.pad(torch.rot90(x, r, [2, 3]), (0, pad, 0, pad), mode='replicate')
                    orx = self.lut_interp(orx, lut, self.bitwith, self.interval, scale, pad, mode)
                    orx = torch.rot90(orx, -r, [2, 3])
                    pred += orx
                    pred = lut_round(pred)

            x = lut_round(torch.clamp(pred / avg_factor + bias, 0, 255))

        if self.phase == 'train':
            x = x / 255.0
        return x

