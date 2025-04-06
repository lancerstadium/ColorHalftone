import os
import numpy as np
import struct
import torch
import torch.nn as nn

# 数据类型映射
dtype_map = {
    'float32': 1,
    'int32': 2,
    'float64': 3,
    'int64': 4,
    'int16': 5,
    'uint8': 6,
    'complex64': 7,
    'complex128': 8,
    'bool': 9,
    'uint32': 10,
    'uint64': 11,
    'int8': 12
}

# 读取并转换 .npy 文件为 .bin 文件
def convert_npy_to_bin(npy_file, bin_file=None):
    # 载入.npy文件
    data = np.load(npy_file)
    # 截取文件路径，替换.npy为.bin
    if bin_file is None:
        bin_file = os.path.splitext(npy_file)[0] + '.bin'

    # 获取数据类型和转换为对应的数字
    dtype = str(data.dtype)
    dtype_code = dtype_map.get(dtype, -1)  # 默认为-1，表示未支持的类型

    if dtype_code == -1:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # 获取数据的形状
    shape = data.shape
    ndim = len(shape)

    # 打开目标二进制文件进行写入
    with open(bin_file, 'wb') as f:
        # 写入数据类型（数字表示）
        f.write(struct.pack('i', dtype_code))

        # 写入张量的维度数
        f.write(struct.pack('i', ndim))

        # 写入每个维度的大小
        for dim in shape:
            f.write(struct.pack('i', dim))

        # 写入数据（按类型保存）
        # 根据数据类型，选择合适的写入方式
        if dtype == 'float32':
            dtype_format = 'f'  # 对应 float32
        elif dtype == 'int32':
            dtype_format = 'i'  # 对应 int32
        elif dtype == 'float64':
            dtype_format = 'd'  # 对应 float64
        elif dtype == 'int64':
            dtype_format = 'q'  # 对应 int64
        elif dtype == 'int16':
            dtype_format = 'h'  # 对应 int16
        elif dtype == 'uint8':
            dtype_format = 'B'  # 对应 uint8
        elif dtype == 'complex64':
            dtype_format = '2f'  # 对应 complex64
        elif dtype == 'complex128':
            dtype_format = '2d'  # 对应 complex128
        elif dtype == 'bool':
            dtype_format = '?'  # 对应 bool
        elif dtype == 'uint32':
            dtype_format = 'I'  # 对应 uint32
        elif dtype == 'uint64':
            dtype_format = 'Q'  # 对应 uint64
        elif dtype == 'int8':
            dtype_format = 'b'  # 对应 int8
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        # 将数据展平并写入
        for element in data.flatten():
            f.write(struct.pack(dtype_format, element))

        print(f"Converted {npy_file} to {bin_file}")



import torch

def convert_u8_to_bit(ts: torch.Tensor, msb_width: int = 8) -> torch.Tensor:
    # 输入验证
    if msb_width < 1 or msb_width > 8:
        raise ValueError("msb_width必须在1到8之间")

    # 转换为uint8并保持设备一致
    ts_u8 = torch.clamp(ts, 0, 255).to(torch.uint8)
    N, C, H, W = ts_u8.shape

    # 生成uint8类型的位掩码
    bit_positions = 7 - torch.arange(msb_width, device=ts.device)
    mask = (2 ** bit_positions).to(torch.uint8)  # 显式指定类型 [msb_width]

    # 维度扩展用于广播
    ts_expanded = ts_u8.unsqueeze(-1)         # [N, C, H, W, 1]
    mask = mask.view(1, 1, 1, 1, -1)          # [1, 1, 1, 1, msb_width]

    # 执行位运算（确保相同数据类型）
    bits = (ts_expanded & mask).ne(0).to(torch.float32)  # [N, C, H, W, msb_width]

    # 重组通道维度
    bits = bits.permute(0, 1, 4, 2, 3)       # [N, C, msb_width, H, W]
    return bits.reshape(N, C * msb_width, H, W)  # [N, C*msb_width, H, W]
