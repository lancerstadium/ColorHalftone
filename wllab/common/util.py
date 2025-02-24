import os
import numpy as np
import struct

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
