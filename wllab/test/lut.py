import torch
import torch.nn as nn
import numpy as np
import os

QBITS = 4
SCALE_FACTOR = 2**6  # 缩放因子可调参数

class IntLUTConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=1):
        super().__init__()
        self.is_depthwise = kernel_size > 1
        
        # 初始化卷积权重（深度卷积需要特殊形状）
        if self.is_depthwise:
            assert out_c == in_c, "Depthwise要求输入输出通道相同"
            self.weights_shape = (out_c, 1, kernel_size, kernel_size)
        else:
            self.weights_shape = (out_c, in_c)
            
        self.register_buffer('weights', torch.randint(-8, 8, self.weights_shape, dtype=torch.int8))
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.offset = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # 输入预处理（uint8 -> int8）
        if x.dtype == torch.uint8:
            x = x.to(torch.int8) - 128  # [-128, 127]
        
        # 量化到4bit范围（保留符号）
        x = torch.clamp(x, -8, 7).to(torch.int8)
        
        # 转换为浮点进行卷积运算（保持梯度）
        x_float = x.float()
        
        # 卷积实现
        if self.is_depthwise:
            # 深度卷积实现
            weights = self.weights.view(-1, 1, *self.weights.shape[2:]).float()
            x_conv = nn.functional.conv2d(
                x_float, weights,
                groups=self.weights.size(0),
                padding=(self.weights.shape[-1]//2,)*2
            )
        else:
            # 逐点卷积实现（通道变换）
            weights = self.weights.view(*self.weights.shape, 1, 1).float()
            x_conv = nn.functional.conv2d(x_float, weights)
        
        # 应用可学习参数
        output = (x_conv * self.scale + self.offset * SCALE_FACTOR) / SCALE_FACTOR
        return torch.clamp(torch.round(output), -128, 127).to(torch.int8)

    def build_lut(self):
        weights = self.weights.numpy()
        scale = self.scale.item() * SCALE_FACTOR
        offset = int(self.offset.item() * 127)
        
        # 构建符号扩展的LUT（处理负数）
        if self.is_depthwise:
            lut = np.zeros((weights.shape[0], 16, 16), dtype=np.int8)
            for oc in range(weights.shape[0]):
                kernel = weights[oc, 0]
                for msb in range(16):
                    for lsb in range(16):
                        # 符号扩展处理（4bit转8bit）
                        val = ((msb << QBITS) | lsb) - 128  # [-128, 127]
                        patch = np.array([[val]*3]*3)
                        output = (patch * kernel).sum()
                        lut[oc, msb, lsb] = np.clip(int(output * scale + offset), -128, 127)
        else:
            lut = np.zeros((weights.shape[0], 16, 16), dtype=np.int8)
            for oc in range(weights.shape[0]):
                for msb in range(16):
                    for lsb in range(16):
                        val = ((msb << QBITS) | lsb) - 128  # [-128, 127]
                        output = (val * weights[oc]).sum()
                        lut[oc, msb, lsb] = np.clip(int(output * scale + offset), -128, 127)
        return lut

class IntLUTEngine:
    def __init__(self, model):
        self.module_queue = []
        self.lut_cache = {}
        self._parse_model(model)
        
    def _parse_model(self, module: nn.Module):
        # 模型解析逻辑保持不变
        for name, child in module.named_children():
            if isinstance(child, nn.Sequential):
                if len(child) >= 2 and all(isinstance(m, IntLUTConv) for m in child[:2]):
                    self.module_queue.append((name, 'res_block', child))
                elif any(isinstance(m, nn.PixelShuffle) for m in child):
                    self.module_queue.append((name, 'upsample', child))
            elif isinstance(child, (IntLUTConv, nn.ReLU, nn.PixelShuffle)):
                type_tag = 'lut' if isinstance(child, IntLUTConv) else \
                          'act' if isinstance(child, nn.ReLU) else 'pxsf'
                self.module_queue.append((name, type_tag, child))
                if isinstance(child, IntLUTConv):
                    self.lut_cache[name] = child.build_lut()
            elif len(list(child.children())) > 0:
                self._parse_model(child)

    def _execute_lut(self, x, name, is_depthwise):
        lut = self.lut_cache[name]
        B, C, H, W = x.shape
        
        # 将int8分解为高低4bit（带符号处理）
        msb = ((x.int() + 128) >> QBITS).clamp(0, 15).long()  # [0,15]
        lsb = ((x.int() + 128) & 0x0F).clamp(0, 15).long()    # [0,15]
        
        if is_depthwise:
            output = torch.zeros_like(x, dtype=torch.int32)
            for c in range(C):
                output[:,c] = torch.from_numpy(lut[c, msb[:,c], lsb[:,c]])
        else:
            OC = lut.shape[0]
            output = torch.zeros(B, OC, H, W, dtype=torch.int32)
            for oc in range(OC):
                output[:,oc] = torch.sum(torch.from_numpy(lut[oc, msb, lsb]), dim=1)
        return output
    

    def save_luts(self, save_dir):
        """保存所有LUT到指定目录"""
        os.makedirs(save_dir, exist_ok=True)
        for name, lut in self.lut_cache.items():
            # 查找对应的卷积模块
            module = next((m for n, t, m in self.module_queue if t == 'lut' and n == name), None)
            if module is None:
                continue
            
            # 准备二进制数据
            scale = np.float32(module.scale.item())
            offset = np.float32(module.offset.item())
            lut_np = lut.numpy() if isinstance(lut, torch.Tensor) else lut
            
            # 文件格式：
            # [scale(4B)][offset(4B)][dim_count(4B)][dims...][data...]
            with open(os.path.join(save_dir, f"{name}.lut"), "wb") as f:
                # 写入参数
                f.write(scale.tobytes())
                f.write(offset.tobytes())
                
                # 写入维度信息
                dims = np.int32(lut_np.ndim)
                f.write(dims.tobytes())
                f.write(np.array(lut_np.shape, dtype=np.int32).tobytes())
                
                # 写入LUT数据
                f.write(lut_np.astype(np.int8).tobytes())

    def load_luts(self, load_dir):
        """从目录加载所有LUT"""
        self.lut_cache.clear()
        for fname in os.listdir(load_dir):
            if not fname.endswith(".lut"):
                continue
            
            # 解析文件名
            name = os.path.splitext(fname)[0]
            path = os.path.join(load_dir, fname)
            
            with open(path, "rb") as f:
                # 读取参数
                scale = np.frombuffer(f.read(4), dtype=np.float32)[0]
                offset = np.frombuffer(f.read(4), dtype=np.float32)[0]
                
                # 读取维度信息
                dim_count = np.frombuffer(f.read(4), dtype=np.int32)[0]
                shape = tuple(np.frombuffer(f.read(4*dim_count), dtype=np.int32))
                
                # 读取LUT数据
                data = np.frombuffer(f.read(), dtype=np.int8).reshape(shape)
            
            # 更新模块参数
            module = next((m for _, t, m in self.module_queue if t == 'lut' and m._get_name() == name), None)
            if module:
                module.scale.data.copy_(torch.tensor(scale))
                module.offset.data.copy_(torch.tensor(offset))
            
            # 缓存LUT
            self.lut_cache[name] = data

    def forward(self, x):
        # 输入预处理（uint8 -> int8）
        x = x.to(torch.int8) - 128
        
        # 执行计算图
        for name, type_tag, module in self.module_queue:
            if type_tag == 'lut':
                x = self._execute_lut(x, name, module.is_depthwise)
                x = (x * module.scale + module.offset * SCALE_FACTOR) // SCALE_FACTOR
                x = torch.clamp(x, -128, 127).to(torch.int8)
            elif type_tag == 'act':
                x = torch.where(x < 0, torch.zeros_like(x), x)
            elif type_tag == 'res_block':
                identity = x.clone()
                for m in module:
                    if isinstance(m, IntLUTConv):
                        x = self._execute_lut(x, m)
                        x = (x * m.scale + m.offset * SCALE_FACTOR) // SCALE_FACTOR
                        x = torch.clamp(x, -128, 127).to(torch.int8)
                x = self._safe_add(x, identity)
            elif type_tag == 'pxsf':
                x = module(x.to(torch.float32)).to(torch.int8)
        # 输出后处理（int8 -> uint8）
        return (x.to(torch.float) + 128).clamp(0, 255).to(torch.uint8)
    
    def _safe_add(self, a, b):
        result = a.int() + b.int()
        return torch.clamp(result, -128, 127).to(torch.int8)

class SRNet(nn.Module):
    def __init__(self, in_c=3, upscale=2):
        super().__init__()
        self.conv1 = IntLUTConv(in_c, upscale**2, 1)
        self.conv2 = IntLUTConv(upscale**2, upscale**2, 3)
        self.conv3 = IntLUTConv(upscale**2, in_c*upscale**2, 1)
        self.pxsf = nn.PixelShuffle(upscale)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.pxsf(x.float()).to(torch.int8)

def engine_test(upscale=2, bsize=32):
    model = SRNet(upscale=upscale)
    engine = IntLUTEngine(model)
    
    # 生成测试输入（uint8范围）
    dummy_input = torch.randint(0, 255, (1,3,bsize,bsize), dtype=torch.uint8)
    # 保存LUT测试
    engine.save_luts("./luts")

    # 引擎推理
    engine_out = engine.forward(dummy_input.clone())
    
    # 模型推理（模拟硬件行为）
    model_in = dummy_input.to(torch.int8) - 128
    model_out = model(model_in)
    model_out = (model_out.to(torch.float) + 128).clamp(0, 255).to(torch.uint8)
    
    print(f"输入尺寸: {dummy_input.shape}")
    print(f"引擎输出: {engine_out.shape} {engine_out.dtype}")
    print(f"模型输出: {model_out.shape} {model_out.dtype}")
    assert engine_out.shape == (1,3,bsize*upscale,bsize*upscale)
    print("验证通过!")

if __name__ == "__main__":
    engine_test(upscale=2, bsize=32)
