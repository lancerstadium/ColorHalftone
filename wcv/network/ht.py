import torch
import torch.nn as nn
import torch.nn.functional as F

# 深度卷积模块
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# 特征提取模块（加深特征提取网络）
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, num_features=64):
        super(FeatureExtractor, self).__init__()
        self.blocks = nn.Sequential(
            DepthwiseSeparableConv(in_channels, num_features),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(num_features, num_features),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.blocks(x)  # 输出: [B, num_features, H, W]


# 偏移分支和颜色分支（加深结构并引入残差连接）
class OffsetAndColorBranch(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=32,  block_size=3, scale=4):
        super(OffsetAndColorBranch, self).__init__()
        self.block_size = block_size
        self.scale = scale
        self.in_channels = in_channels
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, self.in_channels * scale * scale, kernel_size=self.block_size, stride=self.block_size, padding=0)
        )
        self.color_predictor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, self.in_channels * scale * scale, kernel_size=self.block_size, stride=self.block_size, padding=0)
        )

    def forward(self, x, residual):
        # 根据 block_size 动态计算池化后的尺寸
        output_size  = (x.size(2) // self.block_size, x.size(3) // self.block_size)
        
        # 使用自适应池化，将图像池化为 output_size 大小
        B, C, H, W = x.shape
        adpresidual = F.adaptive_avg_pool2d(residual, output_size)
        adpresidual = adpresidual.repeat((1,1,self.scale,self.scale)).view(B, self.in_channels, output_size[0] * self.scale, output_size[1] * self.scale)
        
        offsets = torch.tanh(self.offset_predictor(x).view(B, self.in_channels, output_size[0] * self.scale, output_size[1] * self.scale) + adpresidual) # 偏移范围 [-1, 1]
        quantized_colors = torch.sigmoid(self.color_predictor(x).view(B, self.in_channels, output_size[0] * self.scale, output_size[1] * self.scale) + adpresidual)  # 颜色范围 [0, 1]
        return offsets, quantized_colors


# 深度可分离卷积模块
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        # 深度卷积
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        # 点卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 先进行深度卷积，再用点卷积来增加输出的通道数
        return self.pointwise(self.depthwise(x))

# 分类分支（加深网络并添加全卷积）
class ClassificationBranch(nn.Module):
    def __init__(self, in_channels, num_classes=32, block_size=3, scale=4):
        super(ClassificationBranch, self).__init__()
        self.num_classes = num_classes
        self.block_size = block_size
        self.in_channels = in_channels
        self.scale = scale

        # 使用多个深度可分离卷积加深网络
        self.conv1 = DepthwiseSeparableConv(self.in_channels, self.in_channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = DepthwiseSeparableConv(self.in_channels * 2, self.in_channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv3 = DepthwiseSeparableConv(self.in_channels * 2, self.in_channels * num_classes * scale * scale, kernel_size=self.block_size, stride=self.block_size, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape

        # 通过多个深度可分离卷积块
        x = self.conv1(x)  # 输出: [B, in_channels*2, H, W]
        x = self.conv2(x)  # 输出: [B, in_channels*2, H, W]
        x = self.conv3(x)  # 输出: [B, 3 * num_classes, H/3, W/3]

        # 获取 H_new 和 W_new 是卷积后输出的高度和宽度
        _, _, H_new, W_new = x.shape

        # 调整形状: [B, 3, num_classes, H/3 * scale, W/3 * scale]
        x = x.view(B, self.in_channels, self.num_classes, H_new * self.scale, W_new * self.scale).permute(0, 1, 3, 4, 2).contiguous()

        # 获取分类索引: [B, 3, H/3, W/3]
        class_indices = torch.argmax(x, dim=-1)
        return class_indices


# 查找表模块
class LookupTable(nn.Module):
    def __init__(self, num_classes=64, block_size=3):
        super(LookupTable, self).__init__()
        self.templates = nn.Parameter(torch.randn(num_classes, block_size, block_size))  # 模板参数

    def forward(self, class_indices):
        """
        Args:
            class_indices: 分类索引 [B, 3, H/3, W/3]
        Returns:
            templates: 模板张量 [B, 3, H/3, W/3, block_size, block_size]
        """
        B, C, H, W = class_indices.size()
        templates = self.templates[class_indices.view(-1)]  # 查找模板
        templates = torch.round(torch.clamp(templates.view(B, C, H, W, self.templates.size(1), self.templates.size(2)),0,1))
        return templates


# 重组模块
class ReassembleModule(nn.Module):
    def __init__(self, block_size=3, recue_ratio=0.2,scale=4):
        super(ReassembleModule, self).__init__()
        self.block_size = block_size
        self.recue_ratio = recue_ratio
        self.scale = scale

    def split_into_blocks(self, x, block_size):
        """
        将图像切分为块。
        Args:
            x: 输入图像 (B, C, H, W)
            block_size: 块大小
        Returns:
            切分后的块 (B, C, H_blocks, W_blocks, block_size, block_size)
        """
        B, C, H, W = x.size()
        H_blocks = H // block_size
        W_blocks = W // block_size
        x = x.view(B, C, H_blocks, block_size, W_blocks, block_size)
        x = x.permute(0, 1, 2, 4, 3, 5)  # 调整维度: (B, C, H_blocks, W_blocks, block_size, block_size)
        return x

    def forward(self, x, templates, quantized_colors, offsets):
        """
        Args:
            x: 输入图像 (B, C, H, W)
            templates: 模板张量 (B, C, H_blocks, W_blocks, block_size, block_size)
            quantized_colors: 量化颜色值 (B, C, H_blocks, W_blocks)
            offsets: 偏移值 (B, C, H_blocks, W_blocks)
        Returns:
            output: 重组后的大图 (B, C, H, W)
        """
        residual_blocks = self.split_into_blocks(x, self.block_size)
        R_B, R_C, R_H, R_W, _, _ = residual_blocks.size()
        residual_blocks = residual_blocks.repeat((1, 1, 1, 1, self.scale, self.scale)).view(R_B, R_C, R_H * self.scale, R_W * self.scale, self.block_size, self.block_size)
        B, C, H_blocks, W_blocks, block_size, _ = templates.size()
        H, W = H_blocks * block_size, W_blocks * block_size

        # 偏移逻辑
        shifted_templates = torch.roll(templates, shifts=2, dims=-1)  # 按最后一个维度右移两位
        applied_templates = torch.where(offsets.unsqueeze(-1).unsqueeze(-1) > 0, templates, shifted_templates)

        # 与操作：模板块与 residual 块按位相乘
        # merged_blocks = applied_templates * residual_blocks

        # 颜色映射
        quantized_colors_expanded = quantized_colors.unsqueeze(-1).unsqueeze(-1).expand(
            B, C, H_blocks, W_blocks, block_size, block_size
        )
        colored_templates = (1 - applied_templates) * ((1 - self.recue_ratio) * quantized_colors_expanded + self.recue_ratio * residual_blocks)

        # 将colored_templates限制为0或1
        colored_templates = torch.round(torch.sigmoid(colored_templates))  # 将浮动值转换为0或1

        # 重组为大图
        reconstructed_image = colored_templates.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)

        # 按 recon_ratio 融合 residual 和重组结果
        output = reconstructed_image

        return output


# 主模型
class HalftoneNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=64, num_features=64, block_size=3, scale=4):
        super(HalftoneNet, self).__init__()
        self.block_size = block_size
        self.scale = scale
        self.in_channels = in_channels
        self.feature_extractor = FeatureExtractor(self.in_channels, self.in_channels)
        self.offset_and_color_branch = OffsetAndColorBranch(self.in_channels, num_features, block_size=self.block_size,scale=self.scale)
        self.classification_branch = ClassificationBranch(self.in_channels, num_classes, block_size=self.block_size,scale=self.scale)
        self.lookup_table = LookupTable(num_classes, self.block_size)
        self.reassemble = ReassembleModule(self.block_size,scale=self.scale)

    def forward(self, x):
        features = self.feature_extractor(x)
        offsets, quantized_colors = self.offset_and_color_branch(features, x)
        class_indices = self.classification_branch(features)
        templates = self.lookup_table(class_indices)
        output = self.reassemble(x, templates, quantized_colors, offsets)
        return output, class_indices, offsets


# 测试代码
if __name__ == "__main__":
    from torchsummary import summary
    input_image = torch.randn(8, 1, 48, 48).to(device="cuda:0")
    model = HalftoneNet(in_channels=1, num_classes=64, num_features=128, block_size=3, scale=1).to(device="cuda:0")
    summary(model, input_size=(1, 48, 48))
    output, class_indices, offsets = model(input_image)
    print(f"Output shape: {output.shape}")  # 应为: [8, 3, 48*scale, 48*scale]
    print(f"Class indices shape: {class_indices.shape}")  # 应为: [8, 3, 16*scale, 16*scale]
    print(f"Offsets shape: {offsets.shape}")  # 应为: [8, 3, 16*scale, 16*scale]
