import torch
import torchvision
from torchvision.transforms import ToTensor, Resize, Normalize, RandomCrop, Grayscale, CenterCrop
from torch.utils.data import DataLoader

from wllab.network.lut import SRNet, SPF_LUT_net
from wllab.network.ht import HalftoneNet
from wllab.data.data import SingleDataset, PairedDataset
from wllab.task.eval import evaluate_sr, evaluate_ht



def EVAL_SR():
    # 数据预处理
    transform = torchvision.transforms.Compose([
        # Grayscale(num_output_channels=1),  # 灰度化为单通道
        ToTensor(),  # 转换为Tensor
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用标准 ImageNet 均值和标准差
    ])

    pdataset = PairedDataset(
        image_dir1="../dataset/DIV2K/LR/X4",
        image_dir2="../dataset/DIV2K/HR",
        transform1=transform,
        transform2=transform,
        max_images=10,
        crop_size=(48, 48),
        upscale_factor=188/48,
        is_DIV2K=True
    )

    pdataloader = DataLoader(pdataset, batch_size=10, shuffle=False)

    # 初始化模型
    model = SRNet(mode='SxN', nf=64, upscale=4, dense=True)

    # 开始评估
    evaluate_sr(model, pdataloader=pdataloader, save_dir="./results")

def EVAL_HT():
    # 数据预处理
    transform1 = torchvision.transforms.Compose([
        Grayscale(num_output_channels=1),  # 灰度化为单通道
        CenterCrop(192),  # 随机裁剪为 50x50
        ToTensor(),  # 转换为Tensor
    ])
    # 数据集
    idataset = SingleDataset(
        image_dir="../dataset/DIV2K/LR/X4",
        transform=transform1,
        max_images=10
    )
    idataloader = DataLoader(idataset, batch_size=16, shuffle=False)

    # 初始化模型
    model = HalftoneNet(in_channels=1, num_classes=64, num_features=128, block_size=3, scale=1)

    # 加载保存的模型权重
    checkpoint_path = "./checkpoints/latest_model.pth"
    checkpoint = torch.load(checkpoint_path)
    # checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    # 开始评估
    evaluate_ht(model, idataloader, save_dir="./results")

# 主函数
if __name__ == "__main__":
    EVAL_SR()

