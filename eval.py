import torch
import torchvision
from torchvision.transforms import ToTensor, Resize, Normalize, RandomCrop, Grayscale, CenterCrop
from torch.utils.data import DataLoader

from wcv.network.lut import SRNet
from wcv.network.ht import HalftoneNet
from wcv.data.data import SingleDataset
from wcv.task.eval import evaluate_sr, evaluate_ht



def EVAL_SR():
    # 数据预处理
    transform1 = torchvision.transforms.Compose([
        # Grayscale(num_output_channels=1),  # 灰度化为单通道
        CenterCrop(48),  # 随机裁剪为 50x50
        ToTensor(),  # 转换为Tensor
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用标准 ImageNet 均值和标准差
    ])
    transform2 = torchvision.transforms.Compose([
        # Grayscale(num_output_channels=1),  # 灰度化为单通道
        CenterCrop(188),  # 随机裁剪为 50x50
        ToTensor(),  # 转换为Tensor
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用标准 ImageNet 均值和标准差
    ])

    # 数据集
    idataset = SingleDataset(
        image_dir="/home/lancer/item/DIV2K/LR/X4",
        transform=transform1,
        max_images=10
    )
    odataset = SingleDataset(
        image_dir="/home/lancer/item/DIV2K/HR",
        transform=transform2,
        max_images=10
    )
    idataloader = DataLoader(idataset, batch_size=16, shuffle=False)
    odataloader = DataLoader(odataset, batch_size=16, shuffle=False)

    # 初始化模型
    model = SRNet(mode='SxN', nf=64, upscale=4, dense=True)

    # 加载保存的模型权重
    checkpoint_path = "./checkpoints/latest_model.pth"
    checkpoint = torch.load(checkpoint_path)
    # checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 开始评估
    evaluate_sr(model, idataloader, odataloader, save_dir="./results")

def EVAL_HT():
    # 数据预处理
    transform1 = torchvision.transforms.Compose([
        Grayscale(num_output_channels=1),  # 灰度化为单通道
        CenterCrop(192),  # 随机裁剪为 50x50
        ToTensor(),  # 转换为Tensor
    ])
    # 数据集
    idataset = SingleDataset(
        image_dir="/home/lexer/item/DIV2K/DIV2K_train_LR_bicubic/X4",
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

