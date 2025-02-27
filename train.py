import torchvision
from torchvision.transforms import ToTensor, Resize, Normalize, RandomCrop, Grayscale, CenterCrop, Pad
from torch.utils.data import DataLoader

from wllab.network.lut import SRNet, SPF_LUT_net, MuLUT, BaseSRNets, DepthwiseLUT, PointwiseONE, PointwiseLUT, LogicLUTNet, TinyLUTNet, TinyLUTNetOpt
from wllab.network.ht import HalftoneNet
from wllab.data.data import SingleDataset, PairedDataset
from wllab.task.train import train_ht, train_sr

def TRAIN_HT():
    # 数据预处理
    transform1 = torchvision.transforms.Compose([
        Grayscale(num_output_channels=1),  # 灰度化为单通道
        CenterCrop(48),  # 随机裁剪为 50x50
        ToTensor(),  # 转换为Tensor
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用标准 ImageNet 均值和标准差
    ])

    # 数据集
    idataset = SingleDataset(
        image_dir="../dataset/DIV2K/LR/X4",
        transform=transform1,
        max_images=800
    )
    idataloader = DataLoader(idataset, batch_size=16, shuffle=False)
    # 初始化模型
    model = HalftoneNet(in_channels=1, num_classes=64, num_features=128, block_size=3, scale=1)
    
    # 开始训练
    train_ht(
        model=model,
        dataloader=idataloader,
        num_epochs=200,
        lr=1e-5,
        lambda1=1.0,
        lambda2=0.05,
        lambda3=8.0,
        lambda4=0.2,
        sparsity_threshold=0.2,
        save_path="./checkpoints"
    )

def TRAIN_SR():
    # 数据预处理
    transform1 = torchvision.transforms.Compose([
        # Grayscale(num_output_channels=1),  # 灰度化为单通道
        ToTensor(),  # 转换为Tensor
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 使用标准 ImageNet 均值和标准差
        # Pad([0,0,1,1], fill=0, padding_mode='reflect')
    ])

    transform2 = torchvision.transforms.Compose([
        # Grayscale(num_output_channels=1),  # 灰度化为单通道
        ToTensor(),  # 转换为Tensor
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用标准 ImageNet 均值和标准差
    ])

    pdataset = PairedDataset(
        image_dir1="../dataset/DIV2K/LR/X4",
        image_dir2="../dataset/DIV2K/HR",
        transform1=transform1,
        transform2=transform2,
        max_images=800,
        crop_size=(48, 48),
        upscale_factor=4,
        is_DIV2K=True
    )
    pdataloader = DataLoader(pdataset, batch_size=8, shuffle=False)

    # 初始化模型
    # model = SRNet(mode='SxN', nf=64, upscale=4, dense=True)
    # model = BaseSRNets(nf=64, scale=4, modes="sdy", stages=2)
    # model = HalftoneNet(in_channels=3, num_classes=64, num_features=128, block_size=3, scale=4)
    # model = LogicLUTNet(kernel_size=3, upscale=4, n_feature=64)
    model = TinyLUTNetOpt(upscale=4, n_feature=64)

    # 开始训练
    train_sr(
        model=model,
        pdataloader=pdataloader,
        load_path=None,
        num_epochs=2000,
        save_path="./checkpoints",
        is_self_ensemble=True,
        pad=2,
        is_rev=True,
        is_acc=False
    )



# 主函数
if __name__ == "__main__":
    TRAIN_SR()
    # import torch
    # I = torch.randn(1, 3, 50, 50)
    # model = TinyLUTNetOpt()
    # O = model(I)
    # print(O.shape)
