import torchvision
from torchvision.transforms import ToTensor, Resize, Normalize, RandomCrop, Grayscale, CenterCrop, Pad, RandomHorizontalFlip, RandomVerticalFlip
from torch.utils.data import DataLoader
from wllab.common.util import convert_u8_to_bit
from wllab.network.logic import get_logic_model
from wllab.network.cf import PatchClassifier
from wllab.network.lut import SRNet, SPF_LUT_net, MuLUT, BaseSRNets, DepthwiseLUT, PointwiseONE, PointwiseLUT, LogicLUTNet, TinyLUTNet, TinyLUTNetOpt, VarLUTNet
from wllab.network.ht import HalftoneNet
from wllab.data.data import SingleDataset, PairedDataset, PatchDataset
from wllab.task.train import train_ht, train_sr, train_cf

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
    pdataloader = DataLoader(pdataset, batch_size=4, shuffle=False)

    # 初始化模型
    # model = SRNet(mode='SxN', nf=64, upscale=4, dense=True)
    # model = BaseSRNets(nf=64, scale=4, modes="sdy", stages=2)
    # model = HalftoneNet(in_channels=3, num_classes=64, num_features=128, block_size=3, scale=4)
    # model = LogicLUTNet(kernel_size=3, upscale=4, n_feature=64)
    # model = TinyLUTNetOpt(upscale=4, n_feature=16)
    model = VarLUTNet(upscale=4, n_feature=16, in_ch=3)

    # 开始训练
    train_sr(
        model=model,
        pdataloader=pdataloader,
        load_path=None,
        num_epochs=2000,
        save_path="./checkpoints",
        is_self_ensemble=True,
        pad=0,
        is_rev=False,
        is_acc=False
    )


def TRAIN_CF():
    patch_size = 32
    dateset = PatchDataset(
        # image_dir="../dataset/DIV2K/LR/X4",
        image_dir="../half-tone/dataset/VOC2012/train/raw",
        patch_size=patch_size,
        threshold=0.2,
        max_images=800
    )
    dataloader = DataLoader(dateset, batch_size=32, shuffle=True, num_workers=4)
    # model = PatchClassifier(in_channels=3, patch_size=patch_size)
    model, loss_fn, optimizator = get_logic_model(
        grad_factor=1.1,
        input_ndim=6144,
        layer_neurons=[8400,4200,2100,1060,540,280,150],
        nclasses=1,
        tau=150
    )
    # import torch
    # model.load_state_dict(torch.load("./checkpoints/classifier.pth"))
    train_cf(
        model=model,
        dataloader=dataloader,
        num_epochs=80,
        save_path="./checkpoints/logic_classifier.pth",
        lr=0.001,
        bit_cvt=True,
        msb_width=6
    )


# 主函数
if __name__ == "__main__":
    TRAIN_CF()
    # TRAIN_SR()
    # import torch
    # I = torch.randn(1, 3, 32, 32)
    # model = PatchClassifier(in_channels=3, patch_size=32)
    # O = model(I)
    # print(O.shape)
