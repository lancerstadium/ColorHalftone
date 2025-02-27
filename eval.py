import torch
import torchvision
from torchvision.transforms import ToTensor, Resize, Normalize, RandomCrop, Grayscale, CenterCrop
from torch.utils.data import DataLoader

from wllab.common.lut import lut_load
from wllab.network.lut import SRNet, SPF_LUT_net, MuLUT, LogicLUTNet, TinyLUTNetOpt
from wllab.network.ed import EnDeNet
from wllab.network.ht import HalftoneNet
from wllab.data.data import SingleDataset, PairedDataset
from wllab.task.eval import evaluate_sr, evaluate_st, evaluate_ht, save_image



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

def EVAL_LUT_SR():
    # 新建MuLUT模型，加载LUT
    # model = MuLUT("./lut/", 2, ['s', 'd', 'y'], "x4_4b_i8", 4, 4)
    # model = BaseLUT(None, 2, ['s', 'd', 'y'], 8, 4, 4)
    # lut_load(model, ['s', 'd', 'y'], 2, 8, 4, 4, './lut')
    # lut_load(model, ['s', 'd', 'y'], 2, 8, 4, 4, './lut', '', '_c1')
    # lut_load(model, ['s', 'd', 'y'], 2, 8, 4, 4, './lut', '', '_c2')
    # model = LogicLUTNet(kernel_size=3, upscale=4, n_feature=64)
    model = TinyLUTNetOpt(upscale=4, n_feature=16)

    ints = torch.rand(1, 1, 48, 48)
    # print(model.weight_s1_s[0:4])
    print(model(ints).shape)

    # 数据预处理
    transform = torchvision.transforms.Compose([
        # Grayscale(num_output_channels=1),  # 灰度化为单通道
        ToTensor(),  # 转换为Tensor
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用标准 ImageNet 均值和标准差
    ])

    vdict = {
        'Set5'      : 5,
        'Set14'     : 14,
        'BSD100'    : 100,
        # 'Urban100'  : 100,
        # 'Manga109'  : 104
    }

    for k, v in vdict.items():

        vdataset = PairedDataset(
            image_dir1=f"../dataset/{k}/LR/X4",
            image_dir2=f"../dataset/{k}/HR",
            transform1=transform,
            transform2=transform,
            max_images=v,
            upscale_factor=4,
            is_crop=False
        )
        vdataloader = DataLoader(vdataset, batch_size=1, shuffle=False)

        evaluate_sr(
            model,
            vdataloader,
            './checkpoints/latest_model.pth',
            f'./results/{k}',
            pad=2,
            is_rev=True
        )

def EVAL_LUT_SR1():
    # 新建MuLUT模型，加载LUT
    model = MuLUT("./lut/MuLUT", 2, ['s', 'd', 'y'], "x4_4b_i8", 4, 4)
    # lut_load(model, ['s', 'd', 'y'], 2, 8, 4, 4, './lut')

    ints = torch.rand(1, 1, 48, 48)
    # print(model.weight_s1_s[0:4])
    # print(model(ints).shape)

    # 数据预处理
    transform = torchvision.transforms.Compose([
        ToTensor(),  # 转换为Tensor
    ])

    vdataset = SingleDataset(
        image_dir=f"./test/org",
        transform=transform,
        max_images=10
    )
    vdataloader = DataLoader(vdataset, batch_size=1, shuffle=False)
    model.eval()

    # 开始评估
    for i, org in enumerate(vdataloader):
        out = model(org)
        img = vdataset.image_files[i]
        print(f"Saving {img}:")
        save_image(out[0,:,:,:], f"./test/py/{img}")


def EVAL_ST():
    # 数据预处理
    transform = torchvision.transforms.Compose([
        # Grayscale(num_output_channels=1),  # 灰度化为单通道
        CenterCrop(512),
        ToTensor(),  # 转换为Tensor
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用标准 ImageNet 均值和标准差
    ])

    cdataset = SingleDataset(
        image_dir="../ICCP/input/content",
        transform=transform,
        max_images=10
    )

    sdataset = SingleDataset(
        image_dir="../ICCP/input/style",
        transform=transform,
        max_images=10
    )

    cdataloader = DataLoader(cdataset, batch_size=10, shuffle=False)
    sdataloader = DataLoader(sdataset, batch_size=1, shuffle=False)

    # 初始化模型
    model = EnDeNet()

    model.load("../ICCP/models/finetuned_encoder_iter_160000.pth", 
               "../ICCP/models/finetuned_decoder_iter_160000.pth",
               "../ICCP/models/finetuned_mcc_iter_160000.pth")

    # 开始评估
    evaluate_st(model, cdataloader=cdataloader, sdataloader=sdataloader, save_dir="./results")
    
    


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
    EVAL_LUT_SR()

