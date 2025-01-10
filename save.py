
import torch.optim
import torchvision
from torchvision.transforms import ToTensor, Resize, Normalize, RandomCrop, Grayscale, CenterCrop
from torch.utils.data import DataLoader
from torchsummary import summary

from wllab.network.lut import BaseSRNets, SPF_LUT_net, MuLUT, SRNet, print_network
from wllab.common.lut import lut_save, lut_load, lut_compress, BaseLUT
from wllab.data.data import SingleDataset, PairedDataset
from wllab.task.train import finetune_lut_sr
from wllab.task.eval import evaluate_sr, evaluate_ht

def model_summary(model):
    summary(model, (1, 48, 48))



if __name__ == "__main__":
    import torch
    # 读入训练好的模型，并采样保存LUT
    # model = BaseSRNets(nf=64, scale=4, modes="sdy", stages=2)
    model = SRNet(mode='SxN', nf=64, upscale=4, dense=True)
    checkpoint = torch.load("./checkpoints/SRNet_x4_f64.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    lut_save(model, ['s', 'd', 'y'], 2, 8, 4, 4, save_dir='./lut')
    # lut_compress(model, ['s', 'd', 'y'], 2, 8, 4, 4, 'xyzt', 2, 5, './lut')


    # 新建MuLUT模型，加载LUT
    model = MuLUT("./lut", 2, ['s', 'd', 'y'], "x4_4b_i8", 4, 4)
    # model = BaseLUT(None, 2, ['s', 'd', 'y'], 8, 4, 4)
    lut_load(model, ['s', 'd', 'y'], 2, 8, 4, 4, './lut')
    # lut_load(model, ['s', 'd', 'y'], 2, 8, 4, 4, './lut', '', '_c1')
    # lut_load(model, ['s', 'd', 'y'], 2, 8, 4, 4, './lut', '', '_c2')

    ints = torch.rand(1, 1, 48, 48)
    print(model.weight_s1_s[0:4])
    print(model(ints).shape)

    # 数据预处理
    transform = torchvision.transforms.Compose([
        # Grayscale(num_output_channels=1),  # 灰度化为单通道
        ToTensor(),  # 转换为Tensor
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用标准 ImageNet 均值和标准差
    ])

    pdataset = PairedDataset(
        image_dir1="../dataset/DIV2K/LR/X4",
        image_dir2="../dataset/DIV2K/HR",
        transform1=transform,
        transform2=transform,
        max_images=800,
        crop_size=(48, 48),
        upscale_factor=4,
        is_DIV2K=True
    )
    pdataloader = DataLoader(pdataset, batch_size=16, shuffle=False)

    vdataset = PairedDataset(
        image_dir1="../dataset/DIV2K/LR/X4",
        image_dir2="../dataset/DIV2K/HR",
        transform1=transform,
        transform2=transform,
        max_images=10,
        crop_size=(48, 48),
        upscale_factor=4,
        is_DIV2K=True
    )
    vdataloader = DataLoader(vdataset, batch_size=16, shuffle=False)


    evaluate_sr(
        model,
        vdataloader,
        None,
        'results',
        0
    )

    # 微调并评估模型
    # finetune_lut_sr(
    #     model, 
    #     pdataloader, 
    #     vdataloader, 
    #     None, 
    #     200, 
    #     10, 
    #     ['s', 'd', 'y'],
    #     2,
    #     8,
    #     4,
    #     4,
    #     torch.optim.Adam,
    #     0.001,
    #     0.0001,
    #     (0.9,0.999),
    #     1e-8,
    #     0,
    #     './lut/MuLUT',
    #     False,
    #     0
    # )
    
