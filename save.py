
import torchvision
from torchvision.transforms import ToTensor, Resize, Normalize, RandomCrop, Grayscale, CenterCrop
from torch.utils.data import DataLoader
from torchsummary import summary

from wllab.network.lut import BaseSRNets, SPF_LUT_net, MuLUT, print_network
from wllab.common.lut import lut_save, lut_load
from wllab.data.data import SingleDataset, PairedDataset
from wllab.task.eval import evaluate_sr, evaluate_ht

def model_summary(model):
    summary(model, (1, 48, 48))




if __name__ == "__main__":
    import torch
    # 读入训练好的模型，并采样保存LUT
    model = BaseSRNets(nf=64, scale=4, modes="sdy", stages=2)
    print_network(model)
    lut_save(model, ['s', 'd', 'y'], 2, 8, 4, 4, 64, save_dir="./lut/")

    # 新建MuLUT模型，加载LUT
    model = MuLUT("./lut/", 2, ['s', 'd', 'y'], "x4_4b_i8", 4, 4)
    lut_load(model, ['s', 'd', 'y'], 2, 8, 4, 4, 64, load_dir="./lut/")
    ints = torch.rand(1, 1, 48, 48)
    print(model(ints).shape)
    print(model.weight_s1_s.shape)

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
    pdataloader = DataLoader(pdataset, batch_size=16, shuffle=False)

    # 开始评估
    evaluate_sr(model, pdataloader=pdataloader, save_dir="./results")