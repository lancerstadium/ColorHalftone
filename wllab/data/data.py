from torch.utils.data import Dataset
from PIL import Image
import os
import random
import numpy as np

class SingleDataset(Dataset):
    '''
    单图数据集
    '''
    def __init__(self, image_dir, transform=None, max_images=None):
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)[:max_images]  # 控制最大图像数量
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
    

class PairedDataset(Dataset):
    '''
    成对数据集
    '''
    def __init__(self, image_dir1, image_dir2, transform1=None, transform2=None, max_images=None, crop_size=(256, 256), upscale_factor=1, is_DIV2K=False):
        """
        :param image_dir1: 图像1的文件夹路径（低分辨率图像）
        :param image_dir2: 图像2的文件夹路径（超分图像）
        :param transform1: 对图像1应用的变换
        :param transform2: 对图像2应用的变换
        :param max_images: 最大图像数量
        :param crop_size: 裁剪尺寸（高度，宽度），基于图像1
        :param upscale_factor: 超分图像的放大因子（如1x, 2x, 3x, 4x）
        :param is_DIV2K: 是否使用 DIV2K 数据集
        """
        self.image_dir1 = image_dir1
        self.image_dir2 = image_dir2
        self.image_files = os.listdir(image_dir1)[:max_images]  # 控制最大图像数量
        self.transform1 = transform1
        self.transform2 = transform2
        self.crop_size = crop_size  # 裁剪尺寸，基于图像1的尺寸
        self.upscale_factor = upscale_factor  # 超分图像的放大因子
        self.is_DIV2K = is_DIV2K

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if self.is_DIV2K:
            img_path1 = os.path.join(self.image_dir1, self.image_files[idx])
            img_path2 = os.path.join(self.image_dir2, self.image_files[idx][0:4] + '.png')
        else:
            img_path1 = os.path.join(self.image_dir1, self.image_files[idx])
            img_path2 = os.path.join(self.image_dir2, self.image_files[idx])
        
        # 打开图像
        image1 = Image.open(img_path1).convert("RGB")
        image2 = Image.open(img_path2).convert("RGB")
        
        # 获取裁剪区域的尺寸
        crop_height, crop_width = self.crop_size
        
        # 获取相同的随机裁剪位置
        i, j, h, w = self.get_random_crop_params(image1.size, (crop_height, crop_width))
        
        # 对两张图像应用同步裁剪
        image1 = image1.crop((j, i, j + w, i + h))
        
        # 对于image2，裁剪区域的尺寸应该按比例调整
        # image2的裁剪位置和尺寸按比例调整
        image2 = image2.crop((
            int(j * self.upscale_factor), 
            int(i * self.upscale_factor), 
            int((j + w) * self.upscale_factor), 
            int((i + h) * self.upscale_factor)
        ))
        
        # 应用 transform1 和 transform2
        if self.transform1:
            image1 = self.transform1(image1)
        if self.transform2:
            image2 = self.transform2(image2)
        
        return image1, image2

    def get_random_crop_params(self, image_size, crop_size):
        """
        计算随机裁剪区域的位置
        :param image_size: 图像的尺寸 (width, height)
        :param crop_size: 裁剪区域的尺寸 (crop_height, crop_width)
        :return: 裁剪位置和尺寸 (top, left, crop_height, crop_width)
        """
        width, height = image_size
        crop_height, crop_width = crop_size
        top = random.randint(0, height - crop_height)
        left = random.randint(0, width - crop_width)
        return top, left, crop_height, crop_width