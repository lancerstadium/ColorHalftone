from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import random
import numpy as np
import torch.nn.functional as F

class SingleDataset(Dataset):
    '''
    单图数据集
    :param image_dir: 图像文件夹路径
    :param transform: 转换
    :param max_images: 最大图像数量
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
    :param image_dir1: 图像1的文件夹路径
    :param image_dir2: 图像2的文件夹路径
    :param transform1: 对图像1应用的变换
    :param transform2: 对图像2应用的变换
    :param max_images: 最大图像数量
    :param crop_size: 裁剪尺寸(高度，宽度)
    :param upscale_factor: 超分图像的放大因子
    :param is_DIV2K: 是否使用DIV2K数据集
    '''
    def __init__(self, image_dir1, image_dir2, transform1=None, transform2=None, max_images=None, crop_size=(256, 256), upscale_factor=1, is_DIV2K=False, is_crop=True):
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
        self.is_crop = is_crop

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
        
        if self.is_crop:
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
        else:
            image1 = torch.from_numpy(np.array(image1).transpose(2, 0, 1)).float()
        if self.transform2:
            image2 = self.transform2(image2)
        else:
            image2 = torch.from_numpy(np.array(image2).transpose(2, 0, 1)).float()
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
    



class PatchDataset(Dataset):
    def __init__(self, image_dir, transform=None, max_images=800, patch_size=32, threshold=0.5):
        """
        image_list: 可以是图像路径列表或已加载的numpy数组列表
        transform: 数据增强操作
        max_images: 最大处理的图像数量
        """
        # 存储原始图像引用或路径
        self.image_dir = image_dir
        self.image_list = os.listdir(image_dir)[:max_images]  # 控制最大图像数量
        self.patch_size = patch_size
        self.threshold = threshold
        self.transform = transform
        
        # 预计算所有可能的patch位置 (image_idx, y, x)
        self.patch_positions = []
        for img_idx, img_ref in enumerate(self.image_list):
            # 动态获取图像尺寸
            img_path = os.path.join(self.image_dir, self.image_list[img_idx])
            if isinstance(img_path, str):  # 从文件加载
                with Image.open(img_path) as img:
                    h, w = img.size[1], img.size[0]
            else:  # 如果是numpy数组
                h, w = img_ref.shape[:2]
            
            # 生成网格坐标 (50%重叠)
            y_coords = range(0, h - self.patch_size + 1, self.patch_size // 2)
            x_coords = range(0, w - self.patch_size + 1, self.patch_size // 2)
            self.patch_positions.extend([(img_idx, y, x) for y in y_coords for x in x_coords])

        # 缓存系统
        self.current_img_idx = None
        self.cached_img = None
        self.cached_gradient = None

    def __len__(self):
        return len(self.patch_positions)

    def __getitem__(self, idx):
        img_idx, y, x = self.patch_positions[idx]
        
        # 按需加载图像 -------------------------------------------------
        if img_idx != self.current_img_idx:
            img_ref = self.image_list[img_idx]
            img_path = os.path.join(self.image_dir, self.image_list[img_idx])
            if isinstance(img_path, str):  # 从文件加载
                with Image.open(img_path) as img:
                    img_array = np.array(img.convert('RGB'))  # 强制转RGB
            else:  # 直接使用numpy数组
                img_array = img_ref if img_ref.ndim ==3 else img_ref[..., np.newaxis]
            
            # 转换为Tensor并缓存
            if img_array.ndim == 3:
                self.cached_img = torch.from_numpy(img_array).permute(2,0,1).float() / 255.0
            else:
                self.cached_img = torch.from_numpy(img_array).unsqueeze(0).float() / 255.0
            
            # 计算并缓存梯度图
            self.cached_gradient = self._compute_gradient(self.cached_img)
            self.current_img_idx = img_idx

        # 动态提取patch ------------------------------------------------
        patch = self.cached_img[:,
                  y:y+self.patch_size,
                  x:x+self.patch_size]
        
        # 提取对应的梯度区域
        grad_patch = self.cached_gradient[
                     y:y+self.patch_size,
                     x:x+self.patch_size]
        
        # 计算特征 -----------------------------------------------------
        grad_mean = grad_patch.mean()
        patch_var = torch.var(patch, dim=(1,2), unbiased=False).mean() + 1e-6
        score = 0.6*grad_mean + 0.4*patch_var
        label = 1 if score > self.threshold else 0

        # 数据增强 -----------------------------------------------------
        if self.transform:
            patch = self.transform(patch)

        return patch, label

    def _compute_gradient(self, img_tensor):
        """PyTorch实现的Sobel梯度计算"""
        if img_tensor.shape[0] > 1:  # 多通道转灰度
            gray = img_tensor.mean(dim=0, keepdim=True)
        else:
            gray = img_tensor
        
        # 定义Sobel核
        sobel_x = torch.tensor([[[[1, 0, -1], 
                                [2, 0, -2], 
                                [1, 0, -1]]]], 
                             dtype=torch.float32, device=gray.device)
        
        sobel_y = torch.tensor([[[[1, 2, 1], 
                                [0, 0, 0], 
                                [-1, -2, -1]]]],
                             dtype=torch.float32, device=gray.device)
        
        # 计算梯度
        grad_x = torch.nn.functional.conv2d(gray.unsqueeze(0), sobel_x, padding=1)
        grad_y = torch.nn.functional.conv2d(gray.unsqueeze(0), sobel_y, padding=1)
        
        return torch.sqrt(grad_x.pow(2) + grad_y.pow(2)).squeeze()
    

