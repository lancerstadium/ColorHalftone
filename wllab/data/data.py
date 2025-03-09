from torch.utils.data import Dataset
from torch.utils.data import DataLoader
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
    def __init__(self, image_dir, transform=None, max_images=800, 
                patch_size=32, threshold=0.7, balance_ratio=0.5):
        """
        balance_ratio: 正样本在每张图像中的最小比例 (0.3表示至少30%的样本是正类)
        """
        self.image_dir = image_dir
        self.image_list = sorted([f for f in os.listdir(image_dir) 
                                if f.lower().endswith(('png', 'jpg', 'jpeg'))])[:max_images]
        self.patch_size = patch_size
        self.threshold = threshold
        self.transform = transform
        self.balance_ratio = balance_ratio
        
        # 存储优化后的patch位置信息
        self.patch_positions = []  # (img_idx, y, x, is_positive)
        self.class_weights = []     # 用于后续加权采样
        
        # 第一遍：分析每张图像的样本分布
        self._precompute_balance()
        
        # 第二遍：构建平衡数据集
        self._build_balanced_set()
        
        # 缓存系统
        self.current_img_idx = -1
        self.cached_img = None
        self.cached_gradient = None

    def _precompute_balance(self):
        """预计算每张图像的正负样本分布"""
        self.image_stats = []
        for img_idx, img_name in enumerate(self.image_list):
            img_path = os.path.join(self.image_dir, img_name)
            with Image.open(img_path) as img:
                w, h = img.size
                
            # 生成所有可能位置
            y_coords = range(0, h - self.patch_size + 1, self.patch_size)
            x_coords = range(0, w - self.patch_size + 1, self.patch_size)
            total_patches = len(y_coords) * len(x_coords)
            
            # 估计正样本比例（避免全图扫描）
            # 使用中心区域采样快速评估
            sample_positions = [
                (h//2 - self.patch_size, w//2 - self.patch_size),
                (h//2, w//2),
                (0, 0),
                (h - self.patch_size, w - self.patch_size)
            ]
            positive_count = sum(self._is_positive(img_path, y, x) 
                              for y, x in sample_positions if y >=0 and x >=0)
            
            estimated_ratio = positive_count / len(sample_positions) if sample_positions else 0
            self.image_stats.append({
                'size': (w, h),
                'estimated_ratio': max(0.1, min(0.9, estimated_ratio))  # 防止极端值
            })

    def _build_balanced_set(self):
        """构建平衡的样本集"""
        for img_idx, img_name in enumerate(self.image_list):
            img_path = os.path.join(self.image_dir, img_name)
            w, h = self.image_stats[img_idx]['size']
            est_ratio = self.image_stats[img_idx]['estimated_ratio']
            
            # 动态计算采样步长
            base_stride = self.patch_size
            pos_stride = int(base_stride * (1 - est_ratio))
            neg_stride = int(base_stride * est_ratio)
            
            # 生成候选位置
            pos_samples = []
            neg_samples = []
            for y in range(0, h - self.patch_size + 1, base_stride):
                for x in range(0, w - self.patch_size + 1, base_stride):
                    if self._is_positive(img_path, y, x):
                        pos_samples.append((img_idx, y, x))
                    else:
                        neg_samples.append((img_idx, y, x))
            
            # 动态平衡采样
            target_pos = int(len(pos_samples + neg_samples) * self.balance_ratio)
            target_neg = int(target_pos * (1 - self.balance_ratio) / self.balance_ratio)
            
            # 保证最小样本量
            target_pos = max(target_pos, 1)
            target_neg = max(target_neg, 1)
            
            # 随机选择样本
            selected_pos = random.sample(pos_samples, min(target_pos, len(pos_samples)))
            selected_neg = random.sample(neg_samples, min(target_neg, len(neg_samples)))
            

            # 添加到patch位置列表
            if len(selected_pos) == 0 or len(selected_neg) == 0:
                continue

            self.patch_positions.extend(selected_pos + selected_neg)
            self.class_weights.extend([1.0/len(selected_pos)]*len(selected_pos) + 
                                   [1.0/len(selected_neg)]*len(selected_neg))

    def _is_positive(self, img_path, y, x):
        """快速判断是否为阳性样本"""
        try:
            with Image.open(img_path) as img:
                patch = img.crop((x, y, x+self.patch_size, y+self.patch_size))
                patch_tensor = torch.from_numpy(np.array(patch)).permute(2,0,1).float() / 255.0
                
                # 快速特征计算
                if patch_tensor.shape[0] > 1:
                    gray = patch_tensor.mean(dim=0, keepdim=True)
                else:
                    gray = patch_tensor
                
                # 近似梯度计算（仅用中心区域）
                center = gray[:, self.patch_size//4:3*self.patch_size//4, 
                             self.patch_size//4:3*self.patch_size//4]
                grad_x = center[:, :, 2:] - center[:, :, :-2]
                grad_y = center[:, 2:, :] - center[:, :-2, :]
                grad_magnitude = (grad_x.abs() + grad_y.abs()).mean()
                
                # 快速方差计算
                var = patch_tensor.view(patch_tensor.shape[0], -1).var(dim=1).mean()
                
                return (0.6*grad_magnitude + 0.4*var) > self.threshold
        except:
            return False

    def __len__(self):
        return len(self.patch_positions)

    def __getitem__(self, idx):
        img_idx, y, x = self.patch_positions[idx]
        
        # 按需加载图像 -------------------------------------------------
        if img_idx != self.current_img_idx:
            img_path = os.path.join(self.image_dir, self.image_list[img_idx])
            with Image.open(img_path) as img:
                img_array = np.array(img.convert('RGB'))
            
            self.cached_img = torch.from_numpy(img_array).permute(2,0,1).float() / 255.0
            self.cached_gradient = self._compute_gradient(self.cached_img)
            self.current_img_idx = img_idx

        # 动态提取patch和标签 ------------------------------------------
        patch = self.cached_img[:, y:y+self.patch_size, x:x+self.patch_size]
        grad_patch = self.cached_gradient[y:y+self.patch_size, x:x+self.patch_size]
        
        # 计算精确标签
        grad_mean = grad_patch.mean()
        patch_var = torch.var(patch, dim=(1,2)).mean() + 1e-6
        label = 1 if (0.6*grad_mean + 0.4*patch_var) > self.threshold else 0

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
    
    def get_balanced_loader(self, batch_size=64):
        """获取平衡数据的DataLoader"""
        weights = torch.DoubleTensor(self.class_weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        return DataLoader(self, batch_size=batch_size, sampler=sampler, num_workers=4)

