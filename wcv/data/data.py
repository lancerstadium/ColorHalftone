

from torch.utils.data import Dataset
from PIL import Image
import os

# 数据集定义
class SingleDataset(Dataset):
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

        # 显示加载进度
        #if idx % 100 == 0:  # 每 100 张输出一次进度
            #print(f"Loading image {idx + 1}/{len(self.image_files)}: {self.image_files[idx]}")
        return image