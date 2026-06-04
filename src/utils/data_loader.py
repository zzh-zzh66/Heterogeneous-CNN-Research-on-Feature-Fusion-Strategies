# -*- coding: utf-8 -*-
"""
CIFAR-10数据加载模块

本模块负责：
1. CIFAR-10数据集的下载和加载
2. 数据预处理（标准化、数据增强）
3. 返回PyTorch DataLoader供训练使用

CIFAR-10数据集简介：
-------------------
- 60000张32×32彩色图像
- 10个类别：airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50000张训练样本，10000张测试样本
- 由Alex Krizhevsky等人收集
"""

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


# CIFAR-10 类别名称
# 按数据集默认顺序排列
CIFAR10_CLASSES = (
    'airplane',   # 类别0: 飞机
    'automobile', # 类别1: 汽车
    'bird',       # 类别2: 鸟
    'cat',        # 类别3: 猫
    'deer',       # 类别4: 鹿
    'dog',        # 类别5: 狗
    'frog',       # 类别6: 青蛙
    'horse',      # 类别7: 马
    'ship',       # 类别8: 船
    'truck',      # 类别9: 卡车
)


def get_transforms():
    """
    获取数据预处理transform

    预处理流程：
    -----------
    训练集:
        1. RandomCrop: 在图像四周补0像素，再随机裁剪回原尺寸
                      作用：数据增强，让模型学习到略有偏移的物体
        2. RandomHorizontalFlip: 50%概率水平翻转
                                 作用：数据增强，适用于大多数图像
        3. ToTensor: 将HWC格式转为CHW格式，像素值[0,255]→[0,1]
        4. Normalize: 标准化到均值0、方差1
                      使用CIFAR-10的预计算均值和标准差

    测试集:
        1. ToTensor: 格式转换
        2. Normalize: 标准化（与训练集相同参数）
        注意：测试集不进行数据增强！

    CIFAR-10标准化参数：
    ------------------
    均值 (mean):  [0.4914, 0.4822, 0.4465]  # 对应R, G, B三个通道
    标准差 (std): [0.2470, 0.2435, 0.2616]

    这些参数是通过计算整个训练集所有像素的均值和标准差得到的。
    """
    # 训练集transform：包含数据增强
    transform_train = transforms.Compose([
        # 1. 随机裁剪
        # 先在图像四周各补4像素（默认用0填充），再随机裁剪回32×32
        # 这使得模型能够学习到物体在图像中略有偏移的情况
        transforms.RandomCrop(32, padding=4),

        # 2. 随机水平翻转
        # 以50%概率水平翻转图像，增加数据多样性
        # 适用于CIFAR-10中的大多数类别（飞机翻转后仍是飞机）
        transforms.RandomHorizontalFlip(p=0.5),

        # 3. 转换为Tensor
        # 自动完成：HWC(高×宽×通道)→CHW(通道×高×宽)
        #          像素值从[0,255]归一化到[0,1]
        transforms.ToTensor(),

        # 4. 标准化
        # 公式: output = (input - mean) / std
        # 将数据调整为均值为0、标准差为1的分布
        # 有助于加速模型收敛
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],  # R, G, B通道均值
            std=[0.2470, 0.2435, 0.2616]    # R, G, B通道标准差
        ),
    ])

    # 测试集transform：不包含数据增强
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        ),
    ])

    return transform_train, transform_test


def get_cifar10_loaders(
    batch_size: int = 128,
    num_workers: int = 0,
    data_dir: str = './data',
    val_ratio: float = 0.1,
    random_seed: int = 42
):
    """
    获取CIFAR-10训练集、验证集和测试集的DataLoader

    划分方式：
    ---------
    训练集 (50,000) → 训练 (45,000) + 验证 (5,000)  分层抽样
    测试集 (10,000) → 保持原样，仅最终评估使用

    Args:
        batch_size: 每个batch的样本数量，默认128
        num_workers: 数据加载的子进程数，默认0（Windows推荐0）
                    Linux/Mac可以设为2-4加速数据加载
        data_dir: 数据集保存目录，默认'./data'
        val_ratio: 验证集比例，默认0.1（10%）
        random_seed: 分层划分的随机种子，保证可复现

    Returns:
        train_loader: 训练集DataLoader (45,000样本，含数据增强)
        val_loader:   验证集DataLoader (5,000样本，无增强)
        test_loader:  测试集DataLoader (10,000样本，无增强)

    DataLoader参数说明：
    -----------------
    - batch_size: 每批加载的样本数
    - shuffle: 是否在每个epoch打乱数据
               训练集需要shuffle，验证/测试集不需要
    - num_workers: 并行加载数据的进程数
                   Windows下建议0，Linux/Mac可以设更高
    - drop_last: 是否丢弃最后不完整的batch
                 训练集设为True保证每个batch大小一致
                 验证/测试集设为False利用所有数据

    Example:
        >>> train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=128)
        >>> for images, labels in train_loader:
        ...     print(images.shape)  # [128, 3, 32, 32]
        ...     break
    """
    # 获取预处理transform
    transform_train, transform_test = get_transforms()

    # 加载完整训练集（先不用transform，因为要分层划分索引）
    full_train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,           # 数据保存路径
        train=True,               # True=训练集，False=测试集
        download=True,            # 自动下载
        transform=None            # 暂不设置，划分后再分别设置
    )

    # 获取所有标签用于分层抽样
    targets = np.array(full_train_dataset.targets)

    # 分层抽样：按类别比例划分训练/验证索引
    train_indices, val_indices = train_test_split(
        np.arange(len(targets)),
        test_size=val_ratio,
        stratify=targets,
        random_state=random_seed
    )

    # 用索引构造子集，分别设置transform
    train_dataset = Subset(
        CIFAR10WithTransform(full_train_dataset, transform_train),
        train_indices
    )
    val_dataset = Subset(
        CIFAR10WithTransform(full_train_dataset, transform_test),
        val_indices
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,             # 训练集需要打乱
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True            # 丢弃最后不完整的batch
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,            # 验证集无需打乱
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False           # 验证集利用所有数据
    )

    # 测试集
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,              # 测试集
        download=True,
        transform=transform_test   # 不包含数据增强
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,             # 测试集不需要打乱
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False           # 测试集利用所有数据
    )

    return train_loader, val_loader, test_loader


class CIFAR10WithTransform(torch.utils.data.Dataset):
    """
    包装Dataset，允许在Subset之后重新设置transform

    由于Subset会继承原始dataset的transform，
    但训练集和验证集需要不同的transform，所以需要这个包装类。
    """

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
        self.targets = base_dataset.targets

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


if __name__ == '__main__':
    print("测试CIFAR-10数据加载...")
    train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=128)

    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    print(f"测试批次数: {len(test_loader)}")

    # 获取一个batch
    images, labels = next(iter(train_loader))
    print(f"\n一个batch的形状:")
    print(f"  images: {images.shape}")  # [B, 3, 32, 32]
    print(f"  labels: {labels.shape}")  # [B]
