# -*- coding: utf-8 -*-
"""
Branch1: 浅层纹理分支

设计理念：
---------
Branch1专注于捕获图像的浅层局部特征，如纹理、边缘方向、细小色块等。
由于使用小卷积核（3×3）和较浅的网络结构（仅3层卷积），
每个神经元的感受野较小，天然适合提取局部细节信息。

感受野分析：
-----------
- Conv1 (3×3, pad=1): 每个神经元"看到"输入的3×3区域
- Conv2 (3×3, pad=1): 感受野扩大到 3+(3-1) = 5×5
- Conv3 (3×3, pad=1): 感受野扩大到 5+(3-1) = 7×7

网络结构：
---------
输入(3,32,32) → Conv3×3→32 → Conv3×3→64 → Pool2×2 → Conv3×3→64 → GAP → 输出(64,)

特点：
-----
- 卷积核最小（3×3）：最精细的局部特征提取
- 层数最少（3层）：保留最多的原始细节信息
- 感受野最小（最终约7×7）：专注于局部纹理
"""

import torch
import torch.nn as nn


class BranchTexture(nn.Module):
    """
    浅层纹理分支

    Architecture:
        Input(3,32,32)
            ↓
        Conv2d(3→32, 3×3) + BatchNorm + ReLU
            ↓
        Conv2d(32→64, 3×3) + BatchNorm + ReLU
            ↓
        MaxPool2d(2×2)  ← 尺寸减半: 32→16
            ↓
        Conv2d(64→64, 3×3) + BatchNorm + ReLU
            ↓
        AdaptiveAvgPool2d(1×1)  ← 全局池化
            ↓
        输出: [batch_size, 64]

    Attributes:
        features: 卷积层序列
        gap: 全局平均池化层

    Example:
        >>> branch = BranchTexture()
        >>> x = torch.randn(4, 3, 32, 32)  # 4张32×32的RGB图像
        >>> out = branch(x)
        >>> print(out.shape)  # torch.Size([4, 64])
    """

    def __init__(self):
        """
        初始化Branch1

        网络结构详解：
        1. Conv1: 3通道输入，32通道输出，3×3卷积，padding=1保持尺寸不变
                   BatchNorm对32个特征图分别做标准化
                   ReLU增加非线性

        2. Conv2: 32通道输入，64通道输出，3×3卷积
                   将特征维度扩展到64维

        3. MaxPool: 2×2最大池化，将特征图尺寸从32×32降为16×16
                    同时减少计算量和参数量

        4. Conv3: 64通道输入，64通道输出，3×3卷积
                   保持64维特征，继续提取局部纹理

        5. GAP: 全局平均池化，将任意尺寸的特征图变为1×1
                输出维度=通道数=64
        """
        super(BranchTexture, self).__init__()

        # 第一层：3通道 → 32通道，3×3卷积
        # Sequential容器按顺序执行其中的层
        self.features = nn.Sequential(
            # Conv1: 输入3通道，输出32通道，卷积核3×3，padding=1保证尺寸不变
            nn.Conv2d(
                in_channels=3,          # 输入通道数：RGB图像=3
                out_channels=32,         # 输出通道数：32个卷积核产生32个特征图
                kernel_size=3,           # 卷积核大小：3×3
                padding=1                # 边缘填充：左右上下各填充1像素
                                        # 输出尺寸 = 输入尺寸 + 2*padding - kernel_size + 1
                                        #           = 32 + 2*1 - 3 + 1 = 32 (保持不变)
            ),
            nn.BatchNorm2d(32),         # 对32个特征图做BatchNorm，加速训练
            nn.ReLU(inplace=True),       # ReLU激活函数，inplace=True节省内存

            # Conv2: 32通道 → 64通道，继续卷积
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # MaxPool: 2×2最大池化，尺寸减半
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 输出尺寸: 32→16，特征图数量保持64

            # Conv3: 保持64通道，继续提取局部特征
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 全局平均池化层：将任意尺寸的特征图压缩为1×1
        # 输出维度 = 输入通道数 = 64
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，shape [batch_size, 3, 32, 32]
               - batch_size: 批次大小
               - 3: RGB三个通道
               - 32×32: 图像宽高

        Returns:
            输出张量，shape [batch_size, 64]
               - 64: 提取的纹理特征维度
        """
        # 依次经过卷积层提取特征
        x = self.features(x)
        # 全局平均池化：将特征图压缩为向量
        x = self.gap(x)
        # 展平：从 [B, 64, 1, 1] 变为 [B, 64]
        x = x.view(x.size(0), -1)
        return x


class SingleBranch1(nn.Module):
    """
    单分支Branch1分类器（用于独立训练单个分支）

    这是Branch1加上一个分类头的完整模型，
    用于实验中的单分支baseline对比。
    """

    def __init__(self):
        """
        初始化单分支分类模型

        结构：Branch1 → 全连接层 → 10分类输出
        """
        super(SingleBranch1, self).__init__()
        # 纹理特征提取器
        self.backbone = BranchTexture()
        # 分类头：64维特征 → 10类分类
        self.classifier = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像 [batch_size, 3, 32, 32]

        Returns:
            分类 logits [batch_size, 10]
        """
        # 提取特征
        features = self.backbone(x)
        # 分类
        logits = self.classifier(features)
        return logits
