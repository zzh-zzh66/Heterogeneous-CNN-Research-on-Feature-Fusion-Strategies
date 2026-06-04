# -*- coding: utf-8 -*-
"""
Branch2: 中层边缘/形状分支

设计理念：
---------
Branch2专注于捕获图像的中层特征，如边缘组合、局部形状、部件级结构等。
使用中等卷积核（5×5）和中等网络深度（4层卷积），
在保持一定局部细节的同时，能够提取更大范围的形状信息。

感受野分析：
-----------
- Conv1 (5×5, pad=2): 每个神经元"看到"输入的5×5区域
- Conv2 (5×5, pad=2): 感受野扩大到 5+(5-1) = 9×9
- Conv3 (5×5, pad=2): 感受野扩大到 9+(5-1) = 13×13
- Conv4 (5×5, pad=2): 感受野扩大到 13+(5-1) = 17×17

网络结构：
---------
输入(3,32,32) → Conv5×5→32 → Conv5×5→64 → Pool2×2 → Conv5×5→64×2 → GAP → 输出(64,)

特点：
-----
- 卷积核中等（5×5）：比Branch1更大的感受野
- 层数中等（4层）：比Branch1更深的特征抽象
- 感受野中等（最终约17×17）：适合捕获部件级形状
"""

import torch
import torch.nn as nn


class BranchEdge(nn.Module):
    """
    中层边缘/形状分支

    Architecture:
        Input(3,32,32)
            ↓
        Conv2d(3→32, 5×5) + BatchNorm + ReLU
            ↓
        Conv2d(32→64, 5×5) + BatchNorm + ReLU
            ↓
        MaxPool2d(2×2)  ← 尺寸减半: 32→16
            ↓
        Conv2d(64→64, 5×5) + BatchNorm + ReLU
            ↓
        Conv2d(64→64, 5×5) + BatchNorm + ReLU
            ↓
        AdaptiveAvgPool2d(1×1)  ← 全局池化
            ↓
        输出: [batch_size, 64]

    设计说明：
    ---------
    与Branch1的主要区别：
    1. 卷积核从3×3增大到5×5，每个神经元初始感受野更大
    2. 层数从3层增加到4层，可以提取更复杂的形状特征
    3. 5×5卷积可以用两个3×3卷积近似替代（VGG风格），
       但5×5在早期层可以更直接地建立大感受野
    """

    def __init__(self):
        """
        初始化Branch2

        网络结构详解：
        1. Conv1: 3通道 → 32通道，5×5卷积
                   5×5卷积的感受野比3×3更大，更适合捕获边缘信息

        2. Conv2: 32通道 → 64通道，5×5卷积
                   将特征维度扩展到64维

        3. MaxPool: 2×2最大池化，尺寸减半
                     减少计算量，同时略微增大有效感受野

        4. Conv3-Conv4: 两层5×5卷积
                          继续在缩小后的特征图上提取形状特征
                          两层的堆叠可以捕获更复杂的形状组合

        5. GAP: 全局平均池化
        """
        super(BranchEdge, self).__init__()

        # 卷积特征提取器
        self.features = nn.Sequential(
            # Conv1: 3通道 → 32通道，5×5卷积
            # padding=2 保证输出尺寸与输入相同（32×32）
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=5,          # 5×5卷积核，比Branch1的3×3更大
                padding=2                # padding=2 使输出尺寸保持32×32
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Conv2: 32通道 → 64通道
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # MaxPool: 尺寸减半，32×32 → 16×16
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv3: 64通道 → 64通道
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Conv4: 64通道 → 64通道，继续提取形状特征
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 全局平均池化层
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，shape [batch_size, 3, 32, 32]

        Returns:
            输出张量，shape [batch_size, 64]
        """
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return x


class SingleBranch2(nn.Module):
    """
    单分支Branch2分类器（用于独立训练单个分支）
    """

    def __init__(self):
        """初始化单分支分类模型"""
        super(SingleBranch2, self).__init__()
        # 边缘特征提取器
        self.backbone = BranchEdge()
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
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
