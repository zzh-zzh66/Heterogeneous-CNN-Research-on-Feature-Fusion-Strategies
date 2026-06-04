# -*- coding: utf-8 -*-
"""
Branch3: 高层语义分支

设计理念：
---------
Branch3专注于捕获图像的高层语义特征，如物体整体轮廓、类别判别性信息等。
使用大卷积核（7×7）建立初始大感受野，配合深层网络（5层卷积），
最终能够捕获整个图像的全局语义信息。

感受野分析（关键！）：
-------------------
- Conv1 (7×7, pad=3): 每个神经元初始"看到"输入的7×7区域 ← 最大卷积核
- Conv2 (3×3, pad=1): 感受野扩大到 7+(3-1) = 9×9
- Conv3 (3×3, pad=1): 感受野扩大到 9+(3-1) = 11×11
- MaxPool (2×2): 尺寸减半，感受野翻倍 → 11×2 = 22×22
- Conv4 (3×3, pad=1): 感受野扩大到 22+(3-1) = 24×24
- Conv5 (3×3, pad=1): 感受野扩大到 24+(3-1) = 26×26
- GAP (1×1): 每个神经元看到整个32×32图像 ← 全局感受野

网络结构：
---------
输入(3,32,32)
    ↓
Conv7×7→32 (大感受野，建立初始语义基础)
    ↓
Conv3×3→64 (×2) (深层抽象，VGG风格)
    ↓
MaxPool2×2 (尺寸减半，感受野翻倍)
    ↓
Conv3×3→128 (×2) (高层特征，维度翻倍)
    ↓
GAP → 输出(128,)

特点：
-----
- 首层7×7：建立最大的初始感受野，最快获取全局信息
- 层数最深（5层）：最深度的特征抽象
- 维度最高（128维）：最丰富的特征表示
- 包含MaxPool：在中层降维，感受野翻倍，加速全局信息汇聚
"""

import torch
import torch.nn as nn


class BranchSemantic(nn.Module):
    """
    高层语义分支

    Architecture:
        Input(3,32,32)
            ↓
        Conv2d(3→32, 7×7) + BatchNorm + ReLU  ← 最大卷积核
            ↓
        Conv2d(32→64, 3×3) + BatchNorm + ReLU
            ↓
        Conv2d(64→64, 3×3) + BatchNorm + ReLU
            ↓
        MaxPool2d(2×2)  ← 关键：尺寸减半，感受野翻倍
            ↓
        Conv2d(64→128, 3×3) + BatchNorm + ReLU  ← 维度翻倍
            ↓
        Conv2d(128→128, 3×3) + BatchNorm + ReLU
            ↓
        AdaptiveAvgPool2d(1×1)
            ↓
        输出: [batch_size, 128]  ← 最高维度

    设计说明：
    ---------
    与Branch1/Branch2的关键区别：

    1. 首层使用7×7卷积核：
       - 这是三个分支中最大的卷积核
       - 初始感受野达到7×7，远大于Branch1(3×3)和Branch2(5×5)
       - 更适合捕获大范围的空间结构

    2. 包含MaxPool层：
       - Branch1和Branch2在所有卷积后都没有MaxPool
       - Branch3在中间层使用MaxPool，使尺寸减半
       - MaxPool后，感受野"物理上"翻倍（每2个像素合并为1个）

    3. 输出维度最高（128 vs 64）：
       - 高层语义信息更丰富，需要更多维度来表达
       - 维度翻倍发生在MaxPool之后，不会大幅增加计算量
    """

    def __init__(self):
        """
        初始化Branch3

        网络结构详解：
        1. Conv1: 3通道 → 32通道，7×7卷积
                   这是本项目最大的卷积核，初始感受野7×7
                   pad=3保证输出尺寸不变（32×32）

        2. Conv2-Conv3: 两层3×3卷积（VGG风格）
                         两个连续的3×3卷积可以近似替代一个5×5卷积
                         但参数量更少（2×9 vs 25），且多了非线性

        3. MaxPool: 关键层！
                    将特征图尺寸减半（32→16）
                    这一步使得感受野"物理上"翻倍
                    是Branch3能够捕获全局信息的关键

        4. Conv4-Conv5: 两层3×3卷积，维度翻倍（64→128）
                         在缩小后的特征图上提取高层语义
                         128维输出是最丰富的特征表示
        """
        super(BranchSemantic, self).__init__()

        # 卷积特征提取器
        self.features = nn.Sequential(
            # Conv1: 3通道 → 32通道，7×7卷积
            # 7×7是三个分支中最大的卷积核，建立最大的初始感受野
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=7,          # 7×7大卷积核
                padding=3               # padding=3 使输出尺寸保持32×32
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Conv2: 32通道 → 64通道，3×3卷积
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Conv3: 64通道 → 64通道，3×3卷积
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # MaxPool: 尺寸减半，32×32 → 16×16
            # 这是Branch3的关键：MaxPool使感受野物理上翻倍
            # 结合前面的卷积，最终感受野可以覆盖整个图像
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv4: 64通道 → 128通道，维度翻倍
            nn.Conv2d(
                in_channels=64,
                out_channels=128,         # 维度翻倍：64→128
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Conv5: 128通道 → 128通道，高层语义精炼
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
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
            输出张量，shape [batch_size, 128]
            注意：输出维度是128，比Branch1和Branch2的64维更高
        """
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return x


class SingleBranch3(nn.Module):
    """
    单分支Branch3分类器（用于独立训练单个分支）
    """

    def __init__(self):
        """初始化单分支分类模型"""
        super(SingleBranch3, self).__init__()
        # 语义特征提取器
        self.backbone = BranchSemantic()
        # 分类头：128维特征 → 10类分类
        self.classifier = nn.Linear(128, 10)

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
