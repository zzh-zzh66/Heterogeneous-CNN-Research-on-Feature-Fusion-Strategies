# -*- coding: utf-8 -*-
"""
SingleLargeCNN: 单分支大CNN基线模型

设计理念：
---------
这是一个参数量与HeteroFusion（~0.6M）相当的单分支CNN，作为核心对比基线。
回答核心问题："为什么要用多分支异构融合，而不是一个更大的单分支网络？"

设计原则：
- 总参数量与M5异构融合持平（~0.6M），确保公平对比
- 使用VGG风格堆叠卷积（3×3），简单可靠
- 包含MaxPool进行渐进式下采样
- 不使用多分支结构、不引入正交正则化

结构：
------
输入(3,32,32)
    ↓
Conv3×3→48 + Conv3×3→48 + MaxPool  (32→16)
    ↓
Conv3×3→96 + Conv3×3→96 + MaxPool (16→8)
    ↓
Conv3×3→192 + Conv3×3→192 + GAP
    ↓
FC(192→128) → ReLU → Dropout(0.3)
    ↓
FC(128→10) → 输出10类

参数量估算：
- Conv: 3×3×(3×48+48×48+48×96+96×96+96×192+192×192) ≈ 644K
- BN: 2×(48+48+96+96+192+192) = 1,344（但只有3个BN层=672）
- FC: 192×128+128 + 128×10+10 ≈ 26K
- Total: ~0.67M
"""

import torch
import torch.nn as nn


class SingleLargeCNN(nn.Module):
    """
    单分支大CNN基线

    参数量目标：≈0.6M，与HeteroFusion(0.63M)持平

    Architecture:
        Input(3,32,32)
            ↓
        Conv3×3→48 + BatchNorm + ReLU
        Conv3×3→48 + BatchNorm + ReLU
        MaxPool2×2  ← 32→16
            ↓
        Conv3×3→96 + BatchNorm + ReLU
        Conv3×3→96 + BatchNorm + ReLU
        MaxPool2×2  ← 16→8
            ↓
        Conv3×3→192 + BatchNorm + ReLU
        Conv3×3→192 + BatchNorm + ReLU
        AdaptiveAvgPool2d(1×1)  ← GAP
            ↓
        FC(192→128) + ReLU + Dropout(0.3)
            ↓
        FC(128→10)
            ↓
        输出: [B, 10]

    参数量：
    - Conv layers: ~0.64M
    - FC layers: 192×128 + 128×10 ≈ 26K
    - Total: ~0.67M

    与HeteroFusion的对比：
    - SingleLargeCNN: 1个"全能专家"（~0.67M参数）
    - HeteroFusion: 3个"专精专家"协作（0.63M参数）
    - 相同的参数量预算下，看哪种结构设计更有效
    """

    def __init__(self, dropout: float = 0.3):
        """
        初始化单分支大CNN

        Args:
            dropout: Dropout概率，默认0.3
        """
        super(SingleLargeCNN, self).__init__()

        # -------------------------------------------------------------------------
        # 特征提取器（VGG风格，逐步增加通道数）
        # -------------------------------------------------------------------------
        self.features = nn.Sequential(
            # Stage 1: 3→48通道，32×32
            nn.Conv2d(3, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),

            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),

            # 下采样：32×32 → 16×16
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 2: 48→96通道，16×16
            nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            # 下采样：16×16 → 8×8
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 3: 96→192通道，8×8
            nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        # -------------------------------------------------------------------------
        # 全局平均池化：将8×8→1×1，输出192维
        # -------------------------------------------------------------------------
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        # -------------------------------------------------------------------------
        # 分类头（由 training.py 统一调用）
        # -------------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：仅提取特征，不经过分类头

        这与 Branch1/2/3 的行为保持一致：
        forward() 返回特征向量，classifier 由 training.py 统一调用

        Args:
            x: 输入图像 [B, 3, 32, 32]

        Returns:
            features: 特征向量 [B, 192]
        """
        # 卷积特征提取
        x = self.features(x)

        # 全局平均池化
        x = self.gap(x)

        # 展平为特征向量 [B, 192]
        x = x.view(x.size(0), -1)

        return x
