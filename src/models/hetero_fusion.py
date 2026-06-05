# -*- coding: utf-8 -*-
"""
HeteroFusion: 异构CNN特征融合模型，带有惩罚项的正交正则化，强制模型提取互补的不同的特征。

核心思想：
---------
HeteroFusion是本项目的核心模型，它将三个结构差异化的CNN分支
（Branch1/Branch2/Branch3）的输出特征拼接在一起，通过全连接层
进行分类融合。

融合策略：
---------
特征拼接融合 (Concatenation Fusion)
    Branch1输出: [B, 64]  ← 纹理特征
    Branch2输出: [B, 64]  ← 边缘特征
    Branch3输出: [B, 128] ← 语义特征
        ↓
    拼接: Concat([B,64], [B,64], [B,128]) = [B, 256]
        ↓
    全连接: FC(256→128) → ReLU → Dropout
        ↓
    全连接: FC(128→10) → 10类分类输出

为什么拼接有效：
--------------
拼接保留了各分支的完整信息，而不是丢弃一部分。
理论上，如果三个分支学到互补的特征，拼接后的特征应该比
任何一个分支都更丰富、更具有判别性。
"""

import torch
import torch.nn as nn

from .branch1 import BranchTexture
from .branch2 import BranchEdge
from .branch3 import BranchSemantic


class HeteroFusion(nn.Module):
    """
    异构CNN特征融合模型

    Architecture:
        Input(3,32,32)
            │
            ├──→ BranchTexture ──→ [B, 64]
            ├──→ BranchEdge ──────→ [B, 64]
            └──→ BranchSemantic ──→ [B, 128]
                                    │
                               Concat
                                    │
                              [B, 256] ← 64+64+128
                                    │
                               FC(256→128)
                                    │
                               ReLU + Dropout
                                    │
                               FC(128→10)
                                    │
                              输出: [B, 10]

    Attributes:
        branch1: BranchTexture实例，纹理特征提取器
        branch2: BranchEdge实例，边缘特征提取器
        branch3: BranchSemantic实例，语义特征提取器
        classifier: 全连接分类头
        lambda_orth: 正交正则化系数

    Example:
        >>> model = HeteroFusion(lambda_orth=0.05)
        >>> x = torch.randn(4, 3, 32, 32)
        >>> out = model(x)
        >>> print(out.shape)  # torch.Size([4, 10])

        # 获取特征用于计算正交损失
        >>> f1, f2, f3 = model(x, return_features=True)
        >>> print(f1.shape, f2.shape, f3.shape)  # torch.Size([4, 64]) ×3
    """

    def __init__(self, lambda_orth: float = 0.0, dropout: float = 0.3):
        """
        初始化异构融合模型

        Args:
            lambda_orth: 正交正则化系数，控制特征解耦强度
                        - 0: 不使用正交正则化
                        - 0.01-0.1: 推荐范围
                        - 过大的值会损害分类性能

            dropout: Dropout概率，防止过拟合
                     默认0.3，在全连接层使用
        """
        super(HeteroFusion, self).__init__()

        # -------------------------------------------------------------------------
        # 三个异构分支
        # -------------------------------------------------------------------------
        # Branch1: 纹理分支，输出64维
        self.branch1 = BranchTexture()

        # Branch2: 边缘分支，输出64维
        self.branch2 = BranchEdge()

        # Branch3: 语义分支，输出128维
        self.branch3 = BranchSemantic()

        # -------------------------------------------------------------------------
        # 正交正则化系数
        # -------------------------------------------------------------------------
        # 保存lambda_orth，以便在外部访问和记录
        self.lambda_orth = lambda_orth

        # -------------------------------------------------------------------------
        # 分类头
        # -------------------------------------------------------------------------
        # 输入维度: 64(Branch1) + 64(Branch2) + 128(Branch3) = 256
        # 中间维度: 128
        # 输出维度: 10（CIFAR-10分类）
        self.classifier = nn.Sequential(
            # 第一层全连接: 256 → 128
            nn.Linear(in_features=64 + 64 + 128, out_features=128),
            nn.ReLU(inplace=True),       # ReLU激活，增加非线性
            nn.Dropout(p=dropout),       # Dropout防止过拟合

            # 第二层全连接: 128 → 10
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        前向传播

        Args:
            x: 输入图像张量，shape [batch_size, 3, 32, 32]
            return_features: 如果为True，返回三个分支的特征而不进行分类
                            用于计算正交正则化损失

        Returns:
            如果return_features=False:
                分类logits，shape [batch_size, 10]
            如果return_features=True:
                (f1, f2, f3) 三个分支的特征向量元组
        """
        # -------------------------------------------------------------------------
        # 三个分支并行提取特征
        # -------------------------------------------------------------------------
        # Branch1: 提取纹理特征 [B, 64]
        f1 = self.branch1(x)

        # Branch2: 提取边缘特征 [B, 64]
        f2 = self.branch2(x)

        # Branch3: 提取语义特征 [B, 128]
        f3 = self.branch3(x)

        # 如果只需要特征（用于正交损失计算），直接返回
        if return_features:
            return f1, f2, f3

        # -------------------------------------------------------------------------
        # 特征融合
        # -------------------------------------------------------------------------
        # 拼接三个分支的特征
        # torch.cat: 在维度1（特征维度）上拼接
        # 输入: f1=[B,64], f2=[B,64], f3=[B,128]
        # 输出: [B, 64+64+128] = [B, 256]
        concat_features = torch.cat([f1, f2, f3], dim=1)

        # -------------------------------------------------------------------------
        # 分类
        # -------------------------------------------------------------------------
        # 通过全连接层进行分类
        logits = self.classifier(concat_features)

        return logits

    def get_orth_loss(self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor) -> torch.Tensor:
        """
        计算正交正则化损失

        这个方法用于在训练循环中计算正交正则化项。
        具体的数学推导见 utils/orth_loss.py

        Args:
            f1: Branch1的输出特征 [B, 64]
            f2: Branch2的输出特征 [B, 64]
            f3: Branch3的输出特征 [B, 128]

        Returns:
            正交损失标量张量
        """
        # 如果lambda_orth为0或负数，不使用正交正则化
        if self.lambda_orth <= 0:
            # 返回0张量（标量），不影响总损失
            return torch.tensor(0.0, device=f1.device, dtype=f1.dtype)

        # 动态导入避免循环依赖
        from utils.orth_loss import OrthogonalLoss

        # 创建正交损失计算器并计算
        orth_loss_fn = OrthogonalLoss()
        orth_loss = orth_loss_fn(f1, f2, f3)

        return orth_loss


def count_parameters(model: nn.Module) -> int:
    """
    计算模型的可训练参数数量

    Args:
        model: PyTorch模型

    Returns:
        可训练参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: nn.Module, model_name: str = "Model"):
    """
    打印模型信息（参数量等）

    Args:
        model: PyTorch模型
        model_name: 模型名称，用于打印
    """
    total_params = count_parameters(model)
    print(f"{model_name}:")
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 总参数量(M): {total_params / 1e6:.3f}M")

    # 打印各分支参数量
    if hasattr(model, 'branch1'):
        print(f"  - Branch1参数量: {count_parameters(model.branch1):,}")
    if hasattr(model, 'branch2'):
        print(f"  - Branch2参数量: {count_parameters(model.branch2):,}")
    if hasattr(model, 'branch3'):
        print(f"  - Branch3参数量: {count_parameters(model.branch3):,}")
    if hasattr(model, 'classifier'):
        print(f"  - 分类头参数量: {count_parameters(model.classifier):,}")
