# -*- coding: utf-8 -*-
"""
HomoEnsemble: 同构集成模型（对比基准）

核心思想：
---------
HomoEnsemble使用三个结构完全相同的Branch2网络进行特征融合。
它的存在是为了与HeteroFusion（异构融合）进行对比，
以验证"结构差异"是否确实能带来额外的性能提升。

设计对照实验的重要性：
--------------------
科学实验中需要对照组来验证因果关系。
- HeteroFusion（实验组）：3个结构不同的分支
- HomoEnsemble（对照组）：3个结构相同的分支

如果HeteroFusion > HomoEnsemble，则证明：
    "结构差异带来的互补性是有价值的"

如果两者相近，则说明：
    "多分支融合本身就有效，结构差异不是关键因素"

控制变量：
---------
HomoEnsemble和HeteroFusion的主要区别：
- HeteroFusion: Branch1(64) + Branch2(64) + Branch3(128) = 256维
- HomoEnsemble: Branch2(64) + Branch2(64) + Branch2(64) = 192维

注意：由于维度不同，融合层的输入维度也不同：
- HeteroFusion: 256 → 128
- HomoEnsemble: 192 → 128

但两者参数量相近，可以进行公平的性能对比。
"""

import torch
import torch.nn as nn

from .branch2 import BranchEdge


class HomoEnsemble(nn.Module):
    """
    同构集成模型（3个相同的Branch2）

    Architecture:
        Input(3,32,32)
            │
            ├──→ BranchEdge ──→ [B, 64]
            ├──→ BranchEdge ──→ [B, 64]
            └──→ BranchEdge ──→ [B, 64]
                                 │
                            Concat
                                 │
                      [B, 64+64+64] = [B, 192]
                                 │
                            FC(192→128)
                                 │
                          ReLU + Dropout
                                 │
                            FC(128→10)
                                 │
                        输出: [B, 10]

    Design Note:
        选择Branch2作为同构集成的基础分支，是因为：
        1. Branch2的参数量适中（比Branch3小，比Branch1大）
        2. Branch2的性能在三个单分支中处于中间水平
        3. 用作对比基准最合适
    """

    def __init__(self, dropout: float = 0.3):
        """
        初始化同构集成模型

        Args:
            dropout: Dropout概率，防止过拟合
        """
        super(HomoEnsemble, self).__init__()

        # -------------------------------------------------------------------------
        # 三个同构分支（都是BranchEdge）
        # -------------------------------------------------------------------------
        # 注意：三个分支虽然结构相同，但初始化权重不同
        # 因此即使输入相同，训练后学到的特征也会有差异
        # 这就是集成学习的基本原理：多样性 + 聚合

        self.branch_a = BranchEdge()
        self.branch_b = BranchEdge()  # 与branch_a结构完全相同
        self.branch_c = BranchEdge()  # 与branch_a、branch_b结构完全相同

        # -------------------------------------------------------------------------
        # 分类头
        # -------------------------------------------------------------------------
        # 输入维度: 64(a) + 64(b) + 64(c) = 192
        # 注意：比HeteroFusion的256维少，因为都是Branch2（各64维）
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64 * 3, out_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        前向传播

        Args:
            x: 输入图像张量，shape [batch_size, 3, 32, 32]
            return_features: 如果为True，返回三个分支的特征

        Returns:
            如果return_features=False: 分类logits [batch_size, 10]
            如果return_features=True: (f_a, f_b, f_c)
        """
        # -------------------------------------------------------------------------
        # 三个分支并行提取特征
        # -------------------------------------------------------------------------
        # 虽然三个分支结构相同，但初始化权重不同
        # 经过训练后，它们的输出会略有差异
        # 这种差异是集成学习提升性能的关键来源

        f_a = self.branch_a(x)
        f_b = self.branch_b(x)
        f_c = self.branch_c(x)

        if return_features:
            return f_a, f_b, f_c

        # -------------------------------------------------------------------------
        # 特征融合
        # -------------------------------------------------------------------------
        concat_features = torch.cat([f_a, f_b, f_c], dim=1)

        # -------------------------------------------------------------------------
        # 分类
        # -------------------------------------------------------------------------
        logits = self.classifier(concat_features)

        return logits

    def get_orth_loss(self, f_a: torch.Tensor, f_b: torch.Tensor,
                     f_c: torch.Tensor) -> torch.Tensor:
        """
        计算正交正则化损失（同构模型也支持，但通常不用）

        注意：同构集成模型通常不使用正交正则化，
        因为三个分支本来就相同，正交约束没有意义。

        这个方法仅为保持接口一致性。
        """
        # 同构模型不使用正交正则化
        return torch.tensor(0.0, device=f_a.device, dtype=f_a.dtype)
