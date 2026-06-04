# -*- coding: utf-8 -*-
"""
HeteroFusionCoop: 带协作预测奖励的异构CNN特征融合模型
因为惩罚项的实验结果反而不如λ=0的情况，所以突发奇想，试试奖励机制，看看能不能优化结果

核心思想：
---------
在 HeteroFusion 基础上，为每个分支增加独立的轻量级分类头。
训练时不仅优化融合预测，还鼓励融合预测优于各分支独立预测，
以奖励"1+1+1 > 3"的协作效应。

与 HeteroFusion 的区别：
-----------------------
1. 增加了三个分支独立分类头（参数增量仅 ~2,590，+0.41%）
2. 训练时多出一项协作奖励损失（鼓励融合优于单分支）

数学原理：
---------
L_total = L_CE(fused) + λ_aux * [L_CE(b1) + L_CE(b2) + L_CE(b3)]
          - λ_coop * max(0, min(L_CE(b1), L_CE(b2), L_CE(b3))_detach - L_CE(fused))

解释：
- 前两项：融合分类 + 辅助单分支分类（确保各分支不退化）
- 第三项：协作奖励——融合越好于最优单分支，奖励越大
  - best_single 使用 detach()，保证梯度只优化融合方向
  - clamp(min=0) 确保融合更差时不会惩罚

预期行为：
---------
- 训练初期：单分支弱，协作奖励≈0，相当于标准训练
- 训练中后期：单分支逐渐强大，协作奖励生效，推动融合学习互补信息
- 最终：融合准确率应高于 λ=0 的纯异构融合
"""

import torch
import torch.nn as nn

from .branch1 import BranchTexture
from .branch2 import BranchEdge
from .branch3 import BranchSemantic


class HeteroFusionCoop(nn.Module):
    """
    带协作预测奖励的异构融合模型

    Architecture:
        Input(3,32,32)
            │
            ├──→ BranchTexture ──→ [B, 64] ──→ head_b1(64→10)
            ├──→ BranchEdge ──────→ [B, 64] ──→ head_b2(64→10)
            └──→ BranchSemantic ──→ [B, 128] ─→ head_b3(128→10)
                                    │
                               Concat
                                    │
                              [B, 256]
                                    │
                               FC(256→128)
                                    │
                               ReLU + Dropout
                                    │
                               FC(128→10)
                                    │
                              输出: [B, 10]

    Attributes:
        branch1/2/3: 三个异构特征提取分支
        head_b1/b2/b3: 各分支独立分类头（轻量级 Linear 层）
        classifier: 融合特征分类头
        lambda_orth: 正交正则化系数（保留但默认不使用）
        lambda_coop: 协作奖励系数

    Example:
        >>> model = HeteroFusionCoop(lambda_coop=0.1, lambda_aux=0.3)
        >>> x = torch.randn(4, 3, 32, 32)
        >>>
        >>> # 仅获取融合logits（评估用）
        >>> out = model(x)
        >>> print(out.shape)  # torch.Size([4, 10])
        >>>
        >>> # 获取特征和单分支logits（训练用）
        >>> f1, f2, f3, b1_out, b2_out, b3_out = model(x, return_features=True, return_branch_logits=True)
    """

    def __init__(
        self,
        lambda_orth: float = 0.0,
        lambda_coop: float = 0.1,
        lambda_aux: float = 0.3,
        dropout: float = 0.3
    ):
        """
        初始化协作融合模型

        Args:
            lambda_orth: 正交正则化系数（保留接口，默认为0不使用）
            lambda_coop: 协作奖励强度，控制融合优于单分支的奖励幅度
                         推荐范围 0.05~0.5，默认 0.1
            lambda_aux: 辅助单分支分类损失权重
                        推荐范围 0.1~0.5，默认 0.3
            dropout: Dropout概率，默认0.3
        """
        super(HeteroFusionCoop, self).__init__()

        # -------------------------------------------------------------------------
        # 三个异构分支（与原HeteroFusion完全相同）
        # -------------------------------------------------------------------------
        self.branch1 = BranchTexture()   # [B, 64]
        self.branch2 = BranchEdge()      # [B, 64]
        self.branch3 = BranchSemantic()  # [B, 128]

        # -------------------------------------------------------------------------
        # 分支独立分类头（新增，轻量级）
        # -------------------------------------------------------------------------
        # 每个分支一个 Linear(特征_dim → 10)，参数极少
        self.head_b1 = nn.Linear(64, 10)   # 650 参数
        self.head_b2 = nn.Linear(64, 10)   # 650 参数
        self.head_b3 = nn.Linear(128, 10)  # 1290 参数

        # -------------------------------------------------------------------------
        # 融合分类头（与原HeteroFusion完全相同）
        # -------------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64 + 64 + 128, out_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=128, out_features=10)
        )

        # -------------------------------------------------------------------------
        # 超参数
        # -------------------------------------------------------------------------
        self.lambda_orth = lambda_orth
        self.lambda_coop = lambda_coop
        self.lambda_aux = lambda_aux

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_branch_logits: bool = False
    ):
        """
        前向传播

        Args:
            x: 输入图像张量，shape [batch_size, 3, 32, 32]
            return_features: 是否返回中间特征向量
            return_branch_logits: 是否返回各分支独立分类logits（训练时用）

        Returns:
            默认模式: logits [batch_size, 10]
            return_features=True:
                (f1, f2, f3) 三个分支特征元组
            return_features=True + return_branch_logits=True:
                (f1, f2, f3, logits_b1, logits_b2, logits_b3)
        """
        # -------------------------------------------------------------------------
        # 三个分支并行提取特征
        # -------------------------------------------------------------------------
        f1 = self.branch1(x)  # [B, 64]
        f2 = self.branch2(x)  # [B, 64]
        f3 = self.branch3(x)  # [B, 128]

        # 如果只需要特征和单分支logits（训练时用）
        if return_features and return_branch_logits:
            logits_b1 = self.head_b1(f1)  # [B, 10]
            logits_b2 = self.head_b2(f2)  # [B, 10]
            logits_b3 = self.head_b3(f3)  # [B, 10]
            return f1, f2, f3, logits_b1, logits_b2, logits_b3

        # 如果只需要特征（用于正交损失计算等）
        if return_features:
            return f1, f2, f3

        # -------------------------------------------------------------------------
        # 默认模式：融合分类
        # -------------------------------------------------------------------------
        concat_features = torch.cat([f1, f2, f3], dim=1)  # [B, 256]
        logits = self.classifier(concat_features)          # [B, 10]
        return logits

    def get_orth_loss(self, f1, f2, f3):
        """计算正交正则化损失（保留接口，异步时按需动态导入）"""
        if self.lambda_orth <= 0:
            return torch.tensor(0.0, device=f1.device, dtype=f1.dtype)
        from utils.orth_loss import OrthogonalLoss
        orth_loss_fn = OrthogonalLoss()
        return orth_loss_fn(f1, f2, f3)


def count_parameters(model: nn.Module) -> int:
    """计算模型的可训练参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: nn.Module, model_name: str = "Model"):
    """打印模型信息"""
    total_params = count_parameters(model)
    print(f"{model_name}:")
    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 总参数量(M): {total_params / 1e6:.3f}M")

    if hasattr(model, 'branch1'):
        print(f"  - Branch1 参数量: {count_parameters(model.branch1):,}")
    if hasattr(model, 'branch2'):
        print(f"  - Branch2 参数量: {count_parameters(model.branch2):,}")
    if hasattr(model, 'branch3'):
        print(f"  - Branch3 参数量: {count_parameters(model.branch3):,}")
    if hasattr(model, 'head_b1'):
        head_params = (count_parameters(model.head_b1)
                       + count_parameters(model.head_b2)
                       + count_parameters(model.head_b3))
        print(f"  - 单分支分类头参数量: {head_params:,}")
    if hasattr(model, 'classifier'):
        print(f"  - 融合分类头参数量: {count_parameters(model.classifier):,}")
