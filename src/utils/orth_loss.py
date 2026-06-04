# -*- coding: utf-8 -*-
"""
正交正则化损失函数

核心思想：
---------
正交正则化的目标是：让三个CNN分支学到的特征尽量"互补"而非"冗余"。

问题背景：
---------
在没有约束的情况下，三个分支虽然结构不同，但很可能会学到相似的特征。
因为它们共享相同的输入和相同的分类目标，梯度下降会自然地让所有分支
都去学习"对分类最有帮助"的特征——而这些特征往往是相似的。

正交正则化的解决方案：
-------------------
通过惩罚不同分支特征之间的相关性，迫使各分支探索不同的特征空间。

数学原理：
---------
我们希望：不同分支的输出特征在统计上是正交的（不相关的）。

对于特征矩阵 F1 [B, d1], F2 [B, d2]：
- 如果它们的列向量都两两正交，说明信息是互补的
- 相关性可以通过协方差矩阵 C = F1^T @ F2 来度量
- 协方差矩阵的F范数越小，说明越接近正交

具体步骤：
---------
1. 列归一化：消除特征幅度的影响，只关注方向
   F_normalized = F / ||F||_2

2. 计算交叉协方差矩阵
   C12 = F1^T @ F2  [d1, d2]

3. 计算F范数惩罚
   ||C12||_F^2 = sum(C12^2)  # 所有元素的平方和

4. 总损失
   L_orth = (||C12||_F^2 + ||C13||_F^2 + ||C23||_F^2) / 3

最终损失函数：
------------
L_total = L_CE + lambda * L_orth

其中：
- L_CE: 分类交叉熵损失
- lambda: 正则化强度超参数
- L_orth: 正交正则化损失
"""

import torch
import torch.nn as nn


class OrthogonalLoss(nn.Module):
    """
    正交正则化损失

    通过惩罚不同分支特征之间的交叉协方差F范数，
    鼓励各分支学习统计上互补的特征表示。

    数学公式：
    ---------
    给定三个分支的特征 f1 ∈ R^{B×d1}, f2 ∈ R^{B×d2}, f3 ∈ R^{B×d3}：

    L_orth = (1/3) × (||C12||_F^2 + ||C13||_F^2 + ||C23||_F^2)

    其中：
    - C12 = (f1/||f1||)^T @ (f2/||f2||)  ∈ R^{d1×d2}
    - C13 = (f1/||f1||)^T @ (f3/||f3||)  ∈ R^{d1×d3}
    - C23 = (f2/||f2||)^T @ (f3/||f3||)  ∈ R^{d2×d3}
    - ||C||_F^2 = Σ_i Σ_j C_{ij}^2  (F范数的平方)

    为什么除以3：
    ------------
    因为我们计算了三对分支的协方差，所以取平均。

    Example:
        >>> orth_loss_fn = OrthogonalLoss()
        >>> f1 = torch.randn(128, 64)  # Branch1输出
        >>> f2 = torch.randn(128, 64)  # Branch2输出
        >>> f3 = torch.randn(128, 128) # Branch3输出
        >>> loss = orth_loss_fn(f1, f2, f3)
        >>> print(loss)  # 标量张量
    """

    def __init__(self):
        """初始化正交损失计算器"""
        super(OrthogonalLoss, self).__init__()

    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor
    ) -> torch.Tensor:
        """
        计算正交正则化损失

        Args:
            f1: Branch1的输出特征，shape [batch_size, 64]
            f2: Branch2的输出特征，shape [batch_size, 64]
            f3: Branch3的输出特征，shape [batch_size, 128]

        Returns:
            L_orth: 正交损失标量张量
        """
        # -------------------------------------------------------------------------
        # 第一步：列归一化
        # -------------------------------------------------------------------------
        # 对每个特征矩阵的每一列做L2归一化
        # 这样可以去掉幅度的影响，只关注方向（相关性）
        #
        # F_norm: [B, d]
        # norm: [1, d]  # keepdim保持维度以便广播
        # F_normalized = F / norm
        f1_norm = self._normalize(f1)
        f2_norm = self._normalize(f2)
        f3_norm = self._normalize(f3)

        # -------------------------------------------------------------------------
        # 第二步：计算交叉协方差矩阵
        # -------------------------------------------------------------------------
        # C12 = F1^T @ F2  ∈ R^{d1×d2}
        # C12[i,j] = f1[:, i] 与 f2[:, j] 的点积
        #          = 它们余弦相似度的分子
        #
        # 如果两个分支完全正交，C12 ≈ 0
        c12 = f1_norm.T @ f2_norm  # [64, 64]
        c13 = f1_norm.T @ f3_norm  # [64, 128]
        c23 = f2_norm.T @ f3_norm  # [64, 128]

        # -------------------------------------------------------------------------
        # 第三步：计算F范数平方
        # -------------------------------------------------------------------------
        # ||C||_F^2 = Σ_i Σ_j C_{ij}^2 = ||C||_F.item() ** 2
        #
        # torch.norm(C, p='fro') 计算F范数
        # 再平方得到F范数的平方
        c12_loss = torch.norm(c12, p='fro') ** 2
        c13_loss = torch.norm(c13, p='fro') ** 2
        c23_loss = torch.norm(c23, p='fro') ** 2

        # -------------------------------------------------------------------------
        # 第四步：汇总
        # -------------------------------------------------------------------------
        # 取三对的平均作为最终的正交损失
        orth_loss = (c12_loss + c13_loss + c23_loss) / 3.0

        return orth_loss

    @staticmethod
    def _normalize(F: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        列归一化：对特征矩阵的每一列做L2归一化

        Args:
            F: 特征矩阵，shape [batch_size, feature_dim]
            eps: 防止除零的小常数

        Returns:
            归一化后的特征矩阵，shape不变

        Example:
            >>> F = torch.randn(128, 64)
            >>> F_norm = OrthogonalLoss._normalize(F)
            >>> print(F_norm.shape)  # [128, 64]
            >>> # 验证：每列的L2范数应该约为1
            >>> col_norms = F_norm.norm(dim=0)
            >>> print(col_norms[:5])  # 应该都接近1
        """
        # 计算每列的L2范数
        # F.norm(dim=0): 对dim=0（batch维度）求范数，结果shape=[1, d]
        # keepdim=True: 保持维度以便广播
        norm = F.norm(dim=0, keepdim=True)

        # 归一化：每列除以其L2范数
        return F / (norm + eps)


def compute_pairwise_similarity(
    f1: torch.Tensor,
    f2: torch.Tensor
) -> float:
    """
    计算两个特征矩阵的平均余弦相似度

    当两个特征矩阵维度相同时，计算行向量之间的余弦相似度。
    当维度不同时，使用交叉协方差矩阵F范数来度量。

    Args:
        f1: 特征矩阵1，shape [B, d1]
        f2: 特征矩阵2，shape [B, d2]

    Returns:
        平均余弦相似度（标量）

    Note:
        余弦相似度范围是[-1, 1]：
        - 1: 完全相同方向
        - 0: 正交
        - -1: 完全相反方向

        对于我们希望学到的互补特征，这个值应该越小越好。
    """
    f1_norm = f1 / (f1.norm(dim=1, keepdim=True) + 1e-8)
    f2_norm = f2 / (f2.norm(dim=1, keepdim=True) + 1e-8)

    if f1.shape[1] == f2.shape[1]:
        cosine = (f1_norm * f2_norm).sum(dim=1)
        return cosine.mean().item()
    else:
        cross_cov = f1_norm.T @ f2_norm
        n_elements = cross_cov.numel()
        return (cross_cov.norm(p='fro').item() ** 2) / max(n_elements, 1)


def compute_all_pairwise_similarities(
    f1: torch.Tensor,
    f2: torch.Tensor,
    f3: torch.Tensor
) -> dict:
    """
    计算所有三对分支的余弦相似度

    Args:
        f1, f2, f3: 三个分支的特征矩阵

    Returns:
        包含三对相似度的字典
    """
    sim_12 = compute_pairwise_similarity(f1, f2)
    sim_13 = compute_pairwise_similarity(f1, f3)
    sim_23 = compute_pairwise_similarity(f2, f3)

    return {
        'sim_12': sim_12,  # Branch1 vs Branch2
        'sim_13': sim_13,  # Branch1 vs Branch3
        'sim_23': sim_23,  # Branch2 vs Branch3
        'avg': (sim_12 + sim_13 + sim_23) / 3
    }


if __name__ == '__main__':
    # 测试正交损失
    print("测试正交正则化损失...")

    orth_loss_fn = OrthogonalLoss()

    # 创建测试数据
    # batch_size=128, 各分支特征维度
    f1 = torch.randn(128, 64)
    f2 = torch.randn(128, 64)
    f3 = torch.randn(128, 128)

    # 计算损失
    loss = orth_loss_fn(f1, f2, f3)
    print(f"正交损失: {loss.item():.4f}")

    # 测试完全相同特征的情况（损失应该很大）
    f1_same = torch.randn(128, 64)
    loss_same = orth_loss_fn(f1_same, f1_same, f1_same)
    print(f"相同特征的正交损失: {loss_same.item():.4f}")

    # 测试正交特征的情况（损失应该很小）
    # 创建正交特征：随机生成然后正交化
    f1_orth = torch.randn(128, 64)
    # 使用QR分解得到正交基
    Q, _ = torch.linalg.qr(f1_orth)
    f1_orth = Q[:, :64]
    loss_orth = orth_loss_fn(f1_orth, f1_orth, f1_orth)
    print(f"正交特征的正交损失: {loss_orth.item():.4f}")

    print("\n测试余弦相似度计算...")
    sims = compute_all_pairwise_similarities(f1, f2, f3)
    for key, val in sims.items():
        print(f"  {key}: {val:.4f}")
