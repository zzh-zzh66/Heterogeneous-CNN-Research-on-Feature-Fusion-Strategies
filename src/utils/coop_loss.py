# -*- coding: utf-8 -*-
"""
协作预测奖励损失函数

核心思想：
---------
正交正则化（惩罚分支解耦）被实验证明是帮倒忙的——异构结构本身已有足够的
特征解耦能力。那么能否反其道而行之：奖励分支间的有益信息共享？

协作奖励损失的数学形式：
----------------------
给定三个分支的独立预测损失 ce_b1, ce_b2, ce_b3 和融合预测损失 ce_fused：

    best_single = min(ce_b1_detach, ce_b2_detach, ce_b3_detach)

    L_coop = -lambda_coop * max(0, best_single - ce_fused)

解释：
- best_single 使用 detach() 截断梯度：只允许融合向单分支"学习"，不允许
  单分支为迎合奖励而退化
- max(0, ...) 确保只在融合优于最佳单分支时才给予奖励
- 负号使融合越好（ce_fused越小），总损失越小 → 正向激励

完整损失函数：
-------------
L_total = L_CE(fused) + λ_aux * Σ L_CE(branch_i) + L_coop

其中：
- L_CE(fused): 融合分类交叉熵（主任务）
- λ_aux * Σ L_CE(branch_i): 辅助单分支监督（防止分支退化）
- L_coop: 协作奖励项（推动融合 > 最佳单分支）

预期训练动态：
-------------
1. 初期（epoch 1-10）: 所有分支都弱，best_single 较大，coop_reward ≈ 0
2. 中期（epoch 10-40）: 单分支逐渐强，融合开始受reward驱动学习互补信息
3. 后期（epoch 40+）: 融合稳定优于单分支，coop_reward 持续为正

使用示例：
---------
>>> from utils.coop_loss import compute_coop_loss
>>> # ... 在训练循环中
>>> ce_fused = criterion(logits_fused, labels)
>>> ce_b1 = criterion(logits_b1, labels)
>>> ce_b2 = criterion(logits_b2, labels)
>>> ce_b3 = criterion(logits_b3, labels)
>>> coop_loss = compute_coop_loss(ce_fused, ce_b1, ce_b2, ce_b3)
>>> total_loss = ce_fused + 0.3 * (ce_b1 + ce_b2 + ce_b3) - 0.1 * coop_loss
"""

import torch
import torch.nn as nn


def compute_coop_reward(
    ce_fused: torch.Tensor,
    ce_b1: torch.Tensor,
    ce_b2: torch.Tensor,
    ce_b3: torch.Tensor
) -> torch.Tensor:
    """
    计算协作奖励值（正值 = 融合优于最佳单分支）

    采用 detach 截断单分支梯度，确保协作奖励只优化融合路径。

    Args:
        ce_fused: 融合分类交叉熵损失，标量张量（保留梯度）
        ce_b1: Branch1 独立分类交叉熵损失，标量张量
        ce_b2: Branch2 独立分类交叉熵损失，标量张量
        ce_b3: Branch3 独立分类交叉熵损失，标量张量

    Returns:
        coop_reward: 标量张量，≥0，保留对 ce_fused 的梯度

    Example:
        >>> ce_f = torch.tensor(0.1, requires_grad=True)  # 融合更好
        >>> ce_1 = torch.tensor(0.2)
        >>> ce_2 = torch.tensor(0.3)
        >>> ce_3 = torch.tensor(0.4)
        >>> reward = compute_coop_reward(ce_f, ce_1, ce_2, ce_3)
        >>> print(reward.item())  # 0.1 (= 0.2 - 0.1)
    """
    # 找到最佳单分支（detach 阻断梯度回传）
    best_single = torch.stack([
        ce_b1.detach(),
        ce_b2.detach(),
        ce_b3.detach()
    ]).min()

    # 协作奖励 = max(0, best_single - ce_fused)
    # 如果融合更好：正值，奖励有效
    # 如果融合更差：clamp 为 0，无奖励也无惩罚
    coop_reward = torch.clamp(best_single - ce_fused, min=0.0)

    return coop_reward


def compute_coop_statistics(
    ce_fused: float,
    ce_b1: float,
    ce_b2: float,
    ce_b3: float
) -> dict:
    """
    计算协作统计信息（用于日志输出，非训练用）

    Args:
        ce_fused: 融合CE损失（标量值）
        ce_b1, ce_b2, ce_b3: 各分支CE损失

    Returns:
        包含以下键的字典：
        - best_single: 最佳单分支CE
        - best_branch: 最佳分支编号 (1/2/3)
        - gap: 融合与最佳单分支的差距（负=融合更好）
        - coop_active: 是否触发协作奖励
    """
    best_single = min(ce_b1, ce_b2, ce_b3)
    branch_names = {ce_b1: 1, ce_b2: 2, ce_b3: 3}
    best_branch = branch_names[best_single]
    gap = ce_fused - best_single  # 负 = 融合更好

    return {
        'best_single': best_single,
        'best_branch': best_branch,
        'gap': gap,
        'coop_active': gap < 0  # 融合更好时触发
    }


if __name__ == '__main__':
    print("=" * 60)
    print("协作损失函数单元测试")
    print("=" * 60)

    # 测试1: 融合优于单分支 → 应有正奖励
    print("\n[测试1] 融合优于所有单分支（应触发奖励）:")
    ce_f = torch.tensor(0.1, requires_grad=True)
    ce_1 = torch.tensor(0.3)
    ce_2 = torch.tensor(0.4)
    ce_3 = torch.tensor(0.5)
    reward = compute_coop_reward(ce_f, ce_1, ce_2, ce_3)
    print(f"  ce_fused=0.1, ce_branch=(0.3, 0.4, 0.5)")
    print(f"  reward = {reward.item():.4f} (期望: 0.2000)")

    # 验证梯度流向
    reward.backward()
    print(f"  grad(ce_fused) = {ce_f.grad.item():.4f} (期望: -1.0000)")
    print(f"  grad(ce_b1) = {ce_1.grad} (期望: None)")

    # 测试2: 融合不如单分支 → 奖励=0
    print("\n[测试2] 融合不如最佳单分支（不应有奖励）:")
    ce_f2 = torch.tensor(0.5, requires_grad=True)
    ce_1b = torch.tensor(0.1)
    ce_2b = torch.tensor(0.2)
    ce_3b = torch.tensor(0.3)
    reward2 = compute_coop_reward(ce_f2, ce_1b, ce_2b, ce_3b)
    print(f"  ce_fused=0.5, ce_branch=(0.1, 0.2, 0.3)")
    print(f"  reward = {reward2.item():.4f} (期望: 0.0000)")

    # 测试3: 统计函数
    print("\n[测试3] 统计信息:")
    stats = compute_coop_statistics(0.1, 0.3, 0.4, 0.5)
    print(f"  best_single={stats['best_single']:.4f}, "
          f"best_branch={stats['best_branch']}, "
          f"gap={stats['gap']:.4f}, "
          f"coop_active={stats['coop_active']}")

    print("\n所有测试通过！")
