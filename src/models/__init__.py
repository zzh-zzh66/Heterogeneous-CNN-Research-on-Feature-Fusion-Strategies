# -*- coding: utf-8 -*-
"""
模型定义模块

本模块包含项目中所有神经网络模型的定义：
- Branch1: 浅层纹理分支（3×3卷积核，浅层网络）
- Branch2: 中层边缘分支（5×5卷积核，中层网络）
- Branch3: 高层语义分支（7×7卷积核，深层网络）
- HeteroFusion: 异构融合完整模型
- HomoEnsemble: 同构集成模型（对比基准）
- SingleLargeCNN: 单分支大CNN基线（M0，核心对比基线）

所有模型都继承自 torch.nn.Module。
"""

from .branch1 import BranchTexture
from .branch2 import BranchEdge
from .branch3 import BranchSemantic
from .hetero_fusion import HeteroFusion
from .hetero_fusion_coop import HeteroFusionCoop
from .homo_ensemble import HomoEnsemble
from .single_large import SingleLargeCNN

# 导出所有模型类，方便主程序导入
__all__ = [
    'BranchTexture',       # Branch1: 浅层纹理分支
    'BranchEdge',          # Branch2: 中层边缘分支
    'BranchSemantic',      # Branch3: 高层语义分支
    'HeteroFusion',        # 异构融合模型
    'HeteroFusionCoop',    # 协作奖励异构融合模型 (★ 新增)
    'HomoEnsemble',        # 同构集成模型
    'SingleLargeCNN',      # M0: 单分支大CNN基线
]
