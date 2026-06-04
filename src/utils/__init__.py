# -*- coding: utf-8 -*-
"""
工具模块

本模块包含项目中使用的各种工具函数和类：
- data_loader: CIFAR-10数据加载与预处理
- orth_loss: 正交正则化损失函数
- training: 训练循环封装
- visualization: Grad-CAM可视化、多分支热力图、IoU分析
"""

from .data_loader import get_cifar10_loaders, get_transforms, CIFAR10_CLASSES
from .orth_loss import OrthogonalLoss
from .training import train_one_epoch, evaluate, save_results
from .visualization import (
    plot_training_curves,
    plot_accuracy_comparison,
    plot_lambda_ablation,
    plot_feature_similarity,
    compute_gradcam,
    plot_gradcam,
    plot_multi_branch_gradcam,
    compute_branch_iou,
    plot_confusion_matrix,
    plot_per_class_accuracy,
)

__all__ = [
    # 数据加载
    'get_cifar10_loaders',   # 获取CIFAR-10数据加载器
    'get_transforms',         # 获取数据预处理transform
    'CIFAR10_CLASSES',       # CIFAR-10类别列表

    # 正交损失
    'OrthogonalLoss',        # 正交正则化损失类

    # 训练工具
    'train_one_epoch',       # 训练一个epoch
    'evaluate',              # 评估模型
    'save_results',          # 保存实验结果

    # 可视化
    'plot_training_curves',       # 绘制训练曲线
    'plot_accuracy_comparison',   # 绘制准确率对比图
    'plot_lambda_ablation',       # 绘制λ消融实验图
    'plot_feature_similarity',    # 绘制特征相似度热力图
    'compute_gradcam',            # 计算Grad-CAM
    'plot_gradcam',               # 绘制Grad-CAM热力图（单模型）
    'plot_multi_branch_gradcam',   # 绘制多分支Grad-CAM（异构模型）
    'compute_branch_iou',          # 计算分支间IoU重叠度
    'plot_confusion_matrix',       # 绘制混淆矩阵
    'plot_per_class_accuracy',     # 绘制各类别准确率
]
