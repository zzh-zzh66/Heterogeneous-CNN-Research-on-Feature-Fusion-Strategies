# -*- coding: utf-8 -*-
"""
可视化模块

本模块提供各种可视化功能：
1. 训练曲线：损失和准确率随epoch的变化
2. 准确率对比柱状图：不同模型的性能对比
3. λ消融曲线：正交正则化系数的影响
4. 特征相似度热力图：分析分支间的特征相关性
5. Grad-CAM热力图：可视化各分支关注的图像区域

依赖：
-----
- matplotlib
- pandas
- seaborn（可选，用于更美观的图表）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

# 设置中文字体（如果可用）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass


def plot_training_curves(
    results_dir: str,
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    绘制训练曲线（损失和准确率）

    读取训练日志CSV文件，绘制：
    - 训练/验证损失曲线
    - 训练/验证准确率曲线

    Args:
        results_dir: 结果目录（包含train_log.csv）
        save_path: 保存路径，默认在results_dir下
        show: 是否显示图表
    """
    # 读取训练日志
    csv_path = os.path.join(results_dir, 'train_log.csv')
    if not os.path.exists(csv_path):
        print(f"警告: 找不到训练日志 {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -------------------------------------------------------------------------
    # 左图：损失曲线
    # -------------------------------------------------------------------------
    ax = axes[0]
    epochs = df['epoch']

    ax.plot(epochs, df['train_loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in df.columns:
        ax.plot(epochs, df['val_loss'], label='Validation Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # 右图：准确率曲线
    # -------------------------------------------------------------------------
    ax = axes[1]
    ax.plot(epochs, df['train_acc'], label='Training Accuracy', linewidth=2)
    if 'val_acc' in df.columns:
        ax.plot(epochs, df['val_acc'], label='Validation Accuracy', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Training and Validation Accuracy', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存或显示
    if save_path is None:
        save_path = os.path.join(results_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"训练曲线已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_accuracy_comparison(
    results_dirs: dict,
    title: str = "Model Accuracy Comparison",
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    绘制模型准确率对比柱状图

    Args:
        results_dirs: 字典，键为模型名称，值为结果目录
                      例如: {'Branch1': '../results/M1', 'Branch2': '../results/M2'}
        title: 图表标题
        save_path: 保存路径
        show: 是否显示图表
    """
    model_names = []
    accuracies = []

    # 读取各模型的准确率
    for name, dir_path in results_dirs.items():
        results_path = os.path.join(dir_path, 'test_results.json')
        if os.path.exists(results_path):
            import json
            with open(results_path, 'r') as f:
                results = json.load(f)
            model_names.append(name)
            accuracies.append(results['acc'])

    if not model_names:
        print("警告: 没有找到任何结果文件")
        return

    # 排序（从高到低）
    sorted_data = sorted(zip(model_names, accuracies), key=lambda x: x[1], reverse=True)
    model_names, accuracies = zip(*sorted_data)

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 不同颜色区分不同类型的模型
    colors = []
    for name in model_names:
        if 'Single' in name or 'Branch' in name:
            colors.append('#4ECDC4')  # 单分支用青色
        elif 'Homo' in name:
            colors.append('#FFE66D')  # 同构用黄色
        else:
            colors.append('#FF6B6B')  # 异构用红色

    bars = ax.bar(model_names, accuracies, color=colors, edgecolor='black', linewidth=1.2)

    # 在柱子上方标注数值
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, axis='y', alpha=0.3)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4ECDC4', label='Single Branch'),
        Patch(facecolor='#FFE66D', label='Homo Ensemble'),
        Patch(facecolor='#FF6B6B', label='Hetero Fusion'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    if save_path is None:
        save_path = '../results/accuracy_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"准确率对比图已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_lambda_ablation(
    results_dirs: dict,
    title: str = "Orthogonal Regularization Strength (λ) Ablation",
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    绘制λ消融实验曲线

    展示正交正则化系数λ对准确率的影响

    Args:
        results_dirs: 字典，键为λ值（如'0', '0.01', '0.05'），值为结果目录
        title: 图表标题
        save_path: 保存路径
        show: 是否显示图表
    """
    import json

    lambdas = []
    accuracies = []

    # 读取各λ值对应的准确率
    for lam_str, dir_path in results_dirs.items():
        try:
            lam = float(lam_str)
            results_path = os.path.join(dir_path, 'test_results.json')
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
                lambdas.append(lam)
                accuracies.append(results['acc'])
        except ValueError:
            continue

    if not lambdas:
        print("警告: 没有找到有效的λ实验结果")
        return

    # 按λ值排序
    sorted_data = sorted(zip(lambdas, accuracies))
    lambdas, accuracies = zip(*sorted_data)

    # 绘制折线图
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(lambdas, accuracies, 'o-', linewidth=2, markersize=8, color='#FF6B6B')

    # 标注最优点
    best_idx = np.argmax(accuracies)
    best_lambda = lambdas[best_idx]
    best_acc = accuracies[best_idx]
    ax.scatter([best_lambda], [best_acc], color='green', s=200, zorder=5, marker='*')
    ax.annotate(f'Best: λ={best_lambda}, Acc={best_acc:.2f}%',
                xy=(best_lambda, best_acc),
                xytext=(best_lambda + 0.02, best_acc - 2),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='green'))

    ax.set_xlabel('Orthogonal Regularization Strength (λ)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')  # 使用对数刻度更好地展示λ的变化

    plt.tight_layout()

    if save_path is None:
        save_path = '../results/lambda_ablation.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"λ消融曲线已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_feature_similarity(
    features_dict: dict,
    title: str = "Feature Similarity Heatmap",
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    绘制分支间特征相似度热力图

    Args:
        features_dict: 字典，包含各分支特征的统计信息
                      格式: {'Branch1-Branch2': 0.45, 'Branch1-Branch3': 0.32, ...}
        title: 图表标题
        save_path: 保存路径
        show: 是否显示图表
    """
    # 创建相似度矩阵
    branches = ['Branch1', 'Branch2', 'Branch3']
    n = len(branches)
    similarity_matrix = np.zeros((n, n))

    # 填充对角线（自身相似度为1）
    np.fill_diagonal(similarity_matrix, 1.0)

    # 填充其他位置
    pairs = [
        ('Branch1', 'Branch2', '12'),
        ('Branch1', 'Branch3', '13'),
        ('Branch2', 'Branch3', '23'),
    ]
    for b1, b2, key in pairs:
        i, j = branches.index(b1), branches.index(b2)
        sim = features_dict.get(f'sim_{key}', 0.0)
        similarity_matrix[i, j] = sim
        similarity_matrix[j, i] = sim

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(similarity_matrix, cmap='RdYlGn', vmin=0, vmax=1)

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', fontsize=11)

    # 设置刻度标签
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(branches, fontsize=11)
    ax.set_yticklabels(branches, fontsize=11)

    # 在每个格子中添加数值
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.3f}',
                          ha='center', va='center', fontsize=12,
                          color='white' if similarity_matrix[i, j] < 0.5 else 'black')

    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path is None:
        save_path = '../results/feature_similarity.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"特征相似度热力图已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


class GradCAM:
    """
    Grad-CAM 实现

    用于可视化 CNN 模型在图像上关注的区域
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._forward_handle = None
        self._backward_handle = None

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor: torch.Tensor, target_class: int = None,
                 classifier: nn.Module = None):
        self.model.eval()

        self._forward_handle = self.target_layer.register_forward_hook(self.forward_hook)
        self._backward_handle = self.target_layer.register_full_backward_hook(self.backward_hook)

        output = self.model(input_tensor)
        if classifier is not None:
            output = classifier(output)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)

        self._forward_handle.remove()
        self._backward_handle.remove()

        return heatmap.detach().cpu().numpy()


def plot_gradcam(
    model: nn.Module,
    images: torch.Tensor,
    save_dir: str,
    class_names: Tuple[str, ...],
    target_layer_name: str = None,
    num_samples: int = 8,
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    绘制 Grad-CAM 热力图

    Args:
        model: PyTorch 模型
        images: 输入图像张量 [N, 3, 32, 32]
        save_dir: 保存目录
        class_names: 类别名称
        target_layer_name: 目标层名称（可选，默认使用模型的最后一个卷积层）
        num_samples: 可视化的样本数量
        save_path: 保存路径
        show: 是否显示
    """
    model.eval()

    if target_layer_name is None:
        target_layer = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layer_name = name
                target_layer = module
        if target_layer is None:
            print("  警告: 未找到卷积层，跳过 Grad-CAM 可视化")
            return
    else:
        target_layer = dict(model.named_modules())[target_layer_name]

    num_samples = min(num_samples, images.shape[0])
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))

    if num_samples == 1:
        axes = axes.reshape(2, 1)

    branch_models = (hasattr(model, 'features') and hasattr(model, 'gap'))
    classifier = getattr(model, 'classifier', None) if branch_models else None
    model_device = next(model.parameters()).device

    for idx in range(num_samples):
        img = images[idx].cpu()
        input_tensor = images[idx:idx+1].to(model_device)

        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam.generate(input_tensor, classifier=classifier)

        img_display = img.permute(1, 2, 0).numpy()
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())

        axes[0, idx].imshow(img_display)
        axes[0, idx].axis('off')
        axes[0, idx].set_title(f'Original', fontsize=8)

        axes[1, idx].imshow(img_display)
        axes[1, idx].imshow(heatmap, cmap='jet', alpha=0.6, extent=[0, 32, 32, 0])
        axes[1, idx].axis('off')
        axes[1, idx].set_title(f'Grad-CAM', fontsize=8)

    plt.suptitle(f'Grad-CAM Visualization ({target_layer_name})', fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(save_dir, 'gradcam_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Grad-CAM 热力图已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def compute_gradcam(
    model: torch.nn.Module,
    images: torch.Tensor,
    target_layer_names: List[str],
    device: torch.device
) -> dict:
    """
    计算Grad-CAM热力图

    Grad-CAM (Gradient-weighted Class Activation Mapping) 使用梯度信息
    来生成类别判别性热力图，可视化网络关注图像的哪些区域。

    原理：
    -----
    1. 反向传播计算目标类别对特征图的梯度
    2. 对每个通道的梯度取平均（全局平均池化）作为权重
    3. 用权重对特征图进行加权求和
    4. ReLU激活，只保留正贡献的区域

    Args:
        model: PyTorch模型
        images: 输入图像，shape [B, 3, 32, 32]
        target_layer_names: 要可视化的层名称列表
        device: 计算设备

    Returns:
        热力图字典，键为层名称，值为热力图张量
    """
    # 目前为简化版本，返回placeholder
    # 完整的Grad-CAM实现需要hooks来捕获中间激活
    print("提示: Grad-CAM可视化需要完整的hooks实现")
    print("建议使用 captum 或 torchray 库")

    return {}


def plot_confusion_matrix(
    labels: List[int],
    predictions: List[int],
    save_dir: str,
    class_names: Tuple[str, ...],
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    绘制混淆矩阵

    Args:
        labels: 真实标签列表
        predictions: 预测结果列表
        save_dir: 保存目录
        class_names: 类别名称元组
        save_path: 保存路径
        show: 是否显示图表
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(labels, predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Accuracy'}
    )

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  混淆矩阵已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_per_class_accuracy(
    labels: List[int],
    predictions: List[int],
    save_dir: str,
    class_names: Tuple[str, ...],
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    绘制各类别准确率柱状图

    Args:
        labels: 真实标签列表
        predictions: 预测结果列表
        save_dir: 保存目录
        class_names: 类别名称元组
        save_path: 保存路径
        show: 是否显示图表
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, predictions)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.RdYlGn(per_class_acc)
    bars = ax.bar(range(len(class_names)), per_class_acc * 100, color=colors, edgecolor='black')

    for i, (bar, acc) in enumerate(zip(bars, per_class_acc)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc*100:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim(0, 105)
    ax.grid(True, axis='y', alpha=0.3)

    ax.axhline(y=np.mean(per_class_acc) * 100, color='red', linestyle='--',
               label=f'Mean: {np.mean(per_class_acc)*100:.1f}%')
    ax.legend()

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(save_dir, 'per_class_accuracy.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  各类别准确率已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_multi_branch_gradcam(
    model: nn.Module,
    images: torch.Tensor,
    save_dir: str,
    class_names: Tuple[str, ...],
    num_samples: int = 8,
    show: bool = False
) -> dict:
    """
    为异构融合模型的每个分支分别生成 Grad-CAM 热力图

    与 plot_gradcam 的区别：
    - plot_gradcam: 对整个模型生成一张总热力图
    - plot_multi_branch_gradcam: 对每个分支分别生成热力图，
      可以直观看出 Branch1 关注纹理、Branch2 关注边缘、Branch3 关注语义

    输出文件：
    - branch1_texture_gradcam.png: Branch1 纹理分支的热力图
    - branch2_edge_gradcam.png: Branch2 边缘分支的热力图
    - branch3_semantic_gradcam.png: Branch3 语义分支的热力图

    Args:
        model: 异构融合模型 (HeteroFusion)
        images: 输入图像张量 [N, 3, 32, 32]
        save_dir: 保存目录
        class_names: 类别名称元组
        num_samples: 可视化的样本数量
        show: 是否显示图表

    Returns:
        dict: 各分支生成的 heatmaps 列表
            {'branch1': [...], 'branch2': [...], 'branch3': [...]}
    """
    # 检查模型是否有多分支结构
    if not (hasattr(model, 'branch1') and hasattr(model, 'branch2') and hasattr(model, 'branch3')):
        print("  警告: 模型不包含多分支结构，跳过分支Grad-CAM")
        return {}

    model.eval()
    device = next(model.parameters()).device
    num_samples = min(num_samples, images.shape[0])

    # 定义各分支信息
    # 每个分支需要定位到最后一个卷积层（用于Grad-CAM）
    branch_info = []

    def find_last_conv(sequential):
        """在Sequential中逆序查找最后一个nn.Conv2d"""
        for module in reversed(list(sequential.children())):
            if isinstance(module, nn.Conv2d):
                return module
        return None

    # Branch1: 纹理分支
    b1_conv = find_last_conv(model.branch1.features)
    if b1_conv is not None:
        branch_info.append(('branch1_texture', 'Branch1: Texture (浅层纹理)', b1_conv))

    # Branch2: 边缘/形状分支
    b2_conv = find_last_conv(model.branch2.features)
    if b2_conv is not None:
        branch_info.append(('branch2_edge', 'Branch2: Edge (中层边缘)', b2_conv))

    # Branch3: 语义分支
    b3_conv = find_last_conv(model.branch3.features)
    if b3_conv is not None:
        branch_info.append(('branch3_semantic', 'Branch3: Semantic (深层语义)', b3_conv))

    all_heatmaps = {}

    for branch_key, branch_title, target_layer in branch_info:
        # 创建 GradCAM 实例
        gradcam = GradCAM(model, target_layer)

        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 5))
        if num_samples == 1:
            axes = axes.reshape(2, 1)

        heatmaps = []

        for idx in range(num_samples):
            img = images[idx].cpu()
            input_tensor = images[idx:idx+1].to(device)

            # 生成热力图 (多分支模型不需要单独传入classifier)
            heatmap = gradcam.generate(input_tensor, classifier=None)
            heatmaps.append(heatmap)

            # 显示原图
            img_display = img.permute(1, 2, 0).numpy()
            img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)

            axes[0, idx].imshow(img_display)
            axes[0, idx].axis('off')
            if idx == 0:
                axes[0, idx].set_ylabel('Original', fontsize=9)

            # 显示热力图叠加
            axes[1, idx].imshow(img_display)
            axes[1, idx].imshow(heatmap, cmap='jet', alpha=0.5, extent=[0, 32, 32, 0])
            axes[1, idx].axis('off')
            if idx == 0:
                axes[1, idx].set_ylabel('Grad-CAM', fontsize=9)

        plt.suptitle(f'{branch_title}', fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f'{branch_key}_gradcam.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [{branch_key}] Grad-CAM 已保存: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        all_heatmaps[branch_key] = heatmaps

    return all_heatmaps


def compute_iou(heatmap1: np.ndarray, heatmap2: np.ndarray, threshold: float = 0.3) -> float:
    """
    计算两张热力图之间的 IoU (Intersection over Union) 重叠度

    用于量化不同分支关注区域的重叠程度：
    - IoU 高 (接近1): 两个分支关注相同的区域（特征冗余）
    - IoU 低 (接近0): 两个分支关注不同的区域（特征互补）

    计算过程：
    ---------
    1. 二值化：将热力图按阈值转为 0/1 掩码
    2. 计算交并比：IoU = |A ∩ B| / |A ∪ B|

    Args:
        heatmap1: 热力图1, shape [H, W]
        heatmap2: 热力图2, shape [H, W]
        threshold: 二值化阈值 (默认0.3，即热力图值的前30%视为关注区域)

    Returns:
        float: IoU值, 范围 [0, 1]
    """
    # 二值化
    mask1 = (heatmap1 > threshold).astype(np.float32)
    mask2 = (heatmap2 > threshold).astype(np.float32)

    intersection = (mask1 * mask2).sum()
    union = mask1.sum() + mask2.sum() - intersection

    if union < 1e-8:
        return 0.0

    return float(intersection / union)


def compute_branch_iou(
    model: nn.Module,
    images: torch.Tensor,
    save_dir: str,
    device: torch.device = None,
    num_samples: int = 30,
    threshold: float = 0.3
) -> dict:
    """
    计算异构融合模型三个分支之间的热力图 IoU 重叠度

    这是论文的关键定量分析指标，用于验证：
    - λ=0 时，三个分支可能关注重叠区域（IoU高）
    - λ>0 时，正交约束使分支关注不同区域（IoU低）

    输出文件：
    - branch_iou_results.json: 包含各分支对之间的IoU值

    Args:
        model: 异构融合模型 (HeteroFusion)
        images: 输入图像张量 [N, 3, 32, 32]
        save_dir: 保存目录
        device: 计算设备
        num_samples: 用于计算IoU的样本数量（默认30）
        threshold: 二值化阈值

    Returns:
        dict: IoU结果
            {
                'iou_b1_b2': float,  # Branch1 vs Branch2 平均IoU
                'iou_b1_b3': float,  # Branch1 vs Branch3 平均IoU
                'iou_b2_b3': float,  # Branch2 vs Branch3 平均IoU
                'mean_iou': float,   # 三者平均IoU
                'num_samples': int,
                'threshold': float
            }
    """
    if not (hasattr(model, 'branch1') and hasattr(model, 'branch2') and hasattr(model, 'branch3')):
        print("  警告: 模型不包含多分支结构，跳过IoU计算")
        return {}

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    num_samples = min(num_samples, images.shape[0])

    # 找到每个分支的最后一个卷积层
    def find_last_conv(sequential):
        for module in reversed(list(sequential.children())):
            if isinstance(module, nn.Conv2d):
                return module
        return None

    b1_conv = find_last_conv(model.branch1.features)
    b2_conv = find_last_conv(model.branch2.features)
    b3_conv = find_last_conv(model.branch3.features)

    if any(c is None for c in [b1_conv, b2_conv, b3_conv]):
        print("  警告: 无法找到所有分支的卷积层，跳过IoU计算")
        return {}

    # 收集各分支的热力图
    b1_heatmaps = []
    b2_heatmaps = []
    b3_heatmaps = []

    print(f"\n  [IoU分析] 正在计算 {num_samples} 个样本的分支热力图重叠度...")

    for idx in range(num_samples):
        input_tensor = images[idx:idx+1].to(device)

        # 为每个分支分别创建GradCAM并生成热力图
        gradcam1 = GradCAM(model, b1_conv)
        hm1 = gradcam1.generate(input_tensor, classifier=None)
        b1_heatmaps.append(hm1)

        gradcam2 = GradCAM(model, b2_conv)
        hm2 = gradcam2.generate(input_tensor, classifier=None)
        b2_heatmaps.append(hm2)

        gradcam3 = GradCAM(model, b3_conv)
        hm3 = gradcam3.generate(input_tensor, classifier=None)
        b3_heatmaps.append(hm3)

    # 计算两两之间的 IoU
    iou_12_list = [compute_iou(h1, h2, threshold) for h1, h2 in zip(b1_heatmaps, b2_heatmaps)]
    iou_13_list = [compute_iou(h1, h3, threshold) for h1, h3 in zip(b1_heatmaps, b3_heatmaps)]
    iou_23_list = [compute_iou(h2, h3, threshold) for h2, h3 in zip(b2_heatmaps, b3_heatmaps)]

    iou_12_mean = float(np.mean(iou_12_list))
    iou_13_mean = float(np.mean(iou_13_list))
    iou_23_mean = float(np.mean(iou_23_list))
    mean_iou = (iou_12_mean + iou_13_mean + iou_23_mean) / 3.0

    results = {
        'iou_b1_b2': round(iou_12_mean, 4),
        'iou_b1_b3': round(iou_13_mean, 4),
        'iou_b2_b3': round(iou_23_mean, 4),
        'mean_iou': round(mean_iou, 4),
        'num_samples': num_samples,
        'threshold': threshold
    }

    # 保存结果
    iou_path = os.path.join(save_dir, 'branch_iou_results.json')
    with open(iou_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  [IoU分析] 结果已保存: {iou_path}")
    print(f"    B1-B2 IoU: {iou_12_mean:.4f}")
    print(f"    B1-B3 IoU: {iou_13_mean:.4f}")
    print(f"    B2-B3 IoU: {iou_23_mean:.4f}")
    print(f"    平均 IoU:   {mean_iou:.4f}")

    # 绘制 IoU 柱状图
    _plot_iou_comparison(iou_12_mean, iou_13_mean, iou_23_mean, mean_iou, save_dir)

    return results


def _plot_iou_comparison(
    iou_12: float, iou_13: float, iou_23: float, mean_iou: float, save_dir: str
) -> None:
    """
    绘制分支间 IoU 重叠度柱状图

    Args:
        iou_12: Branch1 vs Branch2 IoU
        iou_13: Branch1 vs Branch3 IoU
        iou_23: Branch2 vs Branch3 IoU
        mean_iou: 平均 IoU
        save_dir: 保存目录
    """
    labels = ['B1-B2\n(Tex-Edge)', 'B1-B3\n(Tex-Sem)', 'B2-B3\n(Edge-Sem)', 'Mean']
    values = [iou_12, iou_13, iou_23, mean_iou]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('IoU (Intersection over Union)', fontsize=12)
    ax.set_title('Branch Feature Overlap Analysis (IoU of Grad-CAM Heatmaps)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.3 + 0.05)
    ax.grid(True, axis='y', alpha=0.3)

    # 添加参考线说明
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='High overlap threshold')
    ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Low overlap threshold')
    ax.legend(fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'branch_iou_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  [IoU分析] 对比图已保存: {save_path}")
    plt.close()


if __name__ == '__main__':
    print("可视化模块测试...")
    print("请确保 results 目录中包含 train_log.csv 和 test_results.json")
