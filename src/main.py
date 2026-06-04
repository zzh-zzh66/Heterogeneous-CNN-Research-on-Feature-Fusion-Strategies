# -*- coding: utf-8 -*-
"""
主程序入口

本程序是整个项目的入口，负责：
1. 解析命令行参数
2. 创建模型
3. 加载数据
4. 执行训练
5. 保存结果

使用方法：
---------
单模型训练:
    python main.py --model hetero_fusion --lambda_orth 0.05

批量实验:
    使用 run_all_experiments.bat (Windows) 或 run_all_experiments.sh (Linux/Mac)
"""

import os
import sys
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_args
from utils.data_loader import get_cifar10_loaders, CIFAR10_CLASSES
from utils.training import train_one_epoch, evaluate, save_results
from utils.visualization import (
    plot_training_curves,
    plot_accuracy_comparison,
    plot_lambda_ablation,
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_gradcam,
    plot_multi_branch_gradcam,
    compute_branch_iou,
)

# 导入模型
from models import (
    BranchTexture,
    BranchEdge,
    BranchSemantic,
    HeteroFusion,
    HeteroFusionCoop,
    HomoEnsemble,
    SingleLargeCNN,
)


def set_seed(seed: int = 42) -> None:
    """
    设置随机种子以确保结果可复现

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 让CuDNN使用确定性算法（可能牺牲一点速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(model_name: str, args) -> nn.Module:
    """
    根据模型名称创建模型实例

    Args:
        model_name: 模型名称（来自命令行参数）
        args: 命令行参数

    Returns:
        PyTorch模型实例
    """
    if model_name == 'single_b1':
        # 单分支Branch1 + 分类头
        model = BranchTexture()
        # 添加分类头
        model.classifier = nn.Linear(64, 10)
        save_dir = f"{args.save_dir}/M1_branch1"

    elif model_name == 'single_b2':
        model = BranchEdge()
        model.classifier = nn.Linear(64, 10)
        save_dir = f"{args.save_dir}/M2_branch2"

    elif model_name == 'single_b3':
        model = BranchSemantic()
        model.classifier = nn.Linear(128, 10)
        save_dir = f"{args.save_dir}/M3_branch3"

    elif model_name == 'homo_ensemble':
        # 同构集成：3个Branch2拼接
        model = HomoEnsemble(dropout=args.dropout)
        save_dir = f"{args.save_dir}/M4_homo_ensemble"

    elif model_name == 'hetero_fusion':
        # 异构融合：3个不同分支拼接
        model = HeteroFusion(
            lambda_orth=args.lambda_orth,
            dropout=args.dropout
        )
        # 保存目录包含λ值
        lambda_str = str(args.lambda_orth).replace('.', 'p')
        save_dir = f"{args.save_dir}/M5_hetero_lambda{lambda_str}"

    elif model_name == 'hetero_fusion_coop':
        # 协作奖励异构融合：3个不同分支 + 协作奖励
        model = HeteroFusionCoop(
            lambda_orth=args.lambda_orth,
            lambda_coop=args.lambda_coop,
            lambda_aux=args.lambda_aux,
            dropout=args.dropout
        )
        coop_str = str(args.lambda_coop).replace('.', 'p')
        aux_str = str(args.lambda_aux).replace('.', 'p')
        save_dir = f"{args.save_dir}/M6_hetero_coop_c{coop_str}_a{aux_str}"

    elif model_name == 'single_large':
        # 单分支大CNN（M0，核心对比基线）
        model = SingleLargeCNN(dropout=args.dropout)
        save_dir = f"{args.save_dir}/M0_single_large"

    else:
        raise ValueError(f"未知的模型类型: {model_name}")

    return model, save_dir


def get_model_display_name(model_name: str, args) -> str:
    """
    获取模型的友好显示名称

    Args:
        model_name: 模型名称
        args: 命令行参数

    Returns:
        友好显示名称
    """
    name_map = {
        'single_b1': 'Single Branch1 (Texture)',
        'single_b2': 'Single Branch2 (Edge)',
        'single_b3': 'Single Branch3 (Semantic)',
        'homo_ensemble': 'Homo Ensemble (3×Branch2)',
        'hetero_fusion': f'Hetero Fusion (λ={args.lambda_orth})',
        'hetero_fusion_coop': f'Hetero Fusion Coop (λc={args.lambda_coop}, λa={args.lambda_aux})',
        'single_large': 'Single Large CNN (M0 Baseline)',
    }
    return name_map.get(model_name, model_name)


def print_model_info(model: nn.Module, model_name: str) -> None:
    """
    打印模型信息

    Args:
        model: PyTorch模型
        model_name: 模型名称
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print(f"模型: {model_name}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"参数量(M): {total_params / 1e6:.3f}M")
    print("=" * 60)


def main():
    """
    主函数
    """
    # -------------------------------------------------------------------------
    # 解析命令行参数
    # -------------------------------------------------------------------------
    args = get_args()

    # 设置随机种子
    set_seed(args.seed)

    # -------------------------------------------------------------------------
    # 设置计算设备
    # -------------------------------------------------------------------------
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("异构CNN特征融合 - CIFAR-10图像分类")
    print("=" * 60)
    print(f"计算设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"随机种子: {args.seed}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 创建模型
    # -------------------------------------------------------------------------
    print("\n[1/5] 创建模型...")
    model, save_dir = create_model(args.model, args)
    model = model.to(device)

    os.makedirs(save_dir, exist_ok=True)

    model_display_name = get_model_display_name(args.model, args)
    print_model_info(model, model_display_name)

    # -------------------------------------------------------------------------
    # 加载数据
    # -------------------------------------------------------------------------
    print("\n[2/5] 加载CIFAR-10数据...")
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=data_dir
    )
    print(f"训练集: {len(train_loader.dataset)} 样本, {len(train_loader)} batches")
    print(f"验证集: {len(val_loader.dataset)} 样本, {len(val_loader)} batches")
    print(f"测试集: {len(test_loader.dataset)} 样本, {len(test_loader)} batches")
    print(f"批次大小: {args.batch_size}")

    # -------------------------------------------------------------------------
    # 设置优化器和损失函数
    # -------------------------------------------------------------------------
    print("\n[3/5] 设置优化器和损失函数...")

    # 优化器：Adam
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    print(f"优化器: Adam (lr={args.lr}, weight_decay={args.weight_decay})")

    # 学习率调度器：ReduceLROnPlateau
    # 当验证集损失不再下降时，降低学习率
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=args.patience,
        factor=args.factor,
        min_lr=1e-6
    )
    print(f"学习率调度: ReduceLROnPlateau (patience={args.patience}, factor={args.factor})")

    # 损失函数：交叉熵
    criterion = nn.CrossEntropyLoss()
    print(f"损失函数: CrossEntropyLoss")

    # -------------------------------------------------------------------------
    # 训练
    # -------------------------------------------------------------------------
    print(f"\n[4/5] 开始训练...")
    print(f"训练轮数: {args.epochs}")
    print(f"正交正则化系数 λ: {args.lambda_orth}")
    print("-" * 60)

    train_history = []
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # 训练一个epoch
        train_stats = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            lambda_orth=args.lambda_orth,
            lambda_coop=args.lambda_coop if args.model == 'hetero_fusion_coop' else 0.0,
            lambda_aux=args.lambda_aux,
            log_interval=args.log_interval
        )

        # 评估
        val_stats = evaluate(
            model=model,
            test_loader=val_loader,
            criterion=criterion,
            device=device
        )

        # 更新学习率-根据损失更新（min）-根据准确率更新（max）
        scheduler.step(val_stats['loss'])
        # scheduler.step(val_stats['acc'])

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 记录历史
        history = {
            'epoch': epoch + 1,
            'train_loss': train_stats['loss'],
            'train_acc': train_stats['acc'],
            'ce_loss': train_stats['ce_loss'],
            'orth_loss': train_stats['orth_loss'],
            'coop_loss': train_stats.get('coop_loss', 0.0),
            'aux_loss': train_stats.get('aux_loss', 0.0),
            'val_loss': val_stats['loss'],
            'val_acc': val_stats['acc'],
            'lr': current_lr
        }
        train_history.append(history)

        # 记录是否刷新最佳准确率
        is_best = val_stats['acc'] > best_acc

        # 保存最佳模型
        if is_best:
            best_acc = val_stats['acc']
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

        epoch_time = time.time() - epoch_start

        # 打印简洁的epoch进度
        improved = " ★" if is_best else ""
        extra_info = ""
        if args.lambda_orth > 0:
            extra_info += f" | Orth: {train_stats['orth_loss']:.6f} ({(train_stats['orth_loss']/train_stats['loss']*100):.1f}%)"
        if args.model == 'hetero_fusion_coop':
            coop_val = train_stats.get('coop_loss', 0.0)
            aux_val = train_stats.get('aux_loss', 0.0)
            extra_info += f" | Coop: {coop_val:.4f} | Aux: {aux_val:.4f}"
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_stats['loss']:.4f}  Acc: {train_stats['acc']:.2f}% | "
              f"CE: {train_stats['ce_loss']:.4f}{extra_info} | "
              f"Val Loss: {val_stats['loss']:.4f}  Acc: {val_stats['acc']:.2f}%{improved} | "
              f"LR: {current_lr:.1e} | "
              f"Time: {epoch_time:.1f}s")

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"训练完成！总时间: {total_time/60:.1f} 分钟")
    print(f"最佳验证准确率: {best_acc:.2f}%")

    # 加载最佳模型进行最终评估
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    # -------------------------------------------------------------------------
    # 保存结果
    # -------------------------------------------------------------------------
    print(f"\n[5/5] 保存结果...")

    test_result = evaluate(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        return_predictions=True
    )
    print(f"最终测试准确率: {test_result['acc']:.2f}%")

    config = {
        'model': args.model,
        'model_display_name': model_display_name,
        'lambda_orth': args.lambda_orth,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'seed': args.seed,
        'total_params': sum(p.numel() for p in model.parameters()),
        'device': str(device),
        'total_time_minutes': total_time / 60
    }

    save_results(
        save_dir=save_dir,
        model_name=model_display_name,
        train_history=train_history,
        test_result=test_result,
        config=config,
        model=None
    )

    # -------------------------------------------------------------------------
    # 生成可视化图表
    # -------------------------------------------------------------------------
    print("\n生成可视化图表...")
    try:
        plot_training_curves(save_dir)
        predictions = test_result.get('predictions', [])
        labels = test_result.get('labels', [])
        if predictions and labels:
            plot_confusion_matrix(labels, predictions, save_dir, CIFAR10_CLASSES)
            plot_per_class_accuracy(labels, predictions, save_dir, CIFAR10_CLASSES)

        print("  生成 Grad-CAM 热力图...")
        model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth'), map_location=device))
        model.eval()
        sample_images, _ = next(iter(test_loader))
        sample_images = sample_images[:8].to(device)

        # 多分支模型：为每个分支分别生成热力图 + IoU分析
        if args.model in ('hetero_fusion', 'hetero_fusion_coop'):
            print("  [多分支分析] 为每个分支分别生成 Grad-CAM...")
            plot_multi_branch_gradcam(model, sample_images, save_dir, CIFAR10_CLASSES)

            print("  [IoU分析] 计算分支间热力图重叠度...")
            compute_branch_iou(model, sample_images, save_dir, device=device, num_samples=30)
        else:
            plot_gradcam(model, sample_images, save_dir, CIFAR10_CLASSES)
    except Exception as e:
        print(f"  生成可视化图表失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("实验全部完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()