# -*- coding: utf-8 -*-
"""
训练循环模块

本模块封装了训练和评估的核心逻辑：
- train_one_epoch: 训练一个epoch
- evaluate: 在测试集上评估模型
- save_results: 保存训练结果到文件

设计考虑：
--------
将训练循环封装成独立函数的好处：
1. 代码复用：不同模型可以使用相同的训练逻辑
2. 可维护性：修改训练逻辑只需改一处
3. 可测试性：可以单独测试每个组件
"""

import os
import json
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 0,
    lambda_orth: float = 0.0,
    lambda_coop: float = 0.0,
    lambda_aux: float = 0.3,
    log_interval: int = 10
) -> Dict[str, float]:
    """
    训练一个epoch

    训练流程：
    ---------
    for each batch:
        1. 前向传播：获取预测结果
        2. 计算损失：
           - 分类损失 (CrossEntropy)
           - 正交正则化损失 (可选)
        3. 反向传播：计算梯度
        4. 梯度下降：更新参数
        5. 统计：累计损失和准确率

    Args:
        model: PyTorch模型
        train_loader: 训练数据加载器
        optimizer: 优化器（如Adam）
        criterion: 损失函数（如CrossEntropyLoss）
        device: 计算设备（cuda或cpu）
        epoch: 当前epoch编号（用于打印）
        lambda_orth: 正交正则化系数，0表示不使用
        lambda_coop: 协作奖励系数，控制融合>单分支的奖励幅度
        lambda_aux: 辅助单分支分类损失权重
        log_interval: 打印日志的batch间隔

    Returns:
        包含以下指标的字典：
        - loss: 总损失均值
        - ce_loss: 分类损失均值
        - orth_loss: 正交损失均值
        - coop_loss: 协作奖励均值
        - aux_loss: 辅助损失均值
        - acc: 训练准确率（%）

    Example:
        >>> model = HeteroFusion(lambda_orth=0.05)
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> criterion = nn.CrossEntropyLoss()
        >>> stats = train_one_epoch(model, train_loader, optimizer, criterion, device)
        >>> print(f"Loss: {stats['loss']:.4f}, Acc: {stats['acc']:.2f}%")
    """
    # 设置模型为训练模式
    # 这会启用Dropout和BatchNorm的训练行为
    model.train()

    # 统计量初始化
    total_loss = 0.0       # 总损失累计
    total_ce = 0.0          # 分类损失累计
    total_orth = 0.0        # 正交损失累计
    total_coop = 0.0        # 协作奖励累计
    total_aux = 0.0         # 辅助损失累计
    correct = 0             # 正确预测数
    total = 0               # 总样本数
    num_batches = len(train_loader)  # 总批次数

    # 遍历所有batch
    for batch_idx, (images, labels) in enumerate(train_loader):
        # -------------------------------------------------------------------------
        # 数据移至计算设备
        # -------------------------------------------------------------------------
        images = images.to(device)  # [B, 3, 32, 32]
        labels = labels.to(device)    # [B,]

        # -------------------------------------------------------------------------
        # 前向传播
        # -------------------------------------------------------------------------
        optimizer.zero_grad()  # 清空梯度缓存

        # 单分支模型 (有 features+gap) vs 多分支融合模型
        is_single_branch = hasattr(model, 'features') and hasattr(model, 'gap')
        is_coop_model = False

        if not is_single_branch:
            # 检测是否为协作奖励模型
            is_coop_model = hasattr(model, 'head_b1') \
                            and hasattr(model, 'head_b2') \
                            and hasattr(model, 'head_b3')

            if is_coop_model and lambda_coop > 0:
                f1, f2, f3, logits_b1, logits_b2, logits_b3 = \
                    model(images, return_features=True, return_branch_logits=True)
                logits = model.classifier(
                    torch.cat([f1, f2, f3], dim=1)
                )
                has_three_branches = True
            else:
                f1, f2, f3 = model(images, return_features=True)
                logits = model.classifier(
                    torch.cat([f1, f2, f3], dim=1)
                )
                has_three_branches = True
        else:
            features = model(images)
            logits = model.classifier(features)
            has_three_branches = False

        # -------------------------------------------------------------------------
        # 计算损失
        # -------------------------------------------------------------------------
        # 分类损失：交叉熵
        ce_loss = criterion(logits, labels)

        # -------------------------------------------------------------------------
        # 分段损失计算：协作奖励 > 正交约束 > 纯CE
        # -------------------------------------------------------------------------
        coop_loss = torch.tensor(0.0, device=device)
        aux_loss = torch.tensor(0.0, device=device)
        orth_loss = torch.tensor(0.0, device=device)

        if lambda_coop > 0 and is_coop_model:
            # ★ 协作奖励模式
            # 1. 计算各分支独立分类损失（辅助监督）
            ce_b1 = criterion(logits_b1, labels)
            ce_b2 = criterion(logits_b2, labels)
            ce_b3 = criterion(logits_b3, labels)
            aux_loss = ce_b1 + ce_b2 + ce_b3

            # 2. 计算协作奖励：融合是否优于最佳单分支
            from .coop_loss import compute_coop_reward
            coop_reward = compute_coop_reward(ce_loss, ce_b1, ce_b2, ce_b3)
            coop_loss = coop_reward

            # 3. 总损失 = 融合CE + 辅助单分支 - 协作奖励
            loss = ce_loss + lambda_aux * aux_loss - lambda_coop * coop_loss
        elif lambda_orth > 0 and has_three_branches:
            # 正交正则化损失（仅融合模型支持）
            from .orth_loss import OrthogonalLoss
            orth_loss_fn = OrthogonalLoss()
            orth_loss = orth_loss_fn(f1, f2, f3)
            loss = ce_loss + lambda_orth * orth_loss
        else:
            loss = ce_loss

        # -------------------------------------------------------------------------
        # 反向传播
        # -------------------------------------------------------------------------
        loss.backward()

        # 梯度裁剪：防止梯度爆炸（可选）
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()  # 更新参数

        # -------------------------------------------------------------------------
        # 统计
        # -------------------------------------------------------------------------
        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_orth += orth_loss.item()
        total_coop += coop_loss.item()
        total_aux += aux_loss.item()

        # 计算准确率
        _, predicted = logits.max(1)  # 取最大概率的类别作为预测
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # -------------------------------------------------------------------------
        # 打印日志（仅当log_interval > 1时打印batch级别进度）
        # -------------------------------------------------------------------------
        if log_interval > 1 and (batch_idx + 1) % log_interval == 0:
            current_acc = 100.0 * correct / total
            current_loss = total_loss / (batch_idx + 1)
            current_ce = total_ce / (batch_idx + 1)
            current_orth = total_orth / (batch_idx + 1)
            current_coop = total_coop / (batch_idx + 1)
            current_aux = total_aux / (batch_idx + 1)
            orth_ratio = current_orth / current_loss * 100 if current_loss > 0 else 0
            extra_info = ""
            if current_coop > 0:
                extra_info += f"Coop: {current_coop:.4f} | Aux: {current_aux:.4f} | "
            print(f"  Batch {batch_idx+1}/{num_batches} | "
                  f"Loss: {current_loss:.4f} | "
                  f"CE: {current_ce:.4f} | "
                  f"Orth: {current_orth:.6f}({orth_ratio:.1f}%) | "
                  f"{extra_info}"
                  f"Acc: {current_acc:.2f}%")

    # -------------------------------------------------------------------------
    # 返回统计结果
    # -------------------------------------------------------------------------
    return {
        'loss': total_loss / num_batches,
        'ce_loss': total_ce / num_batches,
        'orth_loss': total_orth / num_batches,
        'coop_loss': total_coop / num_batches,
        'aux_loss': total_aux / num_batches,
        'acc': 100.0 * correct / total
    }


def evaluate(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    return_predictions: bool = False
) -> Dict[str, float]:
    """
    在测试集上评估模型

    评估流程：
    ---------
    1. 设置模型为评估模式
    2. 遍历测试集，只做前向传播（不计算梯度）
    3. 统计损失和准确率

    Args:
        model: PyTorch模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 计算设备
        return_predictions: 是否返回预测结果

    Returns:
        包含以下指标的字典：
        - loss: 测试损失均值
        - acc: 测试准确率（%）
        如果return_predictions=True，还包含：
        - predictions: 所有预测结果
        - labels: 所有真实标签
    """
    # 设置模型为评估模式
    # 这会禁用Dropout，并使用BatchNorm的running统计量
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_predictions = [] if return_predictions else None
    all_labels = [] if return_predictions else None

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            if logits.shape[-1] != 10 and hasattr(model, 'classifier'):
                logits = model.classifier(logits)

            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if return_predictions:
                all_predictions.extend(predicted.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

    result = {
        'loss': total_loss / total,
        'acc': 100.0 * correct / total
    }

    if return_predictions:
        result['predictions'] = all_predictions
        result['labels'] = all_labels

    return result


def save_results(
    save_dir: str,
    model_name: str,
    train_history: List[Dict],
    test_result: Dict,
    config: Dict,
    model: Optional[nn.Module] = None
) -> None:
    """
    保存实验结果到文件

    保存内容包括：
    1. train_log.csv: 每个epoch的训练历史
    2. test_results.json: 最终测试结果
    3. config.json: 实验配置
    4. best_model.pth: 最佳模型权重（如果提供model）

    Args:
        save_dir: 保存目录
        model_name: 模型名称（用于日志）
        train_history: 训练历史列表，每个元素是一个epoch的统计字典
        test_result: 测试结果字典
        config: 实验配置字典
        model: PyTorch模型（可选，用于保存权重）
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 保存训练历史为CSV
    # -------------------------------------------------------------------------
    # CSV格式便于后续分析和绘图
    df = pd.DataFrame(train_history)
    csv_path = os.path.join(save_dir, 'train_log.csv')
    df.to_csv(csv_path, index=False)
    print(f"  训练历史已保存: {csv_path}")

    # -------------------------------------------------------------------------
    # 保存测试结果为JSON
    # -------------------------------------------------------------------------
    # JSON格式可以保存嵌套结构（只保存指标，不保存预测/标签列表）
    json_result = {
        'loss': test_result['loss'],
        'acc': test_result['acc']
    }
    test_results_path = os.path.join(save_dir, 'test_results.json')
    with open(test_results_path, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, indent=2, ensure_ascii=False)
    print(f"  测试结果已保存: {test_results_path}")

    # -------------------------------------------------------------------------
    # 保存实验配置为JSON
    # -------------------------------------------------------------------------
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  实验配置已保存: {config_path}")

    # -------------------------------------------------------------------------
    # 保存模型权重（如果提供了model）
    # -------------------------------------------------------------------------
    if model is not None:
        model_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(model.state_dict(), model_path)
        print(f"  模型权重已保存: {model_path}")

    # -------------------------------------------------------------------------
    # 打印摘要
    # -------------------------------------------------------------------------
    print(f"\n=== 实验完成: {model_name} ===")
    print(f"  测试准确率: {test_result['acc']:.2f}%")
    print(f"  结果保存目录: {save_dir}")


def load_model_weights(
    model: nn.Module,
    weight_path: str,
    device: torch.device
) -> nn.Module:
    """
    加载模型权重

    Args:
        model: PyTorch模型（架构需要与保存时一致）
        weight_path: 权重文件路径
        device: 计算设备

    Returns:
        加载了权重的模型
    """
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    return model
