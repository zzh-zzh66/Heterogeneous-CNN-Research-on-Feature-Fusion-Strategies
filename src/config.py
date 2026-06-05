# -*- coding: utf-8 -*-
"""
命令行参数配置文件

本模块定义了程序运行时所有可配置的参数，
包括模型选择、训练超参数、输出路径等。
通过 argparse 模块实现，方便实验时灵活调整。
"""

import argparse


def get_args():
    """
    创建并返回命令行参数解析器

    Returns:
        argparse.Namespace: 解析后的参数对象

    使用示例:
        python main.py --model hetero_fusion --lambda_orth 0.05 --epochs 80

    参数说明:
        --model: 选择要训练的模型类型
        --lambda_orth: 正交正则化系数，默认为0表示不使用
        --epochs: 训练轮数，默认为80
        --batch_size: 批次大小，默认为128
        --lr: 初始学习率，默认为0.001
        --seed: 随机种子，用于结果复现
        --save_dir: 结果保存目录
    """
    parser = argparse.ArgumentParser(
        description='异构CNN特征融合CIFAR-10图像分类 - 人工智能基础课程论文'
    )

    # -------------------------------------------------------------------------
    # 模型配置
    # -------------------------------------------------------------------------
    # model: 选择要训练的模型类型
    #   - single_b1: 仅Branch1（纹理分支）独立训练
    #   - single_b2: 仅Branch2（边缘分支）独立训练
    #   - single_b3: 仅Branch3（语义分支）独立训练
    #   - homo_ensemble: 同构集成（3个Branch2拼接融合）
    #   - hetero_fusion: 异构融合（3个异构分支拼接融合）
    parser.add_argument(
        '--model',
        type=str,
        default='hetero_fusion',
        choices=[
            'single_b1',            # 单分支Branch1
            'single_b2',            # 单分支Branch2
            'single_b3',            # 单分支Branch3
            'homo_ensemble',        # 同构集成
            'hetero_fusion',        # 异构融合
            'hetero_fusion_coop',   # 协作奖励异构融合 (M6/M7)
            'hetero_fusion_phased', # 分阶段λ调度异构融合 (M8)
            'single_large'          # 单分支大CNN基线(M0)
        ],
        help='选择模型类型'
    )

    # lambda_orth: 正交正则化系数
    #   - 0: 不使用正交正则化
    #   - 0.01: 轻度正交约束
    #   - 0.05: 中度正交约束（推荐起始值）
    #   - 0.1: 强度正交约束
    #   - 0.5: 过度约束（可能损害性能）
    parser.add_argument(
        '--lambda_orth',
        type=float,
        default=0.0,
        help='正交正则化系数λ，控制特征解耦强度（默认0表示不使用）'
    )

    # 支持 --orth_lambda 作为别名
    parser.add_argument(
        '--orth_lambda',
        type=float,
        default=0.0,
        help='正交正则化系数的别名，与--lambda_orth功能相同'
    )

    # lambda_coop: 协作奖励系数（仅 hetero_fusion_coop 模型使用）
    parser.add_argument(
        '--lambda_coop',
        type=float,
        default=0.1,
        help='协作奖励系数，控制融合优于单分支的奖励幅度（默认0.1）'
    )

    # lambda_aux: 辅助单分支分类损失权重（仅 hetero_fusion_coop 模型使用）
    parser.add_argument(
        '--lambda_aux',
        type=float,
        default=0.3,
        help='辅助单分支分类损失权重，防止分支退化（默认0.3）'
    )

    # -------------------------------------------------------------------------
    # 分阶段训练配置（仅 hetero_fusion_phased / M8 使用）
    # -------------------------------------------------------------------------
    parser.add_argument(
        '--use_phased',
        action='store_true',
        help='启用分阶段λ调度（差异化→协作转型→协作稳定）'
    )
    parser.add_argument(
        '--orth_start',
        type=float,
        default=0.01,
        help='阶段一正交约束初始系数（默认0.01）'
    )
    parser.add_argument(
        '--orth_end',
        type=float,
        default=0.0,
        help='阶段三正交约束终值系数（默认0.0）'
    )
    parser.add_argument(
        '--coop_start',
        type=float,
        default=0.0,
        help='阶段一协作奖励初始系数（默认0.0）'
    )
    parser.add_argument(
        '--coop_end',
        type=float,
        default=0.05,
        help='阶段三协作奖励终值系数（默认0.05）'
    )
    parser.add_argument(
        '--phase1_epochs',
        type=int,
        default=30,
        help='阶段一（差异化播种期）epoch数（默认30）'
    )
    parser.add_argument(
        '--phase2_epochs',
        type=int,
        default=30,
        help='阶段二（协作转型期）epoch数（默认30）'
    )

    # -------------------------------------------------------------------------
    # 训练配置
    # -------------------------------------------------------------------------
    # epochs: 训练的总轮数
    parser.add_argument(
        '--epochs',
        type=int,
        default=80,
        help='训练轮数（默认80）'
    )

    # batch_size: 每个batch的样本数量
    # 显存占用约：batch_size * 3 * 32 * 32 * 4字节 ≈ batch_size * 9KB
    # RTX 3060 6GB显存建议batch_size <= 256
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='批次大小（默认128）'
    )

    # lr: 初始学习率
    # Adam优化器建议初始学习率0.001，SGD建议0.01-0.1
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='初始学习率（默认0.001）'
    )

    # weight_decay: L2正则化系数，用于防止过拟合
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
        help='权重衰减系数（默认1e-4）'
    )

    # dropout: Dropout概率，防止过拟合
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Dropout概率（默认0.3）'
    )

    # patience: 学习率调度器的容忍轮数
    # ReduceLROnPlateau会在验证集loss不再下降的patience个epoch后降低学习率
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='学习率调度器patience（默认5）'
    )

    # factor: 学习率衰减因子
    # 每次调整学习率 = 当前学习率 * factor
    parser.add_argument(
        '--factor',
        type=float,
        default=0.5,
        help='学习率衰减因子（默认0.5）'
    )

    # -------------------------------------------------------------------------
    # 实验配置
    # -------------------------------------------------------------------------
    # seed: 随机种子，用于结果复现
    # 设置相同seed可以保证每次训练结果一致
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认42）'
    )

    # device: 计算设备
    #   - 'auto': 自动选择（优先GPU）
    #   - 'cuda': 强制使用GPU
    #   - 'cpu': 强制使用CPU
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='计算设备：auto/cuda/cpu（默认auto）'
    )

    # save_dir: 实验结果保存目录（绝对路径，不受运行目录影响）
    import os as _os
    _DEFAULT_SAVE_DIR = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'results'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=_DEFAULT_SAVE_DIR,
        help=f'结果保存目录（默认{_DEFAULT_SAVE_DIR}）'
    )

    # log_interval: 每隔多少个epoch输出一次训练日志
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='日志输出间隔（默认每1个epoch输出一次）'
    )

    # num_workers: 数据加载的子进程数
    # Windows建议设为0，Linux/Mac可以设为2-4
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='数据加载线程数（默认0）'
    )

    # -------------------------------------------------------------------------
    # 可视化配置
    # -------------------------------------------------------------------------
    # visualize: 是否生成可视化图表
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='训练后是否生成可视化图表（Grad-CAM等）'
    )

    # gradcam_samples: Grad-CAM可视化的样本数量
    parser.add_argument(
        '--gradcam_samples',
        type=int,
        default=10,
        help='Grad-CAM可视化样本数量（默认10）'
    )

    # -------------------------------------------------------------------------
    # 解析参数
    # -------------------------------------------------------------------------
    args = parser.parse_args()

    # 处理别名：如果用户使用了 --orth_lambda，则同步到 --lambda_orth
    if args.orth_lambda > 0:
        args.lambda_orth = args.orth_lambda

    return args


if __name__ == '__main__':
    # 测试配置：打印所有参数
    args = get_args()
    print("=== 配置参数 ===")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
