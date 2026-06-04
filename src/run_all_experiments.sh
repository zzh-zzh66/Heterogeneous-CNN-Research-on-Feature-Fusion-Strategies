#!/bin/bash
#
# 批量训练脚本 - Linux/Mac版本
#
# 本脚本用于自动运行所有实验配置
# 实验顺序：
#   1. 单分支模型 (M1, M2, M3)
#   2. 同构集成模型 (M4)
#   3. 异构融合模型，不同λ值 (M5)
#

# 进入src目录
cd "$(dirname "$0")/src"

echo "============================================================"
echo "异构CNN特征融合 - CIFAR-10分类 批量训练"
echo "============================================================"
echo ""

# 获取当前时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="../results/batch_log_${TIMESTAMP}.txt"

# 创建结果目录
mkdir -p ../results/M1_branch1
mkdir -p ../results/M2_branch2
mkdir -p ../results/M3_branch3
mkdir -p ../results/M4_homo_ensemble
mkdir -p ../results/M5_hetero_lambda0
mkdir -p ../results/M5_hetero_lambda0p01
mkdir -p ../results/M5_hetero_lambda0p05
mkdir -p ../results/M5_hetero_lambda0p1
mkdir -p ../results/M5_hetero_lambda0p5

echo "开始时间: $(date)"
echo "日志文件: $LOGFILE"
echo ""

# 训练函数
train_model() {
    local model=$1
    local lambda=$2
    local desc=$3

    echo "============================================================"
    echo "$desc"
    echo "============================================================"
    echo ""

    if [ "$lambda" == "none" ]; then
        python main.py --model "$model" --save_dir ../results --log_interval 10 2>&1 | tee -a "$LOGFILE"
    else
        python main.py --model "$model" --lambda_orth "$lambda" --save_dir ../results --log_interval 10 2>&1 | tee -a "$LOGFILE"
    fi

    echo ""
}

# 开始训练
train_model "single_b1" "none" "[1/9] 训练单分支 Branch1 (纹理分支)"
train_model "single_b2" "none" "[2/9] 训练单分支 Branch2 (边缘分支)"
train_model "single_b3" "none" "[3/9] 训练单分支 Branch3 (语义分支)"
train_model "homo_ensemble" "none" "[4/9] 训练同构集成模型"
train_model "hetero_fusion" "0" "[5/9] 训练异构融合模型 (λ=0, 无正则化)"
train_model "hetero_fusion" "0.01" "[6/9] 训练异构融合模型 (λ=0.01)"
train_model "hetero_fusion" "0.05" "[7/9] 训练异构融合模型 (λ=0.05)"
train_model "hetero_fusion" "0.1" "[8/9] 训练异构融合模型 (λ=0.1)"
train_model "hetero_fusion" "0.5" "[9/9] 训练异构融合模型 (λ=0.5)"

echo ""
echo "============================================================"
echo "所有实验训练完成！"
echo "结束时间: $(date)"
echo "============================================================"
echo ""
echo "请查看 ../results 目录下的各实验结果"
echo "训练日志已保存到: $LOGFILE"
