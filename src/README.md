# 异构CNN特征融合与正交正则化 - CIFAR-10图像分类

## 项目简介

本项目实现了一种基于**异构CNN多分支特征融合**的图像分类方法，并引入了**正交正则化约束**来促进不同分支学习互补的特征表示。

### 核心思想

- 设计三种结构差异化的CNN分支，分别捕获图像的**浅层纹理**、**中层边缘**和**高层语义**特征
- 通过特征拼接融合实现互补，提升分类性能
- 引入正交正则化惩罚项，强制不同分支学习统计上互补的特征，减少特征冗余

### 数据集

- **CIFAR-10**：32×32 RGB图像，10个类别，50000训练样本 + 10000测试样本

---

## 环境配置

### 环境要求

- Python 3.11
- PyTorch 2.5+ (with CUDA 12.1)
- torchvision
- matplotlib
- pandas
- numpy

### 安装步骤

```bash
# 1. 创建conda环境
conda create -n ai_paper python=3.11 -y
conda activate ai_paper

# 2. 安装PyTorch (需要梯子或国内镜像)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 安装其他依赖
pip install matplotlib pandas numpy
```

---

## 快速开始

### 单模型训练示例

```bash
# 进入src目录
cd src

# 训练单分支Branch1（纹理分支）
python main.py --model single_b1

# 训练单分支Branch2（边缘分支）
python main.py --model single_b2

# 训练单分支Branch3（语义分支）
python main.py --model single_b3

# 训练同构集成模型
python main.py --model homo_ensemble

# 训练异构融合模型（无正交正则）
python main.py --model hetero_fusion --lambda_orth 0

# 训练异构融合模型（加正交正则，λ=0.05）
python main.py --model hetero_fusion --lambda_orth 0.05

# 训练协作奖励异构融合模型（默认参数 λ_coop=0.1, λ_aux=0.3）
python main.py --model hetero_fusion_coop

# 训练协作奖励异构融合模型（自定义参数）
python main.py --model hetero_fusion_coop --lambda_coop 0.05 --lambda_aux 0.5

# 调参示例：不同协作强度对比
python main.py --model hetero_fusion_coop --lambda_coop 0.2 --lambda_aux 0.3
python main.py --model hetero_fusion_coop --lambda_coop 0.1 --lambda_aux 0.1
```

### 批量训练

```bash
# Windows
.\run_all_experiments.bat

# Linux/Mac
bash run_all_experiments.sh
```

---

## 项目结构

```
src/
├── main.py              # 主程序入口
├── config.py             # 命令行参数配置
├── models/               # 模型定义
│   ├── branch1.py        # Branch1: 浅层纹理分支
│   ├── branch2.py        # Branch2: 中层边缘分支
│   ├── branch3.py        # Branch3: 高层语义分支
│   ├── hetero_fusion.py  # 异构融合完整模型
│   ├── hetero_fusion_coop.py  # 协作奖励异构融合模型
│   └── homo_ensemble.py  # 同构集成模型（对比基准）
├── utils/                # 工具模块
│   ├── data_loader.py    # CIFAR-10数据加载
│   ├── orth_loss.py      # 正交正则化损失
│   ├── coop_loss.py      # 协作奖励损失
│   ├── training.py       # 训练循环封装
│   └── visualization.py  # Grad-CAM可视化
└── run_all_experiments.sh/bat  # 批量训练脚本

results/                   # 实验结果输出目录（训练后自动生成）
doc/                      # 论文方案文档
```

---

## 模型架构

### 三个异构分支

| 分支 | 卷积核 | 层数 | 输出维度 | 聚焦特征 |
|------|--------|------|----------|----------|
| Branch1（纹理） | 3×3 | 3 | 64 | 局部纹理、边缘细节 |
| Branch2（边缘） | 5×5 | 4 | 64 | 边缘组合、局部形状 |
| Branch3（语义） | 7×7+3×3 | 5 | 128 | 全局语义、物体轮廓 |

### 融合方式

三个分支的GAP（全局平均池化）输出拼接后，经过全连接层分类：
```
Concat[64, 64, 128] → FC 256→128 → ReLU → Dropout → FC 128→10
```

---

## 实验模型

| 模型 | 说明 |
|------|------|
| Single-Branch1/2/3 | 单分支独立训练 |
| Homo-Ensemble | 3个同构Branch2拼接融合 |
| Hetero-Fusion | 3个异构分支拼接融合 |
| Hetero-Fusion+Orth | 异构融合 + 正交正则化 |
| Hetero-Fusion+Coop | 异构融合 + 协作奖励机制 |

---

## 论文说明

本项目对应课程论文《基于异构CNN特征融合与正交正则化的CIFAR-10图像分类研究》。

详细方案设计见 `doc/` 目录下的文档。

---

## 作者

- 姓名：待填写
- 学号：待填写
- 专业：待填写
- 课程：人工智能基础
- 提交日期：2026年6月17日
