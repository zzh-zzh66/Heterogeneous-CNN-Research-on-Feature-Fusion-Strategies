"""生成图3-3：策略方向双轴对比图（Test Accuracy + Mean IoU）.改了路径不知道可不可以直接运行，要检查"""
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import numpy as np
import os

# ── 数据 ──────────────────────────────────────────────
data = [
    # (x标签, Acc, IoU, 组别)
    # 外部参照
    ("M0\nSingle Large",   90.42, None, "外部参照"),
    ("M4\nHomo Ensemble",  87.57, None, "外部参照"),
    # 异构零基线
    ("M5\nλ=0\n(零基线)", 87.48, 0.1457, "异构零基线"),
    # 差异化约束
    ("λ=0.001", 85.67, 0.2165, "差异化约束"),
    ("λ=0.01",  86.82, 0.0009, "差异化约束"),
    ("λ=0.05",  85.95, 0.0025, "差异化约束"),
    ("λ=0.1",   85.43, 0.1368, "差异化约束"),
    ("λ=0.5",   85.93, 0.0027, "差异化约束"),
    # 合作损失调控
    ("λc=0\nλa=0.3",    87.87, 0.1591, "合作损失调控"),
    ("λc=0.05\nλa=0.3", 87.60, 0.2400, "合作损失调控"),
    ("λc=0.1\nλa=0.3",  87.43, 0.2054, "合作损失调控"),
    ("λc=0.1\nλa=0.15", 87.87, 0.1860, "合作损失调控"),
    # 组合实验
    ("M7\nλ=0.01\nλc=0.05\nλa=0.3", 85.72, 0.2773, "组合实验"),
    ("M7\nλ=0.01\nλc=0\nλa=0.3",  86.58, 0.2698, "组合实验"),
]

labels = [d[0] for d in data]
accs   = [d[1] for d in data]
ious   = [d[2] for d in data]
groups = [d[3] for d in data]

group_colors = {
    "外部参照":      "#999999",
    "异构零基线":    "#222222",
    "差异化约束":    "#2166AC",
    "合作损失调控":  "#D6604D",
    "组合实验":      "#4DAF4A",
}
bar_colors = [group_colors[g] for g in groups]

fig, ax1 = plt.subplots(figsize=(17, 6))
x = np.arange(len(labels))
bar_w = 0.38

# 左轴: Accuracy 柱状图
bars = ax1.bar(x, accs, bar_w, color=bar_colors, edgecolor='white', linewidth=0.5, zorder=3)
ax1.set_ylabel("Test Accuracy (%)", fontsize=12, color="#333")
ax1.set_ylim(82, 92)
ax1.tick_params(axis='y', labelcolor="#333")

for bar, acc in zip(bars, accs):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25,
             f"{acc:.2f}", ha='center', va='bottom', fontsize=8, color="#333", fontweight='bold')

# 右轴: IoU 折线 + 散点
ax2 = ax1.twinx()
iou_indices = [i for i, v in enumerate(ious) if v is not None]
iou_x = [x[i] for i in iou_indices]
iou_y = [ious[i] for i in iou_indices]

diff_indices = [i for i in iou_indices if groups[i] == "差异化约束"]
coop_indices = [i for i in iou_indices if groups[i] == "合作损失调控"]
combined_indices = [i for i in iou_indices if groups[i] == "组合实验"]
zero_idx = [i for i in iou_indices if groups[i] == "异构零基线"][0]

# 零基线 -> 差异化连线
line_diff_x = [x[zero_idx]] + [x[i] for i in diff_indices]
line_diff_y = [ious[zero_idx]] + [ious[i] for i in diff_indices]
ax2.plot(line_diff_x, line_diff_y, '-', color='#4393C3', linewidth=1.8, alpha=0.6, zorder=2)

# 零基线 -> 协作连线
line_coop_x = [x[zero_idx]] + [x[i] for i in coop_indices]
line_coop_y = [ious[zero_idx]] + [ious[i] for i in coop_indices]
ax2.plot(line_coop_x, line_coop_y, '-', color='#F4A582', linewidth=1.8, alpha=0.6, zorder=2)

# 协作末尾 -> 组合实验连线
if combined_indices:
    last_coop_idx = coop_indices[-1]
    line_comb_x = [x[last_coop_idx]] + [x[i] for i in combined_indices]
    line_comb_y = [ious[last_coop_idx]] + [ious[i] for i in combined_indices]
    ax2.plot(line_comb_x, line_comb_y, '-', color='#4DAF4A', linewidth=1.8, alpha=0.6, zorder=2)

# IoU 散点
iou_colors_dots = [
    "#4393C3" if groups[i] == "差异化约束" else "#F4A582" if groups[i] == "合作损失调控" else "#4DAF4A" if groups[i] == "组合实验" else "#444444"
    for i in iou_indices
]
ax2.scatter(iou_x, iou_y, s=100, c=iou_colors_dots, marker='D',
            edgecolors='white', linewidth=1, zorder=5, clip_on=False)

# IoU 数值标注
offsets = {}
for xi, yi in zip(iou_x, iou_y):
    offsets[(round(xi, 1), round(yi, 4))] = offsets.get((round(xi, 1), round(yi, 4)), 0) + 1
for idx, (xi, yi) in enumerate(zip(iou_x, iou_y)):
    off = 0.018 + 0.012 * (offsets.get((round(xi, 1), round(yi, 4)), 1) - 1)
    ax2.text(xi, yi + off, f"{yi:.4f}", ha='center', va='bottom',
             fontsize=8, color=iou_colors_dots[idx], fontweight='bold')

ax2.set_ylabel("Mean IoU", fontsize=12, color="#D6604D")
ax2.set_ylim(0, 0.34)
ax2.tick_params(axis='y', labelcolor="#D6604D")

# X轴
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=8, rotation=25, ha='right')

# 网格
ax1.yaxis.grid(True, linestyle='--', alpha=0.25, zorder=0)
ax1.set_axisbelow(True)

# 图例（图表下方）
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Patch(facecolor='#999999', label='外部参照'),
    Patch(facecolor='#222222', label='异构零基线'),
    Patch(facecolor='#2166AC', label='差异化约束 (Acc)'),
    Patch(facecolor='#D6604D', label='合作损失调控 (Acc)'),
    Patch(facecolor='#4DAF4A', label='组合实验 (Acc)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#4393C3',
           markersize=8, label='Mean IoU (差异化约束)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#F4A582',
           markersize=8, label='Mean IoU (合作损失调控)'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#4DAF4A',
           markersize=8, label='Mean IoU (组合实验)'),
]
ax1.legend(handles=legend_elements, loc='upper center',
           bbox_to_anchor=(0.5, 1.12), fontsize=9, ncol=9,
           framealpha=0.9, edgecolor='#ccc')

plt.tight_layout()
plt.subplots_adjust(top=0.88)

# 保存
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "image")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "fig5-3_strategy_comparison.png")
plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"图5-3 已保存到: {output_path}")
