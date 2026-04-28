
# =============================================================================
# London 分难度攻击实验 — 可视化分析
# 读取 london_difficulty_results.json，生成论文级别图表
# =============================================================================

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.font_manager as fm
import os

# ── 中文字体配置 ──
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

RESULT_DIR = r'C:\Users\wb.zhoushujie\PyCharmMiscProject\results'
result_path = os.path.join(RESULT_DIR, 'london_difficulty_results.json')

with open(result_path, 'r', encoding='utf-8') as f:
    all_results = json.load(f)

# ── 配置 ──
difficulties = ['Easy', 'Medium', 'Hard']
diff_cn = ['简单', '中等', '困难']  # 中文难度标签
x_pos = np.arange(len(difficulties))

# 模型分组和样式（中文名 + 英文名映射）
model_groups = {
    'Ours Ensemble':   {'color': '#E74C3C', 'marker': '*',  'ls': '-',  'lw': 3.0, 'ms': 16, 'zorder': 10, 'cn': '本文方法'},
    'CatBoost':        {'color': '#3498DB', 'marker': 's',  'ls': '--', 'lw': 1.8, 'ms': 8,  'zorder': 5,  'cn': 'CatBoost'},
    'XGBoost':         {'color': '#2ECC71', 'marker': 'D',  'ls': '--', 'lw': 1.8, 'ms': 8,  'zorder': 5,  'cn': 'XGBoost'},
    'LightGBM':        {'color': '#9B59B6', 'marker': '^',  'ls': '--', 'lw': 1.8, 'ms': 8,  'zorder': 5,  'cn': 'LightGBM'},
    'Random Forest':   {'color': '#F39C12', 'marker': 'o',  'ls': '-.',  'lw': 1.5, 'ms': 7,  'zorder': 4,  'cn': '随机森林'},
    'Transformer':     {'color': '#1ABC9C', 'marker': 'p',  'ls': '-.',  'lw': 1.5, 'ms': 9,  'zorder': 4,  'cn': 'Transformer'},
    'Logistic Reg.':   {'color': '#95A5A6', 'marker': 'v',  'ls': ':',  'lw': 1.2, 'ms': 7,  'zorder': 3,  'cn': '逻辑回归'},
    'MLP':             {'color': '#BDC3C7', 'marker': '<',  'ls': ':',  'lw': 1.2, 'ms': 7,  'zorder': 3,  'cn': 'MLP'},
    'Isolation Forest':{'color': '#E67E22', 'marker': 'x',  'ls': ':',  'lw': 1.2, 'ms': 8,  'zorder': 3,  'cn': '孤立森林'},
}

# 全局样式 — 高清输出
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'legend.fontsize': 10,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
            'figure.dpi': 300,
            'savefig.dpi': 1600,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# =============================================================================
# 图1: AUC vs 攻击难度 (所有模型)
# =============================================================================
fig1, ax1 = plt.subplots(figsize=(10, 6.5))

for mname, style in model_groups.items():
    aucs = [all_results[d][mname]['auc'] for d in difficulties]
    ax1.plot(x_pos, aucs, color=style['color'], marker=style['marker'],
             linestyle=style['ls'], linewidth=style['lw'], markersize=style['ms'],
             label=style['cn'], zorder=style['zorder'])

ax1.set_xticks(x_pos)
ax1.set_xticklabels(diff_cn)
ax1.set_xlabel('攻击难度级别')
ax1.set_ylabel('OOF AUC')
ax1.set_title('检测性能随攻击难度变化曲线（London Smart Meters 数据集）')
ax1.legend(loc='lower left', framealpha=0.9, ncol=2)
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax1.set_ylim(bottom=max(0.45, min(all_results['Hard'][m]['auc'] for m in model_groups) - 0.05))

fig1.tight_layout()
fig1_path = os.path.join(RESULT_DIR, 'fig1_auc_vs_difficulty.png')
fig1.savefig(fig1_path)
print(f"[OK] 图1 已保存: {fig1_path}")
plt.close(fig1)

# =============================================================================
# 图2: F1 vs 攻击难度 (有 F1 的模型)
# =============================================================================
fig2, ax2 = plt.subplots(figsize=(10, 6.5))

for mname, style in model_groups.items():
    f1s = [all_results[d][mname]['f1'] for d in difficulties]
    if f1s[0] is None:  # Transformer 没有 F1
        continue
    ax2.plot(x_pos, f1s, color=style['color'], marker=style['marker'],
             linestyle=style['ls'], linewidth=style['lw'], markersize=style['ms'],
             label=style['cn'], zorder=style['zorder'])

ax2.set_xticks(x_pos)
ax2.set_xticklabels(diff_cn)
ax2.set_xlabel('攻击难度级别')
ax2.set_ylabel('OOF F1 分数')
ax2.set_title('F1分数随攻击难度变化曲线（London Smart Meters 数据集）')
ax2.legend(loc='lower left', framealpha=0.9, ncol=2)
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

fig2.tight_layout()
fig2_path = os.path.join(RESULT_DIR, 'fig2_f1_vs_difficulty.png')
fig2.savefig(fig2_path)
print(f"[OK] 图2 已保存: {fig2_path}")
plt.close(fig2)

# =============================================================================
# 图3: 性能退化率柱状图 (简单 → 困难 的 AUC 下降幅度)
# =============================================================================
fig3, ax3 = plt.subplots(figsize=(11, 6))

model_names_sorted = sorted(
    model_groups.keys(),
    key=lambda m: all_results['Easy'][m]['auc'] - all_results['Hard'][m]['auc'],
    reverse=True  # 退化最大的排前面
)

bars_x = np.arange(len(model_names_sorted))
bar_colors = [model_groups[m]['color'] for m in model_names_sorted]
bar_labels_cn = [model_groups[m]['cn'] for m in model_names_sorted]
degradations = [all_results['Easy'][m]['auc'] - all_results['Hard'][m]['auc']
                for m in model_names_sorted]

bars = ax3.bar(bars_x, degradations, color=bar_colors, width=0.6, edgecolor='white', linewidth=0.5)

# 在柱子上标注数值
for bar, deg in zip(bars, degradations):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
             f'{deg:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax3.set_xticks(bars_x)
ax3.set_xticklabels(bar_labels_cn, rotation=30, ha='right', fontsize=11)
ax3.set_ylabel('AUC 退化幅度（简单 → 困难）')
ax3.set_title('各模型在攻击难度提升下的性能退化对比')
ax3.axhline(y=0, color='black', linewidth=0.5)

fig3.tight_layout()
fig3_path = os.path.join(RESULT_DIR, 'fig3_degradation_bar.png')
fig3.savefig(fig3_path)
print(f"[OK] 图3 已保存: {fig3_path}")
plt.close(fig3)

# =============================================================================
# 图4: 热力图 — 模型 × 难度 的 AUC 矩阵
# =============================================================================
fig4, ax4 = plt.subplots(figsize=(9, 6.5))

model_order = ['Ours Ensemble', 'CatBoost', 'XGBoost', 'LightGBM',
               'Random Forest', 'Transformer', 'Logistic Reg.', 'MLP', 'Isolation Forest']
model_order_cn = [model_groups[m]['cn'] for m in model_order]

heatmap_data = np.array([
    [all_results[d][m]['auc'] for d in difficulties]
    for m in model_order
])

im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
ax4.set_xticks(np.arange(len(difficulties)))
ax4.set_yticks(np.arange(len(model_order)))
ax4.set_xticklabels(diff_cn)
ax4.set_yticklabels(model_order_cn)

# 在每个格子中标注数值
for i in range(len(model_order)):
    for j in range(len(difficulties)):
        val = heatmap_data[i, j]
        color = 'white' if val < 0.7 else 'black'
        ax4.text(j, i, f'{val:.4f}', ha='center', va='center',
                 fontsize=11, fontweight='bold', color=color)

ax4.set_title('AUC 热力图：模型 × 攻击难度')
fig4.colorbar(im, ax=ax4, label='AUC', shrink=0.8)

fig4.tight_layout()
fig4_path = os.path.join(RESULT_DIR, 'fig4_heatmap.png')
fig4.savefig(fig4_path)
print(f"[OK] 图4 已保存: {fig4_path}")
plt.close(fig4)

print("\n所有图表已生成完毕！")
print(f"图表保存目录: {RESULT_DIR}")
