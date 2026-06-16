# 会话记录 2026-06-12 — Point B v2 出版级证据链

## 关联 Spec / Change
- 母 spec（已归档）：`openspec/changes/archive/2026-06-11-design-complementary-grid-anomaly-innovation/`
  - 能力：`grid-anomaly-innovation-planning`、`self-supervised-normal-pattern-detection`
  - 定位：点 A（监督判别 G3）+ 点 B（自监督正常模式建模 → 异常偏离 → 阶段定位）互补体系
- 本轮 change（已归档）：`openspec/changes/archive/2026-06-12-pointb-v2-publication-evidence/`
  - 合并入主 specs：`pointb-localization-evidence`、`pointb-statistical-validation`

## 基线（不可动，禁止编造）
- 点 A = G3 监督 GBDT 集成：AUC=0.875983，F1=0.516921
- 红线：点 B 不声称超过 G3 全局 AUC/F1；点 B 是「定位/偏离」算法创新，非评价指标。

## 本轮做了什么
1. 修复并验证 Point B v2 = 自监督正常流形（掩码重构+未来预测）+ 弱监督 attention-MIL 阶段定位。
   - 关键修复：通道 z-score 标准化（ss loss 爆炸 90848→2.18）；仅在正常用户上训重构（IoU 0.212→0.339）；deviation 引导注意力（零初始内容注意力 + dev_gain=2.0）；类不平衡 pos_weight=10.721（分离度翻正 +0.199，AUC 0.879）。
2. 出版证据 4 项全部完成：
   - 多 seed 统计稳健性（5 seed + 95%CI + Wilcoxon）
   - 弱形态边界（一次可证伪尝试，门未达成，诚实记录）
   - 真实用户定性定位热力图
   - A+B 联合互补叙事
3. 加入公平基线 `random_interval`（随机连续区间，长度=真实异常长度，位置随机），替换退化的 predict-all uniform 作为 headline 基线。

## 最终结果（5 seed 全量，已验证）
| 来源 | mean IoU |
|---|---|
| MIL attention | **0.159** [0.111, 0.207] |
| uniform（退化 predict-all） | 0.118 |
| random_interval（公平） | 0.113 |
| deviation | 0.082 |

- 注意力分离度稳健为正：**0.198 ± 0.020** [0.172, 0.223]
- pa-F1：0.336 ± 0.084
- Wilcoxon `mil_vs_random_interval`：p=4.44e-13，median_diff=0 → 逐用户中位数平局（MIL 双峰：命中或归零）
- 按形态：zero 0.302（强）、sudden_drop 0.117、sustained_low 0.108、slow_drift 0.108（弱）

## 诚实结论（已写入 evidence_summary.json）
- MIL 在 **均值 IoU / pa-F1 / 注意力分离度** 上稳超所有基线（含公平基线，均值 +41%）。
- 但 **逐用户中位数** 与公平随机区间打平；gradual slow_drift 无法可靠定位。
- A+B 互补：~50% 的 G3-positive 用户获得置信连续 B 区间，常落在后期月份。

## 核心文件
- `sgcc_phase4_pointb_v2_localization.py`（v2 核心）
- `sgcc_phase4_self_supervised.py`（依赖：NormalPatternTransformer、load_sgcc）
- `sgcc_phase4_pointb_v2_aggregate.py`（多 seed 聚合 + Wilcoxon）
- `sgcc_phase4_pointb_v2_heatmap.py`（定性热力图）
- `sgcc_phase4_pointb_ab_complementarity.py`（A+B 联合）
- 结果：`results/phase4_pointb_v2_evidence/`（evidence_summary.json、pointb_v2_multiseed_stats.json、heatmaps/、ab_*）
- 点 A 基线：`results/sgcc_phase3_g3_artifacts.npz`
- 重跑脚本：`run_pointb_v2_reseed.ps1`（5 seed + 聚合，纯英文提示防 GBK 乱码）

## 下一步建议（按优先级）
1. **论文/专利写作**：用 evidence_summary.json 的诚实结论填 1.3 研究内容 + 实验章节。卖点 = 「弱监督 MIL 阶段定位 + 自监督正常流形」方法创新 + A(WHO)/B(WHEN+WHAT) 互补，**不碰 G3 全局指标**。
2. **缓解中位数平局**（可选研究方向，需新 change）：MIL 双峰是因「命中或归零」；可试 top-k 区间软标签 / 区间长度先验校准，使非命中用户也有部分重叠，提升中位数 IoU。需先开新 OpenSpec change 并预设可证伪门。
3. **真实标注小样本验证**（若能拿到真实异常月份标签）：当前 IoU 全部来自合成注入；少量真实月份标注可把定位证据从「合成」升级为「真实」，论文说服力更强。
