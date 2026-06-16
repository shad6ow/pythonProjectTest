## Context

当前项目已包含 SGCC 智能电表异常用电检测代码与点 A 专利材料。点 A 技术路线是多源用电行为特征、随机矩阵理论信号增强、双路径 Transformer 深度表征、PCA 压缩和 CatBoost/XGBoost/LightGBM 集成判别。代码中已有 Phase 1 GBDT 特征工程、Phase 2 RMT 特征增强、Phase 3 Transformer + GBDT 集成实验。

从审稿视角看，前一版“全局—局部多证据可信评分”更偏评价与复核决策，容易被理解为二次评分或指标组合，方法创新力度不足。点 B 更适合调整为“基于自监督正常用电模式建模的异常偏离检测方法”：利用正常用户或低风险用户学习正常用电行为流形，通过掩码重构、未来窗口预测和隐空间正常原型距离生成新的异常偏离特征，并定位异常月份。

已读取的论文 1.3 节表明，稳妥写法是“指出已有问题 → 提出针对性方法 → 说明方法组成 → 给出实验验证方式”。本项目应把点 A 和点 B 写成互补关系：点 A 是监督判别，学习异常边界；点 B 是自监督正常建模，学习正常行为流形。两者融合后，既保留点 A 对已知异常的判别能力，又补充点 B 对未知异常、隐蔽异常和阶段性异常的偏离检测能力。

## Goals / Non-Goals

**Goals:**

- 将点 B 定位为“基于自监督正常用电模式建模的异常偏离检测方法”。
- 明确点 B 的创新边界：不是评价指标、不是多模型一致性比较，而是正常模式学习和异常偏离特征生成。
- 让点 B 解决点 A 的不足：点 A 给出异常概率，点 B 给出重构误差、预测误差、隐空间正常原型距离和异常月份定位。
- 设计可实现流程：复用点 A 的月度多通道序列，训练自监督正常模式模型，输出偏离特征并融合到 GBDT 集成。
- 设计可验证实验：偏离特征有效性、点 A + 点 B 消融、弱标签鲁棒性、异常阶段定位分析。
- 支撑论文/专利表达：形成“监督判别 + 自监督正常模式建模”的双创新体系。

**Non-Goals:**

- 不再把点 B 写成全局—局部可信评分或复核排序指标。
- 不把点 B 写成只有多模型一致性的后处理方法。
- 不在本变更中重写点 A 的核心模型。
- 不编造 AUC、F1、召回率、Top-K 命中率等实验数据。
- 不声称点 B 必然提升所有指标；点 B 的收益必须通过实际实验验证。

## Decisions

### Decision 1: 点 B 采用自监督正常模式建模，而非可信评分指标

可信评分类方案容易被质疑为指标拼接或工程解释。自监督正常模式建模能形成实质方法模块：模型先从正常用户或低风险用户学习正常用电序列规律，再用重构误差、预测误差和隐空间原型距离度量待测用户偏离正常行为流形的程度。

Alternatives considered:
- 多证据可信评分：业务解释性强，但方法创新不足。
- 相似用户群体对照检测：已有相关思想较多，单独写创新性偏弱。
- 新增图神经网络：需要台区、线路或拓扑数据支撑，当前数据字段不明确。

### Decision 2: 点 B 复用点 A 的月度多通道序列

点 B 不重新设计数据体系，而是沿用点 A 的月度多通道序列表示。每个用户表示为 \(X_i \in R^{T \times C}\)，其中 \(T\) 为月份数，\(C\) 为通道数。通道可包括月均值、月标准差、月最大值、月零值率、月缺失率、相对基准偏离、累计偏离、月度排名偏离和季节性相关特征。

这样能保证点 B 与点 A 在数据层面关联紧密，并复用现有 SGCC 特征工程代码。

### Decision 3: 自监督任务采用掩码重构 + 未来窗口预测

点 B 至少包含两个自监督任务：

1. 掩码重构：随机遮蔽部分月份或通道，模型根据上下文恢复原始序列，学习正常用电内部结构。
2. 未来窗口预测：使用前期月份预测后期窗口，增强对后期下降、持续偏移、局部低谷等异常形态的敏感性。

联合任务比单一 AutoEncoder 更贴合长周期电表序列，因为它同时约束局部上下文恢复能力和跨期趋势预测能力。

### Decision 4: 异常偏离度由三类信号构成

点 B 输出三类核心偏离信号：

\[
D_i = a \cdot RecError_i + b \cdot PredError_i + c \cdot LatentDistance_i
\]

其中：
- \(RecError_i\)：掩码重构误差；
- \(PredError_i\)：未来窗口预测误差；
- \(LatentDistance_i\)：用户隐空间表示到正常原型的距离。

这些偏离信号不是最终结论，可作为新增特征输入点 A 的 GBDT 集成模型。权重初版可采用归一化平均，后续通过验证集或消融实验调整。

### Decision 5: 点 B 必须支持异常月份定位

重构误差和预测误差需要保留月份维度，输出误差最高的月份或连续异常区间。该能力能补充点 A 只输出用户级异常概率的不足，服务论文图示和专利说明。

### Decision 6: 实验评价必须以真实点 A G3 为融合基线

点 A 的最终基线不是仅包含人工特征和 RMT 的 `A_plus_RMT`，而是 Phase 3 已验证的 G3：多源人工特征、RMT、Transformer_PCA16 表征和 CatBoost/XGBoost/LightGBM GBDT 集成。Phase 4 的点 B 融合必须在 G3 基础上验证，比较对象应写为：

1. G3：HandCraft + RMT + Transformer_PCA16 + GBDT Ensemble。
2. G4_rec：G3 + reconstruction deviation features。
3. G4_pred：G3 + prediction deviation features。
4. G4_latent：G3 + normal-prototype latent deviation features。
5. G4_selected：G3 + screened Point B complementary features。

任何只以 `A_plus_RMT` 为基线的实验只能作为弱基线诊断，不能作为论文最终融合结论。

### Decision 7: 点 B 需要尺度无关和多原型正常模式建模

当前点 B 原始重构/预测误差可能受用户用电规模支配，单一正常原型也难以覆盖异质正常用电模式。后续点 B 应优先改为尺度无关偏离分数和多原型正常流形建模：

1. 使用用户内标准化、通道级标准化或正常参考分布校准，避免 raw MSE 直接反映负荷规模。
2. 使用正常用户聚类或低风险正常用户子集构建多个正常原型，输出 nearest-prototype distance、cluster residual 和 density-style deviation。
3. 对重构误差、预测误差和月份定位误差做正常参考 percentile / z-score 校准。
4. 加强后期窗口、连续低谷、突降和 top-k error concentration 等异常形态特征。

### Decision 8: 先诊断 G3 盲区，再决定是否融合

当前 full Point B feature concatenation 未能提升真实 G3，LR stacking 也存在不安全风险，因此不能把“融合提升”作为主结论。点 B 暂不放弃，但必须先证明它是否补充点 A 的盲区：G3 false negatives、G3 阈值附近边界样本、后期/连续异常月份。

诊断应只读取已有产物，不重新训练重模型：

1. 使用 `results/sgcc_phase3_g3_artifacts.npz` 的 `labels` 和 `oof_ensemble` 确定真实 G3 基线、最佳 F1 阈值、false negatives 和边界带样本。
2. 使用 `results/phase4_self_supervised/sgcc_phase4_self_supervised_features.csv` 评估每个点 B 特征在 G3 predicted-negative 子集内区分 false negatives 与 true negatives 的 AUC。
3. 使用 `results/phase4_self_supervised/sgcc_phase4_month_errors.npz` 统计 G3 false negatives 的高误差月份是否集中在后期窗口或连续区间。
4. 同时记录点 B 特征与 G3 分数的相关性，优先保留“G3 盲区内有信号且与 G3 不完全冗余”的特征。

只有盲区诊断显示点 B 对 G3 false negatives 或边界样本有补充信号时，才继续 gated/residual fusion：

\[
Meta_i = [s^A_i, B_i, s^A_i \times B_i, |rank(s^A_i)-rank(B_i)|]
\]

其中 \(s^A_i\) 为 G3 OOF 风险分数，\(B_i\) 为筛选后的点 B 偏离特征。AUC 和 F1 需要分开优化。full feature concatenation 和不安全 stacking 只能作为失败诊断，不作为论文主结论。

所有数值必须来自实际运行，未运行前只写“待验证”。

### Decision 9: 自监督改进聚焦 G3 predicted-negative rescue，而非全局融合

最新盲区诊断显示，点 B 在 G3 predicted-negative 区域存在弱但可用的 rescue 信号：G3 基线 AUC=0.875983、F1=0.516921、precision=0.510100、recall=0.523928；G3 predicted-negative 中 false negatives=1721、true negatives=36938；`ss_combined_score` 在该区域的 AUC=0.619434，`ss_rec_mean`=0.612512，`ss_pred_mean`=0.610572。与此同时，boundary-band AUC 只有约 0.46-0.48，说明边界融合和全局融合当前不是优先方向。

因此保留自监督点 B，但目标收窄为“用后期窗口异常偏离补救 G3 预测为负的漏检样本”。G3 false negative 的高误差月份集中在后期月份，重构/预测 top months 包括 25、30、31、32、33、26。下一步不做重训练和不做 full concat，而是从已有月度误差中提取 late-window deviation features：last6、last10、month25_33 error sums，late/early ratios，以及 top-k concentration in late window。

融合方式只允许安全的 G3 predicted-negative gated rescue：保持 G3 阳性预测不被点 B 覆盖，只在 G3 分数低于最佳 F1 阈值的样本中，用盲区诊断为正的后期偏离特征挑选少量待救回用户。full Point B feature concatenation、全局 LR stacking 和 boundary/global fusion 只作为失败诊断，不作为论文或专利主结论，除非后续实际结果证明其优于 predicted-negative gated rescue。

### Decision 10: 当前 late-window 简单 gate 不足，下一步转向高精度 rescue 与 clean-normal 自监督

最新 late-window 诊断显示，后期窗口特征确实包含 G3 predicted-negative rescue 信号，但强度仍弱于普通点 B 分数。当前较好的 late-window 特征包括 `late_rec_last10_sum`（neg_region_auc=0.608891）、`late_combined_last10_sum`（0.608343）、`late_pred_last10_sum`（0.607122）和 `late_rec_month25_33_sum`（0.606076），均低于 `ss_combined_score` 的 0.619434。boundary AUC 仍约为 0.456-0.458，因此边界融合继续降级。

简单 gated rescue 也暂未成功：当前 best F1 仅从 G3 的 0.516921 变为约 0.517006，提升可忽略；同时 AUC 降到约 0.86585，且 q=0.95 gate 选出的 1933 个样本只有约 117 个异常，selected precision=0.0605，低于全量异常率。这说明当前 late-window gate 能排序但不能高精度选中可救回的 false negatives。

下一步研究重点改为两条线：

1. 高精度 predicted-negative rescue：在 G3 predicted-negative 区域内组合普通点 B 信号（`ss_combined_score`、`ss_rec_mean`、`ss_pred_mean`）与 late-window 信号（`late_rec_last10_sum`、`late_rec_month25_33_sum`），并测试更严格的 gate 分位数 q=0.975、0.985、0.99、0.995。目标不是覆盖更多样本，而是提高 selected precision 和 fixed-threshold F1。
2. clean-normal 自监督重训：不再默认使用全部 label=0 用户训练正常流形，而是使用 label=0 且 G3 risk score 位于最低 50%/60%/70% 的低风险正常用户训练点 B，验证是否能提高 G3 predicted-negative FN-vs-TN AUC。

在上述两条线未出现稳定提升之前，不能把 late-window gate 或当前点 B 写成成功融合方法。保留自监督方向，但必须以“提高 G3 漏检区救回精度”为核心优化目标。

- [Risk] 正常训练集被异常样本污染 → Mitigation: 优先使用标签正常样本，并结合点 A 低风险样本筛选；训练后检查高误差正常样本。
- [Risk] 自监督模型复杂度过高 → Mitigation: 初版使用轻量 Transformer Encoder，控制隐藏维度、层数和训练轮数。
- [Risk] 重构任务学到复制输入而非正常规律 → Mitigation: 使用随机掩码输入，只在遮蔽位置或加权位置计算主要重构损失。
- [Risk] 点 B 不一定提升 AUC → Mitigation: 同时评估偏离特征有效性、弱标签鲁棒性和异常阶段定位价值。
- [Risk] 与点 A 的 Transformer 看起来重复 → Mitigation: 强调训练目标不同：点 A 是监督分类表征，点 B 是自监督正常模式学习。

### Decision 11: 点 B v2 重定位为「自监督正常流形 + 弱监督异常阶段定位」算法，而非误差后处理

诊断证据已经表明：把点 B 当成「提升 G3 全局 AUC/F1 的融合分数」不成立（boundary AUC≈0.46，全局 concat 使 AUC 反降，late-window/morphology/dual gate 无有效增益，盲区 AUC 天花板约 0.63）。同时，点 B v1 的「异常月份定位」只是对月级误差取 argmax，是误差后处理，不构成算法创新。

因此点 B v2 改为一个**端到端弱监督异常阶段定位算法**，把「定位」本身设计成可学习、可度量的任务，而不是事后排序：

1. **问题形式化（多示例学习 MIL）**：每个用户视为一个「月份袋」\(X_i=\{x_{i,1},\dots,x_{i,T}\}\)。正常用户=所有月份正常；异常用户=至少存在一个异常月份或异常区间。仅有用户级弱标签 \(y_i\in\{0,1\}\)，没有月级标签。这正是窃电检测的真实场景：知道用户违章，但不知道违章发生在哪几个月。

2. **自监督正常流形分支**：沿用掩码重构 + 未来窗口预测，仅在正常/低风险（clean50）用户上训练，得到月级偏离度 \(d_{i,m}\)（重构误差、预测误差、原型距离的标定值）。

3. **弱监督定位头（核心创新）**：在月级偏离上加一个**注意力 MIL 聚合**，用月级注意力权重 \(a_{i,m}\) 把 \(d_{i,m}\) 聚合成用户级异常分 \(s_i=\sum_m a_{i,m} d_{i,m}\)，再用用户级弱标签做 BCE 监督。注意力分布即为异常发生概率的时间定位。

4. **区间结构正则（让定位输出连续区间而非散点）**：
\[
L = \mathrm{BCE}(s_i, y_i) + \lambda_{ss} L_{ss} + \lambda_{tv}\sum_m |a_{i,m}-a_{i,m-1}| + \lambda_{sp}\|a_{i,\cdot}\|_1 + \lambda_{neg} \mathbb{1}[y_i=0]\,\|a_{i,\cdot}\|_2^2
\]
其中 \(L_{ss}\) 是自监督重构/预测损失；TV 项促使注意力形成连续阶段；稀疏项抑制全月份均匀激活；负样本项约束正常用户注意力平坦、偏离低。最终输出异常区间 \([\hat{s}_i,\hat{e}_i]\)、区间置信度和用户级异常分。

5. **可度量的定位验证（关键，把 proxy 变 ground-truth）**：在正常用户上**人工注入合成异常**（突降、持续低谷、零值、缓变），注入月份已知，构成月级 ground-truth。用 IoU、point-adjusted F1、定位精度/召回评估 \(a_{i,m}\) 与注入区间的吻合度。这使「异常阶段定位」成为有真值、可复现、可投稿的指标，而不是事后人工 proxy。

与点 A 的互补关系因此从「分数拼接」升级为「能力分工」：点 A 用监督判别给用户级异常概率；点 B v2 用「自监督正常流形 + 弱监督 MIL 定位」在仅有用户级标签时**反推异常发生的时间区间**，并对点 A 低风险区的隐蔽阶段性异常给出可定位、可解释的输出。算法新颖性落在「自监督正常建模驱动的弱监督时序定位目标函数 + 区间结构正则 + 合成注入真值验证」，不再是新评价指标，也不再是误差后处理。

评价口径：
- 点 B v2 **不**以超过 G3 全局 AUC/F1 为成功标准。
- 成功标准是：合成注入实验上的定位 IoU/point-adjusted F1 显著优于「argmax 后处理」和「随机/均匀注意力」基线；且正常用户注意力显著低于异常用户。
- 用户级 AUC 仅作合理性 sanity check，不作为主结论。
- 未实测前，所有定位数值写「待验证」，不得编造。

- [Risk] 弱监督定位塌缩到单点或全激活 → Mitigation: TV + 稀疏 + 负样本平坦正则联合约束，并在合成注入集上调正则系数。
- [Risk] 注意力被用电规模主导 → Mitigation: 偏离度使用用户内标准化和正常参考标定，输入做 per-user scaling。
- [Risk] 合成注入分布与真实异常不一致 → Mitigation: 注入形态参照真实 G3 阳性样本的月级误差形态（突降/持续/零值/缓变），并报告多种注入形态下的定位指标。
- [Risk] 与点 A 重复 → Mitigation: 点 A 无时序定位输出且为全监督；点 B v2 是弱监督时序定位，二者任务与监督信号不同。
