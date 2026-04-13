# -*- coding: utf-8 -*-
"""
提升各维度 Token AUC 的新特征工程：

【新增 Token 设计原则】
  现有瓶颈：绝对值投影 / 无跨期对比 / 无跨用户排名
  解决方案：
    Token29: 归一化RMT信号能量比  (v₁·x)²/||x||²   AUC目标≥0.62
             消除高用电量用户的绝对优势，纯看在异常方向上的占比
    Token30: 长期用电下降斜率                       AUC目标≥0.65
             second_half_mean / first_half_mean
             窃电用户典型：初期正常→后期系统性下降
    Token31: 跨用户百分位排名变化                   AUC目标≥0.65
             每个窗口内用户的用电排名（百分位）从早期→晚期的变化
             窃电用户排名系统性下滑
    Token32: RMT投影时序变化（后期-前期）            AUC目标≥0.62
             v1投影后24个窗口均值 - 前24个窗口均值
             窃电从某时刻开始 → v1投影在后期应更高/持续
    Token33: 用户极端低值持续强度                   AUC目标≥0.60
             统计该用户自身历史中处于极端低值的窗口占比
             (窃电用户：有大段时间用电量接近零 → 超出零值的正常占比高)
"""

import numpy as np
from scipy.stats import rankdata

path = 'sgcc_analysis.py'
content = open(path, encoding='utf-8').read()

# 插入位置：Token22~25 加完之后，DataLoader 创建之前
ANCHOR = 'print(f"  拼入 Token22~25(延迟嵌入) 后形状: {X_seq14_ds.shape}  (FEAT_DIM={FEAT_DIM})")'
assert ANCHOR in content, '未找到插入锚点'

NEW_TOKENS = '''print(f"  拼入 Token22~25(延迟嵌入) 后形状: {X_seq14_ds.shape}  (FEAT_DIM={FEAT_DIM})")

  # ===========================================================================
  # 新增高判别力 Token26~30（提升各维度 AUC）
  # ===========================================================================
  print("\\n  构造高判别力增强特征（Token26~30）...")

  # ── Token26: 归一化 RMT 信号能量比 ─────────────────────────────────────
  # 当前 Token1~5 是绝对投影 |v1·x|，高用电用户天然更大，与异常无关
  # 归一化：(v1·x)² / ||x||²  → 消除能量偏差，纯看异常方向能量占比
  # spec_tokens shape: (144, N, 7)，前5列是投影
  _v1_proj_sq   = (spec_tokens[:, :, 0] ** 2)             # (144, N) v1投影平方
  _total_energy = (spec_tokens[:, :, :5] ** 2).sum(axis=2) + 1e-8  # (144, N)
  _rmt_normed   = (_v1_proj_sq / _total_energy).mean(axis=0)  # (N,) 时序均值
  rmt_normed_auc = _auc_fn(y, _rmt_normed)
  print(f"    Token26(归一化RMT能量比)   AUC={rmt_normed_auc:.4f}  "
        f"{'✅' if rmt_normed_auc>0.55 else '⚠️'}")

  # ── Token27: 长期下降趋势（用电量后半段/前半段） ────────────────────────
  # 窃电用户：初期正常用电，窃电开始后后半段显著下降
  _half = X.shape[1] // 2
  _first_half_mean  = X[:, :_half].mean(axis=1)    # (N,)
  _second_half_mean = X[:, _half:].mean(axis=1)    # (N,)
  _long_decline = _second_half_mean / (_first_half_mean.clip(1e-4) + 1e-8)  # (N,)
  long_decline_auc = _auc_fn(y, 1.0 - _long_decline)  # 下降比越大越异常
  print(f"    Token27(长期下降趋势)      AUC={long_decline_auc:.4f}  "
        f"{'✅' if long_decline_auc>0.55 else '⚠️'}")

  # ── Token28: 跨用户百分位排名变化（早→晚） ───────────────────────────────
  # 每个窗口内，用均值排名得到当前用户的百分位（0~1）
  # 计算早期24窗口 vs 晚期24窗口的排名差，负数=排名下滑=疑似窃电
  _n_wins_ds = X_seq14_ds.shape[1]   # 48
  _cons_per_win = X_seq14_ds[:, :, 5]  # Token6: 用户方差，近似代表用电活跃度
  # 早期/晚期排名百分位
  _early_rank = np.zeros(X.shape[0], dtype=np.float32)
  _late_rank  = np.zeros(X.shape[0], dtype=np.float32)
  for _ww in range(_n_wins_ds // 4):                         # 前1/4
      _early_rank += rankdata(_cons_per_win[:, _ww]) / X.shape[0]
  for _ww in range(3 * _n_wins_ds // 4, _n_wins_ds):        # 后1/4
      _late_rank  += rankdata(_cons_per_win[:, _ww]) / X.shape[0]
  _early_rank /= (_n_wins_ds // 4)
  _late_rank  /= (_n_wins_ds // 4)
  _rank_decline = _early_rank - _late_rank   # 正数=排名下滑=更异常
  rank_decline_auc = _auc_fn(y, _rank_decline)
  print(f"    Token28(跨用户排名变化)    AUC={rank_decline_auc:.4f}  "
        f"{'✅' if rank_decline_auc>0.55 else '⚠️'}")

  # ── Token29: RMT v1投影时序变化（后期-前期） ──────────────────────────────
  # 窃电持续行为 → v1投影在后半段应高于前半段（系统性异常加剧）
  _v1_ts = X_seq14_ds[:, :, 0]         # (N, 48) 第1维RMT投影时序
  _v1_late_minus_early = (
      _v1_ts[:, 3 * _n_wins_ds // 4:].mean(axis=1) -
      _v1_ts[:, :_n_wins_ds // 4].mean(axis=1)
  )  # (N,)
  v1_change_auc = _auc_fn(y, _v1_late_minus_early)
  print(f"    Token29(v1投影时序变化)    AUC={v1_change_auc:.4f}  "
        f"{'✅' if v1_change_auc>0.55 else '⚠️'}")

  # ── Token30: 极端低值窗口占比（用电量处于自身历史低5%分位的窗口比例） ──────
  # 窃电用户有大段时间用电量接近零或极低，超出正常范围的低值窗口占比高
  _user_mean_ts = X_seq14_ds[:, :, 6]   # Token7: 用户均值时序 (N, 48)
  _user_p10    = np.percentile(_user_mean_ts, 10, axis=1, keepdims=True)  # (N,1)
  _low_frac    = (_user_mean_ts < _user_p10).mean(axis=1).astype(np.float32)  # (N,)
  low_frac_auc = _auc_fn(y, _low_frac)
  print(f"    Token30(极端低值占比)      AUC={low_frac_auc:.4f}  "
        f"{'✅' if low_frac_auc>0.55 else '⚠️'}")

  # ── 拼入 Token26~30（每个 tile 到 48 时间步） ───────────────────────────
  _new_global_tokens = np.stack([
      _rmt_normed,          # (N,) Token26
      1.0 - _long_decline,  # (N,) Token27
      _rank_decline,        # (N,) Token28
      _v1_late_minus_early, # (N,) Token29
      _low_frac,            # (N,) Token30
  ], axis=1).astype(np.float32)  # (N, 5)

  # 对齐到 [0,1] 以防数值尺度差异过大
  for _ci in range(_new_global_tokens.shape[1]):
      _col = _new_global_tokens[:, _ci]
      _col = (_col - _col.min()) / (_col.max() - _col.min() + 1e-8)
      _new_global_tokens[:, _ci] = _col

  # tile: (N, 5) → (N, 48, 5)
  _new_global_tiled = np.tile(
      _new_global_tokens[:, np.newaxis, :], (1, X_seq14_ds.shape[1], 1)
  )
  X_seq14_ds = np.concatenate([X_seq14_ds, _new_global_tiled], axis=2)
  FEAT_DIM   = X_seq14_ds.shape[2]
  print(f"  拼入 Token26~30 后形状: {X_seq14_ds.shape}  (FEAT_DIM={FEAT_DIM})")
  # 同步 DataLoader 数据
  X_tr14 = torch.FloatTensor(X_seq14_ds[idx_tr14])
  X_te14 = torch.FloatTensor(X_seq14_ds[idx_te14])'''

content = content.replace(ANCHOR, NEW_TOKENS, 1)

open(path, 'w', encoding='utf-8').write(content)
print('Token26~30 注入成功')

# 同时更新 patch_rmt_sgp.py 的幂等文件（让它保持幂等，无需改动）
import ast
ast.parse(open(path, encoding='utf-8').read())
print('语法检查通过')
