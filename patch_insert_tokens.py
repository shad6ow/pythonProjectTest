# -*- coding: utf-8 -*-
"""在行1332后（Token22~25之后，DataLoader之前）插入 Token26~30，顶格代码"""

NEW_CODE = '''\

# =============================================================================
# 新增高判别力 Token26~30（提升各维度 Token AUC）
# 现有瓶颈：绝对值投影/无跨期对比/无跨用户排名
# Token26: 归一化RMT能量比  (v1·x)²/||x||²  → AUC目标≥0.62
# Token27: 长期下降趋势  后半段/前半段均值  → AUC目标≥0.65
# Token28: 跨用户百分位排名变化 早→晚       → AUC目标≥0.65
# Token29: v1投影时序变化 后期-前期          → AUC目标≥0.62
# Token30: 极端低值窗口占比                  → AUC目标≥0.60
# =============================================================================
from scipy.stats import rankdata as _rankdata
print("\\n  构造高判别力增强特征（Token26~30）...")

# ── Token26: 归一化RMT信号能量比 ─────────────────────────────────────────
# 绝对投影 |v1·x| 对高用电量用户天然偏大，与异常无关
# 归一化：(v1·x)² / sum((vk·x)² for k=1..5)  纯看异常方向占比
_v1_proj_sq   = (spec_tokens[:, :, 0] ** 2)               # (144, N)
_total_energy = (spec_tokens[:, :, :5] ** 2).sum(axis=2) + 1e-8  # (144, N)
_tok26 = (_v1_proj_sq / _total_energy).mean(axis=0).astype(np.float32)  # (N,)
_t26_auc = max(_auc_fn(y, _tok26), _auc_fn(y, -_tok26))
print(f"    Token26(归一化RMT能量比)   AUC={_t26_auc:.4f}  {'✅' if _t26_auc>0.55 else '⚠️'}")

# ── Token27: 长期用电下降趋势 ─────────────────────────────────────────────
# 窃电用户：初期正常，某时刻后后半段系统性下降 → 后半/前半 < 1
_half = X.shape[1] // 2
_tok27_raw = (X[:, _half:].mean(axis=1) /
              (X[:, :_half].mean(axis=1).clip(1e-4) + 1e-8))  # (N,)
# 下降比越大越异常（下降意味着 ratio<1，取负值使"异常更大"）
_tok27 = (1.0 - _tok27_raw).astype(np.float32)
_t27_auc = max(_auc_fn(y, _tok27), _auc_fn(y, -_tok27))
print(f"    Token27(长期下降趋势)      AUC={_t27_auc:.4f}  {'✅' if _t27_auc>0.55 else '⚠️'}")

# ── Token28: 跨用户百分位排名变化（早→晚） ───────────────────────────────
# 同一时间窗口内用户消费排名，窃电用户排名系统性下滑
_nw = X_seq14_ds.shape[1]   # 48
_cpw = X_seq14_ds[:, :, 6].astype(np.float64)  # Token7用户均值时序 (N,48)
_er  = np.zeros(X.shape[0], dtype=np.float32)
_lr  = np.zeros(X.shape[0], dtype=np.float32)
for _ww in range(_nw // 4):                     # 前1/4
    _er += (_rankdata(_cpw[:, _ww]) / X.shape[0]).astype(np.float32)
for _ww in range(3 * _nw // 4, _nw):            # 后1/4
    _lr += (_rankdata(_cpw[:, _ww]) / X.shape[0]).astype(np.float32)
_tok28 = _er / (_nw // 4) - _lr / (_nw // 4)   # 正数=排名下滑=异常
_t28_auc = max(_auc_fn(y, _tok28), _auc_fn(y, -_tok28))
print(f"    Token28(跨用户排名变化)    AUC={_t28_auc:.4f}  {'✅' if _t28_auc>0.55 else '⚠️'}")

# ── Token29: RMT v1投影时序变化（后期均值 - 前期均值） ─────────────────────
# 窃电持续：v1投影在后半段高于前半段（系统性异常加剧）
_v1s = X_seq14_ds[:, :, 0]  # (N,48) 第1维RMT投影时序
_tok29 = (_v1s[:, 3*_nw//4:].mean(axis=1) -
          _v1s[:, :_nw//4].mean(axis=1)).astype(np.float32)  # (N,)
_t29_auc = max(_auc_fn(y, _tok29), _auc_fn(y, -_tok29))
print(f"    Token29(v1投影时序变化)    AUC={_t29_auc:.4f}  {'✅' if _t29_auc>0.55 else '⚠️'}")

# ── Token30: 极端低值窗口占比 ─────────────────────────────────────────────
# 用户自身历史中低于P10的窗口占比（窃电用户有大段接近零的时期）
_um = X_seq14_ds[:, :, 6].astype(np.float64)  # (N,48)
_p10 = np.percentile(_um, 10, axis=1, keepdims=True)
_tok30 = (_um < _p10).mean(axis=1).astype(np.float32)  # (N,)
_t30_auc = max(_auc_fn(y, _tok30), _auc_fn(y, -_tok30))
print(f"    Token30(极端低值占比)      AUC={_t30_auc:.4f}  {'✅' if _t30_auc>0.55 else '⚠️'}")

# ── 归一化 & 拼入（tiled to 48 steps） ─────────────────────────────────
_new_g = np.stack([_tok26, _tok27, _tok28, _tok29, _tok30], axis=1)  # (N,5)
for _ci in range(_new_g.shape[1]):
    _c = _new_g[:, _ci]
    _new_g[:, _ci] = (_c - _c.min()) / (_c.max() - _c.min() + 1e-8)
_new_g_tiled = np.tile(_new_g[:, np.newaxis, :],
                        (1, X_seq14_ds.shape[1], 1))   # (N,48,5)
X_seq14_ds = np.concatenate([X_seq14_ds, _new_g_tiled], axis=2)
FEAT_DIM   = X_seq14_ds.shape[2]
print(f"  拼入 Token26~30 后形状: {X_seq14_ds.shape}  (FEAT_DIM={FEAT_DIM})")

'''

path = 'sgcc_analysis.py'
lines = open(path, encoding='utf-8').readlines()

# 找到目标插入行（"拼入 Token22~25" 那一行之后）
insert_after = None
for i, line in enumerate(lines):
    if '拼入 Token22~25(延迟嵌入) 后形状' in line:
        insert_after = i
        break

assert insert_after is not None, '未找到目标行'
print(f'在第 {insert_after+1} 行后插入')

new_code_lines = [l + '\n' if not l.endswith('\n') else l
                  for l in NEW_CODE.split('\n')]
lines = lines[:insert_after+1] + new_code_lines + lines[insert_after+1:]
open(path, 'w', encoding='utf-8').writelines(lines)

import ast
ast.parse(open(path, encoding='utf-8').read())
print('语法检查通过，Token26~30 已注入')
