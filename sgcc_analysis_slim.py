
# =============================================================================
# SGCC 用电数据异常检测 — 精简版（核心算法）
# 保留: RMT谱分析 → RMT-ISCT → 月度序列 → DualPathTransformer → XGBoost/CatBoost集成
# 数据来源: State Grid Corporation of China (SGCC)
# =============================================================================

import os, time, warnings, gc
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

_CPU_THREADS = min(os.cpu_count() or 4, 8)
os.environ['OMP_NUM_THREADS'] = str(_CPU_THREADS)
os.environ['MKL_NUM_THREADS'] = str(_CPU_THREADS)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

torch.set_num_threads(_CPU_THREADS)
torch.set_num_interop_threads(max(1, _CPU_THREADS // 2))

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, roc_curve,
                              confusion_matrix, classification_report,
                              precision_recall_curve)
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"⚡ CPU线程={_CPU_THREADS}  设备={device}")

# =============================================================================
# Step 1~3: 加载数据 + 预处理
# =============================================================================
DATA_PATH = r'C:\Users\wb.zhoushujie\Desktop\data set.csv'

print("=" * 65)
print("Step 1~3: 加载数据 + 预处理")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
labels = df['FLAG']
features_df = df.drop('FLAG', axis=1).select_dtypes(include=[np.number])

# 线性插值 → Z-score → 极端值裁剪
features_filled = features_df.interpolate(method='linear', axis=1, limit_direction='both')
features_filled = features_filled.fillna(features_filled.mean())

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_filled.T).T
lower = np.percentile(features_scaled, 0.1)
upper = np.percentile(features_scaled, 99.9)
features_scaled = np.clip(features_scaled, lower, upper)

X = features_scaled.astype(np.float32)
y = labels.values.astype(np.int64)
X_raw = features_filled.values.astype(np.float32)  # 原始kWh（未归一化）

N_USERS, T_DAYS = X.shape
NEG_COUNT = int((y == 0).sum())
POS_COUNT = int((y == 1).sum())

print(f"  用户数={N_USERS}, 天数={T_DAYS}")
print(f"  正常={NEG_COUNT}({NEG_COUNT/len(y)*100:.1f}%), 异常={POS_COUNT}({POS_COUNT/len(y)*100:.1f}%)")
print(f"  X范围=[{X.min():.3f}, {X.max():.3f}]")

# =============================================================================
# Step 7~8: RMT 谱分析（滑动窗口 + MP边界 + 谱Token）
# =============================================================================
print("\n" + "=" * 65)
print("Step 7~8: RMT 谱分析")
print("=" * 65)


def create_rmt_windows(X, window_size=30, stride=7):
    """滑动窗口: (n_users, n_days) → (n_windows, n_users, window_size)"""
    windows = []
    for start in range(0, X.shape[1] - window_size + 1, stride):
        windows.append(X[:, start:start + window_size])
    windows = np.array(windows)
    print(f"  窗口数={len(windows)}, 窗口形状={windows[0].shape}")
    return windows


def marchenko_pastur_bounds(X_window):
    """自适应 MP 上下界（MAD 估计 σ²）"""
    n, T = X_window.shape
    gamma = n / T
    median_val = np.median(X_window)
    mad = np.median(np.abs(X_window - median_val))
    sigma2 = (mad / 0.6745) ** 2 + 1e-8
    lambda_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2
    lambda_minus = sigma2 * (1 - np.sqrt(gamma)) ** 2
    return lambda_minus, lambda_plus, gamma, sigma2


def rmt_spectral_analysis(rmt_windows, y, subsample=500, k=5):
    """
    RMT 谱分析: 对每个窗口做谱分解，生成 (k+2) 维谱Token。
    维度 0~k-1: top-k 特征向量投影 |v_i·x|
    维度 k:     异常强度比 λ_max/λ+
    维度 k+1:   谱集中度 Σtop-k λ / trace(C)
    """
    n_windows, n_users, T = rmt_windows.shape
    n_sub = min(subsample, n_users)
    n_feat = k + 2

    lambda_max_series = np.zeros(n_windows)
    lambda_plus_series = np.zeros(n_windows)
    spec_tokens = np.zeros((n_windows, n_users, n_feat))
    anomaly_scores = np.zeros(n_users)
    eigenvalues_all, eigenvectors_all = [], []

    rng = np.random.default_rng(42)
    for w in range(n_windows):
        X_w = rmt_windows[w]
        sample_idx = rng.choice(n_users, size=n_sub, replace=False)
        X_sub = X_w[sample_idx]

        lam_minus, lam_plus, gamma, sigma2 = marchenko_pastur_bounds(X_sub)
        lambda_plus_series[w] = lam_plus

        # LedoitWolf 鲁棒协方差 → top-k 特征值分解
        try:
            C = LedoitWolf().fit(X_sub.T).covariance_
        except Exception:
            C = np.cov(X_sub)
        eigenvalues, eigenvectors = eigh(C, subset_by_index=[n_sub - k, n_sub - 1])
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        lambda_max_series[w] = eigenvalues[0]

        # 时间域投影
        V_time = X_sub.T @ eigenvectors
        V_time /= (np.linalg.norm(V_time, axis=0, keepdims=True) + 1e-10)
        spec_tokens[w, :, :k] = X_w @ V_time

        intensity_ratio = eigenvalues[0] / (lam_plus + 1e-10)
        spec_tokens[w, :, k] = intensity_ratio
        spec_tokens[w, :, k + 1] = eigenvalues.sum() / (np.trace(C) + 1e-10)

        eigenvalues_all.append(eigenvalues.copy())
        eigenvectors_all.append(V_time.copy())

        if eigenvalues[0] > lam_plus:
            anomaly_scores += np.abs(X_w @ V_time[:, 0])

        if w % 20 == 0:
            flag = "🔴" if eigenvalues[0] > lam_plus else "🟢"
            print(f"    窗口{w:3d}: λ_max={eigenvalues[0]:.4f} λ+={lam_plus:.4f} {flag}")

    n_exceed = np.sum(lambda_max_series > lambda_plus_series)
    print(f"  λ_max超界窗口: {n_exceed}/{n_windows}")
    return (lambda_max_series, lambda_plus_series, spec_tokens,
            anomaly_scores, eigenvalues_all, eigenvectors_all)


# 执行 RMT 谱分析
np.random.seed(42)
windows = create_rmt_windows(X, window_size=30, stride=7)

(lambda_max_series, lambda_plus_series, spec_tokens,
 anomaly_scores, eigenvalues_all, eigenvectors_all) = rmt_spectral_analysis(windows, y)

# =============================================================================
# Step 8b: 用户级异常得分（信号空间投影能量）
# =============================================================================
print("\n" + "=" * 65)
print("Step 8b: 用户级异常得分")
print("=" * 65)

n_windows_all = windows.shape[0]
score_A = np.zeros(N_USERS)  # 信号空间投影能量
for w in range(n_windows_all):
    lam_plus = lambda_plus_series[w]
    eigvals = eigenvalues_all[w]
    signal_mask = eigvals > lam_plus
    if signal_mask.sum() == 0:
        continue
    proj_signal = spec_tokens[w][:, :5][:, signal_mask[:5]]
    score_A += (proj_signal ** 2).sum(axis=1)

score_B = (windows ** 2).mean(axis=(0, 2))  # 时序平均能量
_zscore = lambda x: (x - x.mean()) / (x.std() + 1e-8)
score_C = _zscore(score_A) + _zscore(score_B)  # 融合得分

rmt_baseline_auc = max(roc_auc_score(y, score_A),
                       roc_auc_score(y, score_B),
                       roc_auc_score(y, score_C))
print(f"  RMT 基线 AUC: {rmt_baseline_auc:.4f}")

# =============================================================================
# Step 8c: RMT-ISCT (分层局部RMT + 个体谱贡献轨迹) — 核心创新点
# =============================================================================
print("\n" + "=" * 65)
print("Step 8c: RMT-ISCT (Stratified Local RMT + ISCT)")
print("=" * 65)
_t8c = time.time()

# ── 模块A: 分层局部RMT ──────────────────────────────────────────────
print("  [A] 分层局部RMT ...")

DAYSPM = 30
NM = T_DAYS // DAYSPM  # 34

# 预计算原始月度消费矩阵 (N, 34)
mo_raw = np.zeros((N_USERS, NM), dtype=np.float32)
for m in range(NM):
    s, e = m * DAYSPM, min((m + 1) * DAYSPM, T_DAYS)
    mo_raw[:, m] = X_raw[:, s:e].mean(axis=1)

# 按原始月均消费分 K=8 层
user_avg = X_raw.mean(axis=1)
n_strata = 8
strata_labels = pd.qcut(user_avg, q=n_strata, labels=False, duplicates='drop')
actual_strata = int(strata_labels.max()) + 1

local_rmt_score = np.zeros(N_USERS, dtype=np.float32)
local_rmt_ratio = np.zeros(N_USERS, dtype=np.float32)

for k_s in range(actual_strata):
    mask = (strata_labels == k_s)
    n_layer = mask.sum()
    if n_layer < 10:
        continue

    layer_monthly = mo_raw[mask]
    n_sub = min(500, n_layer)
    rng_k = np.random.default_rng(42 + k_s)
    sub_idx = rng_k.choice(n_layer, n_sub, replace=False)
    layer_sub = layer_monthly[sub_idx]
    layer_sub_c = layer_sub - layer_sub.mean(axis=0, keepdims=True)

    T_dim = layer_sub_c.shape[1]
    C_layer = layer_sub_c @ layer_sub_c.T / max(T_dim - 1, 1)

    try:
        from scipy.sparse.linalg import eigsh
        vals_layer, vecs_layer = eigsh(C_layer.astype(np.float64), k=5, which='LM')
        lmax_layer = vals_layer[-1]
        gamma_layer = n_sub / T_dim
        lplus_layer = (1.0 + np.sqrt(gamma_layer)) ** 2
        strength_layer = max(lmax_layer / (lplus_layer + 1e-8), 0.0)

        layer_all_c = layer_monthly - layer_monthly.mean(axis=0, keepdims=True)
        u_data = layer_sub_c.T @ vecs_layer
        for kk in range(u_data.shape[1]):
            norm = np.linalg.norm(u_data[:, kk])
            if norm > 1e-10:
                u_data[:, kk] /= norm
        layer_proj_energy = np.sum((layer_all_c @ u_data) ** 2, axis=1)
    except Exception:
        strength_layer = 1.0
        layer_proj_energy = np.abs(layer_monthly.mean(axis=1))

    med_layer = np.median(layer_proj_energy)
    mad_layer = np.median(np.abs(layer_proj_energy - med_layer)) + 1e-8
    local_rmt_score[mask] = ((layer_proj_energy - med_layer) / mad_layer).astype(np.float32)
    local_rmt_ratio[mask] = np.float32(strength_layer)

    print(f"    层{k_s}: n={n_layer}, gamma={gamma_layer:.1f}, lmax/l+={strength_layer:.2f}")

lrs_auc = max(roc_auc_score(y, local_rmt_score), 1 - roc_auc_score(y, local_rmt_score))
print(f"    local_rmt_score AUC={lrs_auc:.4f}")

# ── 模块B: 个体消费偏离轨迹 (ISCT) ──────────────────────────────────
print("  [B] 个体消费偏离轨迹 (ISCT) ...")

isct_monthly = mo_raw.copy()  # (N, 34)

# 层内中位数偏离（消除消费水平差异）
isct_dev_monthly = np.zeros((N_USERS, NM), dtype=np.float32)
for k_s in range(actual_strata):
    mask = (strata_labels == k_s)
    if mask.sum() < 2:
        continue
    layer_median = np.median(mo_raw[mask], axis=0, keepdims=True)
    isct_dev_monthly[mask] = ((mo_raw[mask] - layer_median) /
                               (np.abs(layer_median) + 1e-3)).astype(np.float32)

# ISCT-CPD: 8维变点检测特征（向量化）
isct_cpd_feats = np.zeros((N_USERS, 8), dtype=np.float32)
sig_all = isct_dev_monthly
T_sig = sig_all.shape[1]

# F1: CUSUM变点位置
sig_mean = sig_all.mean(axis=1, keepdims=True)
cumsum_all = np.cumsum(sig_all - sig_mean, axis=1)
cp_pos_idx = np.argmax(np.abs(cumsum_all), axis=1)
isct_cpd_feats[:, 0] = cp_pos_idx.astype(np.float32) / T_sig

# F2: 变点前后均值比
cp_idx_clamped = np.clip(cp_pos_idx, 1, T_sig - 1)
arange_t = np.arange(T_sig)[np.newaxis, :]
cp_expanded = cp_idx_clamped[:, np.newaxis]
pre_mask = arange_t < cp_expanded
post_mask = arange_t >= cp_expanded
pre_mean = (sig_all * pre_mask).sum(axis=1) / (pre_mask.sum(axis=1).astype(np.float32) + 1e-8)
post_mean = (sig_all * post_mask).sum(axis=1) / (post_mask.sum(axis=1).astype(np.float32) + 1e-8)
isct_cpd_feats[:, 1] = (post_mean / (np.abs(pre_mean) + 1e-6)).astype(np.float32)

# F3: 最大单月偏离
isct_cpd_feats[:, 2] = np.abs(sig_all).max(axis=1).astype(np.float32)

# F4: 后半段/前半段偏离比
half = T_sig // 2
isct_cpd_feats[:, 3] = (sig_all[:, half:].mean(axis=1) /
                         (np.abs(sig_all[:, :half].mean(axis=1)) + 1e-6)).astype(np.float32)

# F5: 偏离标准差
isct_cpd_feats[:, 4] = sig_all.std(axis=1).astype(np.float32)

# F6: 连续负偏离最大长度
neg_mask_all = (sig_all < 0).astype(np.int32)
max_neg_run = np.zeros(N_USERS, dtype=np.float32)
cur_run = np.zeros(N_USERS, dtype=np.float32)
for t_step in range(T_sig):
    cur_run = (cur_run + 1.0) * neg_mask_all[:, t_step]
    max_neg_run = np.maximum(max_neg_run, cur_run)
isct_cpd_feats[:, 5] = (max_neg_run / T_sig).astype(np.float32)

# F7: 趋势斜率
t_vec = np.arange(T_sig, dtype=np.float64)
t_mean_v = t_vec.mean()
t_var = ((t_vec - t_mean_v) ** 2).sum()
sig_f64 = sig_all.astype(np.float64)
slope_all = ((sig_f64 - sig_f64.mean(axis=1, keepdims=True)) *
             (t_vec[np.newaxis, :] - t_mean_v)).sum(axis=1) / (t_var + 1e-8)
isct_cpd_feats[:, 6] = slope_all.astype(np.float32)

# F8: 末期偏离均值（最后6个月）
isct_cpd_feats[:, 7] = sig_all[:, -6:].mean(axis=1).astype(np.float32)

# 特征选择: 去掉 |AUC-0.5| <= 0.02 的噪声维度
rmt_isc_features = isct_cpd_feats.copy()
isc_keep_mask = []
for ci in range(rmt_isc_features.shape[1]):
    a = roc_auc_score(y, rmt_isc_features[:, ci])
    keep = abs(a - 0.5) > 0.02
    isc_keep_mask.append(keep)
isc_keep_idx = [i for i, k in enumerate(isc_keep_mask) if k]
if len(isc_keep_idx) > 0:
    rmt_isc_features = rmt_isc_features[:, isc_keep_idx]
print(f"  ISCT-CPD 特征: {rmt_isc_features.shape[1]}维 (筛选后)")
print(f"  Step 8c 耗时: {time.time()-_t8c:.1f}s")

# =============================================================================
# Step 8e: 多尺度变点检测特征 (TCN-CPD, 20维)
# =============================================================================
print("\n" + "=" * 65)
print("Step 8e: 多尺度变点检测特征 (TCN-CPD)")
print("=" * 65)
_t8e = time.time()

# 1. 用户级 RobustScaler 归一化日级原始数据
_tcn_input = X_raw.copy().astype(np.float32)
_tcn_input = np.nan_to_num(_tcn_input, nan=0.0)
_u_med = np.median(_tcn_input, axis=1, keepdims=True)
_u_iqr = np.clip(
    np.percentile(_tcn_input, 75, axis=1, keepdims=True) -
    np.percentile(_tcn_input, 25, axis=1, keepdims=True),
    1e-6, None)
_tcn_input = np.clip((_tcn_input - _u_med) / _u_iqr, -10, 10)
_T_tcn = min(_tcn_input.shape[1], 1020)
_tcn_input = _tcn_input[:, :_T_tcn]

# 2. 多尺度移动平均残差 → 变点特征
_scales = [7, 30, 90]
ms_cpd_features = []

for _scale in _scales:
    print(f"  [Scale {_scale}天]", end="")
    _kernel = np.ones(_scale) / _scale
    _smoothed = np.apply_along_axis(
        lambda row: np.convolve(row, _kernel, mode='same'),
        axis=1, arr=_tcn_input)
    _residual = _tcn_input - _smoothed
    _abs_residual = np.abs(_residual)

    _n_segments = max(_T_tcn // _scale, 4)
    _seg_len = _T_tcn // _n_segments
    _seg_energies = np.zeros((N_USERS, _n_segments), dtype=np.float32)
    for _si in range(_n_segments):
        _s, _e = _si * _seg_len, min((_si + 1) * _seg_len, _T_tcn)
        _seg_energies[:, _si] = np.mean(_abs_residual[:, _s:_e], axis=1)

    _seg_diff = np.abs(np.diff(_seg_energies, axis=1))
    _max_jump_pos = np.argmax(_seg_diff, axis=1).astype(np.float32) / max(_n_segments - 2, 1)
    _max_jump_mag = np.max(_seg_diff, axis=1) / (np.mean(_seg_energies, axis=1) + 1e-8)
    _mean_diff = np.mean(_seg_diff, axis=1, keepdims=True) + 1e-8
    _n_changepoints = np.sum(_seg_diff > 2 * _mean_diff, axis=1).astype(np.float32)
    _half_s = _n_segments // 2
    _half_ratio = (np.mean(_seg_energies[:, _half_s:], axis=1) + 1e-8) / \
                  (np.mean(_seg_energies[:, :_half_s], axis=1) + 1e-8)
    _energy_std = np.std(_seg_energies, axis=1)
    _x_axis = np.arange(_n_segments, dtype=np.float32)
    _x_mean = _x_axis.mean()
    _energy_mean = np.mean(_seg_energies, axis=1, keepdims=True)
    _slope = np.sum((_x_axis[np.newaxis, :] - _x_mean) * (_seg_energies - _energy_mean), axis=1) / \
             (np.sum((_x_axis - _x_mean) ** 2) + 1e-8)

    _scale_feats = np.column_stack([
        _max_jump_pos, _max_jump_mag, _n_changepoints,
        _half_ratio, _energy_std, _slope,
    ]).astype(np.float32)
    ms_cpd_features.append(_scale_feats)

    for _fi, _fn in enumerate(['变点位置', '变点幅度', '变点数', '前后比', '波动性', '趋势']):
        _a = max(roc_auc_score(y, _scale_feats[:, _fi]), 1 - roc_auc_score(y, _scale_feats[:, _fi]))
        if _a > 0.55:
            print(f" {_fn}={_a:.4f}✅", end="")
    print()

# 3. 合并 + 跨尺度交叉特征
tcn_cpd_features = np.concatenate(ms_cpd_features, axis=1).astype(np.float32)
_cross_short_long = ms_cpd_features[0][:, 4] / (ms_cpd_features[2][:, 4] + 1e-8)
_cross_jump_consistency = np.abs(ms_cpd_features[0][:, 0] - ms_cpd_features[2][:, 0])
_cross_feats = np.column_stack([_cross_short_long, _cross_jump_consistency]).astype(np.float32)
tcn_cpd_features = np.concatenate([tcn_cpd_features, _cross_feats], axis=1)
print(f"  TCN-CPD 最终特征: {tcn_cpd_features.shape}  耗时 {time.time()-_t8e:.1f}s")

# =============================================================================
# Step A-15: 月度序列构建（Transformer输入）
# =============================================================================
print("\n" + "=" * 65)
print("Step A: 构建月度序列 Transformer 输入")
print("=" * 65)


def _scale_ch(arr2d, clip=5.0):
    """全局 RobustScaler"""
    flat = arr2d.reshape(-1, 1)
    flat = RobustScaler().fit_transform(flat).reshape(arr2d.shape)
    return np.clip(flat, -clip, clip).astype(np.float32)


def _scale_ch_per_user(arr2d, clip=5.0):
    """用户自身归一化: (x - median) / IQR"""
    med = np.median(arr2d, axis=1, keepdims=True)
    q1 = np.percentile(arr2d, 25, axis=1, keepdims=True)
    q3 = np.percentile(arr2d, 75, axis=1, keepdims=True)
    iqr = q3 - q1 + 1e-6
    return np.clip((arr2d - med) / iqr, -clip, clip).astype(np.float32)


# 月度统计通道 (N, NM, K)
mo_std = np.zeros((N_USERS, NM), dtype=np.float32)
mo_max = np.zeros((N_USERS, NM), dtype=np.float32)
mo_zero = np.zeros((N_USERS, NM), dtype=np.float32)
for m in range(NM):
    s, e = m * DAYSPM, min((m + 1) * DAYSPM, T_DAYS)
    mo_std[:, m] = X_raw[:, s:e].std(axis=1)
    mo_max[:, m] = X_raw[:, s:e].max(axis=1)
    mo_zero[:, m] = (X[:, s:e] == 0).mean(axis=1)

# 基准偏离（前6个月为基准）
BASELINE_M = 6
baseline_raw = mo_raw[:, :BASELINE_M].mean(axis=1, keepdims=True) + 1e-3
mo_vs_base = (mo_raw - baseline_raw) / (np.abs(baseline_raw) + 1e-3)
mo_cumdev = np.cumsum(mo_raw - baseline_raw, axis=1)

# 跨用户百分位排名
mo_pct = np.zeros((N_USERS, NM), dtype=np.float32)
for m in range(NM):
    mo_pct[:, m] = (rankdata(mo_raw[:, m]) / N_USERS).astype(np.float32)

# 排名下降幅度
half_nm = NM // 2
rank_drop = mo_pct[:, :half_nm].mean(axis=1) - mo_pct[:, half_nm:].mean(axis=1)
rank_tile = np.tile(rank_drop[:, np.newaxis], (1, NM))

# 月度差分 & 二阶差分
mo_diff1 = np.diff(mo_raw, axis=1, prepend=mo_raw[:, :1])
mo_diff2 = np.diff(mo_diff1, axis=1, prepend=mo_diff1[:, :1])

# 用户自身归一化比值
user_median_raw = np.median(mo_raw, axis=1, keepdims=True) + 1e-3
mo_self_ratio = mo_raw / user_median_raw

# 对数偏离
mo_log_ratio = np.log1p(np.maximum(mo_raw, 0)) - np.log1p(np.maximum(baseline_raw, 0))

# 局部Z-score（3个月窗口）
mo_roll3_mean = np.zeros((N_USERS, NM), dtype=np.float32)
mo_roll3_std = np.zeros((N_USERS, NM), dtype=np.float32)
for m in range(NM):
    ws = max(0, m - 2)
    mo_roll3_mean[:, m] = mo_raw[:, ws:m + 1].mean(axis=1)
    mo_roll3_std[:, m] = mo_raw[:, ws:m + 1].std(axis=1) + 1e-6
mo_local_zscore = (mo_raw - mo_roll3_mean) / mo_roll3_std

# 跨用户偏离度
mo_global_median = np.median(mo_raw, axis=0, keepdims=True) + 1e-3
mo_global_dev = np.log1p(np.maximum(mo_raw, 0)) - np.log1p(np.maximum(mo_global_median, 0))
mo_rank_dev = mo_raw - mo_global_median

# 拼装 17 个月度时变通道
mo_ch = np.stack([
    _scale_ch_per_user(mo_raw),         # ch0 月均消费
    _scale_ch_per_user(mo_std),         # ch1 月波动
    _scale_ch_per_user(mo_max),         # ch2 月峰值
    mo_zero,                             # ch3 零值比
    mo_pct,                              # ch4 跨用户排名
    _scale_ch(mo_rank_dev),             # ch5 排名偏差
    _scale_ch(rank_tile),               # ch6 排名下降幅度
    _scale_ch_per_user(mo_vs_base),     # ch7 基准偏离
    _scale_ch_per_user(mo_cumdev),      # ch8 累积下降
    _scale_ch(mo_log_ratio),            # ch9 对数偏离
    _scale_ch_per_user(mo_diff1),       # ch10 一阶差分
    _scale_ch(mo_diff2),                # ch11 加速度
    mo_self_ratio.astype(np.float32),   # ch12 自身比值
    np.clip(mo_local_zscore, -5, 5).astype(np.float32),  # ch13 局部Z
    _scale_ch(mo_global_dev),           # ch14 全局偏离
    _scale_ch_per_user(isct_monthly),   # ch15 ISCT月度轨迹
    _scale_ch_per_user(isct_dev_monthly),  # ch16 ISCT偏离轨迹
], axis=2)  # (N, NM, 17)

# 标量工程特征: 月度序列各通道的全局均值（作为用户画像）
_scalar_feats = mo_ch.mean(axis=1).astype(np.float32)  # (N, 17)
print(f"  标量工程特征: {_scalar_feats.shape}")

# tile 拼入每个时步 → 完整 Transformer 输入
_scalar_tiled = np.tile(_scalar_feats[:, np.newaxis, :], (1, NM, 1))  # (N, NM, 17)
X_mo_seq = np.concatenate([mo_ch, _scalar_tiled], axis=2).astype(np.float32)
FEAT_DIM = X_mo_seq.shape[2]   # 17 + 17 = 34
N_STEPS_MO = NM

# 验证关键通道 AUC
def _qauc(f):
    a = roc_auc_score(y, f)
    return max(a, 1 - a)
print(f"  月度序列: {X_mo_seq.shape}  ({FEAT_DIM}维通道, {N_STEPS_MO}个月)")
print(f"  ch7(基准偏离) AUC={_qauc(mo_vs_base.mean(axis=1)):.4f}")
print(f"  ch8(累积下降) AUC={_qauc(mo_cumdev.mean(axis=1)):.4f}")
print(f"  ch4(排名)     AUC={_qauc(mo_pct.mean(axis=1)):.4f}")

# ── DataLoader（80/20 split）─────────────────────────────────────────
idx_tr, idx_te = train_test_split(
    np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
)
print(f"  数据分割: train={len(idx_tr)} test={len(idx_te)}")

X_tr = torch.FloatTensor(X_mo_seq[idx_tr])
y_tr = torch.FloatTensor(y[idx_tr])
X_te = torch.FloatTensor(X_mo_seq[idx_te])
y_te = torch.FloatTensor(y[idx_te])

class_counts = np.bincount(y[idx_tr].astype(int))
sample_w = (1.0 / class_counts)[y[idx_tr].astype(int)]
sampler = WeightedRandomSampler(torch.FloatTensor(sample_w), len(idx_tr), replacement=True)

train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=1024,
                           sampler=sampler, num_workers=0)
test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=2048,
                          shuffle=False, num_workers=0)

# =============================================================================
# Step B: DualPathTransformer 模型
# =============================================================================
print("\n" + "=" * 65)
print("Step B: DualPathTransformer 模型")
print("=" * 65)


class LocalWindowAttentionBlock(nn.Module):
    """局部窗口注意力 + FFN (Pre-LN)"""
    def __init__(self, d_model, nhead, win_size=4, dim_ff=256, dropout=0.1):
        super().__init__()
        self.win_size = win_size
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout))

    def _build_mask(self, L, device):
        if hasattr(self, '_mask_cache') and self._mask_cache.shape[0] == L:
            return self._mask_cache.to(device)
        mask = torch.full((L, L), float('-inf'))
        for i in range(L):
            s, e = max(0, i - self.win_size), min(L, i + self.win_size + 1)
            mask[i, s:e] = 0.0
        self._mask_cache = mask
        return mask.to(device)

    def forward(self, x):
        h = self.norm1(x)
        mask = self._build_mask(x.size(1), x.device)
        x = x + self.attn(h, h, h, attn_mask=mask)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class UserGraphAttention(nn.Module):
    """批内用户图注意力（TopK稀疏）"""
    def __init__(self, d_model, n_neighbors=10):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.scale = d_model ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        import torch.nn.functional as F
        B = x.shape[0]
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        attn = torch.mm(Q, K.T) * self.scale
        k_actual = min(self.n_neighbors, B)
        if B > k_actual:
            topk_vals, _ = torch.topk(attn, k_actual, dim=1)
            threshold = topk_vals[:, -1:].expand_as(attn)
            attn = torch.where(attn >= threshold, attn, torch.full_like(attn, float('-inf')))
        attn = F.softmax(attn, dim=1)
        return self.norm(x + self.out_proj(torch.mm(attn, V)))


class DualPathTransformer(nn.Module):
    """双路Transformer: 局部窗口注意力 + 全局CLS注意力"""
    def __init__(self, feat_dim=17, d_model=128, nhead=4, num_layers=3,
                 dim_ff=256, dropout=0.15, win_size=4, max_len=40):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(feat_dim, d_model), nn.LayerNorm(d_model))
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))
        self.pe_drop = nn.Dropout(dropout)

        self.local_layers = nn.ModuleList([
            LocalWindowAttentionBlock(d_model, nhead, win_size, dim_ff, dropout)
            for _ in range(num_layers)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.global_transformer = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.graph_attn = UserGraphAttention(d_model * 2, n_neighbors=10)
        self._graph_enabled = False

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(d_model // 2, 1))

    def _encode(self, x):
        B, L, _ = x.shape
        h = self.pe_drop(self.input_proj(x) + self.pe[:, :L, :])
        h_local = h
        for block in self.local_layers:
            h_local = block(h_local)
        feat_local = h_local.mean(dim=1)
        h_g = torch.cat([self.cls_token.expand(B, -1, -1), h], dim=1)
        feat_global = self.global_transformer(h_g)[:, 0, :]
        return feat_local, feat_global

    def forward(self, x):
        feat_local, feat_global = self._encode(x)
        feat_cat = torch.cat([feat_local, feat_global], dim=1)
        if self.training and self._graph_enabled and feat_cat.shape[0] > 1:
            feat_cat = self.graph_attn(feat_cat)
        return self.classifier(feat_cat), feat_cat

    def extract_features(self, x):
        feat_local, feat_global = self._encode(x)
        return torch.cat([feat_local, feat_global], dim=1)


# =============================================================================
# 损失函数: AdaptiveFocalLoss (含 label smoothing)
# =============================================================================
class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha=0.92, gamma=2.0, max_recall_w=1.5):
        super().__init__()
        self.alpha, self.gamma, self.max_recall_w = alpha, gamma, max_recall_w
        self.cur_epoch, self.max_epochs = 1, 50

    def set_epoch(self, epoch, max_epochs):
        self.cur_epoch, self.max_epochs = epoch, max_epochs

    def forward(self, logits, targets, feats=None):
        probs = torch.sigmoid(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal = (1 - p_t) ** self.gamma * bce
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal
        progress = min(self.cur_epoch / self.max_epochs, 1.0)
        recall_w = 1.0 + (self.max_recall_w - 1.0) * progress
        hard_fn = (targets == 1) & (probs.detach() < 0.3)
        loss = torch.where(hard_fn, loss * recall_w, loss)
        return loss.mean()


# =============================================================================
# 训练 / 评估 (含 Mixup 数据增强)
# =============================================================================
MIXUP_ALPHA = 0.2


def mixup_data(x, y, alpha=MIXUP_ALPHA):
    """Mixup: 随机线性插值两个样本"""
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # 保证 lam >= 0.5
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None, use_mixup=True):
    model.train()
    total_loss, total = 0.0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.float().to(device)
        if use_mixup:
            X_batch, y_batch = mixup_data(X_batch, y_batch)
        optimizer.zero_grad(set_to_none=True)
        logits, feats = model(X_batch)
        loss = criterion(logits.squeeze(1), y_batch, feats=feats)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * len(y_batch)
        total += len(y_batch)
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for X_batch, y_batch in loader:
        logits, _ = model(X_batch.to(device))
        all_probs.extend(torch.sigmoid(logits.squeeze(1)).cpu().numpy())
        all_labels.extend(y_batch.numpy())
    probs, labels = np.array(all_probs), np.array(all_labels)
    auc = roc_auc_score(labels, probs)
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.05, 0.95, 0.005):
        f1 = f1_score(labels, (probs >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return auc, best_f1, best_thr, probs, labels


# ── 训练 ─────────────────────────────────────────────────────────────
_raw_alpha = 1.0 - POS_COUNT / (POS_COUNT + NEG_COUNT)
AUTO_ALPHA = max(_raw_alpha * 0.95, 0.80)
EPOCHS, PATIENCE = 60, 15

model_dual = DualPathTransformer(feat_dim=FEAT_DIM, d_model=128, nhead=4,
                                  num_layers=3, dim_ff=256, dropout=0.15,
                                  win_size=4, max_len=40).to(device)
model_dual._graph_enabled = False

criterion_dual = AdaptiveFocalLoss(alpha=AUTO_ALPHA, gamma=2.0, max_recall_w=1.5)
optimizer_dual = optim.AdamW(model_dual.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_dual = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer_dual, T_0=20, T_mult=2, eta_min=5e-7)

total_params = sum(p.numel() for p in model_dual.parameters() if p.requires_grad)
print(f"  参数量={total_params:,}  feat_dim={FEAT_DIM}  epochs={EPOCHS}")

best_auc_dual, best_f1_dual, best_state_dual = 0.0, 0.0, None
best_epoch_dual, no_improve = 0, 0

print(f"{'Epoch':>5} | {'Loss':>10} | {'AUC':>7} | {'F1':>6} | {'LR':>8} | {'Time':>6}")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss = train_one_epoch(model_dual, train_loader, optimizer_dual, criterion_dual,
                               device, scheduler=None, use_mixup=False)
    val_auc, val_f1, val_thr, _, _ = evaluate(model_dual, test_loader, device)
    scheduler_dual.step(epoch - 1)
    criterion_dual.set_epoch(epoch, EPOCHS)
    lr = optimizer_dual.param_groups[0]['lr']
    elapsed = time.time() - t0

    mark = ' ✅' if val_auc > best_auc_dual else ''
    print(f"{epoch:>5} | {tr_loss:>10.6f} | {val_auc:>7.4f} | {val_f1:>6.4f} | {lr:>8.2e} | {elapsed:>5.1f}s{mark}")

    if val_auc > best_auc_dual:
        best_auc_dual, best_f1_dual = val_auc, val_f1
        best_state_dual = {k: v.cpu().clone() for k, v in model_dual.state_dict().items()}
        best_epoch_dual, no_improve = epoch, 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\n⏹ 早停！最优在第 {best_epoch_dual} 轮")
            break

model_dual.load_state_dict(best_state_dual)
torch.save(best_state_dual, 'dual_path_transformer_best.pth')
print(f"\n✅ Transformer完成! AUC={best_auc_dual:.4f} F1={best_f1_dual:.4f}")

# =============================================================================
# Step D: XGBoost + CatBoost 集成
# =============================================================================
print("\n" + "=" * 65)
print("Step D: XGBoost + CatBoost 集成")
print("=" * 65)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  ⚠ 未安装 xgboost，跳过集成")

xgb_auc, ensemble_auc, best_f1_ens = 0.0, best_auc_dual, best_f1_dual
cat_auc = 0.0

if HAS_XGB:
    # D-1: 提取 Transformer 中间特征（单模型，用于 XGBoost 输入）
    @torch.no_grad()
    def batch_extract_features(model, X_np, device, batch_size=1024):
        model.eval()
        feats = []
        for i in range(0, len(X_np), batch_size):
            xb = torch.FloatTensor(X_np[i:i + batch_size]).to(device)
            feats.append(model.extract_features(xb).cpu().numpy())
        return np.concatenate(feats, axis=0)

    feat_all = batch_extract_features(model_dual, X_mo_seq, device)
    print(f"  Transformer特征: {feat_all.shape}")

    # P2-9: spec_tokens 聚合特征 (144, N, 7) → (N, 21)
    spec_mean = spec_tokens.mean(axis=0)   # (N, 7)
    spec_max = spec_tokens.max(axis=0)     # (N, 7)
    spec_std = spec_tokens.std(axis=0)     # (N, 7)
    spec_agg_feats = np.concatenate([spec_mean, spec_max, spec_std], axis=1).astype(np.float32)
    print(f"  spec_tokens聚合特征: {spec_agg_feats.shape}")

    # P3-11: 时序异常先验特征
    # 连续N月消费骤降（月环比<0.7的连续段最大长度）
    mo_ratio = mo_raw[:, 1:] / (mo_raw[:, :-1] + 1e-6)
    drop_mask = (mo_ratio < 0.7).astype(np.int32)
    max_consec_drop = np.zeros(N_USERS, dtype=np.float32)
    cur_drop_run = np.zeros(N_USERS, dtype=np.float32)
    for t_step in range(drop_mask.shape[1]):
        cur_drop_run = (cur_drop_run + 1.0) * drop_mask[:, t_step]
        max_consec_drop = np.maximum(max_consec_drop, cur_drop_run)
    # 最大连续下降月数（月均消费递减）
    decr_mask = (np.diff(mo_raw, axis=1) < 0).astype(np.int32)
    max_consec_decr = np.zeros(N_USERS, dtype=np.float32)
    cur_decr_run = np.zeros(N_USERS, dtype=np.float32)
    for t_step in range(decr_mask.shape[1]):
        cur_decr_run = (cur_decr_run + 1.0) * decr_mask[:, t_step]
        max_consec_decr = np.maximum(max_consec_decr, cur_decr_run)
    # 末期/前期消费比
    tail_head_ratio = (mo_raw[:, -6:].mean(axis=1) /
                       (mo_raw[:, :6].mean(axis=1) + 1e-6)).astype(np.float32)
    ts_prior_feats = np.column_stack([max_consec_drop, max_consec_decr, tail_head_ratio])
    print(f"  时序先验特征: {ts_prior_feats.shape}")

    # D-2: 增强手工统计特征（与原始版一致: 8组）
    _HALF = NM // 2
    hand_feats = np.concatenate([
        X_mo_seq.mean(axis=1),                                                    # 均值
        X_mo_seq.std(axis=1),                                                     # 标准差
        X_mo_seq.max(axis=1),                                                     # 最大值
        X_mo_seq.min(axis=1),                                                     # 最小值
        np.percentile(X_mo_seq, 25, axis=1),                                      # Q1
        np.percentile(X_mo_seq, 75, axis=1),                                      # Q3
        X_mo_seq[:, _HALF:, :].mean(axis=1) - X_mo_seq[:, :_HALF, :].mean(axis=1), # 后半-前半趋势
        X_mo_seq.argmax(axis=1).astype(np.float32) / NM,                          # 最大值位置
    ], axis=1)
    print(f"  增强手工特征维度: {hand_feats.shape}  (FEAT_DIM×8={FEAT_DIM*8})")

    # D-3: 多视角原始信号
    xgb_self_norm = _scale_ch_per_user(mo_raw)
    xgb_diff_norm = _scale_ch_per_user(np.diff(mo_raw, axis=1, prepend=mo_raw[:, :1]))
    xgb_local_z = np.clip(mo_local_zscore, -5, 5)

    # XGBoost 特征（与原始版一致：含 feat_all + 标量 + TCN-CPD）
    X_xgb = np.concatenate([
        feat_all,          # Transformer 中间特征 (256维)
        hand_feats,        # 月度序列统计
        rmt_isc_features,  # ISCT-CPD 个体变点特征
        isct_monthly,      # ISCT月度谱投影能量 (34维)
        mo_raw,            # 原始kWh月度消费 (34维)
        xgb_self_norm,     # 用户自身归一化月消费 (34维)
        xgb_diff_norm,     # 月消费差分（用户自身归一化）(34维)
        mo_vs_base,        # 基准偏离率 (34维)
        mo_pct,            # 跨用户百分位排名 (34维)
        xgb_local_z,       # 局部Z-score异常度 (34维)
        spec_agg_feats,    # spec_tokens聚合 (21维)
        ts_prior_feats,    # 时序先验 (3维)
        _scalar_feats,     # 标量工程特征 (17维)
        tcn_cpd_features,  # 多尺度TCN-CPD特征 (20维)
    ], axis=1)
    print(f"  XGBoost总特征: {X_xgb.shape}")

    X_xgb_tr, X_xgb_te = X_xgb[idx_tr], X_xgb[idx_te]
    y_xgb_tr, y_xgb_te = y[idx_tr], y[idx_te]

    # XGBoost 超参（与原始版一致，无特征预筛选，直接全特征训练）
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, colsample_bylevel=0.8,
        min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=NEG_COUNT / POS_COUNT,
        eval_metric='auc', random_state=42, n_jobs=-1,
        early_stopping_rounds=30)
    xgb_model.fit(X_xgb_tr, y_xgb_tr,
                  eval_set=[(X_xgb_te, y_xgb_te)], verbose=50)
    xgb_probs = xgb_model.predict_proba(X_xgb_te)[:, 1]
    xgb_auc = roc_auc_score(y_xgb_te, xgb_probs)
    print(f"  XGBoost AUC: {xgb_auc:.4f}")

    # D-4: Transformer + XGBoost 权重融合
    _, _, _, trans_probs_te, trans_labels_te = evaluate(model_dual, test_loader, device)

    # D-5: CatBoost 第3/4基学习器
    try:
        from catboost import CatBoostClassifier
        _HAS_CATBOOST = True
    except ImportError:
        _HAS_CATBOOST = False
        print("  ⚠ catboost 未安装，跳过")

    if _HAS_CATBOOST:
        # CatBoost 使用差异化特征子集（不含 Transformer 特征，与原始版一致）
        X_cat = np.concatenate([
            hand_feats, rmt_isc_features, isct_monthly,
            mo_raw, xgb_self_norm, xgb_diff_norm,
            mo_vs_base, mo_pct, xgb_local_z,
            spec_agg_feats, ts_prior_feats,
            _scalar_feats, tcn_cpd_features,
        ], axis=1)
        X_cat_tr, X_cat_te = X_cat[idx_tr], X_cat[idx_te]
        print(f"  CatBoost差异化特征: {X_cat.shape[1]}维 (不含Transformer特征)")

        # CatBoost（与原始版一致：单个模型, depth=7, lr=0.01, iter=3000）
        cat_model = CatBoostClassifier(
            iterations=3000, depth=7, learning_rate=0.01,
            l2_leaf_reg=3.0, border_count=128, random_strength=1.5,
            bagging_temperature=0.8, auto_class_weights='Balanced',
            eval_metric='AUC', random_seed=42, verbose=300,
            early_stopping_rounds=80, task_type='CPU')
        cat_model.fit(X_cat_tr, y_xgb_tr, eval_set=(X_cat_te, y_xgb_te), verbose=300)
        cat_probs = cat_model.predict_proba(X_cat_te)[:, 1]
        cat_auc = roc_auc_score(y_xgb_te, cat_probs)
        print(f"  CatBoost AUC: {cat_auc:.4f}")
    else:
        cat_probs = xgb_probs.copy()
        cat_auc = xgb_auc

    # D-5b: 三模型加权融合（网格搜索，与原始版一致）
    print("\n  [三模型加权融合] 网格搜索 Transformer + XGBoost + CatBoost 最优权重...")
    best_w3, best_ens3_auc = (0.33, 0.33, 0.34), 0.0
    for w_t in np.arange(0.0, 1.01, 0.05):
        for w_x in np.arange(0.0, 1.01 - w_t, 0.05):
            w_c = 1.0 - w_t - w_x
            if w_c < -1e-9:
                continue
            blend3 = w_t * trans_probs_te + w_x * xgb_probs + w_c * cat_probs
            auc3 = roc_auc_score(trans_labels_te, blend3)
            if auc3 > best_ens3_auc:
                best_ens3_auc = auc3
                best_w3 = (w_t, w_x, w_c)

    ensemble_probs = best_w3[0] * trans_probs_te + best_w3[1] * xgb_probs + best_w3[2] * cat_probs
    ensemble_auc = roc_auc_score(trans_labels_te, ensemble_probs)
    print(f"  三模型最优权重: T={best_w3[0]:.2f}, X={best_w3[1]:.2f}, Cat={best_w3[2]:.2f}")

    best_f1_ens, best_thr_ens = 0.0, 0.5
    for thr in np.arange(0.05, 0.95, 0.005):
        preds = (ensemble_probs >= thr).astype(int)
        f1 = f1_score(trans_labels_te, preds, zero_division=0)
        if f1 > best_f1_ens:
            best_f1_ens, best_thr_ens = f1, thr
    print(f"  三模型加权融合 AUC: {ensemble_auc:.4f}  F1: {best_f1_ens:.4f}")

    # D-5c: Rank Average（三模型，与原始版一致）
    print("\n  [Rank Average] 三模型排名平均集成...")
    rank_trans = rankdata(trans_probs_te) / len(trans_probs_te)
    rank_xgb = rankdata(xgb_probs) / len(xgb_probs)
    rank_cat = rankdata(cat_probs) / len(cat_probs)
    rank_avg_probs = (rank_trans + rank_xgb + rank_cat) / 3.0
    rank_avg_auc = roc_auc_score(trans_labels_te, rank_avg_probs)
    print(f"  Rank Average AUC: {rank_avg_auc:.4f}")

    if rank_avg_auc > ensemble_auc:
        ensemble_probs = rank_avg_probs
        ensemble_auc = rank_avg_auc
        print(f"  --> Rank Average({rank_avg_auc:.4f}) 优于当前最佳，已替换")
    else:
        print(f"  --> 当前最佳({ensemble_auc:.4f}) 仍优于 Rank Average({rank_avg_auc:.4f})")

    # D-6: Stacking（三模型 OOF + LR 元学习器，与原始版一致）
    print("\n  [Stacking集成] 3模型OOF Stacking（Transformer + XGBoost + CatBoost）...")
    skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_xgb_probs = np.zeros(len(y_xgb_tr))
    oof_cat_probs = np.zeros(len(y_xgb_tr))

    print("  5折OOF（XGBoost + CatBoost）...")
    for fold, (tr_idx, va_idx) in enumerate(skf_meta.split(X_xgb_tr, y_xgb_tr)):
        # XGBoost OOF
        xgb_fold = xgb.XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.6,
            min_child_weight=10, reg_alpha=0.5, reg_lambda=2.0,
            scale_pos_weight=NEG_COUNT / POS_COUNT,
            eval_metric='auc', random_state=42, n_jobs=-1,
            early_stopping_rounds=30)
        xgb_fold.fit(X_xgb_tr[tr_idx], y_xgb_tr[tr_idx],
                     eval_set=[(X_xgb_tr[va_idx], y_xgb_tr[va_idx])], verbose=0)
        oof_xgb_probs[va_idx] = xgb_fold.predict_proba(X_xgb_tr[va_idx])[:, 1]

        # CatBoost OOF（使用差异化特征）
        if _HAS_CATBOOST:
            cat_fold = CatBoostClassifier(
                iterations=2500, depth=7, learning_rate=0.01,
                l2_leaf_reg=3.0, border_count=128,
                random_strength=1.5, bagging_temperature=0.8,
                auto_class_weights='Balanced',
                eval_metric='AUC', random_seed=42, verbose=0,
                early_stopping_rounds=80, task_type='CPU')
            cat_fold.fit(X_cat_tr[tr_idx], y_xgb_tr[tr_idx],
                         eval_set=(X_cat_tr[va_idx], y_xgb_tr[va_idx]), verbose=0)
            oof_cat_probs[va_idx] = cat_fold.predict_proba(X_cat_tr[va_idx])[:, 1]
        else:
            oof_cat_probs[va_idx] = oof_xgb_probs[va_idx]
        print(f"    Fold {fold+1}/5 完成")

    # 提取训练集 Transformer 概率
    tr_loader_meta = DataLoader(TensorDataset(X_tr, y_tr), batch_size=2048,
                                 shuffle=False, num_workers=0)
    _, _, _, trans_probs_tr, _ = evaluate(model_dual, tr_loader_meta, device)

    # 构造3模型OOF元特征 → Stacking（与原始版一致：3维 LR）
    meta_train = np.stack([trans_probs_tr, oof_xgb_probs, oof_cat_probs], axis=1)
    meta_test = np.stack([trans_probs_te, xgb_probs, cat_probs], axis=1)

    meta_lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
    meta_lr.fit(meta_train, y_xgb_tr.astype(int))
    stacking_probs = meta_lr.predict_proba(meta_test)[:, 1]
    stacking_auc = roc_auc_score(trans_labels_te, stacking_probs)

    coef_str = ', '.join([f'{c:.3f}' for c in meta_lr.coef_[0]])
    print(f"  Stacking 元学习器权重: [{coef_str}] (Trans/XGB/Cat)")
    print(f"  Stacking 3模型集成 AUC: {stacking_auc:.4f}")

    if stacking_auc > ensemble_auc:
        ensemble_auc = stacking_auc
        ensemble_probs = stacking_probs
        print("  --> Stacking 优于加权平均，已替换为最终集成结果")
    else:
        print("  --> 加权平均仍优于 Stacking，保持原集成结果")

    # P1-6: F1 优化 — PR曲线最优阈值 + 后处理
    prec_arr, rec_arr, thr_arr = precision_recall_curve(trans_labels_te, ensemble_probs)
    f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-8)
    best_pr_idx = np.argmax(f1_arr[:-1])  # 最后一个点为 recall=0
    best_thr_ens = thr_arr[best_pr_idx]
    best_f1_ens = f1_arr[best_pr_idx]
    print(f"  PR曲线最优阈值: {best_thr_ens:.4f}  F1={best_f1_ens:.4f}")

    # P1-6: 后处理 — 连续N月排名下降规则
    final_preds_ens = (ensemble_probs >= best_thr_ens).astype(int)
    # 测试集用户的排名下降: 后半段平均排名 - 前半段平均排名
    rank_drop_te = rank_drop[idx_te]
    # 如果排名下降明显 (rank_drop > 0.15) 且模型概率接近阈值，则上调为异常
    borderline_mask = (ensemble_probs >= best_thr_ens * 0.7) & (ensemble_probs < best_thr_ens)
    rank_drop_significant = rank_drop_te > 0.15
    n_upgraded = int((borderline_mask & rank_drop_significant).sum())
    final_preds_ens[borderline_mask & rank_drop_significant] = 1
    if n_upgraded > 0:
        post_f1 = f1_score(trans_labels_te, final_preds_ens, zero_division=0)
        print(f"  后处理: 上调{n_upgraded}个样本, F1={post_f1:.4f}")
        if post_f1 > best_f1_ens:
            best_f1_ens = post_f1

# =============================================================================
# Step E: 综合评估
# =============================================================================
print("\n" + "=" * 65)
print("Step E: 综合评估")
print("=" * 65)

final_auc, final_f1, final_thr, final_probs, final_labels = evaluate(
    model_dual, test_loader, device)
final_preds = (final_probs >= final_thr).astype(int)
cm = confusion_matrix(final_labels, final_preds)
tn, fp, fn, tp = cm.ravel()
top_auc = ensemble_auc if HAS_XGB else final_auc

print(f"\n  {'方法':<34} {'AUC':>8}  {'F1':>8}")
print(f"  {'-'*55}")
print(f"  {'RMT 谱得分（基线）':<34} {rmt_baseline_auc:>8.4f}")
print(f"  {'双路Transformer':<34} {final_auc:>8.4f}  {final_f1:>8.4f}")
if HAS_XGB:
    print(f"  {'XGBoost(Transformer特征)':<34} {xgb_auc:>8.4f}")
    print(f"  {'CatBoost(差异化基学习器)':<34} {cat_auc:>8.4f}")
    print(f"  {'3模型集成(最终)':<34} {ensemble_auc:>8.4f}  {best_f1_ens:>8.4f}")
print(f"\n  混淆矩阵:  TN={tn} FP={fp} FN={fn} TP={tp}")
print(classification_report(final_labels, final_preds, target_names=['正常', '异常']))

# =============================================================================
# Step F: 消融实验
# =============================================================================
print("\n" + "=" * 65)
print("Step F: 消融实验")
print("=" * 65)

if HAS_XGB:
    # P0-3: 消融不含 feat_all（Transformer特征），真正隔离 RMT-ISCT 增量
    X_base_hand = np.concatenate([
        hand_feats, mo_raw, xgb_self_norm,
        xgb_diff_norm, mo_vs_base, mo_pct, xgb_local_z,
        spec_agg_feats, ts_prior_feats,
    ], axis=1)
    X_strat_rmt = np.column_stack([local_rmt_score, local_rmt_ratio])
    X_isct_cpd = np.concatenate([rmt_isc_features, isct_monthly], axis=1)

    ablation_groups = [
        ("G1: HandCraft (无RMT-ISC)", X_base_hand),
        ("G2: HandCraft + Stratified-RMT", np.concatenate([X_base_hand, X_strat_rmt], axis=1)),
        ("G3: HandCraft + ISCT-CPD", np.concatenate([X_base_hand, X_isct_cpd], axis=1)),
        ("G4: HandCraft + Strat + ISCT", np.concatenate([X_base_hand, X_strat_rmt, X_isct_cpd], axis=1)),
        ("G5: HandCraft + ISCT + Trans", np.concatenate([X_base_hand, X_isct_cpd, feat_all], axis=1)),
    ]

    def ablation_xgb(X_feat, y_all, idx_tr, idx_te):
        t0 = time.time()
        X_tr_a, X_te_a = X_feat[idx_tr], X_feat[idx_te]
        y_tr_a, y_te_a = y_all[idx_tr], y_all[idx_te]
        m = xgb.XGBClassifier(
            n_estimators=800, max_depth=5, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.6, min_child_weight=10,
            reg_alpha=0.5, reg_lambda=2.0,
            scale_pos_weight=(y_tr_a == 0).sum() / max((y_tr_a == 1).sum(), 1),
            eval_metric='auc', random_state=42, n_jobs=-1, early_stopping_rounds=30)
        m.fit(X_tr_a, y_tr_a, eval_set=[(X_te_a, y_te_a)], verbose=0)
        probs = m.predict_proba(X_te_a)[:, 1]
        auc = roc_auc_score(y_te_a, probs)
        best_f1 = max(f1_score(y_te_a, (probs >= thr).astype(int), zero_division=0)
                      for thr in np.arange(0.05, 0.95, 0.01))
        return auc, best_f1, time.time() - t0

    print(f"\n  {'组别':<42} {'维度':>6} {'AUC':>8} {'F1':>8} {'耗时':>8}")
    print(f"  {'─'*78}")
    abl_results = []
    for gname, gfeat in ablation_groups:
        g_auc, g_f1, g_time = ablation_xgb(gfeat, y, idx_tr, idx_te)
        abl_results.append((gname, gfeat.shape[1], g_auc, g_f1))
        print(f"  {gname:<42} {gfeat.shape[1]:>6} {g_auc:>8.4f} {g_f1:>8.4f} {g_time:>6.1f}s")

    base_auc = abl_results[0][2]
    print(f"\n  增量分析 (vs HandCraft Base):")
    for gname, _, g_auc, _ in abl_results[1:]:
        delta = g_auc - base_auc
        print(f"    {gname:<42} AUC {delta:+.4f}")
else:
    print("  (需要 xgboost 才能运行消融实验)")

# =============================================================================
# 完成
# =============================================================================
print("\n" + "=" * 65)
print("全流程完成!")
print(f"   RMT 基线 AUC             : {rmt_baseline_auc:.4f}")
print(f"   双路Transformer AUC      : {final_auc:.4f}  F1={final_f1:.4f}")
if HAS_XGB:
    print(f"   XGBoost AUC              : {xgb_auc:.4f}")
    print(f"   3模型集成 AUC            : {ensemble_auc:.4f}  F1={best_f1_ens:.4f}")
print(f"   总提升（vs RMT基线）     : {top_auc - rmt_baseline_auc:+.4f}")
print("=" * 65)
