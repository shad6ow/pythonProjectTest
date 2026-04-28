
# =============================================================================
# SGCC Phase 3: Transformer 改良版 + 完整消融/对比实验
# 消融: G1=Baseline vs G2=+RMT vs G3=+RMT+Trans_PCA16
# 对比: Transformer单独 vs XGB vs LGB vs CatBoost vs 集成
# 改良: Transformer 256维 → PCA 16维 → GBDT
# =============================================================================

import os, time, warnings, gc
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = str(min(os.cpu_count() or 4, 8))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from scipy.stats import rankdata, skew, kurtosis
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.metrics import precision_recall_curve, f1_score
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb

torch.set_num_threads(min(os.cpu_count() or 4, 8))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(42)
torch.manual_seed(42)

DATA_PATH = r'C:\Users\wb.zhoushujie\Desktop\data set.csv'

print("=" * 70)
print("Phase 3: Transformer 改良版 + 完整消融/对比实验")
print(f"  设备={device}")
print("=" * 70)
t_total = time.time()

# =============================================================================
# Step 1: 加载数据（float32 节省内存）
# =============================================================================
print("\n--- Step 1: 加载数据 ---")
df = pd.read_csv(DATA_PATH, dtype={c: 'float32' for c in
                  pd.read_csv(DATA_PATH, nrows=0).columns
                  if c not in ['CONS_NO', 'FLAG']})
labels = df['FLAG'].values.astype(np.int64)
date_cols = [c for c in df.columns if c not in ['CONS_NO', 'FLAG']]
raw_vals = df[date_cols].values.astype(np.float32)
del df; gc.collect()

N_USERS, T_DAYS = raw_vals.shape
print(f"  用户数={N_USERS}, 天数={T_DAYS}")

dates = pd.to_datetime(date_cols, format='%m/%d/%Y')
day_of_week = dates.dayofweek.values
month_of_year = dates.month.values
is_weekend = (day_of_week >= 5).astype(np.float32)
weekday_mask = (day_of_week < 5)
weekend_mask = (day_of_week >= 5)

# =============================================================================
# Step 2: 缺失模式特征（插值前提取）
# =============================================================================
print("\n--- Step 2: 缺失模式特征 ---")
nan_mask = np.isnan(raw_vals)
miss_ratio = nan_mask.mean(axis=1).astype(np.float32)
half_t = T_DAYS // 2
miss_first_half = nan_mask[:, :half_t].mean(axis=1).astype(np.float32)
miss_second_half = nan_mask[:, half_t:].mean(axis=1).astype(np.float32)
miss_half_diff = miss_second_half - miss_first_half
max_consec_nan = np.zeros(N_USERS, dtype=np.float32)
cur_nan_run = np.zeros(N_USERS, dtype=np.float32)
for t in range(T_DAYS):
    cur_nan_run = (cur_nan_run + 1.0) * nan_mask[:, t]
    max_consec_nan = np.maximum(max_consec_nan, cur_nan_run)
max_consec_nan_ratio = max_consec_nan / T_DAYS
nan_int = nan_mask.astype(np.int32)
nan_diff_arr = np.diff(nan_int, axis=1, prepend=0)
miss_segment_count = (nan_diff_arr == 1).sum(axis=1).astype(np.float32)
zero_mask = (raw_vals == 0) & (~nan_mask)
zero_ratio = zero_mask.sum(axis=1).astype(np.float32) / (~nan_mask).sum(axis=1).clip(1).astype(np.float32)
max_consec_zero = np.zeros(N_USERS, dtype=np.float32)
cur_zero_run = np.zeros(N_USERS, dtype=np.float32)
for t in range(T_DAYS):
    cur_zero_run = (cur_zero_run + 1.0) * zero_mask[:, t]
    max_consec_zero = np.maximum(max_consec_zero, cur_zero_run)
no_signal_ratio = (nan_mask | zero_mask).mean(axis=1).astype(np.float32)
miss_features = np.column_stack([
    miss_ratio, max_consec_nan_ratio,
    miss_first_half, miss_second_half, miss_half_diff,
    miss_segment_count, zero_ratio, max_consec_zero / T_DAYS, no_signal_ratio,
]).astype(np.float32)
print(f"  缺失模式特征: {miss_features.shape[1]}维")

# =============================================================================
# Step 3: 插值填充（分块 float32）
# =============================================================================
print("\n--- Step 3: 数据预处理 ---")
CHUNK = 5000
col_mean = np.nan_to_num(np.nanmean(raw_vals, axis=0), nan=0.0)
for i in range(0, N_USERS, CHUNK):
    chunk = raw_vals[i:i+CHUNK]
    df_chunk = pd.DataFrame(chunk)
    df_chunk = df_chunk.interpolate(method='linear', axis=1, limit_direction='both')
    df_chunk = df_chunk.fillna(pd.Series(col_mean))
    raw_vals[i:i+CHUNK] = df_chunk.values.astype(np.float32)
    del df_chunk; gc.collect()
X_raw = raw_vals
del raw_vals; gc.collect()
print(f"  填充后范围: [{X_raw.min():.2f}, {X_raw.max():.2f}]")

# =============================================================================
# Step 4: 月度聚合
# =============================================================================
print("\n--- Step 4: 月度聚合 ---")
DAYSPM = 30
NM = T_DAYS // DAYSPM  # 34

mo_mean = np.zeros((N_USERS, NM), dtype=np.float32)
mo_std  = np.zeros((N_USERS, NM), dtype=np.float32)
mo_max  = np.zeros((N_USERS, NM), dtype=np.float32)
mo_zero = np.zeros((N_USERS, NM), dtype=np.float32)
mo_nan  = np.zeros((N_USERS, NM), dtype=np.float32)
for m in range(NM):
    s, e = m * DAYSPM, min((m+1)*DAYSPM, T_DAYS)
    mo_mean[:, m] = X_raw[:, s:e].mean(axis=1)
    mo_std[:, m]  = X_raw[:, s:e].std(axis=1)
    mo_max[:, m]  = X_raw[:, s:e].max(axis=1)
    mo_zero[:, m] = (X_raw[:, s:e] == 0).mean(axis=1)
    mo_nan[:, m]  = nan_mask[:, s:e].mean(axis=1)

half_nm = NM // 2
BASELINE_M = 6
baseline_mean = mo_mean[:, :BASELINE_M].mean(axis=1, keepdims=True) + 1e-3
mo_vs_base = (mo_mean - baseline_mean) / (np.abs(baseline_mean) + 1e-3)
mo_cumdev = np.cumsum(mo_mean - baseline_mean, axis=1)
mo_pct = np.zeros((N_USERS, NM), dtype=np.float32)
for m in range(NM):
    mo_pct[:, m] = (rankdata(mo_mean[:, m]) / N_USERS).astype(np.float32)
mo_rank_mean = mo_pct.mean(axis=1, keepdims=True)
mo_rank_dev = mo_pct - mo_rank_mean
rank_drop = mo_pct[:, :half_nm].mean(axis=1) - mo_pct[:, half_nm:].mean(axis=1)
user_median_raw = np.median(mo_mean, axis=1, keepdims=True) + 1e-3
mo_self_ratio = mo_mean / user_median_raw
mo_log_ratio = np.log1p(np.maximum(mo_mean, 0)) - np.log1p(np.maximum(baseline_mean, 0))
mo_roll3_mean = np.zeros((N_USERS, NM), dtype=np.float32)
mo_roll3_std  = np.zeros((N_USERS, NM), dtype=np.float32)
for m in range(NM):
    ws = max(0, m-2)
    mo_roll3_mean[:, m] = mo_mean[:, ws:m+1].mean(axis=1)
    mo_roll3_std[:, m]  = mo_mean[:, ws:m+1].std(axis=1) + 1e-6
mo_local_zscore = np.clip((mo_mean - mo_roll3_mean) / mo_roll3_std, -5, 5)
mo_global_median = np.median(mo_mean, axis=0, keepdims=True) + 1e-3
mo_global_dev = np.log1p(np.maximum(mo_mean, 0)) - np.log1p(np.maximum(mo_global_median, 0))
mo_diff1 = np.diff(mo_mean, axis=1, prepend=mo_mean[:, :1])
mo_diff2 = np.diff(mo_diff1, axis=1, prepend=mo_diff1[:, :1])
mo_cv = mo_std / (mo_mean + 1e-6)
print(f"  月度聚合: {NM}个月")

# =============================================================================
# Step 5: 日历特征
# =============================================================================
print("\n--- Step 5: 日历特征 ---")
weekday_mean_arr = np.nan_to_num(np.nanmean(np.where(weekday_mask[None,:], X_raw, np.nan), axis=1), 0).astype(np.float32)
weekend_mean_arr = np.nan_to_num(np.nanmean(np.where(weekend_mask[None,:], X_raw, np.nan), axis=1), 0).astype(np.float32)
wd_std_arr = np.nan_to_num(np.nanstd(np.where(weekday_mask[None,:], X_raw, np.nan), axis=1), 0).astype(np.float32)
we_std_arr = np.nan_to_num(np.nanstd(np.where(weekend_mask[None,:], X_raw, np.nan), axis=1), 0).astype(np.float32)
monthly_avg_by_cal = np.zeros((N_USERS, 12), dtype=np.float32)
for m_cal in range(12):
    m_mask = (month_of_year == m_cal+1)
    if m_mask.sum() > 0:
        monthly_avg_by_cal[:, m_cal] = np.nan_to_num(
            np.nanmean(np.where(m_mask[None,:], X_raw, np.nan), axis=1), 0).astype(np.float32)
summer_avg = monthly_avg_by_cal[:, [5,6,7]].mean(axis=1)
winter_avg = monthly_avg_by_cal[:, [11,0,1]].mean(axis=1)
summer_winter_ratio = summer_avg / (winter_avg + 1e-6)
yearly_avg = monthly_avg_by_cal.mean(axis=1, keepdims=True) + 1e-6
seasonality_strength = (monthly_avg_by_cal / yearly_avg).std(axis=1)
mo_wd_mean = np.zeros((N_USERS, NM), dtype=np.float32)
mo_we_mean = np.zeros((N_USERS, NM), dtype=np.float32)
for m in range(NM):
    s, e = m*DAYSPM, min((m+1)*DAYSPM, T_DAYS)
    wd_m = weekday_mask[s:e]; we_m = weekend_mask[s:e]
    if wd_m.sum() > 0: mo_wd_mean[:, m] = X_raw[:, s:e][:, wd_m].mean(axis=1)
    if we_m.sum() > 0: mo_we_mean[:, m] = X_raw[:, s:e][:, we_m].mean(axis=1)
mo_wd_we_ratio = mo_wd_mean / (mo_we_mean + 1e-6)
wd_we_ratio = weekday_mean_arr / (weekend_mean_arr + 1e-6)
calendar_features = np.column_stack([
    wd_we_ratio, weekday_mean_arr, weekend_mean_arr, wd_std_arr, we_std_arr,
    summer_winter_ratio, seasonality_strength,
]).astype(np.float32)
print(f"  日历特征: {calendar_features.shape[1]}维")

# =============================================================================
# Step 6: 深层统计特征
# =============================================================================
print("\n--- Step 6: 深层统计特征 ---")
user_mean_g = X_raw.mean(axis=1); user_std_g = X_raw.std(axis=1)
user_median_g = np.median(X_raw, axis=1); user_max_g = X_raw.max(axis=1)
user_q10 = np.percentile(X_raw, 10, axis=1); user_q25 = np.percentile(X_raw, 25, axis=1)
user_q75 = np.percentile(X_raw, 75, axis=1); user_q90 = np.percentile(X_raw, 90, axis=1)
user_iqr = user_q75 - user_q25
user_cv_g = user_std_g / (user_mean_g + 1e-6)
user_skew_g = np.nan_to_num(skew(X_raw, axis=1, nan_policy='omit').astype(np.float32), 0)
user_kurt_g = np.nan_to_num(kurtosis(X_raw, axis=1, nan_policy='omit').astype(np.float32), 0)
extreme_high = (X_raw > (user_q75 + 3*user_iqr)[:, None]).mean(axis=1).astype(np.float32)
extreme_low  = (X_raw < np.maximum(user_q25 - 3*user_iqr, 0)[:, None]).mean(axis=1).astype(np.float32)
t_vec = np.arange(T_DAYS, dtype=np.float64)
t_mean_v = t_vec.mean(); t_var = ((t_vec - t_mean_v)**2).sum()
X_f64 = X_raw.astype(np.float64)
slope_all = ((X_f64 - X_f64.mean(axis=1, keepdims=True)) *
             (t_vec[None,:] - t_mean_v)).sum(axis=1) / (t_var + 1e-8)
del X_f64; gc.collect()
tail_head_ratio = (mo_mean[:, -6:].mean(axis=1) / (mo_mean[:, :6].mean(axis=1) + 1e-6)).astype(np.float32)
mo_ratio_tmp = mo_mean[:, 1:] / (mo_mean[:, :-1] + 1e-6)
drop_mask_tmp = (mo_ratio_tmp < 0.7).astype(np.int32)
max_consec_drop = np.zeros(N_USERS, dtype=np.float32)
cur_drop_run = np.zeros(N_USERS, dtype=np.float32)
for t in range(drop_mask_tmp.shape[1]):
    cur_drop_run = (cur_drop_run+1.0)*drop_mask_tmp[:, t]
    max_consec_drop = np.maximum(max_consec_drop, cur_drop_run)
mo_q25_arr = np.zeros((N_USERS, NM), dtype=np.float32)
mo_q75_arr = np.zeros((N_USERS, NM), dtype=np.float32)
for m in range(NM):
    s, e = m*DAYSPM, min((m+1)*DAYSPM, T_DAYS)
    mo_q25_arr[:, m] = np.percentile(X_raw[:, s:e], 25, axis=1)
    mo_q75_arr[:, m] = np.percentile(X_raw[:, s:e], 75, axis=1)
q25_trend = mo_q25_arr[:, half_nm:].mean(axis=1) - mo_q25_arr[:, :half_nm].mean(axis=1)
q75_trend = mo_q75_arr[:, half_nm:].mean(axis=1) - mo_q75_arr[:, :half_nm].mean(axis=1)
std_trend = mo_std[:, half_nm:].mean(axis=1) / (mo_std[:, :half_nm].mean(axis=1) + 1e-6)
cv_trend  = mo_cv[:, half_nm:].mean(axis=1) - mo_cv[:, :half_nm].mean(axis=1)
deep_stat_features = np.column_stack([
    user_mean_g, user_std_g, user_median_g, user_max_g,
    user_q10, user_q25, user_q75, user_q90, user_iqr, user_cv_g,
    user_skew_g, user_kurt_g, extreme_high, extreme_low,
    slope_all.astype(np.float32), tail_head_ratio, max_consec_drop,
    q25_trend, q75_trend, std_trend, cv_trend,
]).astype(np.float32)
print(f"  深层统计特征: {deep_stat_features.shape[1]}维")

# =============================================================================
# Step 7: TCN-CPD 多尺度变点
# =============================================================================
print("\n--- Step 7: TCN-CPD ---")
_tcn_input = X_raw.copy().astype(np.float32)
_u_med = np.median(_tcn_input, axis=1, keepdims=True)
_u_iqr = np.clip(np.percentile(_tcn_input, 75, axis=1, keepdims=True) -
                  np.percentile(_tcn_input, 25, axis=1, keepdims=True), 1e-6, None)
_tcn_input = np.clip((_tcn_input - _u_med) / _u_iqr, -10, 10)
_T_tcn = min(_tcn_input.shape[1], 1020)
_tcn_input = _tcn_input[:, :_T_tcn]
ms_cpd_features = []
for _scale in [7, 30, 90]:
    _kernel = np.ones(_scale) / _scale
    _smoothed = np.apply_along_axis(lambda r: np.convolve(r, _kernel, 'same'), 1, _tcn_input)
    _abs_residual = np.abs(_tcn_input - _smoothed)
    _n_seg = max(_T_tcn // _scale, 4); _seg_len = _T_tcn // _n_seg
    _seg_e = np.zeros((N_USERS, _n_seg), dtype=np.float32)
    for _si in range(_n_seg):
        _s2, _e2 = _si*_seg_len, min((_si+1)*_seg_len, _T_tcn)
        _seg_e[:, _si] = _abs_residual[:, _s2:_e2].mean(axis=1)
    _sd = np.abs(np.diff(_seg_e, axis=1))
    _mjp = np.argmax(_sd, axis=1).astype(np.float32) / max(_n_seg-2, 1)
    _mjm = np.max(_sd, axis=1) / (_seg_e.mean(axis=1) + 1e-8)
    _ncp = (_sd > 2*(_sd.mean(axis=1, keepdims=True)+1e-8)).sum(axis=1).astype(np.float32)
    _hs = _n_seg//2
    _hr = (_seg_e[:, _hs:].mean(axis=1)+1e-8) / (_seg_e[:, :_hs].mean(axis=1)+1e-8)
    _es = _seg_e.std(axis=1)
    _xm = np.arange(_n_seg, dtype=np.float32); _xmm = _xm.mean()
    _em = _seg_e.mean(axis=1, keepdims=True)
    _sl = ((_xm[None,:]-_xmm)*(_seg_e-_em)).sum(axis=1) / (((_xm-_xmm)**2).sum()+1e-8)
    ms_cpd_features.append(np.column_stack([_mjp, _mjm, _ncp, _hr, _es, _sl]).astype(np.float32))
tcn_cpd_features = np.concatenate(ms_cpd_features, axis=1)
_cf = np.column_stack([ms_cpd_features[0][:,4]/(ms_cpd_features[2][:,4]+1e-8),
                        np.abs(ms_cpd_features[0][:,0]-ms_cpd_features[2][:,0])]).astype(np.float32)
tcn_cpd_features = np.concatenate([tcn_cpd_features, _cf], axis=1)
del _tcn_input; gc.collect()
print(f"  TCN-CPD: {tcn_cpd_features.shape[1]}维")

# =============================================================================
# Step 8: ISCT 变点特征
# =============================================================================
print("\n--- Step 8: ISCT 变点特征 ---")
user_avg_g = X_raw.mean(axis=1)
n_strata = 8
strata_labels = pd.qcut(user_avg_g, q=n_strata, labels=False, duplicates='drop')
isct_dev_monthly = np.zeros((N_USERS, NM), dtype=np.float32)
for k_s in range(int(strata_labels.max())+1):
    mask_s = (strata_labels == k_s)
    if mask_s.sum() < 2: continue
    layer_median = np.median(mo_mean[mask_s], axis=0, keepdims=True)
    isct_dev_monthly[mask_s] = ((mo_mean[mask_s] - layer_median) /
                                 (np.abs(layer_median) + 1e-3)).astype(np.float32)
sig_all = isct_dev_monthly; T_sig = sig_all.shape[1]
isct_cpd_feats = np.zeros((N_USERS, 8), dtype=np.float32)
sig_mean_s = sig_all.mean(axis=1, keepdims=True)
cumsum_all = np.cumsum(sig_all - sig_mean_s, axis=1)
cp_pos_idx = np.argmax(np.abs(cumsum_all), axis=1)
isct_cpd_feats[:, 0] = cp_pos_idx.astype(np.float32) / T_sig
cp_clamped = np.clip(cp_pos_idx, 1, T_sig-1)
ar_t = np.arange(T_sig)[None,:]; cp_exp = cp_clamped[:, None]
pre_m = (sig_all*(ar_t<cp_exp)).sum(axis=1)/((ar_t<cp_exp).sum(axis=1).astype(np.float32)+1e-8)
post_m = (sig_all*(ar_t>=cp_exp)).sum(axis=1)/((ar_t>=cp_exp).sum(axis=1).astype(np.float32)+1e-8)
isct_cpd_feats[:, 1] = (post_m / (np.abs(pre_m)+1e-6)).astype(np.float32)
isct_cpd_feats[:, 2] = np.abs(sig_all).max(axis=1)
hs2 = T_sig//2
isct_cpd_feats[:, 3] = (sig_all[:, hs2:].mean(axis=1) / (np.abs(sig_all[:, :hs2].mean(axis=1))+1e-6))
isct_cpd_feats[:, 4] = sig_all.std(axis=1)
neg_mask2 = (sig_all < 0).astype(np.int32)
mx_neg = np.zeros(N_USERS, dtype=np.float32); cr2 = np.zeros(N_USERS, dtype=np.float32)
for t_s in range(T_sig):
    cr2 = (cr2+1.0)*neg_mask2[:, t_s]; mx_neg = np.maximum(mx_neg, cr2)
isct_cpd_feats[:, 5] = mx_neg / T_sig
t_vs = np.arange(T_sig, dtype=np.float64); tmv = t_vs.mean(); tvv = ((t_vs-tmv)**2).sum()
sf64 = sig_all.astype(np.float64)
isct_cpd_feats[:, 6] = ((sf64-sf64.mean(axis=1,keepdims=True))*(t_vs[None,:]-tmv)).sum(axis=1)/(tvv+1e-8)
isct_cpd_feats[:, 7] = sig_all[:, -6:].mean(axis=1)
isc_keep = [ci for ci in range(8) if abs(roc_auc_score(labels, isct_cpd_feats[:, ci])-0.5) > 0.03]
isct_cpd_feats = isct_cpd_feats[:, isc_keep] if isc_keep else isct_cpd_feats
print(f"  ISCT-CPD: {isct_cpd_feats.shape[1]}维 (保留{len(isc_keep)}/8)")

# =============================================================================
# Step 9: RMT 2特征（Phase 2 已验证正贡献）
# =============================================================================
print("\n--- Step 9: RMT 2特征 ---")
t_rmt = time.time()

def _mp_upper_edge(n_samples, n_features, sigma2=1.0):
    ratio = n_features / max(n_samples, n_features + 1)
    return sigma2 * (1 + np.sqrt(ratio)) ** 2

# F1: B_snr_global
mo_norm_f64 = mo_mean.astype(np.float64)
mo_med_g = np.median(mo_norm_f64, axis=1, keepdims=True)
mo_iqr_g = np.clip(np.percentile(mo_norm_f64, 75, axis=1, keepdims=True) -
                    np.percentile(mo_norm_f64, 25, axis=1, keepdims=True), 1e-6, None)
mo_norm_g = np.clip((mo_norm_f64 - mo_med_g) / mo_iqr_g, -5, 5)
X_cent_g = mo_norm_g - mo_norm_g.mean(axis=0, keepdims=True)
cov_g = (X_cent_g.T @ X_cent_g) / (N_USERS - 1)
eigvals_g, eigvecs_g = np.linalg.eigh(cov_g)
eigvals_g = np.maximum(eigvals_g, 0)
sigma2_g = np.median(eigvals_g[eigvals_g > 0]) if (eigvals_g > 0).sum() > 0 else 1.0
mp_up_g = _mp_upper_edge(N_USERS, NM, sigma2=sigma2_g)
sig_mask_g = eigvals_g > mp_up_g
noise_mask_g = ~sig_mask_g
proj_sig = X_cent_g @ eigvecs_g[:, sig_mask_g] if sig_mask_g.sum() > 0 else np.zeros((N_USERS, 1))
proj_noise = X_cent_g @ eigvecs_g[:, noise_mask_g] if noise_mask_g.sum() > 0 else np.ones((N_USERS, 1))
energy_sig = (proj_sig ** 2).sum(axis=1)
energy_noise = (proj_noise ** 2).sum(axis=1) + 1e-8
B_snr_global = (energy_sig / energy_noise).astype(np.float32)

# F2: A_late_signal_ratio
n_strata_rmt = 8
strata_labels_rmt = pd.qcut(user_avg_g, q=n_strata_rmt, labels=False, duplicates='drop')
A_late_signal_ratio = np.zeros(N_USERS, dtype=np.float32)
for k_s in range(int(strata_labels_rmt.max()) + 1):
    mask_k = (strata_labels_rmt == k_s)
    n_layer = mask_k.sum()
    if n_layer < 20:
        continue
    layer_data = mo_norm_g[mask_k]
    layer_cent = layer_data - layer_data.mean(axis=0, keepdims=True)
    cov_layer = (layer_cent.T @ layer_cent) / (n_layer - 1)
    evals_l, evecs_l = np.linalg.eigh(cov_layer)
    evals_l = np.maximum(evals_l, 0)
    sigma2_l = np.median(evals_l[evals_l > 0]) if (evals_l > 0).sum() > 0 else 1.0
    mp_up_l = _mp_upper_edge(n_layer, NM, sigma2=sigma2_l)
    sig_mask_l = evals_l > mp_up_l
    if sig_mask_l.sum() == 0:
        continue
    V_sig = evecs_l[:, sig_mask_l]
    late_start = NM * 2 // 3
    late_data = layer_cent[:, late_start:]
    full_proj = (layer_cent @ V_sig) ** 2
    late_proj = np.zeros((n_layer, sig_mask_l.sum()))
    late_cent = np.zeros_like(layer_cent)
    late_cent[:, late_start:] = layer_cent[:, late_start:]
    late_proj = (late_cent @ V_sig) ** 2
    full_energy = full_proj.sum(axis=1) + 1e-8
    late_energy = late_proj.sum(axis=1)
    A_late_signal_ratio[mask_k] = (late_energy / full_energy).astype(np.float32)

rmt_features = np.column_stack([B_snr_global, A_late_signal_ratio]).astype(np.float32)
print(f"  RMT特征: {rmt_features.shape[1]}维  耗时={time.time()-t_rmt:.1f}s")
print(f"    B_snr_global AUC={max(roc_auc_score(labels, B_snr_global), 1-roc_auc_score(labels, B_snr_global)):.4f}")
print(f"    A_late_signal AUC={max(roc_auc_score(labels, A_late_signal_ratio), 1-roc_auc_score(labels, A_late_signal_ratio)):.4f}")

# =============================================================================
# Step 10: 组装 GBDT 特征矩阵
# =============================================================================
print("\n--- Step 10: 组装 GBDT 特征矩阵 ---")

# 月度序列展平 → GBDT特征
feat_blocks_baseline = [
    mo_mean, mo_std, mo_max, mo_zero, mo_nan,        # 5 × 34 = 170
    mo_vs_base, mo_cumdev, mo_pct, mo_rank_dev,       # 4 × 34 = 136
    mo_self_ratio, mo_log_ratio,                       # 2 × 34 = 68
    mo_roll3_mean, mo_roll3_std, mo_local_zscore,      # 3 × 34 = 102
    mo_global_dev, mo_diff1, mo_diff2, mo_cv,          # 4 × 34 = 136  → 小计612
    mo_wd_we_ratio,                                    # 34
    rank_drop[:, None],                                # 1
    miss_features,                                      # 9
    calendar_features,                                  # 7
    deep_stat_features,                                # 21
    tcn_cpd_features,                                  # 20
    isct_cpd_feats,                                    # ~6
]
X_baseline = np.concatenate(feat_blocks_baseline, axis=1).astype(np.float32)
print(f"  G1 Baseline 特征: {X_baseline.shape[1]}维")

X_g2 = np.concatenate([X_baseline, rmt_features], axis=1).astype(np.float32)
print(f"  G2 +RMT 特征: {X_g2.shape[1]}维")

# =============================================================================
# Step 11: 构建 Transformer 月度序列输入 + 模型定义
# =============================================================================
print("\n--- Step 11: 构建 Transformer 输入 + 模型 ---")

def _scale_ch(arr2d, clip=5.0):
    flat = arr2d.reshape(-1, 1)
    flat = RobustScaler().fit_transform(flat).reshape(arr2d.shape)
    return np.clip(flat, -clip, clip).astype(np.float32)

def _scale_ch_per_user(arr2d, clip=5.0):
    med = np.median(arr2d, axis=1, keepdims=True)
    q1 = np.percentile(arr2d, 25, axis=1, keepdims=True)
    q3 = np.percentile(arr2d, 75, axis=1, keepdims=True)
    iqr = q3 - q1 + 1e-6
    return np.clip((arr2d - med) / iqr, -clip, clip).astype(np.float32)

rank_tile = np.tile(rank_drop[:, np.newaxis], (1, NM))
isct_monthly = mo_mean.copy()  # 月均消费作为ISCT轨迹通道
mo_rank_dev_ch = mo_mean - mo_global_median  # 排名偏差通道

mo_ch = np.stack([
    _scale_ch_per_user(mo_mean),         # ch0  月均消费
    _scale_ch_per_user(mo_std),          # ch1  月波动
    _scale_ch_per_user(mo_max),          # ch2  月峰值
    mo_zero,                              # ch3  零值比
    mo_pct,                               # ch4  跨用户排名
    _scale_ch(mo_rank_dev_ch),           # ch5  排名偏差
    _scale_ch(rank_tile),                # ch6  排名下降幅度
    _scale_ch_per_user(mo_vs_base),      # ch7  基准偏离
    _scale_ch_per_user(mo_cumdev),       # ch8  累积下降
    _scale_ch(mo_log_ratio),             # ch9  对数偏离
    _scale_ch_per_user(mo_diff1),        # ch10 一阶差分
    _scale_ch(mo_diff2),                 # ch11 加速度
    mo_self_ratio.astype(np.float32),    # ch12 自身比值
    np.clip(mo_local_zscore, -5, 5).astype(np.float32),  # ch13 局部Z
    _scale_ch(mo_global_dev),            # ch14 全局偏离
    _scale_ch_per_user(isct_monthly),    # ch15 ISCT月度轨迹
    _scale_ch_per_user(isct_dev_monthly),# ch16 ISCT偏离轨迹
], axis=2)  # (N, NM, 17)

_scalar_feats = mo_ch.mean(axis=1).astype(np.float32)  # (N, 17)
_scalar_tiled = np.tile(_scalar_feats[:, np.newaxis, :], (1, NM, 1))
X_mo_seq = np.concatenate([mo_ch, _scalar_tiled], axis=2).astype(np.float32)
FEAT_DIM = X_mo_seq.shape[2]   # 34
print(f"  Transformer输入: {X_mo_seq.shape}  ({FEAT_DIM}维通道, {NM}个月)")

# ── DualPathTransformer 模型定义 ────────────────────────────────────

class LocalWindowAttentionBlock(nn.Module):
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


class DualPathTransformer(nn.Module):
    def __init__(self, feat_dim=34, d_model=128, nhead=4, num_layers=3,
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
        feat_cat = torch.cat([feat_local, feat_global], dim=1)  # 256维
        return self.classifier(feat_cat), feat_cat

    def extract_features(self, x):
        feat_local, feat_global = self._encode(x)
        return torch.cat([feat_local, feat_global], dim=1)  # 256维


class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha=0.92, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal = (1 - p_t) ** self.gamma * bce
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * focal).mean()

# =============================================================================
# Step 12: 训练 Transformer（全量训练，80/20 split）
# =============================================================================
print("\n--- Step 12: 训练 Transformer ---")

from sklearn.model_selection import train_test_split
idx_tr, idx_te = train_test_split(
    np.arange(N_USERS), test_size=0.2, random_state=42, stratify=labels)

X_tr = torch.FloatTensor(X_mo_seq[idx_tr])
y_tr = torch.FloatTensor(labels[idx_tr].astype(np.float32))
X_te = torch.FloatTensor(X_mo_seq[idx_te])
y_te = torch.FloatTensor(labels[idx_te].astype(np.float32))

class_counts = np.bincount(labels[idx_tr])
sample_w = (1.0 / class_counts)[labels[idx_tr]]
sampler = WeightedRandomSampler(torch.FloatTensor(sample_w), len(idx_tr), replacement=True)

train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=1024,
                           sampler=sampler, num_workers=0)
test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=2048,
                          shuffle=False, num_workers=0)

NEG_COUNT = int((labels == 0).sum())
POS_COUNT = int((labels == 1).sum())
AUTO_ALPHA = max((1.0 - POS_COUNT / (POS_COUNT + NEG_COUNT)) * 0.95, 0.80)
EPOCHS, PATIENCE = 60, 10

model = DualPathTransformer(feat_dim=FEAT_DIM, d_model=128, nhead=4,
                             num_layers=3, dim_ff=256, dropout=0.15,
                             win_size=4, max_len=40).to(device)
criterion = AdaptiveFocalLoss(alpha=AUTO_ALPHA, gamma=2.0)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=5e-7)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  参数量={total_params:,}  feat_dim={FEAT_DIM}  epochs={EPOCHS}")

best_auc_trans, best_state = 0.0, None
best_epoch, no_improve = 0, 0

print(f"{'Epoch':>5} | {'Loss':>10} | {'AUC':>7} | {'F1':>6} | {'LR':>8}")
print("-" * 50)

for epoch in range(1, EPOCHS + 1):
    # 训练
    model.train()
    total_loss, total = 0.0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, feats = model(X_batch)
        loss = criterion(logits.squeeze(1), y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        total += len(y_batch)
    tr_loss = total_loss / total

    # 验证
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits, _ = model(X_batch.to(device))
            all_probs.extend(torch.sigmoid(logits.squeeze(1)).cpu().numpy())
            all_labels.extend(y_batch.numpy())
    val_probs = np.array(all_probs)
    val_labels = np.array(all_labels)
    val_auc = roc_auc_score(val_labels, val_probs)
    best_f1_val = max(f1_score(val_labels, (val_probs >= thr).astype(int), zero_division=0)
                      for thr in np.arange(0.1, 0.9, 0.01))

    scheduler.step(epoch - 1)
    lr = optimizer.param_groups[0]['lr']
    mark = ' *BEST*' if val_auc > best_auc_trans else ''
    if epoch % 5 == 0 or val_auc > best_auc_trans:
        print(f"{epoch:>5} | {tr_loss:>10.6f} | {val_auc:>7.4f} | {best_f1_val:>6.4f} | {lr:>8.2e}{mark}")

    if val_auc > best_auc_trans:
        best_auc_trans = val_auc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_epoch, no_improve = epoch, 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\nEarly stop epoch={best_epoch}")
            break

model.load_state_dict(best_state)
print(f"\n[OK] Transformer done: best AUC={best_auc_trans:.4f} @ epoch {best_epoch}")

# =============================================================================
# Step 13: 提取 256维 Transformer 特征 -> PCA 降到 16维
# =============================================================================
print("\n--- Step 13: Transformer 256维 -> PCA 16维 ---")

@torch.no_grad()
def batch_extract(model, X_np, device, bs=2048):
    model.eval()
    feats = []
    for i in range(0, len(X_np), bs):
        xb = torch.FloatTensor(X_np[i:i+bs]).to(device)
        feats.append(model.extract_features(xb).cpu().numpy())
    return np.concatenate(feats, axis=0)

trans_feat_256 = batch_extract(model, X_mo_seq, device)
print(f"  Transformer原始特征: {trans_feat_256.shape}")

PCA_DIM = 16
pca_model = PCA(n_components=PCA_DIM, random_state=42)
trans_pca16 = pca_model.fit_transform(trans_feat_256).astype(np.float32)
explained = pca_model.explained_variance_ratio_.sum()
print(f"  PCA {PCA_DIM}维: 方差解释率={explained:.4f} ({explained*100:.1f}%)")
print(f"  trans_pca16 shape: {trans_pca16.shape}")

# 构建 G3 特征
X_g3 = np.concatenate([X_g2, trans_pca16], axis=1).astype(np.float32)
print(f"  G3 +RMT+Trans_PCA16 特征: {X_g3.shape[1]}维")

# Transformer 单独AUC（用验证集的概率）
trans_only_auc = best_auc_trans
print(f"\n  Transformer 单独 AUC (80/20 val): {trans_only_auc:.4f}")

# =============================================================================
# Step 14: 消融实验 — G1 vs G2 vs G3（5折 OOF）
# =============================================================================
print("\n" + "=" * 70)
print("Step 14: Ablation - G1 vs G2 vs G3")
print("=" * 70)

def run_oof_ensemble(X_feat, y, group_name, n_splits=5):
    """跑 CatBoost+XGBoost+LightGBM 5折OOF集成, 返回 ensemble AUC"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    N = len(y)
    oof_cat = np.zeros(N, dtype=np.float64)
    oof_xgb = np.zeros(N, dtype=np.float64)
    oof_lgb = np.zeros(N, dtype=np.float64)

    cat_aucs, xgb_aucs, lgb_aucs = [], [], []

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_feat, y)):
        X_tr_f, X_va_f = X_feat[tr_idx], X_feat[va_idx]
        y_tr_f, y_va_f = y[tr_idx], y[va_idx]

        # CatBoost
        cat_model = CatBoostClassifier(
            iterations=1500, depth=6, learning_rate=0.03,
            l2_leaf_reg=5.0, random_seed=42+fold_i,
            auto_class_weights='Balanced',
            verbose=0, eval_metric='AUC', early_stopping_rounds=100)
        cat_model.fit(X_tr_f, y_tr_f, eval_set=(X_va_f, y_va_f), verbose=0)
        oof_cat[va_idx] = cat_model.predict_proba(X_va_f)[:, 1]
        cat_aucs.append(roc_auc_score(y_va_f, oof_cat[va_idx]))

        # XGBoost
        pos_w = (y_tr_f == 0).sum() / max((y_tr_f == 1).sum(), 1)
        xgb_model = xgb.XGBClassifier(
            n_estimators=1500, max_depth=6, learning_rate=0.03,
            reg_alpha=0.5, reg_lambda=2.0, scale_pos_weight=pos_w,
            subsample=0.8, colsample_bytree=0.6,
            tree_method='hist', random_state=42+fold_i,
            eval_metric='auc', early_stopping_rounds=100, verbosity=0)
        xgb_model.fit(X_tr_f, y_tr_f, eval_set=[(X_va_f, y_va_f)], verbose=False)
        oof_xgb[va_idx] = xgb_model.predict_proba(X_va_f)[:, 1]
        xgb_aucs.append(roc_auc_score(y_va_f, oof_xgb[va_idx]))

        # LightGBM（对齐Phase 1超参）
        pos_w_lgb = (y_tr_f == 0).sum() / max((y_tr_f == 1).sum(), 1)
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1000, max_depth=-1, num_leaves=63,
            learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=pos_w_lgb,
            metric='auc', random_state=42+fold_i, n_jobs=-1,
            verbose=-1)
        lgb_model.fit(X_tr_f, y_tr_f,
                      eval_set=[(X_va_f, y_va_f)],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
        oof_lgb[va_idx] = lgb_model.predict_proba(X_va_f)[:, 1]
        lgb_aucs.append(roc_auc_score(y_va_f, oof_lgb[va_idx]))

        print(f"    [{group_name}] Fold {fold_i+1}: Cat={cat_aucs[-1]:.4f} XGB={xgb_aucs[-1]:.4f} LGB={lgb_aucs[-1]:.4f}")

    # OOF AUC 加权 Rank 融合（弱模型自动降权）
    from scipy.stats import rankdata as _rd
    auc_c = roc_auc_score(y, oof_cat)
    auc_x = roc_auc_score(y, oof_xgb)
    auc_l = roc_auc_score(y, oof_lgb)
    w_c = auc_c ** 2; w_x = auc_x ** 2; w_l = auc_l ** 2
    w_sum = w_c + w_x + w_l
    oof_rank = (w_c * _rd(oof_cat) + w_x * _rd(oof_xgb) + w_l * _rd(oof_lgb)) / w_sum
    ens_auc = roc_auc_score(y, oof_rank)

    cat_mean = np.mean(cat_aucs)
    xgb_mean = np.mean(xgb_aucs)
    lgb_mean = np.mean(lgb_aucs)

    # Best F1
    prec, rec, thrs = precision_recall_curve(y, oof_rank)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best_f1 = f1s.max()

    print(f"    [{group_name}] OOF: Cat={cat_mean:.4f}+/-{np.std(cat_aucs):.4f}"
          f"  XGB={xgb_mean:.4f}+/-{np.std(xgb_aucs):.4f}"
          f"  LGB={lgb_mean:.4f}+/-{np.std(lgb_aucs):.4f}")
    print(f"    [{group_name}] 集成 OOF AUC={ens_auc:.4f}  F1={best_f1:.4f}")

    return {
        'ens_auc': ens_auc, 'best_f1': best_f1,
        'cat_auc': cat_mean, 'xgb_auc': xgb_mean, 'lgb_auc': lgb_mean,
        'oof_rank': oof_rank,
    }

# G1: Baseline
print("\n>> G1: Baseline")
res_g1 = run_oof_ensemble(X_baseline, labels, "G1")

# G2: +RMT
print("\n>> G2: +RMT")
res_g2 = run_oof_ensemble(X_g2, labels, "G2")

# G3: +RMT +Trans_PCA16
print("\n>> G3: +RMT+Trans_PCA16")
res_g3 = run_oof_ensemble(X_g3, labels, "G3")

# =============================================================================
# Step 15: 对比实验表
# =============================================================================
# Step 15: 基线对比实验（有监督分类器 + 无监督异常检测）
# =============================================================================
print("\n" + "=" * 70)
print("Step 15: 基线对比实验（有监督分类器 + 无监督异常检测）")
print("=" * 70)

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
# from sklearn.svm import OneClassSVM, SVC  # 已注释：SVM 耗时过长且效果不佳
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Best F1 工具函数
def _best_f1(y_true, scores):
    prec, rec, _ = precision_recall_curve(y_true, scores)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    return f1s.max()

# 对 G3 特征做 RobustScaler（LR/MLP 对尺度敏感）
from sklearn.preprocessing import RobustScaler as RS2
X_g3_scaled = RS2().fit_transform(X_g3)

# # 为 OCSVM 和 SVM-RBF 做 PCA 降维（已注释：SVM 相关实验已移除）
# pca_svm = PCA(n_components=32, random_state=42)
# X_g3_pca32 = pca_svm.fit_transform(X_g3_scaled).astype(np.float32)

skf_ad = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------- 15a: 有监督分类器对比（公平对比：同样的G3特征矩阵） ----------
print("\n--- 15a: 有监督分类器（同样G3特征, 5折OOF）---")

# oof_svc     = np.zeros(N_USERS, dtype=np.float64)   # SVM (RBF) — 已注释
oof_rf      = np.zeros(N_USERS, dtype=np.float64)   # Random Forest
oof_lr      = np.zeros(N_USERS, dtype=np.float64)   # Logistic Regression
oof_mlp     = np.zeros(N_USERS, dtype=np.float64)   # MLP

for fold_i, (tr_idx, va_idx) in enumerate(skf_ad.split(X_g3, labels)):
    y_tr, y_va = labels[tr_idx], labels[va_idx]

    # --- SVM (RBF) —— 已注释（耗时过长，AUC仅0.51） ---
    # t0 = time.time()
    # svc = SVC(kernel='rbf', gamma='scale', C=1.0, probability=True,
    #           class_weight='balanced', random_state=42)
    # svc.fit(X_g3_pca32[tr_idx], y_tr)
    # oof_svc[va_idx] = svc.predict_proba(X_g3_pca32[va_idx])[:, 1]
    # t_svc = time.time() - t0

    # --- Random Forest ---
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=300, max_depth=12,
                                 class_weight='balanced',
                                 random_state=42+fold_i, n_jobs=-1)
    rf.fit(X_g3[tr_idx], y_tr)
    oof_rf[va_idx] = rf.predict_proba(X_g3[va_idx])[:, 1]
    t_rf = time.time() - t0

    # --- Logistic Regression ---
    t0 = time.time()
    lr = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced',
                            solver='lbfgs', random_state=42)
    lr.fit(X_g3_scaled[tr_idx], y_tr)
    oof_lr[va_idx] = lr.predict_proba(X_g3_scaled[va_idx])[:, 1]
    t_lr = time.time() - t0

    # --- MLP ---
    t0 = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                         max_iter=200, early_stopping=True,
                         validation_fraction=0.1,
                         random_state=42+fold_i)
    mlp.fit(X_g3_scaled[tr_idx], y_tr)
    oof_mlp[va_idx] = mlp.predict_proba(X_g3_scaled[va_idx])[:, 1]
    t_mlp = time.time() - t0

    # auc_svc_f = roc_auc_score(y_va, oof_svc[va_idx])
    auc_rf_f  = roc_auc_score(y_va, oof_rf[va_idx])
    auc_lr_f  = roc_auc_score(y_va, oof_lr[va_idx])
    auc_mlp_f = roc_auc_score(y_va, oof_mlp[va_idx])
    print(f"  Fold {fold_i+1}: RF={auc_rf_f:.4f}({t_rf:.0f}s) "
          f"LR={auc_lr_f:.4f}({t_lr:.0f}s) MLP={auc_mlp_f:.4f}({t_mlp:.0f}s)")

# auc_svc_oof = roc_auc_score(labels, oof_svc)
auc_rf_oof  = roc_auc_score(labels, oof_rf)
auc_lr_oof  = roc_auc_score(labels, oof_lr)
auc_mlp_oof = roc_auc_score(labels, oof_mlp)

# f1_svc = _best_f1(labels, oof_svc)
f1_rf  = _best_f1(labels, oof_rf)
f1_lr  = _best_f1(labels, oof_lr)
f1_mlp = _best_f1(labels, oof_mlp)

print(f"\n  有监督分类器 OOF 汇总:")
# print(f"    SVM (RBF, PCA32):    AUC={auc_svc_oof:.4f}  F1={f1_svc:.4f}")
print(f"    Random Forest:       AUC={auc_rf_oof:.4f}  F1={f1_rf:.4f}")
print(f"    Logistic Regression: AUC={auc_lr_oof:.4f}  F1={f1_lr:.4f}")
print(f"    MLP (128-64):        AUC={auc_mlp_oof:.4f}  F1={f1_mlp:.4f}")

# ---------- 15b: 无监督异常检测基线（作为参考下界）----------
print("\n--- 15b: 无监督异常检测（IF / LOF, 作为参考下界）---")

oof_if  = np.zeros(N_USERS, dtype=np.float64)
oof_lof = np.zeros(N_USERS, dtype=np.float64)
# oof_ocsvm = np.zeros(N_USERS, dtype=np.float64)  # 已注释：OCSVM 耗时过长

for fold_i, (tr_idx, va_idx) in enumerate(skf_ad.split(X_g3, labels)):
    X_tr_ad = X_g3_scaled[tr_idx]
    X_va_ad = X_g3_scaled[va_idx]
    # X_tr_pca = X_g3_pca32[tr_idx]  # 已注释：仅 OCSVM 使用
    # X_va_pca = X_g3_pca32[va_idx]

    # Isolation Forest
    t0 = time.time()
    iso = IsolationForest(n_estimators=300, contamination=0.1,
                          random_state=42+fold_i, n_jobs=-1)
    iso.fit(X_tr_ad)
    oof_if[va_idx] = -iso.decision_function(X_va_ad)
    t_if = time.time() - t0

    # LOF
    t0 = time.time()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1,
                              novelty=True, n_jobs=-1)
    lof.fit(X_tr_ad)
    oof_lof[va_idx] = -lof.decision_function(X_va_ad)
    t_lof = time.time() - t0

    # # One-Class SVM — 已注释（耗时过长，AUC仅约0.51）
    # t0 = time.time()
    # ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    # ocsvm.fit(X_tr_pca)
    # oof_ocsvm[va_idx] = -ocsvm.decision_function(X_va_pca)
    # t_ocsvm = time.time() - t0

    auc_if  = roc_auc_score(labels[va_idx], oof_if[va_idx])
    auc_lof = roc_auc_score(labels[va_idx], oof_lof[va_idx])
    # auc_ocsvm = roc_auc_score(labels[va_idx], oof_ocsvm[va_idx])
    print(f"  Fold {fold_i+1}: IF={auc_if:.4f}({t_if:.0f}s) LOF={auc_lof:.4f}({t_lof:.0f}s)")

auc_if_oof    = roc_auc_score(labels, oof_if)
auc_lof_oof   = roc_auc_score(labels, oof_lof)
# auc_ocsvm_oof = roc_auc_score(labels, oof_ocsvm)

# 修正方向
if auc_if_oof < 0.5:
    oof_if = -oof_if; auc_if_oof = 1 - auc_if_oof
if auc_lof_oof < 0.5:
    oof_lof = -oof_lof; auc_lof_oof = 1 - auc_lof_oof
# if auc_ocsvm_oof < 0.5:
#     oof_ocsvm = -oof_ocsvm; auc_ocsvm_oof = 1 - auc_ocsvm_oof

f1_if    = _best_f1(labels, oof_if)
f1_lof   = _best_f1(labels, oof_lof)
# f1_ocsvm = _best_f1(labels, oof_ocsvm)

print(f"\n  无监督基线 OOF 汇总:")
print(f"    Isolation Forest:  AUC={auc_if_oof:.4f}  F1={f1_if:.4f}")
print(f"    LOF:               AUC={auc_lof_oof:.4f}  F1={f1_lof:.4f}")
# print(f"    One-Class SVM:     AUC={auc_ocsvm_oof:.4f}  F1={f1_ocsvm:.4f}")  # 已注释

# =============================================================================
# Step 16: 完整对比实验表
# =============================================================================
print("\n" + "=" * 70)
print("Step 16: 完整对比实验表")
print("=" * 70)

print("\n" + "=" * 72)
print("                         消融实验结果")
print("=" * 72)
print(f"{'配置':<25} {'特征维度':>8} {'OOF AUC':>9} {'OOF F1':>9}")
print("-" * 72)
print(f"{'G1: Baseline':<25} {X_baseline.shape[1]:>8} {res_g1['ens_auc']:>9.4f} {res_g1['best_f1']:>9.4f}")
print(f"{'G2: +RMT':<25} {X_g2.shape[1]:>8} {res_g2['ens_auc']:>9.4f} {res_g2['best_f1']:>9.4f}")
print(f"{'G3: +RMT+Trans_PCA16':<25} {X_g3.shape[1]:>8} {res_g3['ens_auc']:>9.4f} {res_g3['best_f1']:>9.4f}")
print("-" * 72)
print(f"{'G2-G1 (RMT gain)':<25} {'+' + str(rmt_features.shape[1]):>8} {res_g2['ens_auc']-res_g1['ens_auc']:>+9.4f} {res_g2['best_f1']-res_g1['best_f1']:>+9.4f}")
print(f"{'G3-G2 (Trans gain)':<25} {'+' + str(PCA_DIM):>8} {res_g3['ens_auc']-res_g2['ens_auc']:>+9.4f} {res_g3['best_f1']-res_g2['best_f1']:>+9.4f}")
print(f"{'G3-G1 (Total gain)':<25} {'+' + str(X_g3.shape[1]-X_baseline.shape[1]):>8} {res_g3['ens_auc']-res_g1['ens_auc']:>+9.4f} {res_g3['best_f1']-res_g1['best_f1']:>+9.4f}")
print("=" * 72)

print("\n" + "=" * 78)
print("          完整模型对比实验 (公平对比: 同样G3特征矩阵)")
print("=" * 78)
print(f"{'Model':<35} {'Type':<12} {'OOF AUC':>9} {'OOF F1':>9}")
print("-" * 78)
print(f"{'Logistic Regression (G3)':<35} {'Supervised':<12} {auc_lr_oof:>9.4f} {f1_lr:>9.4f}")
# print(f"{'SVM-RBF (G3, PCA32)':<35} {'Supervised':<12} {auc_svc_oof:>9.4f} {f1_svc:>9.4f}")
print(f"{'MLP (G3)':<35} {'Supervised':<12} {auc_mlp_oof:>9.4f} {f1_mlp:>9.4f}")
print(f"{'Random Forest (G3)':<35} {'Supervised':<12} {auc_rf_oof:>9.4f} {f1_rf:>9.4f}")
print(f"{'Transformer (standalone)':<35} {'Supervised':<12} {trans_only_auc:>9.4f} {'  -':>9}")
print(f"{'CatBoost (G3)':<35} {'GBDT':<12} {res_g3['cat_auc']:>9.4f} {'  -':>9}")
print(f"{'XGBoost (G3)':<35} {'GBDT':<12} {res_g3['xgb_auc']:>9.4f} {'  -':>9}")
print(f"{'LightGBM (G3)':<35} {'GBDT':<12} {res_g3['lgb_auc']:>9.4f} {'  -':>9}")
print(f"{'Ours (RMT+Trans+GBDT Ensemble)':<35} {'Ensemble':<12} {res_g3['ens_auc']:>9.4f} {res_g3['best_f1']:>9.4f}")
print("-" * 78)
print(f"{'Isolation Forest (G3)':<35} {'Unsuperv.':<12} {auc_if_oof:>9.4f} {f1_if:>9.4f}")
print(f"{'LOF (G3)':<35} {'Unsuperv.':<12} {auc_lof_oof:>9.4f} {f1_lof:>9.4f}")
# print(f"{'One-Class SVM (G3, PCA32)':<35} {'Unsuperv.':<12} {auc_ocsvm_oof:>9.4f} {f1_ocsvm:>9.4f}")  # 已注释
print("=" * 78)

total_elapsed = time.time() - t_total
print(f"\nTotal time: {total_elapsed/60:.1f} min")
print("Phase 3 Done!")

# =============================================================================
# Step 17: 可视化图表（论文级，参考 london_difficulty_visualize.py 风格）
# =============================================================================
print("\n" + "=" * 70)
print("Step 17: 生成可视化图表")
print("=" * 70)

import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# ── 中文字体配置 ──
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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

RESULT_DIR = r'C:\Users\wb.zhoushujie\PyCharmMiscProject\results'
os.makedirs(RESULT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 图1: 消融实验柱状图 — G1 vs G2 vs G3 的 AUC & F1 对比
# ─────────────────────────────────────────────────────────────────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5.5))

groups = ['G1: Baseline', 'G2: +RMT', 'G3: +RMT+Trans']
group_aucs = [res_g1['ens_auc'], res_g2['ens_auc'], res_g3['ens_auc']]
group_f1s  = [res_g1['best_f1'], res_g2['best_f1'], res_g3['best_f1']]
group_dims = [X_baseline.shape[1], X_g2.shape[1], X_g3.shape[1]]
colors_abl = ['#95A5A6', '#3498DB', '#E74C3C']
x_abl = np.arange(len(groups))

# AUC 子图
ax = axes1[0]
bars = ax.bar(x_abl, group_aucs, color=colors_abl, width=0.55,
              edgecolor='white', linewidth=0.8)
for bar, val, dim in zip(bars, group_aucs, group_dims):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
            f'{val:.4f}\n({dim}维)', ha='center', va='bottom',
            fontsize=10, fontweight='bold')
ax.set_xticks(x_abl)
ax.set_xticklabels(groups, fontsize=11)
ax.set_ylabel('OOF AUC')
ax.set_title('消融实验 — 集成模型 AUC')
ax.set_ylim(bottom=min(group_aucs) - 0.02, top=max(group_aucs) + 0.025)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

# F1 子图
ax = axes1[1]
bars = ax.bar(x_abl, group_f1s, color=colors_abl, width=0.55,
              edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, group_f1s):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
            f'{val:.4f}', ha='center', va='bottom',
            fontsize=10, fontweight='bold')
ax.set_xticks(x_abl)
ax.set_xticklabels(groups, fontsize=11)
ax.set_ylabel('OOF F1')
ax.set_title('消融实验 — 集成模型 F1')
ax.set_ylim(bottom=min(group_f1s) - 0.02, top=max(group_f1s) + 0.025)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

fig1.suptitle('SGCC 数据集消融实验（Ablation Study）', fontsize=16, fontweight='bold', y=1.02)
fig1.tight_layout()
fig1_path = os.path.join(RESULT_DIR, 'sgcc_fig1_ablation.png')
fig1.savefig(fig1_path)
print(f"[OK] 图1 消融实验柱状图 已保存: {fig1_path}")
plt.close(fig1)

# ─────────────────────────────────────────────────────────────────────────────
# 图2: 完整模型对比 — 横向柱状图（按 AUC 降序排列）
# ─────────────────────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(11, 7))

# 汇总所有模型结果
model_results = {
    '本文方法 (RMT+Trans+Ensemble)': {'auc': res_g3['ens_auc'], 'f1': res_g3['best_f1'], 'type': 'Ensemble',    'color': '#E74C3C'},
    'CatBoost (G3)':                 {'auc': res_g3['cat_auc'], 'f1': None,               'type': 'GBDT',        'color': '#3498DB'},
    'XGBoost (G3)':                  {'auc': res_g3['xgb_auc'], 'f1': None,               'type': 'GBDT',        'color': '#2ECC71'},
    'LightGBM (G3)':                 {'auc': res_g3['lgb_auc'], 'f1': None,               'type': 'GBDT',        'color': '#9B59B6'},
    '随机森林 (G3)':                   {'auc': auc_rf_oof,        'f1': f1_rf,              'type': 'Supervised',  'color': '#F39C12'},
    'Transformer (standalone)':       {'auc': trans_only_auc,    'f1': None,               'type': 'Supervised',  'color': '#1ABC9C'},
    '逻辑回归 (G3)':                   {'auc': auc_lr_oof,        'f1': f1_lr,              'type': 'Supervised',  'color': '#95A5A6'},
    'MLP (G3)':                       {'auc': auc_mlp_oof,       'f1': f1_mlp,             'type': 'Supervised',  'color': '#BDC3C7'},
    '孤立森林 (G3)':                   {'auc': auc_if_oof,        'f1': f1_if,              'type': 'Unsuperv.',   'color': '#E67E22'},
    'LOF (G3)':                       {'auc': auc_lof_oof,       'f1': f1_lof,             'type': 'Unsuperv.',   'color': '#D35400'},
}

# 按 AUC 降序排列
sorted_models = sorted(model_results.keys(), key=lambda m: model_results[m]['auc'], reverse=True)
y_pos = np.arange(len(sorted_models))
bar_aucs = [model_results[m]['auc'] for m in sorted_models]
bar_colors = [model_results[m]['color'] for m in sorted_models]

bars = ax2.barh(y_pos, bar_aucs, color=bar_colors, height=0.6,
                edgecolor='white', linewidth=0.5)

for bar, m_name in zip(bars, sorted_models):
    auc_val = model_results[m_name]['auc']
    m_type = model_results[m_name]['type']
    ax2.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2.,
             f'{auc_val:.4f}  [{m_type}]', ha='left', va='center',
             fontsize=9, fontweight='bold')

ax2.set_yticks(y_pos)
ax2.set_yticklabels(sorted_models, fontsize=11)
ax2.set_xlabel('OOF AUC')
ax2.set_title('SGCC 数据集 — 完整模型对比（按 AUC 降序）', fontsize=14, fontweight='bold')
ax2.invert_yaxis()
ax2.set_xlim(left=min(bar_aucs) - 0.05, right=max(bar_aucs) + 0.08)
ax2.axvline(x=res_g3['ens_auc'], color='#E74C3C', linestyle='--',
            linewidth=1.0, alpha=0.5, label='本文方法')
ax2.legend(loc='lower right', fontsize=9)

fig2.tight_layout()
fig2_path = os.path.join(RESULT_DIR, 'sgcc_fig2_model_comparison.png')
fig2.savefig(fig2_path)
print(f"[OK] 图2 模型对比横向柱状图 已保存: {fig2_path}")
plt.close(fig2)

# ─────────────────────────────────────────────────────────────────────────────
# 图3: GBDT 组件在消融中的性能贡献热力图
# ─────────────────────────────────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 5))

# 构建热力图矩阵: 行=GBDT组件, 列=消融组
components = ['CatBoost', 'XGBoost', 'LightGBM', '集成 (Ensemble)']
ablation_groups_cn = ['G1: Baseline', 'G2: +RMT', 'G3: +RMT+Trans']

heatmap_data = np.array([
    [res_g1['cat_auc'], res_g2['cat_auc'], res_g3['cat_auc']],
    [res_g1['xgb_auc'], res_g2['xgb_auc'], res_g3['xgb_auc']],
    [res_g1['lgb_auc'], res_g2['lgb_auc'], res_g3['lgb_auc']],
    [res_g1['ens_auc'], res_g2['ens_auc'], res_g3['ens_auc']],
])

im = ax3.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=heatmap_data.min() - 0.01, vmax=heatmap_data.max() + 0.005)
ax3.set_xticks(np.arange(len(ablation_groups_cn)))
ax3.set_yticks(np.arange(len(components)))
ax3.set_xticklabels(ablation_groups_cn, fontsize=11)
ax3.set_yticklabels(components, fontsize=11)

# 在每个格子中标注数值
for i in range(len(components)):
    for j in range(len(ablation_groups_cn)):
        val = heatmap_data[i, j]
        # 自动选择文字颜色
        normalized = (val - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-8)
        text_color = 'white' if normalized < 0.3 else 'black'
        ax3.text(j, i, f'{val:.4f}', ha='center', va='center',
                 fontsize=12, fontweight='bold', color=text_color)

ax3.set_title('GBDT 组件 × 消融配置 AUC 热力图（SGCC 数据集）', fontsize=13, fontweight='bold')
fig3.colorbar(im, ax=ax3, label='OOF AUC', shrink=0.8)

fig3.tight_layout()
fig3_path = os.path.join(RESULT_DIR, 'sgcc_fig3_gbdt_heatmap.png')
fig3.savefig(fig3_path)
print(f"[OK] 图3 GBDT组件热力图 已保存: {fig3_path}")
plt.close(fig3)

# ─────────────────────────────────────────────────────────────────────────────
# 图4: 消融增量贡献柱状图（每一步带来的 AUC 增益）
# ─────────────────────────────────────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(8, 5))

increments = ['Baseline (G1)', '+RMT (G2-G1)', '+Trans_PCA16 (G3-G2)']
inc_values = [
    res_g1['ens_auc'],
    res_g2['ens_auc'] - res_g1['ens_auc'],
    res_g3['ens_auc'] - res_g2['ens_auc'],
]
inc_colors = ['#95A5A6', '#3498DB', '#E74C3C']

bars = ax4.bar(np.arange(len(increments)), inc_values, color=inc_colors,
               width=0.55, edgecolor='white', linewidth=0.8)

for bar, val in zip(bars, inc_values):
    label = f'{val:.4f}' if val < 0.5 else f'{val:.4f}'
    y_offset = 0.001 if val > 0 else -0.003
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + y_offset,
             label, ha='center', va='bottom',
             fontsize=11, fontweight='bold')

ax4.set_xticks(np.arange(len(increments)))
ax4.set_xticklabels(increments, fontsize=10)
ax4.set_ylabel('AUC / AUC 增益')
ax4.set_title('各模块对集成模型 AUC 的贡献分解（SGCC 数据集）', fontsize=13, fontweight='bold')
ax4.axhline(y=0, color='black', linewidth=0.5)

fig4.tight_layout()
fig4_path = os.path.join(RESULT_DIR, 'sgcc_fig4_ablation_increments.png')
fig4.savefig(fig4_path)
print(f"[OK] 图4 消融增量贡献图 已保存: {fig4_path}")
plt.close(fig4)

print(f"\n所有 SGCC 可视化图表已生成完毕！")
print(f"图表保存目录: {RESULT_DIR}")
print(f"  - sgcc_fig1_ablation.png         消融实验柱状图")
print(f"  - sgcc_fig2_model_comparison.png  完整模型对比横向柱状图")
print(f"  - sgcc_fig3_gbdt_heatmap.png      GBDT组件性能热力图")
print(f"  - sgcc_fig4_ablation_increments.png 消融增量贡献图")
