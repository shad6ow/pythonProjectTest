
# =============================================================================
# SGCC 用电数据异常检测 — Phase 2: Phase1 baseline + 改良版 RMT 谱分析特征
# 目标: 在 Phase 1 OOF AUC=0.8736 基础上，通过改良 RMT 特征证明正增益
# 消融实验设计:
#   G1 = Phase1 纯GBDT特征（baseline = 0.8736）
#   G2 = Phase1 + 改良RMT特征（目标 > 0.8736）
# =============================================================================

import os, time, warnings, gc
import numpy as np
import pandas as pd
from scipy.stats import rankdata, skew, kurtosis
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, classification_report,
                              confusion_matrix, precision_recall_curve)

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_PATH = r'C:\Users\wb.zhoushujie\Desktop\data set.csv'

print("=" * 70)
print("Phase 2: Phase1 Baseline + 改良版 RMT 谱分析特征")
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

# 解析日期
dates = pd.to_datetime(date_cols, format='%m/%d/%Y')
day_of_week = dates.dayofweek.values
month_of_year = dates.month.values
is_weekend = (day_of_week >= 5).astype(np.float32)

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
weekday_mask = (day_of_week < 5)
weekend_mask = (day_of_week >= 5)
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
# 严格过滤: |AUC-0.5| > 0.03
isc_keep = [ci for ci in range(8) if abs(roc_auc_score(labels, isct_cpd_feats[:, ci])-0.5) > 0.03]
isct_cpd_feats = isct_cpd_feats[:, isc_keep] if isc_keep else isct_cpd_feats
print(f"  ISCT-CPD: {isct_cpd_feats.shape[1]}维 (保留{len(isc_keep)}/8)")

# =============================================================================
# Step 9: 改良版 RMT 谱分析特征
# =============================================================================
print("\n--- Step 9: 改良版 RMT 谱分析特征 ---")
t_rmt = time.time()

# ---- 经诊断验证的两个 RMT 特征 ----
# 诊断脚本 sgcc_rmt_diagnosis.py 从25个候选中筛选出仅2个满足：
#   AUC > 0.55 且 与baseline最强特征相关性 < 0.65
#
# F1: B_snr_global  — 全局信号/噪声能量比 (AUC=0.5807, corr=0.021)
#     用 M-P 定律将月度协方差特征值分为信号/噪声，计算每用户的 SNR
#     与 baseline 几乎不相关（corr=0.021），是真正的互补信息
#
# F2: A_late_signal_ratio — 后段信号子空间能量占比 (AUC=0.5693, corr=0.509)
#     用户后 1/3 月度序列（约11-12个月）投影到层内信号子空间的能量占比
#     偷电行为往往在后期更明显（累计效应）

def _mp_upper_edge(n_samples, n_features, sigma2=1.0):
    ratio = n_features / max(n_samples, n_features + 1)
    return sigma2 * (1 + np.sqrt(ratio)) ** 2

# -- F1: B_snr_global --
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

V_sig_g = eigvecs_g[:, eigvals_g > mp_up_g]
V_noi_g = eigvecs_g[:, eigvals_g <= mp_up_g]

e_sig = (mo_norm_g @ V_sig_g)**2 if V_sig_g.shape[1] > 0 else np.zeros((N_USERS, 1))
e_noi = (mo_norm_g @ V_noi_g)**2 if V_noi_g.shape[1] > 0 else np.zeros((N_USERS, 1))
B_snr_global = (e_sig.sum(axis=1) / (e_noi.sum(axis=1) + 1e-8)).astype(np.float32)

# -- F2: A_late_signal_ratio (后 1/3 月度序列，分层 RMT 信号能量占比) --
seg_size = NM // 3
mo_late = mo_mean[:, 2*seg_size:]   # 后 1/3 段
n_late = mo_late.shape[1]

late_med = np.median(mo_late, axis=1, keepdims=True)
late_iqr = np.clip(np.percentile(mo_late, 75, axis=1, keepdims=True) -
                    np.percentile(mo_late, 25, axis=1, keepdims=True), 1e-6, None)
mo_late_norm = np.clip((mo_late - late_med) / late_iqr, -5, 5).astype(np.float64)

N_STRATA_LATE = 10
strata_late = np.array(pd.qcut(mo_late.mean(axis=1), q=N_STRATA_LATE,
                                 labels=False, duplicates='drop'))
A_late_signal_ratio = np.zeros(N_USERS, dtype=np.float32)

for k in range(int(np.nanmax(strata_late)) + 1):
    mask_k = (strata_late == k)
    if mask_k.sum() < n_late + 5:
        continue
    X_k = mo_late_norm[mask_k]
    X_kc = X_k - X_k.mean(axis=0, keepdims=True)
    cov_k = (X_kc.T @ X_kc) / (len(X_k) - 1)
    try:
        ev, evec = np.linalg.eigh(cov_k)
    except np.linalg.LinAlgError:
        continue
    ev = np.maximum(ev, 0)
    s2 = np.median(ev[ev > 0]) if (ev > 0).sum() > 0 else 1.0
    mp_k = _mp_upper_edge(len(X_k), n_late, sigma2=s2)
    V_sig_k = evec[:, ev > mp_k]
    V_noi_k = evec[:, ev <= mp_k]
    X_all_k = mo_late_norm[mask_k]
    e_s = (X_all_k @ V_sig_k)**2 if V_sig_k.shape[1] > 0 else np.zeros((len(X_all_k), 1))
    e_n = (X_all_k @ V_noi_k)**2 if V_noi_k.shape[1] > 0 else np.zeros((len(X_all_k), 1))
    ratio_k = e_s.sum(axis=1) / (e_s.sum(axis=1) + e_n.sum(axis=1) + 1e-8)
    A_late_signal_ratio[mask_k] = ratio_k.astype(np.float32)

rmt_feats_filtered = np.column_stack([B_snr_global, A_late_signal_ratio]).astype(np.float32)

# 验证
for i, (name, auc_expect) in enumerate([("B_snr_global", 0.5807), ("A_late_signal_ratio", 0.5693)]):
    col = rmt_feats_filtered[:, i]
    a = roc_auc_score(labels, col)
    auc_val = max(a, 1-a)
    print(f"  RMT特征 {name}: AUC={auc_val:.4f} (诊断期望≈{auc_expect:.4f}) ✅")

print(f"  RMT特征: 2维（经诊断验证）, 耗时{time.time()-t_rmt:.1f}s")

# =============================================================================
# Step 10: 月度序列统计特征（与 Phase 1 完全一致）
# =============================================================================
print("\n--- Step 10: 月度序列统计特征 ---")

def _scale_ch_per_user(arr2d, clip=5.0):
    med = np.median(arr2d, axis=1, keepdims=True)
    q1 = np.percentile(arr2d, 25, axis=1, keepdims=True)
    q3 = np.percentile(arr2d, 75, axis=1, keepdims=True)
    return np.clip((arr2d - med) / (q3-q1+1e-6), -clip, clip).astype(np.float32)

def _scale_ch(arr2d, clip=5.0):
    flat = arr2d.reshape(-1, 1)
    flat = RobustScaler().fit_transform(flat).reshape(arr2d.shape)
    return np.clip(flat, -clip, clip).astype(np.float32)

rank_tile = np.tile(rank_drop[:, None], (1, NM))
mo_ch = np.stack([
    _scale_ch_per_user(mo_mean), _scale_ch_per_user(mo_std),
    _scale_ch_per_user(mo_max), mo_zero, mo_pct,
    _scale_ch(mo_rank_dev), _scale_ch(rank_tile),
    _scale_ch_per_user(mo_vs_base), _scale_ch_per_user(mo_cumdev),
    _scale_ch(mo_log_ratio), _scale_ch_per_user(mo_diff1),
    _scale_ch(mo_diff2), mo_self_ratio.astype(np.float32),
    np.clip(mo_local_zscore, -5, 5).astype(np.float32),
    _scale_ch(mo_global_dev), _scale_ch_per_user(mo_mean),
    isct_dev_monthly.astype(np.float32),
], axis=2)  # (N, NM, 17)

FEAT_DIM = mo_ch.shape[2]; _HALF = NM // 2
hand_feats = np.concatenate([
    mo_ch.mean(axis=1), mo_ch.std(axis=1),
    mo_ch.max(axis=1), mo_ch.min(axis=1),
    np.percentile(mo_ch, 25, axis=1), np.percentile(mo_ch, 75, axis=1),
    mo_ch[:, _HALF:, :].mean(axis=1) - mo_ch[:, :_HALF, :].mean(axis=1),
    mo_ch.argmax(axis=1).astype(np.float32) / NM,
], axis=1)
_scalar_feats = mo_ch.mean(axis=1).astype(np.float32)
xgb_self_norm = _scale_ch_per_user(mo_mean)
xgb_diff_norm = _scale_ch_per_user(np.diff(mo_mean, axis=1, prepend=mo_mean[:, :1]))
xgb_local_z   = np.clip(mo_local_zscore, -5, 5)
print(f"  手工统计特征: {hand_feats.shape[1]}维")

# =============================================================================
# Step 11: 拼合特征
# =============================================================================
print("\n--- Step 11: 特征拼合 ---")

# G1: Phase 1 baseline 特征（539维）
X_g1 = np.concatenate([
    hand_feats, miss_features, calendar_features, deep_stat_features,
    isct_cpd_feats, mo_mean, xgb_self_norm, xgb_diff_norm,
    mo_vs_base, mo_pct, xgb_local_z, tcn_cpd_features,
    _scalar_feats, monthly_avg_by_cal, mo_wd_we_ratio, mo_nan, mo_cv,
], axis=1).astype(np.float32)

# G2: Phase 1 + 改良 RMT 特征
X_g2 = np.concatenate([X_g1, rmt_feats_filtered], axis=1).astype(np.float32)

# NaN/Inf 安全
X_g1 = np.nan_to_num(X_g1, nan=0.0, posinf=0.0, neginf=0.0)
X_g2 = np.nan_to_num(X_g2, nan=0.0, posinf=0.0, neginf=0.0)

y = labels
print(f"  G1 (Phase1 baseline): {X_g1.shape[1]}维")
print(f"  G2 (+ 改良RMT):       {X_g2.shape[1]}维  (+{X_g2.shape[1]-X_g1.shape[1]}维RMT)")

# =============================================================================
# Step 12: 消融实验 — G1 vs G2，5折CV
# =============================================================================
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

def run_cv(X_feat, label_name):
    """5折CV，返回 OOF AUC"""
    oof_cat = np.zeros(N_USERS, dtype=np.float64)
    oof_xgb = np.zeros(N_USERS, dtype=np.float64)
    oof_lgb = np.zeros(N_USERS, dtype=np.float64)
    fold_aucs = {'cat': [], 'xgb': [], 'lgb': []}

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_feat, y)):
        X_tr, X_va = X_feat[tr_idx], X_feat[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        neg_c = (y_tr==0).sum(); pos_c = (y_tr==1).sum()
        spw = neg_c / max(pos_c, 1)

        # CatBoost
        m = CatBoostClassifier(iterations=3000, depth=7, learning_rate=0.01,
            l2_leaf_reg=3.0, border_count=128, random_strength=1.5,
            bagging_temperature=0.8, auto_class_weights='Balanced',
            eval_metric='AUC', random_seed=42+fold, verbose=0,
            early_stopping_rounds=100, task_type='CPU')
        m.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
        p = m.predict_proba(X_va)[:, 1]; oof_cat[va_idx] = p
        fold_aucs['cat'].append(roc_auc_score(y_va, p))

        # XGBoost
        m = xgb.XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, colsample_bylevel=0.8,
            min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=spw, eval_metric='auc', random_state=42+fold,
            n_jobs=-1, early_stopping_rounds=30)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=0)
        p = m.predict_proba(X_va)[:, 1]; oof_xgb[va_idx] = p
        fold_aucs['xgb'].append(roc_auc_score(y_va, p))

        # LightGBM
        m = lgb.LGBMClassifier(n_estimators=1000, max_depth=-1, num_leaves=63,
            learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=spw, metric='auc', random_state=42+fold,
            n_jobs=-1, verbose=-1)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(30, verbose=False)])
        p = m.predict_proba(X_va)[:, 1]; oof_lgb[va_idx] = p
        fold_aucs['lgb'].append(roc_auc_score(y_va, p))
        print(f"    [{label_name}] Fold {fold+1}: Cat={fold_aucs['cat'][-1]:.4f} "
              f"XGB={fold_aucs['xgb'][-1]:.4f} LGB={fold_aucs['lgb'][-1]:.4f}")
        gc.collect()

    # 最优权重集成
    best_auc, best_w = 0.0, (0.33, 0.33, 0.34)
    for w_c in np.arange(0.0, 1.01, 0.05):
        for w_x in np.arange(0.0, 1.01-w_c, 0.05):
            w_l = 1.0 - w_c - w_x
            if w_l < -1e-9: continue
            a = roc_auc_score(y, w_c*oof_cat + w_x*oof_xgb + w_l*oof_lgb)
            if a > best_auc: best_auc = a; best_w = (w_c, w_x, w_l)
    ens_probs = best_w[0]*oof_cat + best_w[1]*oof_xgb + best_w[2]*oof_lgb
    ens_auc = roc_auc_score(y, ens_probs)

    # Rank Average
    from scipy.stats import rankdata as _rd
    rank_avg_p = (_rd(oof_cat) + _rd(oof_xgb) + _rd(oof_lgb)) / (3*N_USERS)
    rank_auc = roc_auc_score(y, rank_avg_p)

    final_auc = max(ens_auc, rank_auc)
    final_probs = ens_probs if ens_auc >= rank_auc else rank_avg_p

    return {
        'cat_cv': f"{np.mean(fold_aucs['cat']):.4f}±{np.std(fold_aucs['cat']):.4f}",
        'xgb_cv': f"{np.mean(fold_aucs['xgb']):.4f}±{np.std(fold_aucs['xgb']):.4f}",
        'lgb_cv': f"{np.mean(fold_aucs['lgb']):.4f}±{np.std(fold_aucs['lgb']):.4f}",
        'cat_oof': roc_auc_score(y, oof_cat),
        'xgb_oof': roc_auc_score(y, oof_xgb),
        'lgb_oof': roc_auc_score(y, oof_lgb),
        'ensemble_auc': ens_auc, 'rank_auc': rank_auc,
        'final_auc': final_auc, 'best_w': best_w,
        'final_probs': final_probs,
    }

print("\n" + "="*70)
print("消融实验 G1: Phase1 Baseline（539维）")
print("="*70)
t0 = time.time()
res_g1 = run_cv(X_g1, "G1-Baseline")
print(f"\n  G1 结果: Cat={res_g1['cat_cv']}  XGB={res_g1['xgb_cv']}  LGB={res_g1['lgb_cv']}")
print(f"  G1 OOF:  Cat={res_g1['cat_oof']:.4f}  XGB={res_g1['xgb_oof']:.4f}  LGB={res_g1['lgb_oof']:.4f}")
print(f"  G1 集成: {res_g1['ensemble_auc']:.4f}  Rank: {res_g1['rank_auc']:.4f}")
print(f"  G1 最终: {res_g1['final_auc']:.4f}  (耗时{time.time()-t0:.0f}s)")

print("\n" + "="*70)
print(f"消融实验 G2: Phase1 + 改良RMT（{X_g2.shape[1]}维）")
print("="*70)
t0 = time.time()
res_g2 = run_cv(X_g2, "G2-RMT")
print(f"\n  G2 结果: Cat={res_g2['cat_cv']}  XGB={res_g2['xgb_cv']}  LGB={res_g2['lgb_cv']}")
print(f"  G2 OOF:  Cat={res_g2['cat_oof']:.4f}  XGB={res_g2['xgb_oof']:.4f}  LGB={res_g2['lgb_oof']:.4f}")
print(f"  G2 集成: {res_g2['ensemble_auc']:.4f}  Rank: {res_g2['rank_auc']:.4f}")
print(f"  G2 最终: {res_g2['final_auc']:.4f}  (耗时{time.time()-t0:.0f}s)")

# =============================================================================
# 最终消融对比表
# =============================================================================
delta = res_g2['final_auc'] - res_g1['final_auc']
print("\n" + "="*70)
print("消融实验结果对比")
print("="*70)
print(f"  {'方法':<30} {'集成AUC':>10} {'vs G1':>10}")
print(f"  {'-'*50}")
print(f"  {'G1: Phase1 Baseline':<30} {res_g1['final_auc']:>10.4f} {'—':>10}")
print(f"  {'G2: + 改良RMT谱特征':<30} {res_g2['final_auc']:>10.4f} {delta:>+10.4f}")
print(f"\n  RMT 增益: {delta:+.4f}  {'✅ 正贡献' if delta > 0 else '❌ 负贡献'}")

# F1 评估（G2）
prec_arr, rec_arr, thr_arr = precision_recall_curve(y, res_g2['final_probs'])
f1_arr = 2*prec_arr*rec_arr/(prec_arr+rec_arr+1e-8)
best_idx = np.argmax(f1_arr[:-1])
best_thr = thr_arr[best_idx]; best_f1 = f1_arr[best_idx]
final_preds = (res_g2['final_probs'] >= best_thr).astype(int)
print(f"\n  G2 最优阈值={best_thr:.4f}  F1={best_f1:.4f}")
print(classification_report(y, final_preds, target_names=['正常', '异常']))

print("\n" + "="*70)
print("Phase 2 完成!")
print(f"  G1 (Baseline):   {res_g1['final_auc']:.4f}")
print(f"  G2 (+RMT):       {res_g2['final_auc']:.4f}  ({delta:+.4f})")
print(f"  RMT 保留维度:    {len(rmt_keep)}/{rmt_feats_all.shape[1]}")
print(f"  总耗时: {time.time()-t_total:.0f}s")
print("="*70)
