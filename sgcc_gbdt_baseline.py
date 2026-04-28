
# =============================================================================
# SGCC 用电数据异常检测 — Phase 1: 纯GBDT特征工程 Baseline
# 目标: 通过精细化特征工程 + CatBoost/XGBoost/LightGBM 5折CV集成
#       达到 AUC 0.84+，为后续 RMT/Transformer 增量验证提供强 baseline
# =============================================================================

import os, sys, time, warnings, gc

# 修复Windows GBK编码问题
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from scipy.stats import rankdata, skew, kurtosis
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, classification_report,
                              confusion_matrix, precision_recall_curve)

warnings.filterwarnings('ignore')
np.random.seed(42)

_CPU_THREADS = min(os.cpu_count() or 4, 8)

# =============================================================================
# Step 1: 加载原始数据
# =============================================================================
DATA_PATH = r'C:\Users\wb.zhoushujie\Desktop\data set.csv'

print("=" * 70)
print("Phase 1: 纯GBDT特征工程 Baseline")
print("=" * 70)

t_total = time.time()
df = pd.read_csv(DATA_PATH)
labels = df['FLAG'].values.astype(np.int64)
date_cols = [c for c in df.columns if c not in ['CONS_NO', 'FLAG']]
raw_vals = df[date_cols].values.astype(np.float64)  # (N, 1034) 原始日用电量含NaN

N_USERS, T_DAYS = raw_vals.shape
NEG_COUNT = int((labels == 0).sum())
POS_COUNT = int((labels == 1).sum())

print(f"  用户数={N_USERS}, 天数={T_DAYS}")
print(f"  正常={NEG_COUNT}({NEG_COUNT/len(labels)*100:.1f}%), "
      f"异常={POS_COUNT}({POS_COUNT/len(labels)*100:.1f}%)")

# 解析日期信息
dates = pd.to_datetime(date_cols, format='%m/%d/%Y')
day_of_week = dates.dayofweek.values  # 0=Mon, 6=Sun
month_of_year = dates.month.values
is_weekend = (day_of_week >= 5).astype(np.float32)

# =============================================================================
# Step 2: 缺失模式特征（在插值之前提取！）
# =============================================================================
print("\n--- Step 2: 缺失模式特征 ---")
t_step = time.time()

nan_mask = np.isnan(raw_vals)  # True = 缺失

# F_miss_1: 用户缺失率
miss_ratio = nan_mask.mean(axis=1).astype(np.float32)

# F_miss_2: 连续缺失最大段长度
max_consec_nan = np.zeros(N_USERS, dtype=np.float32)
cur_nan_run = np.zeros(N_USERS, dtype=np.float32)
for t in range(T_DAYS):
    cur_nan_run = (cur_nan_run + 1.0) * nan_mask[:, t]
    max_consec_nan = np.maximum(max_consec_nan, cur_nan_run)
max_consec_nan_ratio = max_consec_nan / T_DAYS

# F_miss_3: 前半段 vs 后半段缺失比
half_t = T_DAYS // 2
miss_first_half = nan_mask[:, :half_t].mean(axis=1).astype(np.float32)
miss_second_half = nan_mask[:, half_t:].mean(axis=1).astype(np.float32)
miss_half_diff = miss_second_half - miss_first_half

# F_miss_4: 缺失段数量（连续NaN段的个数）
nan_int = nan_mask.astype(np.int32)
nan_diff = np.diff(nan_int, axis=1, prepend=0)
miss_segment_count = (nan_diff == 1).sum(axis=1).astype(np.float32)

# F_miss_5: 零值比率（在原始数据上统计，NaN不算零）
zero_mask = (raw_vals == 0) & (~nan_mask)
zero_ratio = zero_mask.sum(axis=1).astype(np.float32) / (~nan_mask).sum(axis=1).clip(1).astype(np.float32)

# F_miss_6: 连续零值最大段长度
max_consec_zero = np.zeros(N_USERS, dtype=np.float32)
cur_zero_run = np.zeros(N_USERS, dtype=np.float32)
zero_int = zero_mask.astype(np.float32)
for t in range(T_DAYS):
    cur_zero_run = (cur_zero_run + 1.0) * zero_int[:, t]
    max_consec_zero = np.maximum(max_consec_zero, cur_zero_run)
max_consec_zero_ratio = max_consec_zero / T_DAYS

# F_miss_7: 缺失+零值联合比率（"无信号"天占比）
no_signal_ratio = (nan_mask | zero_mask).mean(axis=1).astype(np.float32)

miss_features = np.column_stack([
    miss_ratio, max_consec_nan_ratio,
    miss_first_half, miss_second_half, miss_half_diff,
    miss_segment_count,
    zero_ratio, max_consec_zero_ratio,
    no_signal_ratio,
]).astype(np.float32)

# 验证缺失特征区分度
for i, name in enumerate(['缺失率', '最大连续NaN比', '前半NaN', '后半NaN',
                            'NaN前后差', 'NaN段数', '零值率', '最大连续零比', '无信号率']):
    a = max(roc_auc_score(labels, miss_features[:, i]),
            1 - roc_auc_score(labels, miss_features[:, i]))
    if a > 0.52:
        print(f"  {name}: AUC={a:.4f} ✅")

print(f"  缺失模式特征: {miss_features.shape[1]}维, 耗时{time.time()-t_step:.1f}s")

# =============================================================================
# Step 3: 数据预处理（插值填充）
# =============================================================================
print("\n--- Step 3: 数据预处理 ---")

# 线性插值 → 均值填充（float32 节省内存，分块避免OOM）
X_raw = raw_vals.astype(np.float32)  # 直接转float32，节省一半内存
del raw_vals; gc.collect()

CHUNK = 5000
col_mean = np.nanmean(X_raw, axis=0)  # 全局列均值，用于最终兜底填充
col_mean = np.nan_to_num(col_mean, nan=0.0)

for i in range(0, N_USERS, CHUNK):
    chunk = X_raw[i:i+CHUNK]  # (chunk_size, T)
    # 逐行线性插值（pandas行插值，float32即可）
    df_chunk = pd.DataFrame(chunk)
    df_chunk = df_chunk.interpolate(method='linear', axis=1, limit_direction='both')
    # 仍有NaN的（全NaN行）用列均值填充
    df_chunk = df_chunk.fillna(pd.Series(col_mean))
    X_raw[i:i+CHUNK] = df_chunk.values.astype(np.float32)
    del df_chunk
    gc.collect()

print(f"  填充后范围: [{X_raw.min():.2f}, {X_raw.max():.2f}]")

# =============================================================================
# Step 4: 月度聚合 + 月度特征
# =============================================================================
print("\n--- Step 4: 月度聚合 + 月度特征 ---")
t_step = time.time()

DAYSPM = 30
NM = T_DAYS // DAYSPM  # 34个月

# 月度均值/std/max/min/零值比
mo_mean = np.zeros((N_USERS, NM), dtype=np.float32)
mo_std = np.zeros((N_USERS, NM), dtype=np.float32)
mo_max = np.zeros((N_USERS, NM), dtype=np.float32)
mo_min = np.zeros((N_USERS, NM), dtype=np.float32)
mo_zero = np.zeros((N_USERS, NM), dtype=np.float32)
mo_nan = np.zeros((N_USERS, NM), dtype=np.float32)  # 月度缺失率

for m in range(NM):
    s, e = m * DAYSPM, min((m + 1) * DAYSPM, T_DAYS)
    mo_mean[:, m] = X_raw[:, s:e].mean(axis=1)
    mo_std[:, m] = X_raw[:, s:e].std(axis=1)
    mo_max[:, m] = X_raw[:, s:e].max(axis=1)
    mo_min[:, m] = X_raw[:, s:e].min(axis=1)
    mo_zero[:, m] = (X_raw[:, s:e] == 0).mean(axis=1)
    mo_nan[:, m] = nan_mask[:, s:e].mean(axis=1)  # 原始NaN率

# 月度差分 & 二阶差分
mo_diff1 = np.diff(mo_mean, axis=1, prepend=mo_mean[:, :1])
mo_diff2 = np.diff(mo_diff1, axis=1, prepend=mo_diff1[:, :1])

# 基准偏离（前6个月为基准）
BASELINE_M = 6
baseline_mean = mo_mean[:, :BASELINE_M].mean(axis=1, keepdims=True) + 1e-3
mo_vs_base = (mo_mean - baseline_mean) / (np.abs(baseline_mean) + 1e-3)
mo_cumdev = np.cumsum(mo_mean - baseline_mean, axis=1)

# 跨用户百分位排名
mo_pct = np.zeros((N_USERS, NM), dtype=np.float32)
for m in range(NM):
    mo_pct[:, m] = (rankdata(mo_mean[:, m]) / N_USERS).astype(np.float32)

# 排名偏差 & 排名下降
mo_rank_mean = mo_pct.mean(axis=1, keepdims=True)
mo_rank_dev = mo_pct - mo_rank_mean
half_nm = NM // 2
rank_drop = mo_pct[:, :half_nm].mean(axis=1) - mo_pct[:, half_nm:].mean(axis=1)

# 用户自身归一化
user_median_raw = np.median(mo_mean, axis=1, keepdims=True) + 1e-3
mo_self_ratio = mo_mean / user_median_raw

# 对数偏离
mo_log_ratio = np.log1p(np.maximum(mo_mean, 0)) - np.log1p(np.maximum(baseline_mean, 0))

# 局部Z-score（3个月窗口）
mo_roll3_mean = np.zeros((N_USERS, NM), dtype=np.float32)
mo_roll3_std = np.zeros((N_USERS, NM), dtype=np.float32)
for m in range(NM):
    ws = max(0, m - 2)
    mo_roll3_mean[:, m] = mo_mean[:, ws:m + 1].mean(axis=1)
    mo_roll3_std[:, m] = mo_mean[:, ws:m + 1].std(axis=1) + 1e-6
mo_local_zscore = np.clip((mo_mean - mo_roll3_mean) / mo_roll3_std, -5, 5)

# 跨用户偏离度
mo_global_median = np.median(mo_mean, axis=0, keepdims=True) + 1e-3
mo_global_dev = np.log1p(np.maximum(mo_mean, 0)) - np.log1p(np.maximum(mo_global_median, 0))

print(f"  月度聚合: {NM}个月, 耗时{time.time()-t_step:.1f}s")

# =============================================================================
# Step 5: 周期性/日历特征
# =============================================================================
print("\n--- Step 5: 周期性/日历特征 ---")
t_step = time.time()

# 工作日 vs 周末用电比
weekday_mask = (day_of_week < 5)
weekend_mask = (day_of_week >= 5)
weekday_mean = np.nanmean(np.where(weekday_mask[np.newaxis, :], X_raw, np.nan), axis=1)
weekend_mean = np.nanmean(np.where(weekend_mask[np.newaxis, :], X_raw, np.nan), axis=1)
weekday_mean = np.nan_to_num(weekday_mean, nan=0.0).astype(np.float32)
weekend_mean = np.nan_to_num(weekend_mean, nan=0.0).astype(np.float32)
wd_we_ratio = weekday_mean / (weekend_mean + 1e-6)

# 工作日/周末用电差异的std（刻画稳定性）
wd_vals = np.where(weekday_mask[np.newaxis, :], X_raw, np.nan)
we_vals = np.where(weekend_mask[np.newaxis, :], X_raw, np.nan)
wd_std = np.nan_to_num(np.nanstd(wd_vals, axis=1), nan=0.0).astype(np.float32)
we_std = np.nan_to_num(np.nanstd(we_vals, axis=1), nan=0.0).astype(np.float32)

# 月度季节性（按实际月份聚合，而非固定30天窗口）
monthly_avg_by_cal = np.zeros((N_USERS, 12), dtype=np.float32)
for m_cal in range(12):
    m_mask = (month_of_year == m_cal + 1)
    if m_mask.sum() > 0:
        monthly_avg_by_cal[:, m_cal] = np.nanmean(
            np.where(m_mask[np.newaxis, :], X_raw, np.nan), axis=1)
        monthly_avg_by_cal[:, m_cal] = np.nan_to_num(monthly_avg_by_cal[:, m_cal], nan=0.0)

# 夏季(6-8) vs 冬季(12-2) 用电比
summer_months = [6, 7, 8]
winter_months = [12, 1, 2]
summer_avg = monthly_avg_by_cal[:, [m-1 for m in summer_months]].mean(axis=1)
winter_avg = monthly_avg_by_cal[:, [m-1 for m in winter_months]].mean(axis=1)
summer_winter_ratio = summer_avg / (winter_avg + 1e-6)

# 季节性偏离: 每月与全年均值的偏离标准差（越大越有季节波动）
yearly_avg = monthly_avg_by_cal.mean(axis=1, keepdims=True) + 1e-6
seasonality_strength = (monthly_avg_by_cal / yearly_avg).std(axis=1)

# 工作日/周末的异常变异度（月度粒度）
mo_wd_mean = np.zeros((N_USERS, NM), dtype=np.float32)
mo_we_mean = np.zeros((N_USERS, NM), dtype=np.float32)
for m in range(NM):
    s, e = m * DAYSPM, min((m + 1) * DAYSPM, T_DAYS)
    wd_m = weekday_mask[s:e]
    we_m = weekend_mask[s:e]
    if wd_m.sum() > 0:
        mo_wd_mean[:, m] = X_raw[:, s:e][:, wd_m].mean(axis=1)
    if we_m.sum() > 0:
        mo_we_mean[:, m] = X_raw[:, s:e][:, we_m].mean(axis=1)
mo_wd_we_ratio = mo_wd_mean / (mo_we_mean + 1e-6)

calendar_features = np.column_stack([
    wd_we_ratio, weekday_mean, weekend_mean, wd_std, we_std,
    summer_winter_ratio, seasonality_strength,
]).astype(np.float32)

# 验证日历特征区分度
for i, name in enumerate(['工作日/周末比', '工作日均值', '周末均值', '工作日std',
                            '周末std', '夏/冬比', '季节强度']):
    a = max(roc_auc_score(labels, calendar_features[:, i]),
            1 - roc_auc_score(labels, calendar_features[:, i]))
    if a > 0.52:
        print(f"  {name}: AUC={a:.4f} ✅")

print(f"  日历特征: {calendar_features.shape[1]}维, 耗时{time.time()-t_step:.1f}s")

# =============================================================================
# Step 6: 深层统计特征
# =============================================================================
print("\n--- Step 6: 深层统计特征 ---")
t_step = time.time()

# 全局用户统计
user_mean = X_raw.mean(axis=1)
user_std = X_raw.std(axis=1)
user_median = np.median(X_raw, axis=1)
user_max = X_raw.max(axis=1)
user_min = X_raw.min(axis=1)
user_q25 = np.percentile(X_raw, 25, axis=1)
user_q75 = np.percentile(X_raw, 75, axis=1)
user_q10 = np.percentile(X_raw, 10, axis=1)
user_q90 = np.percentile(X_raw, 90, axis=1)
user_iqr = user_q75 - user_q25
user_range = user_max - user_min
user_cv = user_std / (user_mean + 1e-6)  # 变异系数

# 偏度 & 峰度
user_skew = skew(X_raw, axis=1, nan_policy='omit').astype(np.float32)
user_kurt = kurtosis(X_raw, axis=1, nan_policy='omit').astype(np.float32)
user_skew = np.nan_to_num(user_skew, nan=0.0)
user_kurt = np.nan_to_num(user_kurt, nan=0.0)

# 极端值频率：超过 3*IQR 的天数比
extreme_high = X_raw > (user_q75 + 3 * user_iqr)[:, np.newaxis]
extreme_low = X_raw < np.maximum(user_q25 - 3 * user_iqr, 0)[:, np.newaxis]
extreme_high_ratio = extreme_high.mean(axis=1).astype(np.float32)
extreme_low_ratio = extreme_low.mean(axis=1).astype(np.float32)

# 趋势特征（全序列线性回归斜率）
t_vec = np.arange(T_DAYS, dtype=np.float64)
t_mean_v = t_vec.mean()
t_var = ((t_vec - t_mean_v) ** 2).sum()
X_f64 = X_raw.astype(np.float64)
slope_all = ((X_f64 - X_f64.mean(axis=1, keepdims=True)) *
             (t_vec[np.newaxis, :] - t_mean_v)).sum(axis=1) / (t_var + 1e-8)

# 末期/前期比（时序趋势强信号）
tail_head_ratio = (mo_mean[:, -6:].mean(axis=1) /
                   (mo_mean[:, :6].mean(axis=1) + 1e-6)).astype(np.float32)

# 连续N月消费骤降（月环比<0.7）
mo_ratio = mo_mean[:, 1:] / (mo_mean[:, :-1] + 1e-6)
drop_mask = (mo_ratio < 0.7).astype(np.int32)
max_consec_drop = np.zeros(N_USERS, dtype=np.float32)
cur_drop_run = np.zeros(N_USERS, dtype=np.float32)
for t in range(drop_mask.shape[1]):
    cur_drop_run = (cur_drop_run + 1.0) * drop_mask[:, t]
    max_consec_drop = np.maximum(max_consec_drop, cur_drop_run)

# 连续递减月数
decr_mask = (np.diff(mo_mean, axis=1) < 0).astype(np.int32)
max_consec_decr = np.zeros(N_USERS, dtype=np.float32)
cur_decr_run = np.zeros(N_USERS, dtype=np.float32)
for t in range(decr_mask.shape[1]):
    cur_decr_run = (cur_decr_run + 1.0) * decr_mask[:, t]
    max_consec_decr = np.maximum(max_consec_decr, cur_decr_run)

# 分位数变化轨迹: Q25/Q75 在前后半段的变化
mo_q25 = np.zeros((N_USERS, NM), dtype=np.float32)
mo_q75 = np.zeros((N_USERS, NM), dtype=np.float32)
for m in range(NM):
    s, e = m * DAYSPM, min((m + 1) * DAYSPM, T_DAYS)
    mo_q25[:, m] = np.percentile(X_raw[:, s:e], 25, axis=1)
    mo_q75[:, m] = np.percentile(X_raw[:, s:e], 75, axis=1)
q25_trend = mo_q25[:, half_nm:].mean(axis=1) - mo_q25[:, :half_nm].mean(axis=1)
q75_trend = mo_q75[:, half_nm:].mean(axis=1) - mo_q75[:, :half_nm].mean(axis=1)

# 波动率变化趋势: 后半段std vs 前半段std
std_trend = mo_std[:, half_nm:].mean(axis=1) / (mo_std[:, :half_nm].mean(axis=1) + 1e-6)

# 月度变异系数变化
mo_cv = mo_std / (mo_mean + 1e-6)
cv_trend = mo_cv[:, half_nm:].mean(axis=1) - mo_cv[:, :half_nm].mean(axis=1)

deep_stat_features = np.column_stack([
    user_mean, user_std, user_median, user_max, user_min,
    user_q10, user_q25, user_q75, user_q90,
    user_iqr, user_range, user_cv,
    user_skew, user_kurt,
    extreme_high_ratio, extreme_low_ratio,
    slope_all.astype(np.float32),
    tail_head_ratio, max_consec_drop, max_consec_decr,
    q25_trend, q75_trend, std_trend, cv_trend,
]).astype(np.float32)

for i, name in enumerate(['均值', 'std', '中位数', '最大值', '最小值',
                            'Q10', 'Q25', 'Q75', 'Q90',
                            'IQR', '极差', 'CV',
                            '偏度', '峰度',
                            '高极端频率', '低极端频率',
                            '趋势斜率', '末/前比', '连续骤降', '连续递减',
                            'Q25趋势', 'Q75趋势', 'std趋势', 'CV趋势']):
    a = max(roc_auc_score(labels, deep_stat_features[:, i]),
            1 - roc_auc_score(labels, deep_stat_features[:, i]))
    if a > 0.55:
        print(f"  {name}: AUC={a:.4f} ✅")

print(f"  深层统计特征: {deep_stat_features.shape[1]}维, 耗时{time.time()-t_step:.1f}s")

# =============================================================================
# Step 7: 多尺度变点检测特征 (TCN-CPD)（复用原版有效逻辑）
# =============================================================================
print("\n--- Step 7: 多尺度变点检测特征 (TCN-CPD) ---")
t_step = time.time()

# 用户级 RobustScaler 归一化
_tcn_input = X_raw.copy().astype(np.float32)
_u_med = np.median(_tcn_input, axis=1, keepdims=True)
_u_iqr = np.clip(
    np.percentile(_tcn_input, 75, axis=1, keepdims=True) -
    np.percentile(_tcn_input, 25, axis=1, keepdims=True),
    1e-6, None)
_tcn_input = np.clip((_tcn_input - _u_med) / _u_iqr, -10, 10)
_T_tcn = min(_tcn_input.shape[1], 1020)
_tcn_input = _tcn_input[:, :_T_tcn]

_scales = [7, 30, 90]
ms_cpd_features = []

for _scale in _scales:
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

tcn_cpd_features = np.concatenate(ms_cpd_features, axis=1).astype(np.float32)
_cross_short_long = ms_cpd_features[0][:, 4] / (ms_cpd_features[2][:, 4] + 1e-8)
_cross_jump_consistency = np.abs(ms_cpd_features[0][:, 0] - ms_cpd_features[2][:, 0])
_cross_feats = np.column_stack([_cross_short_long, _cross_jump_consistency]).astype(np.float32)
tcn_cpd_features = np.concatenate([tcn_cpd_features, _cross_feats], axis=1)

print(f"  TCN-CPD特征: {tcn_cpd_features.shape[1]}维, 耗时{time.time()-t_step:.1f}s")

# =============================================================================
# Step 8: ISCT 变点特征（复用原版有效逻辑，但不含分层RMT）
# =============================================================================
print("\n--- Step 8: ISCT 变点特征 ---")
t_step = time.time()

# 按用户月均消费分层（用于层内偏离计算）
user_avg = X_raw.mean(axis=1)
n_strata = 8
strata_labels = pd.qcut(user_avg, q=n_strata, labels=False, duplicates='drop')

# 层内中位数偏离
isct_dev_monthly = np.zeros((N_USERS, NM), dtype=np.float32)
for k_s in range(int(strata_labels.max()) + 1):
    mask = (strata_labels == k_s)
    if mask.sum() < 2:
        continue
    layer_median = np.median(mo_mean[mask], axis=0, keepdims=True)
    isct_dev_monthly[mask] = ((mo_mean[mask] - layer_median) /
                               (np.abs(layer_median) + 1e-3)).astype(np.float32)

# ISCT-CPD: 8维变点检测特征
sig_all = isct_dev_monthly
T_sig = sig_all.shape[1]

isct_cpd_feats = np.zeros((N_USERS, 8), dtype=np.float32)

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

# F3-F8: 与原版一致
isct_cpd_feats[:, 2] = np.abs(sig_all).max(axis=1).astype(np.float32)

half_sig = T_sig // 2
isct_cpd_feats[:, 3] = (sig_all[:, half_sig:].mean(axis=1) /
                         (np.abs(sig_all[:, :half_sig].mean(axis=1)) + 1e-6)).astype(np.float32)
isct_cpd_feats[:, 4] = sig_all.std(axis=1).astype(np.float32)

neg_mask_all = (sig_all < 0).astype(np.int32)
max_neg_run = np.zeros(N_USERS, dtype=np.float32)
cur_run = np.zeros(N_USERS, dtype=np.float32)
for t_s in range(T_sig):
    cur_run = (cur_run + 1.0) * neg_mask_all[:, t_s]
    max_neg_run = np.maximum(max_neg_run, cur_run)
isct_cpd_feats[:, 5] = (max_neg_run / T_sig).astype(np.float32)

t_vec_sig = np.arange(T_sig, dtype=np.float64)
t_mean_sig = t_vec_sig.mean()
t_var_sig = ((t_vec_sig - t_mean_sig) ** 2).sum()
sig_f64 = sig_all.astype(np.float64)
slope_sig = ((sig_f64 - sig_f64.mean(axis=1, keepdims=True)) *
             (t_vec_sig[np.newaxis, :] - t_mean_sig)).sum(axis=1) / (t_var_sig + 1e-8)
isct_cpd_feats[:, 6] = slope_sig.astype(np.float32)
isct_cpd_feats[:, 7] = sig_all[:, -6:].mean(axis=1).astype(np.float32)

# 特征筛选: 去掉 |AUC-0.5| <= 0.02 的噪声维度
isc_keep = []
for ci in range(isct_cpd_feats.shape[1]):
    a = roc_auc_score(labels, isct_cpd_feats[:, ci])
    if abs(a - 0.5) > 0.02:
        isc_keep.append(ci)
if len(isc_keep) > 0:
    isct_cpd_feats = isct_cpd_feats[:, isc_keep]

print(f"  ISCT-CPD特征: {isct_cpd_feats.shape[1]}维, 耗时{time.time()-t_step:.1f}s")

# =============================================================================
# Step 9: 月度序列统计特征（8组 × FEAT_DIM）
# =============================================================================
print("\n--- Step 9: 月度序列统计特征 ---")

def _scale_ch_per_user(arr2d, clip=5.0):
    med = np.median(arr2d, axis=1, keepdims=True)
    q1 = np.percentile(arr2d, 25, axis=1, keepdims=True)
    q3 = np.percentile(arr2d, 75, axis=1, keepdims=True)
    iqr = q3 - q1 + 1e-6
    return np.clip((arr2d - med) / iqr, -clip, clip).astype(np.float32)

def _scale_ch(arr2d, clip=5.0):
    flat = arr2d.reshape(-1, 1)
    flat = RobustScaler().fit_transform(flat).reshape(arr2d.shape)
    return np.clip(flat, -clip, clip).astype(np.float32)

# 拼装月度多通道（用于统计特征提取）
rank_tile = np.tile(rank_drop[:, np.newaxis], (1, NM))
mo_ch = np.stack([
    _scale_ch_per_user(mo_mean),        # ch0
    _scale_ch_per_user(mo_std),         # ch1
    _scale_ch_per_user(mo_max),         # ch2
    mo_zero,                             # ch3
    mo_pct,                              # ch4
    _scale_ch(mo_rank_dev),             # ch5
    _scale_ch(rank_tile),               # ch6
    _scale_ch_per_user(mo_vs_base),     # ch7
    _scale_ch_per_user(mo_cumdev),      # ch8
    _scale_ch(mo_log_ratio),            # ch9
    _scale_ch_per_user(mo_diff1),       # ch10
    _scale_ch(mo_diff2),                # ch11
    mo_self_ratio.astype(np.float32),   # ch12
    np.clip(mo_local_zscore, -5, 5).astype(np.float32),  # ch13
    _scale_ch(mo_global_dev),           # ch14
    _scale_ch_per_user(mo_mean),        # ch15 (替代原ISCT通道)
    isct_dev_monthly.astype(np.float32),  # ch16
], axis=2)  # (N, NM, 17)

FEAT_DIM = mo_ch.shape[2]
_HALF = NM // 2

# 8组统计特征
hand_feats = np.concatenate([
    mo_ch.mean(axis=1),                                                    # 均值
    mo_ch.std(axis=1),                                                     # std
    mo_ch.max(axis=1),                                                     # max
    mo_ch.min(axis=1),                                                     # min
    np.percentile(mo_ch, 25, axis=1),                                      # Q1
    np.percentile(mo_ch, 75, axis=1),                                      # Q3
    mo_ch[:, _HALF:, :].mean(axis=1) - mo_ch[:, :_HALF, :].mean(axis=1),  # 后-前趋势
    mo_ch.argmax(axis=1).astype(np.float32) / NM,                         # 最大值位置
], axis=1)

# 标量特征
_scalar_feats = mo_ch.mean(axis=1).astype(np.float32)

print(f"  手工统计特征: {hand_feats.shape[1]}维")

# =============================================================================
# Step 10: 多视角原始月度信号
# =============================================================================
xgb_self_norm = _scale_ch_per_user(mo_mean)
xgb_diff_norm = _scale_ch_per_user(np.diff(mo_mean, axis=1, prepend=mo_mean[:, :1]))
xgb_local_z = np.clip(mo_local_zscore, -5, 5)

# =============================================================================
# Step 11: 拼合所有特征 → GBDT 输入
# =============================================================================
print("\n--- Step 11: 特征拼合 ---")

X_all = np.concatenate([
    hand_feats,            # 月度序列统计 (FEAT_DIM*8)
    miss_features,         # 缺失模式 (9维) ← NEW
    calendar_features,     # 日历/周期性 (7维) ← NEW
    deep_stat_features,    # 深层统计 (24维) ← NEW
    isct_cpd_feats,        # ISCT变点 (~8维)
    mo_mean,               # 原始月消费 (34维)
    xgb_self_norm,         # 用户自身归一化 (34维)
    xgb_diff_norm,         # 差分 (34维)
    mo_vs_base,            # 基准偏离 (34维)
    mo_pct,                # 排名 (34维)
    xgb_local_z,           # 局部Z (34维)
    tcn_cpd_features,      # TCN-CPD (20维)
    _scalar_feats,         # 标量 (17维)
    monthly_avg_by_cal,    # 日历月均消费 (12维) ← NEW
    mo_wd_we_ratio,        # 月度工作日/周末比 (34维) ← NEW
    mo_nan,                # 月度缺失率 (34维) ← NEW
    mo_cv,                 # 月度变异系数 (34维) ← NEW
], axis=1).astype(np.float32)

y = labels

print(f"  总特征维度: {X_all.shape}")
print(f"  [NEW] 缺失模式={miss_features.shape[1]}, 日历={calendar_features.shape[1]}, "
      f"深层统计={deep_stat_features.shape[1]}, 日历月均={monthly_avg_by_cal.shape[1]}, "
      f"月度工作日/周末比={mo_wd_we_ratio.shape[1]}, 月度NaN={mo_nan.shape[1]}, "
      f"月度CV={mo_cv.shape[1]}")

# NaN/Inf 安全处理
X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

# =============================================================================
# Step 12: 5折 StratifiedKFold CV — CatBoost + XGBoost + LightGBM 集成
# =============================================================================
print("\n" + "=" * 70)
print("Step 12: 5折CV — CatBoost + XGBoost + LightGBM 集成")
print("=" * 70)

import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_cat = np.zeros(N_USERS, dtype=np.float64)
oof_xgb = np.zeros(N_USERS, dtype=np.float64)
oof_lgb = np.zeros(N_USERS, dtype=np.float64)

fold_aucs_cat = []
fold_aucs_xgb = []
fold_aucs_lgb = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y)):
    print(f"\n  === Fold {fold+1}/{N_FOLDS} ===")
    X_tr, X_va = X_all[tr_idx], X_all[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    neg_c = (y_tr == 0).sum()
    pos_c = (y_tr == 1).sum()
    spw = neg_c / max(pos_c, 1)

    # ---- CatBoost ----
    t0 = time.time()
    cat_model = CatBoostClassifier(
        iterations=3000, depth=7, learning_rate=0.01,
        l2_leaf_reg=3.0, border_count=128, random_strength=1.5,
        bagging_temperature=0.8, auto_class_weights='Balanced',
        eval_metric='AUC', random_seed=42 + fold, verbose=0,
        early_stopping_rounds=100, task_type='CPU')
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
    cat_probs = cat_model.predict_proba(X_va)[:, 1]
    oof_cat[va_idx] = cat_probs
    cat_auc = roc_auc_score(y_va, cat_probs)
    fold_aucs_cat.append(cat_auc)
    print(f"    CatBoost  AUC={cat_auc:.4f}  iters={cat_model.best_iteration_}  ({time.time()-t0:.0f}s)")

    # ---- XGBoost ----
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, colsample_bylevel=0.8,
        min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=spw,
        eval_metric='auc', random_state=42 + fold, n_jobs=-1,
        early_stopping_rounds=30)
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=0)
    xgb_probs = xgb_model.predict_proba(X_va)[:, 1]
    oof_xgb[va_idx] = xgb_probs
    xgb_auc = roc_auc_score(y_va, xgb_probs)
    fold_aucs_xgb.append(xgb_auc)
    print(f"    XGBoost   AUC={xgb_auc:.4f}  ({time.time()-t0:.0f}s)")

    # ---- LightGBM ----
    t0 = time.time()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000, max_depth=-1, num_leaves=63,
        learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=spw,
        metric='auc', random_state=42 + fold, n_jobs=-1,
        verbose=-1)
    lgb_model.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    lgb_probs = lgb_model.predict_proba(X_va)[:, 1]
    oof_lgb[va_idx] = lgb_probs
    lgb_auc = roc_auc_score(y_va, lgb_probs)
    fold_aucs_lgb.append(lgb_auc)
    print(f"    LightGBM  AUC={lgb_auc:.4f}  ({time.time()-t0:.0f}s)")

    gc.collect()

# =============================================================================
# Step 13: OOF 集成评估
# =============================================================================
print("\n" + "=" * 70)
print("Step 13: OOF 集成评估")
print("=" * 70)

print(f"\n  CatBoost  CV AUC: {np.mean(fold_aucs_cat):.4f} ± {np.std(fold_aucs_cat):.4f}  "
      f"(folds: {[f'{a:.4f}' for a in fold_aucs_cat]})")
print(f"  XGBoost   CV AUC: {np.mean(fold_aucs_xgb):.4f} ± {np.std(fold_aucs_xgb):.4f}  "
      f"(folds: {[f'{a:.4f}' for a in fold_aucs_xgb]})")
print(f"  LightGBM  CV AUC: {np.mean(fold_aucs_lgb):.4f} ± {np.std(fold_aucs_lgb):.4f}  "
      f"(folds: {[f'{a:.4f}' for a in fold_aucs_lgb]})")

# OOF 全量 AUC
cat_oof_auc = roc_auc_score(y, oof_cat)
xgb_oof_auc = roc_auc_score(y, oof_xgb)
lgb_oof_auc = roc_auc_score(y, oof_lgb)
print(f"\n  OOF全量 CatBoost AUC: {cat_oof_auc:.4f}")
print(f"  OOF全量 XGBoost  AUC: {xgb_oof_auc:.4f}")
print(f"  OOF全量 LightGBM AUC: {lgb_oof_auc:.4f}")

# 网格搜索三模型权重
print("\n  [权重搜索] CatBoost + XGBoost + LightGBM ...")
best_w3, best_ens_auc = (0.33, 0.33, 0.34), 0.0
for w_c in np.arange(0.0, 1.01, 0.05):
    for w_x in np.arange(0.0, 1.01 - w_c, 0.05):
        w_l = 1.0 - w_c - w_x
        if w_l < -1e-9:
            continue
        blend = w_c * oof_cat + w_x * oof_xgb + w_l * oof_lgb
        auc_b = roc_auc_score(y, blend)
        if auc_b > best_ens_auc:
            best_ens_auc = auc_b
            best_w3 = (w_c, w_x, w_l)

ensemble_probs = best_w3[0] * oof_cat + best_w3[1] * oof_xgb + best_w3[2] * oof_lgb
ensemble_auc = roc_auc_score(y, ensemble_probs)

print(f"  最优权重: Cat={best_w3[0]:.2f}, XGB={best_w3[1]:.2f}, LGB={best_w3[2]:.2f}")
print(f"  加权融合 OOF AUC: {ensemble_auc:.4f}")

# Rank Average
rank_cat = rankdata(oof_cat) / N_USERS
rank_xgb = rankdata(oof_xgb) / N_USERS
rank_lgb = rankdata(oof_lgb) / N_USERS
rank_avg = (rank_cat + rank_xgb + rank_lgb) / 3.0
rank_avg_auc = roc_auc_score(y, rank_avg)
print(f"  Rank Average OOF AUC: {rank_avg_auc:.4f}")

final_probs = ensemble_probs if ensemble_auc >= rank_avg_auc else rank_avg
final_auc = max(ensemble_auc, rank_avg_auc)
final_method = "加权融合" if ensemble_auc >= rank_avg_auc else "Rank Average"
print(f"  最终选择: {final_method} AUC={final_auc:.4f}")

# F1 最优阈值
prec_arr, rec_arr, thr_arr = precision_recall_curve(y, final_probs)
f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-8)
best_pr_idx = np.argmax(f1_arr[:-1])
best_thr = thr_arr[best_pr_idx]
best_f1 = f1_arr[best_pr_idx]

final_preds = (final_probs >= best_thr).astype(int)
cm = confusion_matrix(y, final_preds)
tn, fp, fn, tp = cm.ravel()

print(f"\n  最优阈值: {best_thr:.4f}  F1={best_f1:.4f}")
print(f"  混淆矩阵: TN={tn} FP={fp} FN={fn} TP={tp}")
print(classification_report(y, final_preds, target_names=['正常', '异常']))

# =============================================================================
# Step 14: 特征重要性 Top20
# =============================================================================
print("\n--- 特征重要性 Top20 (最后一折 CatBoost) ---")
try:
    importances = cat_model.get_feature_importance()
    top_idx = np.argsort(importances)[::-1][:20]
    for rank, idx in enumerate(top_idx):
        print(f"  {rank+1:2d}. Feature[{idx:3d}] importance={importances[idx]:.2f}")
except Exception:
    pass

# =============================================================================
# 完成
# =============================================================================
print("\n" + "=" * 70)
print("Phase 1 完成!")
print(f"  总特征维度: {X_all.shape[1]}")
print(f"  CatBoost  CV AUC: {np.mean(fold_aucs_cat):.4f} ± {np.std(fold_aucs_cat):.4f}")
print(f"  XGBoost   CV AUC: {np.mean(fold_aucs_xgb):.4f} ± {np.std(fold_aucs_xgb):.4f}")
print(f"  LightGBM  CV AUC: {np.mean(fold_aucs_lgb):.4f} ± {np.std(fold_aucs_lgb):.4f}")
print(f"  集成 OOF AUC: {final_auc:.4f}  F1={best_f1:.4f}")
print(f"  总耗时: {time.time()-t_total:.0f}s")
print("=" * 70)
