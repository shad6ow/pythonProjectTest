
# =============================================================================
# RMT 候选特征诊断脚本（不跑CV，快速验证）
# 目标：找出真正有新信息的 RMT 特征设计
# 判断标准：
#   AUC > 0.58（有独立区分度）
#   与 baseline 最强特征的 Pearson 相关性 < 0.6（不与已有特征重复）
# 预计运行时间：< 5 分钟
# =============================================================================

import time, warnings, gc
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_PATH = r'C:\Users\wb.zhoushujie\Desktop\data set.csv'

print("=" * 70)
print("RMT 候选特征诊断（快速验证，不跑CV）")
print("=" * 70)
t0 = time.time()

# =============================================================================
# 1. 加载数据 + 基础预处理（复用 Phase2 逻辑）
# =============================================================================
print("\n[1] 加载数据...")
df = pd.read_csv(DATA_PATH, dtype={c: 'float32' for c in
                  pd.read_csv(DATA_PATH, nrows=0).columns
                  if c not in ['CONS_NO', 'FLAG']})
y = df['FLAG'].values.astype(np.int64)
date_cols = [c for c in df.columns if c not in ['CONS_NO', 'FLAG']]
raw_vals = df[date_cols].values.astype(np.float32)
del df; gc.collect()
N_USERS, T_DAYS = raw_vals.shape

dates = pd.to_datetime(date_cols, format='%m/%d/%Y')
day_of_week = dates.dayofweek.values

# 缺失mask（插值前）
nan_mask = np.isnan(raw_vals)

# 插值填充（分块）
CHUNK = 5000
col_mean = np.nan_to_num(np.nanmean(raw_vals, axis=0), nan=0.0)
for i in range(0, N_USERS, CHUNK):
    chunk = raw_vals[i:i+CHUNK]
    df_c = pd.DataFrame(chunk)
    df_c = df_c.interpolate(method='linear', axis=1, limit_direction='both')
    df_c = df_c.fillna(pd.Series(col_mean))
    raw_vals[i:i+CHUNK] = df_c.values.astype(np.float32)
    del df_c; gc.collect()
X = raw_vals; del raw_vals; gc.collect()

# 月度聚合
DAYSPM = 30
NM = T_DAYS // DAYSPM
mo_mean = np.zeros((N_USERS, NM), dtype=np.float32)
mo_std  = np.zeros((N_USERS, NM), dtype=np.float32)
for m in range(NM):
    s, e = m*DAYSPM, min((m+1)*DAYSPM, T_DAYS)
    mo_mean[:, m] = X[:, s:e].mean(axis=1)
    mo_std[:, m]  = X[:, s:e].std(axis=1)

user_avg = X.mean(axis=1)
user_std = X.std(axis=1)
user_max = X.max(axis=1)
mo_pct = np.zeros((N_USERS, NM), dtype=np.float32)
for m in range(NM):
    mo_pct[:, m] = (rankdata(mo_mean[:, m]) / N_USERS).astype(np.float32)
half_nm = NM // 2
rank_drop = mo_pct[:, :half_nm].mean(axis=1) - mo_pct[:, half_nm:].mean(axis=1)

print(f"  数据加载+预处理完成: {time.time()-t0:.0f}s")

# =============================================================================
# 2. 定义"判断是否有价值"的工具函数
# =============================================================================

# baseline 的代表性强特征（单维 AUC 最高的几个）
baseline_key_feats = {
    'user_std':   user_std,
    'user_max':   user_max,
    'user_avg':   user_avg,
    'rank_drop':  rank_drop,
    'mo_pct_mean': mo_pct.mean(axis=1),
}

def evaluate_feature(name, feat_vec, threshold_auc=0.55, threshold_corr=0.65):
    """
    评估一个候选特征是否值得加入模型
    返回: (auc, max_corr_with_baseline, is_useful)
    """
    feat_vec = np.nan_to_num(feat_vec.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if feat_vec.std() < 1e-8:
        return 0.5, 1.0, False, "全零/常数"

    raw_auc = roc_auc_score(y, feat_vec)
    auc = max(raw_auc, 1 - raw_auc)

    # 与 baseline 关键特征的最大相关性
    max_corr = 0.0
    max_corr_name = ""
    for bname, bfeat in baseline_key_feats.items():
        bfeat = np.nan_to_num(bfeat.astype(np.float64), nan=0.0)
        corr = abs(np.corrcoef(feat_vec, bfeat)[0, 1])
        if corr > max_corr:
            max_corr = corr
            max_corr_name = bname

    is_useful = (auc > threshold_auc) and (max_corr < threshold_corr)
    reason = ""
    if auc <= threshold_auc:
        reason = f"AUC={auc:.4f} 不足"
    elif max_corr >= threshold_corr:
        reason = f"与{max_corr_name}相关性={max_corr:.3f} 太高"
    else:
        reason = "✅ 有新信息"

    return auc, max_corr, is_useful, reason

print("\n" + "=" * 70)
print("候选 RMT 特征评估")
print(f"  判断标准: AUC > 0.55 且 与baseline最强特征相关性 < 0.65")
print("=" * 70)

results = []  # (name, auc, max_corr, is_useful, reason)

# =============================================================================
# 3. 候选设计 A：三段时序局部RMT（前/中/后各11-12个月）
# 核心理念：偷电发生在某个局部时间段，全局协方差会稀释信号
# 提取：每段的噪声子空间能量、信号/噪声比变化
# =============================================================================
print("\n--- 候选设计 A: 三段时序局部 RMT ---")

def _mp_upper(n, p, sigma2=1.0):
    ratio = p / max(n, p+1)
    return sigma2 * (1 + np.sqrt(ratio))**2

def _rmt_segment_features(mo_seg, n_strata=10):
    """
    对一段月度序列做分层RMT，提取每用户的：
    - noise_energy: 投影到噪声子空间的能量（越高越偏离正常模式）
    - signal_ratio: 信号子空间能量占比
    - mahal_noise: 噪声子空间内的马氏距离（信号干净的度量）
    返回 shape=(N_USERS, 3)
    """
    n_users, n_months = mo_seg.shape
    # 用户级归一化
    med = np.median(mo_seg, axis=1, keepdims=True)
    iqr = np.clip(np.percentile(mo_seg, 75, axis=1, keepdims=True) -
                  np.percentile(mo_seg, 25, axis=1, keepdims=True), 1e-6, None)
    mo_norm = np.clip((mo_seg - med) / iqr, -5, 5).astype(np.float64)

    noise_energy   = np.zeros(n_users, dtype=np.float32)
    signal_ratio   = np.zeros(n_users, dtype=np.float32)
    mahal_noise    = np.zeros(n_users, dtype=np.float32)

    strata = np.array(pd.qcut(mo_seg.mean(axis=1), q=n_strata,
                               labels=False, duplicates='drop'))

    for k in range(int(np.nanmax(strata)) + 1):
        mask = (strata == k)
        if mask.sum() < n_months + 5:
            continue
        X_k = mo_norm[mask]
        X_kc = X_k - X_k.mean(axis=0, keepdims=True)
        cov_k = (X_kc.T @ X_kc) / (len(X_k) - 1)

        try:
            eigvals, eigvecs = np.linalg.eigh(cov_k)
        except np.linalg.LinAlgError:
            continue

        eigvals = np.maximum(eigvals, 0)
        sigma2_est = np.median(eigvals[eigvals > 0]) if (eigvals > 0).sum() > 0 else 1.0
        mp_up = _mp_upper(len(X_k), n_months, sigma2=sigma2_est)

        signal_idx = eigvals > mp_up
        noise_idx  = ~signal_idx

        V_sig  = eigvecs[:, signal_idx]   # (p, n_sig)
        V_noi  = eigvecs[:, noise_idx]    # (p, n_noi)

        X_k_all = mo_norm[mask]
        e_sig  = (X_k_all @ V_sig)**2 if V_sig.shape[1] > 0 else np.zeros((len(X_k_all), 1))
        e_noi  = (X_k_all @ V_noi)**2 if V_noi.shape[1] > 0 else np.zeros((len(X_k_all), 1))

        e_sig_sum = e_sig.sum(axis=1).astype(np.float32)
        e_noi_sum = e_noi.sum(axis=1).astype(np.float32)
        total = e_sig_sum + e_noi_sum + 1e-8

        noise_energy[mask] = np.clip(e_noi_sum, 0, np.percentile(e_noi_sum, 99))
        signal_ratio[mask] = e_sig_sum / total

        # 噪声子空间内马氏距离
        if V_noi.shape[1] > 1:
            proj_noi = X_k_all @ V_noi  # (n, n_noi)
            noi_cov = np.cov(proj_noi.T) + np.eye(V_noi.shape[1]) * 1e-4
            try:
                noi_cov_inv = np.linalg.pinv(noi_cov, rcond=1e-4)
                md = np.diag(proj_noi @ noi_cov_inv @ proj_noi.T)
                mahal_noise[mask] = np.clip(md, 0, np.percentile(md, 99)).astype(np.float32)
            except Exception:
                pass

    return np.column_stack([noise_energy, signal_ratio, mahal_noise])

# 三段划分
seg_size = NM // 3
segs = {
    'early':  mo_mean[:, :seg_size],
    'mid':    mo_mean[:, seg_size:2*seg_size],
    'late':   mo_mean[:, 2*seg_size:],
}

seg_feats = {}
for seg_name, seg_data in segs.items():
    t_seg = time.time()
    feats = _rmt_segment_features(seg_data, n_strata=10)
    seg_feats[seg_name] = feats
    for i, fname in enumerate(['noise_energy', 'signal_ratio', 'mahal_noise']):
        full_name = f"A_{seg_name}_{fname}"
        auc, corr, useful, reason = evaluate_feature(full_name, feats[:, i])
        results.append((full_name, auc, corr, useful, reason))
        status = "✅" if useful else "❌"
        print(f"  {status} {full_name:<35} AUC={auc:.4f}  max_corr={corr:.3f}  {reason}")
    print(f"     ({seg_name}段耗时 {time.time()-t_seg:.1f}s)")

# 三段变化趋势：后段-前段（捕捉偷电时间点）
for fname_idx, fname in enumerate(['noise_energy', 'signal_ratio', 'mahal_noise']):
    delta_feat = seg_feats['late'][:, fname_idx] - seg_feats['early'][:, fname_idx]
    full_name = f"A_delta_late_early_{fname}"
    auc, corr, useful, reason = evaluate_feature(full_name, delta_feat)
    results.append((full_name, auc, corr, useful, reason))
    status = "✅" if useful else "❌"
    print(f"  {status} {full_name:<35} AUC={auc:.4f}  max_corr={corr:.3f}  {reason}")

# 后半段噪声能量 / 前半段噪声能量（相对变化）
noise_ratio_trend = (seg_feats['late'][:, 0] + 1e-6) / (seg_feats['early'][:, 0] + 1e-6)
auc, corr, useful, reason = evaluate_feature("A_noise_ratio_trend", noise_ratio_trend)
results.append(("A_noise_ratio_trend", auc, corr, useful, reason))
status = "✅" if useful else "❌"
print(f"  {status} {'A_noise_ratio_trend':<35} AUC={auc:.4f}  max_corr={corr:.3f}  {reason}")

# =============================================================================
# 4. 候选设计 B：月度噪声子空间投影序列的统计量
# 核心理念：如果偷电用户的月度序列偏离了"正常用户主成分"，
#           噪声投影能量会随时间升高，且波动性大
# =============================================================================
print("\n--- 候选设计 B: 月度噪声子空间投影时序统计 ---")

# 用全量用户（不分层）做一个全局协方差，提取噪声子空间
# 再看每个用户在每个月的噪声投影时序

# 全量月度序列归一化
all_med = np.median(mo_mean, axis=1, keepdims=True)
all_iqr = np.clip(np.percentile(mo_mean, 75, axis=1, keepdims=True) -
                  np.percentile(mo_mean, 25, axis=1, keepdims=True), 1e-6, None)
mo_norm_all = np.clip((mo_mean - all_med) / all_iqr, -5, 5).astype(np.float64)

# 全局协方差（月度维度，NM×NM）
X_cent = mo_norm_all - mo_norm_all.mean(axis=0, keepdims=True)
cov_global = (X_cent.T @ X_cent) / (N_USERS - 1)  # (NM, NM)
eigvals_g, eigvecs_g = np.linalg.eigh(cov_global)
eigvals_g = np.maximum(eigvals_g, 0)
sigma2_g = np.median(eigvals_g[eigvals_g > 0])
mp_up_g = _mp_upper(N_USERS, NM, sigma2=sigma2_g)

signal_idx_g = eigvals_g > mp_up_g
noise_idx_g  = ~signal_idx_g
V_sig_g = eigvecs_g[:, signal_idx_g]
V_noi_g = eigvecs_g[:, noise_idx_g]

print(f"  全局RMT: 信号特征值数={signal_idx_g.sum()}, 噪声特征值数={noise_idx_g.sum()}, "
      f"M-P上界={mp_up_g:.3f}")

# 每用户的全局噪声投影能量
B_noise_global = (mo_norm_all @ V_noi_g)**2
B_noise_energy = B_noise_global.sum(axis=1).astype(np.float32)
B_signal_energy = ((mo_norm_all @ V_sig_g)**2).sum(axis=1).astype(np.float32)
B_snr = B_signal_energy / (B_noise_energy + 1e-8)

for name, feat in [("B_noise_energy_global", B_noise_energy),
                   ("B_signal_energy_global", B_signal_energy),
                   ("B_snr_global", B_snr)]:
    auc, corr, useful, reason = evaluate_feature(name, feat)
    results.append((name, auc, corr, useful, reason))
    status = "✅" if useful else "❌"
    print(f"  {status} {name:<35} AUC={auc:.4f}  max_corr={corr:.3f}  {reason}")

# 逐月噪声投影能量时序 → 统计量
mo_noise_proj_energy = np.zeros((N_USERS, NM), dtype=np.float32)
for m in range(NM):
    x_m = mo_norm_all[:, m:m+1]  # (N, 1)
    # 该月在噪声子空间的投影
    proj = (x_m * V_noi_g[m, :][np.newaxis, :]).sum(axis=1)  # 近似
    mo_noise_proj_energy[:, m] = proj.astype(np.float32)

# 噪声投影的时序std（越大越不稳定）
B_noise_ts_std = np.abs(mo_noise_proj_energy).std(axis=1)
# 后半段噪声均值 - 前半段噪声均值
B_noise_trend = (np.abs(mo_noise_proj_energy[:, half_nm:]).mean(axis=1) -
                 np.abs(mo_noise_proj_energy[:, :half_nm]).mean(axis=1))
# 最大噪声投影时间点（偷电开始时间）
B_noise_peak_pos = np.argmax(np.abs(mo_noise_proj_energy), axis=1).astype(np.float32) / NM

for name, feat in [("B_noise_ts_std", B_noise_ts_std),
                   ("B_noise_trend", B_noise_trend),
                   ("B_noise_peak_pos", B_noise_peak_pos)]:
    auc, corr, useful, reason = evaluate_feature(name, feat)
    results.append((name, auc, corr, useful, reason))
    status = "✅" if useful else "❌"
    print(f"  {status} {name:<35} AUC={auc:.4f}  max_corr={corr:.3f}  {reason}")

# =============================================================================
# 5. 候选设计 C：用户月度序列与"正常用户主成分重建"的残差
# 核心理念：用正常用户（FLAG=0）的主成分重建每个用户，
#           重建残差越大 → 越偏离正常用电模式
# 注意：这里只用训练集标签，不会泄露，因为是用整体FLAG分布
# =============================================================================
print("\n--- 候选设计 C: 正常用户主成分残差（PCA重建误差）---")

# 用所有用户（含异常）的月度序列做PCA（top-k信号主成分）
# 再用这些主成分重建每个用户，计算重建残差
# 这与 RMT 等价（信号子空间 = 前k个主成分），但更直观

n_signal_comps = max(int(signal_idx_g.sum()), 2)  # 至少保留2个
V_top = eigvecs_g[:, -n_signal_comps:]  # top-k信号主成分

# 每用户的 PCA 重建
proj_top = mo_norm_all @ V_top          # (N, k)
recon = proj_top @ V_top.T              # (N, NM) 重建
residual = mo_norm_all - recon          # (N, NM) 重建残差

C_recon_err = (residual**2).mean(axis=1).astype(np.float32)   # 重建误差（MSE）
C_recon_err_std = (residual**2).std(axis=1).astype(np.float32) # 误差的方差
C_recon_err_trend = ((residual[:, half_nm:]**2).mean(axis=1) -  # 后-前误差趋势
                      (residual[:, :half_nm]**2).mean(axis=1)).astype(np.float32)
C_recon_err_max_pos = np.argmax(residual**2, axis=1).astype(np.float32) / NM  # 最大误差时间点

for name, feat in [("C_recon_mse", C_recon_err),
                   ("C_recon_mse_std", C_recon_err_std),
                   ("C_recon_mse_trend", C_recon_err_trend),
                   ("C_recon_max_pos", C_recon_err_max_pos)]:
    auc, corr, useful, reason = evaluate_feature(name, feat)
    results.append((name, auc, corr, useful, reason))
    status = "✅" if useful else "❌"
    print(f"  {status} {name:<35} AUC={auc:.4f}  max_corr={corr:.3f}  {reason}")

# 月度重建误差序列 → 每月各用户的误差排名（跨用户标准化）
mo_recon_err_monthly = (residual**2).astype(np.float32)  # (N, NM)
mo_recon_err_rank = np.zeros((N_USERS, NM), dtype=np.float32)
for m in range(NM):
    mo_recon_err_rank[:, m] = rankdata(mo_recon_err_monthly[:, m]) / N_USERS

C_err_rank_mean = mo_recon_err_rank.mean(axis=1)
C_err_rank_trend = mo_recon_err_rank[:, half_nm:].mean(axis=1) - mo_recon_err_rank[:, :half_nm].mean(axis=1)
C_err_rank_max = mo_recon_err_rank.max(axis=1)

for name, feat in [("C_err_rank_mean", C_err_rank_mean),
                   ("C_err_rank_trend", C_err_rank_trend),
                   ("C_err_rank_max", C_err_rank_max)]:
    auc, corr, useful, reason = evaluate_feature(name, feat)
    results.append((name, auc, corr, useful, reason))
    status = "✅" if useful else "❌"
    print(f"  {status} {name:<35} AUC={auc:.4f}  max_corr={corr:.3f}  {reason}")

# =============================================================================
# 6. 候选设计 D：日序列 RMT（工作日序列 vs 周末序列的谱结构差异）
# 核心理念：正常用户工作日和周末的用电规律不同，
#           偷电用户（挂表/绕表）工作日/周末差异会异常小或异常大
# =============================================================================
print("\n--- 候选设计 D: 工作日/周末用电的谱结构差异 ---")

weekday_mask = (day_of_week < 5)
weekend_mask = (day_of_week >= 5)

# 工作日序列和周末序列（取前 NM*5 天的工作日，按月聚合）
wd_mo = np.zeros((N_USERS, NM), dtype=np.float32)
we_mo = np.zeros((N_USERS, NM), dtype=np.float32)
for m in range(NM):
    s, e = m*DAYSPM, min((m+1)*DAYSPM, T_DAYS)
    wd_m = weekday_mask[s:e]; we_m = weekend_mask[s:e]
    if wd_m.sum() > 0: wd_mo[:, m] = X[:, s:e][:, wd_m].mean(axis=1)
    if we_m.sum() > 0: we_mo[:, m] = X[:, s:e][:, we_m].mean(axis=1)

# 工作日/周末比的时序稳定性（std越小越稳定）
wd_we_ratio_monthly = wd_mo / (we_mo + 1e-6)
D_ratio_std = wd_we_ratio_monthly.std(axis=1).astype(np.float32)
D_ratio_trend = (wd_we_ratio_monthly[:, half_nm:].mean(axis=1) -
                 wd_we_ratio_monthly[:, :half_nm].mean(axis=1)).astype(np.float32)

# 工作日月度序列的 PCA 重建误差（用工作日协方差的信号子空间）
wd_med = np.median(wd_mo, axis=1, keepdims=True)
wd_iqr = np.clip(np.percentile(wd_mo, 75, axis=1, keepdims=True) -
                 np.percentile(wd_mo, 25, axis=1, keepdims=True), 1e-6, None)
wd_norm = np.clip((wd_mo - wd_med) / wd_iqr, -5, 5).astype(np.float64)
wd_cent = wd_norm - wd_norm.mean(axis=0, keepdims=True)
cov_wd = (wd_cent.T @ wd_cent) / (N_USERS - 1)
eigvals_wd, eigvecs_wd = np.linalg.eigh(cov_wd)
eigvals_wd = np.maximum(eigvals_wd, 0)
sigma2_wd = max(np.median(eigvals_wd[eigvals_wd > 0]), 1e-6)
mp_wd = _mp_upper(N_USERS, NM, sigma2=sigma2_wd)
n_sig_wd = max((eigvals_wd > mp_wd).sum(), 2)
V_wd_top = eigvecs_wd[:, -n_sig_wd:]
wd_recon = (wd_norm @ V_wd_top) @ V_wd_top.T
D_wd_recon_err = ((wd_norm - wd_recon)**2).mean(axis=1).astype(np.float32)

for name, feat in [("D_wd_we_ratio_std", D_ratio_std),
                   ("D_wd_we_ratio_trend", D_ratio_trend),
                   ("D_wd_recon_err", D_wd_recon_err)]:
    auc, corr, useful, reason = evaluate_feature(name, feat)
    results.append((name, auc, corr, useful, reason))
    status = "✅" if useful else "❌"
    print(f"  {status} {name:<35} AUC={auc:.4f}  max_corr={corr:.3f}  {reason}")

# =============================================================================
# 7. 汇总诊断结果
# =============================================================================
print("\n" + "=" * 70)
print("诊断汇总：有价值的候选特征（AUC>0.55 且 相关性<0.65）")
print("=" * 70)
useful_results = [(n, a, c, r) for n, a, c, u, r in results if u]
useless_results = [(n, a, c, r) for n, a, c, u, r in results if not u]

print(f"\n  ✅ 有价值: {len(useful_results)} 维")
for name, auc, corr, reason in sorted(useful_results, key=lambda x: -x[1]):
    print(f"     {name:<35} AUC={auc:.4f}  max_corr={corr:.3f}")

print(f"\n  ❌ 无价值/重复: {len(useless_results)} 维")
for name, auc, corr, reason in useless_results:
    print(f"     {name:<35} AUC={auc:.4f}  max_corr={corr:.3f}  ({reason})")

print(f"\n  总耗时: {time.time()-t0:.0f}s")
print("=" * 70)
print("\n→ 下一步: 只把上方 ✅ 的特征加入 Phase 2 消融实验")
