# =============================================================================
# patch_feature_boost.py
# 特征工程增强补丁：目标 AUC ≥ 0.90
# 在 sgcc_analysis.py 的 Step A 完成后、Step B 训练前插入此补丁
#
# 新增特征（全部有文献/理论支撑，与 RMT 协同）：
#   F31: λ_max 时序差分均值（RMT谱变化率）         AUC目标 ≥ 0.68
#   F32: λ_max 时序差分标准差（RMT谱波动强度）      AUC目标 ≥ 0.65
#   F33: λ_max 超界连续窗口最长段                  AUC目标 ≥ 0.65
#   F34: Permutation Entropy（排列熵）              AUC目标 ≥ 0.68
#   F35: Sample Entropy（样本熵，近似版）           AUC目标 ≥ 0.65
#   F36: 用电量谱熵（FFT功率谱熵）                  AUC目标 ≥ 0.66
#   F37: KPSS平稳性近似统计量                       AUC目标 ≥ 0.62
#   F38: 用电量 Hurst 指数（长程相关性）             AUC目标 ≥ 0.65
# =============================================================================

import numpy as np
import time
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

print("\n" + "=" * 65)
print("patch_feature_boost: 新增8维高判别力特征 (F31~F38)")
print("=" * 65)

# ── 依赖检查 ──────────────────────────────────────────────────────────────
# 本补丁需要在 sgcc_analysis.py 执行后运行，依赖以下变量：
#   X                  : (N_users, T) 预处理后用电量
#   y                  : (N_users,) 标签
#   lambda_max_series  : (N_windows,) 每窗口 λ_max
#   lambda_plus_series : (N_windows,) 每窗口 MP 上界
#   X_seq14_ds         : (N_users, 48, FEAT_DIM) 当前特征矩阵
#   FEAT_DIM           : 当前特征维度
#   N_W_ds             : 时间步数（48）
#   tile_user_feat     : 将 (N,) 平铺到 (N, N_W_ds) 的函数

_t_boost = time.time()
N_USERS, T_DAYS = X.shape

def _robust_tile(arr_1d, n_w, clip=5.0):
    """RobustScaler 归一化后平铺为 (N, n_w) 的时间轴特征"""
    arr = arr_1d.astype(np.float32).reshape(-1, 1)
    arr = RobustScaler().fit_transform(arr).flatten()
    arr = np.clip(arr, -clip, clip)
    return np.tile(arr[:, np.newaxis], (1, n_w)).astype(np.float32)

def _auc(feat, labels):
    try:
        a = roc_auc_score(labels, feat)
        return max(a, 1 - a)
    except Exception:
        return 0.5

# =============================================================================
# F31 & F32: λ_max 时序差分特征（直接基于 RMT 谱序列）
# 理论：窃电用户导致协方差矩阵结构突变，λ_max 的差分序列方差更大
# 文献：Vinayak et al., "Spectral methods for NTL detection", 2020
# =============================================================================
print("\n  [F31/F32] RMT λ_max 时序差分特征...")

lmax_diff = np.diff(lambda_max_series)          # (N_windows-1,)
lmax_diff_mean = np.abs(lmax_diff).mean()       # 标量：全局平均变化率
lmax_diff_std  = lmax_diff.std()                # 标量：变化率标准差

# 用户级特征：用户在每窗口的 Token1(v1投影) 与 λ_max 差分的加权乘积
# 思路：在 λ_max 剧烈变化的窗口，若该用户 v1 投影也大 → 该用户是驱动者
v1_proj_time = X_seq14_ds[:, :, 0].astype(np.float64)  # (N, 48)

# 将 lmax_diff 插值到 48 步
_idx_orig = np.linspace(0, len(lmax_diff) - 1, N_W_ds)
lmax_diff_48 = np.interp(_idx_orig, np.arange(len(lmax_diff)), np.abs(lmax_diff))
# (N, 48) × (48,) → 加权后对时间轴求均值
tok_f31 = (v1_proj_time * lmax_diff_48[np.newaxis, :]).mean(axis=1).astype(np.float32)
tok_f32 = (v1_proj_time * (lmax_diff_48[np.newaxis, :] ** 2)).mean(axis=1).astype(np.float32)

auc_f31 = _auc(tok_f31, y)
auc_f32 = _auc(tok_f32, y)
print(f"    F31(λ_max差分×v1投影均值)  AUC={auc_f31:.4f}  {'✅' if auc_f31>=0.60 else '⚠️'}")
print(f"    F32(λ_max差分²×v1投影均值) AUC={auc_f32:.4f}  {'✅' if auc_f32>=0.60 else '⚠️'}")

# =============================================================================
# F33: λ_max 超界连续窗口最长段（用户级加权）
# 理论：窃电行为持续时间长 → 该用户在连续异常窗口中 v1 投影持续高
# =============================================================================
print("\n  [F33] RMT 连续超界窗口加权特征...")

exceed_mask_48 = np.interp(
    _idx_orig,
    np.arange(len(lambda_max_series)),
    (lambda_max_series > lambda_plus_series).astype(float)
) > 0.5  # (48,) bool

# 找连续超界段，计算每段长度
_changes = np.diff(exceed_mask_48.astype(int), prepend=0, append=0)
_starts  = np.where(_changes == 1)[0]
_ends    = np.where(_changes == -1)[0]
_run_lengths = _ends - _starts  # 每段长度

if len(_run_lengths) > 0:
    # 构造"连续段权重"：超界段内权重 = 段长度，非超界段 = 0
    exceed_weights = np.zeros(N_W_ds, dtype=np.float32)
    for s, e, rl in zip(_starts, _ends, _run_lengths):
        exceed_weights[s:e] = rl
    tok_f33 = (v1_proj_time * exceed_weights[np.newaxis, :]).mean(axis=1).astype(np.float32)
else:
    tok_f33 = v1_proj_time.mean(axis=1).astype(np.float32)

auc_f33 = _auc(tok_f33, y)
print(f"    F33(连续超界段加权投影)     AUC={auc_f33:.4f}  {'✅' if auc_f33>=0.60 else '⚠️'}")

# =============================================================================
# F34: 排列熵（Permutation Entropy）
# 理论：窃电用户用电序列的复杂度异常（过低=固定模式，过高=随机噪声）
# 文献：Bandt & Pompe, PRL 2002；在NTL检测中广泛使用
# 算法：对每用户计算 order=3 的排列熵，O(N*T) 向量化实现
# =============================================================================
print("\n  [F34] 排列熵（Permutation Entropy）...")

def permutation_entropy_batch(X_data, order=3, delay=1):
    """
    向量化批量排列熵计算
    X_data: (N, T)
    返回: (N,) 归一化排列熵，值域 [0,1]
    """
    N, T = X_data.shape
    T_valid = T - (order - 1) * delay
    if T_valid <= 0:
        return np.zeros(N, dtype=np.float32)

    # 构造嵌入矩阵：(N, T_valid, order)
    indices = np.array([
        np.arange(T_valid) + i * delay for i in range(order)
    ]).T  # (T_valid, order)
    X_emb = X_data[:, indices]  # (N, T_valid, order)

    # 获取每个时间点的排名（argsort of argsort = rank）
    ranks = np.argsort(np.argsort(X_emb, axis=2), axis=2)  # (N, T_valid, order)

    # 将排名转为唯一整数模式（用 order 进制编码）
    multipliers = (order ** np.arange(order)).astype(np.int64)
    patterns = (ranks * multipliers[np.newaxis, np.newaxis, :]).sum(axis=2)  # (N, T_valid)

    # 计算每用户各模式频率，再算香农熵
    max_pattern = order ** order
    entropy = np.zeros(N, dtype=np.float32)
    for i in range(N):
        counts = np.bincount(patterns[i], minlength=max_pattern)
        probs  = counts[counts > 0] / T_valid
        entropy[i] = -np.sum(probs * np.log2(probs + 1e-10))

    # 归一化到 [0,1]：最大熵 = log2(order!)
    import math
    max_entropy = np.log2(math.factorial(order))
    return (entropy / (max_entropy + 1e-8)).astype(np.float32)

tok_f34 = permutation_entropy_batch(X, order=3, delay=1)
auc_f34 = _auc(tok_f34, y)
print(f"    F34(排列熵 order=3)         AUC={auc_f34:.4f}  {'✅' if auc_f34>=0.60 else '⚠️'}")

# =============================================================================
# F35: 谱熵（FFT 功率谱熵）
# 理论：窃电用户的周期性遭到破坏 → 功率谱更均匀（高熵）或极端集中（低熵）
# =============================================================================
print("\n  [F35] FFT 功率谱熵...")

def spectral_entropy_batch(X_data, n_freq=50):
    """
    X_data: (N, T) → 对每用户计算 FFT 后的功率谱熵 (N,)
    """
    fft_amp = np.abs(np.fft.rfft(X_data, axis=1))[:, 1:n_freq+1]  # (N, n_freq)
    power   = fft_amp ** 2
    power_sum = power.sum(axis=1, keepdims=True) + 1e-10
    prob    = power / power_sum
    entropy = -np.sum(prob * np.log2(prob + 1e-10), axis=1)
    max_ent = np.log2(n_freq)
    return (entropy / (max_ent + 1e-8)).astype(np.float32)

tok_f35 = spectral_entropy_batch(X, n_freq=50)
auc_f35 = _auc(tok_f35, y)
print(f"    F35(FFT功率谱熵)            AUC={auc_f35:.4f}  {'✅' if auc_f35>=0.60 else '⚠️'}")

# =============================================================================
# F36: 近似样本熵（ApEn 近似，避免 O(T²) 的精确 SampEn）
# 理论：高样本熵 = 序列更随机 = 可能是随机窃电模式
# 近似算法：用滑动窗口标准差替代精确 ApEn，速度 O(N*T)
# =============================================================================
print("\n  [F36] 近似样本熵...")

def approx_sample_entropy(X_data, m=3, win=7):
    """
    近似版：对每个长度 m 的滑动窗口计算局部标准差，取均值作为复杂度估计
    X_data: (N, T)
    """
    N, T = X_data.shape
    if T <= m + win:
        return np.zeros(N, dtype=np.float32)
    result = np.zeros(N, dtype=np.float32)
    count  = 0
    for start in range(0, T - m - win + 1, win):
        seg = X_data[:, start:start + win]  # (N, win)
        result += seg.std(axis=1)
        count  += 1
    return (result / max(count, 1)).astype(np.float32)

tok_f36 = approx_sample_entropy(X, m=3, win=7)
auc_f36 = _auc(tok_f36, y)
print(f"    F36(近似样本熵)             AUC={auc_f36:.4f}  {'✅' if auc_f36>=0.58 else '⚠️'}")

# =============================================================================
# F37: KPSS 近似平稳性统计量（无需 statsmodels）
# 理论：KPSS 检验 H0 = 序列平稳；窃电用户趋势性下降 → KPSS 统计量大
# 近似公式：η = (1/T²σ²) Σ_t S_t²，S_t = Σ_{s≤t} (x_s - x̄)
# 文献：Kwiatkowski et al., Journal of Econometrics, 1992
# =============================================================================
print("\n  [F37] KPSS 近似平稳性统计量...")

def kpss_approx_batch(X_data):
    """
    向量化 KPSS 统计量：η̂ = (1/T²σ̂²) Σ_t S_t²
    X_data: (N, T) → (N,)
    """
    N, T = X_data.shape
    x_mean = X_data.mean(axis=1, keepdims=True)          # (N, 1)
    x_demean = X_data - x_mean                            # (N, T)
    S = np.cumsum(x_demean, axis=1)                       # (N, T) 累积和
    sigma2 = (x_demean ** 2).mean(axis=1) + 1e-10        # (N,) 残差方差
    eta = (S ** 2).mean(axis=1) / (T * sigma2)            # (N,)
    return eta.astype(np.float32)

tok_f37 = kpss_approx_batch(X)
auc_f37 = _auc(tok_f37, y)
print(f"    F37(KPSS近似统计量)         AUC={auc_f37:.4f}  {'✅' if auc_f37>=0.58 else '⚠️'}")

# =============================================================================
# F38: Hurst 指数（R/S 分析，长程相关性）
# 理论：正常用电有长程相关性（H>0.5）；窃电破坏相关性（H→0.5 或 <0.5）
# 近似算法：对3个子段计算 R/S，线性拟合斜率 = Hurst
# 文献：Hurst, 1951；在时间序列异常检测中广泛应用
# =============================================================================
print("\n  [F38] Hurst 指数（R/S 分析）...")

def hurst_rs_batch(X_data, n_segments=4):
    """
    向量化 R/S Hurst 指数估计
    X_data: (N, T) → (N,)
    """
    N, T = X_data.shape
    seg_sizes = np.array([T // (2 ** i) for i in range(1, n_segments + 1)
                          if T // (2 ** i) >= 8])
    if len(seg_sizes) < 2:
        return np.full(N, 0.5, dtype=np.float32)

    log_n_list, log_rs_list = [], []

    for seg_size in seg_sizes:
        n_segs = T // seg_size
        if n_segs == 0:
            continue
        X_cut = X_data[:, :n_segs * seg_size].reshape(N, n_segs, seg_size)
        # 每段均值偏差
        seg_mean = X_cut.mean(axis=2, keepdims=True)
        Y = X_cut - seg_mean                              # (N, n_segs, seg_size)
        Z = np.cumsum(Y, axis=2)                          # 累积偏差
        R = Z.max(axis=2) - Z.min(axis=2)               # 极差 (N, n_segs)
        S = X_cut.std(axis=2) + 1e-10                    # 标准差
        RS = (R / S).mean(axis=1)                        # (N,) 平均 R/S
        log_n_list.append(np.log(seg_size))
        log_rs_list.append(np.log(RS + 1e-10))           # (N,)

    if len(log_n_list) < 2:
        return np.full(N, 0.5, dtype=np.float32)

    log_n_arr  = np.array(log_n_list)                    # (K,)
    log_rs_arr = np.stack(log_rs_list, axis=1)           # (N, K)

    # 逐用户线性回归斜率（向量化最小二乘）
    n_pts  = len(log_n_arr)
    x_mean = log_n_arr.mean()
    x_c    = log_n_arr - x_mean                          # (K,)
    y_c    = log_rs_arr - log_rs_arr.mean(axis=1, keepdims=True)  # (N, K)
    hurst  = (y_c * x_c[np.newaxis, :]).sum(axis=1) / ((x_c ** 2).sum() + 1e-10)
    return np.clip(hurst, 0.0, 1.0).astype(np.float32)

tok_f38 = hurst_rs_batch(X, n_segments=4)
auc_f38 = _auc(tok_f38, y)
print(f"    F38(Hurst指数 R/S)          AUC={auc_f38:.4f}  {'✅' if auc_f38>=0.58 else '⚠️'}")

# =============================================================================
# 汇总：将 F31~F38 拼入 X_seq14_ds
# =============================================================================
print("\n  汇总新特征 AUC:")
new_feats = {
    'F31(λ_max差分×v1投影)': tok_f31,
    'F32(λ_max差分²×v1投影)': tok_f32,
    'F33(连续超界段加权)': tok_f33,
    'F34(排列熵)': tok_f34,
    'F35(谱熵)': tok_f35,
    'F36(近似样本熵)': tok_f36,
    'F37(KPSS统计量)': tok_f37,
    'F38(Hurst指数)': tok_f38,
}

valid_new = 0
for name, feat in new_feats.items():
    auc_val = _auc(feat, y)
    mark = '✅' if auc_val >= 0.60 else ('⚠️' if auc_val >= 0.55 else '❌')
    if auc_val >= 0.55:
        valid_new += 1
    print(f"    {name:<28} AUC={auc_val:.4f}  {mark}")

print(f"\n  有效新特征(AUC≥0.55): {valid_new}/8")

# 平铺 + RobustScaler + 拼入
for name, feat in new_feats.items():
    tiled = _robust_tile(feat, N_W_ds)
    X_seq14_ds = np.concatenate(
        [X_seq14_ds, tiled[:, :, np.newaxis]], axis=-1
    ).astype(np.float32)

FEAT_DIM = X_seq14_ds.shape[-1]
print(f"\n  拼入 F31~F38 后 X_seq14_ds 形状: {X_seq14_ds.shape}  (FEAT_DIM={FEAT_DIM})")
print(f"  耗时: {time.time() - _t_boost:.1f}s")
print("=" * 65)
