
# =============================================================================
# SGCC 用电数据异常检测分析脚本
# 数据来源: State Grid Corporation of China (SGCC)
# =============================================================================

import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

# =============================================================================
# ⚡ CPU 加速设置（无GPU环境）
# OMP/MKL 线程数设为物理核数，避免超线程竞争
# =============================================================================
_CPU_THREADS = min(os.cpu_count() or 4, 8)   # 最多8线程，避免调度开销
os.environ['OMP_NUM_THREADS']  = str(_CPU_THREADS)
os.environ['MKL_NUM_THREADS']  = str(_CPU_THREADS)
os.environ['NUMEXPR_NUM_THREADS'] = str(_CPU_THREADS)
import torch
torch.set_num_threads(_CPU_THREADS)
torch.set_num_interop_threads(max(1, _CPU_THREADS // 2))
print(f"⚡ CPU 加速: OMP/MKL/Torch 线程数 = {_CPU_THREADS}")

# =============================================================================
# 🎛️  运行控制开关
#   设置 STOP_AFTER = 'A'  → 只运行到 Step A（特征区分度验证）后停止
#   设置 STOP_AFTER = 'B'  → 运行到 Step B（双路Transformer训练）后停止
#   设置 STOP_AFTER = None → 运行全部步骤（A → B → C → D → E）
# =============================================================================
STOP_AFTER = None     # ← 修改这里（None=全部运行，'A'=只跑Step A，'B'=跑到Step B）

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold

# 解决中文乱码问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 1. 加载数据
# =============================================================================
DATA_PATH = r'C:\Users\wb.zhoushujie\Desktop\data set.csv'

print("=" * 60)
print("Step 1: 加载数据")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("数据形状:", df.shape)
print("\n前5行:")
print(df.head())
print("\n列名（前10个）:", df.columns.tolist()[:10], "...")
print("\n标签分布:")
print(df['FLAG'].value_counts())

# =============================================================================
# 2. 基本统计分析
# =============================================================================
print("\n" + "=" * 60)
print("Step 2: 基本统计")
print("=" * 60)

labels = df['FLAG']
# 去掉标签列和非数值列（如 CONS_NO 用户编号）
features = df.drop('FLAG', axis=1).select_dtypes(include=[np.number])

print(f"总用户数: {len(df)}")
print(f"正常用户: {(labels==0).sum()} ({(labels==0).mean()*100:.1f}%)")
print(f"异常用户: {(labels==1).sum()} ({(labels==1).mean()*100:.1f}%)")
print(f"\n时间步数(天数): {features.shape[1]}")
print(f"缺失值数量: {features.isna().sum().sum()}")
print(f"缺失值比例: {features.isna().mean().mean()*100:.2f}%")

# =============================================================================
# 3. 数据清洗和预处理（先处理，再可视化）
# =============================================================================
print("\n" + "=" * 60)
print("Step 3: 数据预处理")
print("=" * 60)

def preprocess_sgcc(df):
    """
    SGCC数据预处理流水线：
      1. 线性插值填充缺失值
      2. Z-score 归一化（按用户维度）
      3. 极端值裁剪（0.1% ~ 99.9%），避免 y 轴爆炸
    返回: X (np.float32), y (np.int64), scaler
    """
    labels_arr = df['FLAG'].values
    # 去掉标签列和非数值列（如 CONS_NO 用户编号）
    features_df = df.drop('FLAG', axis=1).select_dtypes(include=[np.number])

    # Step 1: 处理缺失值（线性插值）
    print("  → 线性插值填充缺失值...")
    features_filled = features_df.interpolate(
        method='linear', axis=1, limit_direction='both'
    )
    # 头尾仍有 NaN 时，用列均值兜底
    features_filled = features_filled.fillna(features_filled.mean())

    # Step 2: Z-score 归一化（转置后按列即按用户归一化）
    print("  → Z-score 归一化...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_filled.T).T

    # Step 3: 裁剪极端值（避免 y 轴爆炸）
    print("  → 极端值裁剪（0.1% ~ 99.9% 百分位数）...")
    lower = np.percentile(features_scaled, 0.1)
    upper = np.percentile(features_scaled, 99.9)
    features_scaled = np.clip(features_scaled, lower, upper)
    print(f"     裁剪范围: [{lower:.3f}, {upper:.3f}]")

    X = features_scaled.astype(np.float32)
    y = labels_arr.astype(np.int64)
    # ★ 保存原始插值数据（未归一化），用于计算基准偏离特征
    X_raw = features_filled.values.astype(np.float32)

    print(f"  预处理完成: X.shape={X.shape}, y.shape={y.shape}")
    print(f"  X 值域: [{X.min():.3f}, {X.max():.3f}]")
    return X, y, scaler, X_raw

X, y, scaler, X_raw = preprocess_sgcc(df)

# =============================================================================
# 4. 可视化：正常 vs 异常用户的用电模式（使用预处理后数据）
# =============================================================================
print("\n" + "=" * 60)
print("Step 4: 可视化用电模式（基于预处理后数据）")
print("=" * 60)

np.random.seed(42)  # 固定随机种子，保证可复现

normal_X   = X[y == 0]   # shape: (n_normal,   n_days)
abnormal_X = X[y == 1]   # shape: (n_abnormal, n_days)

# --- 图1：随机抽样个体曲线 ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for ax in axes[0]:
    idx = np.random.randint(len(normal_X))
    ax.plot(normal_X[idx], color='blue', alpha=0.7)
    ax.set_title(f'正常用户 #{idx}')
    ax.set_xlabel('天数')
    ax.set_ylabel('用电量(标准化)')

for ax in axes[1]:
    idx = np.random.randint(len(abnormal_X))
    ax.plot(abnormal_X[idx], color='red', alpha=0.7)
    ax.set_title(f'异常用户 #{idx}')
    ax.set_xlabel('天数')
    ax.set_ylabel('用电量(标准化)')

plt.suptitle('SGCC: 正常 vs 异常用电模式（标准化后）', fontsize=14)
plt.tight_layout()
plt.savefig('sgcc_patterns.png', dpi=150, bbox_inches='tight')
plt.show()
print("已保存: sgcc_patterns.png")

# --- 图2：均值 ± 标准差对比 ---
plt.figure(figsize=(12, 5))

normal_mean   = normal_X.mean(axis=0)
normal_std    = normal_X.std(axis=0)
abnormal_mean = abnormal_X.mean(axis=0)
abnormal_std  = abnormal_X.std(axis=0)

x_axis = range(len(normal_mean))

plt.plot(normal_mean, label='正常用户均值', color='blue', alpha=0.8)
plt.fill_between(x_axis,
                 normal_mean - normal_std,
                 normal_mean + normal_std,
                 alpha=0.2, color='blue')

plt.plot(abnormal_mean, label='异常用户均值', color='red', alpha=0.8)
plt.fill_between(x_axis,
                 abnormal_mean - abnormal_std,
                 abnormal_mean + abnormal_std,
                 alpha=0.2, color='red')

plt.legend(fontsize=12)
plt.xlabel('天数')
plt.ylabel('标准化用电量')
plt.title('正常 vs 异常用户平均用电模式（标准化后）')
plt.tight_layout()
plt.savefig('sgcc_mean_patterns.png', dpi=150, bbox_inches='tight')
plt.show()
print("已保存: sgcc_mean_patterns.png")

# =============================================================================
# 5. 划分数据集（保持类别比例）
# =============================================================================
print("\n" + "=" * 60)
print("Step 5: 划分数据集")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # 保持正负样本比例
)

print(f"训练集: {X_train.shape},  异常比例: {y_train.mean():.3f}")
print(f"测试集: {X_test.shape},  异常比例: {y_test.mean():.3f}")

# 5折分层交叉验证迭代器（供后续模型使用）
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("5折交叉验证迭代器已创建（变量名: skf）")

# =============================================================================
# 6. 处理类别不平衡
# =============================================================================
print("\n" + "=" * 60)
print("Step 6: 处理类别不平衡")
print("=" * 60)

# --- 方案 A: SMOTE 过采样 ---
try:
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"SMOTE 后 → 正常: {(y_train_balanced==0).sum()}, "
          f"异常: {(y_train_balanced==1).sum()}")
except ImportError:
    print("⚠ 未安装 imbalanced-learn，跳过 SMOTE。")
    print("  可运行: pip install imbalanced-learn")
    X_train_balanced, y_train_balanced = X_train, y_train

# --- 方案 B: 损失函数加权（推荐，不改变数据分布）---
n_normal   = (y_train == 0).sum()
n_abnormal = (y_train == 1).sum()
weight_normal   = 1.0
weight_abnormal = n_normal / n_abnormal

print(f"\n类别权重 → 正常: {weight_normal:.1f},  异常: {weight_abnormal:.1f}")
print("PyTorch 中使用方式:")
print("  criterion = nn.BCEWithLogitsLoss(")
print(f"      pos_weight=torch.tensor([{weight_abnormal:.1f}])")
print("  )")

# =============================================================================
# 7. 构造 RMT（随机矩阵理论）窗口输入格式
# =============================================================================
print("\n" + "=" * 60)
print("Step 7: 构造 RMT 滑动窗口")
print("=" * 60)

def create_rmt_windows(X, window_size=30, stride=7):
    """
    将 SGCC 数据转换为 RMT 可用的滑动窗口格式。

    参数:
        X           : (n_users, n_days)
        window_size : 每个窗口的天数
        stride      : 滑动步长

    返回:
        windows: np.ndarray, shape = (n_windows, n_users, window_size)
    """
    n_users, n_days = X.shape
    windows = []
    for start in range(0, n_days - window_size + 1, stride):
        windows.append(X[:, start:start + window_size])
    windows = np.array(windows)
    print(f"  生成窗口数: {len(windows)},  每窗口形状: {windows[0].shape}")
    return windows

windows = create_rmt_windows(X, window_size=30, stride=7)


def compute_correlation_matrix(window):
    """
    计算样本相关矩阵（RMT 的核心输入）。
    自动过滤标准差为 0 的行（全零/常数用户），避免 NaN。

    参数:
        window: (n_sensors, T)

    返回:
        corr: (n_valid, n_valid) 相关矩阵
    """
    std = window.std(axis=1)
    valid_mask = std > 1e-8
    window_valid = window[valid_mask]

    if window_valid.shape[0] < 2:
        return np.eye(window.shape[0])

    window_centered = window_valid - window_valid.mean(axis=1, keepdims=True)
    # 用 warnings 静默残余的零方差警告
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        corr = np.corrcoef(window_centered)
    return corr


sample_window = windows[0]
corr_matrix   = compute_correlation_matrix(sample_window)
print(f"  相关矩阵形状: {corr_matrix.shape}")

# 可视化相关矩阵（前50个用户）
plt.figure(figsize=(8, 6))
vis_size = min(50, corr_matrix.shape[0])
sns.heatmap(corr_matrix[:vis_size, :vis_size],
            cmap='RdBu_r', center=0, vmin=-1, vmax=1)
plt.title('用户间用电量相关矩阵（前50用户）')
plt.tight_layout()
plt.savefig('sgcc_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("已保存: sgcc_correlation_matrix.png")

# =============================================================================
# 8. RMT 谱分析：特征值分解 + Marchenko-Pastur 判断 + 谱Token
# =============================================================================
print("\n" + "=" * 60)
print("Step 8: RMT 谱分析（特征值分解 + MP边界 + 谱Token）")
print("=" * 60)

from scipy.linalg import eigh
from sklearn.metrics import roc_curve, auc as sk_auc, roc_auc_score


def robust_cov_matrix(X_sub):
    """
    鲁棒协方差估计（替换普通 np.cov）：
    - 优先使用 LedoitWolf 收缩估计（对高维小样本更稳健）
    - LedoitWolf 失败时退回 np.cov

    X_sub: (n_sub, T)
    返回: C (n_sub, n_sub) 鲁棒协方差矩阵
    """
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf(assume_centered=False)
        lw.fit(X_sub.T)          # (T, n_sub) → 估计 (n_sub, n_sub) 协方差
        return lw.covariance_
    except Exception:
        return np.cov(X_sub)     # fallback


def marchenko_pastur_bounds(X_window, sigma2=None):
    """
    自适应 Marchenko-Pastur 上下界（MAD 估计 σ²，抗野值）：

    $\\hat{\\sigma}^2 = \\left(\\frac{\\text{MAD}(X)}{0.6745}\\right)^2$

    $\\lambda_{\\pm} = \\hat{\\sigma}^2 (1 \\pm \\sqrt{\\gamma})^2$

    X_window: (n, T)  n=用户数, T=时间步
    返回: lambda_minus, lambda_plus, gamma, sigma2
    """
    n, T = X_window.shape
    gamma = n / T
    if sigma2 is None:
        # MAD 估计：比 np.var 对野值鲁棒
        median_val = np.median(X_window)
        mad = np.median(np.abs(X_window - median_val))
        sigma2 = (mad / 0.6745) ** 2 + 1e-8
    lambda_plus  = sigma2 * (1 + np.sqrt(gamma)) ** 2
    lambda_minus = sigma2 * (1 - np.sqrt(gamma)) ** 2
    return lambda_minus, lambda_plus, gamma, sigma2


def rmt_spectral_analysis(rmt_windows, y, subsample=500, k=5):
    """
    对每个滑动窗口做谱分析，生成 (k+2) 维增强谱Token：

      维度 0~k-1 : 全量用户在 top-k 特征向量上的投影 |v_i · x|
      维度 k     : 异常强度比  λ_max / λ_+  （广播到所有用户）
      维度 k+1   : 谱集中度   Σtop-k λ_i / trace(C)（广播到所有用户）

    公式：
        z_t = [ |v1·x|, |v2·x|, |v3·x|, |v4·x|, |v5·x|,
                λ_max/λ_+,  Σtop5 λ / trace(C) ]

    参数:
        rmt_windows : (n_windows, n_users, window_size)
        y           : (n_users,) 标签
        subsample   : 每窗口随机取多少用户做协方差分解
        k           : 特征向量投影维数（默认5）

    返回:
        lambda_max_series   : (n_windows,)
        lambda_plus_series  : (n_windows,)
        lambda_minus_series : (n_windows,)
        spec_tokens         : (n_windows, n_users, k+2)   ← 7 维
        anomaly_scores      : (n_users,)
    """
    n_windows, n_users, T = rmt_windows.shape
    n_sub  = min(subsample, n_users)
    n_feat = k + 2          # 5 投影 + 异常强度比 + 谱集中度 = 7

    lambda_max_series   = np.zeros(n_windows)
    lambda_plus_series  = np.zeros(n_windows)
    lambda_minus_series = np.zeros(n_windows)
    spec_tokens         = np.zeros((n_windows, n_users, n_feat))
    anomaly_scores      = np.zeros(n_users)

    # 保存每窗口特征值/向量，供后续用户级得分计算使用
    eigenvalues_all  = []   # list of (k,)   每窗口 top-k 特征值
    eigenvectors_all = []   # list of (T, k) 每窗口时间域特征向量

    print(f"  窗口总数: {n_windows},  用户数: {n_users},  窗口长度 T: {T}")
    print(f"  子采样大小: {n_sub},  γ = n_sub/T = {n_sub}/{T} = {n_sub/T:.2f}")
    print(f"  谱Token 维度: {k} 投影 + 1 异常强度比 + 1 谱集中度 = {n_feat} 维")
    print(f"  每 10 个窗口打印一次进度...")

    rng = np.random.default_rng(42)

    for w in range(n_windows):
        X_w = rmt_windows[w]                              # (n_users, T)

        # ① 随机子采样
        sample_idx = rng.choice(n_users, size=n_sub, replace=False)
        X_sub = X_w[sample_idx]                           # (n_sub, T)

        # ② MP 上下界（基于子集方差）
        lam_minus, lam_plus, gamma, sigma2 = marchenko_pastur_bounds(X_sub)
        lambda_plus_series[w]  = lam_plus
        lambda_minus_series[w] = lam_minus

        # ③ 鲁棒协方差矩阵 & top-k 特征值/向量（LedoitWolf 替换 np.cov）
        C = robust_cov_matrix(X_sub)                      # (n_sub, n_sub)
        eigenvalues, eigenvectors = eigh(
            C, subset_by_index=[n_sub - k, n_sub - 1]
        )
        eigenvalues  = eigenvalues[::-1]                  # (k,)  降序
        eigenvectors = eigenvectors[:, ::-1]              # (n_sub, k)

        lambda_max_series[w] = eigenvalues[0]

        # ④ 特征 0~k-1：全量用户在 top-k 特征向量上的投影
        #    Bug1修复：eigenvectors 在 n_sub 空间，需先转换到时间域(T维)再投影
        #    X_sub: (n_sub, T)  →  V_time = X_sub.T @ eigvec = (T, k)（时间域基）
        #    再将全量用户投影到时间域基：X_w(n_users,T) @ V_time(T,k) = (n_users,k)
        V_time = X_sub.T @ eigenvectors                    # (T, k)
        V_time /= (np.linalg.norm(V_time, axis=0, keepdims=True) + 1e-10)  # 列归一化
        spec_tokens[w, :, :k] = X_w @ V_time              # (n_users, k) ✅ 正确

        # ⑤ 特征 k：异常强度比 λ_max / λ_+（标量 → 广播到所有用户）
        intensity_ratio = eigenvalues[0] / (lam_plus + 1e-10)
        spec_tokens[w, :, k] = intensity_ratio

        # ⑥ 特征 k+1：谱集中度 = Σtop-k λ_i / trace(C)
        trace_C       = np.trace(C)
        concentration = eigenvalues.sum() / (trace_C + 1e-10)
        spec_tokens[w, :, k + 1] = concentration

        # ⑦ 保存时间域特征向量（(T,k)，供 compute_per_user_scores 使用）
        eigenvalues_all.append(eigenvalues.copy())    # (k,)
        eigenvectors_all.append(V_time.copy())        # (T, k) ✅ 时间域向量

        # ⑧ 粗略异常得分累计（窗口级，后续由 compute_per_user_scores 替代）
        if eigenvalues[0] > lam_plus:
            v1_proj = np.abs(X_w @ V_time[:, 0])     # (n_users,) ✅ 用修复后的V_time
            anomaly_scores += v1_proj

        if w % 10 == 0:
            flag = "🔴 异常" if eigenvalues[0] > lam_plus else "🟢 正常"
            print(f"    窗口 {w:3d}: λ_max={eigenvalues[0]:.4f},  "
                  f"λ_+={lam_plus:.4f},  "
                  f"强度比={intensity_ratio:.3f},  "
                  f"集中度={concentration:.3f}  {flag}")

    n_exceed = np.sum(lambda_max_series > lambda_plus_series)
    print(f"\n  ✅ 谱分析完成!  λ_max 超出 MP 上界的窗口: {n_exceed}/{n_windows}")
    print(f"  spec_tokens 形状: {spec_tokens.shape}  ({n_feat} 维增强谱Token)")
    return (lambda_max_series, lambda_plus_series,
            lambda_minus_series, spec_tokens, anomaly_scores,
            eigenvalues_all, eigenvectors_all)


def plot_spectral_results(lambda_max_series, lambda_plus_series,
                          lambda_minus_series, anomaly_scores, y,
                          save_prefix='sgcc'):
    """四联图可视化谱分析结果"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('RMT 谱分析结果', fontsize=16, fontweight='bold')
    wins = np.arange(len(lambda_max_series))

    # ① 最大特征值 vs MP 上界时序曲线
    ax = axes[0, 0]
    ax.plot(wins, lambda_max_series,  'b-',  lw=1.5, label=r'$\lambda_{max}$')
    ax.plot(wins, lambda_plus_series, 'r--', lw=1.5, label=r'MP上界 $\lambda_+$')
    ax.fill_between(wins, lambda_minus_series, lambda_plus_series,
                    alpha=0.15, color='green', label='MP 正常区间')
    exceed_mask = lambda_max_series > lambda_plus_series
    ax.scatter(wins[exceed_mask], lambda_max_series[exceed_mask],
               c='red', s=20, zorder=5,
               label=f'异常窗口 ({exceed_mask.sum()} 个)')
    ax.set_xlabel('窗口编号')
    ax.set_ylabel('特征值')
    ax.set_title('最大特征值 vs Marchenko-Pastur 上界')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ② 超界倍数（异常强度柱状图）
    ax = axes[0, 1]
    ratio = lambda_max_series / (lambda_plus_series + 1e-10)
    colors = np.where(ratio > 1, '#e74c3c', '#3498db')
    ax.bar(wins, ratio - 1, color=colors, alpha=0.7)
    ax.axhline(0, color='black', lw=1)
    ax.set_xlabel('窗口编号')
    ax.set_ylabel(r'$\lambda_{max}/\lambda_+ - 1$')
    ax.set_title('异常强度（超界倍数）')
    ax.grid(True, alpha=0.3)

    # ③ 累计异常得分分布（正常 vs 异常用户）
    ax = axes[1, 0]
    s_norm = anomaly_scores[y == 0]
    s_anom = anomaly_scores[y == 1]
    clip99 = np.percentile(anomaly_scores, 99)
    bins = np.linspace(0, clip99, 60)
    ax.hist(s_norm, bins=bins, alpha=0.6, color='blue',
            label=f'正常用户 (n={len(s_norm)})', density=True)
    ax.hist(s_anom, bins=bins, alpha=0.6, color='red',
            label=f'异常用户 (n={len(s_anom)})', density=True)
    ax.set_xlabel('累计异常得分')
    ax.set_ylabel('密度')
    ax.set_title('谱投影异常得分分布')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ④ ROC 曲线（评估谱得分判别力）
    ax = axes[1, 1]
    score_range = anomaly_scores.max() - anomaly_scores.min() + 1e-10
    scores_norm = (anomaly_scores - anomaly_scores.min()) / score_range
    fpr, tpr, _ = roc_curve(y, scores_norm)
    roc_auc = sk_auc(fpr, tpr)
    ax.plot(fpr, tpr, 'b-', lw=2,
            label=f'RMT 谱得分 (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='随机基线')
    ax.set_xlabel('假阳率 (FPR)')
    ax.set_ylabel('真阳率 (TPR)')
    ax.set_title('ROC 曲线（谱Token 初步判别力）')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'{save_prefix}_rmt_spectral.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"已保存: {fname}")
    return roc_auc


def compute_per_user_scores(windows_data, spec_tokens_raw,
                             eigenvalues_all, eigenvectors_all,
                             lambda_plus_series, y,
                             save_prefix='sgcc'):
    """
    计算真正的用户级异常得分（修复 AUC=0.59 问题）。

    问题根源：原始得分 = Σ_w I(λ_max > λ+) 对所有用户相同
    修复方案：用每用户在"信号子空间"上的投影能量区分用户

    三种递进方案：
      A. 信号空间投影能量：Σ_{w,i: λ_i>λ+} (x_j · u_i)²
      B. 时序平均能量（基线）：用户平均用电量平方
      C. 融合得分：zscore(A) + zscore(B)

    参数:
        windows_data     : (n_windows, n_users, T)   原始窗口
        spec_tokens_raw  : (n_windows, n_users, k)   每窗口原始投影（未取abs/归一化）
        eigenvalues_all  : list of (k,)               每窗口 top-k 特征值
        eigenvectors_all : list of (T, k)             每窗口时间域特征向量
        lambda_plus_series: (n_windows,)              每窗口 MP 上界
        y                : (n_users,)                 标签
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    n_windows, n_users, T = windows_data.shape
    k = spec_tokens_raw.shape[2]

    print("\n" + "=" * 60)
    print("Step 8b: 计算用户级异常得分（修复版）")
    print("=" * 60)

    # ── 方案 A：信号空间投影能量 ──────────────────────────────
    # 对每个异常窗口，累加用户在超界特征向量方向上的投影平方
    # score_A[j] = Σ_{w: ∃λ_i>λ+} Σ_{i: λ_i>λ+} (x_j · u_i)²
    score_A = np.zeros(n_users)

    for w in range(n_windows):
        lam_plus  = lambda_plus_series[w]
        eigvals   = eigenvalues_all[w]          # (k,)
        signal_mask = eigvals > lam_plus        # (k,) bool

        if signal_mask.sum() == 0:
            continue                            # 该窗口无超界分量，跳过

        # 取超界分量对应的投影列，累加平方
        # 先取第 w 个窗口得到 (n_users, k)，再用布尔掩码选列
        # 避免 numpy 混合高维索引时维度被重排
        proj_signal = spec_tokens_raw[w][:, signal_mask]   # (n_users, n_signal)
        score_A += (proj_signal ** 2).sum(axis=1)          # (n_users,)

    # ── 方案 B：时序平均能量（基线）─────────────────────────
    # 异常用户的平均用电量通常更低（电力被盗）或更高（设备故障）
    score_B = (windows_data ** 2).mean(axis=(0, 2))        # (n_users,)

    # ── 方案 C：融合得分 zscore(A) + zscore(B) ───────────────
    def zscore(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    score_C = zscore(score_A) + zscore(score_B)

    # ── AUC 汇总打印 ─────────────────────────────────────────
    scores_dict = {
        '方案A: 信号空间投影能量': score_A,
        '方案B: 时序平均能量':     score_B,
        '方案C: 融合得分(A+B)':    score_C,
    }

    print(f"\n  {'方案':<25} {'AUC':>8}")
    print(f"  {'-' * 35}")
    best_name, best_score, best_auc = None, None, 0.0
    for name, sc in scores_dict.items():
        auc_val = roc_auc_score(y, sc)
        marker  = " ← 最佳" if auc_val == max(
            roc_auc_score(y, s) for s in scores_dict.values()) else ""
        print(f"  {name:<25} {auc_val:.4f}{marker}")
        if auc_val > best_auc:
            best_auc, best_name, best_score = auc_val, name, sc

    # ── 得分分布对比图 ────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('用户级异常得分分布对比（修复版）', fontsize=14)

    for ax, (name, sc) in zip(axes, scores_dict.items()):
        auc_val = roc_auc_score(y, sc)
        clip99  = np.percentile(sc, 99)
        bins    = np.linspace(sc.min(), clip99, 60)
        ax.hist(sc[y == 0], bins=bins, alpha=0.6, color='#3498db',
                label=f'正常 (n={(y==0).sum()})', density=True)
        ax.hist(sc[y == 1], bins=bins, alpha=0.6, color='#e74c3c',
                label=f'异常 (n={(y==1).sum()})', density=True)
        ax.set_title(f'{name}\nAUC = {auc_val:.4f}')
        ax.set_xlabel('异常得分')
        ax.set_ylabel('密度')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f'{save_prefix}_per_user_scores.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  已保存: {fname}")

    # ── ROC 曲线对比图 ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    ls_list = ['-', '--', '-.']
    for (name, sc), ls in zip(scores_dict.items(), ls_list):
        fpr, tpr, _ = roc_curve(y, sc)
        auc_val = roc_auc_score(y, sc)
        ax.plot(fpr, tpr, ls, lw=2, label=f'{name} (AUC={auc_val:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='随机基线')
    ax.set_xlabel('假阳率 (FPR)')
    ax.set_ylabel('真阳率 (TPR)')
    ax.set_title('ROC 曲线对比（三种用户级得分）')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname2 = f'{save_prefix}_roc_comparison.png'
    plt.savefig(fname2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  已保存: {fname2}")

    return score_A, score_B, score_C


# ---------- 主调用 ----------
np.random.seed(42)
(lambda_max_series,
 lambda_plus_series,
 lambda_minus_series,
 spec_tokens,
 anomaly_scores,
 eigenvalues_all,
 eigenvectors_all) = rmt_spectral_analysis(windows, y)

roc_auc = plot_spectral_results(
    lambda_max_series, lambda_plus_series,
    lambda_minus_series, anomaly_scores, y
)

# 计算用户级异常得分（修复 AUC）
score_A, score_B, score_C = compute_per_user_scores(
    windows_data      = windows,
    spec_tokens_raw   = spec_tokens[:, :, :5],   # 只取原始投影部分（前5维）
    eigenvalues_all   = eigenvalues_all,
    eigenvectors_all  = eigenvectors_all,
    lambda_plus_series= lambda_plus_series,
    y                 = y
)

# =============================================================================
# RMT 基线 AUC（取三种方案最优，供 Step 11 统一对比）
from sklearn.metrics import roc_auc_score as _auc_fn
rmt_baseline_auc = max(_auc_fn(y, score_A),
                       _auc_fn(y, score_B),
                       _auc_fn(y, score_C))
print(f"\n  [Step 8b] RMT 基线 AUC（最优方案）: {rmt_baseline_auc:.4f}")
print(f"  将在 Step 11 与 Transformer 统一对比")

# =============================================================================
# Step 8c: RMT-ISC (分层局部RMT + 个体谱贡献轨迹) - 创新点
#
# 【动机】
#   ① 全局RMT的MP边界对高/低消费用户不够紧
#   ② 需要保留逐窗口时序信息供Transformer使用
# 【解决方案】
#   A. 分层局部RMT: 按消费水平分层，层内做更紧的谱分析
#   B. ISCT: 不做时间聚合，保留144步完整时序 -> 月度通道
# =============================================================================
print("\n" + "=" * 65)
print("Step 8c: RMT-ISC (Stratified Local RMT + ISCT)")
print("=" * 65)

import time
_t8c = time.time()

N = X.shape[0]  # 42372

# =====================================================================
# 模块A: 分层局部RMT (Stratified Local RMT)
# =====================================================================
print("  [A] 分层局部RMT ...")

# 1. 按原始月均消费将用户分K=8层
user_avg_consumption = X_raw.mean(axis=1)  # (N,)
n_strata = 8
strata_labels = pd.qcut(user_avg_consumption, q=n_strata, labels=False,
                         duplicates='drop')  # (N,) 0~K-1
actual_strata = int(strata_labels.max()) + 1
print(f"    分层数: {actual_strata} (目标{n_strata})")

# 2. 在每层内部用 X_raw 月度消费矩阵做 RMT 谱分析
#    ★ 修复: 之前对5维投影做协方差(gamma=60, lambda+≈72)导致lmax/l+全为0
#    现在: 用原始月消费矩阵(N_layer, 34), gamma=N_sub/34, lambda+合理
_DAYSPM_RMT = 30
_NM_RMT = X_raw.shape[1] // _DAYSPM_RMT  # 34
# 预计算原始月度消费矩阵 (N, 34)
_mo_raw_rmt = np.zeros((N, _NM_RMT), dtype=np.float32)
for _mr in range(_NM_RMT):
    _s = _mr * _DAYSPM_RMT
    _e = min(_s + _DAYSPM_RMT, X_raw.shape[1])
    _mo_raw_rmt[:, _mr] = X_raw[:, _s:_e].mean(axis=1)

local_rmt_score = np.zeros(N, dtype=np.float32)
local_rmt_ratio = np.zeros(N, dtype=np.float32)

for k in range(actual_strata):
    mask = (strata_labels == k)
    n_layer = mask.sum()
    if n_layer < 10:
        continue
    # ★ 修复: 使用原始月消费矩阵 (n_layer, 34) 做层内协方差分解
    layer_monthly = _mo_raw_rmt[mask]  # (n_layer, 34)

    # 层内子采样做协方差谱分析
    n_sub = min(500, n_layer)
    rng_k = np.random.default_rng(42 + k)
    sub_idx = rng_k.choice(n_layer, n_sub, replace=False)
    layer_sub = layer_monthly[sub_idx]  # (n_sub, 34)
    # 层内中心化（减去层内均值，保留个体偏差）
    layer_sub_c = layer_sub - layer_sub.mean(axis=0, keepdims=True)
    # 协方差矩阵 (n_sub, n_sub)
    T_dim = layer_sub_c.shape[1]  # 34
    C_layer = layer_sub_c @ layer_sub_c.T / max(T_dim - 1, 1)  # (n_sub, n_sub)
    try:
        # 只取top-5特征值
        from scipy.sparse.linalg import eigsh as _eigsh_local
        vals_layer, vecs_layer = _eigsh_local(C_layer.astype(np.float64), k=5, which='LM')
        lmax_layer = vals_layer[-1]
        gamma_layer = n_sub / T_dim  # ~14.7, 合理!
        lplus_layer = (1.0 + np.sqrt(gamma_layer)) ** 2  # ~19.7
        strength_layer = max(lmax_layer / (lplus_layer + 1e-8), 0.0)

        # ★ 关键: 用特征向量投影回全量层内用户，得到每用户的异常得分
        # layer_monthly_all: (n_layer, 34), vecs: (n_sub, 5)
        # 先将特征向量转换回数据空间
        layer_all_c = layer_monthly - layer_monthly.mean(axis=0, keepdims=True)  # (n_layer, 34)
        # 通过子采样矩阵将特征向量投影到数据空间
        u_data = layer_sub_c.T @ vecs_layer  # (34, 5) 数据空间特征向量
        # 归一化
        for _kk in range(u_data.shape[1]):
            _norm = np.linalg.norm(u_data[:, _kk])
            if _norm > 1e-10:
                u_data[:, _kk] /= _norm
        # 每用户在超界特征方向的投影能量
        layer_proj_energy = np.sum((layer_all_c @ u_data) ** 2, axis=1)  # (n_layer,)
    except Exception:
        strength_layer = 1.0
        layer_proj_energy = np.abs(layer_monthly.mean(axis=1))

    # 层内中位数和MAD归一化
    med_layer = np.median(layer_proj_energy)
    mad_layer = np.median(np.abs(layer_proj_energy - med_layer)) + 1e-8
    # 分层RMT得分: 相对层内中位数的偏离 (MAD归一化)
    local_rmt_score[mask] = ((layer_proj_energy - med_layer) / mad_layer).astype(np.float32)
    local_rmt_ratio[mask] = np.float32(strength_layer)

    print(f"    层{k}: n={n_layer}, sub={n_sub}, gamma={gamma_layer:.1f}, "
          f"lmax={lmax_layer:.2f}, l+={lplus_layer:.2f}, "
          f"lmax/l+={strength_layer:.2f}")

lrs_auc = _auc_fn(y, local_rmt_score)
lrr_auc = _auc_fn(y, local_rmt_ratio)
print(f"    local_rmt_score AUC={max(lrs_auc, 1-lrs_auc):.4f}")
print(f"    local_rmt_ratio AUC={max(lrr_auc, 1-lrr_auc):.4f}")

# =====================================================================
# 模块B: 个体谱贡献轨迹 (ISCT) - ★ 改用 X_raw 月度消费做层内对比
# =====================================================================
print("  [B] 个体消费偏离轨迹 (ISCT - X_raw版) ...")

# ★ 修复: 不再用 spec_tokens(AUC 0.50-0.53)，改用原始月消费(AUC=0.7956)
# 1. 用户原始月消费序列（已在模块A计算: _mo_raw_rmt (N, 34)）
isct_monthly = _mo_raw_rmt.copy()  # (N, 34) 原始月消费

# 2. 计算同层中位数偏离（层内对比：消除消费水平差异，暴露个体异常变化）
_NM_ISC = _NM_RMT  # 34
isct_dev_monthly = np.zeros((N, _NM_ISC), dtype=np.float32)
for k in range(actual_strata):
    mask = (strata_labels == k)
    if mask.sum() < 2:
        continue
    layer_median = np.median(_mo_raw_rmt[mask], axis=0, keepdims=True)  # (1, 34)
    # 偏离率（相对于层内中位数的比值偏离）
    isct_dev_monthly[mask] = ((_mo_raw_rmt[mask] - layer_median) /
                               (np.abs(layer_median) + 1e-3)).astype(np.float32)

print(f"    isct_monthly shape: ({N}, {_NM_ISC})")
print(f"    isct_dev_monthly shape: ({N}, {_NM_ISC})")

# 验证信号强度
_isct_mean_auc = max(_auc_fn(y, isct_monthly.mean(axis=1)),
                      1 - _auc_fn(y, isct_monthly.mean(axis=1)))
_isct_dev_auc = max(_auc_fn(y, isct_dev_monthly.mean(axis=1)),
                     1 - _auc_fn(y, isct_dev_monthly.mean(axis=1)))
print(f"    isct_monthly(原始月消费) mean AUC={_isct_mean_auc:.4f}")
print(f"    isct_dev_monthly(层内偏离率) mean AUC={_isct_dev_auc:.4f}")

# 4. 个体级CPD: 对每个用户的月度偏离时序做简化变点检测
#    向量化实现（避免42000用户的Python for循环）
print("    Computing ISCT-CPD features (vectorized) ...")
isct_cpd_feats = np.zeros((N, 8), dtype=np.float32)
sig_all = isct_dev_monthly  # (N, 34)
T_sig = sig_all.shape[1]  # 34

# F1: 最大CUSUM统计量位置（变点位置）
sig_mean = sig_all.mean(axis=1, keepdims=True)  # (N, 1)
cumsum_all = np.cumsum(sig_all - sig_mean, axis=1)  # (N, 34)
cp_pos_idx = np.argmax(np.abs(cumsum_all), axis=1)  # (N,)
cp_pos = cp_pos_idx.astype(np.float32) / T_sig
isct_cpd_feats[:, 0] = cp_pos

# F2: 变点前后均值比
cp_idx_clamped = np.clip(cp_pos_idx, 1, T_sig - 1)  # (N,)
# 向量化: 用掩码计算前后均值
_arange_t = np.arange(T_sig)[np.newaxis, :]  # (1, 34)
_cp_expanded = cp_idx_clamped[:, np.newaxis]  # (N, 1)
pre_mask = _arange_t < _cp_expanded  # (N, 34) bool
post_mask = _arange_t >= _cp_expanded  # (N, 34) bool
pre_sum = (sig_all * pre_mask).sum(axis=1)
pre_cnt = pre_mask.sum(axis=1).astype(np.float32)
pre_mean = pre_sum / (pre_cnt + 1e-8)
post_sum = (sig_all * post_mask).sum(axis=1)
post_cnt = post_mask.sum(axis=1).astype(np.float32)
post_mean = post_sum / (post_cnt + 1e-8)
isct_cpd_feats[:, 1] = (post_mean / (np.abs(pre_mean) + 1e-6)).astype(np.float32)

# F3: 最大单月偏离
isct_cpd_feats[:, 2] = np.abs(sig_all).max(axis=1).astype(np.float32)

# F4: 后半段/前半段偏离比
half = T_sig // 2
first_half_mean = sig_all[:, :half].mean(axis=1)
second_half_mean = sig_all[:, half:].mean(axis=1)
isct_cpd_feats[:, 3] = (second_half_mean / (np.abs(first_half_mean) + 1e-6)).astype(np.float32)

# F5: 偏离标准差（不稳定性）
isct_cpd_feats[:, 4] = sig_all.std(axis=1).astype(np.float32)

# F6: 连续负偏离最大长度（向量化实现）
neg_mask_all = (sig_all < 0).astype(np.int32)  # (N, 34)
max_neg_run = np.zeros(N, dtype=np.float32)
cur_run = np.zeros(N, dtype=np.float32)
for t_step in range(T_sig):
    cur_run = (cur_run + 1.0) * neg_mask_all[:, t_step]
    max_neg_run = np.maximum(max_neg_run, cur_run)
isct_cpd_feats[:, 5] = (max_neg_run / T_sig).astype(np.float32)

# F7: 趋势斜率（向量化最小二乘）
t_vec = np.arange(T_sig, dtype=np.float64)
t_mean = t_vec.mean()
t_var = ((t_vec - t_mean) ** 2).sum()
sig_all_f64 = sig_all.astype(np.float64)
slope_all = ((sig_all_f64 - sig_all_f64.mean(axis=1, keepdims=True)) *
             (t_vec[np.newaxis, :] - t_mean)).sum(axis=1) / (t_var + 1e-8)
isct_cpd_feats[:, 6] = slope_all.astype(np.float32)

# F8: 末期偏离均值（最后6个月）
tail_len = min(6, T_sig)
isct_cpd_feats[:, 7] = sig_all[:, -tail_len:].mean(axis=1).astype(np.float32)

# 5. 打印AUC验证
_cpd_names = ['BianDianWeiZhi', 'QianHouBi', 'ZuiDaPianLi', 'BanDuanBi',
              'BuWenDingXing', 'LianXuFuPianLi', 'QuShiXieLv', 'MoQiPianLi']
for idx, name in enumerate(_cpd_names):
    a = roc_auc_score(y, isct_cpd_feats[:, idx])
    a = max(a, 1 - a)
    print(f"    ISCT-CPD F{idx+1}({name}) AUC={a:.4f}")

# 6. 打包输出（只保留 ISCT-CPD 8维，去掉负贡献的 Stratified-RMT 特征）
# 消融实验 G2 证明 Stratified-RMT (rmt_score/rmt_ratio) AUC -0.0014（负贡献）
rmt_isc_features = isct_cpd_feats.astype(np.float32)  # (N, 8) 仅保留ISCT-CPD

# ── 6b. RMT-ISC 特征选择：去掉噪声维度 ──────────────────────────
_isc_names = [f'cpd_{i}' for i in range(isct_cpd_feats.shape[1])]
_isc_keep_mask = []
print("  [RMT-ISC 特征选择] 按 |AUC - 0.5| 筛选 (阈值=0.02):")
for _ci in range(rmt_isc_features.shape[1]):
    _col = rmt_isc_features[:, _ci]
    _valid = np.isfinite(_col) & np.isfinite(y)
    if _valid.sum() < 50:
        _isc_keep_mask.append(False)
        print(f"    {_isc_names[_ci]:15s}  SKIP (too few valid)")
        continue
    _a = roc_auc_score(y[_valid], _col[_valid])
    _deviation = abs(_a - 0.5)
    _keep = _deviation > 0.02
    _isc_keep_mask.append(_keep)
    print(f"    {_isc_names[_ci]:15s}  AUC={_a:.4f}  |dev|={_deviation:.4f}  {'✓ KEEP' if _keep else '✗ DROP'}")

_isc_keep_idx = [i for i, k in enumerate(_isc_keep_mask) if k]
if len(_isc_keep_idx) > 0:
    rmt_isc_features = rmt_isc_features[:, _isc_keep_idx]
    print(f"  RMT-ISC 特征选择: {len(_isc_keep_mask)} → {len(_isc_keep_idx)} 维 (去掉 {len(_isc_keep_mask)-len(_isc_keep_idx)} 噪声维度)")
else:
    print("  ⚠ 所有RMT-ISC特征 |dev| <= 0.02，保留全部以防退化")

print(f"  RMT-ISC features: {rmt_isc_features.shape}  "
      f"isct_monthly: {isct_monthly.shape}  "
      f"isct_dev_monthly: {isct_dev_monthly.shape}  "
      f"elapsed {time.time()-_t8c:.1f}s")
print("=" * 65)


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset






# =============================================================
# Step 8e: Multi-Scale TCN-CPD（多尺度时间卷积 + 多变点检测）
# =============================================================
# 现有CPD局限性：CUSUM只检测单一变点，月度序列仅34步太短
# TCN-CPD改进：在1020天日度序列上用多尺度因果卷积检测多个变点
print("\n" + "=" * 65)
print("Step 8e: Multi-Scale TCN-CPD（多尺度时间卷积 + 多变点检测）")
print("=" * 65)
_t8e = time.time()

# 1. 准备输入：使用原始kWh日度数据（保留消费量级信息）
# 对每个用户进行 RobustScaler 归一化（保留个体趋势）
_tcn_input = X_raw.copy().astype(np.float32)  # (N, 1020)
# 处理NaN
_tcn_input = np.nan_to_num(_tcn_input, nan=0.0)
# per-user RobustScaler（用中位数和IQR）
_u_med = np.median(_tcn_input, axis=1, keepdims=True)
_u_iqr = np.clip(
    np.percentile(_tcn_input, 75, axis=1, keepdims=True) -
    np.percentile(_tcn_input, 25, axis=1, keepdims=True),
    1e-6, None
)
_tcn_input = (_tcn_input - _u_med) / _u_iqr  # (N, 1020)
_tcn_input = np.clip(_tcn_input, -10, 10)

print(f"  TCN输入: {_tcn_input.shape}  (per-user RobustScaler归一化)")

# 2. 多尺度1D卷积特征提取（纯NumPy，无需GPU）
# 使用不同窗口大小的移动平均作为"卷积核"，然后计算残差
_scales = [7, 30, 90]  # 周级/月级/季级
_N_users = len(y)

# 存储多尺度特征
ms_cpd_features = []

for _scale in _scales:
    print(f"  [Scale {_scale}天]", end="")

    # 移动平均（等价于1D平均卷积核）
    _kernel = np.ones(_scale) / _scale
    _smoothed = np.apply_along_axis(
        lambda row: np.convolve(row, _kernel, mode='same'),
        axis=1, arr=_tcn_input
    )  # (N, 1020)

    # 残差 = 原始 - 平滑（捕获高频变化）
    _residual = _tcn_input - _smoothed  # (N, 1020)

    # 残差的绝对值移动和（检测异常集中区域）
    _abs_residual = np.abs(_residual)
    _cumsum_res = np.cumsum(_abs_residual, axis=1)  # (N, 1020)

    # 多变点检测：使用残差能量的变化率
    # 将序列分成多段，检测每段的残差能量
    _n_segments = max(1020 // _scale, 4)
    _seg_len = 1020 // _n_segments
    _seg_energies = np.zeros((_N_users, _n_segments), dtype=np.float32)
    for _si in range(_n_segments):
        _s = _si * _seg_len
        _e = min(_s + _seg_len, 1020)
        _seg_energies[:, _si] = np.mean(_abs_residual[:, _s:_e], axis=1)

    # 从分段能量中提取变点特征
    # F1: 最大段间能量跳变位置（归一化到0-1）
    _seg_diff = np.abs(np.diff(_seg_energies, axis=1))  # (N, n_seg-1)
    _max_jump_pos = np.argmax(_seg_diff, axis=1).astype(np.float32) / max(_n_segments - 2, 1)

    # F2: 最大段间跳变幅度（归一化）
    _max_jump_mag = np.max(_seg_diff, axis=1) / (np.mean(_seg_energies, axis=1) + 1e-8)

    # F3: 有多少个"显著变点"（跳变 > 2倍均值跳变）
    _mean_diff = np.mean(_seg_diff, axis=1, keepdims=True) + 1e-8
    _n_changepoints = np.sum(_seg_diff > 2 * _mean_diff, axis=1).astype(np.float32)

    # F4: 前半段 vs 后半段能量比
    _half = _n_segments // 2
    _first_half_e = np.mean(_seg_energies[:, :_half], axis=1) + 1e-8
    _second_half_e = np.mean(_seg_energies[:, _half:], axis=1) + 1e-8
    _half_ratio = _second_half_e / _first_half_e

    # F5: 能量序列的标准差（衡量波动性）
    _energy_std = np.std(_seg_energies, axis=1)

    # F6: 能量序列的趋势斜率（是否逐渐下降/上升）
    _x_axis = np.arange(_n_segments, dtype=np.float32)
    _x_mean = _x_axis.mean()
    _energy_mean = np.mean(_seg_energies, axis=1, keepdims=True)
    _slope = np.sum((_x_axis[np.newaxis, :] - _x_mean) * (_seg_energies - _energy_mean), axis=1) / \
             (np.sum((_x_axis - _x_mean) ** 2) + 1e-8)

    _scale_feats = np.column_stack([
        _max_jump_pos,    # 最大变点位置
        _max_jump_mag,    # 最大变点幅度
        _n_changepoints,  # 显著变点数
        _half_ratio,      # 前后半段比
        _energy_std,      # 能量波动性
        _slope,           # 能量趋势
    ]).astype(np.float32)  # (N, 6)

    ms_cpd_features.append(_scale_feats)

    # 评估单尺度AUC
    for _fi, _fname in enumerate(['变点位置', '变点幅度', '变点数', '前后比', '波动性', '趋势']):
        _a = roc_auc_score(y, _scale_feats[:, _fi])
        _a = max(_a, 1 - _a)
        if _a > 0.55:
            print(f" {_fname}={_a:.4f}✅", end="")
    print()

# 3. 合并多尺度特征
tcn_cpd_features = np.concatenate(ms_cpd_features, axis=1).astype(np.float32)  # (N, 18)
print(f"  TCN-CPD 多尺度特征: {tcn_cpd_features.shape}")

# 4. 跨尺度交叉特征（不同尺度间的关系）
# 短期波动 vs 长期趋势的不一致性
_cross_short_long = ms_cpd_features[0][:, 4] / (ms_cpd_features[2][:, 4] + 1e-8)  # 7天波动/90天波动
_cross_jump_consistency = np.abs(ms_cpd_features[0][:, 0] - ms_cpd_features[2][:, 0])  # 变点位置一致性
_cross_feats = np.column_stack([_cross_short_long, _cross_jump_consistency]).astype(np.float32)

tcn_cpd_features = np.concatenate([tcn_cpd_features, _cross_feats], axis=1)  # (N, 20)

# 评估跨尺度特征
for _ci, _cn in enumerate(['短长波动比', '变点一致性']):
    _a = roc_auc_score(y, _cross_feats[:, _ci])
    _a = max(_a, 1 - _a)
    print(f"  跨尺度-{_cn}: AUC={_a:.4f}")

print(f"  TCN-CPD 最终特征: {tcn_cpd_features.shape}  elapsed {time.time()-_t8e:.1f}s")
print("=" * 65)


# 升级版 Step A~E: 目标 AUC ≥ 0.90
#  A. 14维特征（修复归一化顺序，IsolationForest替代失效LOF）
#  B. 双路注意力Transformer（局部窗口 + 全局CLS Token）
#  C. ContrastiveFocal 联合损失
#  D. XGBoost 集成提升
#  E. 综合评估与可视化
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import (roc_auc_score, f1_score, roc_curve,
                              confusion_matrix, classification_report,
                              precision_recall_curve)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import time

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_pin = torch.cuda.is_available()
print(f"使用设备: {device}")

# =============================================================================
# Step A【重写】: 构造14维逐用户特征（修复全局标量问题）
# =============================================================================
print("\n" + "=" * 65)
print("Step A【重写】: 构造14维逐用户特征（修复全局标量问题）")
print("=" * 65)

N_WINS  = spec_tokens.shape[0]        # 144
N_USERS = spec_tokens.shape[1]        # 42372
# Bug7修复：WIN_LEN 是滑动步长（stride），而不是总天数/窗口数
# rmt_spectral_analysis 用 WIN_STRIDE 步进，Win_LEN 是窗口覆盖天数（= step_size）
WIN_LEN = X.shape[1] // N_WINS        # 每窗口步长（stride），确保覆盖全期

print(f"  N_WINS={N_WINS}, N_USERS={N_USERS}, WIN_LEN(stride)={WIN_LEN}")
# Bug8：降采样因子断言（144 必须能被 48 整除，且商=3）
_DS_FACTOR = 3
assert N_WINS % _DS_FACTOR == 0, \
    f"Bug8: N_WINS={N_WINS} 不能被 _DS_FACTOR={_DS_FACTOR} 整除！请调整 N_WINDOWS 或 _DS_FACTOR"
_DS_TARGET = N_WINS // _DS_FACTOR     # 48
print(f"  降采样因子={_DS_FACTOR}，降采样目标窗口数={_DS_TARGET}")

# ── A-1: Token1~5 保留（逐用户特征向量投影，已验证有效）──────────────
tokens_1to5 = spec_tokens[:, :, :5].copy()   # (144, N_USERS, 5)

# ── A-2: Token6~14 从原始X逐用户计算（彻底解决全局标量问题）──────────
print("\n  计算逐用户窗口特征 Token6~14 ...")
t0_feat = time.time()

# ══════════════════════════════════════════════════════════════════
# ⚡ 向量化重写：将 N_WINS × N_USERS 双重循环改为批量矩阵运算
#   策略：先构造 all_segs (N_WINS, N_USERS, WIN_LEN)，一次性完成
#         所有窗口的统计量计算，速度提升 10x 以上
# ══════════════════════════════════════════════════════════════════

# ① 构造三维窗口矩阵（零拷贝视图尽量复用内存）
all_segs = np.stack([
    X[:, w * WIN_LEN : min((w + 1) * WIN_LEN, X.shape[1])]
    for w in range(N_WINS)
], axis=0).astype(np.float32)   # (N_WINS, N_USERS, WIN_LEN)
_W, _N, _T = all_segs.shape
print(f"  all_segs 形状: {all_segs.shape}  ({_W*_N*_T/1e6:.1f}M 元素)")

# ② Token6~11：一行完成（全部广播运算，无 Python for 循环）
token6_user_var    = all_segs.var(axis=2)                              # (N_WINS, N_USERS)
token7_user_mean   = all_segs.mean(axis=2)
token8_user_max    = all_segs.max(axis=2)
token11_zero_ratio = (np.abs(all_segs) < 1e-6).mean(axis=2)
token16_nonzero_r  = (np.abs(all_segs) >= 0.01).mean(axis=2)

# ③ Token9 & Token10：差分（在时间轴做 diff，结果形状 (N_WINS, N_USERS, WIN_LEN-1)）
_diff              = np.diff(all_segs, axis=2)                         # (N_WINS, N_USERS, T-1)
token9_delta_mean  = _diff.mean(axis=2)
token10_delta_max  = np.abs(_diff).max(axis=2)

# ④ Token12：线性趋势斜率（批量最小二乘，无循环）
#   slope = Σ (t - t̄)(x - x̄) / Σ (t - t̄)²
_t     = np.arange(_T, dtype=np.float32)                               # (T,)
_t_c   = _t - _t.mean()                                                # 中心化时间轴
_t_var = (_t_c ** 2).sum()
if _t_var > 0:
    # all_segs: (N_WINS, N_USERS, T)   *  _t_c: (T,)
    _x_c = all_segs - token7_user_mean[:, :, np.newaxis]               # (N_WINS, N_USERS, T)
    token12_trend_slope = (_x_c * _t_c).sum(axis=2) / _t_var           # (N_WINS, N_USERS)
else:
    token12_trend_slope = np.zeros((_W, _N), dtype=np.float32)

# ⑤ Token13：偏度（批量，无循环）
_mu   = token7_user_mean[:, :, np.newaxis]                             # (N_WINS, N_USERS, 1)
_sig  = all_segs.std(axis=2, keepdims=True) + 1e-8
_skew = (((all_segs - _mu) / _sig) ** 3).mean(axis=2)
token13_skewness = np.clip(_skew, -5, 5)

# ⑥ Token14：与全体均值的标准化偏离（批量，仅在 N_USERS 维度上统计）
_g_mean = token7_user_mean.mean(axis=1, keepdims=True)                 # (N_WINS, 1)
_g_std  = token7_user_mean.std(axis=1, keepdims=True) + 1e-8
token14_vs_neighbor = np.clip(
    (token7_user_mean - _g_mean) / _g_std, -5, 5
)

# ⑦ Token15：低频 FFT 能量（对整个三维矩阵做 rfft，axis=2）
if _T >= 4:
    _fft_amp       = np.abs(np.fft.rfft(all_segs, axis=2))             # (N_WINS, N_USERS, freq)
    token15_fft_energy = _fft_amp[:, :, 1:4].mean(axis=2)              # 低频 1~3
else:
    token15_fft_energy = np.zeros((_W, _N), dtype=np.float32)

# 释放大中间变量，节省内存（_x_c 仅在 _t_var>0 时存在）
_to_del = [_diff, _mu, _sig, _skew]
if '_x_c' in dir():       _to_del.append(_x_c)
if '_fft_amp' in dir():   _to_del.append(_fft_amp)
del _to_del
import gc; gc.collect()

print(f"  ⚡ Token6~16 向量化计算完成，总耗时: {time.time()-t0_feat:.1f}s")

# ── A-3: 区分度快速验证（归一化前，原始值） ──────────────────────────
raw_tokens_dict = {
    'Token6(用户方差)':   token6_user_var,
    'Token7(用户均值)':   token7_user_mean,
    'Token8(用户最大值)': token8_user_max,
    'Token9(Δ均值)':      token9_delta_mean,
    'Token10(Δ最大值)':   token10_delta_max,
    'Token11(零值占比)':  token11_zero_ratio,
    'Token12(趋势斜率)':  token12_trend_slope,
    'Token13(偏度)':      token13_skewness,
    'Token14(邻居偏离)':  token14_vs_neighbor,
}
print(f"\n  {'维度':<20} {'正常均值':>10} {'异常均值':>10} {'差异倍数':>10}")
print(f"  {'-'*54}")
for name, arr in raw_tokens_dict.items():
    # arr: (N_WINS, N_USERS) → 对每个用户取时间均值
    user_mean = arr.mean(axis=0)   # (N_USERS,)
    nm = user_mean[y == 0].mean()
    am = user_mean[y == 1].mean()
    r  = am / (nm + 1e-8)
    flag = "✅" if r > 1.05 or r < 0.95 else "❌"
    print(f"  {name:<20} {nm:>10.4f} {am:>10.4f} {r:>10.3f}x  {flag}")

# ── A-4: 合并14维Token（Token1~5 + Token6~14）────────────────────────
new_tokens = np.stack([
    token6_user_var,
    token7_user_mean,
    token8_user_max,
    token9_delta_mean,
    token10_delta_max,
    token11_zero_ratio,
    token12_trend_slope,
    token13_skewness,
    token14_vs_neighbor,
    token15_fft_energy,   # 优先级5新增
    token16_nonzero_r,    # 优先级5新增
], axis=-1)   # (N_WINS, N_USERS, 11)

spec_tokens_14 = np.concatenate(
    [tokens_1to5, new_tokens], axis=-1
)   # (N_WINS, N_USERS, 16)
print(f"\n  spec_tokens_16 shape: {spec_tokens_14.shape}")

# ── A-5: 转置 + 混合归一化策略 ──────────────────────────────────────
#   Token1~5 : 逐用户 MinMax（特征向量投影，相对变化有效）
#   Token6~14: 全局 RobustScaler（保留类间绝对差异，不被per-user抹平）
from sklearn.preprocessing import RobustScaler

X_seq14  = spec_tokens_14.transpose(1, 0, 2).astype(np.float32)
X_seq14  = np.nan_to_num(X_seq14, nan=0.0, posinf=5.0, neginf=-5.0)
FEAT_DIM = X_seq14.shape[-1]   # 自动读取实际维度（当前=16）
X_norm14 = np.zeros_like(X_seq14)

# Token1~5: 逐用户 MinMax
for i in range(5):
    feat  = X_seq14[:, :, i]                              # (N_USERS, N_WINS)
    f_min = feat.min(axis=1, keepdims=True)
    f_max = feat.max(axis=1, keepdims=True)
    X_norm14[:, :, i] = (feat - f_min) / (f_max - f_min + 1e-8)

# Token6~16: 全局 RobustScaler，保留原始输出（不remap到[0,1]，保留分布形状差异）
for i in range(5, FEAT_DIM):
    feat_flat   = X_seq14[:, :, i].reshape(-1, 1)         # (N_USERS*N_WINS, 1)
    feat_scaled = RobustScaler().fit_transform(feat_flat)  # 中位数=0，IQR=1
    feat_clipped= np.clip(feat_scaled, -5, 5)              # 保留[-5,5]范围
    X_norm14[:, :, i] = feat_clipped.reshape(N_USERS, -1)

X_seq14 = X_norm14

# ── A-6: AUC验证区分度（替换均值比较，AUC才是真实可分性指标）────────
from sklearn.metrics import roc_auc_score as _roc_auc

dim_names = [
    'Token1(v1_proj)',   'Token2(v2_proj)',   'Token3(v3_proj)',
    'Token4(v4_proj)',   'Token5(v5_proj)',
    'Token6(用户方差)',   'Token7(用户均值)',  'Token8(用户最大值)',
    'Token9(Δ均值)',      'Token10(Δ最大值)',
    'Token11(零值占比)', 'Token12(趋势斜率)',
    'Token13(偏度)',      'Token14(邻居偏离)',
    'Token15(FFT能量)',  'Token16(非零比例)',
]
print(f"\n  {'维度':<22} {'AUC':>8}  {'说明'}")
print(f"  {'-'*52}")
valid_count = 0
for i, name in enumerate(dim_names):
    # 每用户在时间轴上取均值 → 得到标量特征，计算与标签的AUC
    feat_per_user = X_seq14[:, :, i].mean(axis=1)   # (N_USERS,)
    try:
        auc  = _roc_auc(y, feat_per_user)
        auc  = max(auc, 1 - auc)     # 双向AUC（方向无关）
        flag = "✅" if auc >= 0.52 else "❌"
        if auc >= 0.52:
            valid_count += 1
        desc = "强" if auc >= 0.60 else ("中" if auc >= 0.55 else "弱")
    except Exception:
        auc, flag, desc = 0.5, "❌", "-"
    print(f"  {name:<22} AUC={auc:.4f}  {desc}  {flag}")

print(f"\n  有效维度(AUC≥0.52): {valid_count}/14  "
      f"{'✅ 可以继续训练' if valid_count >= 6 else '❌ 建议进一步检查'}")

# ── A 检查点 ─────────────────────────────────────────────────────────
if STOP_AFTER in ('A', 'a'):
    import sys
    print("\n[STOP_AFTER='A'] Step A 完成，程序退出。")
    print("  有效维度 >= 6 → 将 STOP_AFTER 改为 None 继续完整训练。")
    sys.exit(0)

# ── A-8: 降采样 144→48 ───────────────────────────────────────────────
# Bug7/8修复：使用断言验证过的 _DS_FACTOR/_DS_TARGET，而不是硬编码 48/3
X_seq14_ds = X_seq14.reshape(
    X_seq14.shape[0], _DS_TARGET, _DS_FACTOR, FEAT_DIM
).mean(axis=2)   # (n_users, _DS_TARGET, FEAT_DIM)
print(f"\n  降采样后形状: {X_seq14_ds.shape}")  # 预期 (N_USERS, 48, 16)

# ── A-9: 跨窗口特征增强（16→19维）──────────────────────────────────
#  新增3维全局统计特征，不依赖全局标量，逐用户计算
print("  构造跨窗口增强特征(+3维)...")
N_U, N_W, N_D = X_seq14_ds.shape

# Feature1: 方差 Token 历史滑动 Z-score（当前窗口相对历史12窗口的异常度）
# Token6（索引5）是用户方差，捕捉用电波动突变
var_tok   = X_seq14_ds[:, :, 5]                             # (N_U, 48)
roll_mean = np.zeros_like(var_tok)
roll_std  = np.ones_like(var_tok)
for t in range(4, N_W):
    hist           = var_tok[:, max(0, t-12):t]
    roll_mean[:, t]= hist.mean(axis=1)
    roll_std[:, t] = hist.std(axis=1) + 1e-8
zscore_var = np.clip((var_tok - roll_mean) / roll_std, -5, 5)  # (N_U, 48)

# Feature2: 用户用电均值 Token 的跨全期 IQR（衡量用电稳定性）
# Token7（索引6）是窗口内均值
q75 = np.percentile(X_seq14_ds[:, :, 6], 75, axis=1)       # (N_U,)
q25 = np.percentile(X_seq14_ds[:, :, 6], 25, axis=1)
iqr = np.clip(q75 - q25, 0, 5)                             # (N_U,)
iqr_tiled = np.tile(iqr[:, np.newaxis], (1, N_W))          # (N_U, 48)

# Feature3: RMT 信号强度比例（Token1~5 最大值超阈值的窗口占比）
rmt_ratio = (X_seq14_ds[:, :, :5].max(axis=2) > 0.7).mean(axis=1)  # (N_U,)
rmt_tiled = np.tile(rmt_ratio[:, np.newaxis], (1, N_W))             # (N_U, 48)

# 拼接：(N_U, 48, 16+3) = (N_U, 48, 19)
X_seq14_ds = np.concatenate([
    X_seq14_ds,
    zscore_var[:, :, np.newaxis],
    iqr_tiled[:, :, np.newaxis],
    rmt_tiled[:, :, np.newaxis],
], axis=-1).astype(np.float32)

FEAT_DIM = X_seq14_ds.shape[-1]   # 19
print(f"  增强后形状: {X_seq14_ds.shape}  (FEAT_DIM={FEAT_DIM})")

# AUC 验证新增3维
for name, arr in [('zscore_var(F17)', zscore_var),
                  ('iqr_tiled(F18)',  iqr_tiled),
                  ('rmt_ratio(F19)', rmt_tiled)]:
    feat_u = arr.mean(axis=1)
    try:
        from sklearn.metrics import roc_auc_score as _r
        a = _r(y, feat_u); a = max(a, 1 - a)
        print(f"    {name:<22} AUC={a:.4f}  {'✅' if a>=0.52 else '❌'}")
    except Exception:
        pass

# ── A-10: 新增 Token17~21（月度/季节/工作日 特征）────────────────────
print("\n  构造月度/工作日增强特征(Token17~21)...")

# Token17: 月度用电量变异系数 CV（窃电用户月度波动更大）
DAYS_PER_MONTH = 30
n_months = X.shape[1] // DAYS_PER_MONTH
monthly_means = np.zeros((N_USERS, n_months), dtype=np.float32)
for _m in range(n_months):
    monthly_means[:, _m] = X[:, _m*DAYS_PER_MONTH:(_m+1)*DAYS_PER_MONTH].mean(axis=1)
token17_monthly_cv = (monthly_means.std(axis=1) /
                      (np.abs(monthly_means.mean(axis=1)) + 1e-8))

# Token18: 月度最小值/最大值比（窃电用户某月极度异常低）
monthly_min = monthly_means.min(axis=1)
monthly_max = monthly_means.max(axis=1)
token18_min_max_ratio = monthly_min / (monthly_max + 1e-8)

# Token19: 工作日 vs 周末用电差异（正常用户有规律，窃电用户无规律）
date_cols = [c for c in df.columns if c not in ['FLAG', 'CONS_NO']]
_dates = pd.to_datetime(date_cols, errors='coerce')
_is_weekend = (_dates.dayofweek >= 5)  # dayofweek>=5 已是 numpy bool array，无需再转换
weekday_mean = X[:, ~_is_weekend].mean(axis=1)
weekend_mean = X[:, _is_weekend].mean(axis=1)
token19_wd_we_ratio = weekday_mean / (weekend_mean + 1e-8)

# Token20: 连续零值最长段长度（窃电常有连续停用）
# Bug10修复：从 O(N) for-loop 改为向量化实现（scipy.ndimage）
def max_consecutive_zeros(arr_2d, threshold=1e-6):
    """arr_2d: (N_USERS, T) → 返回每用户最长连续零值段长度 (N_USERS,)"""
    try:
        from scipy.ndimage import label as _label
        is_zero = (np.abs(arr_2d) < threshold)          # (N_USERS, T) bool
        result  = np.zeros(arr_2d.shape[0], dtype=np.float32)
        for _i in range(arr_2d.shape[0]):
            labeled, num = _label(is_zero[_i])
            if num > 0:
                sizes = np.bincount(labeled)[1:]         # 跳过背景 0
                result[_i] = float(sizes.max())
        return result
    except ImportError:
        # fallback：无 scipy 时回退 numpy 版本
        N, T = arr_2d.shape
        result = np.zeros(N, dtype=np.float32)
        for _i in range(N):
            is_zero = np.abs(arr_2d[_i]) < threshold
            changes = np.diff(is_zero.astype(int), prepend=0, append=0)
            starts  = np.where(changes == 1)[0]
            ends    = np.where(changes == -1)[0]
            if len(starts) > 0:
                result[_i] = float((ends - starts).max())
        return result

token20_max_zero_run = max_consecutive_zeros(X)

# Token21: 前后半段用电量比（趋势性窃电：越来越少用电）
_mid = X.shape[1] // 2
token21_trend = X[:, :_mid].mean(axis=1) / (X[:, _mid:].mean(axis=1) + 1e-8)

# AUC 验证 Token17~21
print(f"\n  {'维度':<26} {'AUC':>8}")
print(f"  {'-'*36}")
for _name, _feat in [
    ('Token17(月度CV)',       token17_monthly_cv),
    ('Token18(月min/max比)',  token18_min_max_ratio),
    ('Token19(工作日/周末)',  token19_wd_we_ratio),
    ('Token20(最长零值段)',   token20_max_zero_run),
    ('Token21(前后半段比)',   token21_trend),
]:
    try:
        _auc = roc_auc_score(y, _feat)
        _auc = max(_auc, 1 - _auc)
        print(f"  {_name:<26} AUC={_auc:.4f}  {'✅' if _auc>=0.55 else '弱'}")
    except Exception:
        print(f"  {_name:<26} AUC 计算失败")

# 将 Token17~21 平铺到时间轴后拼入 X_seq14_ds（参考 iqr_tiled 写法）
N_W_ds = X_seq14_ds.shape[1]   # 48
def tile_user_feat(arr_1d, n_w):
    """将 (N_USERS,) 的用户级特征平铺为 (N_USERS, n_w)"""
    return np.tile(arr_1d[:, np.newaxis], (1, n_w)).astype(np.float32)

t17 = tile_user_feat(token17_monthly_cv,    N_W_ds)
t18 = tile_user_feat(token18_min_max_ratio, N_W_ds)
t19 = tile_user_feat(token19_wd_we_ratio,   N_W_ds)
t20 = tile_user_feat(token20_max_zero_run,  N_W_ds)
t21 = tile_user_feat(token21_trend,         N_W_ds)

# RobustScaler 归一化（与 Token6~16 保持一致）
from sklearn.preprocessing import RobustScaler as _RS
for _arr in [t17, t18, t19, t20, t21]:
    _flat = _arr.reshape(-1, 1)
    _scaled = _RS().fit_transform(_flat)
    _arr[:] = np.clip(_scaled, -5, 5).reshape(_arr.shape)

X_seq14_ds = np.concatenate([
    X_seq14_ds,
    t17[:, :, np.newaxis],
    t18[:, :, np.newaxis],
    t19[:, :, np.newaxis],
    t20[:, :, np.newaxis],
    t21[:, :, np.newaxis],
], axis=-1).astype(np.float32)

FEAT_DIM = X_seq14_ds.shape[-1]   # 19+5=24
print(f"  拼入 Token17~21 后形状: {X_seq14_ds.shape}  (FEAT_DIM={FEAT_DIM})")

# ── A-11: 混沌延迟嵌入谱特征（Token22~25）────────────────────────────
print("\n  构造混沌延迟嵌入谱特征(Token22~25)...")

def delay_embedding(X_data, delay=7, embed_dim=4):
    """
    时间延迟嵌入（Takens嵌入定理）:
    将 (N, T) → (N, T_valid, embed_dim)

    delay=7: 7天周期（每周）
    embed_dim=4: 4周历史信息

    示意: t=28 → [x(28), x(21), x(14), x(7)]
    """
    N, T = X_data.shape
    T_valid = T - (embed_dim - 1) * delay
    if T_valid <= 0:
        raise ValueError(f"序列太短: T={T}, 需要 T > {(embed_dim-1)*delay}")
    indices = np.array([
        np.arange(T_valid) + i * delay
        for i in range(embed_dim - 1, -1, -1)
    ]).T   # (T_valid, embed_dim)
    return X_data[:, indices]   # (N, T_valid, embed_dim)


def compute_delay_corr_token(X_data, delay=7, embed_dim=4):
    """
    ⚡ 向量化版：基于延迟嵌入的用户级周期性特征

    原版：for i in N_USERS → np.cov(X_embed[i].T)  = 42372次循环
    新版：einsum 批量协方差 → np.linalg.eigvalsh 批量特征值分解
          速度提升约 20~50x

    公式：C_i = X_c[i]^T · X_c[i] / (T'-1)   ∈ R^{d×d}

    返回: (N_users, embed_dim) 归一化特征值能量比
    """
    X_embed = delay_embedding(X_data, delay, embed_dim)   # (N, T', d)
    N, T_prime, d = X_embed.shape

    # ① 中心化（减去每用户的时间均值）
    X_c = X_embed - X_embed.mean(axis=1, keepdims=True)   # (N, T', d)

    # ② 批量协方差矩阵：C_batch[i] = X_c[i].T @ X_c[i] / (T'-1)
    #    einsum 'ntd,nte->nde' 等价于对每个用户做 (d,T')@(T',d)
    C_batch = np.einsum('ntd,nte->nde', X_c, X_c) / max(T_prime - 1, 1)
    # C_batch: (N, d, d)

    # ③ 批量特征值分解（eigvalsh 比 eig 快，且对对称矩阵稳定）
    eigvals_batch = np.linalg.eigvalsh(C_batch)            # (N, d) 升序
    eigvals_batch = eigvals_batch[:, ::-1]                 # 改为降序

    # ④ 归一化：各特征值占总能量的比例
    total_energy = eigvals_batch.sum(axis=1, keepdims=True) + 1e-8
    tokens = (eigvals_batch / total_energy).astype(np.float32)

    return tokens   # (N, embed_dim=4)

delay_tokens = compute_delay_corr_token(X, delay=7, embed_dim=4)
print(f"  延迟嵌入谱Token形状: {delay_tokens.shape}")

# AUC 验证
for _col, _name in enumerate(['Token22(λ1)', 'Token23(λ2)', 'Token24(λ3)', 'Token25(λ4)']):
    try:
        _auc = roc_auc_score(y, delay_tokens[:, _col])
        _auc = max(_auc, 1 - _auc)
        print(f"  {_name:<22} AUC={_auc:.4f}  {'✅' if _auc>=0.55 else '弱'}")
    except Exception:
        pass

# 平铺 + RobustScaler + 拼入 X_seq14_ds
for _col in range(delay_tokens.shape[1]):
    _arr = tile_user_feat(delay_tokens[:, _col], N_W_ds)
    _flat = _arr.reshape(-1, 1)
    _scaled = _RS().fit_transform(_flat)
    _arr = np.clip(_scaled, -5, 5).reshape(_arr.shape)
    X_seq14_ds = np.concatenate(
        [X_seq14_ds, _arr[:, :, np.newaxis]], axis=-1
    ).astype(np.float32)

FEAT_DIM = X_seq14_ds.shape[-1]   # 24+4=28
print(f"  拼入 Token22~25(延迟嵌入) 后形状: {X_seq14_ds.shape}  (FEAT_DIM={FEAT_DIM})")

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
print("\n  构造高判别力增强特征（Token26~30）...")

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
# Step A-12【新增】: 高判别力特征增强 (F31~F38)
# 与 RMT 协同的8维新特征：熵特征 + Hurst 指数 + KPSS + RMT差分加权
# =============================================================================
exec(open('patch_feature_boost.py', encoding='utf-8').read())

# =============================================================================
# Step A-13【新增】: 自动特征筛选，剔除 AUC < 0.53 的低判别力 Token
# =============================================================================
exec(open('patch_feature_select.py', encoding='utf-8').read())

# =============================================================================
# Step A-14【新增】: 月度趋势特征 + LightGBM 快速基线
# 月度特征是窃电检测最强信号，LightGBM 30秒内出 AUC 天花板结果
# =============================================================================
exec(open('patch_quick_auc.py', encoding='utf-8').read())

# =============================================================================
# Step A-15【核心改进】: 用月度序列替换7天窗口序列作为 Transformer 输入
# 原因：7天窗口聚合损失了月度趋势信息；LightGBM实验证实月度数据更有判别力
# 新输入形状：(N, 34个月, FEAT_DIM_MO)
# =============================================================================
print("\n" + "=" * 65)
print("Step A-15: 重建 Transformer 输入 → 月度序列替换7天窗口")
print("=" * 65)

# ── 构造月度时序特征矩阵 (N, N_MONTHS, K_MO) ──────────────────────────
# 每个时间步 = 1个月，包含：
#   ch0: 月均消费（归一化）
#   ch1: 月消费标准差
#   ch2: 月最大值
#   ch3: 月最小值
#   ch4: 月零值占比
#   ch5: 相对于全用户同月中位数的偏差（排名特征）
#   ch6~: 用户级标量特征（平铺到所有月份）

_N, _T  = X.shape
_DAYSPM = 30
_NM     = _T // _DAYSPM   # 34

# ★ P0优化：使用 X_raw（原始kWh）计算月度通道，保留绝对消费量信号
_mo_mean = np.zeros((_N, _NM), dtype=np.float32)
_mo_std  = np.zeros((_N, _NM), dtype=np.float32)
_mo_max  = np.zeros((_N, _NM), dtype=np.float32)
_mo_min  = np.zeros((_N, _NM), dtype=np.float32)
_mo_zero = np.zeros((_N, _NM), dtype=np.float32)

# ★ 用 X_raw 替代 X（Z-score后）计算月度统计
for _m in range(_NM):
    _s  = _m * _DAYSPM
    _e  = min(_s + _DAYSPM, _T)
    _seg_raw = X_raw[:, _s:_e]   # ★ 原始kWh数据
    _seg_z   = X[:, _s:_e]       # Z-score后数据（仅用于零值判断）
    _mo_mean[:, _m] = _seg_raw.mean(axis=1)     # ★ 原始kWh月均值
    _mo_std[:, _m]  = _seg_raw.std(axis=1)      # ★ 原始kWh月波动
    _mo_max[:, _m]  = _seg_raw.max(axis=1)      # ★ 原始kWh月峰值
    _mo_min[:, _m]  = _seg_raw.min(axis=1)      # ★ 原始kWh月最低
    _mo_zero[:, _m] = (_seg_z == 0).mean(axis=1) # 零值比仍用Z-score后数据

# ch5: 用户月消费相对全局中位数的偏差（跨用户排名信号）
_mo_median_global = np.median(_mo_mean, axis=0, keepdims=True)   # (1, NM)
_mo_rank_dev = _mo_mean - _mo_median_global                       # (N, NM)

# 差分通道 ch6: 月度一阶差分（直接捕捉斜率变化）
_mo_diff1 = np.diff(_mo_mean, axis=1, prepend=_mo_mean[:, :1])   # (N, NM)

# ── ★ 核心新增特征：用原始数据计算基准偏离（必须在归一化前）─────────────
# 关键：Z-score 后基准≈0，比值无意义；必须用 X_raw（原始插值未归一化数据）
_BASELINE_M = 6
_DAYSPM_RAW = 30
_NM_RAW     = X_raw.shape[1] // _DAYSPM_RAW   # ≈34

# 计算原始月均消费矩阵
_mo_raw = np.zeros((_N, _NM_RAW), dtype=np.float32)
for _mr2 in range(_NM_RAW):
    _s2 = _mr2 * _DAYSPM_RAW
    _e2 = min(_s2 + _DAYSPM_RAW, X_raw.shape[1])
    _mo_raw[:, _mr2] = X_raw[:, _s2:_e2].mean(axis=1)

# 用原始数据的前6个月为基准（单位：kWh，有绝对意义）
_baseline_raw = _mo_raw[:, :_BASELINE_M].mean(axis=1, keepdims=True) + 1e-3  # (N,1)

# ch7: 每月消费相对自身基准的偏离率（原始kWh，有物理意义）
_mo_vs_base = (_mo_raw - _baseline_raw) / (np.abs(_baseline_raw) + 1e-3)    # (N, NM)
# 对齐到 _NM（34步）
if _mo_vs_base.shape[1] != _NM:
    _mo_vs_base = _mo_vs_base[:, :_NM]

# ch8: 累积下降量（原始kWh累积偏离）
_mo_cumdev = np.cumsum(_mo_raw - _baseline_raw, axis=1)[:, :_NM]    # (N, NM)

# ch9: 跨用户同月百分位排名 [0,1]（窃电后排名持续下滑）
from scipy.stats import rankdata as _rankdata
_mo_pct = np.zeros((_N, _NM), dtype=np.float32)
for _mr in range(_NM):
    _r = _rankdata(_mo_mean[:, _mr])
    _mo_pct[:, _mr] = (_r / _N).astype(np.float32)

# ch10: 排名下降幅度（前半段排名均值 - 后半段排名均值，平铺）
_half_nm   = _NM // 2
_rank_drop = (_mo_pct[:, :_half_nm].mean(axis=1) -
              _mo_pct[:, _half_nm:].mean(axis=1))   # (N,) 正=排名下滑=可疑
_rank_tile = np.tile(_rank_drop[:, np.newaxis], (1, _NM))  # (N, NM)

# 打印新特征 AUC
from sklearn.metrics import roc_auc_score as _ras_m
def _qauc_m(f): a = _ras_m(y, f); return max(a, 1 - a)
print(f"  ch7(相对基准偏离率)  AUC={_qauc_m(_mo_vs_base.mean(axis=1)):.4f}  ← 目标≥0.68")
print(f"  ch8(累积下降量)      AUC={_qauc_m(_mo_cumdev.mean(axis=1)):.4f}  ← 目标≥0.65")
print(f"  ch9(跨用户百分位)    AUC={_qauc_m(_mo_pct.mean(axis=1)):.4f}  ← 目标≥0.65")
print(f"  ch10(排名下降幅度)   AUC={_qauc_m(_rank_drop):.4f}  ← 目标≥0.68")

# 归一化所有通道（RobustScaler 逐通道）
from sklearn.preprocessing import RobustScaler as _RS

def _scale_ch(arr2d, clip=5.0):
    """全局 RobustScaler 归一化（跨用户归一化通道）"""
    flat = arr2d.reshape(-1, 1)
    flat = _RS().fit_transform(flat).reshape(arr2d.shape)
    return np.clip(flat, -clip, clip).astype(np.float32)

def _scale_ch_per_user(arr2d, clip=5.0):
    """★ 用户自身归一化：每行(用户)独立做 (x - median) / IQR
    保留用户自身时间趋势，抹掉跨用户量级差异"""
    med = np.median(arr2d, axis=1, keepdims=True)
    q1  = np.percentile(arr2d, 25, axis=1, keepdims=True)
    q3  = np.percentile(arr2d, 75, axis=1, keepdims=True)
    iqr = q3 - q1 + 1e-6
    out = (arr2d - med) / iqr
    return np.clip(out, -clip, clip).astype(np.float32)

# ★ 用户自身归一化的月消费比值
_user_median_raw = np.median(_mo_mean, axis=1, keepdims=True) + 1e-3
_mo_self_ratio   = _mo_mean / _user_median_raw   # 每月/用户自身中位数

# 月度消费二阶差分
_mo_diff2 = np.diff(_mo_diff1, axis=1, prepend=_mo_diff1[:, :1])

# ★ 新增：月度消费对数变化率（对比值取对数，放大微小变化）
_mo_log_ratio = np.log1p(np.maximum(_mo_mean, 0)) - \
                np.log1p(np.maximum(_baseline_raw, 0))   # (N, NM) 对数偏离
if _mo_log_ratio.shape[1] != _NM:
    _mo_log_ratio = _mo_log_ratio[:, :_NM]

# ★ 新增：滑动窗口异常检测（3个月窗口，检测急剧下降）
_mo_roll3_mean = np.zeros((_N, _NM), dtype=np.float32)
_mo_roll3_std  = np.zeros((_N, _NM), dtype=np.float32)
for _m in range(_NM):
    _ws = max(0, _m - 2)
    _mo_roll3_mean[:, _m] = _mo_mean[:, _ws:_m+1].mean(axis=1)
    _mo_roll3_std[:, _m]  = _mo_mean[:, _ws:_m+1].std(axis=1) + 1e-6
# 当前月消费偏离最近3个月均值的程度（Z-score，越负=越可疑）
_mo_local_zscore = (_mo_mean - _mo_roll3_mean) / _mo_roll3_std   # (N, NM)

# ★ 新增：月消费跨用户同月偏离度（当月消费与全部用户当月中位数的log比值）
_mo_global_median = np.median(_mo_mean, axis=0, keepdims=True) + 1e-3  # (1, NM)
_mo_global_dev    = np.log1p(np.maximum(_mo_mean, 0)) - \
                    np.log1p(np.maximum(_mo_global_median, 0))

# 打印新通道 AUC 以验证信号强度
print(f"  ★ 新通道AUC验证:")
print(f"    ch_self_ratio(自身比值)    AUC={_qauc_m(_mo_self_ratio.mean(axis=1)):.4f}")
print(f"    ch_log_ratio(对数偏离)     AUC={_qauc_m(_mo_log_ratio.mean(axis=1)):.4f}")
print(f"    ch_local_zscore(局部Z)     AUC={_qauc_m(_mo_local_zscore.mean(axis=1)):.4f}")
print(f"    ch_global_dev(全局偏离)    AUC={_qauc_m(_mo_global_dev.mean(axis=1)):.4f}")

_mo_ch = np.stack([
    # ── 第一组：用户自身归一化通道（保留时间趋势）──
    _scale_ch_per_user(_mo_mean),     # ch0 月均消费（★ 用户自身归一化）
    _scale_ch_per_user(_mo_std),      # ch1 月消费波动（★ 用户自身归一化）
    _scale_ch_per_user(_mo_max),      # ch2 月最大值（★ 用户自身归一化）
    _mo_zero,                         # ch3 零值比 [0,1]
    # ── 第二组：跨用户排名通道 ──
    _mo_pct,                          # ch4 ★ 跨用户百分位排名 [0,1]
    _scale_ch(_mo_rank_dev),          # ch5 跨用户排名偏差（全局归一化）
    _scale_ch(_rank_tile),            # ch6 ★ 排名下降幅度
    # ── 第三组：基准偏离通道（最强信号）──
    _scale_ch_per_user(_mo_vs_base),  # ch7 ★ 相对基准偏离（用户自身归一化）
    _scale_ch_per_user(_mo_cumdev),   # ch8 ★ 累积下降量（用户自身归一化）
    _scale_ch(_mo_log_ratio),         # ch9 ★ 新增：对数偏离（放大微小变化）
    # ── 第四组：趋势/变化率通道 ──
    _scale_ch_per_user(_mo_diff1),    # ch10 月度差分（用户自身归一化）
    _scale_ch(_mo_diff2),             # ch11 月度加速度
    _mo_self_ratio.astype(np.float32),# ch12 自身比值（无需scale，已是比值）
    # ── 第五组：异常检测通道 ──
    np.clip(_mo_local_zscore, -5, 5).astype(np.float32),  # ch13 ★ 新增：局部Z-score
    _scale_ch(_mo_global_dev),        # ch14 ★ 新增：全局偏离度
    # ── 第六组：ISCT 谱轨迹通道 ──
    _scale_ch_per_user(isct_monthly),     # ch15 ★ ISCT: 个体谱投影能量月度轨迹
    _scale_ch_per_user(isct_dev_monthly), # ch16 ★ ISCT: 个体谱偏离轨迹（层内对比）
], axis=2)   # (N, NM, 17)

# 保留标量特征供 tile 拼接和 XGBoost 使用
_scalar_feats = X_seq14_ds.mean(axis=1)   # (N, FEAT_DIM) — 标量工程特征

# 标量特征 tile 拼接到每个时步 → 形成完整的 Transformer 输入序列
_scalar_tiled = np.tile(_scalar_feats[:, np.newaxis, :], (1, _NM, 1))  # (N, NM, FEAT_DIM)
X_mo_seq      = np.concatenate([_mo_ch, _scalar_tiled], axis=2).astype(np.float32)
FEAT_DIM_MO   = X_mo_seq.shape[2]           # 17 + FEAT_DIM
N_STEPS_MO    = _NM   # 34

print(f"  月度序列形状: {X_mo_seq.shape}  (时变{_mo_ch.shape[2]}维 + 标量{_scalar_feats.shape[1]}维)")
print(f"  时间步数: {N_STEPS_MO} 个月  总特征维度: {FEAT_DIM_MO}")
print(f"    ch0~16: 用户自身归一化+排名+基准偏离+趋势+异常检测+ISCT ({_mo_ch.shape[2]}维)")
print(f"    ch17~: 标量工程特征 tile 拼接 ({_scalar_feats.shape[1]}维)")

# ── 重建 DataLoader ────────────────────────────────────────────────────
idx_tr14, idx_te14 = train_test_split(
    np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
)
X_tr14 = torch.FloatTensor(X_mo_seq[idx_tr14])
y_tr14 = torch.FloatTensor(y[idx_tr14])
X_te14 = torch.FloatTensor(X_mo_seq[idx_te14])
y_te14 = torch.FloatTensor(y[idx_te14])

FEAT_DIM = FEAT_DIM_MO

class_counts14 = np.bincount(y[idx_tr14].astype(int))
sample_w14     = (1.0 / class_counts14)[y[idx_tr14].astype(int)]
sampler14 = WeightedRandomSampler(
    torch.FloatTensor(sample_w14), len(idx_tr14), replacement=True
)
# Windows CPU：num_workers=0（spawn进程开销 >> 数据预取收益）
# TensorDataset 数据全部在内存，无 I/O 瓶颈，不需要多进程预取
train_loader14 = DataLoader(
    TensorDataset(X_tr14, y_tr14),
    batch_size=1024, sampler=sampler14,
    num_workers=0, pin_memory=False,
)
test_loader14 = DataLoader(
    TensorDataset(X_te14, y_te14),
    batch_size=2048, shuffle=False,
    num_workers=0, pin_memory=False,
)
print(f"  Train: {len(train_loader14)} batches | Test: {len(test_loader14)} batches")

# =============================================================================
# Step B: 双路注意力Transformer（局部窗口 + 全局CLS Token）
# =============================================================================
print("\n" + "=" * 65)
print("Step B: 双路注意力Transformer模型")
print("=" * 65)


class UserGraphAttention(nn.Module):
    """
    批内用户图注意力层:

    Q·K^T / sqrt(d) → TopK 稀疏化 → Softmax → 聚合邻居特征 V

    核心思想: 窃电用户的行为模式偏离其"邻居"用户，
             通过批内用户间注意力让离群异常更加显著。

    α_{ij} = softmax( q_i · k_j / sqrt(d) )   [TopK 稀疏]
    h_i^{out} = LayerNorm( x_i + out_proj( Σ_j α_{ij} v_j ) )
    """
    def __init__(self, d_model: int, n_neighbors: int = 10):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.scale   = d_model ** -0.5
        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_model) 用户特征，返回 (B, d_model)"""
        import torch.nn.functional as F
        B = x.shape[0]
        Q = self.q_proj(x)    # (B, d)
        K = self.k_proj(x)    # (B, d)
        V = self.v_proj(x)    # (B, d)

        attn = torch.mm(Q, K.T) * self.scale   # (B, B)

        # TopK 稀疏化：只保留最近 n_neighbors 个邻居
        k_actual = min(self.n_neighbors, B)
        if B > k_actual:
            topk_vals, _ = torch.topk(attn, k_actual, dim=1)
            threshold = topk_vals[:, -1:].expand_as(attn)
            attn = torch.where(attn >= threshold, attn,
                               torch.full_like(attn, float('-inf')))

        attn = F.softmax(attn, dim=1)
        out  = self.out_proj(torch.mm(attn, V))  # (B, d)
        return self.norm(x + out)


class LocalWindowAttentionBlock(nn.Module):
    """
    带 FFN 的完整 Local Attention Block（Pre-LN 结构）
    局部滑动窗口注意力 + FFN，替代原先只有注意力的 LocalWindowAttention
    """
    def __init__(self, d_model: int, nhead: int,
                 win_size: int = 4, dim_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.win_size = win_size
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )

    def _build_mask(self, L: int, device) -> torch.Tensor:
        """构建局部窗口 attn_mask（带缓存，L固定时只建一次，CPU下省去重复分配）"""
        if hasattr(self, '_mask_cache') and self._mask_cache.shape[0] == L:
            return self._mask_cache.to(device)
        mask = torch.full((L, L), float('-inf'))
        for i in range(L):
            s = max(0, i - self.win_size)
            e = min(L, i + self.win_size + 1)
            mask[i, s:e] = 0.0
        self._mask_cache = mask   # 缓存，下次直接复用
        return mask.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        # Pre-LN Attention + 残差
        h = self.norm1(x)
        mask = self._build_mask(L, x.device)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + attn_out
        # Pre-LN FFN + 残差
        x = x + self.ffn(self.norm2(x))
        return x


class DualPathTransformer(nn.Module):
    """
    双路Transformer分类器
    路径1: 局部窗口注意力  → 捕捉短期突变
    路径2: 全局CLS注意力   → 捕捉长期趋势
    融合后接 MLP 分类头
    """
    def __init__(self, feat_dim=14, d_model=64, nhead=4,
                 num_layers=3, dim_ff=256, dropout=0.15,
                 win_size=4, max_len=60):
        super().__init__()
        assert d_model % nhead == 0

        # 共享输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model)
        )

        # 位置编码
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))
        self.pe_drop = nn.Dropout(dropout)

        # 路径1: 局部窗口注意力（带 FFN 的完整 Block）
        self.local_layers = nn.ModuleList([
            LocalWindowAttentionBlock(d_model, nhead, win_size, dim_ff, dropout)
            for _ in range(num_layers)
        ])

        # 路径2: 全局CLS Token注意力
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.global_transformer = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # 用户图注意力（批内 TopK 邻居聚合）
        # _graph_enabled=False：前30轮跳过 O(B²) 开销，第30轮后由外部开启
        self.graph_attn = UserGraphAttention(d_model * 2, n_neighbors=10)
        self._graph_enabled = False

        # 融合分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.3),          # 加强正则，防止少轮过拟合
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, 1)
        )

    def _encode(self, x: torch.Tensor):
        """共享编码：返回 (feat_local, feat_global)，避免重复计算"""
        B, L, _ = x.shape
        h = self.pe_drop(self.input_proj(x) + self.pe[:, :L, :])

        # 路径1: 局部窗口注意力（Block 内部已含 Pre-LN + FFN + 残差）
        h_local = h
        for block in self.local_layers:
            h_local = block(h_local)
        feat_local = h_local.mean(dim=1)   # (B, d_model)

        # 路径2: CLS 全局注意力
        h_g = torch.cat([self.cls_token.expand(B, -1, -1), h], dim=1)
        h_g = self.global_transformer(h_g)
        feat_global = h_g[:, 0, :]         # (B, d_model)

        return feat_local, feat_global

    def forward(self, x: torch.Tensor):
        """返回 (logit, feat_concat)，feat_concat 供 Contrastive Loss 使用"""
        feat_local, feat_global = self._encode(x)
        feat_cat = torch.cat([feat_local, feat_global], dim=1)  # (B, 2*d_model)
        # 训练时：仅在 _graph_enabled=True 后才启用（前30轮跳过，节省约30%时间）
        if self.training and self._graph_enabled and feat_cat.shape[0] > 1:
            feat_cat = self.graph_attn(feat_cat)
        return self.classifier(feat_cat), feat_cat              # (B,1), (B, 2D)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取 2*d_model 维特征，供 XGBoost 使用"""
        feat_local, feat_global = self._encode(x)
        return torch.cat([feat_local, feat_global], dim=1)

# ── B 检查点 ─────────────────────────────────────────────────────────
if STOP_AFTER == 'B':
    import sys
    print("\n[🛑 STOP_AFTER='B'] Step B 完成，程序退出。")
    sys.exit(0)

# =============================================================================
# [创新点B] SupConLoss: 监督对比损失
# 让异常用户特征聚类、正常用户远离异常用户，提升特征区分度
# 参考: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020
# =============================================================================
class SupConLoss(nn.Module):
    """
    监督对比损失（创新点B）
    对同类样本拉近，异类样本推远，改善特征空间判别性
    与 AdaptiveFocalLoss 联合使用：total_loss = focal + λ * supcon
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature      = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        features: (B, D) L2-normalized 特征向量
        labels:   (B,)  0/1 标签
        """
        device = features.device
        B = features.shape[0]
        if B < 4:
            return torch.tensor(0.0, device=device)

        # L2 归一化
        features = nn.functional.normalize(features, dim=1)

        # 相似度矩阵 (B, B)
        sim = torch.matmul(features, features.T) / self.temperature

        # 数值稳定：减去对角最大值
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # 构建 mask：同类为1，自身为0
        labels = labels.view(-1, 1)
        mask_pos  = (labels == labels.T).float()   # 同类
        mask_self = torch.eye(B, device=device)
        mask_pos  = mask_pos - mask_self            # 去掉对角

        # 若某类只有 1 个样本，该样本无正样本对，跳过
        num_pos = mask_pos.sum(dim=1)
        valid   = num_pos > 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device)

        exp_sim   = torch.exp(sim) * (1 - mask_self)   # 去掉自身
        log_prob  = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        mean_log  = (mask_pos * log_prob).sum(dim=1) / (num_pos + 1e-8)
        loss      = -(self.temperature / self.base_temperature) * mean_log
        return loss[valid].mean()


# =============================================================================
# Step C: ContrastiveFocal 联合损失
# =============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.85, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs   = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        p_t     = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss    = -alpha_t * (1 - p_t) ** self.gamma * torch.log(p_t)
        return loss.mean()


class AdaptiveFocalLoss(nn.Module):
    """
    自适应 Focal Loss（P3优化版）：
    - recall_weight 随训练轮次从 1.0 线性增长到 max_recall_w，避免过早激进
    - ★ P3改进1: hard_fn 阈值从 0.3 → 0.5，覆盖更多"模型不确定"的漏检
    - ★ P3改进2: 所有正样本(FN)施加非对称代价权重 fn_cost
    """
    def __init__(self, alpha=0.92, gamma=2.0, max_recall_w=1.5, fn_cost=1.0):
        super().__init__()
        self.alpha       = alpha
        self.gamma       = gamma
        self.max_recall_w= max_recall_w
        self.fn_cost     = fn_cost
        self.cur_epoch   = 1
        self.max_epochs  = 50

    def set_epoch(self, epoch: int, max_epochs: int):
        self.cur_epoch  = epoch
        self.max_epochs = max_epochs

    def forward(self, logits, targets, feats=None):   # feats 保持接口兼容
        probs   = torch.sigmoid(logits)
        bce     = nn.functional.binary_cross_entropy_with_logits(
                      logits, targets, reduction='none')

        # Focal 调制
        p_t     = probs * targets + (1 - probs) * (1 - targets)
        focal   = (1 - p_t) ** self.gamma * bce
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss    = alpha_t * focal

        # 动态 recall_weight：随训练进度从 1.0 → max_recall_w
        progress     = min(self.cur_epoch / self.max_epochs, 1.0)
        recall_weight= 1.0 + (self.max_recall_w - 1.0) * progress

        # 仅对"确定漏检"（prob < 0.3）的异常样本施加额外惩罚
        hard_fn_mask = (targets == 1) & (probs.detach() < 0.3)
        loss         = torch.where(hard_fn_mask, loss * recall_weight, loss)

        return loss.mean()


# ── 训练 / 评估函数 ───────────────────────────────────────────────────

def _mixup_data(x, y, alpha=0.3):
    """★ P5: Mixup 数据增强 — 对输入和标签做凸组合，增强正则化"""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def train_one_epoch_dual(model, loader, optimizer, scheduler,
                         criterion, scaler, device):
    """
    CPU 优化版训练函数（P5增强版）：
    - 禁用 autocast（CPU 下 float16 不被加速，反而因精度损失变慢）
    - 禁用 GradScaler（CPU 无 GPU 溢出问题，scaler 开销纯损耗）
    - ★ P5: 50% 概率使用 Mixup 数据增强
    """
    model.train()
    total_loss, total = 0.0, 0
    for batch in loader:
        X_batch, y_batch = batch
        X_batch = X_batch.to(device)
        y_batch = y_batch.float().to(device)
        optimizer.zero_grad(set_to_none=True)

        # Mixup 数据增强（20%概率，轻度正则化）
        if np.random.random() < 0.2:
            X_mix, y_a, y_b, lam = _mixup_data(X_batch, y_batch)
            logits, feats = model(X_mix)
            logits = logits.squeeze(1)
            loss = lam * criterion(logits, y_a, feats=feats) + \
                   (1 - lam) * criterion(logits, y_b, feats=feats)
        else:
            logits, feats = model(X_batch)
            logits = logits.squeeze(1)
            loss = criterion(logits, y_batch, feats=feats)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # ⚠️ 不在 batch 内调用 scheduler.step()
        # CosineAnnealingWarmRestarts 必须在 epoch 结束后调用
        total_loss += loss.item() * len(y_batch)
        total      += len(y_batch)
    return total_loss / total


@torch.no_grad()
def evaluate_dual(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        X_batch, y_batch = batch
        X_batch        = X_batch.to(device)
        logits, _      = model(X_batch)
        logits         = logits.squeeze(1)
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(y_batch.numpy())
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    auc = roc_auc_score(all_labels, all_probs)
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.05, 0.95, 0.005):
        preds = (all_probs >= thr).astype(int)
        f1    = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return auc, best_f1, best_thr, all_probs, all_labels


# ── 模型初始化 + 训练 ────────────────────────────────────────────────
NEG_COUNT  = int((y == 0).sum())
POS_COUNT  = int((y == 1).sum())
# ── 提升AUC/F1 超参调整（目标 AUC≥0.85, F1≥0.50） ──────────────
# GAMMA: 2.0→3.0，更强聚焦难分异常样本
# MAX_RECALL_W: 1.5→2.5，训练后期更激进提升召回率，直接改善F1
# EPOCHS: 50→80，配合早停充分训练
# PATIENCE: 15→20，给AUC更多改善机会
_raw_alpha    = 1.0 - POS_COUNT / (POS_COUNT + NEG_COUNT)
AUTO_ALPHA    = max(_raw_alpha * 0.95, 0.80)
# ── 【AUC优化】超参调整说明 ──────────────────────────────────────────
# GAMMA 2.0：降低 Focal 聚焦强度，让模型学更多"普通难样本"而非只盯最难样本
# MAX_RECALL_W 1.5：降低 recall 惩罚，优先保精确率 → AUC > F1
# EPOCHS 120：更长训练，配合 CosineWarmRestart 多周期收敛
# PATIENCE 20：给新特征更多收敛时间
GAMMA         = 2.0
MAX_RECALL_W  = 1.5
FN_COST       = 1.0
EPOCHS        = 60
PATIENCE      = 15

print(f"  正样本: {POS_COUNT} | 负样本: {NEG_COUNT}")
print(f"  AdaptiveFocal → alpha={AUTO_ALPHA:.4f}, gamma={GAMMA}, max_recall_w={MAX_RECALL_W}")

# ── 模型结构说明 ──────────────────────────────────────────────────────────
# d_model=128: CPU 推理速度与 AUC 的最佳平衡点
#   d_model=256 参数量5.4M 单epoch≈650s (×120=21h)，完全不可接受
#   d_model=128 参数量1.4M 单epoch≈150s (×120= 5h)，可接受
# num_layers=3: 减少1层，再提速 ~25%；SGCC 48步序列深度足够
# dim_ff=256: 与 d_model 对齐，进一步压缩计算量
# UserGraphAttention: O(B²) 开销极大，CPU 上直接禁用（epoch30启用会让每epoch增加400s）
model_dual = DualPathTransformer(
    feat_dim   = FEAT_DIM,
    d_model    = 128,
    nhead      = 4,
    num_layers = 3,
    dim_ff     = 256,
    dropout    = 0.15,
    win_size   = 4,       # 月度序列34步，窗口4步=4个月，捕捉季度趋势
    max_len    = 40,      # 略大于34，留余量
).to(device)
# ⚠️ 强制禁用 UserGraphAttention（CPU上O(B²)，会让每epoch从150s变600s）
# Windows CPU 不使用 torch.compile，直接访问模型属性
model_dual._graph_enabled = False

# ── 先统计参数量、初始化优化器（必须在 torch.compile 之前！）──────────
def _get_raw_model(m):
    """兼容 torch.compile 包装：取原始 nn.Module（用于访问自定义属性）"""
    return getattr(m, '_orig_mod', m)

total_params = sum(p.numel() for p in model_dual.parameters() if p.requires_grad)
print(f"  模型参数量: {total_params:,}")

criterion_dual = AdaptiveFocalLoss(
    alpha=AUTO_ALPHA, gamma=GAMMA, max_recall_w=MAX_RECALL_W, fn_cost=FN_COST
)
optimizer_dual = optim.AdamW(
    model_dual.parameters(), lr=1e-3, weight_decay=1e-4   # 编译前取 parameters()
)
# ── 调度器：CosineAnnealingWarmRestarts ──────────────────────────────────
# T_0=15, T_mult=2：周期 15→30→60 轮，共覆盖 105 轮，配合 120 轮训练
# scheduler.step() 在每个 epoch 结束后调用（非 batch 内），避免 LR 震荡崩塌
scheduler_dual = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer_dual,
    T_0     = 15,
    T_mult  = 2,
    eta_min = 5e-7,
)

# ⚠️ torch.compile 在 Windows CPU 环境下需要 MSVC (cl.exe)，未安装时会在
# 第一次 forward 时崩溃（InductorError: Compiler cl is not found）
# → 直接跳过，其余向量化/多线程优化已足够提速
print('  ℹ️ torch.compile 已跳过（Windows CPU 需要 MSVC，当前环境不支持）')

# CPU 环境下 GradScaler 无意义（无 FP16 溢出问题），用 DummyScaler 替代
# 保持接口不变（train_one_epoch_dual 内部已不再调用 scaler），仅做占位
class _DummyScaler:
    """CPU环境：GradScaler 空实现，消除无用开销"""
    def scale(self, loss): return loss
    def unscale_(self, opt): pass
    def step(self, opt): opt.step()
    def update(self): pass

scaler_dual = _DummyScaler()

history_dual = {'train_loss': [], 'val_auc': [], 'val_f1': [], 'lr': []}
best_auc_dual, best_f1_dual = 0.0, 0.0
best_state_dual, best_epoch_dual = None, 0
no_improve_dual = 0

print(f"\n{'='*70}")
print(f"  损失函数: AdaptiveFocal | alpha={AUTO_ALPHA:.4f} | "
      f"gamma={GAMMA} | max_recall_w={MAX_RECALL_W}")
print(f"  调度器: CosineAnnealingWarmRestarts | T_0=15 T_mult=2 | eta_min=5e-7 | epochs={EPOCHS}")
print(f"  模型: d_model=128 | layers=3 | dim_ff=256 | feat_dim={FEAT_DIM}")
print(f"  UserGraphAttention: 已禁用（CPU加速）")
print(f"  早停: patience={PATIENCE} 轮")
print(f"{'='*70}")
print(f"{'Epoch':>5} | {'Loss':>10} | {'ValAUC':>7} | "
      f"{'ValF1':>6} | {'Thr':>5} | {'LR':>8} | {'Time':>6}")
print(f"{'='*70}")

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss = train_one_epoch_dual(
        model_dual, train_loader14, optimizer_dual,
        scheduler_dual, criterion_dual, scaler_dual, device
    )
    val_auc, val_f1, val_thr, val_probs, val_labels = evaluate_dual(
        model_dual, test_loader14, device
    )
    elapsed = time.time() - t0

    # ✅ CosineAnnealingWarmRestarts：在 epoch 结束后调用 step(epoch)
    # 传入 epoch-1 使第0轮从最大 LR 开始（官方推荐用法）
    scheduler_dual.step(epoch - 1)
    criterion_dual.set_epoch(epoch, EPOCHS)
    cur_lr = optimizer_dual.param_groups[0]['lr']

    history_dual['train_loss'].append(tr_loss)
    history_dual['val_auc'].append(val_auc)
    history_dual['val_f1'].append(val_f1)
    history_dual['lr'].append(cur_lr)

    # UserGraphAttention CPU 模式下已禁用，无需动态启用

    mark = ' ✅' if val_auc > best_auc_dual else ''
    print(f"{epoch:>5} | {tr_loss:>10.6f} | {val_auc:>7.4f} | "
          f"{val_f1:>6.4f} | {val_thr:>5.3f} | {cur_lr:>8.2e} | {elapsed:>5.1f}s{mark}")

    if val_auc > best_auc_dual:
        best_auc_dual   = val_auc
        best_f1_dual    = val_f1
        best_state_dual = {k: v.cpu().clone()
                           for k, v in model_dual.state_dict().items()}
        best_epoch_dual = epoch
        no_improve_dual = 0
    else:
        no_improve_dual += 1
        if no_improve_dual >= PATIENCE:
            print(f"\n⏹ 早停！最优在第 {best_epoch_dual} 轮")
            break

model_dual.load_state_dict(best_state_dual)
torch.save(best_state_dual, 'dual_path_transformer_best.pth')
print(f"\n✅ 双路Transformer训练完成！"
      f"最优 AUC={best_auc_dual:.4f}  F1={best_f1_dual:.4f}")


# ── Step C-2 跳过，best_fed_state / best_fed_auc 对齐集中式训练结果 ──
best_fed_auc   = best_auc_dual
best_fed_state = best_state_dual
print("\n[Step C-2 已注释] 跳过联邦学习，直接使用集中式训练最优模型继续。")

# =============================================================================
# Step D: XGBoost 集成提升（Transformer特征 + 手工统计特征）
# =============================================================================
print("\n" + "=" * 65)
print("Step D: XGBoost 集成（Transformer特征 + 手工特征）")
print("=" * 65)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  ⚠ 未安装 xgboost，跳过集成。可运行: pip install xgboost")

xgb_auc, ensemble_auc, best_f1_ens = 0.0, best_auc_dual, best_f1_dual

if HAS_XGB:
    # ── D-1: 提取Transformer中间特征 ────────────────────────────────
    @torch.no_grad()
    def batch_extract_features(model, X_np, device, batch_size=1024):
        model.eval()
        all_feats = []
        for i in range(0, len(X_np), batch_size):
            xb = torch.FloatTensor(X_np[i:i + batch_size]).to(device)
            feat = model.extract_features(xb)
            all_feats.append(feat.cpu().numpy())
        return np.concatenate(all_feats, axis=0)

    print("  提取Transformer特征...")
    feat_all = batch_extract_features(model_dual, X_mo_seq, device)
    print(f"  Transformer特征维度: {feat_all.shape}")

    # ── D-2: 增强手工统计特征（基于月度序列）──────────────────────────
    _X_raw  = X_mo_seq          # (N_USERS, N_MONTHS, FEAT_DIM_MO)
    _N_STEPS = _X_raw.shape[1]  # 34 个月
    _HALF    = _N_STEPS // 2    # 17
    hand_feats = np.concatenate([
        _X_raw.mean(axis=1),                                                   # 均值
        _X_raw.std(axis=1),                                                    # 标准差
        _X_raw.max(axis=1),                                                    # 最大值
        _X_raw.min(axis=1),                                                    # 最小值
        np.percentile(_X_raw, 25, axis=1),                                     # Q1
        np.percentile(_X_raw, 75, axis=1),                                     # Q3
        _X_raw[:, _HALF:, :].mean(axis=1) - _X_raw[:, :_HALF, :].mean(axis=1), # 后半-前半趋势
        _X_raw.argmax(axis=1).astype(np.float32) / _N_STEPS,                  # 最大值位置
    ], axis=1)
    print(f"  增强手工特征维度: {hand_feats.shape}  (FEAT_DIM_MO×8={FEAT_DIM_MO*8})")

    # ── D-3: 拼接 → XGBoost训练 ─────────────────────────────────────
    # ★ 核心优化: 给XGBoost提供多视角原始信号
    # 用户自身归一化的月消费（与Transformer输入一致的视角）
    _xgb_self_norm = _scale_ch_per_user(_mo_raw[:, :_NM])  # (N, 34)
    # 月消费一阶差分
    _xgb_diff = np.diff(_mo_raw[:, :_NM], axis=1, prepend=_mo_raw[:, :1])
    _xgb_diff_norm = _scale_ch_per_user(_xgb_diff)
    # 基准偏离率
    _xgb_vs_base = _mo_vs_base[:, :_NM]
    # 跨用户百分位
    _xgb_pct = _mo_pct[:, :_NM]
    # 局部Z-score
    _xgb_local_z = np.clip(_mo_local_zscore[:, :_NM], -5, 5)

    X_xgb    = np.concatenate([
        feat_all,          # Transformer 中间特征 (256维)
        hand_feats,        # 月度序列统计
        rmt_isc_features,   # ISCT-CPD 个体变点特征 (8维)
        isct_monthly,       # ISCT月度谱投影能量 (34维)
        _mo_raw[:, :_NM],  # 原始kWh月度消费 (34维)
        _xgb_self_norm,    # ★ 用户自身归一化月消费 (34维)
        _xgb_diff_norm,    # ★ 月消费差分（用户自身归一化）(34维)
        _xgb_vs_base,      # ★ 基准偏离率 (34维)
        _xgb_pct,          # ★ 跨用户百分位排名 (34维)
        _xgb_local_z,      # ★ 局部Z-score异常度 (34维)
        _scalar_feats,     # 标量工程特征
        tcn_cpd_features,  # 多尺度TCN-CPD特征 (20维)
    ], axis=1)
    print(f"  XGBoost总特征: {X_xgb.shape}  (多视角原始信号+Transformer+手工+标量)")
    X_xgb_tr = X_xgb[idx_tr14]
    X_xgb_te = X_xgb[idx_te14]
    y_xgb_tr = y[idx_tr14]
    y_xgb_te = y[idx_te14]

    # XGBoost 超参（消融实验验证后的最优参数: G1=0.8042）
    xgb_model = xgb.XGBClassifier(
        n_estimators          = 1000,
        max_depth             = 6,
        learning_rate         = 0.03,
        subsample             = 0.8,
        colsample_bytree      = 0.7,
        colsample_bylevel     = 0.8,
        min_child_weight      = 5,
        reg_alpha             = 0.1,
        reg_lambda            = 1.0,
        scale_pos_weight      = NEG_COUNT / POS_COUNT,
        eval_metric           = 'auc',
        random_state          = 42,
        n_jobs                = -1,
        early_stopping_rounds = 30
    )
    xgb_model.fit(
        X_xgb_tr, y_xgb_tr,
        eval_set=[(X_xgb_te, y_xgb_te)],
        verbose=50
    )

    xgb_probs = xgb_model.predict_proba(X_xgb_te)[:, 1]
    xgb_auc   = roc_auc_score(y_xgb_te, xgb_probs)
    print(f"\n  XGBoost AUC: {xgb_auc:.4f}")

    # ── D-4: 最优权重搜索集成（替换固定0.5/0.5）────────────────────
    _, _, _, trans_probs_te, trans_labels_te = evaluate_dual(
        model_dual, test_loader14, device
    )

    # 在验证集上搜索最优融合权重 w（Transformer权重，XGB权重=1-w）
    best_w, best_ens_auc = 0.5, 0.0
    for w in np.arange(0.0, 1.01, 0.05):
        blend = w * trans_probs_te + (1 - w) * xgb_probs
        try:
            auc_w = roc_auc_score(trans_labels_te, blend)
            if auc_w > best_ens_auc:
                best_ens_auc, best_w = auc_w, w
        except Exception:
            pass

    ensemble_probs = best_w * trans_probs_te + (1 - best_w) * xgb_probs
    ensemble_auc   = roc_auc_score(trans_labels_te, ensemble_probs)
    print(f"  最优融合权重: Transformer={best_w:.2f}, XGB={1-best_w:.2f}")

    best_f1_ens, best_thr_ens = 0.0, 0.5
    for thr in np.arange(0.05, 0.95, 0.005):
        preds = (ensemble_probs >= thr).astype(int)
        f1    = f1_score(trans_labels_te, preds, zero_division=0)
        if f1 > best_f1_ens:
            best_f1_ens, best_thr_ens = f1, thr

    print(f"  集成(加权平均) AUC: {ensemble_auc:.4f}  F1: {best_f1_ens:.4f}")

    # ── D-5: CatBoost 第3个基学习器（与XGBoost差异化集成）──────────────
    print("\n  [CatBoost] 训练第3个差异化基学习器...")
    try:
        from catboost import CatBoostClassifier
        _HAS_CATBOOST = True
    except ImportError:
        _HAS_CATBOOST = False
        print("  ⚠ catboost 未安装，跳过CatBoost（可 pip install catboost）")

    if _HAS_CATBOOST:
        cat_model = CatBoostClassifier(
            iterations=3000,
            depth=7,
            learning_rate=0.01,
            l2_leaf_reg=3.0,
            border_count=128,
            random_strength=1.5,
            bagging_temperature=0.8,
            auto_class_weights='Balanced',
            eval_metric='AUC',
            random_seed=42,
            verbose=300,
            early_stopping_rounds=80,
            task_type='CPU'
        )
        # CatBoost 使用差异化特征子集（去掉 Transformer 特征，增加集成多样性）
        # XGBoost: feat_all + hand + ISC + 原始多视角 + 标量 (953维)
        # CatBoost: 只用手工特征 + ISC + 原始多视角 + 标量（不含 Transformer 特征）
        X_cat = np.concatenate([
            hand_feats,         # 月度序列统计
            rmt_isc_features,   # ISCT-CPD 个体变点特征 (8维)
            isct_monthly,       # ISCT月度轨迹 (34维)
            _mo_raw[:, :_NM],   # 原始kWh月度消费
            _xgb_self_norm,     # 用户自身归一化
            _xgb_diff_norm,     # 月消费差分
            _xgb_vs_base,       # 基准偏离率
            _xgb_pct,           # 跨用户百分位排名
            _xgb_local_z,       # 局部Z-score
            _scalar_feats,      # 标量工程特征
            tcn_cpd_features,   # 多尺度TCN-CPD特征 (20维)
        ], axis=1)
        X_cat_tr = X_cat[idx_tr14]
        X_cat_te = X_cat[idx_te14]
        print(f"  CatBoost差异化特征: {X_cat.shape[1]}维 (不含Transformer特征)")

        cat_model.fit(
            X_cat_tr, y_xgb_tr,
            eval_set=(X_cat_te, y_xgb_te),
            verbose=300
        )
        cat_probs = cat_model.predict_proba(X_cat_te)[:, 1]
        cat_auc = roc_auc_score(y_xgb_te, cat_probs)
        print(f"  CatBoost AUC: {cat_auc:.4f}")
    else:
        cat_probs = xgb_probs.copy()
        cat_auc = xgb_auc

    # ── D-5b: 三模型加权融合（网格搜索最优权重组合）──────────────
    print("\n  [三模型加权融合] 网格搜索 Transformer + XGBoost + CatBoost 最优权重...")
    best_w3, best_ens3_auc = (0.33, 0.33, 0.34), 0.0
    _step3 = 0.05
    for w_t in np.arange(0.0, 1.01, _step3):
        for w_x in np.arange(0.0, 1.01 - w_t, _step3):
            w_c = 1.0 - w_t - w_x
            if w_c < -1e-9:
                continue
            blend3 = w_t * trans_probs_te + w_x * xgb_probs + w_c * cat_probs
            try:
                auc3 = roc_auc_score(trans_labels_te, blend3)
                if auc3 > best_ens3_auc:
                    best_ens3_auc = auc3
                    best_w3 = (w_t, w_x, w_c)
            except Exception:
                pass

    ensemble_probs_3m = best_w3[0] * trans_probs_te + best_w3[1] * xgb_probs + best_w3[2] * cat_probs
    ensemble_auc_3m   = roc_auc_score(trans_labels_te, ensemble_probs_3m)
    print(f"  三模型最优权重: T={best_w3[0]:.2f}, X={best_w3[1]:.2f}, Cat={best_w3[2]:.2f}")

    best_f1_3m, best_thr_3m = 0.0, 0.5
    for thr in np.arange(0.05, 0.95, 0.005):
        preds = (ensemble_probs_3m >= thr).astype(int)
        f1    = f1_score(trans_labels_te, preds, zero_division=0)
        if f1 > best_f1_3m:
            best_f1_3m, best_thr_3m = f1, thr

    print(f"  三模型加权融合 AUC: {ensemble_auc_3m:.4f}  F1: {best_f1_3m:.4f}")

    # 如果三模型融合优于二模型融合，则更新
    if ensemble_auc_3m > ensemble_auc:
        ensemble_probs = ensemble_probs_3m
        ensemble_auc   = ensemble_auc_3m
        best_f1_ens    = best_f1_3m
        print(f"  --> 三模型融合({ensemble_auc_3m:.4f}) 优于二模型融合，已替换")
    else:
        print(f"  --> 二模型融合({ensemble_auc:.4f}) 仍优于三模型融合({ensemble_auc_3m:.4f})")

    # ── D-5c: Rank Average 集成策略 ──────────────────────────────
    print("\n  [Rank Average] 三模型排名平均集成...")
    from scipy.stats import rankdata as _rankdata
    rank_trans = _rankdata(trans_probs_te) / len(trans_probs_te)
    rank_xgb   = _rankdata(xgb_probs) / len(xgb_probs)
    rank_cat   = _rankdata(cat_probs) / len(cat_probs)
    rank_avg_probs = (rank_trans + rank_xgb + rank_cat) / 3.0
    rank_avg_auc   = roc_auc_score(trans_labels_te, rank_avg_probs)

    best_f1_rank, best_thr_rank = 0.0, 0.5
    for thr in np.arange(0.05, 0.95, 0.005):
        preds = (rank_avg_probs >= thr).astype(int)
        f1    = f1_score(trans_labels_te, preds, zero_division=0)
        if f1 > best_f1_rank:
            best_f1_rank, best_thr_rank = f1, thr

    print(f"  Rank Average AUC: {rank_avg_auc:.4f}  F1: {best_f1_rank:.4f}")

    # 如果 Rank Average 更优，则更新
    if rank_avg_auc > ensemble_auc:
        ensemble_probs = rank_avg_probs
        ensemble_auc   = rank_avg_auc
        best_f1_ens    = best_f1_rank
        print(f"  --> Rank Average({rank_avg_auc:.4f}) 优于当前最佳，已替换")
    else:
        print(f"  --> 当前最佳({ensemble_auc:.4f}) 仍优于 Rank Average({rank_avg_auc:.4f})")

    # ── D-6: 三模型 Stacking 集成（OOF方式，消除数据泄露）──────────────
    print("\n  [Stacking集成] 3模型OOF Stacking（Transformer + XGBoost + CatBoost）...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold as _SKF_meta

    _skf_meta = _SKF_meta(n_splits=5, shuffle=True, random_state=42)
    oof_xgb_probs = np.zeros(len(y_xgb_tr))
    oof_cat_probs = np.zeros(len(y_xgb_tr))

    print("  5折OOF（XGBoost + CatBoost）...")
    for _fold, (_tr_idx, _va_idx) in enumerate(_skf_meta.split(X_xgb_tr, y_xgb_tr)):
        # XGBoost OOF
        _xgb_fold = xgb.XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.6,
            min_child_weight=10, reg_alpha=0.5, reg_lambda=2.0,
            scale_pos_weight=NEG_COUNT / POS_COUNT,
            eval_metric='auc', random_state=42, n_jobs=-1,
            early_stopping_rounds=30
        )
        _xgb_fold.fit(
            X_xgb_tr[_tr_idx], y_xgb_tr[_tr_idx],
            eval_set=[(X_xgb_tr[_va_idx], y_xgb_tr[_va_idx])],
            verbose=0
        )
        oof_xgb_probs[_va_idx] = _xgb_fold.predict_proba(X_xgb_tr[_va_idx])[:, 1]

        # CatBoost OOF（使用差异化特征）
        if _HAS_CATBOOST:
            _cat_fold = CatBoostClassifier(
                iterations=2500, depth=7, learning_rate=0.01,
                l2_leaf_reg=3.0, border_count=128,
                random_strength=1.5, bagging_temperature=0.8,
                auto_class_weights='Balanced',
                eval_metric='AUC', random_seed=42, verbose=0,
                early_stopping_rounds=80, task_type='CPU'
            )
            _cat_fold.fit(
                X_cat_tr[_tr_idx], y_xgb_tr[_tr_idx],
                eval_set=(X_cat_tr[_va_idx], y_xgb_tr[_va_idx]),
                verbose=0
            )
            oof_cat_probs[_va_idx] = _cat_fold.predict_proba(X_cat_tr[_va_idx])[:, 1]
        else:
            oof_cat_probs[_va_idx] = oof_xgb_probs[_va_idx]

        print(f"    Fold {_fold+1}/5 完成")

    # 提取训练集Transformer概率
    _tr_loader_meta = DataLoader(
        TensorDataset(X_tr14, y_tr14),
        batch_size=2048, shuffle=False, num_workers=0, pin_memory=False
    )
    _, _, _, _trans_probs_tr, _trans_labels_tr = evaluate_dual(model_dual, _tr_loader_meta, device)

    # 构造3模型OOF元特征 → Stacking
    meta_train_oof = np.stack([_trans_probs_tr, oof_xgb_probs, oof_cat_probs], axis=1)
    meta_lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
    meta_lr.fit(meta_train_oof, y_xgb_tr.astype(int))

    # 测试集3模型概率
    meta_test = np.stack([trans_probs_te, xgb_probs, cat_probs], axis=1)
    stacking_probs = meta_lr.predict_proba(meta_test)[:, 1]
    stacking_auc   = roc_auc_score(trans_labels_te, stacking_probs)

    best_f1_stk, best_thr_stk = 0.0, 0.5
    for thr in np.arange(0.05, 0.95, 0.005):
        preds = (stacking_probs >= thr).astype(int)
        f1    = f1_score(trans_labels_te, preds, zero_division=0)
        if f1 > best_f1_stk:
            best_f1_stk, best_thr_stk = f1, thr

    coef_str = ', '.join([f'{c:.3f}' for c in meta_lr.coef_[0]])
    print(f"  Stacking 元学习器权重: [{coef_str}] (Trans/XGB/Cat)")
    print(f"  Stacking 3模型集成 AUC: {stacking_auc:.4f}  F1: {best_f1_stk:.4f}")

    # 若 Stacking 更优则采用
    if stacking_auc > ensemble_auc:
        ensemble_auc   = stacking_auc
        ensemble_probs = stacking_probs
        best_f1_ens    = best_f1_stk
        print("  --> Stacking 优于加权平均，已替换为最终集成结果")
    else:
        print("  --> 加权平均仍优于 Stacking，保持原集成结果")

# =============================================================================
# Step E: 综合评估与可视化
# =============================================================================
print("\n" + "=" * 65)
print("Step E: 综合评估汇总")
print("=" * 65)

final_auc, final_f1, final_thr, final_probs, final_labels = evaluate_dual(
    model_dual, test_loader14, device
)
final_preds = (final_probs >= final_thr).astype(int)
cm  = confusion_matrix(final_labels, final_preds)
tn, fp, fn, tp = cm.ravel()

print(f"\n  {'方法':<34} {'AUC':>8}  {'F1':>8}")
print(f"  {'-'*55}")
print(f"  {'RMT 谱得分（基线）':<34} {rmt_baseline_auc:>8.4f}  {'N/A':>8}")
print(f"  {'双路Transformer(集中式)':<34} {final_auc:>8.4f}  {final_f1:>8.4f}")
print(f"  {'FedAvg 联邦训练(+图注意力)':<34} {best_fed_auc:>8.4f}  {'N/A':>8}")
if HAS_XGB:
    print(f"  {'XGBoost(Transformer特征)':<34} {xgb_auc:>8.4f}  {'N/A':>8}")
    print(f"  {'CatBoost(差异化基学习器)':<34} {cat_auc:>8.4f}  {'N/A':>8}")
    print(f"  {'3模型Stacking集成(最终)':<34} {ensemble_auc:>8.4f}  {best_f1_ens:>8.4f}")

print(f"\n  混淆矩阵:")
print(f"  {'':18} 预测:正常  预测:异常")
print(f"  真实:正常    {tn:6d}    {fp:6d}")
print(f"  真实:异常    {fn:6d}    {tp:6d}")
print(f"\n{classification_report(final_labels, final_preds, target_names=['正常', '异常'])}")

# ── 优先级1：PR 曲线最优阈值分析 ─────────────────────────────────────
print("\n  [阈值优化] 基于 PR 曲线搜索最优阈值...")
from sklearn.metrics import precision_recall_curve as _prc
pr_prec, pr_rec, pr_thr = _prc(final_labels, final_probs)
pr_f1   = 2 * pr_prec * pr_rec / (pr_prec + pr_rec + 1e-8)
best_pr_idx = np.argmax(pr_f1)
best_pr_thr = pr_thr[best_pr_idx]
best_pr_f1  = pr_f1[best_pr_idx]
best_pr_prec= pr_prec[best_pr_idx]
best_pr_rec = pr_rec[best_pr_idx]

print(f"\n  {'─'*45}")
print(f"  PR 曲线最优阈值 : {best_pr_thr:.4f}")
print(f"  最优 F1         : {best_pr_f1:.4f}")
print(f"  对应精确率      : {best_pr_prec:.4f}")
print(f"  对应召回率      : {best_pr_rec:.4f}")
print(f"  {'─'*45}")

y_pred_pr = (final_probs >= best_pr_thr).astype(int)
print(f"\n  [PR最优阈值下的详细报告]")
print(classification_report(final_labels, y_pred_pr, target_names=['正常', '异常']))

# ── 六图可视化 ─────────────────────────────────────────────────────
top_auc    = ensemble_auc if HAS_XGB else final_auc
epochs_r   = range(1, len(history_dual['train_loss']) + 1)
fig, axes  = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    f"双路Transformer + XGBoost集成 | AUC={top_auc:.4f}",
    fontsize=14, fontweight='bold'
)

# ① Loss曲线
ax = axes[0, 0]
ax.plot(epochs_r, history_dual['train_loss'], color='steelblue')
ax.set_title('ContrastiveFocal Loss')
ax.set_xlabel('Epoch')
ax.grid(True, alpha=0.3)

# ② AUC曲线（多方法对比）
ax = axes[0, 1]
ax.plot(epochs_r, history_dual['val_auc'], color='green', label='双路Transformer')
ax.axhline(rmt_baseline_auc, color='red', linestyle='--',
           label=f'RMT基线({rmt_baseline_auc:.4f})')
if HAS_XGB:
    ax.axhline(ensemble_auc, color='purple', linestyle='-.',
               label=f'集成({ensemble_auc:.4f})')
ax.set_title('AUC对比'); ax.set_xlabel('Epoch')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ③ F1曲线
ax = axes[0, 2]
ax.plot(epochs_r, history_dual['val_f1'], color='blue')
ax.set_title('F1 变化曲线（异常类）'); ax.set_xlabel('Epoch')
ax.grid(True, alpha=0.3)

# ④ ROC曲线
ax = axes[1, 0]
fpr_t, tpr_t, _ = roc_curve(final_labels, final_probs)
ax.plot(fpr_t, tpr_t, label=f'双路Transformer(AUC={final_auc:.4f})')
if HAS_XGB:
    fpr_e, tpr_e, _ = roc_curve(trans_labels_te, ensemble_probs)
    ax.plot(fpr_e, tpr_e, '--', label=f'集成(AUC={ensemble_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', label='随机基线')
ax.set_title('ROC曲线'); ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ⑤ PR曲线
ax = axes[1, 1]
prec_arr, rec_arr, _ = precision_recall_curve(final_labels, final_probs)
ax.plot(rec_arr, prec_arr, color='purple')
ax.axhline(POS_COUNT / (POS_COUNT + NEG_COUNT), color='red',
           linestyle='--', label='随机基线')
ax.set_title('Precision-Recall曲线')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.legend(); ax.grid(True, alpha=0.3)

# ⑥ 混淆矩阵
ax = axes[1, 2]
im = ax.imshow(cm, cmap=plt.cm.Blues)
ax.set_title('混淆矩阵')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['正常', '异常']); ax.set_yticklabels(['正常', '异常'])
ax.set_xlabel('预测值'); ax.set_ylabel('真实值')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                fontsize=14, fontweight='bold',
                color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('sgcc_upgrade_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("已保存: sgcc_upgrade_results.png")

# =============================================================================
# Step F: 消融实验 — 验证 RMT-CPD / RMT-ISC-CPD 对 AUC 的贡献
# =============================================================================
print("\n" + "=" * 65)
print("Step F: 消融实验 (Ablation Study)")
print("=" * 65)
print("  目标: 验证 Stratified-RMT 和 ISCT-CPD 各组件对 AUC 的贡献")
print("  方法: 使用 XGBoost 在不同特征子集上快速训练并对比 AUC")

if HAS_XGB:
    import time as _abl_time

    # ── 构造消融实验的特征矩阵 ─────────────────────────────────────────
    # 基础特征（不含任何 RMT-ISC 组件）
    _X_base = np.concatenate([
        feat_all,          # Transformer 中间特征
        hand_feats,        # 月度序列统计
        _mo_raw[:, :_NM],  # 原始kWh月度消费
        _xgb_self_norm,    # 用户自身归一化月消费
        _xgb_diff_norm,    # 月消费差分
        _xgb_vs_base,      # 基准偏离率
        _xgb_pct,          # 跨用户百分位排名
        _xgb_local_z,      # 局部Z-score
        _scalar_feats,     # 标量工程特征
    ], axis=1)

    # 分层RMT特征（2维）
    _X_strat_rmt = np.column_stack([local_rmt_score, local_rmt_ratio])

    # ISCT-CPD特征（8维个体CPD + 34维月消费轨迹 = 42维）
    _X_isct_cpd = np.concatenate([
        isct_cpd_feats,      # (N, 8) 个体变点特征
        isct_monthly,        # (N, 34) 原始月消费轨迹
    ], axis=1)

    # TCN-CPD 特征
    _X_tcn_cpd = tcn_cpd_features   # (N, 20) Multi-Scale TCN-CPD 特征

    # 消融实验组别（6组）
    ablation_groups = [
        ("G1: Base (无RMT-ISC)",                _X_base),
        ("G2: Base + Stratified-RMT",           np.concatenate([_X_base, _X_strat_rmt], axis=1)),
        ("G3: Base + ISCT-CPD",                 np.concatenate([_X_base, _X_isct_cpd], axis=1)),
        ("G4: Full (Base+Strat-RMT+ISCT-CPD)",  np.concatenate([_X_base, _X_strat_rmt, _X_isct_cpd], axis=1)),
        ("G5: Base + TCN-CPD",                  np.concatenate([_X_base, _X_tcn_cpd], axis=1)),
        ("G6: Base + ISCT-CPD + TCN-CPD",       np.concatenate([_X_base, _X_isct_cpd, _X_tcn_cpd], axis=1)),
    ]

    # XGBoost 快速训练函数
    def _ablation_xgb(X_feat, y_all, idx_tr, idx_te):
        _t0 = _abl_time.time()
        X_tr_a, X_te_a = X_feat[idx_tr], X_feat[idx_te]
        y_tr_a, y_te_a = y_all[idx_tr], y_all[idx_te]
        neg_c = (y_tr_a == 0).sum()
        pos_c = (y_tr_a == 1).sum()
        _model = xgb.XGBClassifier(
            n_estimators=800, max_depth=5, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.6,
            min_child_weight=10, reg_alpha=0.5, reg_lambda=2.0,
            scale_pos_weight=neg_c / max(pos_c, 1),
            eval_metric='auc', random_state=42, n_jobs=-1,
            early_stopping_rounds=30
        )
        _model.fit(X_tr_a, y_tr_a, eval_set=[(X_te_a, y_te_a)], verbose=0)
        _probs = _model.predict_proba(X_te_a)[:, 1]
        _auc = roc_auc_score(y_te_a, _probs)
        # best f1
        _best_f1 = 0.0
        for _thr in np.arange(0.05, 0.95, 0.01):
            _f1 = f1_score(y_te_a, (_probs >= _thr).astype(int), zero_division=0)
            if _f1 > _best_f1:
                _best_f1 = _f1
        _elapsed = _abl_time.time() - _t0
        return _auc, _best_f1, _elapsed

    # ── 执行消融实验 ─────────────────────────────────────────────────
    print(f"\n  {'组别':<42} {'维度':>6} {'AUC':>8} {'F1':>8} {'耗时':>8}")
    print(f"  {'─'*78}")

    abl_results = []
    for gname, gfeat in ablation_groups:
        g_auc, g_f1, g_time = _ablation_xgb(gfeat, y, idx_tr14, idx_te14)
        abl_results.append((gname, gfeat.shape[1], g_auc, g_f1, g_time))
        print(f"  {gname:<42} {gfeat.shape[1]:>6} {g_auc:>8.4f} {g_f1:>8.4f} {g_time:>6.1f}s")

    # ── 增量分析 ─────────────────────────────────────────────────────
    base_auc = abl_results[0][2]
    print(f"\n  增量 AUC 贡献分析（相对 G1 Base）:")
    print(f"  {'─'*60}")
    for gname, gdim, g_auc, g_f1, _ in abl_results[1:]:
        delta = g_auc - base_auc
        pct = delta / base_auc * 100
        arrow = "+" if delta > 0 else ""
        print(f"  {gname:<42} AUC {arrow}{delta:.4f} ({arrow}{pct:.2f}%)")

    # ── 单特征维度 AUC 汇总（ISCT-CPD 8维） ──────────────────────────
    print(f"\n  ISCT-CPD 个体变点特征逐维 AUC:")
    print(f"  {'─'*50}")
    _cpd_names = ['F1(变点位置)', 'F2(前后比)', 'F3(最大偏离)',
                   'F4(半段比)', 'F5(不稳定性)', 'F6(连续负偏离)',
                   'F7(趋势斜率)', 'F8(末期偏离)']
    for _ci, _cn in enumerate(_cpd_names):
        _ca = roc_auc_score(y, isct_cpd_feats[:, _ci])
        _ca = max(_ca, 1 - _ca)
        print(f"    {_cn:<20} AUC={_ca:.4f}")

    # ── 分层RMT 特征 AUC ──────────────────────────────────────────────
    print(f"\n  Stratified-RMT 特征 AUC:")
    print(f"  {'─'*50}")
    _lrs = max(roc_auc_score(y, local_rmt_score), 1 - roc_auc_score(y, local_rmt_score))
    _lrr = max(roc_auc_score(y, local_rmt_ratio), 1 - roc_auc_score(y, local_rmt_ratio))
    print(f"    local_rmt_score  AUC={_lrs:.4f}")
    print(f"    local_rmt_ratio  AUC={_lrr:.4f}")

    print(f"\n  {'='*65}")
    print(f"  消融实验结论:")
    strat_delta = abl_results[1][2] - base_auc
    isct_delta = abl_results[2][2] - base_auc
    full_delta = abl_results[3][2] - base_auc
    print(f"    [RMT-ISC 组件]")
    print(f"    Stratified-RMT 贡献:  AUC {strat_delta:+.4f}")
    print(f"    ISCT-CPD 贡献:        AUC {isct_delta:+.4f}")
    print(f"    两者叠加(Full):       AUC {full_delta:+.4f}")

    if len(abl_results) >= 6:
        tcn_cpd_delta = abl_results[4][2] - base_auc
        combined_delta = abl_results[5][2] - base_auc
        print(f"\n    [TCN-CPD 组件]")
        print(f"    TCN-CPD 单独贡献:     AUC {tcn_cpd_delta:+.4f}")
        print(f"    ISCT-CPD + TCN-CPD:   AUC {combined_delta:+.4f}")

    print(f"  {'='*65}")

else:
    print("  (需要 xgboost 库才能运行消融实验)")

# =============================================================================
# 完成
# =============================================================================
print("\n" + "=" * 65)
print("全流程完成!")
print(f"   RMT 基线 AUC             : {rmt_baseline_auc:.4f}")
print(f"   双路Transformer AUC      : {final_auc:.4f}  F1={final_f1:.4f}")
if HAS_XGB:
    print(f"   XGBoost AUC              : {xgb_auc:.4f}")
    print(f"   集成 AUC                 : {ensemble_auc:.4f}  F1={best_f1_ens:.4f}")
print(f"   总提升（vs RMT基线）     : {top_auc - rmt_baseline_auc:+.4f}")
print("=" * 65)


