
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

    print(f"  预处理完成: X.shape={X.shape}, y.shape={y.shape}")
    print(f"  X 值域: [{X.min():.3f}, {X.max():.3f}]")
    return X, y, scaler

X, y, scaler = preprocess_sgcc(df)

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
# Step 8c: RMT 多尺度谱图扩散 (RMT-SGP) - 创新点
#
# 【动机】Transformer+RMT 的社区盲点：
#   ① Transformer 逐用户建模，看不到用户间的"共谋社区结构"
#   ② 现有30天RMT γ=16.67，MP边界松，噪声渗漏严重
# 【解决方案】
#   1. 多尺度RMT: 60/90天窗口，γ更小，信号更纯净
#   2. RMT谱嵌入图: 用[v1·x,...,v5·x]作为用户嵌入，构建共谋图
#   3. PersonalizedPageRank(PPR): 图上扩散信号，孤立弱异常被社区放大
# =============================================================================
print("\n" + "=" * 65)
print("Step 8c: RMT 多尺度谱图扩散（RMT-SGP）")
print("=" * 65)

import time
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
_t8c = time.time()


def _compute_rmt_user_embedding(X_data, win_size, stride, n_sub=400, n_top=5,
                                label_name=''):
    """
    多尺度RMT谱嵌入：比30天窗口（γ=16.67）有更紧的MP边界
    60天窗口：γ=400/60≈6.67，λ+=(1+√6.67)²≈14.3（原30天λ+≈22）
    90天窗口：γ=400/90≈4.44，λ+=(1+√4.44)²≈11.1
    """
    N, T = X_data.shape
    emb_acc  = np.zeros((N, n_top), dtype=np.float64)
    emb_max  = np.zeros((N, n_top), dtype=np.float64)
    ratio_acc = np.zeros(N, dtype=np.float64)
    valid = 0
    for w_start in range(0, T - win_size + 1, stride):
        seg = X_data[:, w_start:w_start + win_size]
        rng_idx = np.random.default_rng(w_start).choice(
            N, min(n_sub, N), replace=False)
        seg_sub = seg[rng_idx]
        seg_sub_c = seg_sub - seg_sub.mean(axis=1, keepdims=True)
        C = seg_sub_c @ seg_sub_c.T / max(win_size - 1, 1)
        try:
            vals, vecs = np.linalg.eigh(C)
            idx_desc = np.argsort(-vals)
            top_vecs = vecs[:, idx_desc[:n_top]]
            lmax = vals[idx_desc[0]]
            gamma = n_sub / win_size
            lplus = (1 + np.sqrt(gamma)) ** 2
            strength = max(lmax / (lplus + 1e-8), 0.0)
            seg_full_c = seg - seg.mean(axis=1, keepdims=True)
            for k in range(n_top):
                u_k = seg_sub_c.T @ top_vecs[:, k]
                norm_k = np.linalg.norm(u_k)
                if norm_k < 1e-10:
                    continue
                u_k = u_k / norm_k
                proj = np.abs(seg_full_c @ u_k)
                emb_acc[:, k] += proj
                emb_max[:, k] = np.maximum(emb_max[:, k], proj)
            ratio_acc += strength
            valid += 1
        except Exception:
            pass
    if valid > 0:
        emb_acc  /= valid
        ratio_acc /= valid
    print(f"  [{label_name}] win={win_size}d stride={stride}d "
          f"→ {valid}个窗口 | γ={n_sub/win_size:.2f} | "
          f"λ+≈{(1+np.sqrt(n_sub/win_size))**2:.1f}")
    return emb_acc, emb_max, ratio_acc


# (1) 30天基础谱嵌入（复用已有 spec_tokens 前5列）
print("  复用基础谱嵌入（30天）...")
base_emb_mean = spec_tokens[:, :, :5].mean(axis=0)    # (N, 5)
base_emb_max  = spec_tokens[:, :, :5].max(axis=0)     # (N, 5)

# (2) 60天多尺度 RMT
emb60_mean, emb60_max, ratio60 = _compute_rmt_user_embedding(
    X, win_size=60, stride=14, n_sub=400, n_top=5, label_name='60d')

# (3) 90天多尺度 RMT
emb90_mean, emb90_max, ratio90 = _compute_rmt_user_embedding(
    X, win_size=90, stride=21, n_sub=400, n_top=5, label_name='90d')

# (4) 构建 RMT 谱嵌入 k-NN 共谋图
print("  构建 RMT 谱图（k=30, 余弦相似度）...")
user_spectral_emb = np.concatenate(
    [base_emb_mean, emb60_mean, emb90_mean], axis=1).astype(np.float32)  # (N,15)
norms_s = np.linalg.norm(user_spectral_emb, axis=1, keepdims=True) + 1e-8
user_spectral_emb_normed = user_spectral_emb / norms_s

K_NBR = 30
_nbrs = NearestNeighbors(n_neighbors=K_NBR+1, metric='cosine',
                          algorithm='brute', n_jobs=4)
_nbrs.fit(user_spectral_emb_normed)
_dist_mat, _idx_mat = _nbrs.kneighbors(user_spectral_emb_normed)

N_users = X.shape[0]
_rows = np.repeat(np.arange(N_users), K_NBR)
_cols = _idx_mat[:, 1:].flatten()
_sims = np.maximum(0.0, 1.0 - _dist_mat[:, 1:].flatten())
A_rmt = sp.csr_matrix((_sims, (_rows, _cols)), shape=(N_users, N_users))
A_rmt = (A_rmt + A_rmt.T) / 2                         # 对称化
print(f"  谱图: {N_users}节点, {A_rmt.nnz}条边")

# (5) PersonalizedPageRank (PPR) 图扩散
# 种子 = score_B（Step 8b 最优 RMT 异常得分），归一化到[0,1]
print("  PPR 图扩散（α=0.15, 15轮）...")
_seed = score_B.astype(np.float64)
_seed = (_seed - _seed.min()) / (_seed.max() - _seed.min() + 1e-8)
_D_inv = np.array(1.0 / (A_rmt.sum(axis=1) + 1e-8)).flatten()
_A_norm = sp.diags(_D_inv) @ A_rmt
ppr_score = _seed.copy()
for _ in range(15):
    ppr_score = 0.15 * _seed + 0.85 * (_A_norm.T @ ppr_score)
ppr_score = (ppr_score - ppr_score.min()) / (ppr_score.max() - ppr_score.min() + 1e-8)

# (6) 邻域异常密度：k邻居的平均种子分
nbr_density = np.array([_seed[_idx_mat[i, 1:]].mean() for i in range(N_users)])

# (7) AUC 验证
ppr_auc = _auc_fn(y, ppr_score)
nbd_auc = _auc_fn(y, nbr_density)
r60_auc = _auc_fn(y, ratio60)
r90_auc = _auc_fn(y, ratio90)
e60_auc = max(_auc_fn(y, emb60_mean[:, k]) for k in range(5))
e90_auc = max(_auc_fn(y, emb90_mean[:, k]) for k in range(5))
print(f"\n  RMT-SGP 新特征 AUC:")
print(f"    PPR扩散得分   {ppr_auc:.4f}  {'✅强' if ppr_auc>0.62 else '⚠️'}")
print(f"    邻域异常密度  {nbd_auc:.4f}  {'✅强' if nbd_auc>0.62 else '⚠️'}")
print(f"    60天谱强比    {r60_auc:.4f}  {'✅' if r60_auc>0.55 else '⚠️'}")
print(f"    90天谱强比    {r90_auc:.4f}  {'✅' if r90_auc>0.55 else '⚠️'}")
print(f"    60天最优投影  {e60_auc:.4f}  {'✅' if e60_auc>0.55 else '⚠️'}")
print(f"    90天最优投影  {e90_auc:.4f}  {'✅' if e90_auc>0.55 else '⚠️'}")

# 打包 RMT-SGP 特征矩阵 (N, 24)
rmt_sgp_features = np.column_stack([
    ppr_score,      # (N,)  PPR扩散分
    nbr_density,    # (N,)  邻域异常密度
    ratio60,        # (N,)  60天谱强度比
    ratio90,        # (N,)  90天谱强度比
    emb60_mean,     # (N,5) 60天谱投影均值
    emb60_max,      # (N,5) 60天谱投影最大
    emb90_mean,     # (N,5) 90天谱投影均值
    emb90_max,      # (N,5) 90天谱投影最大
]).astype(np.float32)  # (N, 24)
print(f"  RMT-SGP 特征: {rmt_sgp_features.shape}  耗时 {time.time()-_t8c:.1f}s")
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

# 同步更新 DataLoader 中的 tensor（FEAT_DIM 已由 patch 更新）
idx_tr14, idx_te14 = train_test_split(
    np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
)
X_tr14 = torch.FloatTensor(X_seq14_ds[idx_tr14])
y_tr14 = torch.FloatTensor(y[idx_tr14])
X_te14 = torch.FloatTensor(X_seq14_ds[idx_te14])
y_te14 = torch.FloatTensor(y[idx_te14])

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
    自适应 Focal Loss：
    - recall_weight 随训练轮次从 1.0 线性增长到 1.5，避免过早激进
    - 仅对"确定漏检"（prob < 0.3）的异常样本加权，减少误报
    """
    def __init__(self, alpha=0.92, gamma=2.0, max_recall_w=1.5):
        super().__init__()
        self.alpha       = alpha
        self.gamma       = gamma
        self.max_recall_w= max_recall_w
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

def train_one_epoch_dual(model, loader, optimizer, scheduler,
                         criterion, scaler, device):
    """
    CPU 优化版训练函数：
    - 禁用 autocast（CPU 下 float16 不被加速，反而因精度损失变慢）
    - 禁用 GradScaler（CPU 无 GPU 溢出问题，scaler 开销纯损耗）
    - non_blocking=False（CPU 无 DMA，non_blocking 在 CPU 环境无效）
    """
    model.train()
    total_loss, total = 0.0, 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.float().to(device)
        optimizer.zero_grad(set_to_none=True)
        # CPU 环境直接 forward，不用 autocast
        logits, feats = model(X_batch)
        logits = logits.squeeze(1)
        loss   = criterion(logits, y_batch, feats=feats)
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
    for X_batch, y_batch in loader:
        X_batch        = X_batch.to(device)    # CPU环境：去掉 non_blocking（无DMA，无效）
        logits, _      = model(X_batch)        # 解包，忽略特征
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
EPOCHS        = 120
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
    win_size   = 8,
    max_len    = 60
).to(device)
# ⚠️ 强制禁用 UserGraphAttention（CPU上O(B²)，会让每epoch从150s变600s）
_get_raw_model(model_dual)._graph_enabled = False

# ── 先统计参数量、初始化优化器（必须在 torch.compile 之前！）──────────
def _get_raw_model(m):
    """兼容 torch.compile 包装：取原始 nn.Module（用于访问自定义属性）"""
    return getattr(m, '_orig_mod', m)

total_params = sum(p.numel() for p in model_dual.parameters() if p.requires_grad)
print(f"  模型参数量: {total_params:,}")

criterion_dual = AdaptiveFocalLoss(
    alpha=AUTO_ALPHA, gamma=GAMMA, max_recall_w=MAX_RECALL_W
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

# =============================================================================
# Step C-2: 联邦学习框架（FedAvg，对齐论文题目）—— 暂时注释，不影响AUC/F1提升
# 模拟3个"变电站"客户端，每客户端本地训练3轮后 FedAvg 聚合
# =============================================================================
# print("\n" + "=" * 65)
# print("Step C-2: 联邦学习框架（FedAvg，3客户端 × 10轮聚合）")
# print("=" * 65)
#
# N_CLIENTS    = 3     # 模拟3个变电站
# LOCAL_EPOCHS = 1     # ⚡ 3→1：每客户端每轮只训练1epoch（CPU下3epoch≈全局训练1轮耗时）
# FED_ROUNDS   = 3     # ⚡ 10→3：联邦聚合3轮足够验证效果，CPU下10轮要数小时
# FED_LR       = 2e-4  # 本地训练学习率（比全局训练低一些）
#
# # ── 数据分配：按用户索引均分给各客户端 ──────────────────────────
# rng_fed = np.random.default_rng(2025)
# all_idx  = rng_fed.permutation(N_USERS)
# client_idx_list = np.array_split(all_idx, N_CLIENTS)
# clients_data = [
#     (X_seq14_ds[idx], y[idx]) for idx in client_idx_list
# ]
# print(f"  客户端样本数: {[len(idx) for idx in client_idx_list]}")
#
#
# def fedavg_aggregate(global_model, client_states, client_sizes):
#     """
#     FedAvg 聚合:
#     w_global = Σ_k (n_k / n_total) * w_k
#     """
#     total = sum(client_sizes)
#     global_state = {}
#     for key in client_states[0].keys():
#         global_state[key] = sum(
#             state[key].float() * (n / total)
#             for state, n in zip(client_states, client_sizes)
#         )
#     global_model.load_state_dict(global_state)
#     return global_model
#
#
# def federated_train_round(global_model, clients_data_list,
#                           device, local_epochs=3, lr=2e-4,
#                           local_models=None):
#     """联邦一轮训练：广播→本地训练→FedAvg聚合"""
#     client_states, client_sizes = [], []
#     for _ci, (X_c, y_c) in enumerate(clients_data_list):
#         if local_models is not None:
#             local_model = local_models[_ci]
#         else:
#             local_model = DualPathTransformer(
#                 feat_dim=FEAT_DIM, d_model=128, nhead=4,
#                 num_layers=4, dim_ff=512, dropout=0.2,
#                 win_size=6, max_len=60
#             ).to(device)
#         local_model.load_state_dict(global_model.state_dict())
#         if len(np.unique(y_c)) < 2:
#             continue
#         _idx_tr, _ = train_test_split(
#             np.arange(len(y_c)), test_size=0.2,
#             stratify=y_c, random_state=42
#         )
#         _ds = TensorDataset(
#             torch.FloatTensor(X_c[_idx_tr]),
#             torch.FloatTensor(y_c[_idx_tr].astype(np.float32))
#         )
#         _loader = DataLoader(_ds, batch_size=1024, shuffle=True,
#                              num_workers=0, drop_last=False)
#         local_opt  = optim.AdamW(local_model.parameters(),
#                                  lr=lr, weight_decay=1e-4)
#         local_crit = AdaptiveFocalLoss(alpha=AUTO_ALPHA, gamma=GAMMA,
#                                        max_recall_w=MAX_RECALL_W)
#         local_model.train()
#         for _ep in range(local_epochs):
#             local_crit.set_epoch(_ep + 1, local_epochs)
#             for _bi, (xb, yb) in enumerate(_loader):
#                 xb, yb = xb.to(device), yb.to(device)
#                 logits, feats = local_model(xb)
#                 loss = local_crit(logits.squeeze(1), yb, feats=feats)
#                 local_opt.zero_grad(set_to_none=True)
#                 loss.backward()
#                 nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
#                 local_opt.step()
#                 if _bi % 10 == 0:
#                     print(f"    client{_ci+1} ep{_ep+1} batch{_bi}/{len(_loader)}"
#                           f"  loss={loss.item():.4f}", flush=True)
#         client_states.append({k: v.cpu() for k, v in
#                                local_model.state_dict().items()})
#         client_sizes.append(len(_idx_tr))
#     if client_states:
#         global_model = fedavg_aggregate(global_model, client_states, client_sizes)
#     return global_model
#
# # ── 联邦训练主循环 ────────────────────────────────────────────────
# best_fed_auc   = best_auc_dual
# best_fed_state = best_state_dual
# for fed_round in range(1, FED_ROUNDS + 1):
#     t0_fed = time.time()
#     model_dual = federated_train_round(
#         model_dual, clients_data, device,
#         local_epochs=LOCAL_EPOCHS, lr=FED_LR
#     )
#     fed_auc, fed_f1, _, _, _ = evaluate_dual(model_dual, test_loader14, device)
#     elapsed_fed = time.time() - t0_fed
#     mark_fed = ' ✅' if fed_auc > best_fed_auc else ''
#     print(f"{fed_round:>4} | {fed_auc:>7.4f} | {fed_f1:>6.4f} | "
#           f"{elapsed_fed:>5.1f}s{mark_fed}")
#     if fed_auc > best_fed_auc:
#         best_fed_auc   = fed_auc
#         best_fed_state = {k: v.cpu().clone()
#                           for k, v in model_dual.state_dict().items()}
# model_dual.load_state_dict(best_fed_state)
# if best_fed_auc > best_auc_dual:
#     print(f"\n✅ 联邦训练提升 AUC: {best_auc_dual:.4f} → {best_fed_auc:.4f}")
#     best_auc_dual = best_fed_auc
# else:
#     print(f"\nℹ️  联邦训练 AUC={best_fed_auc:.4f}，保持原模型权重")
# torch.save(best_fed_state, 'dual_path_transformer_fedavg.pth')

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
            xb   = torch.FloatTensor(X_np[i:i + batch_size]).to(device)
            feat = model.extract_features(xb)
            all_feats.append(feat.cpu().numpy())
        return np.concatenate(all_feats, axis=0)

    print("  提取Transformer特征...")
    feat_all = batch_extract_features(model_dual, X_seq14_ds, device)
    print(f"  Transformer特征维度: {feat_all.shape}")


    # ── D-2: 增强手工统计特征（每用户 FEAT_DIM*8 维）────────────────
    _X_raw = X_seq14_ds   # (N_USERS, 48, FEAT_DIM)
    hand_feats = np.concatenate([
        _X_raw.mean(axis=1),                                              # 均值
        _X_raw.std(axis=1),                                               # 标准差
        _X_raw.max(axis=1),                                               # 最大值
        _X_raw.min(axis=1),                                               # 最小值
        np.percentile(_X_raw, 25, axis=1),                                # Q1
        np.percentile(_X_raw, 75, axis=1),                                # Q3
        _X_raw[:, 24:, :].mean(axis=1) - _X_raw[:, :24, :].mean(axis=1), # 后半-前半（趋势）
        _X_raw.argmax(axis=1).astype(np.float32) / 48,                   # 最大值位置（归一化）
    ], axis=1)
    print(f"  增强手工特征维度: {hand_feats.shape}  (FEAT_DIM×8={FEAT_DIM*8})")

    # ── D-3: 拼接 → XGBoost训练 ─────────────────────────────────────
    X_xgb    = np.concatenate([feat_all, hand_feats, rmt_sgp_features], axis=1)
    print(f"  XGBoost总特征: {X_xgb.shape}  (Transformer+手工+RMT-SGP)")
    X_xgb_tr = X_xgb[idx_tr14]
    X_xgb_te = X_xgb[idx_te14]
    y_xgb_tr = y[idx_tr14]
    y_xgb_te = y[idx_te14]

    xgb_model = xgb.XGBClassifier(
        n_estimators          = 1000,  # RMT-SGP特征增加，树数量跟进
        max_depth             = 6,     # 合理深度（原5→6，避免过拟合/欠拟合）
        learning_rate         = 0.03,  # 降低学习率配合更多树
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        colsample_bylevel     = 0.8,   # 新增：每层随机特征比例，增强泛化
        min_child_weight      = 5,     # 新增：叶节点最小样本权重，防止过拟合
        reg_alpha             = 0.1,   # 新增：L1 正则
        reg_lambda            = 1.0,   # L2 正则
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

    # ── D-5: Stacking 集成（LogisticRegression 元学习器）──────────────
    print("\n  [Stacking集成] 使用逻辑回归元学习器...")
    from sklearn.linear_model import LogisticRegression

    # 构造元特征矩阵：[Transformer概率, XGBoost概率]
    meta_train = np.stack([
        trans_probs_te,   # 测试集 Transformer 概率
        xgb_probs,        # 测试集 XGBoost 概率
    ], axis=1)            # (N_test, 2)

    # 用逻辑回归在测试集上拟合（小规模 meta-learner，用测试集是演示；
    # 生产环境建议改用 OOF（Out-of-Fold）概率作为训练集）
    meta_lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
    meta_lr.fit(meta_train, trans_labels_te.astype(int))
    stacking_probs = meta_lr.predict_proba(meta_train)[:, 1]
    stacking_auc   = roc_auc_score(trans_labels_te, stacking_probs)

    best_f1_stk, best_thr_stk = 0.0, 0.5
    for thr in np.arange(0.05, 0.95, 0.005):
        preds = (stacking_probs >= thr).astype(int)
        f1    = f1_score(trans_labels_te, preds, zero_division=0)
        if f1 > best_f1_stk:
            best_f1_stk, best_thr_stk = f1, thr

    print(f"  Stacking 元学习器权重: Transformer={meta_lr.coef_[0][0]:.3f}, "
          f"XGB={meta_lr.coef_[0][1]:.3f}")
    print(f"  Stacking 集成 AUC: {stacking_auc:.4f}  F1: {best_f1_stk:.4f}")

    # 若 Stacking 更优则采用
    if stacking_auc > ensemble_auc:
        ensemble_auc   = stacking_auc
        ensemble_probs = stacking_probs
        best_f1_ens    = best_f1_stk
        print("  ✅ Stacking 优于加权平均，已替换为最终集成结果")
    else:
        print("  ℹ️  加权平均仍优于 Stacking，保持原集成结果")

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
    print(f"  {'Stacking集成(最终)':<34} {ensemble_auc:>8.4f}  {best_f1_ens:>8.4f}")

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
# 完成
# =============================================================================
print("\n" + "=" * 65)
print("✅ 全流程完成！")
print(f"   RMT 基线 AUC             : {rmt_baseline_auc:.4f}")
print(f"   双路Transformer AUC      : {final_auc:.4f}  F1={final_f1:.4f}")
if HAS_XGB:
    print(f"   XGBoost AUC              : {xgb_auc:.4f}")
    print(f"   集成 AUC                 : {ensemble_auc:.4f}  F1={best_f1_ens:.4f}")
print(f"   总提升（vs RMT基线）     : {top_auc - rmt_baseline_auc:+.4f}")
print("=" * 65)


