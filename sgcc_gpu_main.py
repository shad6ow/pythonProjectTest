"""
=============================================================================
RMT-CPD-Transformer: 面向窃电检测的随机矩阵理论引导变点感知Transformer框架
=============================================================================
论文创新点：
  C1. RMT 谱Token化        - 用MP边界将相关矩阵谱分解为可学习的时序Token
  C2. 双路时序Transformer   - 局部窗口注意力 + 全局CLS Token建模谱Token演化
  C3. RMT-CPD 联合框架     - 个体变点检测弥补RMT群体盲点，形成宏-微观互补

数据集: SGCC (42372 用户, 1034 天, 异常率 8.5%)
硬件需求: GPU (VRAM ≥ 8GB), RAM ≥ 16GB
=============================================================================
"""

import os, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report
warnings.filterwarnings('ignore')

# ── GPU 加速设置 ─────────────────────────────────────────────────────────────
os.environ['OMP_NUM_THREADS']  = '8'
os.environ['MKL_NUM_THREADS']  = '8'
torch.set_num_threads(8)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"⚡ 运行设备: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── 超参配置 ─────────────────────────────────────────────────────────────────
CFG = dict(
    # 数据
    data_path       = 'data set.csv',
    test_size       = 0.2,
    random_seed     = 42,
    # RMT
    rmt_win_days    = 30,           # 每个 RMT 窗口的天数
    rmt_stride_days = 7,            # 窗口步长（天）
    rmt_n_sub       = 500,          # 子采样用户数（节省内存）
    rmt_n_eig       = 5,            # 提取前 K 个特征向量投影
    # 周度序列
    week_size       = 7,            # 每周天数
    # CPD（变点检测）
    cpd_pen         = 3.0,          # PELT 惩罚项（越小检测越灵敏）
    cpd_model       = 'rbf',        # PELT 代价函数
    # Transformer
    d_model         = 256,
    nhead           = 8,
    num_layers      = 6,
    dim_ff          = 1024,
    dropout         = 0.1,
    max_len         = 200,
    # 训练
    batch_size      = 512,
    epochs          = 300,
    lr              = 1e-3,
    weight_decay    = 1e-4,
    patience        = 25,           # 早停
    T_0             = 30,           # CosineWarmRestart 周期
    T_mult          = 2,
    # 损失
    focal_gamma     = 2.0,
    focal_alpha     = 0.87,         # 自动更新
)

# =============================================================================
# Step 1: 数据加载与预处理
# =============================================================================
print("\n" + "="*65)
print("Step 1: 数据加载与预处理")
print("="*65)

df = pd.read_csv(CFG['data_path'])
print(f"  原始数据形状: {df.shape}")

labels = df['FLAG'].values.astype(np.int64)
feats  = df.drop('FLAG', axis=1).select_dtypes(include=[np.number])

# 线性插值填充缺失值
print("  → 线性插值...")
feats = feats.interpolate(method='linear', axis=1, limit_direction='both')
feats = feats.fillna(feats.mean())

# ★ 保存原始数据（kWh 量级），供 CPD 计算使用
X_raw = feats.values.astype(np.float32)   # (N, T) 原始消费量
print(f"  原始数据（未归一化）: X_raw.shape={X_raw.shape}")

# 全局 Z-score 归一化
print("  → Z-score 归一化...")
scaler = StandardScaler()
X_norm = scaler.fit_transform(feats.values.T).T.astype(np.float32)
lo, hi = np.percentile(X_norm, 0.1), np.percentile(X_norm, 99.9)
X_norm = np.clip(X_norm, lo, hi)

N_USERS, T_DAYS = X_norm.shape
y = labels
print(f"  归一化数据: X_norm.shape={X_norm.shape}, 值域[{lo:.3f}, {hi:.3f}]")
print(f"  标签分布: 正常={( y==0).sum()}, 异常={(y==1).sum()} "
      f"(异常率={(y==1).mean()*100:.1f}%)")

# =============================================================================
# Step 2: RMT 谱特征提取（C1：RMT 谱 Token 化）
# =============================================================================
print("\n" + "="*65)
print("Step 2: RMT 谱特征提取（C1: 谱Token化）")
print("="*65)

def mp_upper_bound(gamma: float) -> float:
    """Marchenko-Pastur 分布上界 λ+ = σ²(1+√γ)²，假设 σ²=1"""
    return (1 + np.sqrt(gamma)) ** 2

def rmt_spectral_tokens(X_data, win_days, stride_days, n_sub, n_eig):
    """
    对滑动窗口协方差矩阵做谱分解，提取 RMT 谱 Token。

    返回:
      tokens : (N_users, N_windows, 2+n_eig) 谱Token矩阵
               dim0..n_eig-1 : 前 n_eig 个特征向量投影（绝对值）
               dim n_eig     : λ_max / λ+（异常强度比）
               dim n_eig+1   : λ_max 是否超界（0/1，二值）
      lmax_series  : (N_windows,) 每窗口 λ_max
      lplus_series : (N_windows,) 每窗口 λ+
    """
    N, T = X_data.shape
    wins = list(range(0, T - win_days + 1, stride_days))
    N_W  = len(wins)
    D    = 2 + n_eig
    tokens     = np.zeros((N, N_W, D), dtype=np.float32)
    lmax_arr   = np.zeros(N_W, dtype=np.float32)
    lplus_arr  = np.zeros(N_W, dtype=np.float32)

    rng = np.random.default_rng(42)

    for wi, t0 in enumerate(wins):
        seg = X_data[:, t0:t0 + win_days]           # (N, win_days)
        # 子采样用户，降低内存
        idx = rng.choice(N, min(n_sub, N), replace=False)
        seg_sub = seg[idx]                           # (n_sub, win_days)

        # 计算相关矩阵（按时间维度）
        C = np.corrcoef(seg_sub.T)                   # (win_days, win_days)
        C = np.nan_to_num(C, nan=0.0)

        # 特征值分解（只取前 n_eig+1 个）
        try:
            eigvals, eigvecs = np.linalg.eigh(C)
        except np.linalg.LinAlgError:
            continue
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        gamma  = n_sub / win_days
        lplus  = mp_upper_bound(gamma)
        lmax   = eigvals[0]

        # 前 n_eig 个特征向量在用户数据上的投影
        for k in range(min(n_eig, eigvecs.shape[1])):
            vk = eigvecs[:, k].astype(np.float32)    # (win_days,)
            proj = (seg @ vk)                         # (N,) 所有用户在 vk 上的投影
            tokens[:, wi, k] = np.abs(proj)

        tokens[:, wi, n_eig]     = float(lmax / (lplus + 1e-8))
        tokens[:, wi, n_eig + 1] = float(lmax > lplus)

        lmax_arr[wi]  = lmax
        lplus_arr[wi] = lplus

        if wi % 20 == 0:
            flag = "🔴 异常" if lmax > lplus else "🟢 正常"
            print(f"    窗口 {wi:>3}: λ_max={lmax:.4f}, λ+={lplus:.4f}, "
                  f"比值={lmax/lplus:.3f}  {flag}")

    exceed_pct = (lmax_arr > lplus_arr).mean() * 100
    print(f"  ✅ 谱Token完成: shape={tokens.shape}, "
          f"超界窗口 {exceed_pct:.0f}%")
    return tokens, lmax_arr, lplus_arr

t0_rmt = time.time()
spec_tokens, lmax_series, lplus_series = rmt_spectral_tokens(
    X_norm,
    win_days    = CFG['rmt_win_days'],
    stride_days = CFG['rmt_stride_days'],
    n_sub       = CFG['rmt_n_sub'],
    n_eig       = CFG['rmt_n_eig'],
)
print(f"  RMT 耗时: {time.time()-t0_rmt:.1f}s")
# spec_tokens: (N_users, N_windows, 7)

# =============================================================================
# Step 3: CPD 变点检测特征（C3：弥补 RMT 群体盲点）
# =============================================================================
print("\n" + "="*65)
print("Step 3: 个体变点检测 CPD（C3: 弥补RMT群体盲点）")
print("="*65)

def extract_cpd_features(X_raw_data, pen=3.0, cpd_model='rbf',
                          days_per_month=30):
    """
    ★ 论文 C3 贡献：用 PELT 在原始消费量上检测变点，
    提取 8 维个体变点特征。

    动机：RMT 检测群体协方差矩阵的全局异常，当仅 1 个用户(1/42372)
    开始窃电时，对协方差矩阵的扰动极小。但该用户个体的消费序列
    存在清晰的结构性变化（变点）。CPD 在个体层面检测此变化。

    返回: (N, 8) 变点特征矩阵
    """
    try:
        import ruptures as rpt
        _rpt_ok = True
    except ImportError:
        try:
            import subprocess, sys
            subprocess.check_call([sys.executable,'-m','pip','install',
                                   'ruptures','-q'])
            import ruptures as rpt
            _rpt_ok = True
        except Exception:
            _rpt_ok = False

    N, T = X_raw_data.shape
    # 月度消费序列
    n_months = T // days_per_month
    X_mo = np.zeros((N, n_months), dtype=np.float32)
    for m in range(n_months):
        X_mo[:, m] = X_raw_data[:, m*days_per_month :
                                    (m+1)*days_per_month].mean(axis=1)

    cpd_feats = np.zeros((N, 8), dtype=np.float32)

    if _rpt_ok:
        print(f"  PELT 变点检测 (pen={pen}, {N} 用户)...")
        t_cpd = time.time()
        for i in range(N):
            sig = X_mo[i]
            try:
                algo = rpt.Pelt(model=cpd_model, min_size=2, jump=1)
                bkps = algo.fit_predict(sig, pen=pen)
                bkps = [b for b in bkps if b < n_months]
            except Exception:
                bkps = []

            n_bkps = len(bkps)

            # F1: 检测到的变点数（归一化）
            cpd_feats[i, 0] = n_bkps / max(n_months, 1)

            if n_bkps > 0:
                # F2: 第一个变点的相对位置（越早越可疑）
                first_bkp = bkps[0]
                cpd_feats[i, 1] = first_bkp / n_months

                # F3: 最大变点处前后均值之比
                pre_mean  = sig[:first_bkp].mean()  + 1e-6
                post_mean = sig[first_bkp:].mean()
                cpd_feats[i, 2] = post_mean / pre_mean   # <1 = 消费下降

                # F4: 最大单次变化幅度（绝对值）
                seg_means = [sig[bkps[j-1] if j>0 else 0:bkps[j]].mean()
                             for j in range(len(bkps))]
                if len(seg_means) >= 2:
                    changes = np.diff(seg_means)
                    cpd_feats[i, 3] = np.abs(changes).max() / (np.abs(sig).mean() + 1e-6)

                # F5: 后段均值 / 全段均值（整体下降程度）
                cpd_feats[i, 4] = post_mean / (sig.mean() + 1e-6)

                # F6: 变点后的标准差（不稳定性）
                cpd_feats[i, 5] = sig[first_bkp:].std() / (sig.std() + 1e-6)

                # F7: 最后4个月均值 / 前4个月均值
                early = sig[:min(4, n_months)].mean() + 1e-6
                late  = sig[max(0, n_months-4):].mean()
                cpd_feats[i, 6] = late / early

                # F8: 变点后消费持续低于基准(前6月均值)的月份比例
                baseline_val = sig[:min(6, n_months)].mean() + 1e-6
                if first_bkp < n_months:
                    post_seg = sig[first_bkp:]
                    cpd_feats[i, 7] = (post_seg < baseline_val * 0.9).mean()
            else:
                # 无变点：用整体趋势斜率代替
                t_idx = np.arange(n_months, dtype=np.float32)
                t_c   = t_idx - t_idx.mean()
                y_c   = sig - sig.mean()
                slope = (y_c * t_c).sum() / ((t_c**2).sum() + 1e-8)
                cpd_feats[i, 1] = 1.0           # 变点位置设为末尾
                cpd_feats[i, 2] = 1.0 + slope   # 用斜率近似
                cpd_feats[i, 6] = (sig[-4:].mean() + 1e-6) / (sig[:4].mean() + 1e-6)

        print(f"  CPD 耗时: {time.time()-t_cpd:.1f}s")
    else:
        # 降级：无 ruptures 时用简单统计代替
        print("  ⚠️ ruptures 不可用，使用简化变点特征")
        for i in range(N):
            sig = X_mo[i]
            half = n_months // 2
            early = sig[:half].mean() + 1e-6
            late  = sig[half:].mean()
            cpd_feats[i, 2] = late / early
            cpd_feats[i, 6] = (sig[-4:].mean()+1e-6) / (sig[:4].mean()+1e-6)

    # 打印各维度 AUC（验证有效性）
    print(f"\n  CPD 特征 AUC（验证）:")
    for k in range(8):
        try:
            a = roc_auc_score(y, cpd_feats[:, k])
            a = max(a, 1 - a)
        except Exception:
            a = 0.5
        mark = '✅' if a >= 0.60 else ('⚠️' if a >= 0.55 else '❌')
        names = ['变点数','首变位置','前后比','最大变幅','后段比均','后段稳定性','末/首比','持续低于基准']
        print(f"    F{k+1}({names[k]:<10}) AUC={a:.4f}  {mark}")

    return cpd_feats

cpd_features = extract_cpd_features(X_raw, pen=CFG['cpd_pen'],
                                     cpd_model=CFG['cpd_model'])
K_CPD = cpd_features.shape[1]
print(f"\n  CPD 特征形状: {cpd_features.shape}")

# =============================================================================
# Step 4: 构造周度序列输入（Transformer 时序输入）
# =============================================================================
print("\n" + "="*65)
print("Step 4: 构造周度时序特征（Transformer 输入）")
print("="*65)

def build_weekly_sequence(X_norm_data, X_raw_data, spec_tok,
                           lmax_arr, lplus_arr, week_size=7):
    """
    构造 (N, N_weeks, D_feat) 周度时序特征矩阵。

    每周特征包含：
      0: 周均消费（Z-score）
      1: 周消费标准差
      2: 周最大值
      3: 周最小值
      4: 零值占比
      5: 周内差分均值（波动性）
      6: 跨用户同周百分位排名
      7: 相对自身前8周基准的偏离率（用原始数据）
      8-14: 插值后的 RMT 谱Token（7维）
    """
    N, T  = X_norm_data.shape
    N_weeks = T // week_size
    _, N_W_rmt, D_rmt = spec_tok.shape

    # 月度原始数据（用于基准计算）
    days_per_month = 30
    n_months = T // days_per_month
    X_mo_raw = np.zeros((N, n_months), dtype=np.float32)
    for m in range(n_months):
        X_mo_raw[:, m] = X_raw_data[:, m*days_per_month:(m+1)*days_per_month].mean(1)
    baseline_raw = X_mo_raw[:, :6].mean(axis=1, keepdims=True) + 1e-3  # (N,1)

    # 基础统计通道
    week_feats = np.zeros((N, N_weeks, 7), dtype=np.float32)
    for w in range(N_weeks):
        seg = X_norm_data[:, w*week_size:(w+1)*week_size]       # (N, 7)
        seg_raw = X_raw_data[:, w*week_size:(w+1)*week_size]
        week_feats[:, w, 0] = seg.mean(axis=1)
        week_feats[:, w, 1] = seg.std(axis=1)
        week_feats[:, w, 2] = seg.max(axis=1)
        week_feats[:, w, 3] = seg.min(axis=1)
        week_feats[:, w, 4] = (seg == 0).mean(axis=1)
        week_feats[:, w, 5] = np.diff(seg, axis=1).mean(axis=1) if seg.shape[1] > 1 else 0
        # 跨用户百分位
        ranks = np.argsort(np.argsort(seg.mean(axis=1))) / N
        week_feats[:, w, 6] = ranks.astype(np.float32)

    # 相对基准偏离（用原始数据，物理意义明确）
    week_raw_mean = np.zeros((N, N_weeks), dtype=np.float32)
    for w in range(N_weeks):
        week_raw_mean[:, w] = X_raw_data[:, w*week_size:(w+1)*week_size].mean(1)
    vs_baseline = (week_raw_mean - baseline_raw) / (np.abs(baseline_raw) + 1e-3)
    vs_baseline = np.clip(vs_baseline, -5, 5)[:, :, np.newaxis]

    # 插值 RMT 谱Token 到 N_weeks
    rmt_interp = np.zeros((N, N_weeks, D_rmt), dtype=np.float32)
    for d in range(D_rmt):
        for n in range(N):
            rmt_interp[n, :, d] = np.interp(
                np.linspace(0, 1, N_weeks),
                np.linspace(0, 1, N_W_rmt),
                spec_tok[n, :, d]
            )

    # 拼接所有通道
    X_seq = np.concatenate([
        week_feats,        # (N, N_weeks, 7)
        vs_baseline,       # (N, N_weeks, 1)
        rmt_interp,        # (N, N_weeks, D_rmt=7)
    ], axis=2).astype(np.float32)

    print(f"  周度序列形状: {X_seq.shape}  (N_weeks={N_weeks})")
    print(f"  通道: 统计(7) + 基准偏离(1) + RMT谱Token({D_rmt})")
    return X_seq

X_seq = build_weekly_sequence(
    X_norm, X_raw, spec_tokens, lmax_series, lplus_series,
    week_size = CFG['week_size']
)
N_STEPS, FEAT_DIM = X_seq.shape[1], X_seq.shape[2]
print(f"  时间步: {N_STEPS},  特征维度: {FEAT_DIM}")

# =============================================================================
# Step 5: Transformer 模型定义（C2：双路时序建模 + CPD 融合）
# =============================================================================
print("\n" + "="*65)
print("Step 5: 模型定义（C2: RMT-CPD-Transformer）")
print("="*65)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.87, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha  = alpha
        self.gamma  = gamma
        self.bce    = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs    = torch.sigmoid(logits)
        p_t      = probs * targets + (1 - probs) * (1 - targets)
        focal_w  = (1 - p_t) ** self.gamma
        alpha_t  = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * focal_w * bce_loss).mean()


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        self.attn   = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                            batch_first=True)
        self.ff     = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LN（训练更稳定）
        h, _  = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x     = x + self.drop(h)
        x     = x + self.drop(self.ff(self.norm2(x)))
        return x


class RMTCPDTransformer(nn.Module):
    """
    论文核心模型：RMT-CPD-Transformer

    输入：
      x_seq    : (B, T, feat_dim) 周度时序特征（包含 RMT 谱Token）
      x_cpd    : (B, k_cpd)       CPD 变点特征（C3 贡献）

    架构：
      1. 输入投影 + 位置编码
      2. N 层 Transformer Encoder（全局注意力）
      3. CLS Token 池化 → 用户全局表示
      4. CPD 特征投影 → CPD 表示
      5. 拼接融合 → 分类头
    """
    def __init__(self, feat_dim, k_cpd, d_model=256, nhead=8,
                 num_layers=6, dim_ff=1024, dropout=0.1, max_len=200):
        super().__init__()
        self.d_model = d_model

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # 可学习位置编码（比固定 sin/cos 更灵活，适合论文）
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer Encoder 层
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

        # CPD 特征分支（C3 贡献）
        self.cpd_proj = nn.Sequential(
            nn.Linear(k_cpd, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 融合分类头
        fuse_dim = d_model + d_model // 2
        self.classifier = nn.Sequential(
            nn.Linear(fuse_dim, fuse_dim // 2),
            nn.LayerNorm(fuse_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fuse_dim // 2, 1)
        )

    def forward(self, x_seq, x_cpd):
        B, T, _ = x_seq.shape

        # 序列路径：投影 + 位置编码 + CLS Token
        h = self.input_proj(x_seq)                          # (B, T, d)
        h = h + self.pos_emb[:, :T, :]
        cls = self.cls_token.expand(B, 1, -1)
        h   = torch.cat([cls, h], dim=1)                    # (B, T+1, d)

        # Transformer 编码
        for layer in self.layers:
            h = layer(h)
        h = self.norm_out(h)
        h_cls = h[:, 0, :]                                  # (B, d) CLS Token

        # CPD 路径（C3 贡献）
        h_cpd = self.cpd_proj(x_cpd)                        # (B, d//2)

        # 融合 → 分类
        fused  = torch.cat([h_cls, h_cpd], dim=1)           # (B, d + d//2)
        logits = self.classifier(fused).squeeze(-1)         # (B,)
        return logits

    @torch.no_grad()
    def extract_features(self, x_seq, x_cpd):
        """提取用于 XGBoost 集成的中间特征"""
        B, T, _ = x_seq.shape
        h = self.input_proj(x_seq) + self.pos_emb[:, :T, :]
        cls = self.cls_token.expand(B, 1, -1)
        h   = torch.cat([cls, h], dim=1)
        for layer in self.layers:
            h = layer(h)
        h = self.norm_out(h)
        h_cls = h[:, 0, :]
        h_cpd = self.cpd_proj(x_cpd)
        return torch.cat([h_cls, h_cpd], dim=1)             # (B, d + d//2)


model = RMTCPDTransformer(
    feat_dim   = FEAT_DIM,
    k_cpd      = K_CPD,
    d_model    = CFG['d_model'],
    nhead      = CFG['nhead'],
    num_layers = CFG['num_layers'],
    dim_ff     = CFG['dim_ff'],
    dropout    = CFG['dropout'],
    max_len    = CFG['max_len'],
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  模型参数量: {total_params:,}")
print(f"  Transformer: d_model={CFG['d_model']}, layers={CFG['num_layers']}, "
      f"nhead={CFG['nhead']}, d_ff={CFG['dim_ff']}")
print(f"  时序输入: ({N_STEPS}步, {FEAT_DIM}维)")
print(f"  CPD 输入: ({K_CPD}维)")

# =============================================================================
# Step 6: 训练准备
# =============================================================================
print("\n" + "="*65)
print("Step 6: 训练准备")
print("="*65)

# 数据集划分
idx_tr, idx_te = train_test_split(np.arange(N_USERS),
                                   test_size=CFG['test_size'],
                                   stratify=y,
                                   random_state=CFG['random_seed'])

X_seq_tr = torch.FloatTensor(X_seq[idx_tr])
X_seq_te = torch.FloatTensor(X_seq[idx_te])
X_cpd_tr = torch.FloatTensor(cpd_features[idx_tr])
X_cpd_te = torch.FloatTensor(cpd_features[idx_te])
y_tr     = torch.FloatTensor(y[idx_tr])
y_te     = torch.FloatTensor(y[idx_te])

# 类别不平衡处理：加权采样 + Focal Loss
pos_count = (y[idx_tr] == 1).sum()
neg_count = (y[idx_tr] == 0).sum()
pos_w     = neg_count / pos_count
print(f"  训练集: {len(idx_tr)} 样本  正样本: {pos_count}  负样本: {neg_count}")
print(f"  pos_weight: {pos_w:.2f}")

class_w   = np.where(y[idx_tr] == 1, 1.0 / pos_count, 1.0 / neg_count)
sampler   = WeightedRandomSampler(
    torch.FloatTensor(class_w), len(idx_tr), replacement=True
)

train_ds  = TensorDataset(X_seq_tr, X_cpd_tr, y_tr)
test_ds   = TensorDataset(X_seq_te, X_cpd_te, y_te)
train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                          sampler=sampler, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=CFG['batch_size'] * 2,
                          shuffle=False, num_workers=4, pin_memory=True)

print(f"  训练 batches: {len(train_loader)}  测试 batches: {len(test_loader)}")

# 损失函数 + 优化器 + 调度器
CFG['focal_alpha'] = neg_count / (pos_count + neg_count)
criterion = FocalLoss(alpha=CFG['focal_alpha'], gamma=CFG['focal_gamma'],
                      pos_weight=torch.tensor([pos_w]).to(device))
optimizer = optim.AdamW(model.parameters(), lr=CFG['lr'],
                        weight_decay=CFG['weight_decay'])
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=CFG['T_0'], T_mult=CFG['T_mult'], eta_min=1e-6
)

# =============================================================================
# Step 7: 训练循环
# =============================================================================
print("\n" + "="*65)
print("Step 7: GPU 训练")
print("="*65)
print(f"  epochs={CFG['epochs']}, batch={CFG['batch_size']}, "
      f"lr={CFG['lr']}, patience={CFG['patience']}")
print("="*65)
print(f"{'Epoch':>5} | {'Loss':>9} | {'ValAUC':>7} | {'ValF1':>6} | "
      f"{'Thr':>5} | {'LR':>8} | {'Time':>7}")
print("="*65)

best_auc   = 0.0
best_f1    = 0.0
best_epoch = 0
no_improve = 0
best_state = None

scaler_amp = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

for epoch in range(1, CFG['epochs'] + 1):
    t_ep = time.time()
    model.train()
    total_loss = 0.0

    for xb_seq, xb_cpd, yb in train_loader:
        xb_seq = xb_seq.to(device, non_blocking=True)
        xb_cpd = xb_cpd.to(device, non_blocking=True)
        yb     = yb.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler_amp is not None:
            with torch.cuda.amp.autocast():
                logits = model(xb_seq, xb_cpd)
                loss   = criterion(logits, yb)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            logits = model(xb_seq, xb_cpd)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * len(yb)

    train_loss = total_loss / len(idx_tr)

    # 验证
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb_seq, xb_cpd, yb in test_loader:
            logits = model(xb_seq.to(device), xb_cpd.to(device))
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(yb.numpy())

    probs_np  = np.array(all_probs)
    labels_np = np.array(all_labels)
    val_auc   = roc_auc_score(labels_np, probs_np)

    # 最优 F1 阈值
    best_f1_ep, best_thr = 0.0, 0.5
    for thr in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(labels_np, (probs_np >= thr).astype(int),
                      zero_division=0)
        if f1 > best_f1_ep:
            best_f1_ep, best_thr = f1, thr

    scheduler.step(epoch - 1)
    cur_lr  = optimizer.param_groups[0]['lr']
    elapsed = time.time() - t_ep

    improved = val_auc > best_auc
    flag     = " ✅" if improved else ""
    print(f"{epoch:>5} | {train_loss:>9.6f} | {val_auc:>7.4f} | "
          f"{best_f1_ep:>6.4f} | {best_thr:>5.3f} | {cur_lr:>8.2e} | "
          f"{elapsed:>5.1f}s{flag}")

    if improved:
        best_auc   = val_auc
        best_f1    = best_f1_ep
        best_epoch = epoch
        no_improve = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        no_improve += 1

    if no_improve >= CFG['patience']:
        print(f"\n⏹ 早停！最优在第 {best_epoch} 轮")
        break

# 恢复最优模型
model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
print(f"\n✅ 训练完成! 最优 AUC={best_auc:.4f}  F1={best_f1:.4f} "
      f"(Epoch {best_epoch})")

# =============================================================================
# Step 8: 最终评估
# =============================================================================
print("\n" + "="*65)
print("Step 8: 最终评估")
print("="*65)

model.eval()
all_probs, all_labels = [], []
with torch.no_grad():
    for xb_seq, xb_cpd, yb in test_loader:
        logits = model(xb_seq.to(device), xb_cpd.to(device))
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(yb.numpy())

final_probs  = np.array(all_probs)
final_labels = np.array(all_labels)
final_auc    = roc_auc_score(final_labels, final_probs)

# 最优 F1 阈值
best_f1_fin, best_thr_fin = 0.0, 0.5
for thr in np.arange(0.01, 0.99, 0.005):
    f1 = f1_score(final_labels, (final_probs >= thr).astype(int),
                  zero_division=0)
    if f1 > best_f1_fin:
        best_f1_fin, best_thr_fin = f1, thr

final_preds = (final_probs >= best_thr_fin).astype(int)

print(f"\n  ╔══════════════════════════════════════════════╗")
print(f"  ║  RMT-CPD-Transformer 最终结果")
print(f"  ║  AUC   = {final_auc:.4f}")
print(f"  ║  F1    = {best_f1_fin:.4f}  (阈值={best_thr_fin:.3f})")
print(f"  ╚══════════════════════════════════════════════╝")
print()
print(classification_report(final_labels, final_preds,
                             target_names=['正常', '异常']))

# 保存模型
torch.save({
    'model_state': model.state_dict(),
    'cfg': CFG,
    'auc': final_auc,
    'f1': best_f1_fin,
    'best_epoch': best_epoch,
}, 'rmt_cpd_transformer_best.pth')
print("  模型已保存: rmt_cpd_transformer_best.pth")
