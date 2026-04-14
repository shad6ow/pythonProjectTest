# =============================================================================
# patch_feature_select.py
# 自动特征筛选：剔除 AUC < AUC_THRESHOLD 的低判别力 Token
#
# 使用方式：在 patch_feature_boost.py 执行之后、DataLoader 构建之前 exec 此文件
# 依赖变量：X_seq14_ds, y, FEAT_DIM, N_W_ds
# =============================================================================

import numpy as np
from sklearn.metrics import roc_auc_score

# ── 筛选阈值（可调整） ─────────────────────────────────────────────────────
# 0.51：只删纯噪声；0.53：激进删弱特征（推荐）
AUC_THRESHOLD = 0.53

print("\n" + "=" * 65)
print(f"patch_feature_select: 自动特征筛选 (阈值 AUC ≥ {AUC_THRESHOLD})")
print("=" * 65)

N_USERS_fs, N_W_fs, FEAT_DIM_fs = X_seq14_ds.shape

def _auc_safe(feat, labels):
    try:
        a = roc_auc_score(labels, feat)
        return max(a, 1 - a)
    except Exception:
        return 0.5

# ── 逐维度计算 AUC ─────────────────────────────────────────────────────────
auc_per_dim = np.zeros(FEAT_DIM_fs, dtype=np.float32)
for dim_i in range(FEAT_DIM_fs):
    feat_mean = X_seq14_ds[:, :, dim_i].mean(axis=1)   # (N_USERS,) 时间轴均值
    auc_per_dim[dim_i] = _auc_safe(feat_mean, y)

# ── 打印每维度 AUC ─────────────────────────────────────────────────────────
print(f"\n  {'维度':>4}  {'AUC':>7}  {'状态'}")
print(f"  {'-'*30}")
keep_mask = np.zeros(FEAT_DIM_fs, dtype=bool)
removed_dims, kept_dims = [], []

for i, auc_val in enumerate(auc_per_dim):
    if auc_val >= AUC_THRESHOLD:
        keep_mask[i] = True
        kept_dims.append(i)
        status = f"✅ 保留 (AUC={auc_val:.4f})"
    else:
        removed_dims.append(i)
        status = f"❌ 删除 (AUC={auc_val:.4f})"
    print(f"  {i:>4}  {auc_val:>7.4f}  {status}")

print(f"\n  原始特征数: {FEAT_DIM_fs}")
print(f"  删除特征数: {len(removed_dims)}  → 维度 {removed_dims}")
print(f"  保留特征数: {len(kept_dims)}")

# ── 执行筛选 ─────────────────────────────────────────────────────────────
X_seq14_ds = X_seq14_ds[:, :, keep_mask].astype(np.float32)
FEAT_DIM   = X_seq14_ds.shape[-1]

print(f"\n  筛选后 X_seq14_ds 形状: {X_seq14_ds.shape}  (FEAT_DIM={FEAT_DIM})")
print(f"  特征压缩率: {len(removed_dims)/FEAT_DIM_fs*100:.1f}% 已剔除")
print("=" * 65)
