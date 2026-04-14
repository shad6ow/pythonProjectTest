# =============================================================================
# patch_quick_auc.py
# 快速 AUC 提升补丁（在 patch_feature_select 之后、DataLoader 构建之前执行）
#
# 包含两个模块：
#   Module 1 - 月度趋势特征（最强信号，AUC 目标 ≥ 0.70）
#   Module 2 - LightGBM 快速基线（<1 分钟出 AUC，验证特征天花板）
# =============================================================================

import numpy as np
import time
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

print("\n" + "=" * 65)
print("patch_quick_auc: 月度趋势特征 + LightGBM 快速基线")
print("=" * 65)

_t0 = time.time()
N_USERS_qa, T_DAYS_qa = X.shape   # X: (N, 1034) 原始预处理序列

def _auc(feat, labels):
    try:
        a = roc_auc_score(labels, feat)
        return max(a, 1 - a)
    except Exception:
        return 0.5

def _robust_tile(arr_1d, n_w, clip=5.0):
    arr = arr_1d.astype(np.float32).reshape(-1, 1)
    arr = RobustScaler().fit_transform(arr).flatten()
    arr = np.clip(arr, -clip, clip)
    return np.tile(arr[:, np.newaxis], (1, n_w)).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# Module 1: 月度消费趋势特征
# 理论：窃电用户在某个时间点后开始系统性减少用电量
#   - 月度消费均值的趋势斜率（最强信号）
#   - 月度消费变异系数（窃电用户CV波动更大）
#   - 月度消费最大跌幅（突变检测）
#   - 前期/后期用电比（直接捕捉下降趋势）
# 文献：Nagi et al. 2010; Zheng et al. 2018 (IEEE TSG)
# ─────────────────────────────────────────────────────────────────────────────
print("\n  [Module 1] 月度消费趋势特征...")

# 计算每用户每月平均消费（约30天一个月）
DAYS_PER_MONTH = 30
N_MONTHS = T_DAYS_qa // DAYS_PER_MONTH     # ≈34个月
X_monthly = np.zeros((N_USERS_qa, N_MONTHS), dtype=np.float32)
for m in range(N_MONTHS):
    start = m * DAYS_PER_MONTH
    end   = min(start + DAYS_PER_MONTH, T_DAYS_qa)
    X_monthly[:, m] = X[:, start:end].mean(axis=1)

print(f"    月度消费矩阵形状: {X_monthly.shape}  ({N_MONTHS} 个月)")

# ── M1: 月度趋势斜率（OLS，向量化）────────────────────────────────────────
t_vals  = np.arange(N_MONTHS, dtype=np.float32)
t_mean  = t_vals.mean()
t_c     = t_vals - t_mean                                      # (N_months,)
y_c_mo  = X_monthly - X_monthly.mean(axis=1, keepdims=True)   # (N, N_months)
slope   = (y_c_mo * t_c[np.newaxis, :]).sum(axis=1) / ((t_c ** 2).sum() + 1e-10)
tok_m1  = slope.astype(np.float32)
auc_m1  = _auc(tok_m1, y)
print(f"    M1(月度趋势斜率)        AUC={auc_m1:.4f}  {'✅' if auc_m1>=0.60 else '⚠️'}")

# ── M2: 后期/前期用电比（直接量化下降幅度）────────────────────────────────
half_m      = N_MONTHS // 2
early_mean  = X_monthly[:, :half_m].mean(axis=1) + 1e-8
late_mean   = X_monthly[:, half_m:].mean(axis=1)
tok_m2      = (late_mean / early_mean).astype(np.float32)
auc_m2      = _auc(tok_m2, y)
print(f"    M2(后期/前期用电比)     AUC={auc_m2:.4f}  {'✅' if auc_m2>=0.60 else '⚠️'}")

# ── M3: 月度消费变异系数（CV = std/mean）──────────────────────────────────
mo_mean = X_monthly.mean(axis=1) + 1e-8
mo_std  = X_monthly.std(axis=1)
tok_m3  = (mo_std / mo_mean).astype(np.float32)
auc_m3  = _auc(tok_m3, y)
print(f"    M3(月度消费CV)          AUC={auc_m3:.4f}  {'✅' if auc_m3>=0.58 else '⚠️'}")

# ── M4: 月度消费最大单月跌幅（异常突变检测）──────────────────────────────
mo_diff   = np.diff(X_monthly, axis=1)                 # (N, N_months-1)
tok_m4    = mo_diff.min(axis=1).astype(np.float32)     # 最大跌幅（负值越大越异常）
auc_m4    = _auc(tok_m4, y)
print(f"    M4(最大单月跌幅)        AUC={auc_m4:.4f}  {'✅' if auc_m4>=0.58 else '⚠️'}")

# ── M5: 月度消费趋势加速度（斜率的斜率，捕捉加速下降）─────────────────────
# 对每用户的月度消费做二阶差分均值
mo_diff2  = np.diff(mo_diff, axis=1)                   # (N, N_months-2)
tok_m5    = mo_diff2.mean(axis=1).astype(np.float32)
auc_m5    = _auc(tok_m5, y)
print(f"    M5(月度趋势加速度)      AUC={auc_m5:.4f}  {'✅' if auc_m5>=0.55 else '⚠️'}")

# ── M6: 最低月消费 / 最高月消费（极值比，检测用电量骤降）─────────────────
mo_min    = X_monthly.min(axis=1) + 1e-8
mo_max    = X_monthly.max(axis=1) + 1e-8
tok_m6    = (mo_min / mo_max).astype(np.float32)
auc_m6    = _auc(tok_m6, y)
print(f"    M6(月度极值比min/max)   AUC={auc_m6:.4f}  {'✅' if auc_m6>=0.55 else '⚠️'}")

# ── M7: 月度趋势折点检测（最大下降段起始位置 / 总月数）──────────────────
# 思路：找连续3个月内最大累积跌幅的起始月份
WINDOW3 = 3
best_drop    = np.full(N_USERS_qa, 0.0, dtype=np.float32)
best_pos_rel = np.full(N_USERS_qa, 0.5, dtype=np.float32)  # 默认中间
for s in range(N_MONTHS - WINDOW3):
    drop = X_monthly[:, s + WINDOW3] - X_monthly[:, s]   # (N,) 3个月累积变化
    is_lower = drop < best_drop
    best_drop[is_lower]    = drop[is_lower]
    best_pos_rel[is_lower] = s / N_MONTHS                 # 归一化位置
tok_m7  = best_pos_rel.astype(np.float32)
auc_m7  = _auc(tok_m7, y)
print(f"    M7(最大跌幅月份位置)    AUC={auc_m7:.4f}  {'✅' if auc_m7>=0.55 else '⚠️'}")

# ── M8: 季节性一致性（夏季/冬季用电比的年际变化标准差）──────────────────
# 正常用户夏冬比相对稳定；窃电用户某个时段后比值系统性下降
months_per_year = 12
n_full_years    = N_MONTHS // months_per_year   # ≈2年
if n_full_years >= 2:
    summer_idx = [m for y_i in range(n_full_years)
                  for m in range(y_i * months_per_year + 5,
                                 y_i * months_per_year + 8)
                  if m < N_MONTHS]   # 6~8月
    winter_idx = [m for y_i in range(n_full_years)
                  for m in range(y_i * months_per_year + 11,
                                 y_i * months_per_year + 13)
                  if m < N_MONTHS]   # 12~1月

    if len(summer_idx) > 0 and len(winter_idx) > 0:
        summer_mean = X_monthly[:, summer_idx].mean(axis=1)
        winter_mean = X_monthly[:, winter_idx].mean(axis=1) + 1e-8
        tok_m8 = (summer_mean / winter_mean).astype(np.float32)
    else:
        tok_m8 = np.zeros(N_USERS_qa, dtype=np.float32)
else:
    tok_m8 = np.zeros(N_USERS_qa, dtype=np.float32)
auc_m8  = _auc(tok_m8, y)
print(f"    M8(夏冬用电比)          AUC={auc_m8:.4f}  {'✅' if auc_m8>=0.55 else '⚠️'}")

# ── 月度特征汇总 & 拼入 X_seq14_ds ─────────────────────────────────────
monthly_feats = {
    'M1(月度趋势斜率)':     tok_m1,
    'M2(后期/前期用电比)':  tok_m2,
    'M3(月度CV)':           tok_m3,
    'M4(最大单月跌幅)':     tok_m4,
    'M5(月度趋势加速度)':   tok_m5,
    'M6(月度极值比)':       tok_m6,
    'M7(最大跌幅月份)':     tok_m7,
    'M8(夏冬用电比)':       tok_m8,
}

print(f"\n  月度特征 AUC 汇总:")
valid_mo = 0
N_W_qa   = X_seq14_ds.shape[1]
for name, feat in monthly_feats.items():
    auc_val = _auc(feat, y)
    mark    = '✅' if auc_val >= 0.60 else ('⚠️' if auc_val >= 0.55 else '❌')
    if auc_val >= 0.55:
        valid_mo += 1
    print(f"    {name:<24} AUC={auc_val:.4f}  {mark}")
    if auc_val >= 0.53:   # 只拼入有判别力的
        tiled = _robust_tile(feat, N_W_qa)
        X_seq14_ds = np.concatenate(
            [X_seq14_ds, tiled[:, :, np.newaxis]], axis=-1
        ).astype(np.float32)

FEAT_DIM = X_seq14_ds.shape[-1]
print(f"\n  拼入月度特征后形状: {X_seq14_ds.shape}  (FEAT_DIM={FEAT_DIM})")
print(f"  有效月度特征(AUC≥0.55): {valid_mo}/8")

# ─────────────────────────────────────────────────────────────────────────────
# Module 2: LightGBM 快速基线
# 用所有工程特征（时间轴均值展平）训练 LightGBM，快速估算 AUC 天花板
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n  [Module 2] LightGBM 快速基线...")

try:
    import lightgbm as lgb
    _lgb_ok = True
except ImportError:
    try:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lightgbm', '-q'])
        import lightgbm as lgb
        _lgb_ok = True
    except Exception:
        _lgb_ok = False
        print("    ⚠️ lightgbm 安装失败，跳过 LightGBM 基线")

if _lgb_ok:
    # 构造扁平特征矩阵：(N, FEAT_DIM) 取时间轴均值
    X_flat = X_seq14_ds.mean(axis=1)   # (N, FEAT_DIM)

    # 额外追加月度序列本身作为特征（月度特征直接量化趋势）
    X_lgb  = np.concatenate([X_flat, X_monthly], axis=1).astype(np.float32)

    X_tr_lgb, X_te_lgb, y_tr_lgb, y_te_lgb = train_test_split(
        X_lgb, y, test_size=0.2, random_state=42, stratify=y
    )

    pos_w = (y_tr_lgb == 0).sum() / (y_tr_lgb == 1).sum()

    lgb_model = lgb.LGBMClassifier(
        n_estimators      = 800,
        learning_rate     = 0.05,
        num_leaves        = 63,
        max_depth         = -1,
        min_child_samples = 30,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        reg_alpha         = 0.1,
        reg_lambda        = 0.5,
        scale_pos_weight  = pos_w,   # 处理类别不平衡
        n_jobs            = -1,
        random_state      = 42,
        verbose           = -1,
    )

    _t_lgb = time.time()
    lgb_model.fit(
        X_tr_lgb, y_tr_lgb,
        eval_set           = [(X_te_lgb, y_te_lgb)],
        callbacks          = [lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(period=-1)],
    )
    lgb_elapsed = time.time() - _t_lgb

    lgb_probs = lgb_model.predict_proba(X_te_lgb)[:, 1]
    lgb_auc   = roc_auc_score(y_te_lgb, lgb_probs)

    # 找最优 F1 阈值
    best_f1_lgb, best_thr_lgb = 0.0, 0.5
    for thr in np.arange(0.05, 0.95, 0.005):
        preds = (lgb_probs >= thr).astype(int)
        f1    = f1_score(y_te_lgb, preds, zero_division=0)
        if f1 > best_f1_lgb:
            best_f1_lgb, best_thr_lgb = f1, thr

    print(f"\n  ╔══════════════════════════════════════════╗")
    print(f"  ║  LightGBM 快速基线结果（{lgb_elapsed:.0f}s）")
    print(f"  ║  AUC  = {lgb_auc:.4f}  {'🟢 良好' if lgb_auc>=0.85 else ('🟡 中等' if lgb_auc>=0.78 else '🔴 偏低')}")
    print(f"  ║  F1   = {best_f1_lgb:.4f}  (阈值={best_thr_lgb:.3f})")
    print(f"  ║  特征数: {X_lgb.shape[1]}  (工程特征{X_flat.shape[1]} + 月度{N_MONTHS})")
    print(f"  ╚══════════════════════════════════════════╝")

    # 保存 LightGBM 模型供后续集成使用
    lgb_model_saved = lgb_model
    lgb_te_probs    = lgb_probs
    lgb_te_idx      = np.arange(len(y))[len(y) - len(y_te_lgb):]  # 近似

    # 特征重要性 Top 10
    feat_imp    = lgb_model.feature_importances_
    top10_idx   = np.argsort(feat_imp)[-10:][::-1]
    feat_names  = ([f'ENG_{i}' for i in range(X_flat.shape[1])] +
                   [f'MO_{i}'  for i in range(N_MONTHS)])
    print(f"\n  Top 10 特征重要性:")
    for rank, fi in enumerate(top10_idx):
        print(f"    {rank+1:>2}. {feat_names[fi]:<12} importance={feat_imp[fi]}")

else:
    lgb_model_saved = None
    lgb_te_probs    = None
    print("    LightGBM 不可用，跳过基线评估")

print(f"\n  patch_quick_auc 总耗时: {time.time() - _t0:.1f}s")
print("=" * 65)
