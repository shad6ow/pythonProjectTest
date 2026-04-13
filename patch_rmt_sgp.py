# -*- coding: utf-8 -*-
"""检测锚点并注入 RMT-SGP 代码"""

path = 'sgcc_analysis.py'
content = open(path, encoding='utf-8').read()

# 检测所有锚点
anchors = {
    'A': 'print(f"  将在 Step 11 与 Transformer 统一对比")',
    'C': '    X_xgb    = np.concatenate([feat_all, hand_feats], axis=1)',
    'D_new': '        n_estimators          = 1000,  # RMT-SGP',
    'D_old': '        n_estimators          = 600,   # 适当增加树的数量',
    'E': "  print(f\"  {'RMT 谱得分（基线）':<34} {rmt_baseline_auc:>8.4f}  {'N/A':>8}\")",
}
for k, v in anchors.items():
    print(f'  [{k}] found={v[:40]!r}: {v in content}')

# ── 注入 A: Step 8b 末尾插入 Step 8c ──────────────────────────────
STEP_8C = '''

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
print("\\n" + "=" * 65)
print("Step 8c: RMT 多尺度谱图扩散（RMT-SGP）")
print("=" * 65)

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
print(f"\\n  RMT-SGP 新特征 AUC:")
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
'''

# ── 幂等检查：已注入则跳过 ───────────────────────────────────────────
GUARD = 'Step 8c: RMT'
if GUARD in content:
    print(f'[幂等] Step 8c 已存在，跳过 A 注入')
else:
    a_anchor = 'print(f"  将在 Step 11 与 Transformer 统一对比")'
    assert a_anchor in content, '锚点A未找到'
    content = content.replace(a_anchor, a_anchor + STEP_8C, 1)
    print('[A] Step 8c 注入成功')

# ── 注入 C: X_xgb 拼接行（幂等：已含 rmt_sgp_features 则跳过） ─────
old_c = '    X_xgb    = np.concatenate([feat_all, hand_feats], axis=1)'
new_c = ('    X_xgb    = np.concatenate([feat_all, hand_feats, rmt_sgp_features], axis=1)\n'
         '    print(f"  XGBoost总特征: {X_xgb.shape}  '
         '(Transformer+手工+RMT-SGP)")')
if 'rmt_sgp_features' in content and old_c not in content:
    print('[C] X_xgb 已包含 rmt_sgp_features，跳过')
elif old_c in content:
    content = content.replace(old_c, new_c, 1)
    print('[C] X_xgb 注入成功')
else:
    print('[C] 警告：两个锚点都未找到，请手动检查')

# ── 注入 D: XGBoost 超参（幂等） ─────────────────────────────────
old_d = '        n_estimators          = 600,   # 适当增加树的数量'
if old_d in content:
    new_d = '        n_estimators          = 1000,  # RMT-SGP特征增加，树数量跟进'
    content = content.replace(old_d, new_d, 1)
    print('[D] XGBoost n_estimators 注入成功')
else:
    print('[D] 已是 1000，跳过')

# ── 注入 E: 结果汇总加 PPR 行（幂等） ────────────────────────────
old_e = "  print(f\"  {'RMT 谱得分（基线）':<34} {rmt_baseline_auc:>8.4f}  {'N/A':>8}\")"
new_e = (old_e + '\n'
         '  _ppr_te = _auc_fn(y[idx_te14], ppr_score[idx_te14])\n'
         "  print(f\"  {'RMT-SGP PPR扩散（创新点）':<34} {_ppr_te:>8.4f}  {'N/A':>8}\")")
if 'RMT-SGP PPR' in content:
    print('[E] 结果汇总 PPR 行已存在，跳过')
elif old_e in content:
    content = content.replace(old_e, new_e, 1)
    print('[E] 结果汇总注入成功')
else:
    print('[E] 锚点E未找到，跳过')

open(path, 'w', encoding='utf-8').write(content)
print('\nRMT-SGP patch 完成（幂等版）')
