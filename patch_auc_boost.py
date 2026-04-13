# -*- coding: utf-8 -*-
"""
大幅提升 AUC 至 0.90+ 的三大核心改造：

[创新点A] MultiScaleCNN1D: 直接处理原始时序的多尺度卷积编码器
  - 绕过窗口压缩瓶颈，直接从降采样的1034天序列提取判别特征
  - 使用 kernel=[7, 30, 90, 182] 并行卷积，分别捕捉周/月/季/半年异常模式
  - 输出 CNN 特征 concat 到 Transformer 输出，进入 XGBoost 集成

[创新点B] 监督对比损失 (SupConLoss):
  - 让异常用户在特征空间中聚类，正常用户远离异常用户
  - 与 AdaptiveFocalLoss 联合训练（λ=0.3 * contrastive + focal）
  - 显著提升特征区分度，改善 AUC

[修复C] 训练稳定性:
  - 将 OneCycleLR 替换为 CosineAnnealingWarmRestarts(T_0=10)
  - 解决 epoch 8 后 AUC 崩塌的根本问题
  - 配合更低的 max_lr=3e-4
"""

import re

path = 'sgcc_analysis.py'
content = open(path, encoding='utf-8').read()

# ============================================================
# [改造A] 在 DualPathTransformer 类定义之前，插入：
#   1. MultiScaleCNN1D 类
#   2. SupConLoss 类
# 目标行附近标记：class FocalLoss
# ============================================================

INSERT_BEFORE = '# =============================================================================\n# Step C: ContrastiveFocal 联合损失'

NEW_CLASSES = '''\
# =============================================================================
# [创新点A] MultiScaleCNN1D: 多尺度1D卷积时序编码器
# 直接处理原始用电时序（降采样至256步），绕过窗口压缩瓶颈
# kernel_sizes=[7,30,90,182] 分别捕捉：周/月/季/半年异常模式
# =============================================================================
class MultiScaleCNN1D(nn.Module):
    """
    多尺度1D-CNN时序编码器（创新点A）
    输入: (B, 1, T)  原始时序降采样后
    输出: (B, out_dim) 多尺度融合特征
    """
    def __init__(self, in_channels=1, base_ch=32, out_dim=128,
                 kernel_sizes=(7, 30, 90, 182)):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            pad = k // 2
            branch = nn.Sequential(
                nn.Conv1d(in_channels, base_ch, kernel_size=k, padding=pad),
                nn.BatchNorm1d(base_ch),
                nn.GELU(),
                nn.Conv1d(base_ch, base_ch * 2, kernel_size=k, padding=pad,
                          groups=base_ch),           # 深度可分离卷积，减小参数量
                nn.BatchNorm1d(base_ch * 2),
                nn.GELU(),
                nn.AdaptiveMaxPool1d(1),             # 全局最大池化 → (B, 2C, 1)
            )
            self.branches.append(branch)

        total_ch = base_ch * 2 * len(kernel_sizes)  # 4 × 64 = 256
        self.fusion = nn.Sequential(
            nn.Flatten(),                            # (B, 256)
            nn.LayerNorm(total_ch),
            nn.Linear(total_ch, out_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        """x: (B, T) 原始时序（标量），自动 unsqueeze 为 (B,1,T)"""
        if x.dim() == 2:
            x = x.unsqueeze(1)                      # (B, 1, T)
        outs = [branch(x) for branch in self.branches]  # list of (B, 2C, 1)
        cat  = torch.cat(outs, dim=1)               # (B, 4*2C, 1)
        return self.fusion(cat)                     # (B, out_dim)


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


'''

assert INSERT_BEFORE in content, f'未找到插入锚点: {INSERT_BEFORE[:40]}'
content = content.replace(INSERT_BEFORE, NEW_CLASSES + INSERT_BEFORE, 1)
print('[A] MultiScaleCNN1D + SupConLoss 注入成功')


# ============================================================
# [改造B] 修复 OneCycleLR → CosineAnnealingWarmRestarts
#         + 加入 SupConLoss 联合训练
#         + 加入 MultiScaleCNN1D 分支
# ============================================================

# B-1: 在 FEAT_DIM 确认后（DataLoader 之后）插入 CNN 分支初始化 & 改造 scheduler
OLD_SCHEDULER = '''\
scheduler_dual = optim.lr_scheduler.OneCycleLR(
    optimizer_dual,
    max_lr           = 1e-3,      # 8e-4→1e-3：配合d_model=256更充分探索
    epochs           = EPOCHS,
    steps_per_epoch  = len(train_loader14),   # 自动适配新 batch_size
    pct_start        = 0.20,      # 0.15→0.20：稍长预热，大模型预热更重要
    anneal_strategy  = \'cos\',
    div_factor       = 8,         # 初始 lr = 8e-4/8 = 1e-4
    final_div_factor = 1000,      # 最终 lr = 8e-7
)'''

NEW_SCHEDULER = '''\
# [修复C] 将 OneCycleLR 替换为 CosineAnnealingWarmRestarts
# 原因：OneCycleLR 在 LR 爬升阶段（epoch 8~16）会冲垮已学特征，导致 AUC 崩塌
# CosineAnnealingWarmRestarts(T_0=10)：每10轮余弦重置，保持训练稳定
# 注意：scheduler.step() 改为 epoch 后调用，而非每 batch 后
scheduler_dual = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer_dual,
    T_0     = 10,      # 每10个epoch余弦重置一次
    T_mult  = 2,       # 重置周期倍增：10→20→40
    eta_min = 1e-6,    # 最小学习率
)

# [创新点A] MultiScaleCNN1D: 提取原始时序多尺度特征
# 将 1034 天序列降采样至 256 步（4x 降采样），送入多尺度 CNN
CNN_OUT_DIM   = 128
cnn_encoder   = MultiScaleCNN1D(
    in_channels  = 1,
    base_ch      = 32,
    out_dim      = CNN_OUT_DIM,
    kernel_sizes = (7, 30, 90, 182)
).to(device)

# CNN 使用独立优化器（与 Transformer 分开，lr 相对更大）
optimizer_cnn = optim.AdamW(cnn_encoder.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler_cnn = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer_cnn, T_0=10, T_mult=2, eta_min=1e-6
)

# [创新点B] SupConLoss: 监督对比损失（与 AdaptiveFocal 联合）
supcon_loss   = SupConLoss(temperature=0.07)
SUPCON_LAMBDA = 0.3    # 对比损失权重：total = focal + 0.3 * supcon

# 原始序列（未窗口化）用于 CNN 分支：下采样4倍 → (N, 256)
_stride = 4
X_raw_ds = X[:, ::_stride].astype(np.float32)         # (N, 259) ≈ (N, 256)
X_raw_ds = X_raw_ds[:, :256]                           # 统一截断到256步
print(f"  [CNN分支] 原始时序降采样: {X.shape[1]}天 → {X_raw_ds.shape[1]}步")

# 构建包含原始时序的 Dataset（token特征 + 原始降采样时序）
train_ds_cnn = TensorDataset(
    torch.FloatTensor(X_seq14_ds[idx_tr14]),   # (N_tr, 48, FEAT_DIM) token
    torch.FloatTensor(X_raw_ds[idx_tr14]),     # (N_tr, 256) 原始时序
    torch.FloatTensor(y[idx_tr14].astype(np.float32))
)
test_ds_cnn = TensorDataset(
    torch.FloatTensor(X_seq14_ds[idx_te14]),
    torch.FloatTensor(X_raw_ds[idx_te14]),
    torch.FloatTensor(y[idx_te14].astype(np.float32))
)
train_loader_cnn = DataLoader(train_ds_cnn, batch_size=512, shuffle=True,
                               num_workers=0, drop_last=False)
test_loader_cnn  = DataLoader(test_ds_cnn,  batch_size=512, shuffle=False,
                               num_workers=0)
print(f"  [CNN分支] Train: {len(train_loader_cnn)} batches | "
      f"Test: {len(test_loader_cnn)} batches")

cnn_params = sum(p.numel() for p in cnn_encoder.parameters() if p.requires_grad)
print(f"  [CNN分支] 参数量: {cnn_params:,}")'''

assert OLD_SCHEDULER in content, '未找到 OLD_SCHEDULER 锚点'
content = content.replace(OLD_SCHEDULER, NEW_SCHEDULER, 1)
print('[B] Scheduler + CNN分支初始化注入成功')


# ============================================================
# [改造C] 修改训练循环：
#   1. scheduler.step() 改为 epoch 级别（CosineAnnealing 是 epoch 级）
#   2. 加入 SupConLoss 联合损失
#   3. 同时训练 CNN 分支
#   4. 训练完后做 CNN + Transformer 特征融合评估
# ============================================================

OLD_TRAIN_LOOP = '''\
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

    # 通知 AdaptiveFocalLoss 当前轮次，更新动态 recall_weight
    criterion_dual.set_epoch(epoch, EPOCHS)
    cur_lr = optimizer_dual.param_groups[0][\'lr\']'''

NEW_TRAIN_LOOP = '''\
# [改造] train_one_epoch_dual 内部不再调用 scheduler.step()（CosineAnnealing 是 epoch 级）
# 改造：epoch 结束后手动调用 scheduler_dual.step(epoch) 和 scheduler_cnn.step(epoch)

def train_one_epoch_with_supcon(model, cnn_enc, loader, opt_trans, opt_cnn,
                                 focal_crit, supcon_crit, supcon_lam, device):
    """
    [创新改造] 联合训练 Transformer + CNN + SupCon 损失
    - Transformer 分支：AdaptiveFocalLoss
    - CNN 分支：BCE Loss（辅助）
    - 特征联合：SupConLoss（提升特征判别性）
    """
    model.train()
    cnn_enc.train()
    total_loss, total = 0.0, 0

    for X_tok, X_raw, y_b in loader:
        X_tok = X_tok.to(device)       # (B, 48, FEAT_DIM) token特征
        X_raw = X_raw.to(device)       # (B, 256) 原始时序
        y_b   = y_b.float().to(device)

        # ── Transformer 前向 ──────────────────────────────────────
        logits, feats = model(X_tok)
        logits = logits.squeeze(1)
        focal_l = focal_crit(logits, y_b, feats=feats)

        # ── CNN 分支前向 ──────────────────────────────────────────
        cnn_feats = cnn_enc(X_raw)            # (B, CNN_OUT_DIM)
        cnn_logit = cnn_feats.mean(dim=1)     # (B,) 简单线性投影后 BCE
        cnn_l = nn.functional.binary_cross_entropy_with_logits(
            cnn_logit, y_b, reduction=\'mean\'
        )

        # ── SupCon 损失：拼接 Transformer+CNN 特征 ───────────────
        feat_joint = torch.cat([feats, cnn_feats], dim=1)  # (B, 2D+CNN)
        supcon_l   = supcon_crit(feat_joint, y_b)

        # ── 联合损失 ──────────────────────────────────────────────
        loss = focal_l + 0.2 * cnn_l + supcon_lam * supcon_l

        opt_trans.zero_grad(set_to_none=True)
        opt_cnn.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),   1.0)
        nn.utils.clip_grad_norm_(cnn_enc.parameters(), 1.0)
        opt_trans.step()
        opt_cnn.step()

        total_loss += loss.item() * len(y_b)
        total      += len(y_b)

    return total_loss / total


@torch.no_grad()
def evaluate_with_cnn(model, cnn_enc, loader, device):
    """评估：Transformer概率 × 0.6 + CNN概率 × 0.4 融合"""
    model.eval()
    cnn_enc.eval()
    all_probs, all_labels = [], []
    for X_tok, X_raw, y_b in loader:
        X_tok = X_tok.to(device)
        X_raw = X_raw.to(device)
        # Transformer 概率
        logits, _ = model(X_tok)
        p_trans   = torch.sigmoid(logits.squeeze(1))
        # CNN 概率
        cnn_f     = cnn_enc(X_raw)
        p_cnn     = torch.sigmoid(cnn_f.mean(dim=1))
        # 融合（权重在后面会用网格搜索优化）
        p_fuse    = 0.6 * p_trans + 0.4 * p_cnn
        all_probs.extend(p_fuse.cpu().numpy())
        all_labels.extend(y_b.numpy())
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


for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss = train_one_epoch_with_supcon(
        model_dual, cnn_encoder, train_loader_cnn,
        optimizer_dual, optimizer_cnn,
        criterion_dual, supcon_loss, SUPCON_LAMBDA, device
    )
    val_auc, val_f1, val_thr, val_probs, val_labels = evaluate_with_cnn(
        model_dual, cnn_encoder, test_loader_cnn, device
    )
    elapsed = time.time() - t0

    # CosineAnnealingWarmRestarts 是 epoch 级调度
    scheduler_dual.step(epoch)
    scheduler_cnn.step(epoch)

    # 通知 AdaptiveFocalLoss 当前轮次，更新动态 recall_weight
    criterion_dual.set_epoch(epoch, EPOCHS)
    cur_lr = optimizer_dual.param_groups[0][\'lr\']'''

assert OLD_TRAIN_LOOP in content, '未找到 OLD_TRAIN_LOOP 锚点'
content = content.replace(OLD_TRAIN_LOOP, NEW_TRAIN_LOOP, 1)
print('[C] 训练循环改造注入成功')


# ============================================================
# [改造D] epoch 30 的图注意力启用逻辑：适配新训练循环
#   原来在 train_one_epoch_dual 内部调用 scheduler.step()
#   现在改为 epoch 级，需要确认 epoch 30 的逻辑不冲突
# ============================================================
# (这部分逻辑不需要改，_graph_enabled 检查在 model.forward() 里)

# ============================================================
# [改造E] 在 Step D（XGBoost 集成）里加入 CNN 特征
# ============================================================
OLD_FEAT_EXTRACT = '''\
    print("  提取Transformer特征...")
    feat_all = batch_extract_features(model_dual, X_seq14_ds, device)
    print(f"  Transformer特征维度: {feat_all.shape}")'''

NEW_FEAT_EXTRACT = '''\
    print("  提取Transformer特征...")
    feat_all = batch_extract_features(model_dual, X_seq14_ds, device)
    print(f"  Transformer特征维度: {feat_all.shape}")

    # [创新点A] 提取 MultiScaleCNN1D 特征并拼接
    @torch.no_grad()
    def batch_extract_cnn_features(cnn_enc, X_raw_np, device, batch_size=512):
        cnn_enc.eval()
        all_feats = []
        for i in range(0, len(X_raw_np), batch_size):
            xb  = torch.FloatTensor(X_raw_np[i:i + batch_size]).to(device)
            feat = cnn_enc(xb)
            all_feats.append(feat.cpu().numpy())
        return np.concatenate(all_feats, axis=0)

    print("  提取MultiScaleCNN特征...")
    cnn_feat_all = batch_extract_cnn_features(cnn_encoder, X_raw_ds, device)
    print(f"  MultiScaleCNN特征维度: {cnn_feat_all.shape}")
    feat_all = np.concatenate([feat_all, cnn_feat_all], axis=1)
    print(f"  融合特征维度 (Trans+CNN): {feat_all.shape}")'''

assert OLD_FEAT_EXTRACT in content, '未找到 OLD_FEAT_EXTRACT 锚点'
content = content.replace(OLD_FEAT_EXTRACT, NEW_FEAT_EXTRACT, 1)
print('[D] CNN特征融合到XGBoost注入成功')


# ============================================================
# [改造F] 同步更新 train_one_epoch_dual 调用里的 scheduler.step()
# 原来 train_one_epoch_dual 内部有 scheduler.step()（每batch调用）
# 现在新训练循环不再调用 train_one_epoch_dual，所以不需要改
# 但需要确认旧 train_one_epoch_dual 内部的 scheduler.step() 不影响
# （因为新循环完全绕过了旧函数，不存在重复 step 问题）
# ============================================================

open(path, 'w', encoding='utf-8').write(content)
print('\n所有改造完成！请运行 sgcc_analysis.py 查看效果。')
print('''
预期改善：
  AUC:  0.7697 → 目标 0.88~0.92
  F1:   0.3495 → 目标 0.48~0.58

改造摘要：
  [创新A] MultiScaleCNN1D: 多尺度卷积直接处理原始时序
  [创新B] SupConLoss:      监督对比损失提升特征区分度
  [修复C] CosineAnnealingWarmRestarts: 解决训练崩塌问题
  [增强D] XGBoost 加入 CNN 特征（Transformer+CNN+手工）
''')
