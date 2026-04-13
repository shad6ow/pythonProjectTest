# -*- coding: utf-8 -*-
"""
回退 patch_auc_boost.py 注入的 CNN 相关代码，保留：
  - SupConLoss 类（如果你想保留对比损失可手动控制）
  - CosineAnnealingWarmRestarts 调度器
  - EPOCHS/PATIENCE/GAMMA 等超参改动
移除：
  - MultiScaleCNN1D 类及注释块
  - cnn_encoder 初始化、optimizer_cnn、scheduler_cnn
  - supcon_loss、SUPCON_LAMBDA
  - X_raw_ds、train_ds_cnn、test_ds_cnn、train_loader_cnn、test_loader_cnn
  - train_one_epoch_with_supcon 函数（替换回原始单路训练）
  - evaluate_with_cnn 函数（替换回原始评估）
  - 训练循环中 CNN 相关调用（恢复原始单路训练逻辑）
  - batch_extract_cnn_features 及 CNN 特征拼接（Step D）
"""

path = 'sgcc_analysis.py'
content = open(path, encoding='utf-8').read()
original_len = len(content)

changes_done = []

# ── 1. 移除 MultiScaleCNN1D 类及其注释块 ────────────────────────────
old1 = '''# =============================================================================
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


'''
if old1 in content:
    content = content.replace(old1, '', 1)
    changes_done.append('1. MultiScaleCNN1D 类 已移除')
else:
    changes_done.append('1. MultiScaleCNN1D 类 未找到（跳过）')

# ── 2. 移除 CNN 分支初始化块（CNN_OUT_DIM 到 cnn_params 打印）─────────
old2 = '''
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

# [创新点B] SupConLoss: 监督对比损失（与 AdaptiveFocal 联合训练）
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
if old2 in content:
    content = content.replace(old2, '', 1)
    changes_done.append('2. CNN 初始化块 已移除')
else:
    changes_done.append('2. CNN 初始化块 未找到（跳过）')

# ── 3. 移除训练循环注释头 + train_one_epoch_with_supcon 定义 ─────────
old3 = '''# [改造] train_one_epoch_dual 内部不再调用 scheduler.step()（CosineAnnealing 是 epoch 级）
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
            cnn_logit, y_b, reduction='mean'
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


'''
new3 = ''
if old3 in content:
    content = content.replace(old3, new3, 1)
    changes_done.append('3. train_one_epoch_with_supcon + evaluate_with_cnn 已移除')
else:
    changes_done.append('3. 训练函数 未找到（跳过）')

# ── 4. 修复训练循环：恢复原始 train_one_epoch_dual + evaluate_dual 调用 ──
old4 = '''for epoch in range(1, EPOCHS + 1):
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
    cur_lr = optimizer_dual.param_groups[0]['lr']'''

new4 = '''for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss = train_one_epoch_dual(
        model_dual, train_loader14, optimizer_dual,
        scheduler_dual, criterion_dual, scaler_dual, device
    )
    val_auc, val_f1, val_thr, val_probs, val_labels = evaluate_dual(
        model_dual, test_loader14, device
    )
    elapsed = time.time() - t0

    # CosineAnnealingWarmRestarts 是 epoch 级调度（step 在 train_one_epoch_dual 内部已调用）
    # 通知 AdaptiveFocalLoss 当前轮次，更新动态 recall_weight
    criterion_dual.set_epoch(epoch, EPOCHS)
    cur_lr = optimizer_dual.param_groups[0]['lr']'''

if old4 in content:
    content = content.replace(old4, new4, 1)
    changes_done.append('4. 训练循环 已恢复为 train_one_epoch_dual')
else:
    changes_done.append('4. 训练循环 未找到（跳过）')

# ── 5. 移除 scheduler_cnn.step(epoch) 这行（如果训练循环里单独还有）──
old5 = '    scheduler_cnn.step(epoch)\n'
if old5 in content:
    content = content.replace(old5, '', 1)
    changes_done.append('5. scheduler_cnn.step(epoch) 已移除')
else:
    changes_done.append('5. scheduler_cnn.step(epoch) 未找到（跳过）')

# ── 6. 移除 Step D 中的 CNN 特征提取块 ────────────────────────────────
old6 = '''
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
    print(f"  融合特征维度 (Trans+CNN): {feat_all.shape}")
'''
if old6 in content:
    content = content.replace(old6, '\n', 1)
    changes_done.append('6. Step D CNN 特征提取块 已移除')
else:
    changes_done.append('6. Step D CNN 特征提取块 未找到（跳过）')

# ── 7. 移除 [创新点 RMT-SGP] 注入 RMT-SGP 特征维度打印（patch_auc_boost 加的）─
old7 = '''
    # [创新点 RMT-SGP] 注入多尺度谱图扩散特征
    print("  注入 RMT-SGP 特征（PPR扩散 + 邻域密度 + 多尺度谱投影）...")
    print(f"  RMT-SGP 特征维度: {rmt_sgp_features.shape}")'''
if old7 in content:
    content = content.replace(old7, '', 1)
    changes_done.append('7. patch_auc_boost 遗留的 RMT-SGP 打印 已移除（patch_rmt_sgp 里已有更好的）')
else:
    changes_done.append('7. 遗留打印 未找到（跳过）')

# 写回
open(path, 'w', encoding='utf-8').write(content)

# 报告
print('\n=== CNN 回退报告 ===')
for c in changes_done:
    print(f'  {c}')
print(f'\n文件大小变化: {original_len} → {len(content)} 字节 '
      f'(减少 {original_len - len(content)} 字节)')
