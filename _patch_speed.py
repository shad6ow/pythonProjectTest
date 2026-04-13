"""
一次性打补丁：4项 CPU 训练加速优化
  1. DataLoader num_workers=0（Windows spawn开销大）
  2. UserGraphAttention 延迟到 epoch 30 才启用
  3. 联邦训练预创建 local_models（避免每轮 __init__）
  4. torch.compile（PyTorch 2.0+）
"""
import sys

TARGET = 'sgcc_analysis.py'

with open(TARGET, 'r', encoding='utf-8') as f:
    src = f.read()

results = {}

# ─────────────────────────────────────────────────────────────────────────────
# 修改1: num_workers=0
# ─────────────────────────────────────────────────────────────────────────────
OLD1 = (
    "# CPU 环境：num_workers>0 可并行预取数据，persistent_workers 避免重复fork开销\n"
    "# Windows 下 num_workers 建议 2~4（过多反而因进程通信变慢）\n"
    "_N_WORKERS = min(2, os.cpu_count() or 1)   # Windows 安全上限\n"
    "\n"
    "# CPU下更大的 batch 能充分激活 MKL/OpenBLAS 的矩阵分块优化\n"
    "# 512→1024（train），1024→2048（test），充分利用 BLAS Level-3 GEMM\n"
    "train_loader14 = DataLoader(\n"
    "    TensorDataset(X_tr14, y_tr14),\n"
    "    batch_size=1024, sampler=sampler14,\n"
    "    num_workers=_N_WORKERS,\n"
    "    pin_memory=False,                  # CPU 环境无需 pin_memory\n"
    "    persistent_workers=(_N_WORKERS > 0),  # 保持 worker 存活，避免重复初始化\n"
    "    prefetch_factor=2 if _N_WORKERS > 0 else None,\n"
    ")\n"
    "test_loader14 = DataLoader(\n"
    "    TensorDataset(X_te14, y_te14),\n"
    "    batch_size=2048, shuffle=False,\n"
    "    num_workers=_N_WORKERS,\n"
    "    pin_memory=False,\n"
    "    persistent_workers=(_N_WORKERS > 0),\n"
    "    prefetch_factor=2 if _N_WORKERS > 0 else None,\n"
    ")"
)
NEW1 = (
    "# Windows CPU：num_workers=0（spawn进程开销 >> 数据预取收益）\n"
    "# TensorDataset 数据全部在内存，无 I/O 瓶颈，不需要多进程预取\n"
    "train_loader14 = DataLoader(\n"
    "    TensorDataset(X_tr14, y_tr14),\n"
    "    batch_size=1024, sampler=sampler14,\n"
    "    num_workers=0, pin_memory=False,\n"
    ")\n"
    "test_loader14 = DataLoader(\n"
    "    TensorDataset(X_te14, y_te14),\n"
    "    batch_size=2048, shuffle=False,\n"
    "    num_workers=0, pin_memory=False,\n"
    ")"
)
if OLD1 in src:
    src = src.replace(OLD1, NEW1, 1)
    results['1_dataloader'] = 'OK'
else:
    results['1_dataloader'] = 'NOT FOUND'

# ─────────────────────────────────────────────────────────────────────────────
# 修改2a: DualPathTransformer.__init__ 加 _graph_enabled=False 标志
# ─────────────────────────────────────────────────────────────────────────────
OLD2A = (
    "        # 用户图注意力（批内 TopK 邻居聚合，训练时启用）\n"
    "        self.graph_attn = UserGraphAttention(d_model * 2, n_neighbors=10)"
)
NEW2A = (
    "        # 用户图注意力（批内 TopK 邻居聚合）\n"
    "        # _graph_enabled=False：前30轮跳过 O(B²) 开销，第30轮后由外部开启\n"
    "        self.graph_attn = UserGraphAttention(d_model * 2, n_neighbors=10)\n"
    "        self._graph_enabled = False"
)
if OLD2A in src:
    src = src.replace(OLD2A, NEW2A, 1)
    results['2a_graph_flag'] = 'OK'
else:
    results['2a_graph_flag'] = 'NOT FOUND'

# ─────────────────────────────────────────────────────────────────────────────
# 修改2b: forward 中图注意力判断改为检查 _graph_enabled
# ─────────────────────────────────────────────────────────────────────────────
OLD2B = (
    "        # 训练时：用批内图注意力聚合邻居信息，让离群异常更显著\n"
    "        if self.training and feat_cat.shape[0] > 1:\n"
    "            feat_cat = self.graph_attn(feat_cat)"
)
NEW2B = (
    "        # 训练时：仅在 _graph_enabled=True 后才启用（前30轮跳过，节省约30%时间）\n"
    "        if self.training and self._graph_enabled and feat_cat.shape[0] > 1:\n"
    "            feat_cat = self.graph_attn(feat_cat)"
)
if OLD2B in src:
    src = src.replace(OLD2B, NEW2B, 1)
    results['2b_forward_guard'] = 'OK'
else:
    results['2b_forward_guard'] = 'NOT FOUND'

# ─────────────────────────────────────────────────────────────────────────────
# 修改2c: 训练循环 epoch==30 时触发图注意力
# ─────────────────────────────────────────────────────────────────────────────
OLD2C = (
    "    mark = ' ✅' if val_auc > best_auc_dual else ''\n"
    "    print(f\"{epoch:>5} | {tr_loss:>10.6f} | {val_auc:>7.4f} | \"\n"
    "          f\"{val_f1:>6.4f} | {val_thr:>5.3f} | {cur_lr:>8.2e} | {elapsed:>5.1f}s{mark}\")"
)
NEW2C = (
    "    # Epoch 30 后启用图注意力（模型基本收敛后 O(B²) 开销才值得）\n"
    "    if epoch == 30 and not model_dual._graph_enabled:\n"
    "        model_dual._graph_enabled = True\n"
    "        print(\"  ⚡ Epoch 30: 启用 UserGraphAttention\")\n"
    "\n"
    "    mark = ' ✅' if val_auc > best_auc_dual else ''\n"
    "    print(f\"{epoch:>5} | {tr_loss:>10.6f} | {val_auc:>7.4f} | \"\n"
    "          f\"{val_f1:>6.4f} | {val_thr:>5.3f} | {cur_lr:>8.2e} | {elapsed:>5.1f}s{mark}\")"
)
if OLD2C in src:
    src = src.replace(OLD2C, NEW2C, 1)
    results['2c_epoch30_trigger'] = 'OK'
else:
    results['2c_epoch30_trigger'] = 'NOT FOUND'

# ─────────────────────────────────────────────────────────────────────────────
# 修改3a: federated_train_round 函数签名 + 内部逻辑（接收预创建模型）
# ─────────────────────────────────────────────────────────────────────────────
OLD3A = (
    "def federated_train_round(global_model, clients_data_list,\n"
    "                          device, local_epochs=3, lr=2e-4):\n"
    "    \"\"\"\n"
    "    一轮联邦训练：\n"
    "    1. 广播全局模型参数到各客户端\n"
    "    2. 各客户端本地训练 local_epochs 轮（AdaptiveFocalLoss）\n"
    "    3. FedAvg 聚合，更新全局模型\n"
    "    \"\"\"\n"
    "    client_states, client_sizes = [], []\n"
    "\n"
    "    for X_c, y_c in clients_data_list:\n"
    "        # 复制全局模型到本地\n"
    "        local_model = DualPathTransformer(\n"
    "            feat_dim=FEAT_DIM, d_model=128, nhead=4,\n"
    "            num_layers=4, dim_ff=512, dropout=0.2,\n"
    "            win_size=6, max_len=60\n"
    "        ).to(device)\n"
    "        local_model.load_state_dict(global_model.state_dict())"
)
NEW3A = (
    "def federated_train_round(global_model, clients_data_list,\n"
    "                          device, local_epochs=3, lr=2e-4,\n"
    "                          local_models=None):\n"
    "    \"\"\"\n"
    "    ⚡ 优化版：接收预创建的 local_models，每轮只做 load_state_dict，\n"
    "    不重复调用 DualPathTransformer.__init__（省去权重初始化开销）。\n"
    "    1. load_state_dict 广播全局参数\n"
    "    2. 各客户端本地训练 local_epochs 轮\n"
    "    3. FedAvg 聚合，更新全局模型\n"
    "    \"\"\"\n"
    "    client_states, client_sizes = [], []\n"
    "\n"
    "    for _ci, (X_c, y_c) in enumerate(clients_data_list):\n"
    "        # ⚡ 复用预创建模型，只更新参数，无 __init__ 开销\n"
    "        if local_models is not None:\n"
    "            local_model = local_models[_ci]\n"
    "        else:\n"
    "            local_model = DualPathTransformer(\n"
    "                feat_dim=FEAT_DIM, d_model=128, nhead=4,\n"
    "                num_layers=4, dim_ff=512, dropout=0.2,\n"
    "                win_size=6, max_len=60\n"
    "            ).to(device)\n"
    "        local_model.load_state_dict(global_model.state_dict())"
)
if OLD3A in src:
    src = src.replace(OLD3A, NEW3A, 1)
    results['3a_fed_signature'] = 'OK'
else:
    results['3a_fed_signature'] = 'NOT FOUND'

# ─────────────────────────────────────────────────────────────────────────────
# 修改3b: 联邦训练主循环前预创建 local_models，调用时传入
# ─────────────────────────────────────────────────────────────────────────────
OLD3B = (
    "# ── 联邦训练主循环 ────────────────────────────────────────────────\n"
    "print(f\"\\n{'轮次':>4} | {'ValAUC':>7} | {'ValF1':>6} | {'时间':>6}\")\n"
    "print(f\"{'='*35}\")\n"
    "\n"
    "best_fed_auc   = best_auc_dual   # 以集中式训练结果为初始基准\n"
    "best_fed_state = best_state_dual\n"
    "\n"
    "for fed_round in range(1, FED_ROUNDS + 1):\n"
    "    t0_fed = time.time()\n"
    "    model_dual = federated_train_round(\n"
    "        model_dual, clients_data, device,\n"
    "        local_epochs=LOCAL_EPOCHS, lr=FED_LR,\n"
    "    )"
)
NEW3B = (
    "# ── 联邦训练主循环 ────────────────────────────────────────────────\n"
    "# ⚡ 预创建 N_CLIENTS 个本地模型（每轮复用，避免重复 __init__）\n"
    "_fed_local_models = [\n"
    "    DualPathTransformer(\n"
    "        feat_dim=FEAT_DIM, d_model=128, nhead=4,\n"
    "        num_layers=4, dim_ff=512, dropout=0.2,\n"
    "        win_size=6, max_len=60\n"
    "    ).to(device)\n"
    "    for _ in range(N_CLIENTS)\n"
    "]\n"
    "print(f\"  预创建 {N_CLIENTS} 个联邦本地模型 ✅\")\n"
    "\n"
    "print(f\"\\n{'轮次':>4} | {'ValAUC':>7} | {'ValF1':>6} | {'时间':>6}\")\n"
    "print(f\"{'='*35}\")\n"
    "\n"
    "best_fed_auc   = best_auc_dual   # 以集中式训练结果为初始基准\n"
    "best_fed_state = best_state_dual\n"
    "\n"
    "for fed_round in range(1, FED_ROUNDS + 1):\n"
    "    t0_fed = time.time()\n"
    "    model_dual = federated_train_round(\n"
    "        model_dual, clients_data, device,\n"
    "        local_epochs=LOCAL_EPOCHS, lr=FED_LR,\n"
    "        local_models=_fed_local_models,\n"
    "    )"
)
if OLD3B in src:
    src = src.replace(OLD3B, NEW3B, 1)
    results['3b_fed_prealloc'] = 'OK'
else:
    results['3b_fed_prealloc'] = 'NOT FOUND'

# ─────────────────────────────────────────────────────────────────────────────
# 修改4: torch.compile（模型初始化后插入）
# ─────────────────────────────────────────────────────────────────────────────
OLD4 = (
    "model_dual = DualPathTransformer(\n"
    "    feat_dim   = FEAT_DIM,   # 19\n"
    "    d_model    = 128,        # 64→128，提升表达能力\n"
    "    nhead      = 4,\n"
    "    num_layers = 4,          # 3→4，加深网络\n"
    "    dim_ff     = 512,        # 256→512\n"
    "    dropout    = 0.2,        # 微调\n"
    "    win_size   = 6,          # 局部窗口略扩大\n"
    "    max_len    = 60\n"
    ").to(device)"
)
NEW4 = (
    "model_dual = DualPathTransformer(\n"
    "    feat_dim   = FEAT_DIM,   # 19\n"
    "    d_model    = 128,        # 64→128，提升表达能力\n"
    "    nhead      = 4,\n"
    "    num_layers = 4,          # 3→4，加深网络\n"
    "    dim_ff     = 512,        # 256→512\n"
    "    dropout    = 0.2,        # 微调\n"
    "    win_size   = 6,          # 局部窗口略扩大\n"
    "    max_len    = 60\n"
    ").to(device)\n"
    "\n"
    "# ⚡ torch.compile（PyTorch 2.0+）：CPU 下通常有 10~30% 加速\n"
    "# 首个 batch 会有一次编译开销（约数秒），后续每 batch 均加速\n"
    "if hasattr(torch, 'compile'):\n"
    "    try:\n"
    "        model_dual = torch.compile(model_dual, mode='reduce-overhead')\n"
    "        print('  ⚡ torch.compile 已启用（mode=reduce-overhead）')\n"
    "    except Exception as _ce:\n"
    "        print(f'  ⚠️ torch.compile 不可用，跳过: {_ce}')\n"
    "else:\n"
    "    print('  ℹ️ PyTorch 版本不支持 torch.compile，建议升级到 2.0+')"
)
if OLD4 in src:
    src = src.replace(OLD4, NEW4, 1)
    results['4_torch_compile'] = 'OK'
else:
    results['4_torch_compile'] = 'NOT FOUND'

# ─────────────────────────────────────────────────────────────────────────────
# 写回
# ─────────────────────────────────────────────────────────────────────────────
with open(TARGET, 'w', encoding='utf-8') as f:
    f.write(src)

import io, sys
out = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
out.write("=" * 50 + "\n")
for k, v in results.items():
    status = "[OK]" if v == "OK" else "[!!]"
    out.write(f"  {status} {k}: {v}\n")
out.write("=" * 50 + "\n")
all_ok = all(v == "OK" for v in results.values())
out.write("ALL OK\n" if all_ok else "SOME FAILED\n")
out.flush()
sys.exit(0 if all_ok else 1)
