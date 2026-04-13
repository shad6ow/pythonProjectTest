# -*- coding: utf-8 -*-
"""
一次性修改 sgcc_analysis.py 中影响 AUC/F1 的关键超参：
  1. GAMMA: 2.0 → 3.0（更强聚焦难分异常样本）
  2. MAX_RECALL_W: 1.5 → 2.5（训练后期大幅提升召回权重，改善F1）
  3. EPOCHS: 50 → 80（配合早停，充分训练）
  4. PATIENCE: 15 → 20（给AUC更多改善机会）
  5. AUTO_ALPHA: 固定计算→略收紧（max(...,0.80)，避免极端正类损失）
  6. d_model: 128 → 256（更强表达能力）
  7. dropout: 0.2 → 0.1（减小正则，让模型更充分学习特征）
  8. win_size: 6 → 8（局部窗口扩大，捕捉更长异常模式）
  9. max_lr: 8e-4 → 1e-3（配合d_model=256更充分探索）
  10. pct_start: 0.15 → 0.20（稍长预热，大模型更重要）
"""

path = 'sgcc_analysis.py'
content = open(path, encoding='utf-8').read()

changes = [
    # 1~5: 超参组合
    (
        'AUTO_ALPHA = 1.0 - POS_COUNT / (POS_COUNT + NEG_COUNT)\n'
        'GAMMA         = 2.0\n'
        'MAX_RECALL_W  = 1.5   # AdaptiveFocalLoss 最大召回权重\n'
        'EPOCHS        = 50    # CPU下适当减少轮次：50轮足够收敛，60轮增加20%时间但收益边际\n'
        'PATIENCE      = 15    # 早停更激进：CPU下每轮更贵，15轮无提升即停',
        '# ── 提升AUC/F1 超参调整（目标 AUC≥0.85, F1≥0.50） ──────────────\n'
        '# GAMMA: 2.0→3.0，更强聚焦难分异常样本\n'
        '# MAX_RECALL_W: 1.5→2.5，训练后期更激进提升召回率，直接改善F1\n'
        '# EPOCHS: 50→80，配合早停充分训练\n'
        '# PATIENCE: 15→20，给AUC更多改善机会\n'
        '_raw_alpha    = 1.0 - POS_COUNT / (POS_COUNT + NEG_COUNT)\n'
        'AUTO_ALPHA    = max(_raw_alpha * 0.95, 0.80)   # 略收紧，避免极端正类损失爆炸\n'
        'GAMMA         = 3.0          # 2.0→3.0：更强聚焦难分异常样本\n'
        'MAX_RECALL_W  = 2.5          # 1.5→2.5：训练后期大幅提升召回权重，改善F1\n'
        'EPOCHS        = 80           # 50→80：配合早停，充分训练\n'
        'PATIENCE      = 20           # 15→20：搭配更长训练，给AUC更多改善机会'
    ),
    # 6: d_model
    (
        'd_model    = 128,        # 64→128，提升表达能力',
        'd_model    = 256,        # 128→256：更强表达能力，AUC提升关键'
    ),
    # 7: dropout
    (
        'dropout    = 0.2,        # 微调',
        'dropout    = 0.1,        # 0.2→0.1：减小dropout，让模型更充分学习特征'
    ),
    # 8: win_size
    (
        'win_size   = 6,          # 局部窗口略扩大',
        'win_size   = 8,          # 6→8：局部窗口扩大，捕捉更长异常模式'
    ),
    # 9: max_lr
    (
        'max_lr           = 8e-4,',
        'max_lr           = 1e-3,      # 8e-4→1e-3：配合d_model=256更充分探索'
    ),
    # 10: pct_start
    (
        'pct_start        = 0.15,      # 0.3 → 0.15，更快到达峰值再下降',
        'pct_start        = 0.20,      # 0.15→0.20：稍长预热，大模型预热更重要'
    ),
]

for i, (old, new) in enumerate(changes, 1):
    if old in content:
        content = content.replace(old, new, 1)
        print(f'[{i}] OK  : {old[:50].strip()!r}')
    else:
        print(f'[{i}] MISS: {old[:50].strip()!r}')

open(path, 'w', encoding='utf-8').write(content)
print('\n✅ patch_hyperparams.py 全部完成')
