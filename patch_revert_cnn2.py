# -*- coding: utf-8 -*-
"""移除 CNN 初始化块残留"""
path = 'sgcc_analysis.py'
lines = open(path, encoding='utf-8').readlines()

# 找到 CNN 初始化块的起始和结束行
start_marker = '# [创新点A] MultiScaleCNN1D: 提取原始时序多尺度特征'
end_marker   = 'print(f"  [CNN分支] 参数量: {cnn_params:,}")'

start_idx = None
end_idx   = None
for i, line in enumerate(lines):
    if start_marker in line and start_idx is None:
        start_idx = i
    if end_marker in line and start_idx is not None:
        end_idx = i
        break

if start_idx is not None and end_idx is not None:
    # 删除 start_idx 到 end_idx（含）的所有行
    removed = lines[start_idx:end_idx+1]
    lines = lines[:start_idx] + lines[end_idx+1:]
    open(path, 'w', encoding='utf-8').writelines(lines)
    print(f'已移除 {len(removed)} 行 (行{start_idx+1}~{end_idx+1})')
    print('移除内容预览（前3行）:')
    for r in removed[:3]:
        print(' ', r.rstrip())
else:
    print(f'start_idx={start_idx}, end_idx={end_idx} - 未找到，已跳过')
