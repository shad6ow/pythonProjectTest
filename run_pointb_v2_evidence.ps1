# run_pointb_v2_evidence.ps1
# 顺序批跑 Point B v2 发表级证据实验:
#   1) 平滑变体 (smooth3) 单次全量
#   2) 5 个随机种子全量 (鲁棒性)
#   3) 多种子聚合 + Wilcoxon 显著性
# 每步输出同时打印并写入 logs/ 下日志文件。挂着跑, 回来贴两个关键日志即可。

$ErrorActionPreference = "Stop"
Set-Location "C:\Users\wb.zhoushujie\PyCharmMiscProject"
$py = ".venv\Scripts\python.exe"
$script = "sgcc_phase4_pointb_v2_localization.py"
$logdir = "results\phase4_pointb_v2_evidence\logs"
New-Item -ItemType Directory -Force -Path $logdir | Out-Null

function Run-Step($name, $cmd) {
    $log = Join-Path $logdir "$name.log"
    Write-Host "==================== [$name] start $(Get-Date -Format HH:mm:ss) ===================="
    Invoke-Expression $cmd 2>&1 | Tee-Object -FilePath $log
    Write-Host "==================== [$name] done  $(Get-Date -Format HH:mm:ss)  -> $log ===================="
}

# ---- 节点 2: 平滑变体 (先跑, 单次出结果快) ----
Run-Step "smooth3" "$py $script --epochs 20 --max-users 0 --n-inject 500 --attn-smooth 3 --output-dir results/phase4_pointb_v2_smooth3"

# ---- 节点 1: 5 个随机种子 (长) ----
foreach ($seed in 11, 22, 33, 44, 55) {
    Run-Step "seed$seed" "$py $script --epochs 20 --max-users 0 --n-inject 500 --seed $seed --seed-suffix"
}

# ---- 多种子聚合 + 显著性检验 ----
Run-Step "aggregate" "$py sgcc_phase4_pointb_v2_aggregate.py"

Write-Host ""
Write-Host "全部完成。请贴两个日志给我:"
Write-Host "  1) $logdir\aggregate.log   (多种子 mean/std/CI + Wilcoxon)"
Write-Host "  2) $logdir\smooth3.log     (平滑变体 by_shape IoU, 末尾 JSON)"
