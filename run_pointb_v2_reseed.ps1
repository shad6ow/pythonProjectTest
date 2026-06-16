# run_pointb_v2_reseed.ps1
# Re-run 5 seeds + aggregate after adding fair baseline (random_interval).
# (smooth3 weak-shape gate already concluded; no re-run needed.)
# Overwrites seed*/ metrics so per_user_iou includes random_interval.

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

foreach ($seed in 11, 22, 33, 44, 55) {
    Run-Step "seed$seed" "$py $script --epochs 20 --max-users 0 --n-inject 500 --seed $seed --seed-suffix"
}

Run-Step "aggregate" "$py sgcc_phase4_pointb_v2_aggregate.py"

Write-Host ""
Write-Host "Done. Paste $logdir\aggregate.log (focus: mil_vs_random_interval and random_interval_mean_iou)."
