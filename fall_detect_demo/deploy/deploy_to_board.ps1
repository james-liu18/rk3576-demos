﻿﻿﻿# fall_detect_demo 一键部署脚本 (Windows PC 端)
# 前提: 板子通过 USB 连接本机 (adb 可用); PC 能 ssh 编译机
# 用法:
#   powershell -File deploy_to_board.ps1                       # 首次/完整部署
#   powershell -File deploy_to_board.ps1 -SkipDeps             # 只更新代码+模型
#   powershell -File deploy_to_board.ps1 -CompilePw <密码>      # 指定编译机密码
param(
    [string]$CompileHost = "q@10.66.84.59",
    [string]$CompilePw = "",                       # 编译机密码; 首次部署(需要拉离线包)时必填
    [switch]$SkipDeps                              # 跳过 Python/运行时安装, 只推代码
)
$ErrorActionPreference = "Stop"
$Repo   = $PSScriptRoot          # deploy/ 目录
$Demo   = Split-Path $Repo       # fall_detect_demo/ 根目录
$PkgDir = Join-Path $Repo "pkg"  # 本地离线包缓存
$Plink  = "C:\Program Files\PuTTY\plink.exe"
$Pscp   = "C:\Program Files\PuTTY\pscp.exe"
$Adb    = "E:\platform-tools\adb.exe"

function Rem([string]$cmd) {
    & $Plink -ssh -batch -pw $CompilePw $CompileHost $cmd 2>&1
}

Write-Host "=== 1/5 检查板子连接 ==="
& $Adb get-state 2>$null
if ($LASTEXITCODE -ne 0) { throw "板子未连接 (adb)"; }
& $Adb shell "echo 板子OK:"; & $Adb shell uname -m

if (-not $SkipDeps) {
    Write-Host "=== 2/5 从编译机拉取离线部署包 (有本地缓存则跳过) ==="
    New-Item -ItemType Directory -Force -Path $PkgDir | Out-Null
    $Tar = Join-Path $PkgDir "cpython-3.12-aarch64-install_only.tar.gz"
    if (-not (Test-Path $Tar)) {
        if (-not $CompilePw) { throw "首次部署需要 -CompilePw <编译机密码> 拉取离线包" }
        & $Pscp -batch -pw $CompilePw "${CompileHost}:board_pkg/cpython-3.12-aarch64-install_only.tar.gz" $PkgDir 2>&1 | Select-Object -Last 1
    }
    if (-not (Test-Path (Join-Path $PkgDir "wheels_pkgs"))) {
        if (-not $CompilePw) { throw "首次部署需要 -CompilePw <编译机密码> 拉取离线包" }
        & $Pscp -batch -pw $CompilePw -r "${CompileHost}:board_pkg/wheels_pkgs" $PkgDir 2>&1 | Select-Object -Last 1
    }
    Write-Host "=== 3/5 推送部署包到板子 ==="
    & $Adb shell "mkdir -p /tmp/board_pkg/wheels_pkgs"
    & $Adb push $Tar /tmp/board_pkg/ 2>&1 | Select-Object -Last 1
    & $Adb push (Join-Path $Repo "assets\librknnrt.so") /tmp/board_pkg/ 2>&1 | Select-Object -Last 1
    & $Adb push (Join-Path $PkgDir "wheels_pkgs") /tmp/board_pkg/ 2>&1 | Select-Object -Last 1
    & $Adb push (Join-Path $Repo "board_setup.sh") /tmp/board_pkg/ 2>&1 | Select-Object -Last 1
    Write-Host "=== 4/5 板端安装 Python/运行时 (幂等) ==="
    & $Adb shell "bash /tmp/board_pkg/board_setup.sh"
} else {
    Write-Host "=== 2-4/5 跳过 (-SkipDeps) ==="
}

Write-Host "=== 5/5 推送 demo 代码/模型 并自测 ==="
& $Adb shell "mkdir -p /opt/fall_detect_demo"
Push-Location $Demo
& $Adb push README.md selftest.py main.py pose_detector.py fall_detector.py `
    alarm.py audio_monitor.py event_recorder.py ui.py config.yaml run.sh `
    /opt/fall_detect_demo/ 2>&1 | Select-Object -Last 1
& $Adb push models /opt/fall_detect_demo/ 2>&1 | Select-Object -Last 1
& $Adb push assets /opt/fall_detect_demo/ 2>&1 | Select-Object -Last 1
& $Adb push tools /opt/fall_detect_demo/ 2>&1 | Select-Object -Last 1
Pop-Location
& $Adb shell "chmod +x /opt/fall_detect_demo/run.sh; rm -rf /opt/fall_detect_demo/__pycache__"
& $Adb shell "cd /opt/fall_detect_demo && /opt/py312/bin/python3.12 selftest.py" | Select-String "FAIL|通过"

Write-Host ""
Write-Host "部署完成! 板上运行:"
Write-Host "  cd /opt/fall_detect_demo && ./run.sh            # 带界面"
Write-Host "  cd /opt/fall_detect_demo && ./run.sh --no-ui    # 无界面"
Write-Host "  ./run.sh --test-audio                           # 声音阈值标定"
