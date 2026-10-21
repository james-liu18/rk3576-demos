﻿﻿# 打包自包含离线部署包 demo_fall_detect_bundle.tar.gz
# 内容: py3.12 离线包 + wheels + librknnrt.so + demo 代码 + board_setup.sh
# 板子拿到包后: tar -xzf demo_fall_detect_bundle.tar.gz && cd demo_fall_detect_bundle && sudo bash board_setup.sh
param([string]$Out = "demo_fall_detect_bundle.tar.gz")
$ErrorActionPreference = "Stop"
$Deploy = $PSScriptRoot
$Demo   = Split-Path $Deploy
$PkgDir = Join-Path $Deploy "pkg"

foreach ($f in @((Join-Path $PkgDir "cpython-3.12-aarch64-install_only.tar.gz"),
                 (Join-Path $PkgDir "wheels_pkgs"),
                 (Join-Path $Deploy "assets\librknnrt.so"))) {
    if (-not (Test-Path $f)) { throw "缺少部署资产: $f (先跑一次 deploy_to_board.ps1 或从编译机 board_pkg 拉取)" }
}

$Stage = Join-Path $env:TEMP "demo_fall_detect_bundle_stage"
if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
$B = Join-Path $Stage "demo_fall_detect_bundle"
New-Item -ItemType Directory -Force -Path $B, (Join-Path $B "demo") | Out-Null

Copy-Item (Join-Path $PkgDir "cpython-3.12-aarch64-install_only.tar.gz") $B
Copy-Item (Join-Path $PkgDir "wheels_pkgs") $B -Recurse
Copy-Item (Join-Path $Deploy "assets\librknnrt.so") $B
Copy-Item (Join-Path $Deploy "board_setup.sh") $B

Push-Location $Demo
$files = "README.md","selftest.py","main.py","pose_detector.py","fall_detector.py",`
    "alarm.py","audio_monitor.py","event_recorder.py","ui.py","config.yaml","run.sh"
Copy-Item -Path $files -Destination (Join-Path $B "demo")
Copy-Item models, assets, tools (Join-Path $B "demo") -Recurse
Pop-Location

$OutPath = Join-Path $Demo $Out
if (Test-Path $OutPath) { Remove-Item $OutPath }
tar -czf $OutPath -C $Stage demo_fall_detect_bundle
Remove-Item -Recurse -Force $Stage
"{0}  ({1:N1} MB)" -f $OutPath, ((Get-Item $OutPath).Length / 1MB)
