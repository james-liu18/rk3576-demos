# rk3576-demos — Quectel PI M2 (RK3576) NPU 视觉 Demo 合集

在 Quectel PI M2 (RK3576, Debian trixie) 上实测通过的两个摄像头 NPU demo。

| Demo | 功能 | NPU 用途 |
|------|------|----------|
| [fall_detect_demo](fall_detect_demo/) | 人体跌倒检测 + 异常声音告警（LED 闪烁 / 警笛 / 事件录像） | yolov8n-pose 姿态推理 (~60ms/帧) |
| [style_cam_demo](style_cam_demo/) | 摄像头预览 + 实时风格化特效 + 拍照存储 | 7 种风格迁移 + 4 种人体关键点贴纸 (43~152ms/帧) |

## 仓库结构

```
rk3576-demos/
├── fall_detect_demo/          # 跌倒检测 + 声音告警
│   ├── main.py / *.py         # 主程序与各模块
│   ├── models/yolo_pose.rknn  # 已转好的模型 (fp16, rk3576)
│   ├── deploy/                # 板端安装脚本 + NPU runtime
│   ├── tools/                 # 模型导出/转换、调试工具
│   └── README.md / README_EN.md
└── style_cam_demo/            # 风格化相机
    ├── main.py / *.py         # 主程序与各模块
    ├── models/*.rknn          # 10 个已转好的模型 (fp16)
    ├── tools/                 # 模型下载/转换(含图手术)、板端调试
    └── README.md / README_EN.md
```

## 环境要求

- **板子**: Quectel PI M2 / RK3576，Debian trixie，NPU 驱动 0.9.8，librknnrt ≥ 2.5.0
- **Python**: 3.12（Debian trixie 自带 3.13，装不上 cp312 的 rknn wheel，脚本会自动用 uv 装 3.12 到 `/opt/py312`）
- **外设**: USB 摄像头 (UVC)；fall demo 另需麦克风/喇叭（ALSA `plughw` 设备）
- **模型**: `models/` 下已带全部 .rknn，开箱即跑；重新转换需 x86 Linux PC + rknn-toolkit2

## 快速开始（板端在线部署）

前置：板子可联网，USB 摄像头接好，root（或 sudo）终端。首次部署只需一条命令：

```bash
git clone https://github.com/james-liu18/rk3576-demos.git
cd rk3576-demos
sudo bash fall_detect_demo/deploy/board_setup.sh
# 自动完成: Python 3.12(/opt/py312) + pip 依赖 + PySide6 + librknnrt + 系统工具
#          (libxcb-cursor0 在 apt 源缺失时自动从 Debian 官方池补装)
```

**方式 A：直接在仓库目录运行**（试用/开发，run.sh 自动定位所在目录）：

```bash
cd fall_detect_demo
sudo python3 selftest.py        # 离线逻辑自测（可选）
sudo ./run.sh                   # 带界面; 无显示器用 ./run.sh --no-ui

cd ../style_cam_demo            # 复用 /opt/py312, 无新依赖
sudo ./run.sh
```

**方式 B：部署到 /opt 运行**（正式部署，开机自启/固定路径场景）。fall demo 已由
board_setup.sh 自动复制到 /opt；style_cam 需手动复制一次：

```bash
cd /opt/fall_detect_demo && sudo ./run.sh
sudo cp -r style_cam_demo /opt/ && cd /opt/style_cam_demo && sudo ./run.sh
```

详细说明见各 demo 目录下的 README（中文）/ README_EN（English）：

- [fall_detect_demo/README.md](fall_detect_demo/README.md) — 配置标定（LED GPIO / 音频阈值 / 摄像头）、模型重转、常见问题
- [style_cam_demo/README.md](style_cam_demo/README.md) — 特效清单与性能、PySide6 界面、模型转换要点（踩坑记录）
