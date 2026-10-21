# rk3576-demos — NPU Vision Demo Collection for Quectel PI M2 (RK3576)

Two camera-based NPU demos, verified on Quectel PI M2 (RK3576, Debian trixie).

| Demo | Features | NPU workload |
|------|----------|--------------|
| [demo_fall_detect](demo_fall_detect/) | Human fall detection + abnormal-sound alarm (LED / siren / event recording) | yolov8n-pose inference (~60 ms/frame) |
| [demo_style_camera](demo_style_camera/) | Camera preview + real-time style effects + photo capture | 7 style-transfer models + 4 keypoint sticker modes (43~152 ms/frame) |

## Repository Layout

```
rk3576-demos/
├── demo_fall_detect/          # Fall detection + sound alarm
│   ├── main.py / *.py         # Main program and modules
│   ├── models/yolo_pose.rknn  # Pre-converted model (fp16, rk3576)
│   ├── deploy/                # On-board setup script + NPU runtime
│   ├── tools/                 # Model export/conversion, debug utilities
│   └── README.md / README_EN.md
└── demo_style_camera/            # Style camera
    ├── main.py / *.py         # Main program and modules
    ├── models/*.rknn          # 10 pre-converted models (fp16)
    ├── tools/                 # Model fetch/conversion (incl. graph surgery), board debugging
    └── README.md / README_EN.md
```

## Requirements

- **Board**: Quectel PI M2 / RK3576, Debian trixie, NPU driver 0.9.8, librknnrt ≥ 2.5.0
- **Python**: 3.12 (Debian trixie ships 3.13 which cannot install the cp312 rknn wheel; the setup script automatically installs 3.12 via uv into `/opt/py312`)
- **Peripherals**: USB camera (UVC); fall demo also needs a microphone/speaker (ALSA `plughw` devices)
- **Models**: all `.rknn` files are included under `models/` — runs out of the box; re-conversion requires an x86 Linux PC with rknn-toolkit2

## Quick Start (On-board, online deployment)

Prerequisites: board has internet, USB camera attached, root (or sudo) shell.
One command for first-time setup:

```bash
git clone https://github.com/james-liu18/rk3576-demos.git
cd rk3576-demos
sudo bash demo_fall_detect/deploy/board_setup.sh
# Handles: Python 3.12 (/opt/py312) + pip deps + PySide6 + librknnrt + system tools
#          (libxcb-cursor0 fetched from the Debian pool when missing from apt sources)
```

**Option A: run directly from the repo clone** (trial/dev; run.sh locates its own directory):

```bash
cd demo_fall_detect
sudo python3 selftest.py        # offline logic self-test (optional)
sudo ./run.sh                   # with UI; use ./run.sh --no-ui headless

cd ../demo_style_camera            # reuses /opt/py312, no extra deps
sudo ./run.sh
```

**Option B: run from /opt** (production: autostart / fixed paths). The fall demo
is copied to /opt by board_setup.sh automatically; style_camera needs a one-time
manual copy:

```bash
cd /opt/demo_fall_detect && sudo ./run.sh
sudo cp -r demo_style_camera /opt/ && cd /opt/demo_style_camera && sudo ./run.sh
```

See each demo's README (Chinese) / README_EN (English) for details:

- [demo_fall_detect/README_EN.md](demo_fall_detect/README_EN.md) — configuration & calibration (LED GPIO / audio threshold / camera), model re-conversion, troubleshooting
- [demo_style_camera/README_EN.md](demo_style_camera/README_EN.md) — effects list & benchmarks, PySide6 UI, model conversion notes (hard-won lessons)
