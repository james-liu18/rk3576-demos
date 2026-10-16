# rk3576-demos — NPU Vision Demo Collection for Quectel PI M2 (RK3576)

Two camera-based NPU demos, verified on Quectel PI M2 (RK3576, Debian trixie).

| Demo | Features | NPU workload |
|------|----------|--------------|
| [fall_detect_demo](fall_detect_demo/) | Human fall detection + abnormal-sound alarm (LED / siren / event recording) | yolov8n-pose inference (~60 ms/frame) |
| [style_cam_demo](style_cam_demo/) | Camera preview + real-time style effects + photo capture | 7 style-transfer models + 4 keypoint sticker modes (43~152 ms/frame) |

## Repository Layout

```
rk3576-demos/
├── fall_detect_demo/          # Fall detection + sound alarm
│   ├── main.py / *.py         # Main program and modules
│   ├── models/yolo_pose.rknn  # Pre-converted model (fp16, rk3576)
│   ├── deploy/                # On-board setup script + NPU runtime
│   ├── tools/                 # Model export/conversion, debug utilities
│   └── README.md / README_EN.md
└── style_cam_demo/            # Style camera
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

```bash
sudo apt-get install -y git
git clone https://github.com/james-liu18/rk3576-demos.git
cd rk3576-demos

# ---- fall demo: one-shot setup (Python env / deps / librknnrt) + deploy ----
sudo bash fall_detect_demo/deploy/board_setup.sh
cd /opt/fall_detect_demo
sudo python3 selftest.py        # offline logic self-test (optional)
sudo ./run.sh                   # with UI; use ./run.sh --no-ui headless

# ---- style cam: reuses /opt/py312, no extra deps ----
sudo cp -r style_cam_demo /opt/
cd /opt/style_cam_demo && sudo ./run.sh
```

See each demo's README (Chinese) / README_EN (English) for details:

- [fall_detect_demo/README_EN.md](fall_detect_demo/README_EN.md) — configuration & calibration (LED GPIO / audio threshold / camera), model re-conversion, troubleshooting
- [style_cam_demo/README_EN.md](style_cam_demo/README_EN.md) — effects list & benchmarks, PySide6 UI, model conversion notes (hard-won lessons)
