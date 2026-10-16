# Fall Detection + Abnormal Sound Alarm Demo (RK3576)

Human fall detection based on a USB camera + YOLO-Pose (RKNN/NPU), combined with
microphone-based abnormal sound detection. When an event triggers: LED blinking +
speaker siren + locally saved 10s audio/video recording, with a Tkinter UI.

```
fall_detect_demo/
├── main.py               # Main entry point
├── config.yaml           # All configuration (camera/model/thresholds/LED/recording)
├── run.sh                # On-board launcher wrapper (root borrows desktop user's XWayland)
├── pose_detector.py      # YOLO-Pose inference (rknnlite / opencv-dnn dual runtime)
├── fall_detector.py      # Fall decision state machine (scoring: torso tilt/body ratio/ground proximity/lateral spread)
├── audio_monitor.py      # arecord mic energy detection (RMS dBFS + adaptive noise floor)
├── alarm.py              # LED (gpiod/sysfs/pwm) + aplay siren
├── event_recorder.py     # Ring buffer, saves 10s mp4 + wav + events.jsonl per event
├── ui.py                 # Tkinter UI (falls back to OpenCV window / headless)
├── selftest.py           # Offline self-test (no camera/NPU needed)
├── models/yolo_pose.rknn # Pre-converted model (fp16, rk3576)
├── records/              # Event recordings output (generated at runtime)
├── deploy/
│   ├── board_setup.sh    # One-shot on-board installer (online/offline dual mode)
│   ├── make_bundle.ps1   # PC: build offline deployment bundle
│   ├── deploy_to_board.ps1 # PC: push bundle to board via adb
│   └── assets/librknnrt.so # NPU runtime (aarch64)
└── tools/
    ├── export_onnx.sh    # PC: download yolov8n-pose and export onnx
    ├── convert.py        # PC: onnx -> rknn (rk3576, fp16/int8)
    └── fall_feat_debug.py # Board: print fall features in real time for calibration
```

## 1. PC: Model Preparation

```bash
# 1. Export onnx (default yolov8n-pose; also n|s + 8|11 combos, e.g. "s 11")
./tools/export_onnx.sh n 8

# 2. Convert to rknn (x86 PC + python3.8~3.12)
pip install "rknn-toolkit2>=2.3.0"
python3 tools/convert.py models/yolov8n-pose.onnx rk3576
#    Optional int8 quantization (faster): first fill tools/dataset.txt with calibration images
#    python3 tools/convert.py models/yolov8n-pose.onnx rk3576 i8
```

Copy the generated `models/yolo_pose.rknn` to the board.

> You can verify the whole pipeline on PC first with onnx (cv2.dnn on CPU):
> `python3 main.py --model models/yolov8n-pose.onnx`

## 2. On-board Deployment (RK3576, Debian)

### 0. One-shot deployment (recommended, board has internet)

```bash
git clone https://github.com/james-liu_Quectel/rk3576-demos.git
sudo bash rk3576-demos/fall_detect_demo/deploy/board_setup.sh
```

The script is idempotent and automatically handles: Python 3.12 (`/opt/py312`,
installed online via uv if missing) → pip dependencies → librknnrt.so →
alsa-utils/gpiod → demo code copied to `/opt/fall_detect_demo`.
If offline bundle files are present (cpython tarball / wheels_pkgs, built with
make_bundle.ps1), they are used preferentially.

Then:

```bash
cd /opt/fall_detect_demo
sudo ./run.sh                # with UI (root borrows desktop user's XWayland)
sudo ./run.sh --no-ui        # headless
```

### 1. Manual deployment (offline / customized scenarios)

The SDK's rknn-toolkit-lite2 wheel only goes up to cp312, while Debian trixie
ships python3.13. **Check first: `python3 --version`**

- For 3.12 or lower, install the SDK wheel directly:
  `pip3 install RK3576_Linux6.1_BL01_v1.2.0/external/rknn-toolkit2/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.2-cp3XX-...whl`
- For 3.13: install python3.12 (e.g. `uv venv -p 3.12` or conda) and install the wheel inside it.

```bash
# Other dependencies
apt install -y alsa-utils gpiod python3-gpiod libgles2  # as needed
pip3 install numpy opencv-python pyyaml pillow   # or system python3-opencv
```

### 2. NPU runtime library

```bash
cp RK3576_Linux6.1_BL01_v1.2.0/external/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/
# Verify: dmesg | grep -i rknpu  (driver 0.9.8)
```

### 3. Camera / Microphone / LED

```bash
ls /dev/video*          # USB camera (UVC enabled in kernel)
arecord -l              # list sound cards/PDM mics, determine arecord -D device name
aplay -l                # es8389 codec, speaker via codec + speak_pa_en amplifier
```

LED: the board dts does not define gpio-leds, so the demo drives GPIO directly.
Edit `config.yaml`:

```yaml
alarm:
  led:
    mode: gpiod        # gpiod | sysfs_gpio | sysfs_led | none
    chip: 0
    line: <your LED GPIO number>   # find a free line via gpioinfo gpiochip0
    active_low: false
```

Confirm the LED pin manually with `gpioset gpiochip0 <line>=1` before filling in the config.

### 4. Run

```bash
cd fall_detect_demo                  # /opt/fall_detect_demo after deployment
python3 selftest.py                  # offline logic self-test (optional)
python3 main.py --test-alarm         # test LED + speaker
python3 main.py --test-audio         # observe ambient noise, calibrate abs_threshold_db
sudo ./run.sh                        # main program with UI (LED needs root)
sudo ./run.sh --no-ui                # headless
# or directly: sudo python3 main.py [--no-ui]
```

## 3. Features & Configuration

| Feature | Logic | Key config |
|---|---|---|
| Fall detection | torso-to-vertical angle > 55° or bbox h/w ratio < 1.05 for 4 frames; hip drop as auxiliary | `fall.*` |
| Abnormal sound | RMS energy above absolute threshold (-28dBFS) and 12dB above noise floor, 2 consecutive windows | `audio.*` |
| Alarm | LED blink + siren loop, 10s default; keeps alarming while fall persists | `alarm.*` |
| Recording | 4s pre-event rollback + 10s total, mp4+wav in `records/<date>/`, index `events.jsonl` | `record.*` |

Model is swappable: `model.path` accepts any yolov8/yolo11-pose `.rknn`
(airockchip 116-channel export) or `.onnx` (standard ultralytics export);
post-processing auto-detects both formats.

## 4. Troubleshooting

- **rknn load failure / no NPU**: make sure `librknnrt.so` is installed and the
  wheel matches the Python version; check driver with
  `cat /sys/kernel/debug/rknpu/version`.
- **No alarm sound**: test with `aplay assets/alarm.wav` first; check
  `aplay -l` and fill the device name into `alarm.speaker.device`.
- **Too many false alarms**: raise `audio.abs_threshold_db`,
  `fall.min_trigger_frames`, or `model.conf_threshold`.
- **Camera won't open**: run `v4l2-ctl --list-devices`, adjust `camera.index`.
