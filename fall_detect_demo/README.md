# 跌倒检测 + 异常声音告警 Demo (RK3576)

基于 USB 摄像头 + YOLO-Pose(RKNN/NPU) 的人体跌倒检测，结合麦克风异常声音检测。
触发事件时：LED 闪烁 + 喇叭警笛告警 + 本地保存 10s 音视频录像，带 Tkinter UI。

```
fall_detect_demo/
├── main.py               # 主程序入口
├── config.yaml           # 全部配置（摄像头/模型/阈值/LED/录音）
├── run.sh                # 板端启动包装 (root 借用桌面用户 XWayland 开窗口)
├── pose_detector.py      # YOLO-Pose 推理 (rknnlite / opencv-dnn 双运行时)
├── fall_detector.py      # 跌倒判定状态机 (打分制: 倾角/身体比/贴地/横向分布)
├── audio_monitor.py      # arecord 麦克风能量检测 (RMS dBFS + 自适应底噪)
├── alarm.py              # LED(gpiod/sysfs/pwm) + aplay 警笛
├── event_recorder.py     # 环形缓冲, 事件保存 10s mp4 + wav + events.jsonl
├── ui.py                 # Tkinter UI (回退 OpenCV 窗口 / 无界面)
├── selftest.py           # 离线自测 (不需要摄像头/NPU)
├── models/yolo_pose.rknn # 已转好的模型 (fp16, rk3576)
├── records/              # 事件录像输出 (运行时生成)
├── deploy/
│   ├── board_setup.sh    # 板端一键安装 (在线/离线双模式)
│   ├── make_bundle.ps1   # PC: 打离线部署包
│   ├── deploy_to_board.ps1 # PC: adb 推包到板子
│   └── assets/librknnrt.so # NPU runtime (aarch64)
└── tools/
    ├── export_onnx.sh    # PC: 下载 yolov8n-pose 并导出 onnx
    ├── convert.py        # PC: onnx -> rknn (rk3576, 支持 fp16/int8)
    └── fall_feat_debug.py # 板端: 实时打印跌倒特征用于标定
```

## 一、PC 端：模型准备

```bash
# 1. 导出 onnx (默认 yolov8n-pose; 也可 n|s + 8|11 组合, 如 s 11)
./tools/export_onnx.sh n 8

# 2. 转 rknn (需 x86 PC + python3.8~3.12)
pip install "rknn-toolkit2>=2.3.0"
python3 tools/convert.py models/yolov8n-pose.onnx rk3576
#    可选 int8 量化(更快): 先编辑 tools/dataset.txt 填校准图片
#    python3 tools/convert.py models/yolov8n-pose.onnx rk3576 i8
```

生成的 `models/yolo_pose.rknn` 拷到板端。

> PC 上可先用 onnx 验证整条链路（CPU 跑 cv2.dnn）：
> `python3 main.py --model models/yolov8n-pose.onnx`

## 二、板端部署 (RK3576, Debian)

### 0. 一键部署 (推荐, 板子可联网)

```bash
git clone https://github.com/james-liu18/rk3576-demos.git
sudo bash rk3576-demos/fall_detect_demo/deploy/board_setup.sh
```

脚本幂等可重复执行，自动完成: Python 3.12 (`/opt/py312`, 无则 uv 在线装) →
pip 依赖 → librknnrt.so → alsa-utils/gpiod → demo 代码复制到 `/opt/fall_detect_demo`。
若目录里有离线包 (cpython tarball / wheels_pkgs，用 make_bundle.ps1 制作) 则优先离线安装。

之后:

```bash
cd /opt/fall_detect_demo
sudo ./run.sh                # 带界面 (root 借用桌面用户 XWayland)
sudo ./run.sh --no-ui        # 无显示器
```

### 1. 手动部署 (无网络/定制场景)


SDK 自带 rknn-toolkit-lite2 wheel 只到 cp312，而 Debian trixie 系统是
python3.13。**先确认版本：`python3 --version`**

- 若为 3.12 及以下，直接装 SDK 的 wheel：
  `pip3 install RK3576_Linux6.1_BL01_v1.2.0/external/rknn-toolkit2/rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.2-cp3XX-...whl`
- 若为 3.13：安装 python3.12（如 `uv venv -p 3.12` 或 conda），在该环境内装 wheel。

```bash
# 其他依赖
apt install -y alsa-utils gpiod python3-gpiod libgles2  # 按需
pip3 install numpy opencv-python pyyaml pillow   # 也可用系统 python3-opencv
```

### 2. NPU runtime 库

```bash
cp RK3576_Linux6.1_BL01_v1.2.0/external/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/
# 验证: dmesg | grep -i rknpu  (驱动 0.9.8)
```

### 3. 摄像头 / 麦克风 / LED

```bash
ls /dev/video*          # USB 摄像头 (内核已开 UVC)
arecord -l              # 查看声卡/PDM 麦, 确定 arecord -D 设备名
aplay -l                # es8389 声卡, 喇叭走 codec + speak_pa_en 功放
```

LED：板级 dts 未定义 gpio-leds，demo 默认直接控 GPIO。修改 `config.yaml`：

```yaml
alarm:
  led:
    mode: gpiod        # gpiod | sysfs_gpio | sysfs_led | none
    chip: 0
    line: <你的LED GPIO编号>   # gpioinfo gpiochip0 查找空闲 line
    active_low: false
```

`gpioset gpiochip0 <line>=1` 先手动确认 LED 引脚，再填入配置。

### 4. 运行

```bash
cd fall_detect_demo                  # 部署后为 /opt/fall_detect_demo
python3 selftest.py                  # 离线逻辑自测 (可选)
python3 main.py --test-alarm         # 测试 LED+喇叭
python3 main.py --test-audio         # 观察环境噪声, 标定 abs_threshold_db
sudo ./run.sh                        # 主程序带界面 (LED 权限需 root)
sudo ./run.sh --no-ui                # 无显示器场景
# 或直接: sudo python3 main.py [--no-ui]
```

## 三、功能与配置说明

| 功能 | 判定逻辑 | 关键配置 |
|---|---|---|
| 跌倒检测 | 躯干与垂直夹角>55° 或 框高宽比<1.05 持续 4 帧; 髋部骤降辅助 | `fall.*` |
| 异常声音 | RMS 能量超绝对阈值(-28dBFS)且高于底噪 12dB, 连续 2 窗 | `audio.*` |
| 告警 | LED 闪烁 + 警笛循环, 默认 10s; 跌倒未恢复期间持续告警 | `alarm.*` |
| 录像 | 事件前 4s 回溯 + 共 10s, mp4+wav 存 `records/日期/`, 索引 `events.jsonl` | `record.*` |

模型可换：`model.path` 支持任意 yolov8/yolo11-pose 的 `.rknn`（airockchip
116 通道导出）或 `.onnx`（标准 ultralytics 导出），后处理自动识别两种格式。

## 四、常见问题

- **rknn 加载失败/无 NPU**：确认 `librknnrt.so` 已拷贝、wheel 版本与
  python 匹配；`cat /sys/kernel/debug/rknpu/version` 看驱动版本。
- **没有告警声音**：`aplay assets/alarm.wav` 先测试；检查
  `aplay -l` 设备名填入 `alarm.speaker.device`。
- **误报多**：调高 `audio.abs_threshold_db`、`fall.min_trigger_frames`，
  或调高 `model.conf_threshold`。
- **摄像头打不开**：`v4l2-ctl --list-devices`，改 `camera.index`。
