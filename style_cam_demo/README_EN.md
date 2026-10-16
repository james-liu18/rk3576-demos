# style_cam_demo — Camera Preview + Style Effects + Photo Capture (RK3576)

Second demo built on the fall_detect_demo baseline: USB camera streaming preview,
real-time style effects, one-key photo capture. **All effects run on the NPU**
(style transfer + human keypoint stickers), no CPU filter effects.

| Effect | Mode name | Source model |
|--------|-----------|--------------|
| Miyazaki style | hayao | CartoonGAN Hayao (Yijunmaverick, 160 input) |
| Oil painting | oil | fast-neural-style rain-princess (ONNX model zoo, native 224) |
| Candy | candy | fast-neural-style candy (ONNX model zoo) |
| Glass mosaic | mosaic | fast-neural-style mosaic (ONNX model zoo) |
| Pencil sketch | sketch | CycleGAN photo2sketch (HF fracta1l, lightweight ngf32) |
| Ukiyo-e | ukiyoe | CycleGAN photo->ukiyoe (HF Pim66070135) |
| Ink wash painting | ink | CIPGAN (CycleGAN Chinese ink, .pth self-exported to onnx) |
| Dog head | dog | yolo_pose keypoints + dog ear/nose stickers |
| Pig head | pig | yolo_pose keypoints + pig ear/nose stickers |
| Rabbit head | rabbit | yolo_pose keypoints + rabbit ears/nose/whiskers stickers |
| Clown face | clown | yolo_pose keypoints + wig/eye makeup/red nose/mouth stickers |

Missing models auto-hide their modes without affecting others, so **the board
needs no extra environment** (reuses fall_detect_demo's /opt/py312 +
numpy/opencv/pyyaml/pillow).

## Directory Structure

```
style_cam_demo/
├── main.py                 # Main program (streaming / stylizing / capture threads)
├── stylizer.py             # NPU style models + keypoint stickers (all-NPU, no CPU effects)
├── ui.py                   # UI: PySide6 dark theme (preferred) / Tk / OpenCV window / headless fallback chain
├── config.yaml             # Configuration (camera / effects / saving)
├── run.sh                  # On-board launcher wrapper (for root desktop sessions)
├── selftest.py             # Offline self-test (no camera/board needed)
├── models/                 # Pre-converted rknn (fp16, 192x192 input, oil 224)
│   └── onnx/               # Original onnx, for PC debugging / re-conversion
├── assets/                 # Sticker PNGs (OpenCV drawing fallback if missing)
└── tools/                  # Scripts used on the build PC
    ├── fetch_style_models2.sh   # Download 6 style onnx models (HF)
    ├── export_ink_onnx.py       # CIPGAN ink .pth -> onnx
    ├── fuse_instance_norm.py    # Decomposed InstanceNorm -> native ops (critical!)
    ├── convert_style.py         # onnx -> rknn (with graph surgery)
    └── convert_all2.sh          # One-key batch conversion of 8 style models
```

## Usage (on board)

Deploy to /opt/style_cam_demo first (see deployment steps below):

```bash
cd /opt/style_cam_demo
./run.sh                     # with UI (effect dropdown, space/button to capture)
./run.sh --no-ui             # headless
./run.sh --list-styles       # list available effects
```

Capture: click "Capture" or press space in the UI -> full-resolution stylized
frame saved to `photos/style_<timestamp>.jpg`. Set `save.save_original: true`
to also save the original frame.

## Camera

Default `camera.index: 0`. After swapping cameras, confirm the index with
`v4l2-ctl --list-devices`, or run `python3 main.py --camera N`.

## Deployment

Prerequisite: install the Python environment via fall_detect_demo's
`deploy/board_setup.sh` (/opt/py312 + deps; style_cam adds nothing). Then either:

```bash
# A: git clone (board has internet)
git clone https://github.com/james-liu18/rk3576-demos.git
sudo cp -r rk3576-demos/style_cam_demo /opt/
sudo chmod +x /opt/style_cam_demo/run.sh

# B: adb push (from Windows)
adb push style_cam_demo /opt/
adb shell "chmod +x /opt/style_cam_demo/run.sh"
```

Verify:

```bash
adb shell "cd /opt/style_cam_demo && /opt/py312/bin/python3.12 selftest.py"
```

### PySide6 UI (optional, modern dark theme)

The default Tk UI is plain; once PySide6 is installed it automatically switches
to a Qt dark UI (effect grid / capture flash / thumbnails). Falls back to Tk
when absent, no impact on running:

```bash
adb shell "apt-get install -y libxcb-cursor0"
adb shell "/opt/py312/bin/python3.12 -m pip install --no-cache-dir \
    -i https://pypi.org/simple shiboken6 PySide6_Essentials"
```

run.sh sets `QT_QPA_PLATFORM=xcb` (root borrows the desktop user's XWayland to open windows).

## PC / Build-machine Debugging (no board needed)

Point the model path to a `.onnx` file to use OpenCV DNN (see
config_onnx_debug.yaml):

```bash
python3 main.py --config config_onnx_debug.yaml --test-style any.jpg  # all effects on a static image
python3 main.py --config config_onnx_debug.yaml                       # real-time from camera
```

## Re-converting Models (build PC)

```bash
bash tools/fetch_style_models2.sh          # download 6 style onnx (HuggingFace)
python3 tools/export_ink_onnx.py           # CIPGAN ink: requires cloning /tmp/CIPGAN first
bash tools/convert_all2.sh                 # batch onnx -> rknn
```

### Conversion Notes (hard-won lessons, do not delete)

- **Toolkit version**: use a dedicated venv with rknn-toolkit2 **2.3.2**
  (convert_all2.sh creates it automatically). Both 2.5.0 and 2.3.2 have a
  bookkeeping bug fusing the TF-exported decomposed LayerNorm (build-time
  KeyError `..._tp_rs#1_norm_rs_mul`). On-board librknnrt 2.5.0 is backward
  compatible with 2.3.2 artifacts.
- **Decomposed InstanceNorm fusion**: convert_style.py calls
  `fuse_instance_norm.py` after onnxsim, rewriting the TF-exported 12-nodes-per-group
  decomposed LayerNorm into native InstanceNormalization (adopted only if
  numerical check max diff < 1e-2). This is the key to a successful build.
- **Boundary-truncating Transpose redirection**: the hayao-family TF-exported
  graph is `input -> MirrorPad -> Transpose -> ...`. After truncating the
  boundary Transpose, consumers must be redirected to the **Pad's output**
  (not the original input), otherwise the Pad is bypassed and feature maps
  collapse (256 input yields 248 output, misaligned image edges).
- Graph surgery (fix_graph): reflect/edge Pad -> constant, Relu -> LeakyRelu(**1e-6**).
  Note alpha must NOT be 1.0 — LeakyRelu(1.0) is an identity map, deleting the
  ReLU nonlinearity entirely (once ruined ink/oil outputs); 1e-6 approximates
  ReLU numerically while dodging the NPU's ReLU-fusion-outputs-zero bug.
  Without surgery the on-board NPU would: corrupt memory on reflect pad
  (segfault on second inference), output all zeros via ReLU fusion.
- **TF-exported AnimeGANv2 (LayerNorm semantics) is unusable**: the official TF
  version's "LayerNorm" normalizes over C+H+W jointly, semantically different
  from InstanceNorm (H,W); blindly replacing it destroyed outputs.
  MeanVarianceNormalization / native LayerNormalization / fusion-disabling
  rules were all verified infeasible on RK3576.
- **CartoonGAN's custom InstanceNormalization must be replaced before export**
  (2026-10-16): Yijunmaverick's InstanceNormalization hand-implements
  view/mean/var/expand inside `__call__`, tracing ~30 nodes of dynamic shapes
  per group (Shape/Gather/Expand); numerically correct in the onnx simulator
  but on-board NPU `failed to submit` (Sub:/Sub, 6.6s/frame). Fix: monkeypatch
  to `F.instance_norm(x, None, None, scale, shift)` before export (parameter
  names scale/shift preserved, state dict loads directly; biased variance,
  same semantics). After export: 87 nodes with 21 native InstanceNormalization,
  works on board.
- **AnimeGANv2 (bryandlee) portrait quality was poor, replaced** (2026-10-16):
  celeba_distill had obvious face blur (small model capacity + 192 input below
  training resolution); the 3 modes were replaced with CartoonGAN (better
  director-style brushwork/tones, no blur); input 160 (~35% faster than 192,
  cartoon flat-color style is resolution-insensitive).
- **Always resize input to model input size first**: RKNNLite auto-resizes
  mismatched inputs with poor interpolation (1280x720->256 aliasing), causing
  horizontal stripes/color-block artifacts on board; correct way is
  cv2.resize(INTER_LINEAR/AREA) before inference (built into stylizer.py).
- Input 192x192 (measured 256->192 ~1.8x faster across the board, acceptable
  quality loss by eye; oil keeps native 224); normalization mean/std=127.5
  baked into the rknn, feed RGB uint8 directly on board; output [-1,1]
  (zoo family is 0~255).
- fp16, not quantized. int8 measured **not worth it** (2026-10-16): candy
  int8@256 74ms is slower than fp16@192 (60ms); anime-family int8 175ms vs
  fp16@192 (149ms), InstanceNorm layers fall back to CPU and become the
  bottleneck. auto-hybrid also gained nothing.
- core_mask=3 (dual-core NPU) is ~10% faster than default.
- fast-neural-style from the model zoo uses `--norm imagenet`;
  AnimeGANv2/CycleGAN family (sketch/ukiyoe/ink) uses `--norm anime`.

## Performance Reference (RK3576, librknnrt 2.5.0, measured on board 2026-10-16)

| Mode | Per-frame | Real-time fps |
|------|-----------|---------------|
| Preview (raw) | - | 25 fps (1280x720 MJPG) |
| sketch | 43 ms | 17.1 |
| candy/mosaic | 62~66 ms | 11.6~12.6 |
| oil (224) | 88 ms | 9.1 |
| hayao (CartoonGAN, 160) | 118 ms | 6.7 |
| ukiyoe / ink | ~152 ms | 5.5 |
| dog/pig/rabbit/clown stickers | 52~58 ms | 14+ |

( At 256 input: anime family 300ms/2.9fps, ink 345ms/2.5fps, zoo family
  137ms/6fps; 192 input roughly halved everything. CartoonGAN was tried at
  192 input: 173ms/4.7fps, dropping to 160 gives 118ms/6.8fps with near-lossless
  quality by eye )

## On-board Validation Record (2026-10-16)

- selftest.py all passed; --list-styles lists 12 modes (raw + 7 styles + 4 stickers).
- main.py --test-style renders all effects correctly (visual check passed);
  board_rt_test.py runs all effects in real time stably.
- Model list: models/{hayao, oil, candy, mosaic, sketch, ukiyoe, ink,
  yolo_pose}.rknn. hayao = CartoonGAN director style (HF
  yash0311dev/cartoonify-gan-weights; Shinkai/Paprika/Hosoda weights kept but
  not deployed); sketch = CycleGAN photo2sketch (HF fracta1l, lightweight ngf32);
  ukiyoe = CycleGAN photo->ukiyoe (HF Pim66070135).
- Rabbit/clown heads are pure OpenCV-drawn stickers (positioned by yolo_pose
  keypoints), no asset files needed; use tools/test_sticker_pc.py for visual
  checks on PC.
- Conversion toolchain: build PC /tmp/rknn232venv (rknn-toolkit2 2.3.2),
  simulator comparison via tools/sim_full2.py; single-model rendering on board
  via tools/board_one_style2.py (explicit pre-scaling), real-time regression
  via tools/board_rt_test.py.
