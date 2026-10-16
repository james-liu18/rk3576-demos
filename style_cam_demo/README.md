# style_cam_demo — 摄像头预览 + 风格化特效 + 拍照存储 (RK3576)

基于 fall_detect_demo 基线的第二个 demo：USB 摄像头取流预览，实时风格化特效，
一键拍照保存到本地。**全部特效跑在 NPU 上**（风格迁移 + 人体关键点贴纸），
无 CPU 滤镜特效。

| 特效 | 模式名 | 来源模型 |
|------|--------|----------|
| 宫崎骏风 | hayao | CartoonGAN Hayao (Yijunmaverick, 160 输入) |
| 油画 | oil | fast-neural-style rain-princess (ONNX model zoo, 原生 224) |
| 糖果风 | candy | fast-neural-style candy (ONNX model zoo) |
| 玻璃马赛克 | mosaic | fast-neural-style mosaic (ONNX model zoo) |
| 素描 | sketch | CycleGAN photo2sketch (HF fracta1l, ngf32 轻量) |
| 浮世绘 | ukiyoe | CycleGAN photo->ukiyoe (HF Pim66070135) |
| 水墨画 | ink | CIPGAN (CycleGAN 中国水墨, .pth 自行导出 onnx) |
| 狗头 | dog | yolo_pose 关键点 + 狗耳狗鼻贴纸 |
| 猪头 | pig | yolo_pose 关键点 + 猪耳猪鼻贴纸 |
| 兔头 | rabbit | yolo_pose 关键点 + 兔耳兔鼻胡须贴纸 |
| 小丑头 | clown | yolo_pose 关键点 + 假发/眼妆/红鼻/红嘴贴纸 |

模型缺失时对应模式自动隐藏，不影响其他功能，所以**板子不需要任何额外环境**
（复用 fall_detect_demo 的 /opt/py312 + numpy/opencv/pyyaml/pillow 即可）。

## 目录结构

```
style_cam_demo/
├── main.py                 # 主程序 (取流/风格化/拍照线程)
├── stylizer.py             # NPU 风格模型 + 关键点贴纸封装 (全 NPU, 无 CPU 特效)
├── ui.py                   # UI: PySide6 深色主题 (首选) / Tk / OpenCV 窗口 / 无界面 多级回退
├── config.yaml             # 配置 (摄像头/特效/保存)
├── run.sh                  # 板上启动包装 (root 开窗口用)
├── selftest.py             # 离线自测 (无需摄像头/板子)
├── models/                 # 已转好的 rknn (fp16, 192x192 输入, oil 224)
│   └── onnx/               # 原始 onnx, 供 PC 调试 / 重新转换
├── assets/                 # 贴纸 PNG (缺失时代码用 OpenCV 画)
└── tools/                  # 编译机 (10.66.84.59) 上使用的脚本
    ├── fetch_style_models2.sh   # 下载 6 个风格 onnx (HF)
    ├── export_ink_onnx.py       # CIPGAN 水墨 .pth -> onnx
    ├── fuse_instance_norm.py    # 分解式 InstanceNorm -> 原生算子 (关键!)
    ├── convert_style.py         # onnx -> rknn (含图手术)
    └── convert_all2.sh          # 一键批量转换 8 个风格模型
```

## 使用 (板上)

部署到 /opt/style_cam_demo 后 (见下方部署步骤):

```bash
cd /opt/style_cam_demo
./run.sh                     # 带界面 (特效下拉框切换, 空格/按钮拍照)
./run.sh --no-ui             # 无界面
./run.sh --list-styles       # 列出可用特效
```

拍照: UI 点"拍照"按钮或按空格 → 全分辨率风格化帧存 `photos/style_时间戳.jpg`。
`save.save_original: true` 可同时存一张原图。

## 摄像头

默认 `camera.index: 0`。换摄像头后用 `v4l2-ctl --list-devices` 确认序号，
或 `python3 main.py --camera N`。

## 部署步骤

前置: 先按 fall_detect_demo 的 `deploy/board_setup.sh` 装好 Python 环境
(/opt/py312 + 依赖, style_cam 无新依赖)。然后任选其一:

```bash
# 方式 A: git clone (板子可联网)
git clone https://github.com/james-liu18/rk3576-demos.git
sudo cp -r rk3576-demos/style_cam_demo /opt/
sudo chmod +x /opt/style_cam_demo/run.sh

# 方式 B: adb push (Windows)
adb push style_cam_demo /opt/
adb shell "chmod +x /opt/style_cam_demo/run.sh"
```

验证:

```bash
adb shell "cd /opt/style_cam_demo && /opt/py312/bin/python3.12 selftest.py"
```

### PySide6 界面 (可选, 现代深色主题)

默认 Tk 界面较朴素; 装上 PySide6 后自动切换为 Qt 深色界面
(特效网格/拍照白闪/缩略图), 未安装时自动回退 Tk, 不影响运行:

```bash
adb shell "apt-get install -y libxcb-cursor0"
adb shell "/opt/py312/bin/python3.12 -m pip install --no-cache-dir \
    -i https://pypi.org/simple shiboken6 PySide6_Essentials"
```

run.sh 已设置 `QT_QPA_PLATFORM=xcb` (root 借用普通用户 XWayland 出窗口)。

## PC / 编译机调试 (无需板子)

模型路径改成 `.onnx` 即走 OpenCV DNN（见 config_onnx_debug.yaml）：

```bash
python3 main.py --config config_onnx_debug.yaml --test-style any.jpg  # 静态图全特效
python3 main.py --config config_onnx_debug.yaml                       # 摄像头实时
```

## 重新转换模型 (编译机 10.66.84.59)

```bash
bash tools/fetch_style_models2.sh          # 下载 6 个风格 onnx (HuggingFace)
python3 tools/export_ink_onnx.py           # CIPGAN 水墨: 需先 clone /tmp/CIPGAN
bash tools/convert_all2.sh                 # 批量 onnx -> rknn
```

### 转换要点 (踩过的坑, 勿删)

- **toolkit 版本**: 用 rknn-toolkit2 **2.3.2** 独立 venv
  (`/tmp/rknn232venv`, convert_all2.sh 自动创建)。2.5.0/2.3.2 对 TF 导出的
  分解式 LayerNorm 融合都有簿记 bug (build 时 KeyError `..._tp_rs#1_norm_rs_mul`)。
  板上 librknnrt 2.5.0 向后兼容 2.3.2 产物。
- **分解式 InstanceNorm 融合**: convert_style.py 在 onnxsim 之后调用
  `fuse_instance_norm.py`，把 TF 导出的 12 节点/组的分解式 LayerNorm 重写成
  原生 InstanceNormalization (数值校验 max diff < 1e-2 才采用)。这是能否
  build 成功的关键。
- **截边界 Transpose 重定向**: hayao 系 TF 导出图为
  `输入 -> MirrorPad -> Transpose -> ...`，截掉边界 Transpose 后消费者必须
  重定向到 **Pad 的输出** (而不是原始输入)，否则 Pad 被架空、特征图塌缩
  (256 输入输出只有 248, 画面边缘错位)。
- 图手术 (fix_graph): reflect/edge Pad → constant、Relu → LeakyRelu(**1e-6**)。
  注意 alpha 不能用 1.0 —— LeakyRelu(1.0) 是恒等映射, 等于把 ReLU 非线性整个
  删掉 (曾导致 ink/oil 输出全毁); 1e-6 数值上近似 ReLU, 又躲开 NPU 的
  ReLU 融合输出恒 0 bug。不做手术板端 NPU 会: reflect pad 内存损坏
  (第二次推理段错误)、ReLU 融合输出恒 0。
- **TF 导出的 AnimeGANv2 (LayerNorm 语义) 不可用**: 官方 TF 版的 "LayerNorm"
  是对 C+H+W 整体归一化, 与 InstanceNorm (H,W) 语义不同, 误替换成
  InstanceNorm 后输出全毁。MeanVarianceNormalization /
  原生 LayerNormalization / 禁融合规则等路线均已在 RK3576 上验证不可行。
- **CartoonGAN 自定义 InstanceNormalization 必须先替换再导出** (2026-10-16):
  Yijunmaverick 仓库的 InstanceNormalization 在 `__call__` 里用
  view/mean/var/expand 手工实现, trace 出 ~30 节点/组的动态 Shape 图
  (Shape/Gather/Expand), onnx 模拟器数值正确但板端 NPU `failed to submit`
  (Sub:/Sub, 6.6s/帧)。修法: 导出前 monkeypatch 成
  `F.instance_norm(x, None, None, scale, shift)` (参数名保持 scale/shift,
  state dict 直接加载; 有偏方差语义一致), 导出后 87 节点含 21 个原生
  InstanceNormalization, 板端正常。
- **AnimeGANv2 (bryandlee) 人像效果差被替换** (2026-10-16): celeba_distill
  糊脸明显 (模型容量小 + 192 输入低于训练分辨率), 3 个模式整体换为
  CartoonGAN (导演风格笔触/色调好, 无糊脸); 输入 160 (比 192 快 ~35%,
  卡通扁平色块风格对分辨率不敏感)。
- **喂图必须先缩放到模型输入尺寸**: RKNNLite 对尺寸不符的输入会自动缩放,
  其插值质量差 (1280x720→256 混叠), 板端出横条纹/色块伪影; 正确做法是
  先 cv2.resize(INTER_LINEAR/AREA) 再 inference (stylizer.py 已内置)。
- 输入 192x192 (全线实测 256→192 快 ~1.8x, 目检质量损失可接受; oil 保持
  原生 224)；归一化 mean/std=127.5 写进 rknn，板端直接喂 RGB uint8；
  输出 [-1,1] (zoo 系为 0~255)。
- fp16 非量化。int8 实测**不划算** (2026-10-16): candy int8@256 74ms 反而
  慢于 fp16@192 (60ms); anime 系 int8 175ms 慢于 fp16@192 (149ms),
  InstanceNorm 层回退 CPU 反成瓶颈。auto-hybrid 亦无收益。
- core_mask=3 (双核 NPU) 比默认快 ~10%。
- 模型 zoo 的 fast-neural-style 用 `--norm imagenet`；AnimeGANv2/CycleGAN 系
  (bryandlee/sketch/ukiyoe/ink) 用 `--norm anime`。

## 性能参考 (RK3576, librknnrt 2.5.0, 2026-10-16 板上实测)

| 模式 | 单帧耗时 | 实时 fps |
|------|---------|---------|
| 预览 (原始) | - | 25 fps (1280x720 MJPG) |
| sketch 素描 | 43 ms | 17.1 |
| candy/mosaic 糖果/马赛克 | 62~66 ms | 11.6~12.6 |
| oil 油画 (224) | 88 ms | 9.1 |
| hayao 宫崎骏风 (CartoonGAN, 160) | 118 ms | 6.7 |
| ukiyoe 浮世绘 / ink 水墨画 | ~152 ms | 5.5 |
| dog/pig/rabbit/clown 贴纸 | 52~58 ms | 14+ |

( 256 输入时代: anime 系 300ms/2.9fps, ink 345ms/2.5fps, zoo 系 137ms/6fps;
  192 输入后全线约减半。CartoonGAN 曾试 192 输入: 173ms/4.7fps, 降至 160 后
  118ms/6.8fps, 画质目检几乎无损 )

## 板上验证记录 (2026-10-16)

- selftest.py 全部通过; --list-styles 列出 12 模式 (原始 + 7 风格 + 4 贴纸)。
- main.py --test-style 全特效出图正常目检通过; board_rt_test.py 实时跑
  全部特效稳定。
- 模型清单: models/{hayao, oil, candy, mosaic, sketch,
  ukiyoe, ink, yolo_pose}.rknn。
  hayao = CartoonGAN 导演风格 (HF yash0311dev/cartoonify-gan-weights,
  Shinkai/Paprika/Hosoda 权重备而未部署);
  sketch = CycleGAN photo2sketch (HF fracta1l, ngf32 轻量);
  ukiyoe = CycleGAN photo->ukiyoe (HF Pim66070135)。
- 兔头/小丑头为纯 OpenCV 绘制贴纸 (yolo_pose 关键点定位), 无需素材文件;
  PC 端用 tools/test_sticker_pc.py 目检。
- 转换工具链: 编译机 10.66.84.59 /tmp/rknn232venv (rknn-toolkit2 2.3.2),
  模拟器对比用 tools/sim_full2.py; 板上单模型出图 tools/board_one_style2.py
  (显式预缩放), 实时回归 tools/board_rt_test.py。
