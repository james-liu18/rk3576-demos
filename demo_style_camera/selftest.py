#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线自测: 不依赖摄像头/NPU, 验证特效注册/回退/贴纸绘制与拍照保存链路.

用法: python3 selftest.py   (板上跑会额外加载 rknn, 板外自动降级)
"""
import os
import shutil

import cv2
import numpy as np

from stylizer import (Stylizer, _draw_dog_ears, _draw_dog_nose,
                      _draw_pig_ears, _draw_pig_nose,
                      _draw_rabbit_ears, _draw_rabbit_nose,
                      _draw_clown_hair, _draw_clown_eyes,
                      _draw_clown_nose, _draw_clown_mouth)

FAIL = 0


def check(name, cond):
    global FAIL
    print(("PASS  " if cond else "FAIL  ") + name)
    if not cond:
        FAIL += 1


STICKER_CFG = {"pose_model": "models/yolo_pose.rknn", "assets_dir": "assets"}

# ---------- 1. 模式列表 ----------
st = Stylizer({"input_size": [256, 256], "models": {}})
check("无模型无贴纸时只有 原始", [k for k, _ in st.available_modes()] == ["none"])

st = Stylizer({"input_size": [256, 256], "models": {}}, STICKER_CFG)
keys = [k for k, _ in st.available_modes()]
check("配置 yolo_pose 后出现 狗头/猪头/兔头/小丑头",
      all(k in keys for k in ("dog", "pig", "rabbit", "clown")))
check("模式 key 唯一", len(keys) == len(set(keys)))

# ---------- 2. 贴纸绘制函数 (纯 CPU, 直接验证) ----------
img = np.zeros((480, 640, 3), np.uint8)
_draw_dog_ears(img, (200, 150), (280, 150), 30)
_draw_dog_nose(img, 240, 260, 20)
_draw_pig_ears(img, (360, 150), (440, 150), 35)
_draw_pig_nose(img, 400, 260, 25)
_draw_rabbit_ears(img, (100, 150), (160, 150), 80)
_draw_rabbit_nose(img, 130, 260, 80)
_draw_clown_hair(img, (520.0, 200.0), 80)
_draw_clown_eyes(img, (500.0, 230.0), (540.0, 230.0), 80)
_draw_clown_nose(img, 520, 270, 80)
_draw_clown_mouth(img, 520, 310, 80)
check("OpenCV 绘制狗/猪/兔/小丑贴纸", img.sum() > 0)

# ---------- 3. 贴纸模式推理 (板外无 rknn 自动回退原帧) ----------
img = np.random.randint(0, 255, (480, 640, 3), np.uint8)
for key in ("dog", "pig", "rabbit", "clown"):
    out, ms = st.stylize(img, key)
    check(f"贴纸模式 {key} 可调用 ({ms:.0f}ms, 板外回退原帧)",
          out.shape == img.shape)

# ---------- 4. 不存在的 NPU 模式自动回退原帧 ----------
st3 = Stylizer({"input_size": [256, 256],
                "models": {"ghost": "models/__no_such__.rknn"}}, STICKER_CFG)
out, ms = st3.stylize(img, "ghost")
check("缺失模型回退原帧", np.array_equal(out, img))
check("缺失模型不在模式列表",
      "ghost" not in [k for k, _ in st3.available_modes()])

# ---------- 5. strength 混合 (注入假模型验证) ----------
class _Fake:
    def run(self, frame):
        return np.full_like(frame, 255)

    def release(self):
        pass


st2 = Stylizer({"input_size": [256, 256], "strength": 0.5,
                "models": {"fake": "models/fake_style.rknn"}})
st2._models["fake"] = _Fake()          # 绕过文件加载
out, _ = st2.stylize(img, "fake")
mean = float(out.mean())
exp = (255.0 + float(img.mean())) / 2.0
check(f"strength=0.5 混合生效 (期望均值 {exp:.0f}±10)",
      abs(mean - exp) < 10.0)

# ---------- 6. 拍照保存链路 ----------
save_dir = "selftest_photos"
os.makedirs(save_dir, exist_ok=True)
styled, _ = st.stylize(img, "none")
p = os.path.join(save_dir, "style_test.jpg")
cv2.imwrite(p, styled, [cv2.IMWRITE_JPEG_QUALITY, 95])
ok = os.path.exists(p) and os.path.getsize(p) > 1000
check("JPEG 保存", ok)
back = cv2.imread(p)
check("JPEG 读回尺寸一致", back is not None and back.shape == img.shape)
shutil.rmtree(save_dir, ignore_errors=True)

# ---------- 7. onnx 模型 (有则测推理, 没有则跳过) ----------
onnx_files = []
for d in ("models/onnx",):
    if os.path.isdir(d):
        onnx_files = [os.path.join(d, f) for f in sorted(os.listdir(d))
                      if f.endswith(".onnx")]
if onnx_files:
    # 排除 pose 检测类 onnx (输出非图像, 不适合走风格迁移路径)
    onnx_files = [f for f in onnx_files
                  if "pose" not in os.path.basename(f).lower()
                  and "yolo" not in os.path.basename(f).lower()]
if onnx_files:
    st4 = Stylizer({"input_size": [256, 256],
                    "models": {"test_onnx": onnx_files[0]}})
    check("onnx 模式出现在列表",
          "test_onnx" in [k for k, _ in st4.available_modes()])
    out, ms = st4.stylize(img, "test_onnx")
    check(f"onnx 风格迁移推理 ({ms:.0f}ms)",
          out.shape == img.shape and not np.array_equal(out, img))
else:
    print("SKIP  onnx 模型未下载 (models/onnx/ 为空), ONNX 路径未测")

print("\n" + ("全部通过" if FAIL == 0 else f"{FAIL} 项失败"))
raise SystemExit(1 if FAIL else 0)
