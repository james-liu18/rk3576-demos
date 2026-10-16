#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PC 视觉验证贴纸: 用 onnx pose 模型对测试图贴狗头/猪头 (Windows 可跑).

用法: python tools/test_sticker_pc.py [IMG]
"""
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stylizer import PoseSticker  # noqa: E402

img_path = sys.argv[1] if len(sys.argv) > 1 else "tools/test_person.jpg"
img = cv2.imread(img_path)
assert img is not None, f"读图失败: {img_path}"

st = PoseSticker("models/onnx/yolov8n-pose.onnx", "assets")
os.makedirs("test_out", exist_ok=True)
name = os.path.splitext(os.path.basename(img_path))[0]
for effect in ("dog", "pig", "rabbit", "clown"):
    out, hit = st.apply(img, effect)
    p = f"test_out/{name}_sticker_{effect}.jpg"
    cv2.imwrite(p, out)
    print(f"{effect}: 命中人脸={hit} -> {p}")
