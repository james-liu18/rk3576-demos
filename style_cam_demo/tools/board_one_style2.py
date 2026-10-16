#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""板上单模型出图 v2: 显式预缩放输入, 排除 RKNNLite 自动缩放假设.
board_one_style2.py MODEL.rknn IN.jpg OUT.jpg SIZE [range]
range: pm1 (默认) / image255"""
import sys

import cv2
import numpy as np
from rknnlite.api import RKNNLite

model, img_path, out_path, size = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
out_range = sys.argv[5] if len(sys.argv) > 5 else "pm1"

img = cv2.imread(img_path)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)

rk = RKNNLite()
print("load:", rk.load_rknn(model), flush=True)
print("init:", rk.init_runtime(), flush=True)
out = rk.inference(inputs=[rgb[None].astype(np.uint8)])
o = np.asarray(out).squeeze().astype(np.float32)
print(f"out shape={o.shape} range=[{o.min():.1f},{o.max():.1f}] "
      f"mean={o.mean():.1f}", flush=True)
if o.ndim == 3 and o.shape[0] == 3:
    o = o.transpose(1, 2, 0)
if out_range == "pm1":
    o = (o + 1.0) * 127.5
o = np.clip(o, 0, 255).astype(np.uint8)
cv2.imwrite(out_path, cv2.cvtColor(o, cv2.COLOR_RGB2BGR))
print("saved", out_path, flush=True)
rk.release()
