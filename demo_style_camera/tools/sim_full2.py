#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""编译机: 通用管线复刻 + 模拟器真实图推理.
用法: sim_full2.py ONNX OUT.rknn SIZE SIZE NORM(anime|imagenet) IN.jpg OUT.jpg
复刻 convert_one (trunc/fix/sim/fuse) 后 build, 模拟器跑真实图."""
import os
import sys

import numpy as np
import onnx
from PIL import Image

sys.path.insert(0, "tools")
import convert_style as cs
from fuse_instance_norm import fuse_instance_norm

ONNX, OUT = sys.argv[1], sys.argv[2]
H, W = int(sys.argv[3]), int(sys.argv[4])
NORM = sys.argv[5]
IN_JPG, OUT_JPG = sys.argv[6], sys.argv[7]
pm1 = NORM == "anime"

img = Image.open(IN_JPG).convert("RGB").resize((W, H))
rgb = np.asarray(img, dtype=np.uint8)

dh, dw, layout = cs.onnx_input_info(ONNX)
src = ONNX
outputs = None
if (dh, dw) != (H, W):
    patched = ONNX + f".{H}x{W}.onnx"
    cs.patch_input_size(ONNX, patched, H, W)
    src = patched
if layout == "nhwc":
    trunc, nchw_out = cs.truncate_nchw_boundary(src, src + ".trunc.onnx", H, W)
    if trunc:
        src = trunc
        outputs = [nchw_out]

from rknn.api import RKNN
rk = RKNN(verbose=False)
mean, std = (cs.MEAN_ANIME, cs.STD_ANIME) if pm1 \
    else (cs.MEAN_IMAGENET, cs.STD_IMAGENET)
rk.config(mean_values=mean, std_values=std, target_platform="rk3576",
          disable_rules=["fuse_pad_into_conv", "fuse_align_conv_forward",
                         "fuse_align_conv_backward",
                         "remove_parallel_transpose",
                         "absorb_gather_into_attention_conv"])
fixed = src + ".fixed.onnx"
cs.fix_graph(src, fixed)
import onnxsim
sim, ok = onnxsim.simplify(onnx.load(fixed))
assert ok
fixed = src + ".fixed.sim.onnx"
onnx.save(sim, fixed)
fuse_instance_norm(fixed, fixed + ".inorm.onnx")
if os.path.exists(fixed + ".inorm.onnx"):
    fixed = fixed + ".inorm.onnx"  # 无可融合时沿用原文件 (原生 InstanceNorm)

assert rk.load_onnx(model=fixed, outputs=outputs) == 0, "load_onnx"
assert rk.build(do_quantization=False) == 0, "build"
assert rk.init_runtime() == 0, "init"     # 模拟器
out = rk.inference(inputs=[rgb[None].astype(np.uint8)])
o = np.asarray(out).squeeze().astype(np.float32)
print(f"sim out shape={o.shape} range=[{o.min():.3f},{o.max():.3f}] "
      f"mean={o.mean():.3f}", flush=True)
if o.ndim == 3 and o.shape[0] == 3:
    o = o.transpose(1, 2, 0)
if pm1:
    o = (o + 1.0) * 127.5
o = np.clip(o, 0, 255).astype(np.uint8)
Image.fromarray(o).save(OUT_JPG)
print("saved", OUT_JPG, flush=True)
rk.release()
