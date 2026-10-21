#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""板上: 截断模型输出检查. 用法: board_trunc.py models/candy_trunc_66.rknn"""
import sys

import numpy as np
from rknnlite.api import RKNNLite

img = np.zeros((224, 224, 3), dtype=np.uint8)
img[..., 0] = np.linspace(0, 255, 224, dtype=np.uint8)[None, :]
img[..., 1] = np.linspace(255, 0, 224, dtype=np.uint8)[:, None]
img[..., 2] = 128

rk = RKNNLite()
print('load:', rk.load_rknn(sys.argv[1]), flush=True)
print('init:', rk.init_runtime(), flush=True)
for i in range(2):
    out = rk.inference(inputs=[img[None]])
    if out is None:
        print('iter %d FAILED' % i, flush=True)
        raise SystemExit(1)
    a = np.asarray(out[0]).astype(np.float32)
    print('iter %d shape=%s range=%.2f~%.2f mean=%.2f' % (
        i, a.shape, a.min(), a.max(), a.mean()), flush=True)
rk.release()
