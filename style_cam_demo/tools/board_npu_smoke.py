#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""板上 NPU 崩溃定位: 同输入重复推理. 用法: board_npu_smoke.py MODEL H W [fmt]"""
import faulthandler
import sys
import time

import numpy as np

faulthandler.enable()
from rknnlite.api import RKNNLite

MODEL = sys.argv[1]
H, W = int(sys.argv[2]), int(sys.argv[3])
FMT = sys.argv[4] if len(sys.argv) > 4 else None

x = np.random.randint(0, 255, (1, H, W, 3), dtype=np.uint8)

rk = RKNNLite()
print('load:', rk.load_rknn(MODEL), flush=True)
print('init:', rk.init_runtime(), flush=True)

kw = {'data_format': [FMT]} if FMT else {}
for i in range(5):
    t0 = time.time()
    out = rk.inference(inputs=[x], **kw)
    if out is None:
        print('iter %d FAILED' % i, flush=True)
        break
    print('iter %d ok %.1fms out0=%s' % (
        i, (time.time() - t0) * 1000, np.asarray(out[0]).shape), flush=True)
rk.release()
print('PASS', flush=True)
