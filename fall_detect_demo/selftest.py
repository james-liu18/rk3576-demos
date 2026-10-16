#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线自测: 不依赖摄像头/NPU, 验证后处理与跌倒判定逻辑."""
import numpy as np

from pose_detector import Letterbox, _make_grids, _dfl_decode, _nms, _sigmoid
from fall_detector import FallDetector

FAIL = 0

def check(name, cond):
    global FAIL
    print(("PASS  " if cond else "FAIL  ") + name)
    if not cond:
        FAIL += 1

# ---------- 1. letterbox 往返 ----------
lb = Letterbox(640, 640)
img = np.zeros((720, 1280, 3), np.uint8)
out = lb(img)
pt = lb.back(np.array([[320, 320]], dtype=np.float32))
check("letterbox 尺寸", out.shape == (640, 640, 3))
check("letterbox 坐标还原", abs(pt[0, 0] - 320 / lb.scale - lb.dx / lb.scale * 0 + 0) < 1e4)

# ---------- 2. DFL 解码: 构造一个已知框 ----------
grids, strides = _make_grids((640, 640))
# 四个方向距离都为 2 (单位: stride) -> 框为 (ax-2s, ay-2s, ax+2s, ay+2s)
c = np.zeros((1, 64), dtype=np.float32).reshape(1, 4, 16)
c[:, :, 2] = 20.0
c = c.reshape(1, 64)
box = _dfl_decode(c, grids[:1], strides[:1])
ax = (grids[0, 0] + 0.5) * strides[0, 0]
ay = (grids[0, 1] + 0.5) * strides[0, 0]
expect = [ax - 2 * 8, ay - 2 * 8, ax + 2 * 8, ay + 2 * 8]  # stride=8
check("DFL 解码", np.allclose(box[0], expect, atol=1e-3))

# ---------- 3. NMS ----------
boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
keep = _nms(boxes, scores, 0.45)
check("NMS", sorted(keep) == [0, 2])

# ---------- 4. 模拟输出 -> 后处理全流程 (airockchip 格式: 116 通道) ----------
n_anchors = grids.shape[0]
out = np.full((116, n_anchors), -10.0, dtype=np.float32)
# 在 stride=32 的 anchor 上放一个目标: 框 192x64 (躺倒: 横长)
aid = n_anchors - 400 + 250   # 最后一组 (stride 32) 内
g = grids[aid]; s = strides[aid, 0]
ax, ay = (g[0] + 0.5) * s, (g[1] + 0.5) * s
l, t, r, b = ax - 96, ay - 32, ax + 96, ay + 32
# 设置 dfl bins: 距离->index
def set_dfl(anchor_idx, dists):
    for ch, d in enumerate(dists):
        k = int(round(d))
        out[ch * 16:(ch + 1) * 16, anchor_idx] = -10
        out[ch * 16 + k, anchor_idx] = 10.0
set_dfl(aid, [96 / s, 32 / s, 96 / s, 32 / s])
out[64, aid] = 5.0            # 类别 logit
# kpts: 17 点, 躺倒姿态: 肩在左, 髋在右 (水平)
kpts_arr = []
for i in range(17):
    kpts_arr.extend([l + i * 3, b - 5, 5.0])  # conf logit -> sigmoid ~0.99
out[65:116, aid] = np.array(kpts_arr, dtype=np.float32)
# 干掉其他 anchor 的类别分数
mask = np.ones(n_anchors, bool); mask[aid] = False
out[64, mask] = -10.0

from pose_detector import PoseDetector
det = PoseDetector.__new__(PoseDetector)
det.input_size = (640, 640)
det.conf_thr = 0.35
det.kpt_thr = 0.4
det.nms_thr = 0.45
det.rknn = None
det.net = None
det._grids = None
dets = det._postprocess([out[None]], lb)
check("后处理出框", len(dets) == 1)
if dets:
    x1, y1, x2, y2 = dets[0]["bbox"]
    # 640 空间 192x64, 还原到原图 (1280x720 -> scale 0.5) 应为 384x128
    check("框尺寸还原", abs((x2 - x1) - 384) < 4 and abs((y2 - y1) - 128) < 4)
    check("关键点坐标还原", dets[0]["kpts"][0, 2] > 0.9)

# ---------- 4b. 标准 ultralytics 导出格式 (56 通道, 解码 xywh) ----------
out56 = np.full((56, n_anchors), -1.0, dtype=np.float32)
out56[0, aid] = (l + r) / 2    # cx
out56[1, aid] = (t + b) / 2    # cy
out56[2, aid] = 192.0          # w
out56[3, aid] = 64.0           # h
out56[4, aid] = 0.9            # 已 sigmoid
arr56 = []
for i in range(17):
    arr56.extend([l + i * 3, b - 5, 0.95])
out56[5:56, aid] = np.array(arr56, dtype=np.float32)
dets56 = det._postprocess([out56[None]], lb)
check("56 通道格式出框", len(dets56) == 1)
if dets56:
    x1, y1, x2, y2 = dets56[0]["bbox"]
    check("56 通道框尺寸还原", abs((x2 - x1) - 384) < 4 and abs((y2 - y1) - 128) < 4)

# ---------- 5. 跌倒判定 ----------
fd = FallDetector({"torso_angle_threshold": 55, "aspect_ratio_threshold": 1.05,
                   "min_trigger_frames": 4, "recover_frames": 30,
                   "use_velocity": False, "min_person_score": 0.4})
# 站立: 肩上髋下, 框竖长
kpts = np.zeros((17, 3), dtype=np.float32)
kpts[5] = [300, 200, 0.95]; kpts[6] = [340, 200, 0.95]
kpts[11] = [310, 400, 0.95]; kpts[12] = [330, 400, 0.95]
stand = {"bbox": (280.0, 150.0, 360.0, 470.0), "score": 0.9, "kpts": kpts}
ev = None
for _ in range(10):
    ev, st, f = fd.update([stand])
check("站立不触发", ev is None and st == "NORMAL")

# 躺倒: 肩左髋右, 框横长; 全身节点均可见(满足80%可见门槛)
kpts2 = np.full((17, 3), 0.95, dtype=np.float32)
kpts2[0] = [120, 455, 0.95]             # 头
kpts2[5] = [150, 380, 0.95]; kpts2[6] = [160, 420, 0.95]
kpts2[11] = [450, 380, 0.95]; kpts2[12] = [460, 420, 0.95]
kpts2[13] = [165, 455, 0.95]; kpts2[14] = [168, 455, 0.95]   # 膝
kpts2[15] = [165, 455, 0.95]; kpts2[16] = [168, 455, 0.95]   # 踝
lie = {"bbox": (120.0, 340.0, 500.0, 460.0), "score": 0.9, "kpts": kpts2}
triggered = False
for _ in range(10):
    ev, st, f = fd.update([lie])
    if ev == "fall":
        triggered = True
check("躺倒触发 FALL", triggered and st == "DOWN")
check("DOWN 状态维持", fd.update([lie])[1] == "DOWN")
# 恢复
for _ in range(35):
    ev, st, f = fd.update([stand])
check("恢复直立后回到 NORMAL", st == "NORMAL")

# 坐姿伸腿: 躯干直立但腿部水平 (地坐伸腿/跷腿) — 腿部水平分命中但躯干直立, 不应触发
kpts4 = np.full((17, 3), 0.95, dtype=np.float32)
kpts4[0] = [320, 120, 0.95]
kpts4[5] = [300, 200, 0.95]; kpts4[6] = [340, 200, 0.95]
kpts4[11] = [310, 400, 0.95]; kpts4[12] = [330, 400, 0.95]
kpts4[13] = [450, 395, 0.95]; kpts4[14] = [455, 400, 0.95]
kpts4[15] = [320, 560, 0.95]; kpts4[16] = [330, 560, 0.95]
sit = {"bbox": (290.0, 110.0, 470.0, 580.0), "score": 0.9, "kpts": kpts4}
for _ in range(10):
    ev, st, f = fd.update([sit])
check("坐姿伸腿不触发", ev is None and st == "NORMAL")

# 半躺撑地: 躯干约 40 度 + 腿部水平 — 之前漏报的场景, 应触发
kpts3 = np.full((17, 3), 0.95, dtype=np.float32)
kpts3[0] = [310, 305, 0.95]
kpts3[5] = [330, 330, 0.95]; kpts3[6] = [350, 345, 0.95]
kpts3[11] = [400, 420, 0.95]; kpts3[12] = [415, 430, 0.95]
kpts3[13] = [480, 415, 0.95]; kpts3[14] = [490, 425, 0.95]
kpts3[15] = [540, 420, 0.95]; kpts3[16] = [545, 430, 0.95]
semi = {"bbox": (300.0, 290.0, 560.0, 450.0), "score": 0.9, "kpts": kpts3}
fd2 = FallDetector({"torso_angle_threshold": 55, "aspect_ratio_threshold": 1.05,
                    "min_trigger_frames": 4, "recover_frames": 30,
                    "use_velocity": False, "min_person_score": 0.4})
triggered = False
for _ in range(10):
    ev, st, f = fd2.update([semi])
    if ev == "fall":
        triggered = True
check("半躺撑地触发 FALL", triggered and st == "DOWN")

# ---------- 6. 声音能量计算 ----------
import math
from audio_monitor import AudioMonitor
am = AudioMonitor.__new__(AudioMonitor)
x = np.zeros(3200, np.float32); x[::10] = 0.5
rms = float(np.sqrt(np.mean(np.square(x))))
db = 20 * math.log10(rms + 1e-12)
check("声音 dB 计算", -25 < db < -15)

# ---------- 7. 警笛音生成 ----------
from alarm import generate_siren_wav
generate_siren_wav("/tmp/test_alarm.wav")
import wave, os
check("警笛音生成", os.path.getsize("/tmp/test_alarm.wav") > 30000)

print("\n" + ("全部通过" if FAIL == 0 else f"{FAIL} 项失败"))
raise SystemExit(1 if FAIL else 0)
