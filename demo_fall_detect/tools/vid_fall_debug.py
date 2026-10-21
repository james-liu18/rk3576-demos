#!/usr/bin/env python3
"""视频版跌倒特征诊断: 逐帧打印 检出数/可见度/打分/倾角."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import yaml
from pose_detector import PoseDetector
from fall_detector import FallDetector

src = sys.argv[1]
with open("config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

mp = cfg["model"]["path"]
if not mp.startswith("/"):
    import os
    mp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), mp)
det = PoseDetector(mp,
                   input_size=tuple(cfg["model"].get("input_size", [640, 640])),
                   conf_threshold=cfg["model"].get("conf_threshold", 0.35),
                   kpt_threshold=cfg["model"].get("kpt_threshold", 0.4),
                   nms_threshold=cfg["model"].get("nms_threshold", 0.45))
fd = FallDetector(cfg.get("fall", {}))

cap = cv2.VideoCapture(src)
i = 0
triggered = None
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if os.environ.get("UPSCALE"):
        frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    dets, ms = det.infer(frame)
    event, state, fallen = fd.update(dets, frame.shape[:2])
    if not dets:
        print(f"[{i:3d}] NO-DET")
        i += 1
        continue
    main = max(dets, key=lambda d: d["score"])
    is_f, hip, info = fd._person_features(main, frame.shape[:2])
    mark = ""
    if event == "fall" and triggered is None:
        triggered = i
        mark = " <<< FALL EVENT"
    if i % 5 == 0 or mark or state == "DOWN":
        print(f"[{i:3d}] n={len(dets)} state={state} fallen={int(is_f)} "
              f"sc={main['score']:.2f} vis={info.get('kpt_vis_ratio', 0):.2f} "
              f"fs={info.get('fall_score', 0):4.1f} ang={info.get('torso_angle', -1):5.1f} "
              f"bbox={main['bbox'][2]-main['bbox'][0]:.0f}x{main['bbox'][3]-main['bbox'][1]:.0f}{mark}")
    i += 1
print(f"triggered_at={triggered}")
det.release()
