"""实时打印跌倒特征 (坐姿标定用): 躯干倾角/框高宽比/关键点可见度/髋部下坠"""
import time
import cv2
from pose_detector import PoseDetector
from fall_detector import FallDetector
import yaml

with open("config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cap = cv2.VideoCapture(cfg["camera"]["index"], cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"].get("width", 1280))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"].get("height", 720))
cap.set(cv2.CAP_PROP_FPS, cfg["camera"].get("fps", 30))

det = PoseDetector(cfg["model"]["path"],
                   input_size=tuple(cfg["model"].get("input_size", [640, 640])),
                   conf_threshold=cfg["model"].get("conf_threshold", 0.35),
                   kpt_threshold=cfg["model"].get("kpt_threshold", 0.4),
                   nms_threshold=cfg["model"].get("nms_threshold", 0.45))
fd = FallDetector(cfg.get("fall", {}))

t0 = time.time()
n = 0
while time.time() - t0 < 15:
    ok, frame = cap.read()
    if not ok:
        continue
    n += 1
    dets, ms = det.infer(frame)
    if not dets:
        continue
    main = max(dets, key=lambda d: d["score"])
    is_fallen, hip, info = fd._person_features(main, frame.shape[:2])
    bbox = main["bbox"]
    print(f"[{time.time()-t0:5.1f}s] fallen={int(is_fallen)} "
          f"score={main['score']:.2f} vis_ratio={info.get('kpt_vis_ratio', 0):.2f} "
          f"fall_score={info.get('fall_score', 0):4.1f} "
          f"angle={info.get('torso_angle', -1):5.1f} "
          f"body_ratio={info.get('body_ratio', info.get('aspect_bbox', -1)):4.2f} "
          f"hdist={info.get('hdist', -1):5.2f} hip_drop={info.get('hip_drop', 0):6.1f} "
          f"bbox_wh=({bbox[2]-bbox[0]:.0f}x{bbox[3]-bbox[1]:.0f})")
det.release()
cap.release()
print(f"frames={n} done")
