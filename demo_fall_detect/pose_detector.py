#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO-Pose 推理封装.

支持两种运行时:
  - .rknn  -> rknnlite (RK3576 NPU)
  - .onnx  -> OpenCV DNN (PC/CPU 调试)

同时兼容两种模型导出格式 (自动检测):
  - airockchip 导出: 前 64 通道为 DFL 分类 bin, 类别分数为 logit
  - 标准 ultralytics 导出: 前 4 通道为解码后的 xywh (中心点格式), 分数已 sigmoid

输出统一为: [{bbox:(x1,y1,x2,y2), score, kpts:(17,3) 原图坐标}, ...]
"""
import time

import cv2
import numpy as np

COCO_KPT_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]

# COCO 19 条骨架连线
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


class Letterbox:
    """等比缩放 + 填充, 记录反变换参数."""

    def __init__(self, new_w, new_h):
        self.new_w, self.new_h = new_w, new_h

    def __call__(self, img):
        h, w = img.shape[:2]
        scale = min(self.new_w / w, self.new_h / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.new_h, self.new_w, 3), 114, dtype=np.uint8)
        self.dx = (self.new_w - nw) // 2
        self.dy = (self.new_h - nh) // 2
        self.scale = scale
        canvas[self.dy:self.dy + nh, self.dx:self.dx + nw] = resized
        return canvas

    def back(self, pts):
        """640 坐标 -> 原图坐标, pts: (..., 2)"""
        pts = pts.copy()
        pts[..., 0] = (pts[..., 0] - self.dx) / self.scale
        pts[..., 1] = (pts[..., 1] - self.dy) / self.scale
        return pts


def _nms(boxes, scores, thr):
    """boxes: (N,4) xyxy"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= thr]
    return keep


def _make_grids(input_size):
    """生成 8/16/32 stride 的 anchor 点, 供 DFL 解码 (airockchip 格式用)."""
    hs, ws = [], []
    grids, expanded_strides = [], []
    for stride in (8, 16, 32):
        h, w = input_size[1] // stride, input_size[0] // stride
        gv, gw = np.meshgrid(np.arange(w), np.arange(h))
        grids.append(np.stack((gw, gv), -1).reshape(-1, 2))
        expanded_strides.append(np.full((h * w, 1), stride, dtype=np.float32))
    grids = np.concatenate(grids, 0)
    strides = np.concatenate(expanded_strides, 0)
    return grids, strides


def _dfl_decode(dfl, grids, strides, reg_max=16):
    """DFL bin -> xyxy (输入尺度). dfl: (N, 64), bin 为原始 logit.
    距离以 stride 为单位, 解码后乘回 stride."""
    c = dfl.reshape(-1, 4, reg_max)
    prob = _softmax(c)
    proj = np.arange(reg_max, dtype=np.float32)
    dist = (prob * proj).sum(-1)  # (N,4) l,t,r,b (stride 单位)
    s = strides[:, 0]
    ax = (grids[:, 0] + 0.5) * s
    ay = (grids[:, 1] + 0.5) * s
    x1 = ax - dist[:, 0] * s
    y1 = ay - dist[:, 1] * s
    x2 = ax + dist[:, 2] * s
    y2 = ay + dist[:, 3] * s
    return np.stack([x1, y1, x2, y2], -1)


class PoseDetector:
    def __init__(self, model_path, input_size=(640, 640),
                 conf_threshold=0.35, kpt_threshold=0.4, nms_threshold=0.45,
                 core_mask=0):
        self.model_path = str(model_path)
        self.input_size = tuple(input_size)
        self.conf_thr = conf_threshold
        self.kpt_thr = kpt_threshold
        self.nms_thr = nms_threshold
        self.core_mask = core_mask
        self.rknn = None
        self.net = None
        self.infer_ms = 0.0
        self._grids = None

        if self.model_path.endswith(".rknn"):
            self._init_rknn()
        else:
            self._init_onnx()

    # ---------------- runtime init ----------------
    def _init_rknn(self):
        from rknnlite.api import RKNNLite
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError(f"加载 rknn 模型失败: {self.model_path} (ret={ret})")
        kwargs = {}
        if self.core_mask:
            kwargs["core_mask"] = self.core_mask
        ret = self.rknn.init_runtime(**kwargs)
        if ret != 0:
            raise RuntimeError("RKNNLite init_runtime 失败")

    def _init_onnx(self):
        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # ---------------- inference ----------------
    def infer(self, frame_bgr):
        """输入 BGR 图像, 返回 (detections, infer_ms)."""
        lb = Letterbox(*self.input_size)
        img = lb(frame_bgr)
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t0 = time.time()

        if self.rknn is not None:
            outputs = self.rknn.inference(inputs=[np.expand_dims(blob, 0)])
        else:
            self.net.setInput(blob.transpose(2, 0, 1)[None].astype(np.float32))
            outputs = [self.net.forward()]

        self.infer_ms = (time.time() - t0) * 1000.0
        dets = self._postprocess(outputs, lb)
        return dets, self.infer_ms

    # ---------------- postprocess ----------------
    def _postprocess(self, outputs, lb):
        out = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        out = np.asarray(out)
        if out.ndim == 3 and out.shape[0] == 1:
            out = out[0]  # (C, N) 或 (N, C)
        # 统一成 (C=56, N)
        if out.shape[0] != 56 and out.shape[-1] == 56:
            out = out.T
        if out.ndim != 2 or out.shape[0] < 56:
            return []

        n = out.shape[1]
        c_dim = out.shape[0]
        if c_dim >= 65:
            # airockchip 导出: (64 DFL + 1 类别logit + 51 关键点)
            if self._grids is None:
                self._grids = _make_grids(self.input_size)
            grids, strides = self._grids
            boxes = _dfl_decode(out[:64, :].T, grids, strides)
            score_raw = out[64, :]
            kpt_raw = out[65:116, :]
        else:
            # 标准 ultralytics 导出 (56): 解码后的中心点 xywh (输入尺度)
            boxes_raw = out[:4, :]
            cx, cy, w, h = boxes_raw
            boxes = np.stack([cx - w / 2, cy - h / 2,
                              cx + w / 2, cy + h / 2], -1)
            score_raw = out[4, :]
            kpt_raw = out[5:56, :]

        # 类别分数: logit 则补 sigmoid
        if score_raw.max() > 1.0 or score_raw.min() < 0.0:
            score_raw = _sigmoid(score_raw)

        # 关键点: x,y 已在输入尺度; conf 视导出方式补 sigmoid
        kpts = kpt_raw.T.reshape(n, 17, 3)  # (N,17,3)
        kc = kpts[:, :, 2]
        if kc.max() > 1.0 or kc.min() < 0.0:
            kpts[:, :, 2] = _sigmoid(kc)

        # 过滤 + NMS
        mask = score_raw >= self.conf_thr
        if not mask.any():
            return []
        boxes, scores, kpts = boxes[mask], score_raw[mask], kpts[mask]
        keep = _nms(boxes, scores, self.nms_thr)

        dets = []
        for i in keep:
            k = kpts[i].copy()
            k[:, :2] = lb.back(k[:, :2])  # 关键点回原图
            box = lb.back(boxes[i].reshape(-1, 2)).reshape(-1)  # 框回原图
            dets.append({
                "bbox": (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                "score": float(scores[i]),
                "kpts": k,  # (17,3) 原图坐标
            })
        return dets

    def release(self):
        if self.rknn is not None:
            try:
                self.rknn.release()
            except Exception:
                pass


def draw_pose(frame, det, color=(0, 220, 0), fallen=False):
    """绘制单人骨架与检测框."""
    x1, y1, x2, y2 = map(int, det["bbox"])
    c = (0, 0, 255) if fallen else color
    cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
    cv2.putText(frame, f"person {det['score']:.2f}", (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
    k = det["kpts"]
    for i, j in SKELETON:
        if k[i, 2] > 0.3 and k[j, 2] > 0.3:
            cv2.line(frame, (int(k[i, 0]), int(k[i, 1])),
                     (int(k[j, 0]), int(k[j, 1])), (255, 180, 0), 2)
    for x, y, v in k:
        if v > 0.3:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)
