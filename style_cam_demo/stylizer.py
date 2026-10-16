#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""风格化特效封装 (全 NPU 版).

两类特效 (全部走 NPU):
  - NPU 风格迁移: AnimeGANv2 系模型 (.rknn 走 RK3576 NPU,
    .onnx 走 OpenCV DNN 供 PC 调试), 模型缺失时对应模式自动隐藏
  - NPU 关键点贴纸: yolo_pose.rknn 检测鼻/眼/耳关键点 + 透明 PNG 贴纸
    叠加 (叠加本身 CPU 绘制, 仅几 ms)

AnimeGANv2 预处理实测结论:
  输入 [-1,1] (x/127.5-1, 转换时以 mean=std=127.5 写进 rknn, 板端直接喂
  RGB uint8); 输出也是 [-1,1] -> (out+1)*127.5 再 clip (out_range="pm1").
  旧 fast-neural-style 输出是 0~255 图像尺度 (out_range="image255").
"""
import os
import time

import cv2
import numpy as np

from pose_detector import PoseDetector

# 贴纸用到的 COCO 关键点索引
KPT_NOSE, KPT_L_EYE, KPT_R_EYE, KPT_L_EAR, KPT_R_EAR = 0, 1, 2, 3, 4
KPT_VIS_THR = 0.35          # 鼻+双眼可见门槛
KPT_AUX_THR = 0.30          # 耳朵等辅助点门槛


# ------------------------------------------------------------------
# 风格迁移模型 (AnimeGANv2 系)
# ------------------------------------------------------------------
class StyleModel:
    """单个风格模型: .rknn -> NPU / .onnx -> OpenCV DNN (PC 调试)."""

    def __init__(self, path, input_size=(256, 256), core_mask=0,
                 out_range="pm1", layout="nhwc"):
        self.path = str(path)
        self.size = tuple(input_size)          # (w, h)
        self.out_range = out_range             # pm1 / image255
        self.layout = layout                   # onnx 输入布局 (rknn 恒 NHWC 喂入)
        self.rknn = None
        self.net = None
        if self.path.endswith(".rknn"):
            self._init_rknn(core_mask)
        else:
            self.net = cv2.dnn.readNetFromONNX(self.path)

    def _init_rknn(self, core_mask):
        from rknnlite.api import RKNNLite
        self.rknn = RKNNLite()
        if self.rknn.load_rknn(self.path) != 0:
            raise RuntimeError(f"加载 rknn 失败: {self.path}")
        kwargs = {"core_mask": core_mask} if core_mask else {}
        if self.rknn.init_runtime(**kwargs) != 0:
            raise RuntimeError("RKNNLite init_runtime 失败")

    def run(self, frame_bgr):
        """BGR 全尺寸帧 -> 模型输入尺寸的风格化 BGR uint8."""
        w, h = self.size
        img = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.rknn is not None:
            # 归一化在 NPU 内完成 (convert 时写入 mean/std), 直接喂 uint8
            out = self.rknn.inference(inputs=[rgb[None].astype(np.uint8)])
        else:
            x = rgb.astype(np.float32)
            x = x / 127.5 - 1.0 if self.out_range == "pm1" else x / 255.0
            x = x[None] if self.layout == "nhwc" \
                else x.transpose(2, 0, 1)[None]
            self.net.setInput(x)
            out = self.net.forward()

        out = np.asarray(out).squeeze()
        if out.ndim == 3 and out.shape[0] == 3:   # CHW -> HWC
            out = out.transpose(1, 2, 0)
        if self.out_range == "pm1":               # [-1,1] -> 0~255
            out = (out + 1.0) * 127.5
        out = np.clip(out, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    def release(self):
        if self.rknn is not None:
            try:
                self.rknn.release()
            except Exception:
                pass
        self.rknn = None


# ------------------------------------------------------------------
# 关键点贴纸 (yolo_pose NPU + 透明 PNG / OpenCV 绘制)
# ------------------------------------------------------------------
def _overlay(img, patch, cx, cy, w):
    """RGBA/BGR patch 以 (cx,cy) 为中心、宽 w 等比缩放后 alpha 叠加."""
    if patch is None or w < 2:
        return
    ph, pw = patch.shape[:2]
    h = int(round(w * ph / pw))
    if h < 2:
        return
    dst_w = int(round(w))
    resized = cv2.resize(patch, (dst_w, h), interpolation=cv2.INTER_AREA)
    x0, y0 = int(round(cx - dst_w / 2)), int(round(cy - h / 2))
    H, W = img.shape[:2]
    sx, sy = max(0, -x0), max(0, -y0)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x0 + dst_w - sx), min(H, y0 + h - sy)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    p = resized[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
    if p.shape[2] == 4:
        a = p[:, :, 3:4].astype(np.float32) / 255.0
        roi[:] = (p[:, :, :3].astype(np.float32) * a
                  + roi.astype(np.float32) * (1.0 - a)).astype(np.uint8)
    else:
        roi[:] = p[:, :, :3]


def _draw_dog_ears(img, lc, rc, w):
    """无 PNG 素材时用 OpenCV 画卡通狗耳 (棕色旋转椭圆+粉色内耳)."""
    for (cx, cy), angle in ((lc, 35), (rc, -35)):
        axes = (int(w * 0.55), int(w))
        cv2.ellipse(img, (int(cx), int(cy)), axes, angle, 0, 360,
                    (30, 80, 140), -1)          # 深棕 BGR
        cv2.ellipse(img, (int(cx), int(cy + w * 0.25)),
                    (int(w * 0.3), int(w * 0.6)), angle, 0, 360,
                    (120, 140, 250), -1)        # 粉色内耳 BGR


def _draw_dog_nose(img, cx, cy, r):
    """无 PNG 素材时用 OpenCV 画狗鼻."""
    cv2.ellipse(img, (int(cx), int(cy)), (int(r), int(r * 0.8)), 0, 0, 360,
                (20, 45, 70), -1)               # 深棕黑 BGR
    cv2.ellipse(img, (int(cx - r * 0.3), int(cy - r * 0.25)),
                (int(r * 0.22), int(r * 0.14)), 0, 0, 360,
                (140, 160, 180), -1)            # 高光


def _draw_pig_nose(img, cx, cy, r):
    """OpenCV 画猪鼻 (粉色椭圆 + 两个鼻孔)."""
    cv2.ellipse(img, (int(cx), int(cy)), (int(r * 1.35), int(r)), 0, 0, 360,
                (144, 173, 252), -1)            # 粉 BGR
    dark = (95, 110, 190)
    for dx in (-r * 0.55, r * 0.55):
        cv2.ellipse(img, (int(cx + dx), int(cy)),
                    (max(2, int(r * 0.26)), max(3, int(r * 0.42))),
                    0, 0, 360, dark, -1)


def _draw_pig_ears(img, lc, rc, w):
    """OpenCV 画猪耳 (粉色三角)."""
    top = int(-w * 0.2)
    for (cx, cy), sx in ((lc, -1), (rc, 1)):
        pts = np.array([
            (cx - sx * w * 0.55, cy + w * 0.55),
            (cx + sx * w * 0.15, cy + w * 0.55),
            (cx + sx * w * 0.05, cy - w * 0.45),
        ], np.int32)
        cv2.fillPoly(img, [pts], (150, 180, 250))     # 粉 BGR
        inner = (pts.astype(np.float32) * 0.75
                 + np.float32([cx, cy + top]) * 0.25).astype(np.int32)
        cv2.fillPoly(img, [inner], (110, 130, 220))   # 深粉内耳


def _draw_rabbit_ears(img, lc, rc, fw):
    """OpenCV 画兔耳 (白色长椭圆 + 粉色内耳, 略向外倾斜). fw≈脸宽."""
    for (cx, cy), ang in ((lc, 15), (rc, -15)):
        cv2.ellipse(img, (int(cx), int(cy)),
                    (max(2, int(fw * 0.10)), max(4, int(fw * 0.38))),
                    ang, 0, 360, (250, 248, 252), -1)      # 白 BGR
        cv2.ellipse(img, (int(cx), int(cy + fw * 0.07)),
                    (max(2, int(fw * 0.05)), max(3, int(fw * 0.26))),
                    ang, 0, 360, (185, 175, 255), -1)      # 粉内耳 BGR


def _draw_rabbit_nose(img, cx, cy, fw):
    """OpenCV 画兔鼻 (粉色三角 + 人中 + 胡须)."""
    r = fw * 0.13
    pts = np.array([(cx, cy + r * 0.75),
                    (cx - r, cy - r * 0.45),
                    (cx + r, cy - r * 0.45)], np.float32)
    cv2.fillPoly(img, [pts.astype(np.int32)], (150, 150, 255))   # 粉 BGR
    cv2.line(img, (int(cx), int(cy + r * 0.75)),
             (int(cx), int(cy + r * 2.0)), (170, 130, 170),
             max(2, int(fw * 0.02)))                             # 人中
    for sx in (-1, 1):
        for dy in (-0.2, 0.35):
            cv2.line(img,
                     (int(cx + sx * r * 0.8), int(cy + dy * r)),
                     (int(cx + sx * r * 4.2), int(cy + (dy - 0.6) * r)),
                     (210, 210, 220), 1)                         # 胡须


def _draw_clown_hair(img, eye_c, fw):
    """OpenCV 画小丑假发 (头顶两侧各一簇彩色圆)."""
    colors = ((0, 200, 255), (30, 120, 255), (220, 60, 220), (70, 130, 60))
    offs = ((-0.55, -0.45), (-0.34, -0.68), (-0.10, -0.80), (0.12, -0.72))
    r = max(3, int(fw * 0.16))
    for sx in (-1, 1):
        for (dx, dy), c in zip(offs, colors):
            cv2.circle(img,
                       (int(eye_c[0] + sx * dx * fw),
                        int(eye_c[1] + dy * fw)), r, c, -1)


def _draw_clown_eyes(img, le, re, fw):
    """OpenCV 画小丑眼妆 (白色眼眶 + 彩色菱形)."""
    for (ex, ey), c in ((le, (60, 200, 255)), (re, (255, 160, 60))):
        cv2.ellipse(img, (int(ex), int(ey)),
                    (max(3, int(fw * 0.16)), max(4, int(fw * 0.22))),
                    0, 0, 360, (255, 255, 255), -1)
        pts = np.array([(ex, ey - fw * 0.42),
                        (ex + fw * 0.10, ey - fw * 0.18),
                        (ex, ey + fw * 0.06),
                        (ex - fw * 0.10, ey - fw * 0.18)], np.float32)
        cv2.fillPoly(img, [pts.astype(np.int32)], c)


def _draw_clown_nose(img, cx, cy, fw):
    """OpenCV 画小丑红鼻头."""
    r = fw * 0.22
    cv2.circle(img, (int(cx), int(cy)), int(r), (40, 40, 230), -1)
    cv2.circle(img, (int(cx - r * 0.3), int(cy - r * 0.3)),
               max(2, int(r * 0.22)), (130, 150, 255), -1)   # 高光


def _draw_clown_mouth(img, cx, cy, fw):
    """OpenCV 画小丑大红嘴 (弧线)."""
    cv2.ellipse(img, (int(cx), int(cy)),
                (max(4, int(fw * 0.42)), max(3, int(fw * 0.30))),
                0, 15, 165, (40, 40, 230), max(3, int(fw * 0.09)))


class PoseSticker:
    """人脸贴纸特效: yolo_pose 关键点检测 (NPU) + 贴纸叠加.

    dog   : 狗耳×2 (PNG 优先, 缺失用 OpenCV 画) + 狗鼻 (同上)
    pig   : 猪耳×2 + 猪鼻 (OpenCV 绘制)
    rabbit: 兔耳×2 + 兔鼻胡须 (OpenCV 绘制)
    clown : 彩色假发 + 白眼眶彩菱形 + 红鼻头 + 大红嘴 (OpenCV 绘制)
    """

    def __init__(self, pose_model, assets_dir="assets", core_mask=0):
        self.det = PoseDetector(pose_model, input_size=(640, 640),
                                core_mask=core_mask)
        self.assets_dir = str(assets_dir)
        self._pngs = {}

    def _png(self, name):
        if name not in self._pngs:
            p = os.path.join(self.assets_dir, name + ".png")
            self._pngs[name] = (cv2.imread(p, cv2.IMREAD_UNCHANGED)
                                if os.path.exists(p) else None)
        return self._pngs[name]

    def has_dog_assets(self):
        return (self._png("dog_ears") is not None
                and self._png("dog_nose") is not None)

    def apply(self, frame_bgr, effect):
        """返回 (贴纸后的帧, 是否命中人脸)."""
        dets, _ = self.det.infer(frame_bgr)
        if not dets:
            return frame_bgr, False
        img = frame_bgr
        hit = False
        for d in dets:
            k = d["kpts"]
            if min(k[KPT_NOSE, 2], k[KPT_L_EYE, 2], k[KPT_R_EYE, 2]) \
                    < KPT_VIS_THR:
                continue
            hit = True
            nose = k[KPT_NOSE, :2]
            eye_c = (k[KPT_L_EYE, :2] + k[KPT_R_EYE, :2]) / 2.0
            eye_dist = float(np.linalg.norm(k[KPT_L_EYE, :2]
                                            - k[KPT_R_EYE, :2]))
            ear_ok = min(k[KPT_L_EAR, 2], k[KPT_R_EAR, 2]) > KPT_AUX_THR
            if ear_ok:
                face_w = float(np.linalg.norm(k[KPT_L_EAR, :2]
                                              - k[KPT_R_EAR, :2])) * 1.35
            else:
                face_w = eye_dist * 2.2
            if face_w < 8:                        # 太远, 贴了也看不清
                continue

            if effect == "dog":
                ears, nose_png = self._png("dog_ears"), self._png("dog_nose")
                if ears is not None:
                    # 一对耳朵的宽图: 盖在头顶, 比脸略宽
                    _overlay(img, ears, eye_c[0], eye_c[1] - face_w * 0.52,
                             face_w * 1.35)
                else:
                    ear_l = (eye_c[0] - face_w * 0.32,
                             eye_c[1] - face_w * 0.62)
                    ear_r = (eye_c[0] + face_w * 0.32,
                             eye_c[1] - face_w * 0.62)
                    _draw_dog_ears(img, ear_l, ear_r, face_w * 0.22)
                if nose_png is not None:
                    _overlay(img, nose_png, nose[0], nose[1], face_w * 0.38)
                    tongue = self._png("dog_tongue")
                    if tongue is not None:       # 舌头在鼻子下方
                        _overlay(img, tongue, nose[0], nose[1] + face_w * 0.34,
                                 face_w * 0.3)
                else:
                    _draw_dog_nose(img, nose[0], nose[1], face_w * 0.18)
            elif effect == "rabbit":              # 兔头 (OpenCV 绘制)
                ear_l = (eye_c[0] - face_w * 0.30, eye_c[1] - face_w * 0.72)
                ear_r = (eye_c[0] + face_w * 0.30, eye_c[1] - face_w * 0.72)
                _draw_rabbit_ears(img, ear_l, ear_r, face_w)
                _draw_rabbit_nose(img, nose[0], nose[1], face_w)
            elif effect == "clown":               # 小丑头 (OpenCV 绘制)
                _draw_clown_hair(img, eye_c, face_w)
                _draw_clown_eyes(img, k[KPT_L_EYE, :2], k[KPT_R_EYE, :2],
                                 face_w)
                _draw_clown_nose(img, nose[0], nose[1], face_w)
                _draw_clown_mouth(img, nose[0], nose[1] + face_w * 0.52,
                                  face_w)
            else:                                 # pig (OpenCV 绘制)
                ear_l = (eye_c[0] - face_w * 0.32, eye_c[1] - face_w * 0.62)
                ear_r = (eye_c[0] + face_w * 0.32, eye_c[1] - face_w * 0.62)
                _draw_pig_ears(img, ear_l, ear_r, face_w * 0.35)
                _draw_pig_nose(img, nose[0], nose[1], face_w * 0.22)
        return img, hit

    def release(self):
        self.det.release()


# ------------------------------------------------------------------
# Stylizer: 模式管理 + 统一入口
# ------------------------------------------------------------------
MODE_LABELS = {
    "hayao": "宫崎骏风 (NPU)",
    "oil": "油画 (NPU)",
    "candy": "糖果风 (NPU)",
    "mosaic": "玻璃马赛克 (NPU)",
    "sketch": "素描 (NPU)",
    "ukiyoe": "浮世绘 (NPU)",
    "ink": "水墨画 (NPU)",
    "dog": "狗头 (NPU)",
    "pig": "猪头 (NPU)",
    "rabbit": "兔头 (NPU)",
    "clown": "小丑头 (NPU)",
}
STICKER_MODES = ("dog", "pig", "rabbit", "clown")


class Stylizer:
    def __init__(self, cfg, sticker_cfg=None):
        self.size = tuple(cfg.get("input_size", [256, 256]))
        self.strength = float(cfg.get("strength", 1.0))
        self.core_mask = int(cfg.get("core_mask", 0))
        self.model_cfg = cfg.get("models", {}) or {}
        self.sticker_cfg = sticker_cfg or {}
        self._models = {}              # key -> StyleModel (懒加载)
        self._sticker = None
        self._load_failed = set()
        self.infer_ms = 0.0

    # ---- 配置: value 可以是路径字符串或 {path, out_range, layout} ----
    def _model_spec(self, key):
        v = self.model_cfg.get(key)
        if isinstance(v, dict):
            return v.get("path"), v
        return v, {}

    def available_modes(self):
        modes = [("none", "原始")]
        for key in self.model_cfg:
            path, _ = self._model_spec(key)
            if path and os.path.exists(str(path)):
                modes.append((key, MODE_LABELS.get(key, f"{key} (NPU)")))
        if self._sticker_ready():
            for key in STICKER_MODES:
                if key == "dog" and not self._sticker_ready(dog_assets=True):
                    continue
                modes.append((key, MODE_LABELS[key]))
        return modes

    def _sticker_ready(self, dog_assets=False):
        """贴纸模式可用性: 依赖 yolo_pose 模型 (dog 另需 PNG 素材,
        但 PNG 缺失时代码绘制兜底, 仍然可用)."""
        pm = self.sticker_cfg.get("pose_model")
        return bool(pm) and os.path.exists(str(pm))

    def get_model(self, key):
        if key in self._models:
            return self._models[key]
        if key in self._load_failed:
            return None
        path, spec = self._model_spec(key)
        if not path or not os.path.exists(str(path)):
            return None
        try:
            size = tuple(spec.get("size") or self.size)   # 每模型可覆盖
            m = StyleModel(path, size, self.core_mask,
                           out_range=spec.get("out_range", "pm1"),
                           layout=spec.get("layout", "nhwc"))
        except Exception as e:
            print(f"[style] 模型 {key} 加载失败, 已跳过: {e}")
            self._load_failed.add(key)
            return None
        print(f"[style] 已加载 {key}: {path}")
        self._models[key] = m
        return m

    def get_sticker(self):
        if self._sticker is not None:
            return self._sticker
        if "sticker" in self._load_failed:
            return None
        pm = self.sticker_cfg.get("pose_model")
        if not pm or not os.path.exists(str(pm)):
            return None
        try:
            self._sticker = PoseSticker(
                pm, self.sticker_cfg.get("assets_dir", "assets"),
                self.core_mask)
        except Exception as e:
            print(f"[style] 贴纸模块加载失败, 已跳过: {e}")
            self._load_failed.add("sticker")
            return None
        print(f"[style] 已加载贴纸模块: {pm}")
        return self._sticker

    def stylize(self, frame_bgr, mode_key):
        """返回 (风格化帧, 耗时ms). mode_key=none 时原样返回."""
        t0 = time.time()
        if mode_key == "none":
            return frame_bgr, 0.0

        if mode_key in STICKER_MODES:
            st = self.get_sticker()
            if st is None:
                self.infer_ms = 0.0
                return frame_bgr, 0.0
            out, _ = st.apply(frame_bgr, mode_key)
            self.infer_ms = (time.time() - t0) * 1000.0
            return out, self.infer_ms

        model = self.get_model(mode_key)
        if model is None:
            self.infer_ms = 0.0
            return frame_bgr, 0.0

        styled = model.run(frame_bgr)
        styled = cv2.resize(styled, (frame_bgr.shape[1], frame_bgr.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
        if self.strength < 1.0:
            a = self.strength
            styled = cv2.addWeighted(styled, a, frame_bgr, 1.0 - a, 0)
        self.infer_ms = (time.time() - t0) * 1000.0
        return styled, self.infer_ms

    def release(self):
        for m in self._models.values():
            m.release()
        self._models.clear()
        if self._sticker is not None:
            self._sticker.release()
            self._sticker = None
