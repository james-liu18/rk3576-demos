#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跌倒判定状态机.

判定方式参考 Quectel 官方 demo-fall-alarm-device 的二次验证 (打分制):
  前提: 可见(可信)骨骼节点比例 >= kpt_visible_ratio, 否则不判定
  - 躯干倾角: >=阈值 +2 / >=0.6*阈值(半躺) +1
  - 腿部水平 (膝/踝接近髋部高度, 躺姿标志)   +1.5
  - 半躺组合: 躯干半倾(0.6~1倍阈值)+腿部水平 +0.5
  - 身体长宽比横向 (关键点头顶/脚踝/肩宽)    +2
  - 多数关键点贴近画面底部                   +1.5
  - 关键点横向分布 (x方差/y方差, y跨度)      +1~3
  总分 >= 3 才算"倒地姿态", 单一特征不足以触发 (防坐姿/边缘裁剪误报):
  坐姿(躯干直立)即使跷腿最多 1.5+分布分, 蹲姿(腿部水平但躯干直立)同理.
  另: 髋部快速下坠仅在同时倾角超标时直接判定 (防止坐下动作误报).

整体状态机 (多人场景取置信度最高目标):
  NORMAL --连续N帧"倒地姿态"--> 触发 FALL 事件 -> DOWN
  DOWN   --连续M帧恢复直立--> NORMAL
"""
import math
import time
from collections import deque

import numpy as np

# COCO 索引
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16
SCORE_KEYS = (NOSE, 5, 6, 11, 12, 13, 14, L_ANKLE, R_ANKLE)


def _mid(a, b):
    return (np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)) / 2.0


class FallDetector:
    def __init__(self, cfg):
        self.angle_thr = cfg.get("torso_angle_threshold", 55)
        self.ratio_thr = cfg.get("height_ratio_threshold", 1.2)
        self.aspect_thr = cfg.get("aspect_ratio_threshold", 1.05)
        self.min_kpt_conf = cfg.get("min_kpt_conf", 0.4)
        self.vis_ratio_thr = cfg.get("kpt_visible_ratio", 0.8)  # 可见节点比例门槛
        self.trigger_frames = cfg.get("min_trigger_frames", 4)
        self.recover_frames = cfg.get("recover_frames", 30)
        self.use_velocity = cfg.get("use_velocity", True)
        self.v_drop_px = cfg.get("velocity_drop_px", 110)
        self.v_window = cfg.get("velocity_window", 5)
        self.min_score = cfg.get("min_person_score", 0.40)
        self.cooldown_s = cfg.get("cooldown_s", 15)

        self.state = "NORMAL"          # NORMAL / DOWN
        self.fallen_count = 0
        self.upright_count = 0
        self.last_event_ts = 0.0
        self.hip_history = {}          # 按人(位置键)隔离的髋部y历史, 多人不互相污染
        self.features = {}             # 最近一次特征, 供 UI 显示

    # ---------------- 姿态特征 ----------------
    @staticmethod
    def _person_key(det):
        """按检测框中心粗量化, 作跨帧同人识别键 (用于髋部下坠历史隔离)."""
        x1, y1, x2, y2 = det["bbox"]
        return (int((x1 + x2) / 2) // 48, int((y1 + y2) / 2) // 48)

    def _person_features(self, det, img_hw=None, key=None):
        """返回 (is_fallen, hip_center, info). img_hw=(h,w) 用于底部/分布特征."""
        k = det["kpts"]
        h, w = det["bbox"][3] - det["bbox"][1], det["bbox"][2] - det["bbox"][0]
        kpc = k[:, 2]
        vis_ok = kpc > self.min_kpt_conf
        # 可见(可信)节点比例: 低于门槛视为骨架观测不完整, 不允许判定跌倒
        vis_ratio = float(np.mean(vis_ok))
        info = {"kpt_vis_ratio": round(vis_ratio, 2)}
        if vis_ratio < self.vis_ratio_thr:
            return False, None, info
        hip = None
        angle_ok = False
        torso_len = 0.0

        core = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
        core_vis = sum(1 for i in core if vis_ok[i])

        # 1) 躯干倾角: 肩中点->髋中点 与垂直方向夹角 (分档: 全倒+2 / 半躺+1)
        angle = 0.0
        if core_vis >= 3:
            shoulder = _mid(k[L_SHOULDER][:2], k[R_SHOULDER][:2])
            hip = _mid(k[L_HIP][:2], k[R_HIP][:2])
            v = hip - shoulder
            torso_len = float(np.linalg.norm(v))
            if torso_len > 1e-3:
                cos_a = abs(v[1]) / torso_len
                angle = math.degrees(math.acos(min(1.0, max(-1.0, cos_a))))
                info["torso_angle"] = round(angle, 1)
                angle_ok = angle >= self.angle_thr

        score = 0.0
        if angle_ok:
            score += 2.0
        elif angle >= self.angle_thr * 0.6:
            score += 1.0

        # 1b) 腿部水平: 膝/踝接近髋部高度 -> 躺/半躺 (坐姿腿部朝下不命中)
        #     半躺组合: 躯干半倾 + 腿部水平 额外 +0.5 (斜倚撑地典型形态)
        legs_h = False
        if hip is not None and torso_len > 1e-3:
            leg_pts = [k[i] for i in (13, 14, L_ANKLE, R_ANKLE) if vis_ok[i]]
            if leg_pts:
                dmin = min(abs(float(p[1]) - float(hip[1])) for p in leg_pts)
                legs_h = dmin <= torso_len * 0.45
                info["leg_dy"] = round(dmin / torso_len, 2)
        if legs_h:
            score += 1.5
            if self.angle_thr * 0.6 <= angle < self.angle_thr:
                score += 0.5

        # 2) 身体长宽比 (关键点: 头顶y vs 脚踝y, 肩宽); 关键点不可见才用 bbox 兜底
        ratio_ok = False
        if vis_ok[NOSE] and (vis_ok[L_ANKLE] or vis_ok[R_ANKLE]):
            head_y = float(k[NOSE][1])
            ankle_y = float(max(k[L_ANKLE][1], k[R_ANKLE][1]))
            body_h = abs(ankle_y - head_y)
            body_w = abs(float(k[L_SHOULDER][0]) - float(k[R_SHOULDER][0])) \
                if vis_ok[L_SHOULDER] and vis_ok[R_SHOULDER] else 0.0
            if body_w > 1:
                info["body_ratio"] = round(body_h / body_w, 2)
                ratio_ok = body_h / body_w < self.ratio_thr
        else:
            ratio = h / w if w > 1 else 99
            info["aspect_bbox"] = round(float(ratio), 2)
            ratio_ok = ratio <= self.aspect_thr
        if ratio_ok:
            score += 2.0

        # 3) 贴地: 多数关键点位于画面下部 (参考官方 _analyze_ground_contact)
        if img_hw and core_vis >= 3:
            ih = img_hw[0]
            bottom = ih * 0.7
            cnt = tot = 0
            for i in SCORE_KEYS:
                if vis_ok[i]:
                    tot += 1
                    if k[i][1] > bottom:
                        cnt += 1
            if tot and cnt / tot > 0.6:
                score += 1.5

        # 4) 横向分布: x方差/y方差 大 -> 横躺 (参考官方 _analyze_horizontal_distribution)
        if img_hw and core_vis >= 3:
            pts = [k[i][:2] for i in SCORE_KEYS if vis_ok[i]]
            if len(pts) >= 4:
                pts = np.asarray(pts)
                yv = float(np.var(pts[:, 1]))
                if yv > 1e-6:
                    r = float(np.var(pts[:, 0])) / yv
                    info["hdist"] = round(r, 2)
                    if r > 3:
                        score += 2
                    elif r > 1.5:
                        score += 1
                ys = (pts[:, 1].max() - pts[:, 1].min()) / img_hw[0]
                if ys < 0.25:
                    score += 1

        info["fall_score"] = round(score, 1)
        posture = score >= 3.0

        # 5) 髋部快速下坠: 需同时倾角超标, 防止"坐下"动作误报; 历史按人隔离
        vel_ok = False
        if self.use_velocity and hip is not None and key is not None:
            hist = self.hip_history.setdefault(key, deque(maxlen=10))
            hist.append(float(hip[1]))
            if len(hist) >= 2:
                recent = list(hist)[-self.v_window:]
                drop = recent[-1] - min(recent[:-1]) if len(recent) > 1 else 0
                info["hip_drop"] = round(float(drop), 1)
                vel_ok = drop >= self.v_drop_px and angle_ok

        is_fallen = posture or vel_ok
        info.update({"angle_ok": angle_ok, "ratio_ok": ratio_ok,
                     "vel_ok": vel_ok, "posture": posture})
        return is_fallen, hip, info

    # ---------------- 每帧更新 ----------------
    def update(self, detections, img_hw=None):
        """输入当帧检测结果, 返回 (event: None|'fall', state, fallen_det|None).

        每个检测到的人独立判定 (各自80%可见门槛+打分), 任一人倒地即视为有人倒地;
        恢复需所有可见目标均不处于倒地姿态.
        """
        dets = [d for d in detections if d["score"] >= self.min_score]
        event = None
        fallen_det = None

        if not dets:
            self.fallen_count = 0
            self.features = {}
            # 无目标不重置 DOWN 状态, 防止离开画面误消警
            return event, self.state, None

        # 逐人判定
        results = []
        for det in dets:
            is_fallen, hip, info = self._person_features(
                det, img_hw, key=self._person_key(det))
            results.append((is_fallen, det, info))
        # 清理已消失目标的髋部历史
        alive = {self._person_key(d) for d in dets}
        for key in list(self.hip_history):
            if key not in alive:
                del self.hip_history[key]

        # 任一人倒地即算有人倒地; 标注/特征取"最倒地"(分最高)者
        fallen_rs = [r for r in results if r[0]]
        anyone_fallen = bool(fallen_rs)
        if fallen_rs:
            best = max(fallen_rs, key=lambda r: r[2].get("fall_score", 0))
            fallen_det, self.features = best[1], best[2]
        else:
            # 无人倒地时展示置信度最高者的特征, 便于标定观察
            ref = max(results, key=lambda r: r[1]["score"])
            self.features = ref[2]

        if self.state == "NORMAL":
            self.fallen_count = self.fallen_count + 1 if anyone_fallen else 0
            now = time.time()
            if self.fallen_count >= self.trigger_frames and \
                    now - self.last_event_ts >= self.cooldown_s:
                self.state = "DOWN"
                self.upright_count = 0
                self.last_event_ts = now
                event = "fall"
        else:  # DOWN: 所有人都恢复直立才累计解除
            self.upright_count = 0 if anyone_fallen else self.upright_count + 1
            if self.upright_count >= self.recover_frames:
                self.state = "NORMAL"
                self.fallen_count = 0

        return event, self.state, fallen_det
