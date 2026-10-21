#!/usr/bin/env python3
"""离线演示视频合成工具.

把一段网上的人体跌倒视频喂给真实的 NPU 推理链路 (PoseDetector + FallDetector),
逐帧骨骼标注后合成进真实 UI 模板截图, 再配上警笛音轨与慢速回放, 产出演示 mp4.

用法 (板端, 需 root):
    sudo /opt/py312/bin/python3.12 tools/make_demo_video.py \
        --src /tmp/fall-01-cam0.mp4 --template /tmp/ui_template.png \
        --out-dir /tmp/demo_out
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pose_detector import PoseDetector      # noqa: E402
from fall_detector import FallDetector      # noqa: E402
from ui import annotate_frame               # noqa: E402

FPS = 30
FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"

# ---- UI 模板布局 (模板截图 1426x975, 由 tools/measure_ui.py 实测) ----
VIDEO_RECT = (122, 163, 1166, 746)    # 视频容器整块
BADGE_RECT = (1228, 98, 1352, 144)    # 右上状态徽标
VAL_RECTS = {                         # 右侧指标数值区 (只清数值行, 保留标签)
    "fps":    (1228, 190, 1352, 214),
    "ms":     (1228, 269, 1352, 293),
    "count":  (1228, 348, 1352, 372),
    "noise":  (1228, 424, 1352, 448),
    "sound":  (1228, 506, 1352, 530),
    "bar":    (1243, 533, 1337, 553),   # 声音强度条
}
LOG_RECT = (124, 787, 1166, 900)      # 事件日志框

TITLE_DUR = 3.5
OUTRO_DUR = 2.5


def F(size):
    return ImageFont.truetype(FONT_PATH, size)


def sample_bg(img, x, y, r=3):
    """取 (x,y) 附近 r 范围内的中位色, 作为填充底色."""
    patch = img[y - r:y + r, x - r:x + r]
    b, g, rr = (int(v) for v in np.median(patch.reshape(-1, 3), axis=0))
    return (rr, g, b)   # PIL 用 RGB


def draw_pill(draw, rect, text, fill, font, text_fill=(255, 255, 255)):
    draw.rounded_rectangle(rect, radius=(rect[3] - rect[1]) // 2, fill=fill)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cx = (rect[0] + rect[2]) // 2 - tw // 2
    cy = (rect[1] + rect[3]) // 2 - th // 2 - bbox[1]
    draw.text((cx, cy), text, font=font, fill=text_fill)


class UiCompositor:
    """把标注帧合成进 UI 模板, 并实时刷新徽标/指标/事件日志."""

    def __init__(self, template_path):
        self.tpl = cv2.imread(template_path)
        assert self.tpl is not None, f"模板读取失败: {template_path}"
        if self.tpl.shape[0] % 2:          # libx264 要求高度为偶数
            self.tpl = self.tpl[:-1]
        self.h, self.w = self.tpl.shape[:2]
        self.bgs = {
            "header": sample_bg(self.tpl, 1100, 125),
            "card":   {k: sample_bg(self.tpl, c - 8, (b + d) // 2)
                       for k, (a, b, c, d) in VAL_RECTS.items()},
            "log":    sample_bg(self.tpl, 550, 880),
        }
        self.f_val = F(26)
        self.f_badge = F(20)
        self.f_log = F(17)
        self.events = []              # [(ts, desc, is_fall)]
        self.fps_val = 0.0
        self.ms_val = 0
        self.count_val = 0
        self.noise_db = -21
        self.sound_db = -21
        self.alarm_on = False

    # ---- 底图: 模板 + 覆盖视频区 ----
    def base(self, frame_rgb):
        canvas = self.tpl.copy()   # 保留完整 UI, 仅替换视频区内容
        x0, y0, x1, y1 = VIDEO_RECT
        vh, vw = y1 - y0, x1 - x0
        fh, fw = frame_rgb.shape[:2]
        scale = min(vw / fw, vh / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        resized = cv2.resize(frame_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 0, 0), -1)  # 黑底压掉旧画面
        ox, oy = x0 + (vw - nw) // 2, y0 + (vh - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = resized
        return canvas

    # ---- 动态 UI 元素 ----
    def overlay(self, canvas):
        pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        # 右上徽标
        bx0, by0, bx1, by1 = BADGE_RECT
        d.rectangle([bx0 - 4, by0 - 4, bx1 + 4, by1 + 4], fill=self.bgs["header"])
        if self.alarm_on:
            draw_pill(d, BADGE_RECT, "检测到跌倒!", (200, 40, 40), self.f_badge)
        else:
            draw_pill(d, BADGE_RECT, "正常监控", (40, 150, 70), self.f_badge)
        # 指标数值
        vals = {
            "fps":   f"{self.fps_val:.1f}",
            "ms":    f"{self.ms_val} ms",
            "count": f"{self.count_val}",
            "noise": f"{int(round(self.noise_db))} dB",
            "sound": f"{int(round(self.sound_db))} dB",
        }
        for k, (a, b, c, dd) in VAL_RECTS.items():
            if k == "bar":
                d.rectangle([a, b, c, dd], fill=self.bgs["card"][k])
                ratio = max(0.0, min(1.0, (self.sound_db + 40) / 40))
                bw = int((c - a - 2) * ratio)
                color = (230, 126, 34) if not self.alarm_on else (220, 60, 60)
                if bw > 2:
                    d.rectangle([a + 1, b + 1, a + 1 + bw, dd - 1], fill=color)
            else:
                d.rectangle([a, b, c, dd], fill=self.bgs["card"][k])
                d.text((a + 4, b + 2), vals[k], font=self.f_val,
                       fill=(255, 255, 255))
        # 事件日志 (最新在底部, 最多 4 行)
        lx0, ly0, lx1, ly1 = LOG_RECT
        d.rectangle([lx0 + 2, ly0 + 2, lx1 - 2, ly1 - 2], fill=self.bgs["log"])
        y = ly0 + 14
        for ts, desc, is_fall in self.events[-4:]:
            color = (231, 76, 60) if is_fall else (200, 200, 200)
            d.text((lx0 + 18, y), f"{ts}  {desc}", font=self.f_log, fill=color)
            y += 25
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def compose(self, annotated_frame):
        rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        return self.overlay(self.base(rgb))


def make_title_frame(text_main, text_sub, lines, size=44, h=974, w=1426):
    img = np.full((h, w, 3), 16, np.uint8)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    f1, f2, f3 = F(size), F(24), F(20)
    w1 = d.textbbox((0, 0), text_main, font=f1)[2]
    d.text(((w - w1) // 2, 340), text_main, font=f1, fill=(240, 240, 240))
    w2 = d.textbbox((0, 0), text_sub, font=f2)[2]
    d.text(((w - w2) // 2, 430), text_sub, font=f2, fill=(120, 180, 250))
    y = 530
    for ln in lines:
        wl = d.textbbox((0, 0), ln, font=f3)[2]
        d.text(((w - wl) // 2, y), ln, font=f3, fill=(150, 150, 150))
        y += 36
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def build_audio(siren_wav, total, marks, sr=44100):
    """生成整条音轨: 静音底 + 在 marks 秒处循环叠加警笛."""
    import wave
    with wave.open(siren_wav, "rb") as w:
        assert w.getframerate() == 16000, "siren 需 16k wav"
        data = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
    track = np.zeros(int(total * sr), np.float32)
    for t0, dur in marks:
        i0 = int(t0 * sr)
        seg = np.tile(data, int(np.ceil(dur * 16000 / len(data))))[:int(dur * 16000)]
        seg = cv2.resize(seg.reshape(1, -1), (int(dur * sr), 1),
                         interpolation=cv2.INTER_AREA).ravel()
        i1 = min(i0 + len(seg), len(track))
        track[i0:i1] = np.clip(track[i0:i1] + seg[:i1 - i0] * 0.9, -32768, 32767)
    track16 = track.astype(np.int16)
    out = os.path.join(os.path.dirname(siren_wav), "_track.wav")
    import wave as wv
    with wv.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(track16.tobytes())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="跌倒源视频")
    ap.add_argument("--template", required=True, help="UI 模板截图 png")
    ap.add_argument("--out-dir", default="/tmp/demo_out")
    ap.add_argument("--upscale", type=float, default=1.0,
                    help="推理前放大倍数 (低分辨率源素材建议 2.0)")
    ap.add_argument("--conf", default=os.path.join(ROOT, "config.yaml"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.conf, encoding="utf-8"))
    os.makedirs(args.out_dir, exist_ok=True)
    seq_dir = os.path.join(args.out_dir, "seq")
    shutil.rmtree(seq_dir, ignore_errors=True)
    os.makedirs(seq_dir)

    model_path = cfg["model"]["path"]
    if not os.path.isabs(model_path):
        model_path = os.path.join(ROOT, model_path)
    detector = PoseDetector(
        model_path,
        input_size=tuple(cfg["model"].get("input_size", [640, 640])),
        conf_threshold=cfg["model"].get("conf_threshold", 0.35),
        kpt_threshold=cfg["model"].get("kpt_threshold", 0.4),
        nms_threshold=cfg["model"].get("nms_threshold", 0.45),
        core_mask=cfg["model"].get("core_mask", 0))
    fall_det = FallDetector(cfg.get("fall", {}))

    cap = cv2.VideoCapture(args.src)
    frames_src = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames_src.append(f)
    cap.release()
    print(f"源视频 {len(frames_src)} 帧")

    comp = UiCompositor(args.template)
    ui_frames = []      # 主段合成帧 (BGR)
    event_i = None
    t0 = time.time()
    fps_n, fps_t = 0, t0
    for i, frame in enumerate(frames_src):
        if args.upscale != 1.0:
            frame = cv2.resize(frame, None, fx=args.upscale, fy=args.upscale,
                               interpolation=cv2.INTER_CUBIC)
        dets, ms = detector.infer(frame)
        event, state, fallen_det = fall_det.update(dets, frame.shape[:2])
        if event == "fall":
            event_i = i
            ts = time.strftime("%H:%M:%S")
            comp.events.append((ts, "检测到人体跌倒!", True))
            comp.alarm_on = True
            print(f"[event] 第{i}帧 检测到跌倒 state={state}")
        fi = next((j for j, d in enumerate(dets) if d is fallen_det), None)
        annotated = annotate_frame(frame, dets, fallen_idx=fi)
        comp.ms_val = int(ms)
        comp.count_val = len(dets)
        comp.sound_db = -7.5 if comp.alarm_on else -21 + np.random.uniform(-1, 1)
        comp.noise_db = -21 + np.random.uniform(-0.6, 0.6)
        now = time.time()
        fps_n += 1
        if now - fps_t >= 1.0:
            comp.fps_val = fps_n / (now - fps_t)
            fps_n, fps_t = 0, now
        ui_frames.append(comp.compose(annotated))
        if i % 30 == 0:
            print(f"  {i}/{len(frames_src)} ({time.time()-t0:.0f}s)")
    total_ms = time.time() - t0
    if event_i is None:
        print("警告: 未触发跌倒事件! 换一段视频或检查判定阈值")
    comp.fps_val = len(frames_src) / total_ms
    print(f"推理完成 {total_ms:.0f}s, 平均 {len(frames_src)/total_ms:.1f} fps")

    # ---- 组装帧序列: 标题卡 + 主段 + 慢速回放 + 片尾 ----
    saved = 0

    def save(img, times=1):
        nonlocal saved
        p = os.path.join(seq_dir, f"{saved:05d}.jpg")
        cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        for _ in range(times - 1):
            saved += 1
            shutil.copy(p, os.path.join(seq_dir, f"{saved:05d}.jpg"))
        saved += 1

    title = make_title_frame(
        "跌倒检测 · 异常声音告警", "Quectel Pi M2 (RK3576) NPU 实时演示",
        ["YOLOv8n-Pose 姿态估计 · 多级打分跌倒判定 · 声光告警 · 自动事件录像"],
        h=comp.h)
    for _ in range(int(TITLE_DUR * FPS)):
        save(title)

    main_start = saved / FPS
    for f in ui_frames:
        save(f)
    ev_t = main_start + (event_i / FPS if event_i is not None else 0)

    # 慢速回放: 事件前后各 1.5s/1s, 每帧写 2 次 (0.5x)
    if event_i is not None:
        lo = max(0, event_i - 45)
        hi = min(len(ui_frames) - 1, event_i + 30)
        replay_start = saved / FPS
        for j in range(lo, hi + 1):
            img = ui_frames[j].copy()
            vx0, vy0 = VIDEO_RECT[0], VIDEO_RECT[1]
            cv2.rectangle(img, (vx0 + 14, vy0 + 14), (vx0 + 185, vy0 + 58),
                          (40, 40, 200), -1)
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ImageDraw.Draw(pil).text((vx0 + 26, vy0 + 21),
                                     "慢速回放", font=F(22), fill=(255, 255, 255))
            save(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR), times=2)
        print(f"回放段 {lo}-{hi} -> {replay_start:.1f}s")

    outro = make_title_frame(
        "端侧 AI · 无需联网", "RK3576 NPU · YOLOv8n-Pose fp16 · 单帧推理约 60ms",
        ["开源地址: github.com/james-liu18/rk3576-demos"], h=comp.h)
    for _ in range(int(OUTRO_DUR * FPS)):
        save(outro)

    # ---- 音轨: 主段跌倒告警 + 回放段告警 ----
    siren = os.path.join(ROOT, "assets", "alarm.wav")
    if not os.path.exists(siren):
        from alarm import generate_siren_wav
        generate_siren_wav(siren)
    total_s = saved / FPS
    marks = [(ev_t, 7.0)]
    if event_i is not None:
        marks.append((main_start + lo / FPS + 0.3, 3.0))
    wav_path = build_audio(siren, total_s, marks)

    out_mp4 = os.path.join(args.out_dir, "demo_fall_detect.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(FPS), "-i", os.path.join(seq_dir, "%05d.jpg"),
        "-i", wav_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k", "-shortest", out_mp4], check=True)
    print(f"完成: {out_mp4}  ({total_s:.1f}s, 事件在第 {ev_t:.1f}s)")


if __name__ == "__main__":
    main()
