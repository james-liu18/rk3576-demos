#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跌倒检测 + 异常声音告警 + 本地存储 Demo 主程序.

用法:
  python3 main.py                      # 默认 UI 运行
  python3 main.py --no-ui              # 无界面后台运行
  python3 main.py --test-alarm         # 仅测试 LED+喇叭告警 5s
  python3 main.py --test-audio         # 实时打印麦克风能量 (标定阈值用)
"""
import argparse
import os
import signal
import sys
import threading
import time

import cv2
import numpy as np
import yaml

from alarm import AlarmController
from audio_monitor import AudioMonitor
from event_recorder import EventRecorder
from fall_detector import FallDetector
from ui import AppState, make_ui
from ui import annotate_frame


class App:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.state = AppState()
        self.mode = "run"          # run / test_alarm / test_audio

        # ---- 各模块 ----
        self.alarm = AlarmController(cfg.get("alarm", {}))
        self.audio = AudioMonitor(cfg.get("audio", {}), on_event=self._on_sound)
        self.recorder = EventRecorder(cfg.get("record", {}), self.audio)
        self.fall_det = FallDetector(cfg.get("fall", {}))
        self.detector = None       # 延迟创建 (报错信息更友好)
        self.cap = None

        self._stop = threading.Event()
        self._infer_thread = None
        self._events_ui = []
        self.model_path = args.model or cfg["model"]["path"]
        self.cam_index = args.camera if args.camera is not None else \
            cfg["camera"].get("index", 0)

        if args.test_alarm:
            self.mode = "test_alarm"
        elif args.test_audio:
            self.mode = "test_audio"

    # ---------------- 摄像头 ----------------
    def open_camera(self):
        c = self.cfg["camera"]
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"摄像头打开失败: index={self.cam_index}")
        cap.set(cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*c.get("fourcc", "MJPG")))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.get("width", 1280))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.get("height", 720))
        cap.set(cv2.CAP_PROP_FPS, c.get("fps", 30))
        self.cap = cap
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[camera] 已打开 /dev/video{self.cam_index} 实际分辨率 {int(w)}x{int(h)}")

    # ---------------- 事件处理 ----------------
    def _on_sound(self, level_db):
        # 声音事件只记录不拉警笛 (仅跌倒触发告警)
        self._fire_event("sound", f"异常声音 {level_db}dB", alarm=False)

    def trigger_alarm(self, duration=5):
        print("[alarm] 手动测试告警")
        self.alarm.trigger(duration)
        with self.state.lock:
            self.state.alarm_active = True

    def toggle_mute(self):
        m = self.alarm.toggle_mute()
        with self.state.lock:
            self.state.muted = m
        print(f"[alarm] 静音: {m}")

    def manual_record(self):
        self.recorder.push_event("manual", {"desc": "手动触发录像"})
        print("[record] 手动触发 10s 录像")

    # ---------------- 推理线程 ----------------
    def _infer_loop(self):
        frame_count, t0 = 0, time.time()
        try:
            while not self._stop.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    print("[camera] 读帧失败, 重试...")
                    time.sleep(0.1)
                    continue

                dets, ms = self.detector.infer(frame)
                event, state, fallen_det = self.fall_det.update(
                    dets, frame.shape[:2])
                if event == "fall":
                    self._fire_event("fall", "检测到人体跌倒!")
                elif state == "DOWN" and not self.alarm.active:
                    # 跌倒未恢复期间持续告警
                    self.alarm.trigger()

                # 注意用 identity 查找: dict == 比较含 numpy 数组会抛异常
                fi = next((i for i, d in enumerate(dets) if d is fallen_det),
                          None)
                annotated = annotate_frame(frame, dets, fallen_idx=fi)
                self.recorder.push_frame(frame, annotated)

                frame_count += 1
                now = time.time()
                if now - t0 >= 1.0:
                    with self.state.lock:
                        self.state.fps = frame_count / (now - t0)
                    frame_count, t0 = 0, now

                with self.state.lock:
                    self.state.infer_ms = ms
                    self.state.person_count = len(dets)
                    self.state.fall_state = state
                    self.state.sound_db = self.audio.level_db
                    self.state.noise_db = self.audio.noise_floor_db
                    self.state.alarm_active = self.alarm.active
                    disp = cv2.resize(annotated, (960, int(960 * frame.shape[0] / frame.shape[1]))) \
                        if frame.shape[1] > 1280 else annotated
                    self.state.frame = disp
        finally:
            pass

    def _fire_event(self, etype, desc, alarm=True):
        ts = time.strftime("%H:%M:%S")
        with self.state.lock:
            self._events_ui.append((ts, desc))
            self.state.events = list(self._events_ui[-50:])
        print(f"[event] {ts} {desc}")
        self.recorder.push_event(etype)
        if alarm:
            self.alarm.trigger()

    # ---------------- 生命周期 ----------------
    def start(self):
        if self.mode == "test_alarm":
            print("测试告警 5 秒 (LED + 喇叭)...")
            self.alarm.trigger(5)
            time.sleep(6)
            self.shutdown()
            return False

        if self.mode == "test_audio":
            print("实时声音能量 (Ctrl+C 退出). 对着麦克风说话/拍手观察 dB 变化:")
            self.audio.start()
            while True:
                print(f"\r  当前: {self.audio.level_db:6.1f} dB | "
                      f"底噪: {self.audio.noise_floor_db:6.1f} dB | "
                      f"事件: {self.audio.event_count}", end="")
                time.sleep(0.5)
                if self._stop.is_set():
                    break
            return False

        self.open_camera()
        from pose_detector import PoseDetector
        try:
            self.detector = PoseDetector(
                self.model_path,
                input_size=tuple(self.cfg["model"].get("input_size", [640, 640])),
                conf_threshold=self.cfg["model"].get("conf_threshold", 0.35),
                kpt_threshold=self.cfg["model"].get("kpt_threshold", 0.4),
                nms_threshold=self.cfg["model"].get("nms_threshold", 0.45),
                core_mask=self.cfg["model"].get("core_mask", 0))
        except Exception as e:
            raise RuntimeError(
                f"模型加载失败: {e}\n请先完成模型转换, 详见 README.md") from e

        if self.cfg.get("audio", {}).get("enabled", True):
            self.audio.start()
        self._infer_thread = threading.Thread(target=self._infer_loop,
                                              name="inference", daemon=True)
        self._infer_thread.start()
        return True

    def shutdown(self, *_):
        if self._stop.is_set():
            return
        print("\n[main] 正在退出...")
        self._stop.set()
        self.state.quit = True
        # 先等推理线程收尾再释放资源, 否则退出时 librknnrt 可能段错误
        if self._infer_thread and self._infer_thread.is_alive():
            self._infer_thread.join(timeout=2.0)
        try:
            self.audio.stop()
            self.alarm.release()
            self.recorder.stop()
            if self.detector:
                self.detector.release()
            if self.cap:
                self.cap.release()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="跌倒检测+异常声音告警 Demo")
    parser.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    parser.add_argument("--model", help="覆盖配置中的模型路径 (.rknn/.onnx)")
    parser.add_argument("--camera", type=int, help="覆盖摄像头索引")
    parser.add_argument("--no-ui", action="store_true", help="无界面运行")
    parser.add_argument("--test-alarm", action="store_true", help="测试 LED+喇叭")
    parser.add_argument("--test-audio", action="store_true", help="标定声音阈值")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    app = App(cfg, args)

    if args.test_alarm or args.test_audio:
        try:
            app.start()
        except KeyboardInterrupt:
            pass
        finally:
            app.shutdown()
        return

    signal.signal(signal.SIGINT, app.shutdown)
    signal.signal(signal.SIGTERM, app.shutdown)

    try:
        if not app.start():
            return
        ui = make_ui(app.state, app, cfg.get("ui", {}), no_ui=args.no_ui)
        ui.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
