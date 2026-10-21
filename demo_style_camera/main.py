#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""摄像头预览 + 风格化特效 + 拍照本地存储 Demo 主程序 (RK3576).

用法:
  python3 main.py                    # 默认 UI 运行 (板子/USB 摄像头)
  python3 main.py --no-ui            # 无界面后台运行
  python3 main.py --list-styles      # 列出当前可用特效模式
  python3 main.py --test-style IMG   # 对静态图片应用全部特效 (PC 验证用)
  python3 main.py --test-camera      # 测摄像头取流帧率
"""
import argparse
import os
import queue
import signal
import sys
import threading
import time

import cv2
import numpy as np
import yaml

from stylizer import Stylizer
from ui import AppState, make_ui


class App:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.state = AppState()
        self.mode = "run"          # run / test_camera / test_style

        self.stylizer = Stylizer(cfg.get("style", {}),
                                 cfg.get("sticker", {}))
        self.modes = self.stylizer.available_modes()
        self.mode_keys = [k for k, _ in self.modes]
        self.mode_labels = [lb for _, lb in self.modes]
        self.current_mode = "none"

        save = cfg.get("save", {})
        self.save_dir = save.get("dir", "photos")
        self.save_quality = int(save.get("quality", 95))
        self.save_original = bool(save.get("save_original", False))
        os.makedirs(self.save_dir, exist_ok=True)

        self.cap = None
        self._stop = threading.Event()
        self._worker = None
        self._save_q = queue.Queue(maxsize=8)
        self._saver = None
        self.cam_index = args.camera if args.camera is not None else \
            cfg["camera"].get("index", 0)

        if args.test_camera:
            self.mode = "test_camera"
        elif args.test_style:
            self.mode = "test_style"

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
        print(f"[camera] 已打开 /dev/video{self.cam_index} "
              f"实际分辨率 {int(w)}x{int(h)}")

    # ---------------- 模式/拍照 (UI 线程调用) ----------------
    def set_mode_by_index(self, idx):
        if 0 <= idx < len(self.mode_keys):
            self.current_mode = self.mode_keys[idx]
            print(f"[style] 切换到: {self.mode_labels[idx]}")

    def set_mode_by_key(self, key):
        if key in self.mode_keys:
            self.current_mode = key

    def capture(self):
        """请求拍一张 (worker 线程在下一帧保存)."""
        self.state.capture_request = True

    # ---------------- 工作线程: 取流 + 风格化 ----------------
    def _worker_loop(self):
        frame_count, t0 = 0, time.time()
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                print("[camera] 读帧失败, 重试...")
                time.sleep(0.1)
                continue

            styled, ms = self.stylizer.stylize(frame, self.current_mode)

            if self.state.capture_request:
                self.state.capture_request = False
                try:
                    self._save_q.put_nowait(
                        (styled, frame if self.save_original else None))
                except queue.Full:
                    print("[save] 保存队列满, 丢弃本次拍照")

            frame_count += 1
            now = time.time()
            if now - t0 >= 1.0:
                with self.state.lock:
                    self.state.fps = frame_count / (now - t0)
                frame_count, t0 = 0, now

            with self.state.lock:
                self.state.infer_ms = ms
                self.state.mode_label = dict(self.modes).get(
                    self.current_mode, self.current_mode)
                if styled.shape[1] > 1280:
                    styled = cv2.resize(
                        styled, (1280, int(1280 * styled.shape[0]
                                           / styled.shape[1])))
                self.state.frame = styled

    # ---------------- 保存线程 ----------------
    def _saver_loop(self):
        while True:
            item = self._save_q.get()
            if item is None:
                return
            styled, original = item
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.save_dir, f"style_{ts}.jpg")
            ok = cv2.imwrite(path, styled,
                             [cv2.IMWRITE_JPEG_QUALITY, self.save_quality])
            if ok and original is not None:
                cv2.imwrite(
                    os.path.join(self.save_dir, f"orig_{ts}.jpg"), original,
                    [cv2.IMWRITE_JPEG_QUALITY, self.save_quality])
            desc = "已保存" if ok else "保存失败"
            print(f"[save] {desc}: {path}")
            with self.state.lock:
                self.state.photo_count += 1
                self.state.last_photo = path if ok else ""
                self.state.events.append(
                    (time.strftime("%H:%M:%S"), desc + ": "
                     + os.path.basename(path)))
                self.state.events = self.state.events[-50:]

    # ---------------- 测试模式 ----------------
    def _run_test_camera(self):
        self.open_camera()
        print("测帧率 5 秒 (Ctrl+C 退出)...")
        n, t0 = 0, time.time()
        while time.time() - t0 < 5.0:
            ok, frame = self.cap.read()
            if ok:
                n += 1
        print(f"摄像头帧率: {n / (time.time() - t0):.1f} fps, "
              f"分辨率 {frame.shape[1]}x{frame.shape[0]}")
        return False

    def _run_test_style(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"读图失败: {img_path}")
        out_dir = "test_out"
        os.makedirs(out_dir, exist_ok=True)
        name = os.path.splitext(os.path.basename(img_path))[0]
        for key, label in self.modes:
            # 各模式独立拷贝: 贴纸模式在输入数组上原地作画, 共用会相互叠加
            out, ms = self.stylizer.stylize(img.copy(), key)
            p = os.path.join(out_dir, f"{name}_{key}.jpg")
            cv2.imwrite(p, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"  {label:<16} {ms:6.1f}ms -> {p}")
        return False

    # ---------------- 生命周期 ----------------
    def start(self):
        if self.mode == "test_camera":
            return self._run_test_camera()
        if self.mode == "test_style":
            return self._run_test_style(self._test_img)

        self.open_camera()
        self._saver = threading.Thread(target=self._saver_loop,
                                       name="saver", daemon=True)
        self._saver.start()
        self._worker = threading.Thread(target=self._worker_loop,
                                        name="worker", daemon=True)
        self._worker.start()
        return True

    def shutdown(self, *_):
        if self._stop.is_set():
            return
        print("\n[main] 正在退出...")
        self._stop.set()
        self.state.quit = True
        # 先等工作线程收尾再释放资源
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        try:
            if self._saver:
                self._save_q.put_nowait(None)
                self._saver.join(timeout=2.0)
            self.stylizer.release()
            if self.cap:
                self.cap.release()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="摄像头预览+风格化特效+拍照存储 Demo")
    parser.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    parser.add_argument("--camera", type=int, help="覆盖摄像头索引")
    parser.add_argument("--no-ui", action="store_true", help="无界面运行")
    parser.add_argument("--list-styles", action="store_true",
                        help="列出可用特效模式后退出")
    parser.add_argument("--test-camera", action="store_true",
                        help="测摄像头帧率 5s")
    parser.add_argument("--test-style", metavar="IMG",
                        help="对静态图片应用全部特效 (无需摄像头)")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    app = App(cfg, args)
    app._test_img = getattr(args, "test_style", None)

    if args.list_styles:
        print("可用特效模式:")
        for i, (k, lb) in enumerate(app.modes):
            print(f"  [{i}] {k:<14} {lb}")
        return

    signal.signal(signal.SIGINT, app.shutdown)
    signal.signal(signal.SIGTERM, app.shutdown)

    try:
        if not app.start():
            return
        ui = make_ui(app.state, app, cfg.get("ui", {}), no_ui=args.no_ui,
                     modes=app.mode_labels)
        ui.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
