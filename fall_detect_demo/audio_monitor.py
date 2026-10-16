#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""异常声音检测.

通过 arecord 读取麦克风 PCM (S16_LE 单声道), 滑窗计算 RMS 能量 (dBFS):
  触发条件: 连续 min_chunks 个窗口  RMS > 绝对阈值 且 高于自适应底噪 rel 阈值
同时维护 10s 原始 PCM 环形缓冲, 供事件录像保存音频.
"""
import math
import subprocess
import threading
import time
from collections import deque

import numpy as np


class AudioMonitor:
    def __init__(self, cfg, on_event=None):
        self.cfg = cfg
        self.on_event = on_event          # callback(level_db)
        self.level_db = -90.0             # 当前窗口能量, 供 UI 显示
        self.noise_floor_db = -50.0
        self.event_count = 0
        self._stop = threading.Event()
        self._thread = None
        self.proc = None

        self.sr = cfg.get("sample_rate", 16000)
        self.chunk_s = cfg.get("chunk_s", 0.2)
        self.chunk = int(self.sr * self.chunk_s)
        self.abs_thr = cfg.get("abs_threshold_db", -28)
        self.rel_thr = cfg.get("rel_threshold_db", 12)
        self.alpha = cfg.get("noise_floor_alpha", 0.02)
        self.min_chunks = cfg.get("min_chunks", 2)
        self.cooldown_s = cfg.get("cooldown_s", 8)

        self.devices = [cfg.get("device", "default")] + \
            list(cfg.get("fallback_devices", []))
        self._dev_idx = 0

        ring_seconds = 12
        self.ring = deque(maxlen=int(ring_seconds * self.sr / self.chunk))
        self._lock = threading.Lock()
        self._sinks = []                  # 持续音频块接收者 (录像用)

    # ---------------- lifecycle ----------------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="audio", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def _open_arecord(self):
        # 多设备轮询: 板载麦被桌面 pipewire 占用时自动切备用设备
        dev = self.devices[self._dev_idx % len(self.devices)]
        cmd = [
            "arecord", "-D", dev,
            "-f", "S16_LE", "-r", str(self.sr),
            "-c", str(self.cfg.get("channels", 1)), "-t", "raw", "-q",
        ]
        print(f"[audio] 打开录音设备 {dev}")
        return subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)

    # ---------------- worker ----------------
    def _run(self):
        while not self._stop.is_set():
            try:
                self.proc = self._open_arecord()
            except FileNotFoundError:
                print("[audio] 未找到 arecord, 请安装: apt install alsa-utils")
                return
            self._read_loop()
            if not self._stop.is_set():  # arecord 意外退出则重开
                time.sleep(1.0)

    def _read_loop(self):
        assert self.proc and self.proc.stdout
        hot = 0
        last_event = 0.0
        chunks_seen = 0
        warmup = max(1, int(3.0 / self.chunk_s))  # 启动热身期只学底噪不触发
        while not self._stop.is_set():
            raw = self.proc.stdout.read(self.chunk * 2)
            if not raw or len(raw) < self.chunk * 2:
                # 从未读到数据说明设备打不开(被占/不存在), 换下一个设备
                if chunks_seen == 0:
                    self._dev_idx += 1
                break  # 供外层重连
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

            # RMS dBFS
            rms = float(np.sqrt(np.mean(np.square(samples))) + 1e-12)
            db = 20.0 * math.log10(rms)
            self.level_db = db

            with self._lock:
                self.ring.append(raw)
            for sink in list(self._sinks):
                try:
                    sink(raw)
                except Exception:
                    pass

            # 底噪跟踪: 首块直接初始化为实测值(避免初始-50导致启动误触发), 之后缓慢上/下降
            if chunks_seen == 0:
                self.noise_floor_db = db
            elif db > self.noise_floor_db:
                self.noise_floor_db += self.alpha * (db - self.noise_floor_db)
            else:
                self.noise_floor_db += self.alpha * 0.5 * (db - self.noise_floor_db)
            chunks_seen += 1

            now = time.time()
            if chunks_seen > warmup and db >= self.abs_thr \
                    and db >= self.noise_floor_db + self.rel_thr:
                hot += 1
            else:
                hot = max(0, hot - 1)

            if hot >= self.min_chunks and now - last_event >= self.cooldown_s:
                last_event = now
                self.event_count += 1
                hot = 0
                if self.on_event:
                    try:
                        self.on_event(round(db, 1))
                    except Exception as e:
                        print(f"[audio] 事件回调异常: {e}")

    # ---------------- 供录像取音频 ----------------
    def get_ring_pcm(self):
        with self._lock:
            return b"".join(self.ring)

    def subscribe_chunks(self, sink_fn):
        """注册持续音频块接收函数 (事件后录制用), 在读线程内同步调用."""
        self._sinks.append(sink_fn)

    def unsubscribe_chunks(self, sink_fn):
        if sink_fn in self._sinks:
            self._sinks.remove(sink_fn)
