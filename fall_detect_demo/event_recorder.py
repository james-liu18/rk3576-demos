#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""事件录像: 视频环形缓冲 + 事件触发保存 10s 音视频.

- 视频帧由推理线程 push (带时间戳), 环形缓冲保留 pre_seconds
- 事件到达后: 写出回溯帧 + 持续采集新帧至 clip_seconds, mp4 格式
- 音频: 事件前从 AudioMonitor 环形缓冲取回溯 PCM, 之后订阅实时块
- 音视频用 ffmpeg 合并为单个 mp4 (视频流copy, 音频转AAC); ffmpeg 不可用时退回分开的 mp4+wav
- 所有事件追加写入 events.jsonl
"""
import json
import os
import subprocess
import threading
import time
from collections import deque

import cv2


class EventRecorder:
    def __init__(self, cfg, audio_monitor=None):
        self.cfg = cfg
        self.save_dir = cfg.get("save_dir", "records")
        self.clip_seconds = cfg.get("clip_seconds", 10)
        self.pre_seconds = cfg.get("pre_seconds", 4)
        self.annotated = cfg.get("save_annotated", True)
        self.save_audio = cfg.get("save_audio", True) and audio_monitor is not None
        self.fps = cfg.get("fps", 25)
        self.audio = audio_monitor

        os.makedirs(self.save_dir, exist_ok=True)
        self.frames = deque(maxlen=int(self.pre_seconds * 40))  # 按帧数, 上限40fps
        self.events = deque(maxlen=100)
        self._lock = threading.Lock()
        self._q = deque()
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="recorder",
                                        daemon=True)
        self._thread.start()

    # ---------------- 数据入口 (推理线程调用) ----------------
    def push_frame(self, frame, annotated=None):
        with self._lock:
            self.frames.append((time.time(), frame, annotated))

    def push_event(self, etype, extra=None):
        """事件入口: fall / sound / manual"""
        ts = time.time()
        with self._cv:
            self._q.append((ts, etype, extra or {}))
            self._cv.notify()
        self.events.append({"ts": ts, "type": etype, **(extra or {})})
        return self._event_path(ts, etype)

    def recent_events(self):
        return list(self.events)

    # ---------------- worker ----------------
    def _worker(self):
        while not self._stop.is_set():
            with self._cv:
                if not self._q:
                    self._cv.wait(timeout=0.5)
                if not self._q:
                    continue
                ts, etype, extra = self._q.popleft()
            try:
                self._save_clip(ts, etype, extra)
            except Exception as e:
                print(f"[record] 保存失败: {e}")

    def _save_clip(self, ts, etype, extra):
        day = time.strftime("%Y%m%d", time.localtime(ts))
        out_dir = os.path.join(self.save_dir, day)
        os.makedirs(out_dir, exist_ok=True)
        name = f"{etype}_{time.strftime('%H%M%S', time.localtime(ts))}"
        vpath = os.path.join(out_dir, name + ".mp4")
        apath = os.path.join(out_dir, name + ".wav")

        # ---- 视频帧收集: 回溯 + 前向 ----
        with self._lock:
            backlog = list(self.frames)
        cutoff = ts - self.pre_seconds
        frames = [f for f in backlog if f[0] >= cutoff]
        collected = list(frames)
        need = int(self.clip_seconds * self.fps)
        # 继续收集新帧至满足时长 (最多等 clip+5 秒)
        last_ts = collected[-1][0] if collected else ts
        deadline = time.time() + self.clip_seconds + 5
        while len(collected) < need and time.time() < deadline:
            with self._lock:
                new = [f for f in self.frames if f[0] > last_ts]
            if new:
                collected.extend(new)
                last_ts = new[-1][0]
            else:
                time.sleep(1.0 / self.fps)
        # 限速补齐/裁剪
        if not collected:
            print("[record] 无帧可写, 跳过")
            return

        h, w = collected[0][1].shape[:2]
        n_video = 0
        writer = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"),
                                 self.fps, (w, h))
        for fts, raw, ann in collected:
            img = ann if (self.annotated and ann is not None) else raw
            writer.write(img)
            n_video += 1
        writer.release()

        # ---- 音频 ----
        if self.save_audio and self.audio is not None:
            pcm = bytearray()
            # 回溯部分: 环形缓冲比预录窗长, 只取尾部 pre_seconds, 否则
            # pcm[:target] 会把事件后的实时音频全部截掉
            tail = int(self.pre_seconds * self.audio.sr) * 2  # bytes (s16)
            pcm.extend(self.audio.get_ring_pcm()[-tail:])
            # 前向部分: 订阅实时块直到总时长满足
            target = int(self.clip_seconds * self.audio.sr) * 2  # bytes (s16)
            got_event = threading.Event()

            def sink(b):
                pcm.extend(b)
                if len(pcm) >= target:
                    got_event.set()

            self.audio.subscribe_chunks(sink)
            got_event.wait(timeout=self.clip_seconds + 5)
            self.audio.unsubscribe_chunks(sink)

            import wave
            with wave.open(apath, "wb") as wf:
                wf.setnchannels(self.audio.cfg.get("channels", 1))
                wf.setsampwidth(2)
                wf.setframerate(self.audio.sr)
                wf.writeframes(bytes(pcm[:target]))

            # ---- 合并为单个 mp4: 视频流直接copy, 音频转AAC ----
            # 注意: 不能用 -shortest, 音频空/极短时会把输出截成0帧(RC仍为0);
            # 用 -t 明确取视频时长; 音频为空则只保留视频
            if os.path.getsize(apath) > 100 and n_video > 0:
                try:
                    tmp = vpath + ".mux.mp4"
                    subprocess.run(
                        ["ffmpeg", "-y", "-loglevel", "error",
                         "-i", vpath, "-i", apath,
                         "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                         "-t", f"{n_video / self.fps:.3f}", tmp],
                        check=True, timeout=60)
                    os.replace(tmp, vpath)
                    os.remove(apath)
                    apath = None
                except Exception as e:
                    print(f"[record] 音视频合并失败({e}), 保留分开的 mp4+wav")
            else:
                os.remove(apath)
                apath = None
                print("[record] 无有效音频, 仅保存视频")

        # ---- 事件日志 ----
        rec = {"ts": ts, "iso": time.strftime("%Y-%m-%d %H:%M:%S",
                                              time.localtime(ts)),
               "type": etype, "video": vpath, "audio": apath if self.save_audio else None,
               **extra}
        with open(os.path.join(self.save_dir, "events.jsonl"), "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[record] 已保存 {etype} 事件: {vpath}"
              + (f" + {os.path.basename(apath)}" if (self.save_audio and apath) else ""))

    def _event_path(self, ts, etype):
        day = time.strftime("%Y%m%d", time.localtime(ts))
        name = f"{etype}_{time.strftime('%H%M%S', time.localtime(ts))}"
        return os.path.join(self.save_dir, day, name + ".mp4")

    def stop(self):
        self._stop.set()
        with self._cv:
            self._cv.notify()
        self._thread.join(timeout=3)
