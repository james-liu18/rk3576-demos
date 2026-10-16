#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""告警控制器: LED 闪烁 + 喇叭循环播放警笛.

LED 支持四种模式:
  gpiod      - libgpiod (兼容 v1/v2 Python API), 需要知道 gpiochip 与 line 号
  sysfs_gpio - /sys/class/gpio 传统接口
  sysfs_led  - 内核已注册的 /sys/class/leds/<name> (brightness/trigger)
  pwm        - 裸 PWM sysfs (M2 板载红灯 D0407: PWM0_CH0 = /sys/class/pwm/pwmchip0,
               无 LED class、不能当 GPIO; 顺序 period->polarity->duty->enable,
               polarity 必须 normal)
  none       - 不控灯 (无 LED 硬件时)

喇叭: aplay 循环播放 assets/alarm.wav (首次运行自动生成警笛音).
"""
import math
import os
import shutil
import subprocess
import threading
import time
import wave

import numpy as np


# ------------------------------------------------------------------
# LED 控制
# ------------------------------------------------------------------
class Led:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = cfg.get("mode", "none")
        self.dev = None       # gpiod v1/v2 句柄
        self.line = cfg.get("line", 0)
        self.chip = cfg.get("chip", 0)
        self.active_low = cfg.get("active_low", False)
        self._on = False
        self._init_mode()

    def _init_mode(self):
        if self.mode == "gpiod":
            try:
                self._init_gpiod()
                print(f"[led] gpiod 模式 gpiochip{self.chip} line={self.line}")
            except Exception as e:
                print(f"[led] gpiod 初始化失败({e}), 降级为 sysfs_gpio")
                self.mode = "sysfs_gpio"
                self._init_mode()
        elif self.mode == "sysfs_gpio":
            path = f"/sys/class/gpio/gpio{self.line}"
            if not os.path.exists(path):
                with open("/sys/class/gpio/export", "w") as f:
                    f.write(str(self.line))
                with open(f"{path}/direction", "w") as f:
                    f.write("out")
            print(f"[led] sysfs_gpio 模式 gpio{self.line}")
        elif self.mode == "sysfs_led":
            print(f"[led] sysfs_led 模式: {self.led_path()}")
        elif self.mode == "pwm":
            self._init_pwm()
        else:
            print("[led] 未启用 LED 控制 (mode=none)")

    # ---- pwm sysfs (rockchip pwm v4: 顺序 period->polarity->duty->enable) ----
    def _init_pwm(self):
        # pwm 模式下: chip=pwmchipN 编号, line=pwm 通道号 (M2 LED=pwmchip0/pwm0)
        self.pwm_dir = f"/sys/class/pwm/pwmchip{self.chip}/pwm{self.line}"
        self.period = self.cfg.get("period", 200000)
        try:
            if not os.path.exists(self.pwm_dir):
                with open(f"/sys/class/pwm/pwmchip{self.chip}/export", "w") as f:
                    f.write(str(self.line))

            def w(name, val):
                # period=0 (刚 export) 时写 enable 会 I/O error, 属预期, 容错跳过
                try:
                    with open(f"{self.pwm_dir}/{name}", "w") as f:
                        f.write(str(val))
                    return True
                except OSError as e:
                    print(f"[led] pwm 写 {name}={val} 失败: {e}")
                    return False

            w("period", self.period)              # 必须最先写 period
            w("polarity", "normal")               # 默认 inversed, 不改点不亮
            w("duty_cycle", 0)
            w("enable", 0)
            print(f"[led] pwm 模式 pwmchip{self.chip}/pwm{self.line} "
                  f"period={self.period}")
        except (PermissionError, FileNotFoundError, OSError) as e:
            print(f"[led] pwm 初始化失败: {e}, LED 不启用")
            self.mode = "none"

    def led_path(self):
        return f"/sys/class/leds/{self.cfg.get('name', 'work')}"

    def _init_gpiod(self):
        import gpiod  # noqa
        path = f"/dev/gpiochip{self.chip}"
        if hasattr(gpiod, "Chip"):  # libgpiod v1 API
            chip = gpiod.Chip(path)
            line = chip.get_line(self.line)
            line.request(consumer="fall_detect_demo",
                         type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])
            self.dev = ("v1", chip, line)
        else:                       # libgpiod v2 API
            from gpiod.line import Direction, Value
            req = gpiod.request_lines(
                path, consumer="fall_detect_demo",
                config={self.line: gpiod.LineSettings(
                    direction=Direction.OUTPUT,
                    output_value=Value.INACTIVE)})
            self.dev = ("v2", req)

    def set(self, on: bool):
        if self._on == on:
            return
        self._on = on
        val = (not on) if self.active_low else on
        try:
            if self.mode == "pwm":
                # 亮: duty=period 全占空; 灭: enable=0
                with open(f"{self.pwm_dir}/duty_cycle", "w") as f:
                    f.write(str(self.period if val else 0))
                with open(f"{self.pwm_dir}/enable", "w") as f:
                    f.write("1" if val else "0")
            elif self.mode == "gpiod" and self.dev:
                api = self.dev[0]
                if api == "v1":
                    _, _, line = self.dev
                    line.set_value(1 if val else 0)
                else:
                    from gpiod.line import Value
                    _, req = self.dev
                    req.set_value(self.line,
                                  Value.ACTIVE if val else Value.INACTIVE)
            elif self.mode == "sysfs_gpio":
                with open(f"/sys/class/gpio/gpio{self.line}/value", "w") as f:
                    f.write("1" if val else "0")
            elif self.mode == "sysfs_led":
                # 关闭内核 trigger 后直接写 brightness
                trig = f"{self.led_path()}/trigger"
                if os.access(trig, os.W_OK):
                    with open(trig, "w") as f:
                        f.write("none")
                with open(f"{self.led_path()}/brightness", "w") as f:
                    f.write("255" if val else "0")
        except (PermissionError, FileNotFoundError) as e:
            print(f"[led] 写入失败: {e} (需要 root 或相应权限)")

    def release(self):
        self.set(False)
        if self.mode == "pwm":
            try:
                with open(f"{self.pwm_dir}/enable", "w") as f:
                    f.write("0")
                with open(f"/sys/class/pwm/pwmchip{self.chip}/unexport", "w") as f:
                    f.write(str(self.line))
            except (OSError, FileNotFoundError):
                pass
        if self.dev and self.dev[0] == "v1":
            try:
                self.dev[2].release()
                self.dev[1].close()
            except Exception:
                pass


# ------------------------------------------------------------------
# 警笛音生成
# ------------------------------------------------------------------
def generate_siren_wav(path, sr=16000, seconds=2.0):
    """生成 2s 扫频警笛 (600~1200Hz), 16bit 单声道."""
    n = int(sr * seconds)
    t = np.arange(n) / sr
    f = 900 + 300 * np.sign(np.sin(2 * math.pi * t / seconds))  # 方波调制
    phase = 2 * math.pi * np.cumsum(f) / sr
    sig = 0.75 * np.sin(phase)
    # 淡入淡出防爆音
    env = np.minimum(1.0, np.minimum(t / 0.05, (seconds - t) / 0.05))
    pcm = (sig * env * 32767 * 0.8).astype(np.int16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


# ------------------------------------------------------------------
# 告警控制器
# ------------------------------------------------------------------
class AlarmController:
    def __init__(self, cfg):
        self.duration = cfg.get("duration_s", 10)
        self.led = Led(cfg.get("led", {}))
        self.speaker_cfg = cfg.get("speaker", {})
        self.wav = self.speaker_cfg.get("wav", "assets/alarm.wav")
        self.muted = False
        self._until = 0.0
        self._stop_evt = threading.Event()
        self._thread = None
        self.active = False

        if self.speaker_cfg.get("mode", "aplay") == "aplay" and \
                not os.path.exists(self.wav):
            try:
                generate_siren_wav(self.wav)
                print(f"[alarm] 已生成警笛音: {self.wav}")
            except Exception as e:
                print(f"[alarm] 警笛音生成失败: {e}")

    # ---------------- 外部接口 ----------------
    def trigger(self, duration=None):
        """触发一次告警 (LED 闪烁 + 声音), 重复触发会刷新时长."""
        self._until = time.time() + (duration or self.duration)
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, name="alarm", daemon=True)
        self._thread.start()

    def stop(self):
        self._until = 0.0
        self._stop_evt.set()

    def toggle_mute(self):
        self.muted = not self.muted
        return self.muted

    # ---------------- 内部 ----------------
    def _run(self):
        self.active = True
        ap = None
        period = 0.35  # LED 闪烁周期的一半
        try:
            while time.time() < self._until and not self._stop_evt.is_set():
                # LED: 一直闪
                self.led.set(int(time.time() / period) % 2 == 0)
                # 声音: 未静音时循环 aplay
                if not self.muted and self.speaker_cfg.get("mode", "aplay") == "aplay" \
                        and shutil.which("aplay"):
                    if ap is None or ap.poll() is not None:
                        ap = subprocess.Popen(
                            ["aplay", "-q", "-D",
                             self.speaker_cfg.get("device", "default"), self.wav],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                time.sleep(0.05)
        finally:
            self.led.set(False)
            if ap and ap.poll() is None:
                ap.terminate()
            self.active = False

    def release(self):
        self.stop()
        self._stop_evt.wait(timeout=2)
        self.led.release()
