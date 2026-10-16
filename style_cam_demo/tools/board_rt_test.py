#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""板上实时风格化回归: 逐个特效跑 10 秒, 打印 fps 与单帧耗时.
用法: board_rt_test.py KEY [KEY...]"""
import sys
import time


def main():
    import cv2
    import yaml
    from stylizer import Stylizer

    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    keys = sys.argv[1:]
    st = Stylizer(cfg.get("style", {}), cfg.get("sticker", {}))
    avail = dict(st.available_modes())
    for k in keys:
        if k not in avail:
            print(f"[skip] {k} 不可用")
            continue
        cap = cv2.VideoCapture(cfg["camera"]["index"], cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
        n, ms_sum, t0 = 0, 0.0, time.time()
        while time.time() - t0 < 10.0:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            _, ms = st.stylize(frame, k)
            n += 1
            ms_sum += ms
        cap.release()
        el = time.time() - t0
        print(f"[rt] {k:16s} fps={n / el:5.1f}  infer={ms_sum / max(n, 1):6.1f}ms",
              flush=True)


if __name__ == "__main__":
    main()
