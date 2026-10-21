#!/usr/bin/env python3
"""测量 UI 模板中各动态元素的真实坐标."""
import sys
import cv2
import numpy as np

tpl = cv2.imread(sys.argv[1])
h, w = tpl.shape[:2]
print(f"template: {w}x{h}")
gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

# ---- 视频区: 中央亮内容 bbox ----
mask = (gray > 75).astype(np.uint8)
sub = mask[80:int(h * 0.85), 0:int(w * 0.86)]
cols = sub.sum(axis=0)
rows = sub.sum(axis=1)
xs = np.where(cols > 60)[0]
ys = np.where(rows > 60)[0]
vx0, vx1 = int(xs[0]), int(xs[-1])
vy0, vy1 = int(ys[0]) + 80, int(ys[-1]) + 80
print(f"VIDEO_RECT = ({vx0}, {vy0}, {vx1}, {vy1})")

# ---- 徽标: 右上绿色块 ----
b, g, r = tpl[:, :, 0].astype(int), tpl[:, :, 1].astype(int), tpl[:, :, 2].astype(int)
green = ((g > 110) & (r < 90) & (b < 110)).astype(np.uint8)
q = green[0:int(h * 0.4), int(w * 0.75):]
ys2, xs2 = np.nonzero(q)
if len(xs2):
    bx0, bx1 = int(xs2.min()) + int(w * 0.75), int(xs2.max()) + int(w * 0.75)
    by0, by1 = int(ys2.min()), int(ys2.max())
    print(f"BADGE_RECT = ({bx0-4}, {by0-4}, {bx1+4}, {by1+4})")

# ---- 面板数值: 徽标下方白色文字行聚类 ----
px0 = bx0 if len(xs2) else int(w * 0.86)
white = ((r > 190) & (g > 190) & (b > 190)).astype(np.uint8)
col = white[by1 + 10:int(h * 0.75), px0 - 8:int(w - 10)]
rowsum = col.sum(axis=1)
on = rowsum > 3
segs = []
s = None
for i, v in enumerate(on):
    if v and s is None:
        s = i
    elif not v and s is not None:
        if i - s > 8:
            segs.append((s + by1 + 10, i + by1 + 10))
        s = None
for j, (a, bb) in enumerate(segs):
    print(f"val[{j}] y=({a},{bb})")

# ---- 声音条: 面板内橙色块 ----
orange = ((r > 180) & (g > 80) & (g < 170) & (b < 80)).astype(np.uint8)
q3 = orange[int(h * 0.3):int(h * 0.75), px0 - 8:w - 10]
ys3, xs3 = np.nonzero(q3)
if len(xs3):
    print(f"BAR_RECT = ({int(xs3.min())+px0-8}, {int(ys3.min())+int(h*0.3)}, "
          f"{int(xs3.max())+px0-8}, {int(ys3.max())+int(h*0.3)})")
