#!/bin/bash
# 板端: 逐一验证 5 个风格模型 (2 次推理 + 输出范围)
cd /opt/style_cam_demo
for m in candy mosaic rain_princess udnie pointilism; do
  echo "== $m"
  /opt/py312/bin/python3.12 tools/board_trunc.py "models/$m.rknn" 2>&1 | grep -E 'iter [01]|FAILED'
done
echo ALL_DONE
