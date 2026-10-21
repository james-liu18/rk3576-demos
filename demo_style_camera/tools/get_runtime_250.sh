#!/bin/bash
# 编译机: 检查 toolkit 版本, 下载 librknnrt 2.5.0 runtime, 重转 candy
set -x
python3 -m pip show rknn-toolkit2 | head -2
curl -s --no-check-certificate https://api.github.com/repos/airockchip/rknn-toolkit2/tags -o /tmp/tags.json
python3 - <<'EOF'
import json
tags = json.load(open('/tmp/tags.json'))
print([t['name'] for t in tags][:10])
EOF
# 试几个可能的 raw 路径
for ref in master main v2.5.0; do
  url="https://raw.githubusercontent.com/airockchip/rknn-toolkit2/$ref/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so"
  echo "try $url"
  wget -q --no-check-certificate "$url" -O /tmp/librknnrt_250.so && \
    file /tmp/librknnrt_250.so | grep -q ELF && { echo "OK from $ref"; break; }
done
ls -la /tmp/librknnrt_250.so
file /tmp/librknnrt_250.so
