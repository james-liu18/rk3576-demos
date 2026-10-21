#!/usr/bin/env bash
# 下载开源 fast-neural-style ONNX 模型 (ONNX model zoo)
# 用法: 在有外网的编译机上执行  bash tools/fetch_onnx_models.sh
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)/models/onnx"
mkdir -p "$DIR"
BASE="https://github.com/onnx/models/raw/main/validated/vision/style_transfer/fast_neural_style/model"
for m in mosaic-9 candy-9 rain-princess-9 udnie-9 pointilism-9; do
    out="$DIR/$(echo "$m" | tr '-' '_').onnx"
    if [ -s "$out" ] && [ "$(stat -c%s "$out")" -gt 1000000 ]; then
        echo "已存在: $out"
        continue
    fi
    echo "下载 $m ..."
    wget -q --no-check-certificate -O "$out" "$BASE/$m.onnx"
    echo "  -> $out ($(du -h "$out" | cut -f1))"
done
echo "完成. 模型输入为固定尺寸, 转换时会自动读取."
