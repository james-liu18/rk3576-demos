#!/bin/bash
# 下载新版特色风格 ONNX 模型 (编译机执行, 有外网)
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)/models/onnx"
mkdir -p "$DIR"

dl() {  # dl <url> <out>
    local out="$DIR/$2"
    if [ -s "$out" ] && [ "$(stat -c%s "$out")" -gt 1000000 ]; then
        echo "已存在: $2"
        return 0
    fi
    echo "下载 $2 ..."
    curl -skL --retry 2 -o "$out" "$1" && [ "$(stat -c%s "$out")" -gt 1000000 ] \
        && echo "  -> $2 ($(stat -c%s "$out"))" \
        || { echo "  失败: $2"; rm -f "$out"; return 1; }
}

# 今敏风 (复古赛璐璐动画)
dl "https://huggingface.co/vumichien/AnimeGANv2_Paprika/resolve/main/AnimeGANv2_Paprika.onnx" \
   anime_paprika.onnx || true
# 人像动漫 (AnimeGANv3 JP_face)
dl "https://huggingface.co/vumichien/AnimeGANv3_JP_face/resolve/main/AnimeGANv3_JP_face.onnx" \
   jp_face.onnx || true
# 素描 (AnimeGANv3 PortraitSketch)
dl "https://huggingface.co/vumichien/AnimeGANv3_PortraitSketch/resolve/main/AnimeGANv3_PortraitSketch.onnx" \
   portrait_sketch.onnx || true

echo "---- 当前 onnx: ----"
ls -l "$DIR" | awk '{print $5, $9}'
