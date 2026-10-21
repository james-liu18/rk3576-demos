#!/usr/bin/env bash
# AnimeGANv2 系模型: 修正人像模型 + 全部转 rknn (编译机执行)
set -e
cd /tmp/style_camera_src
mkdir -p models models/onnx

# --- 人像动漫模型: JP_face (人像动漫化) -> Disney -> 兜底保留 Hayao_36 ---
BASE="https://github.com/TachibanaYoshino/AnimeGANv3/releases/download/v1.1.0"
if [ ! -s models/onnx/face_paint.onnx ] || \
   grep -qa "Hayao_36" models/onnx/face_paint.onnx 2>/dev/null; then
    for f in AnimeGANv3_JP_face_v1.0.onnx AnimeGANv3_Disney_v1.onnx; do
        echo "尝试 $f ..."
        if curl -skL -o /tmp/fp.onnx "$BASE/$f" && [ "$(stat -c%s /tmp/fp.onnx)" -gt 1000000 ]; then
            mv /tmp/fp.onnx models/onnx/face_paint.onnx
            echo "人像模型用: $f"
            break
        fi
    done
fi
ls -l models/onnx/

# --- 转换 (fp16, 256x256, anime 归一化, 模拟器自验) ---
for m in anime_hayao anime_shinkai face_paint; do
    if [ -f "models/onnx/$m.onnx" ]; then
        python3 tools/convert_style.py \
            --onnx "models/onnx/$m.onnx" \
            --out "models/$m.rknn" \
            --size 256 256 --norm anime --verify
    else
        echo "跳过 $m (onnx 缺失)"
    fi
done
echo ALL_CONVERTED
ls -l models/*.rknn
