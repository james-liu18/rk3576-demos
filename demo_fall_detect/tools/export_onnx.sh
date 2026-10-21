#!/bin/bash
# ============================================================
# PC 端: 下载 YOLO-Pose 预训练模型并导出 ONNX
# 用法: ./tools/export_onnx.sh [n|s] [8|11]
#   ./tools/export_onnx.sh n 8    -> yolov8n-pose (默认)
#   ./tools/export_onnx.sh s 11   -> yolov11s-pose
# ============================================================
set -e
SIZE=${1:-n}
VER=${2:-8}

cd "$(dirname "$0")/.."
mkdir -p models

pip install "ultralytics>=8.2"

if [ "$VER" = "11" ]; then
    WEIGHT="yolo11${SIZE}-pose.pt"
else
    WEIGHT="yolov8${SIZE}-pose.pt"
fi

python3 - <<EOF
from ultralytics import YOLO
model = YOLO("${WEIGHT}")
path = model.export(format="onnx", imgsz=640, opset=12, simplify=True)
print("导出成功:", path)
EOF

cp ${WEIGHT%.pt}.onnx models/
echo "完成: models/${WEIGHT%.pt}.onnx"
echo "PC 调试:  python3 main.py --model models/${WEIGHT%.pt}.onnx"
echo "转 RKNN:  python3 tools/convert.py models/${WEIGHT%.pt}.onnx rk3576"
