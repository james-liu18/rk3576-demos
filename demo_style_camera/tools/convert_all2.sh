#!/bin/bash
# 批量转换新风格模型 ONNX -> RKNN (编译机, rknn-toolkit2 2.3.2 venv)
# 2.5.0 对 TF 导出图的 fusion 规则有 bookkeeping bug (KeyError _tp_rs#1_norm_rs_mul),
# 2.3.2 正常; 板上 librknnrt 2.5.0 向后兼容 2.3.2 产物 (已回归验证)
# anime 系: fp16, 256x256, mean/std=127.5; oil: imagenet norm; ink: CycleGAN anime norm
set -e
cd /tmp/style_camera_src

VENV=/tmp/rknn232venv
if [ ! -f "$VENV/bin/python3" ]; then
    python3 -m venv --system-site-packages "$VENV"
    "$VENV/bin/pip" -q install rknn-toolkit2==2.3.2
fi
PY="$VENV/bin/python3"
$PY -c "from rknn.api import RKNN; print('toolkit 2.3.2 ready')"
CONV="tools/convert_style.py"

conv() {  # conv <onnx> <rknn名> <norm>
    local onnx="models/onnx/$1" out="models/$2.rknn" norm="$3"
    if [ -s "$out" ]; then echo "已存在, 跳过: $out"; return 0; fi
    echo "===== $onnx -> $out (norm=$norm) ====="
    $PY $CONV --onnx "$onnx" --out "$out" --size 256 256 --norm "$norm" --verify
}

conv anime_hayao.onnx      cartoon   anime
conv anime_shinkai.onnx    shinkai   anime
conv anime_paprika.onnx    paprika   anime
conv jp_face.onnx          portrait  anime
conv portrait_sketch.onnx  sketch    anime
conv rain_princess_9.onnx  oil       imagenet
conv ink.onnx              ink       anime
# face_paint512 (bryandlee) 放弃: 图内 depth-to-space 的 Reshape/Resize
# 常量按 512 折叠死, 改 256 需成倍改写空间常量且 toolkit 常数折叠仍踩坑,
# 且效果与 portrait (JP_face) 高度重合, 不值得

echo ALL_CONVERTED
ls -l models/*.rknn
