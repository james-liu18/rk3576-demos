#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PC 端: ONNX -> RKNN 转换 (目标平台 rk3576).

环境 (x86 PC, python3.8~3.12):
    pip install rknn-toolkit2>=2.3.0

用法:
    python3 tools/convert.py models/yolov8n-pose.onnx rk3576
    python3 tools/convert.py models/yolov8n-pose.onnx rk3576 fp models/yolo_pose_fp.rknn
    python3 tools/convert.py models/yolov8n-pose.onnx rk3576 i8   # int8 量化(更快)

注意:
  - 默认输出 fp16 不量化, 精度优先; int8 需要提供量化校准图片 (见 dataset.txt)
  - 若追求更高 int8 精度, 可参考 rknn_model_zoo yolov8_pose 的混合量化两步法
"""
import argparse
import os
import sys

from rknn.api import RKNN

DEFAULT_RKNN = "models/yolo_pose.rknn"


def parse_arg():
    p = argparse.ArgumentParser()
    p.add_argument("onnx", help="输入 onnx 模型路径")
    p.add_argument("platform", help="目标平台, 如 rk3576")
    p.add_argument("dtype", nargs="?", default="fp",
                   help="fp(默认,fp16) 或 i8(int8量化)")
    p.add_argument("output", nargs="?", default=DEFAULT_RKNN)
    return p.parse_args()


def main():
    args = parse_arg()
    if not os.path.exists(args.onnx):
        sys.exit(f"找不到模型: {args.onnx}")

    rknn = RKNN(verbose=False)

    print("--> config")
    ret = rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                      target_platform=args.platform)
    if ret != 0:
        sys.exit("config 失败")

    print("--> load_onnx")
    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        sys.exit("load_onnx 失败")

    print(f"--> build (dtype={args.dtype})")
    if args.dtype == "i8":
        dataset = os.path.join(os.path.dirname(__file__), "dataset.txt")
        if not os.path.exists(dataset):
            sys.exit("int8 量化需要 dataset.txt (每行一张校准图片路径)")
        ret = rknn.build(do_quantization=True, dataset=dataset)
    else:
        ret = rknn.build(do_quantization=False)
    if ret != 0:
        sys.exit("build 失败")

    print("--> export_rknn")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ret = rknn.export_rknn(args.output)
    if ret != 0:
        sys.exit("export_rknn 失败")

    rknn.release()
    print(f"\n转换完成: {args.output}")
    print("拷贝到板端 models/ 目录后运行: python3 main.py")


if __name__ == "__main__":
    main()
