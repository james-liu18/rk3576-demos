#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CIPGAN 水墨画 CycleGAN 生成器 .pth -> ONNX 导出 (编译机运行).

用法: python3 tools/export_ink_onnx.py [ckpt名] [输出onnx]
默认: CIP_dataset1 -> models/onnx/ink.onnx (256x256, 输入输出均 [-1,1])
"""
import importlib.util
import os
import sys

import torch
import torch.nn as nn

CKPT_DIR = "/tmp/CIPGAN/checkpoints"
NET_SRC = "/tmp/CIPGAN/models/networks.py"


class Wrapper(nn.Module):
    """去掉 forward 的 map 参数 (非 attention 版本不用 map)."""

    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, x):
        return self.g(x, None)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "CIP_dataset1"
    out = sys.argv[2] if len(sys.argv) > 2 else "models/onnx/ink.onnx"
    spec = importlib.util.spec_from_file_location("cipgan_networks", NET_SRC)
    nw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nw)

    net = nw.ResnetGenerator(3, 3, False, 64,
                             norm_layer=nn.InstanceNorm2d, use_dropout=False,
                             n_blocks=9, padding_type="reflect")
    sd = torch.load(os.path.join(CKPT_DIR, name, "latest_net_G.pth"),
                    map_location="cpu")
    net.load_state_dict(sd)
    net.eval()
    print(f"加载 {name} 权重成功 ({len(sd)} 个参数张量)")

    wrapper = Wrapper(net).eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = wrapper(x)
    print(f"torch 前向 OK: out shape={tuple(y.shape)} "
          f"range=[{y.min():.3f}, {y.max():.3f}]")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.onnx.export(wrapper, x, out, opset_version=11,
                      input_names=["input"], output_names=["output"],
                      do_constant_folding=True)
    print(f"导出完成: {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
