#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""风格迁移 ONNX -> RKNN 转换 (在编译机/Ubuntu x86 上运行).

用法:
  # AnimeGANv2 系 (输入输出均 [-1,1], 缩到 256 提速):
  python3 tools/convert_style.py --onnx models/onnx/anime_hayao.onnx \
      --out models/anime_hayao.rknn --size 256 256 --norm anime --verify
  # 批量:
  python3 tools/convert_style.py --dir models/onnx --outdir models \
      --target rk3576 --size 256 256 --norm anime --verify

依赖 (编译机): pip3 install rknn-toolkit2==2.5.0 onnx
说明:
  - 输入尺寸自动从 onnx 读取 (NCHW/NHWC 自动识别); --size 可覆盖, 会直接
    改写 onnx 图输入维度 (AnimeGANv2 全卷积, 512->256 安全, 需 32 的倍数)
  - --norm anime: mean=std=127.5 (等价 x/127.5-1); 默认 imagenet (旧
    fast-neural-style 用)
  - fp16 非量化
  - 板端实测 (RK3576, librknnrt 2.5.0): opset9 图存在两个致命问题,
    必须做图手术, 见 fix_graph():
      1) reflect/edge Pad 在 NPU 上导致内存损坏, 第二次推理段错误
      2) ReLU 融合路径输出恒为 0; LeakyRelu(alpha=1.0) 数学等价可绕过
      int8 量化对 InstanceNorm 网络输出全 0, 不可用
  - --verify: 导出后在模拟器里跑一次随机输入, 打印输出范围 (抓归一化/
    输出尺度配错; NPU 特有 bug 需板上 board_trunc.py 复核)
"""
import argparse
import os

MEAN_IMAGENET = [[0.485, 0.456, 0.406]]
STD_IMAGENET = [[0.229, 0.224, 0.225]]
MEAN_ANIME = [[127.5, 127.5, 127.5]]
STD_ANIME = [[127.5, 127.5, 127.5]]


def fix_graph(onnx_path, fixed_path):
    """图手术: reflect/edge Pad -> constant, Relu -> LeakyRelu(1e-6).

    绕开 RK3576 NPU 的两个 bug (reflect pad 内存损坏 / ReLU 融合输出恒 0).
    注意: alpha 必须 ~0 而不是 1.0! LeakyRelu(1.0) 是恒等映射, 会把
    ReLU 非线性整个删掉 (曾导致 ink/oil 输出全毁); 1e-6 数值上等价 ReLU.
    """
    import onnx
    m = onnx.load(onnx_path)
    n_pad = n_relu = 0
    for node in m.graph.node:
        if node.op_type == 'Pad':
            for attr in node.attribute:
                if attr.name == 'mode' and attr.s in (b'reflect', b'edge'):
                    attr.s = b'constant'
                    n_pad += 1
        elif node.op_type == 'Relu':
            node.op_type = 'LeakyRelu'
            del node.attribute[:]
            a = node.attribute.add()
            a.name = 'alpha'
            a.type = onnx.AttributeProto.FLOAT
            a.f = 1e-6
            n_relu += 1
    print(f"    图手术: pad->{n_pad}, relu->leakyrelu {n_relu} 处")
    onnx.save(m, fixed_path)
    return fixed_path


def patch_input_size(onnx_path, out_path, h, w):
    """改写 onnx 图输入 H/W (全卷积模型缩输入尺寸用, 如 512->256)."""
    import onnx
    m = onnx.load(onnx_path)
    dims = m.graph.input[0].type.tensor_type.shape.dim
    if len(dims) != 4:
        raise RuntimeError(f"输入不是 4 维: {[d.dim_value for d in dims]}")
    vals = [d.dim_value for d in dims]
    if vals[1] == 3:            # NCHW
        dims[2].dim_value = h
        dims[3].dim_value = w
    elif vals[3] == 3:          # NHWC
        dims[1].dim_value = h
        dims[2].dim_value = w
    else:
        raise RuntimeError(f"无法识别输入布局: {vals}")
    onnx.save(m, out_path)
    return out_path


def nchw_wrap(onnx_path, out_path, h, w):
    """NHWC 输入的模型包一层 Transpose 变成 NCHW 输入.

    rknn-toolkit2 对 onnx 输入严格按 NCHW 解释 (NHWC 模型会把 C 认成 H),
    图首插入 Transpose(0,2,3,1) 后即可按 NCHW 声明输入.
    """
    import onnx
    from onnx import TensorProto, helper
    m = onnx.load(onnx_path)
    g = m.graph
    orig = g.input[0]
    orig_name = orig.name
    dims = [d.dim_value for d in orig.type.tensor_type.shape.dim]
    assert len(dims) == 4 and dims[3] == 3, f"非 NHWC 输入: {dims}"
    new_in = helper.make_tensor_value_info(
        "nchw_input", TensorProto.FLOAT, [1, 3, h, w])
    tp = helper.make_node("Transpose", ["nchw_input"], [orig_name],
                          perm=[0, 2, 3, 1], name="nchw_to_nhwc")
    g.node.insert(0, tp)
    orig.CopyFrom(new_in)
    onnx.save(m, out_path)
    return out_path


def _perm_of(node):
    for a in node.attribute:
        if a.name == "perm":
            return list(a.ints)
    return None


def _reorder_pads_nhwc_to_nchw(node, inits):
    """把 Pad 节点的 pads (NHWC 语义 8 元素) 重排为 NCHW 语义.

    TF 导出图的 pads 常量按 NHWC 排列 [bN,bH,bW,bC,eN,eH,eW,eC],
    截掉边界 Transpose 后数据已是 NCHW, pads 须重排为 [bN,bC,bH,bW,...].
    """
    import numpy as np
    import onnx

    def reorder(vals):
        b, e = list(vals[:4]), list(vals[4:])
        return [b[0], b[3], b[1], b[2], e[0], e[3], e[1], e[2]]

    # 属性式 (opset<11)
    for a in node.attribute:
        if a.name == "pads" and len(a.ints) == 8:
            new = reorder(a.ints)
            del a.ints[:]
            a.ints.extend(new)
            return True
    # 输入式 (opset>=11): pads 是 initializer 常量
    if len(node.input) >= 2:
        for init in inits:
            if init.name == node.input[1]:
                arr = onnx.numpy_helper.to_array(init).flatten()
                assert len(arr) == 8, f"pads 长度异常: {arr}"
                new = np.array(reorder(arr), dtype=arr.dtype)
                new_init = onnx.numpy_helper.from_array(
                    new.reshape(init.dims) if list(init.dims) else new,
                    init.name)
                init.CopyFrom(new_init)
                return True
    return False


def truncate_nchw_boundary(onnx_path, out_path, h, w):
    """TF 导出的 NHWC 边界图, 若内部是 NCHW (输入/输出各有一个边界
    Transpose), 直接截掉两个 Transpose, 让图的输入输出都是 NCHW.
    支持输入侧 graph_input -> Pad -> Transpose 结构 (hayao: MirrorPad
    在 Transpose 之前), 此时 pads 常量同步重排 NHWC->NCHW.
    返回 (path, nchw_out_name) / 不适用时返回 (None, None).
    """
    import onnx
    from onnx import TensorProto, helper
    m = onnx.load(onnx_path)
    g = m.graph
    in_name = g.input[0].name
    out_name = g.output[0].name
    t_in = t_out = pad_in = None
    # 输入侧: graph_input -> [Pad] -> Transpose(NHWC->NCHW)
    for n in g.node:
        if n.op_type != "Transpose" or _perm_of(n) != [0, 3, 1, 2]:
            continue
        if in_name in n.input:
            t_in = n
            break
        # 前面垫了一个 Pad (MirrorPad 等)
        for p in g.node:
            if p.op_type == "Pad" and p.output and \
                    p.output[0] in n.input and in_name in p.input:
                t_in = n
                pad_in = p
                break
        if t_in:
            break
    for n in g.node:
        if n.op_type == "Transpose" and out_name in n.output:
            if _perm_of(n) == [0, 2, 3, 1]:              # NCHW -> NHWC
                t_out = n
            break
    if t_in is None or t_out is None:
        return None, None
    if pad_in is not None:
        ok = _reorder_pads_nhwc_to_nchw(pad_in, g.initializer)
        assert ok, "Pad pads 常量重排失败"
        pad_in.input[0] = in_name
        print(f"    输入侧 Pad({pad_in.name}) 保留, pads 重排 NHWC->NCHW")
    # 新 graph input 沿用原名 (Pad/后续节点引用的是这个名字, 改名会断链)
    new_in = helper.make_tensor_value_info(
        in_name, TensorProto.FLOAT, [1, 3, h, w])
    g.input[0].CopyFrom(new_in)
    nchw_out = t_out.input[0]
    # 删掉两个边界 Transpose
    g.node.remove(t_in)
    g.node.remove(t_out)
    # 重定向 t_in 的消费者: 有前置 Pad 时接到 Pad 输出 (否则会架空 Pad,
    # 特征图尺寸塌缩, 输出比输入小 8px); 无 Pad 才直接接新 graph input
    reroute = pad_in.output[0] if pad_in is not None else in_name
    for n in g.node:
        for i, name in enumerate(n.input):
            if name == t_in.output[0]:
                n.input[i] = reroute
    new_out = helper.make_tensor_value_info(
        nchw_out, TensorProto.FLOAT, [1, 3, h, w])
    g.output[0].CopyFrom(new_out)
    onnx.save(m, out_path)
    print(f"    截掉边界 Transpose: 输入 NCHW 1x3x{h}x{w}, 输出 {nchw_out}")
    return out_path, nchw_out


def onnx_input_info(onnx_path):
    """读取 onnx 输入 -> (h, w, layout); 动态维度返回 None 尺寸."""
    import onnx
    m = onnx.load(onnx_path)
    shape = m.graph.input[0].type.tensor_type.shape
    dims = [d.dim_value if d.HasField("dim_value") else None
            for d in shape.dim]
    if len(dims) != 4:
        return None, None, "nchw"
    if dims[1] == 3:
        return dims[2], dims[3], "nchw"
    if dims[3] == 3:
        return dims[1], dims[2], "nhwc"
    return None, None, "nchw"


def sim_verify(rknn, w, h):
    """模拟器跑一次随机输入, 打印输出范围 (抓归一化/输出尺度配错)."""
    import numpy as np
    try:
        if rknn.init_runtime() != 0:
            print("    [verify] 模拟器 init_runtime 失败, 跳过")
            return
        x = np.random.randint(0, 256, (1, h, w, 3), dtype=np.uint8)
        out = rknn.inference(inputs=[x])
        o = np.asarray(out).squeeze().astype(np.float32)
        print(f"    [verify] 输出 shape={o.shape} "
              f"range=[{o.min():.1f}, {o.max():.1f}] mean={o.mean():.1f}")
        if abs(float(o.max()) - float(o.min())) < 1e-3:
            print("    [verify] !! 输出恒定值, 转换有问题")
        elif o.min() >= -1.5 and o.max() <= 1.5:
            print("    [verify] 输出为 [-1,1] 尺度 -> 板端用 out_range: pm1")
        elif o.min() >= -5 and o.max() <= 260:
            print("    [verify] 输出为 0~255 尺度 -> 板端用 out_range: image255")
        else:
            print("    [verify] !! 输出尺度异常, 请人工检查")
    except Exception as e:
        print(f"    [verify] 异常 (不影响产物): {e}")


def convert_one(onnx_path, out_path, target, size=None, quant=False,
                dataset=None, norm="imagenet", verify=False):
    import onnx
    from rknn.api import RKNN
    mean, std = (MEAN_ANIME, STD_ANIME) if norm == "anime" \
        else (MEAN_IMAGENET, STD_IMAGENET)
    dh, dw, layout = onnx_input_info(onnx_path)
    h, w = size or (dh, dw) or (224, 224)
    print(f"\n==> {os.path.basename(onnx_path)}  输入 {w}x{h} ({layout})")

    src = onnx_path
    outputs = None
    if size and (dh, dw) != (h, w):
        patched = onnx_path + f'.{h}x{w}.onnx'
        patch_input_size(onnx_path, patched, h, w)
        print(f"    输入尺寸 {dh}x{dw} -> {h}x{w} (全卷积改写)")
        src = patched
    keep_nhwc = layout == "nhwc" and os.environ.get("KEEP_NHWC") == "1"
    nhwc_in_name = nhwc_out_name = None
    if keep_nhwc:
        _m = onnx.load(src)
        nhwc_in_name = _m.graph.input[0].name
        nhwc_out_name = _m.graph.output[0].name
        print(f"    KEEP_NHWC: 布局保持 NHWC 直转 "
              f"(in={nhwc_in_name}, out={nhwc_out_name})")
    if layout == "nhwc" and not keep_nhwc:
        # 首选: 内部 NCHW 的图直接截掉 NHWC 边界 Transpose (hayao/shinkai)
        trunc, nchw_out = truncate_nchw_boundary(src, src + '.trunc.onnx',
                                                 h, w)
        if trunc:
            src = trunc
            outputs = [nchw_out]
        else:
            # 兜底: 图首包 Transpose 变 NCHW 输入 (内部 NHWC 的图)
            wrapped = src + '.nchw.onnx'
            nchw_wrap(src, wrapped, h, w)
            print(f"    NHWC 输入 -> 包 Transpose 转 NCHW ({w}x{h})")
            src = wrapped
            # 输出若是 NHWC (图尾 Transpose perm=[0,2,3,1]), 截到它的 NCHW 输入
            m = onnx.load(src)
            out_name = m.graph.output[0].name
            for node in m.graph.node:
                if node.op_type == "Transpose" and node.output \
                        and node.output[0] == out_name:
                    perm = [a.ints for a in node.attribute
                            if a.name == "perm"]
                    if perm and list(perm[0]) == [0, 2, 3, 1]:
                        outputs = [node.input[0]]
                        print(f"    输出 NHWC -> 截到 NCHW: {outputs[0]}")
                    break

    rknn = RKNN(verbose=False)
    # toolkit 融合规则 bug (官方提示用 disable_rules 跳过):
    # - fuse_pad_into_conv: "assignment destination is read-only" (2.5.0)
    # - fuse_align_conv_forward/backward: KeyError '..._tp_rs#1_norm_rs_mul'
    #   (分解式 LayerNorm + conv 触发, 系列簿记 bug, 2.3.2/2.5.0 皆然)
    # 不融合只是少了优化, constant Pad 本身 NPU 安全
    rules = ["fuse_pad_into_conv",
             "fuse_align_conv_forward",
             "fuse_align_conv_backward",
             "remove_parallel_transpose",
             "absorb_gather_into_attention_conv"]
    # DISABLE_NORM_RULES=1: 禁掉全部 LayerNorm/InstanceNorm 融合规则,
    # 保留分解式 norm (排查原生算子/融合簿记 bug 用)
    if os.environ.get("DISABLE_NORM_RULES") == "1":
        rules += ["convert_layernorm_to_transpose_layernorm",
                  "convert_layernorm_to_reshape_layernorm",
                  "convert_layernorm_to_exlayernorm",
                  "convert_layernorm_to_exnorm",
                  "convert_exlayernorm_to_transpose_layernorm",
                  "convert_in_to_exnorm",
                  "convert_exnorm_to_exnorm_mul_add",
                  "fuse_norm", "fuse_exnorm_add", "fuse_exnorm_mul",
                  "fuse_exnorm_mul_add", "bypass_norm", "cvt_norm",
                  "instance_norm", "layer_norm", "mean_var_norm"]
    rknn.config(mean_values=mean, std_values=std, target_platform=target,
                disable_rules=rules)
    fixed = src + '.fixed.onnx'
    fix_graph(src, fixed)
    # SKIP_SIM=1: 跳过 onnxsim (排查简化引入的问题)
    if os.environ.get("SKIP_SIM") != "1":
        try:
            import onnxsim
            model = onnx.load(fixed)
            sim, ok = onnxsim.simplify(model)
            if ok:
                fixed = src + '.fixed.sim.onnx'
                onnx.save(sim, fixed)
                print("    onnxsim 简化完成")
            else:
                print("    onnxsim 简化失败, 用原图继续")
        except ImportError:
            print("    未安装 onnxsim (pip install onnxsim), 跳过简化")
    # 分解式 InstanceNorm -> 原生 InstanceNormalization:
    # toolkit 融合 TF 分解式 LayerNorm 有簿记 bug, 融合后直接支持
    # NO_FUSE_NORM=1: 跳过原生 InstanceNormalization 改写, 保留分解式
    # (用于排查 NPU 上原生 InstanceNormalization 算子的问题)
    # KEEP_NHWC 时也跳过: 分解式 norm 保持 TF 原生形态交给 toolkit
    if os.environ.get("NO_FUSE_NORM") != "1" and not keep_nhwc:
        try:
            from fuse_instance_norm import fuse_instance_norm
            inorm = fixed + '.inorm.onnx'
            if fuse_instance_norm(fixed, inorm):
                fixed = inorm
        except RuntimeError as e:
            print(f"    InstanceNorm 融合失败, 用原图继续: {e}")
    else:
        print("    NO_FUSE_NORM=1: 保留分解式 InstanceNorm")
    if keep_nhwc:
        rc = rknn.load_onnx(model=fixed, inputs=[nhwc_in_name],
                            input_size_list=[[1, h, w, 3]],
                            outputs=[nhwc_out_name])
    else:
        rc = rknn.load_onnx(model=fixed, outputs=outputs)
    if rc != 0:
        raise RuntimeError("load_onnx 失败")
    if rknn.build(do_quantization=quant, dataset=dataset) != 0:
        raise RuntimeError("build 失败")
    if rknn.export_rknn(out_path) != 0:
        raise RuntimeError("export_rknn 失败")
    print(f"<== {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    if verify:
        sim_verify(rknn, w, h)
    rknn.release()
    print(f"    config.yaml: out_range={'pm1' if norm == 'anime' else 'image255'}"
          f", input_size=[{w}, {h}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", help="单个 onnx 文件")
    ap.add_argument("--dir", help="批量转换目录下所有 .onnx")
    ap.add_argument("--out", help="输出 .rknn 路径 (--onnx 模式)")
    ap.add_argument("--outdir", default="models", help="批量输出目录")
    ap.add_argument("--target", default="rk3576")
    ap.add_argument("--size", nargs=2, type=int, metavar=("H", "W"),
                    help="覆盖输入尺寸 (全卷积模型直接改写图输入)")
    ap.add_argument("--norm", choices=("imagenet", "anime"), default="imagenet",
                    help="anime=mean/std 127.5 (AnimeGANv2)")
    ap.add_argument("--quant", action="store_true",
                    help="int8 量化 (需 --dataset 校准图片列表)")
    ap.add_argument("--dataset", help="量化校准图片列表 txt")
    ap.add_argument("--verify", action="store_true",
                    help="导出后模拟器跑一次验证输出范围")
    args = ap.parse_args()

    size = tuple(args.size) if args.size else None
    if args.onnx:
        out = args.out or os.path.splitext(
            os.path.basename(args.onnx))[0] + ".rknn"
        convert_one(args.onnx, out, args.target, size=size, quant=args.quant,
                    dataset=args.dataset, norm=args.norm, verify=args.verify)
    elif args.dir:
        os.makedirs(args.outdir, exist_ok=True)
        onnx_files = sorted(f for f in os.listdir(args.dir)
                            if f.endswith(".onnx") and "." not in f.rsplit(
                                ".", 1)[0])
        if not onnx_files:
            raise SystemExit(f"{args.dir} 下没有 onnx 文件, "
                             f"先跑 tools/fetch_anime_models.py")
        for f in onnx_files:
            name = os.path.splitext(f)[0]
            convert_one(os.path.join(args.dir, f),
                        os.path.join(args.outdir, name + ".rknn"),
                        args.target, size=size, quant=args.quant,
                        dataset=args.dataset, norm=args.norm,
                        verify=args.verify)
    else:
        ap.error("需要 --onnx 或 --dir")


if __name__ == "__main__":
    main()
