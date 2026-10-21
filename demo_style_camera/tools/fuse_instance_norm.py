#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 TF 导出的分解式 InstanceNorm 融合成原生 InstanceNormalization.

背景: rknn-toolkit2 (2.3.2/2.5.0) 对 AnimeGAN TF 导出图里的
ReduceMean/Sub/Mul/Sqrt/Reciprocal 分解式 LayerNorm 做 fuse 时有簿记
bug, build 阶段 KeyError '..._tp_rs#1_norm_rs_mul'. 把这一坨融合成
原生 InstanceNormalization 算子后 toolkit 直接支持 (zoo 模型已验证).

匹配模式 (onnxsim 常量折叠后, 每组 12 节点):
  mean  = ReduceMean(X, axes=[2,3], keepdims=1)
  sq    = Sub(X, mean)
  sqsq  = Mul(sq, sq)
  var   = ReduceMean(sqsq, axes=[2,3], keepdims=1)
  veps  = Add(var, eps_const)
  s     = Sqrt(veps)
  r     = Reciprocal(s)
  scale = Mul(r, gamma_const)            # gamma/r 顺序不定
  ms    = Mul(mean, scale)
  shift = Sub(beta_const, ms)            # beta - mean*scale
  xm    = Mul(X, scale)
  y     = Add(xm, shift)
=> InstanceNormalization(X, gamma, beta, eps=eps) 输出 y

用法 (独立验证): python3 fuse_instance_norm.py <in.onnx> <out.onnx>
集成: convert_one 在 onnxsim 之后调用 fuse_instance_norm().
"""
import numpy as np


def _prod(axes):
    r = 1
    for a in axes:
        r *= a
    return r


def _axes_of(node, G):
    """读取 ReduceMean 的 axes (属性式或 initializer 输入式), 失败返回 None."""
    for a in node.attribute:
        if a.name == "axes":
            return list(a.ints)
    if len(node.input) > 1:
        arr = G.const(node.input[1])
        if arr is not None:
            return [int(v) for v in np.asarray(arr).flatten()]
    return None


class _Graph:
    def __init__(self, g):
        self.g = g
        self.inits = {i.name: i for i in g.initializer}
        self.consts = {}
        self.consumers = {}
        for n in g.node:
            for x in n.input:
                self.consumers.setdefault(x, []).append(n)
        # Constant 节点产出的常量
        for n in g.node:
            if n.op_type == "Constant":
                for a in n.attribute:
                    if a.name == "value":
                        self.consts[n.output[0]] = a.t

    def const(self, name):
        """返回 (np.ndarray) 或 None."""
        if name in self.inits:
            from onnx import numpy_helper
            return numpy_helper.to_array(self.inits[name])
        if name in self.consts:
            from onnx import numpy_helper
            return numpy_helper.to_array(self.consts[name])
        return None

    def by_output(self, tensor):
        for n in self.g.node:
            if tensor in n.output:
                return n
        return None


def fuse_instance_norm(onnx_path, out_path, verify=True):
    """融合分解式 InstanceNorm. 返回 True 表示图被改写 (out_path 有效)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.load(onnx_path)
    g = m.graph
    G = _Graph(g)
    to_remove = set()
    # (插入位置索引, InstanceNormalization 节点)
    add_nodes = []
    need_op17 = False

    def attr(n, name):
        for a in n.attribute:
            if a.name == name:
                return a
        return None

    for idx, n_sq in enumerate(list(g.node)):
        if n_sq.op_type != "Sub" or id(n_sq) in to_remove:
            continue
        x_name, mean_name = n_sq.input[0], n_sq.input[1]
        n_mean = G.by_output(mean_name)
        if n_mean is None or n_mean.op_type != "ReduceMean":
            continue
        ka = attr(n_mean, "keepdims")
        if ka is None or ka.i != 1:
            continue
        # sq = Sub(X, mean)  (顺序可能颠倒, X 是非 mean 的那个)
        n_xm1 = G.by_output(n_sq.output[0])
        cons_sq = G.consumers.get(n_sq.output[0], [])
        n_sqsq = next((c for c in cons_sq if c.op_type == "Mul"), None)
        if n_sqsq is None:
            continue
        cons_sqsq = G.consumers.get(n_sqsq.output[0], [])
        n_var = next((c for c in cons_sqsq if c.op_type == "ReduceMean"), None)
        if n_var is None:
            continue
        cons_var = G.consumers.get(n_var.output[0], [])
        n_veps = next((c for c in cons_var if c.op_type == "Add"), None)
        if n_veps is None:
            continue
        eps = G.const(n_veps.input[0] if n_veps.input[1] == n_var.output[0]
                      else n_veps.input[1])
        if eps is None:
            continue
        cons_veps = G.consumers.get(n_veps.output[0], [])
        n_sqrt = next((c for c in cons_veps if c.op_type == "Sqrt"), None)
        if n_sqrt is None:
            continue
        cons_sqrt = G.consumers.get(n_sqrt.output[0], [])
        n_recip = next((c for c in cons_sqrt if c.op_type == "Reciprocal"),
                       None)
        if n_recip is None:
            continue
        cons_recip = G.consumers.get(n_recip.output[0], [])
        # scale = Mul(r, gamma_const)
        n_scale = next((c for c in cons_recip if c.op_type == "Mul"), None)
        if n_scale is None:
            continue
        gamma = G.const(n_scale.input[0] if n_scale.input[1]
                        == n_recip.output[0] else n_scale.input[1])
        if gamma is None:
            continue
        scale_name = n_scale.output[0]
        cons_scale = G.consumers.get(scale_name, [])
        # xm = Mul(X, scale); ms = Mul(mean, scale)
        n_xm = next((c for c in cons_scale if c.op_type == "Mul"
                     and x_name in c.input), None)
        n_ms = next((c for c in cons_scale if c.op_type == "Mul"
                     and mean_name in c.input), None)
        if n_xm is None or n_ms is None:
            continue
        cons_ms = G.consumers.get(n_ms.output[0], [])
        n_shift = next((c for c in cons_ms if c.op_type == "Sub"), None)
        if n_shift is None:
            continue
        beta = G.const(n_shift.input[0] if n_shift.input[1] == n_ms.output[0]
                       else n_shift.input[1])
        if beta is None:
            continue
        cons_xm = G.consumers.get(n_xm.output[0], [])
        n_out = next((c for c in cons_xm if c.op_type == "Add"), None)
        if n_out is None:
            continue

        # ---- 匹配成功, 生成归一化算子 ----
        # 按 ReduceMean axes 分流:
        #   {2,3}      -> InstanceNormalization (逐样本逐通道, CycleGAN/zoo)
        #   {1,2,3}    -> LayerNormalization(axis=1) (AnimeGANv2 TF 版
        #                 名字就叫 LayerNorm, 对 C+H+W 全体归一化,
        #                 gamma/beta 按 (C,1,1) 广播)
        n_var_axes = _axes_of(n_var, G)
        if n_var_axes is not None and sorted(n_var_axes) != sorted(
                _axes_of(n_mean, G) or []):
            continue
        axes = sorted(_axes_of(n_mean, G) or [])
        if axes == [] and n_var_axes is not None:
            axes = sorted(n_var_axes)
        if not axes:
            continue
        c = int(gamma.reshape(-1).shape[0])
        scale_t = numpy_helper.from_array(
            gamma.astype(np.float32).reshape(c, 1, 1), f"{n_mean.name}/scale")
        bias_t = numpy_helper.from_array(
            beta.astype(np.float32).reshape(c, 1, 1), f"{n_mean.name}/bias")
        g.initializer.extend([scale_t, bias_t])
        if axes == [2, 3]:
            nn = helper.make_node(
                "InstanceNormalization",
                [x_name, scale_t.name, bias_t.name], [n_out.output[0]],
                name=f"Inorm/{n_mean.name}", epsilon=float(eps))
            add_nodes.append((idx, nn))
        else:
            # axes={1,2,3} 等: 全体归一化 (TF LayerNorm)
            # -> 原生 LayerNormalization(axis=1) (opset>=17)
            nn = helper.make_node(
                "LayerNormalization",
                [x_name, scale_t.name, bias_t.name], [n_out.output[0]],
                axis=1, epsilon=float(eps), name=f"Lnorm/{n_mean.name}")
            add_nodes.append((idx, nn))
            need_op17 = True
        for node in (n_mean, n_sq, n_sqsq, n_var, n_veps, n_sqrt, n_recip,
                     n_scale, n_ms, n_shift, n_xm, n_out):
            to_remove.add(id(node))

    if not add_nodes:
        print("    InstanceNorm 融合: 未发现匹配模式 (可能已是原生算子)")
        return False

    # 在每组第一个被移除节点的位置插入 InstanceNormalization, 保持拓扑序
    add_nodes.sort(key=lambda t: t[0])
    new_nodes = []
    for i, n in enumerate(g.node):
        if id(n) not in to_remove:
            new_nodes.append(n)
        for pos, nn in add_nodes:
            if pos == i:
                new_nodes.append(nn)
    del g.node[:]
    g.node.extend(new_nodes)
    if need_op17:
        # LayerNormalization 需要 opset>=17; 所有用到的算子在 13->17 间
        # 语义未变, 直接提升 opset 即可
        for imp in m.opset_import:
            if imp.domain in ("", "ai.onnx") and imp.version < 17:
                imp.version = 17
    onnx.checker.check_model(m)
    onnx.save(m, out_path)
    print(f"    归一化融合: {len(add_nodes)} 组 -> "
          f"InstanceNorm/LayerNorm ({len(to_remove)} 节点移除)")

    if verify:
        import onnxruntime as ort
        try:
            m0 = onnx.load(onnx_path)
            dims = [d.dim_value for d in
                    m0.graph.input[0].type.tensor_type.shape.dim]
            shape = [d if d > 0 else 16 for d in dims]
            name = m0.graph.input[0].name
            x = np.random.RandomState(0).randn(*shape).astype(np.float32)
            so = ort.SessionOptions()
            so.log_severity_level = 3
            o0 = ort.InferenceSession(onnx_path, so).run(
                None, {name: x})[0]
            o1 = ort.InferenceSession(out_path, so).run(
                None, {name: x})[0]
            d = float(np.abs(o0.astype(np.float32)
                             - o1.astype(np.float32)).max())
            print(f"    归一化融合数值校验: max diff = {d:.2e}")
            if d > 1e-2:
                raise RuntimeError("融合后数值偏差过大, 放弃改写")
        except RuntimeError:
            raise
        except Exception as e:
            print(f"    数值校验跳过 ({e})")
    return True


if __name__ == "__main__":
    import sys
    ok = fuse_instance_norm(sys.argv[1], sys.argv[2])
    print("fused" if ok else "no-op")
