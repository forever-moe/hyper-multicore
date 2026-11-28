# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Ascend950 MXFP8 PanGu MoE bitwise replay using only HyperMulticore and CANN."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch


def compare_bytes(name, actual, expected):
    import torch

    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {"tensor": name, "bitwise": False, "actual": [list(actual.shape), str(actual.dtype)],
                "expected": [list(expected.shape), str(expected.dtype)]}
    a = actual.detach().contiguous().view(torch.uint8).cpu().reshape(-1)
    b = expected.detach().contiguous().view(torch.uint8).cpu().reshape(-1)
    equal = torch.equal(a, b)
    result = {"tensor": name, "bitwise": equal}
    if not equal:
        different = a != b
        result["different_bytes"] = int(different.sum())
        result["first_different_byte"] = int(different.to(torch.int8).argmax())
    return result


def _run(args):
    import torch
    import torch_npu

    torch.npu.set_device(args.device)
    device_name = torch.npu.get_device_name(args.device)
    if not device_name.startswith("Ascend950"):
        raise RuntimeError(f"This replay requires Ascend950, found {device_name}.")
    import cann_ops_nn  # pylint: disable=unused-import

    from hyper_multicore import MegaMoeExperts, MegaMoeNumericsOptions
    from hyper_multicore.experimental import MegaMoeSchedulerOptions, MegaMoeTuningOptions
    from hyper_multicore.modules.mega_moe.backends.mxfp8 import execution, shared_expert

    device = torch.device("npu", args.device)
    fp8, scale_dtype = torch.float8_e4m3fn, torch_npu.float8_e8m0fnu
    bf16, fp32 = torch.bfloat16, torch.float32
    tokens, hidden, width, experts, top_k = 4096, 3584, 1280, 8, 8
    rng = torch.Generator().manual_seed(5587)

    def sample(shape, scale):
        return (torch.randn(shape, generator=rng) * scale).to(device=device, dtype=bf16)

    def quantize(x, group=None):
        if group is None:
            result = torch_npu.npu_dynamic_mx_quant_with_dual_axis(x, dst_type=fp8, scale_alg=1)
        else:
            result = torch.ops.cann_ops_nn.grouped_dynamic_mx_quant_with_dual_axis(
                x, group, round_mode="rint", scale_alg=1, dst_type=fp8,
            )
        return packed(result)

    def packed(result):
        row, row_scale, col, col_scale = result[:4]
        return row, row_scale.view(torch.uint8), col, col_scale.view(torch.uint8)

    def swiglu(gate_up, group):
        return packed(torch.ops.cann_ops_nn.swiglu_group_quant_with_dual_axis(
            gate_up, weight=None, group_index=group, dst_type=292,
            quant_mode=1, clamp_limit=-1.0, output_origin=False,
        ))

    def swiglu_grad(da, gate_up, group):
        return packed(torch.ops.cann_ops_nn.swiglu_backward_group_quant_with_dual_axis(
            da, gate_up, weight=None, y_origin=None, group_index=group,
            clamp_limit=-1.0, quant_mode=1, dst_type=36,
        ))

    def matmul(a, b, layout, dtype=bf16, group=None):
        if layout[0] == "T":
            a_data, a_scale = a[2].transpose(-1, -2), a[3].transpose(-2, -3)
        else:
            a_data, a_scale = a[:2]
        if layout[1] == "T":
            b_data, b_scale = b[0].transpose(-1, -2), b[1].transpose(-2, -3)
        else:
            b_data, b_scale = b[2:]
        if group is None:
            return torch_npu.npu_quant_matmul(
                a_data, b_data, b_scale, pertoken_scale=a_scale, output_dtype=dtype,
                scale_dtype=scale_dtype, pertoken_scale_dtype=scale_dtype, group_sizes=[1, 1, 32],
            )
        return torch_npu.npu_grouped_matmul(
            [a_data], [b_data], bias=None, scale=[b_scale], per_token_scale=[a_scale],
            group_list=group, group_type=2 if layout == "TN" else 0,
            group_list_type=0, output_dtype=dtype, scale_dtype=scale_dtype,
            per_token_scale_dtype=scale_dtype, split_item=2,
        )[0]

    x = sample((tokens, hidden), 0.25)
    dy = sample((tokens, hidden), 0.125)
    probs = torch.randn((tokens, top_k), generator=rng).softmax(-1).to(device)
    ids = ((torch.arange(tokens)[:, None] + torch.arange(top_k)) % experts).to(device=device, dtype=torch.int32)
    counts = torch.full((experts,), tokens * top_k // experts, device=device, dtype=torch.int64)

    def weights(shapes):
        roots, physical = [], []
        for shape in shapes:
            value = sample(shape, 0.02)
            payload = quantize(value)
            roots.append(torch.nn.Parameter(value.float()))
            physical.append(payload)
        return tuple(roots), tuple(physical)

    routed_roots, routed = weights(((experts, hidden, 2 * width), (experts, width, hidden)))
    shared_roots, shared = weights(((2 * width, hidden), (hidden, width)))

    def native_shared_forward(hidden_states):
        x_mx = quantize(hidden_states)
        gate_up = matmul(x_mx, shared[0], "NT")
        activation = swiglu(gate_up, None)
        return matmul(activation, shared[1], "NT"), (x_mx, gate_up, activation)

    def native_shared_backward(saved, grad):
        x_mx, gate_up, activation = saved
        dy_mx = quantize(grad)
        da = matmul(dy_mx, shared[1], "NN")
        du = swiglu_grad(da, gate_up, None)
        dx = matmul(du, shared[0], "NN")
        dw1, dw2 = matmul(du, x_mx, "TN", fp32), matmul(dy_mx, activation, "TN", fp32)
        return (dx, dw1, dw2), (dy_mx, da, du)

    # EP=1 still exercises PanGu's fused token permutation, grouped MXFP8
    # quantization, split-M/split-K GMM and probability-weighted combine.
    def native_routed():
        group = counts.cumsum(0)
        dispatched, order = torch_npu.npu_moe_token_permute(x, ids)
        input_mx = quantize(dispatched, group)
        up = matmul(input_mx, routed[0], "NN", group=group)
        activation = swiglu(up, group)
        down = matmul(activation, routed[1], "NN", group=group)
        output = torch_npu.npu_moe_token_unpermute(down, order, probs, False, x.shape)
        dy_tokens, dp = torch_npu.npu_moe_token_unpermute_grad(down, dy, order, probs, False, x.shape)
        dy_mx = quantize(dy_tokens, group)
        da = matmul(dy_mx, routed[1], "NT", group=group)
        dw2 = matmul(activation, dy_mx, "TN", fp32, group)
        du = swiglu_grad(da, up, group)
        dx_tokens = matmul(du, routed[0], "NT", group=group)
        dw1 = matmul(input_mx, du, "TN", fp32, group)
        dx = torch_npu.npu_moe_token_permute_grad(x, dx_tokens, ids, order, False)
        return output, dx, dp, dw1, dw2

    class NativeShared(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_arg, w1, w2):
            # Quantized operands are fixed and bound to the supplied logical roots.
            del w1, w2
            output, ctx.saved = native_shared_forward(x_arg)
            return output

        @staticmethod
        def backward(ctx, grad):
            return native_shared_backward(ctx.saved, grad)[0]

    def mega(fused):
        layer = MegaMoeExperts(
            hidden_size=hidden, intermediate_size=width, num_experts=experts, top_k=top_k,
            precision="mxfp8", shared_intermediate_size=width if fused else None,
            numerics_options=MegaMoeNumericsOptions(partial_weight_grad_dtype="float32"),
            tuning_options=MegaMoeTuningOptions(
                dispatch_mode="pull", dispatch_split=4096, combine_split=4096,
                scheduler=MegaMoeSchedulerOptions(queue_ready_lookahead=1),
            ),
        )
        x_arg, probs_arg = x.detach().requires_grad_(), probs.detach().requires_grad_()
        for root, payload in zip(routed_roots + shared_roots, routed + shared):
            root.row_data, root.row_scale, root.col_data, root.col_scale = payload
        try:
            kwargs = {"shared_expert_weights": shared_roots} if fused else {}
            # An intermediate view exposes the routed gradient in the same backward
            # that accumulates both branches into x_arg.
            routed_input = x_arg if fused else x_arg.view_as(x_arg)
            routed_result = layer(routed_input, ids, probs_arg, expert_weights=routed_roots,
                                  tokens_per_expert=counts, **kwargs)
            output = routed_result
            inputs = (x_arg, probs_arg, *routed_roots, *shared_roots)
            if not fused:
                output = routed_result + NativeShared.apply(x_arg, *shared_roots)
                inputs += (routed_input,)
            grads = torch.autograd.grad(output, inputs, dy)
            torch.npu.synchronize()
            if not fused:
                return output, *grads[:-1], routed_result, grads[-1]
            return output, *grads
        finally:
            layer.close()

    native_output, saved = native_shared_forward(x)
    (native_dx, native_dw1, native_dw2), (dy_mx, da, du) = native_shared_backward(saved, dy)
    routed_output, routed_dx, dp, dw1, dw2 = native_routed()
    expected = (routed_output + native_output, routed_dx + native_dx, dp, dw1, dw2, native_dw1, native_dw2)
    labels = ("output", "dx", "dprobs", "routed_dw1", "routed_dw2", "shared_dw1", "shared_dw2")
    stages = {}

    def capture(name, allocate):
        def record(*params, **kwargs):
            buffers = allocate(*params, **kwargs)
            stages[name] = buffers
            return buffers
        return record

    def compare_quant(results, phase, prefix, quant):
        for suffix, want in zip(("row", "row_scale", "col", "col_scale"), quant):
            name = f"{prefix}_{suffix}"
            actual = stages[phase][name]
            # The group path reserves an extra col-scale tile for the local group.
            if suffix == "col_scale":
                actual = actual[:want.shape[0]]
            results.append(compare_bytes(name, actual, want))

    results = {}
    tiling = []
    if args.case in ("both", "separate"):
        separate_labels = (*labels, "routed_output", "routed_dx")
        separate_expected = (*expected, routed_output, routed_dx)
        results["megamoe_with_native_shared"] = [
            compare_bytes(name, actual, want)
            for name, actual, want in zip(separate_labels, mega(False), separate_expected, strict=True)
        ]
    if args.case in ("both", "fused"):
        native_tiler = shared_expert.qbmm_tiling

        def trace_tiling(layout, m, n, k, output_bytes, cubes, *, device):
            key, blocks, data = native_tiler(layout, m, n, k, output_bytes, cubes, device=device)
            tiling.append({"layout": layout, "m_n_k": [m, n, k], "output_bytes": output_bytes,
                           "key": hex(key), "blocks": blocks, "tiling_bytes": len(data),
                           "tiling_hex": data.hex()})
            return key, blocks, data

        with (
            patch.object(shared_expert, "qbmm_tiling", trace_tiling),
            patch.object(execution, "_shared_forward_buffers", capture("forward", execution._shared_forward_buffers)),
            patch.object(execution, "_shared_backward_buffers", capture("backward", execution._shared_backward_buffers)),
        ):
            output = mega(True)
        fused = results["megamoe_with_fused_shared"] = []
        x_mx, gate_up, activation = saved
        compare_quant(fused, "forward", "x", x_mx)
        fused.append(compare_bytes("shared_gate_up", stages["forward"]["gate_up"], gate_up))
        compare_quant(fused, "forward", "activation", activation)
        fused.append(compare_bytes("shared_output", stages["forward"]["output"], native_output))
        compare_quant(fused, "backward", "dy", dy_mx)
        fused.append(compare_bytes("shared_da", stages["backward"]["da"], da))
        compare_quant(fused, "backward", "du", du)
        fused.append(compare_bytes("shared_dx", stages["backward"]["dx"], native_dx))
        fused.extend(compare_bytes(name, actual, want) for name, actual, want in zip(labels, output, expected))
        assert len(tiling) == 6, "Fused replay did not build all six Dense QBMM stages"

    report = {
        "device": device_name, "seed": 5587, "ep": 1, "tokens": tokens,
        "hidden": hidden, "intermediate": width, "experts": experts, "top_k": top_k,
        "bitwise": all(item["bitwise"] for group in results.values() for item in group),
        "cases": results,
        "shared_qbmm_tiling": tiling,
    }
    encoded = json.dumps(report, indent=2)
    print(encoded, flush=True)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["bitwise"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="Visible Ascend950 device index.")
    parser.add_argument("--case", choices=("both", "separate", "fused"), default="both")
    parser.add_argument("--output", type=Path, help="Optional path for the JSON result.")
    return _run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
