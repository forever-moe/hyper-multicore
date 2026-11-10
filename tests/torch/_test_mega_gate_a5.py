# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Ascend950 candidate admission for sigmoid routing and projection fusion."""

import json
import os
from pathlib import Path
import tempfile

import torch
import torch_npu  # pylint: disable=unused-import
from torch.utils.checkpoint import checkpoint

from hyper_multicore.modules.mega_gate import (
    MegaGate, sigmoid_route_from_hidden_states, sigmoid_route_from_logits,
)
from hyper_multicore.diagnostics.profiler import mega_kernel_profile


_DEVICE = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))


def _setup() -> None:
    """Use the exact precision policy required by projection fusion."""
    torch.npu.set_device(_DEVICE)
    assert torch.npu.get_device_name(_DEVICE).startswith("Ascend950"), "Ascend950 admission requires an A5 NPU"
    torch.npu.matmul.allow_hf32 = False
    torch.manual_seed(71)


def _golden(hidden, weight, bias, fixed_indices=None):
    logits = torch.nn.functional.linear(  # pylint: disable=not-callable
        hidden.reshape(-1, weight.shape[1]).float(), weight.float()
    )
    scores = logits.sigmoid()
    indices = (scores + bias).topk(8, sorted=True).indices if fixed_indices is None else fixed_indices
    selected = scores.gather(1, indices)
    return logits, selected / (selected.sum(-1, keepdim=True) + 1.0e-20), indices


def _loss(outputs):
    """Include route scaling/casting, fixed indices, auxiliary and z-loss graph edges."""
    logits, weights, indices = outputs
    fixed = torch.arange(indices.numel(), device=indices.device).reshape_as(indices) % logits.shape[1]
    coefficient = (fixed.float() + 1) / logits.shape[1]
    routed = (weights * 2.5).to(torch.bfloat16).float()
    return ((routed * coefficient).sum() / logits.shape[0] +
            2.0e-4 * logits.softmax(-1).square().mean() +
            1.0e-6 * logits.logsumexp(-1).square().mean())


def _parity(tokens, hidden, experts, dtype, mode):
    data = torch.randn(1, tokens, hidden, device=_DEVICE, dtype=dtype)
    parameter = torch.randn(experts, hidden, device=_DEVICE, dtype=dtype) * 0.02
    bias = torch.randn(experts, device=_DEVICE) * 0.03
    results = []
    for operation in (_golden, sigmoid_route_from_hidden_states):
        x = data.detach().requires_grad_()
        w = parameter.detach().requires_grad_()
        def _forward(xx, ww):
            return operation(xx, ww, bias) if operation is _golden else operation(xx, ww, expert_bias=bias)
        if mode == "none":
            output = _forward(x, w)
        else:
            output = checkpoint(_forward, x, w, use_reentrant=(mode == "reentrant"))
        gradients = torch.autograd.grad(_loss(output), (x, w)) if mode != "reentrant" else None
        if gradients is None:
            _loss(output).backward()
            gradients = x.grad, w.grad
        results.append((*output, *gradients))
    actual_result, expected_result = results[1], results[0]
    selected = actual_result[0].sigmoid().gather(1, actual_result[2])
    torch.testing.assert_close(actual_result[1], selected / (selected.sum(-1, keepdim=True) + 1.0e-20),
                               rtol=1.0e-6, atol=2.0e-7)
    reference_logits = expected_result[0]
    actual_scores = actual_result[0].sigmoid()
    reference_scores = reference_logits.sigmoid()
    chosen_values = (reference_scores + bias).gather(1, actual_result[2])
    top_values = (reference_scores + bias).topk(8, sorted=True).values
    # Only rankings within the measured projection/activation roundoff may change.
    bound = 2 * (actual_scores - reference_scores).abs().amax(-1, keepdim=True) + 2e-7
    assert torch.all((chosen_values - top_values).abs() <= bound)
    changed_rows = (actual_result[2] != expected_result[2]).any(-1).sum().item()
    torch.testing.assert_close(actual_result[0], reference_logits, rtol=2e-3, atol=2e-5)
    # Different valid TopK sets are different differentiable branches. Compare the
    # derivative on the branch actually selected, retaining the slot-based loss.
    x = data.detach().requires_grad_()
    w = parameter.detach().requires_grad_()
    fixed_reference = _golden(x, w, bias, fixed_indices=actual_result[2])
    fixed_gradients = torch.autograd.grad(_loss(fixed_reference), (x, w))
    expected_result = (*fixed_reference, *fixed_gradients)
    for index, (actual, expected) in enumerate(zip(actual_result, expected_result)):
        assert torch.isfinite(actual).all(), f"nonfinite output/gradient {index}"
        if index == 2:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            tolerance = 2.0e-3 if actual.dtype == torch.bfloat16 else 2.0e-5
            torch.testing.assert_close(actual, expected, rtol=2.0e-3, atol=tolerance)
    if changed_rows:
        print(json.dumps({"precision_boundary_rows": changed_rows, "T": tokens, "E": experts}), flush=True)
    print(json.dumps({"T": tokens, "H": hidden, "E": experts, "dtype": str(dtype), "checkpoint": mode}), flush=True)


def test_mega_gate_sigmoid_parity() -> None:
    """Route-only precision for normalized/direct schedules and expert-width tails."""
    _setup()
    for tokens, experts, top_k, normalize in (
        (1, 8, 1, True), (33, 12, 8, True), (65, 64, 16, True),
        (257, 384, 128, True), (33, 384, 384, True), (65, 12, 3, False),
        (4096, 12, 8, True), (4096, 384, 8, True),
    ):
        logits = torch.randn(tokens, experts, device=_DEVICE, requires_grad=True)
        bias = torch.randn(experts, device=_DEVICE) * 0.03
        weights, indices = sigmoid_route_from_logits(logits, top_k=top_k, expert_bias=bias, normalize=normalize)
        scores = logits.sigmoid()
        expected_indices = (scores + bias).topk(top_k, sorted=True).indices
        selected = scores.gather(1, expected_indices)
        expected = selected / (selected.sum(-1, keepdim=True) + 1.0e-20) if top_k > 1 and normalize else selected
        torch.testing.assert_close((scores + bias).gather(1, indices),
                                   (scores + bias).topk(top_k, sorted=True).values, rtol=0, atol=2e-7)
        assert indices.sort(-1).values.diff(dim=-1).ne(0).all()
        selected = scores.gather(1, indices)
        expected = selected / (selected.sum(-1, keepdim=True) + 1.0e-20) if top_k > 1 and normalize else selected
        torch.testing.assert_close(weights, expected, rtol=1.0e-6, atol=2.0e-7)
        upstream = torch.randn_like(weights)
        actual_grad = torch.autograd.grad(weights, logits, upstream, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, logits, upstream)[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2.0e-4, atol=2.0e-6)


def test_mega_gate_sigmoid_direct_workspace() -> None:
    """Unnormalized K384 must not reserve the normalized backward's eight K-wide temporaries."""
    _setup()
    logits = torch.randn(4096, 384, device=_DEVICE, requires_grad=True)
    weights, _ = sigmoid_route_from_logits(logits, top_k=384, normalize=False)
    grad_weights = torch.randn_like(weights)
    # Warm up post kernels and allocator caches before recording the per-call peak.
    torch.autograd.grad(weights, logits, grad_weights, retain_graph=True)
    torch.npu.synchronize(_DEVICE)
    baseline = torch.npu.memory_allocated(_DEVICE)
    torch.npu.reset_peak_memory_stats(_DEVICE)
    torch.autograd.grad(weights, logits, grad_weights)
    torch.npu.synchronize(_DEVICE)
    additional = torch.npu.max_memory_allocated(_DEVICE) - baseline
    assert additional < 48 * 1024 * 1024, f"unexpected normalized workspace in direct backward: {additional}"


def test_mega_gate_projection_parity() -> None:
    """Current batch1/T4096/H5120 and larger T, partial tiles, BF16 and checkpoint."""
    _setup()
    shapes = [(1, 128, 8), (31, 128, 12), (33, 128, 384), (257, 128, 64),
              (4096, 5120, 12), (4096, 5120, 384), (12288, 5120, 12)]
    for tokens, hidden, experts in shapes:
        for dtype in (torch.float32, torch.bfloat16):
            for mode in ("none", "reentrant", "nonreentrant"):
                _parity(tokens, hidden, experts, dtype, mode)


def test_mega_gate_projection_streams_and_training() -> None:
    """Fresh call state, retained graphs, gradient accumulation and async optimizer steps."""
    _setup()
    gate = MegaGate(hidden_size=128, num_experts=12, top_k=8, scoring_func="sigmoid",
                    routed_scaling_factor=2.5).to(_DEVICE)
    data = torch.randn(1, 33, 128, device=_DEVICE, requires_grad=True)
    initial = {name: tensor.detach().clone() for name, tensor in gate.state_dict().items()}
    final = []
    for fused in (False, True):
        gate.load_state_dict(initial)
        def _forward(hidden):
            outputs = (sigmoid_route_from_hidden_states(hidden, gate.weight, expert_bias=gate.bias)
                       if fused else _golden(hidden, gate.weight, gate.bias))
            return outputs[0], outputs[1] * gate.routed_scaling_factor, outputs[2]

        optimizer = torch.optim.SGD(gate.parameters(), lr=1.0e-4, momentum=0.9)
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(2):
                output = checkpoint(_forward, data, use_reentrant=False)
                loss = (output[1] * (output[2].float() + 1)).mean() + output[0].square().mean() * 0.01
                loss.backward()
            optimizer.step()
        final.append(gate.weight.detach().clone())
    torch.testing.assert_close(final[1], final[0], rtol=2.0e-4, atol=2.0e-6)
    assert gate.bias.grad is None
    origin = torch.npu.current_stream(_DEVICE)
    streams = [torch.npu.Stream(device=_DEVICE), torch.npu.Stream(device=_DEVICE)]
    with torch.no_grad():
        reference = _forward(data)
        results = []
        for stream in streams:
            stream.wait_stream(origin)
            with torch.npu.stream(stream):
                results.append(_forward(data))
        for stream in streams:
            origin.wait_stream(stream)
        for result in results:
            for actual, expected in zip(result, reference):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mega_gate_projection_profile() -> None:
    """Export the standard cycle trace with complete MatMul/wait/route/backward records."""
    _setup()
    data = torch.randn(1, 4096, 5120, device=_DEVICE, dtype=torch.bfloat16, requires_grad=True)
    weight = (torch.randn(12, 5120, device=_DEVICE, dtype=torch.bfloat16) * 0.02).requires_grad_()
    with mega_kernel_profile(detailed_task_names=True) as profiler:
        output = checkpoint(sigmoid_route_from_hidden_states, data, weight, use_reentrant=False)
        _loss(output).backward()
        profiler.step()
    destination = os.environ.get("MEGA_GATE_PROFILE_DIR")
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(destination or temporary) / "mega_gate_T4096_H5120_E12_K8.megakernel.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        trace = profiler.export_chrome_trace(path)
    metadata = trace["megaKernelCycleTrace"]
    events = [event for event in trace["traceEvents"] if event.get("ph") == "X"]
    assert metadata["droppedRecordCount"] == 0 and metadata["warnings"] == [], metadata
    assert metadata["invocationCount"] >= 3, metadata
    assert any("MatMul" in event["name"] for event in events)
    assert any("WaitDependency" in event["name"] for event in events)
    assert any("Sigmoid" in event["name"] for event in events)
    print(json.dumps({"path": str(path), "metadata": metadata}), flush=True)
