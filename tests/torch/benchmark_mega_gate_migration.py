# Copyright 2026 Huawei Technologies Co., Ltd.
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
"""Run isolated migration qualification or timings against shared CPU fixtures.

Run source and target in separate processes with their matching vendor activated.
The optional reference root is used only by this qualification tool; the component
and ordinary system tests do not depend on the reference training repository.
"""

import argparse
import hashlib
import importlib
import io
import json
from pathlib import Path
import statistics
import sys
import time
import types

import torch
import torch_npu  # pylint: disable=unused-import
from torch.utils.checkpoint import checkpoint


def _fixture(path):
    generator = torch.Generator().manual_seed(20261109)
    projection = []
    for tokens, hidden, experts, dtype, biased in (
        (1, 64, 8, torch.float32, False), (31, 128, 12, torch.bfloat16, True),
        (32, 128, 16, torch.float32, False), (33, 128, 33, torch.bfloat16, True),
        (129, 5120, 384, torch.bfloat16, True), (4096, 5120, 384, torch.bfloat16, True),
        (6144, 5120, 384, torch.bfloat16, True),
    ):
        projection.append({
            "x": torch.randn(tokens, hidden, generator=generator).to(dtype),
            "w": (torch.randn(experts, hidden, generator=generator) * 0.001383).to(dtype),
            "bias": torch.randn(experts, generator=generator) * 0.03 if biased else None,
            "upstream": torch.randn(tokens, 8, generator=generator),
        })
    route = []
    for tokens, experts, topk, normalize, dtype in (
        (1, 8, 1, True, torch.float32), (33, 12, 8, True, torch.bfloat16),
        (65, 64, 16, True, torch.float32), (257, 384, 128, True, torch.float32),
        (33, 384, 384, True, torch.float32), (65, 12, 3, False, torch.float16),
        (33, 384, 384, False, torch.float32), (4096, 384, 8, True, torch.float32),
    ):
        route.append({"logits": torch.randn(tokens, experts, generator=generator).to(dtype),
                      "bias": torch.randn(experts, generator=generator) * 0.03,
                      "upstream": torch.randn(tokens, topk, generator=generator),
                      "topk": topk, "normalize": normalize})
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"projection": projection, "route": route}, path)


def _implementation(reference):
    if reference is None:
        return importlib.import_module("hyper_multicore.modules.mega_gate")
    # Import only the embedded component; the PanGu package initializer starts
    # unrelated training-framework setup. No computation or native op is mocked.
    namespace = types.ModuleType("PanGu")
    namespace.__path__ = [str(reference.resolve() / "PanGu")]
    sys.modules["PanGu"] = namespace
    return importlib.import_module("PanGu.multicore.modules.mega_gate")


def _device_case(case):
    return {key: value.to("npu:0") if isinstance(value, torch.Tensor) else value for key, value in case.items()}


def _snapshot(values):
    result = []
    for value in values:
        if value is not None:
            assert torch.isfinite(value).all(), "Nonfinite qualification result"
        result.append(None if value is None else value.detach().cpu())
    return result


def _loss(outputs, upstream, branch="both"):
    logits, weights = outputs[:2]
    direct = logits.square().mean() * 1e-3 if branch != "weights" else 0
    routed = (weights * upstream).mean() if branch != "logits" else 0
    return direct + routed


def _projection_case(gate, case, *, branch="both", frozen=None, recompute=False, autocast=False):
    x = case["x"].detach().requires_grad_(frozen != "x")
    w = case["w"].detach().requires_grad_(frozen != "w")
    bias = None if case["bias"] is None else case["bias"].detach().requires_grad_()

    def _forward(hidden, weight):
        return gate.sigmoid_route_from_hidden_states(hidden, weight, expert_bias=bias)

    before = torch.npu.matmul.allow_hf32
    with torch.autocast("npu", enabled=autocast, dtype=torch.bfloat16):
        outputs = checkpoint(_forward, x, w, use_reentrant=False) if recompute else _forward(x, w)
        loss = _loss(outputs, case["upstream"], branch)
    inputs = [value for value in (x, w, bias) if value is not None and value.requires_grad]
    gradients = torch.autograd.grad(loss, inputs, allow_unused=True)
    assert torch.npu.matmul.allow_hf32 == before, "Projection changed the global HF32 setting"
    assert outputs[0].dtype == outputs[1].dtype == torch.float32
    assert outputs[2].dtype == torch.int64 and not outputs[2].requires_grad
    if bias is not None:
        assert gradients[-1] is None, "Selection bias must not receive a gradient"
    return (*outputs, *gradients)


def _train(gate, case, *, resume=False):
    parameter = torch.nn.Parameter(case["w"].float().clone())
    optimizer = torch.optim.SGD([parameter], lr=0.05, momentum=0.9)
    bias = case["bias"].clone()
    records = []
    for step in range(12):
        optimizer.zero_grad(set_to_none=True)

        def _forward(hidden, weight):
            return gate.sigmoid_route_from_hidden_states(hidden, weight, expert_bias=bias)

        outputs = checkpoint(_forward, case["x"], parameter, use_reentrant=False)
        loss = _loss(outputs, case["upstream"])
        loss.backward()
        records.append(torch.stack((loss.detach(), parameter.grad.norm())))
        optimizer.step()
        with torch.no_grad():
            counts = torch.bincount(outputs[2].flatten(), minlength=bias.numel())
            bias.add_((counts.float().mean() - counts.float()).sign() * 1e-4)
        if resume and step == 5:
            with io.BytesIO() as buffer:
                torch.save({"weight": parameter.detach(), "bias": bias,  # pylint: disable=not-callable
                            "optimizer": optimizer.state_dict()}, buffer)
                buffer.seek(0)
                saved = torch.load(buffer, weights_only=True)
            parameter = torch.nn.Parameter(saved["weight"])
            bias = saved["bias"]
            optimizer = torch.optim.SGD([parameter], lr=0.05, momentum=0.9)
            optimizer.load_state_dict(saved["optimizer"])
    momentum = optimizer.state[parameter]["momentum_buffer"]
    return _snapshot((parameter, bias, momentum, torch.stack(records)))


def _qualify(gate, fixture):
    outputs = {}
    for index, cpu_case in enumerate(fixture["route"]):
        case = _device_case(cpu_case)
        logits = case["logits"].requires_grad_()
        bias = case["bias"].requires_grad_()
        weights, indices = gate.sigmoid_route_from_logits(
            logits, top_k=case["topk"], expert_bias=bias, normalize=case["normalize"],
        )
        grad, grad_bias = torch.autograd.grad((weights * case["upstream"]).mean(), (logits, bias),
                                              allow_unused=True)
        assert grad_bias is None
        outputs[f"route/{index}"] = _snapshot((weights, indices, grad))
    for index, cpu_case in enumerate(fixture["projection"]):
        outputs[f"projection/{index}"] = _snapshot(_projection_case(gate, _device_case(cpu_case)))
        print(f"Qualified projection case {index}", flush=True)
    case = _device_case(fixture["projection"][3])
    for branch in ("logits", "weights"):
        outputs[f"branch/{branch}"] = _snapshot(_projection_case(gate, case, branch=branch))
    for frozen in ("x", "w"):
        outputs[f"frozen/{frozen}"] = _snapshot(_projection_case(gate, case, frozen=frozen))
    outputs["checkpoint"] = _snapshot(_projection_case(gate, case, recompute=True))
    for flag in (False, True):
        torch.npu.matmul.allow_hf32 = flag
        outputs[f"autocast_hf32/{flag}"] = _snapshot(_projection_case(gate, case, autocast=True))
    torch.npu.matmul.allow_hf32 = False
    streams = [torch.npu.Stream() for _ in range(2)]
    torch.npu.synchronize()
    serial = _snapshot(_projection_case(gate, case))
    for index, stream in enumerate(streams):
        with torch.npu.stream(stream):
            result = _projection_case(gate, case)
        stream.synchronize()
        actual = _snapshot(result)
        assert all(a is None if b is None else a is not None and torch.equal(a, b)
                   for a, b in zip(actual, serial)), "Stream execution differs from serial execution"
        outputs[f"stream/{index}"] = actual
    continuous, resumed = _train(gate, case), _train(gate, case, resume=True)
    assert all(torch.equal(a, b) for a, b in zip(continuous, resumed)), "Optimizer restore differs"
    outputs["training"] = continuous
    return outputs


def _benchmark(gate, fixture, rounds, iterations):
    package = gate.__name__.rsplit(".", 1)[0]
    projection = importlib.import_module(package + ".mega_gate.projection")
    results = {}
    for cpu_case in fixture["projection"][-2:]:
        case = _device_case(cpu_case)
        x, w, bias = case["x"], case["w"], case["bias"]
        fp32_x, fp32_w = x.float(), w.float()
        runtime = projection._projection_runtime(x.device)  # pylint: disable=protected-access
        native = projection._native_ops()  # pylint: disable=protected-access
        source_x, source_w = x.detach().requires_grad_(), w.detach().requires_grad_()

        def _kernel():
            with torch.no_grad():
                return native._mega_gate_projection(fp32_x, fp32_w, bias, runtime)  # pylint: disable=protected-access

        def _forward():
            with torch.no_grad():
                return gate.sigmoid_route_from_hidden_states(x, w, expert_bias=bias)

        def _train(recompute=False):
            def _call(hidden, weight):
                return gate.sigmoid_route_from_hidden_states(hidden, weight, expert_bias=bias)
            outputs = (checkpoint(_call, source_x, source_w, use_reentrant=False) if recompute
                       else _call(source_x, source_w))
            return torch.autograd.grad(_loss(outputs, case["upstream"]), (source_x, source_w))

        functions = {"kernel": _kernel, "forward": _forward, "forward_backward": _train,
                     "checkpoint": lambda: _train(True)}
        for name, function in functions.items():
            for _ in range(20):
                function()
            torch.npu.synchronize()
            samples, walls = [], []
            base = torch.npu.memory_allocated()
            torch.npu.reset_peak_memory_stats()
            for _ in range(rounds):
                start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                wall = time.perf_counter()
                start.record()
                for _ in range(iterations):
                    function()
                end.record()
                end.synchronize()
                walls.append((time.perf_counter() - wall) * 1000 / iterations)
                samples.append(start.elapsed_time(end) / iterations)
            key = f"T{x.shape[0]}/{name}"
            results[key] = {"device_ms": samples, "wall_ms": walls,
                            "median_device_ms": statistics.median(samples),
                            "peak_extra_bytes": torch.npu.max_memory_allocated() - base,
                            "retained_extra_bytes": torch.npu.memory_allocated() - base}
            print(key, results[key]["median_device_ms"], flush=True)
    return results


def main() -> None:
    """Execute one isolated side of the migration comparison.

    Returns:
        Writes shared input fixtures, qualification tensors or benchmark JSON.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--mode", choices=("generate", "qualify", "benchmark"), required=True)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.mode == "generate":
        if args.fixture.exists():
            raise FileExistsError(args.fixture)
        _fixture(args.fixture)
        return
    if args.output is None:
        parser.error("--output is required for qualification and benchmark runs")
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    gate = _implementation(args.source_root)
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    metadata = {"implementation": gate.__file__, "torch": str(torch.__version__),
                "torch_npu": str(torch_npu.__version__), "device": torch.npu.get_device_name(0),
                "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
                "rounds": args.rounds, "iterations": args.iterations}
    metadata["harness_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "qualify":
        torch.save(_qualify(gate, fixture), args.output)
    else:
        metadata["timings"] = _benchmark(gate, fixture, args.rounds, args.iterations)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
