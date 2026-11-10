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
"""Compare separately captured migration tensors or process/round timings on CPU."""

import argparse
import json
from pathlib import Path
import statistics

import numpy as np
import torch


def _tensors(reference, candidate):
    left = torch.load(reference, map_location="cpu", weights_only=True)
    right = torch.load(candidate, map_location="cpu", weights_only=True)
    assert left.keys() == right.keys(), "Qualification case sets differ"
    result, count = {}, 0
    for key in left:
        assert len(left[key]) == len(right[key]), f"Result count differs for {key}"
        for index, (expected, actual) in enumerate(zip(left[key], right[key])):
            name = f"{key}/{index}"
            if expected is None:
                result[name] = {"equal": actual is None, "kind": "unused_gradient"}
                continue
            count += 1
            assert actual is not None and expected.shape == actual.shape and expected.dtype == actual.dtype, name
            equal = torch.equal(expected.contiguous().view(torch.uint8), actual.contiguous().view(torch.uint8))
            metrics = {"equal": equal, "shape": list(actual.shape), "dtype": str(actual.dtype)}
            if not equal:
                difference = actual.double() - expected.double()
                metrics.update(max_abs=difference.abs().max().item(),
                               relative_l2=(difference.norm() / expected.double().norm().clamp_min(1e-100)).item(),
                               unequal_elements=(actual != expected).sum().item())
            result[name] = metrics
    return {"passed": all(value["equal"] for value in result.values()), "groups": len(left),
            "tensors": count, "results": result}


def _bootstrap(samples, generator):
    data = np.asarray(samples, dtype=float)
    processes = generator.integers(0, data.shape[0], size=(10000, data.shape[0]))
    rounds = generator.integers(0, data.shape[1], size=(*processes.shape, data.shape[1]))
    return np.median(data[processes[..., None], rounds], axis=-1).mean(axis=-1)


def _memory_measurements(before, after, key):
    fields = ("peak_extra_bytes", "retained_extra_bytes")
    return {field: {"source": [run["timings"][key][field] for run in before],
                    "target": [run["timings"][key][field] for run in after]}
            for field in fields if field in before[0]["timings"][key]}


def _timings(references, candidates, limit):
    left = [json.loads(path.read_text(encoding="utf-8")) for path in references]
    right = [json.loads(path.read_text(encoding="utf-8")) for path in candidates]
    assert len(left) == len(right) and len(left) >= 2, "At least two independent processes per side are required"
    fixture = left[0]["fixture_sha256"]
    keys = left[0]["timings"].keys()
    for run in left + right:
        assert run["fixture_sha256"] == fixture and run["timings"].keys() == keys, "Timing inputs differ"
    generator = np.random.default_rng(71)
    result = {}
    for key in keys:
        before = [run["timings"][key]["device_ms"] for run in left]
        after = [run["timings"][key]["device_ms"] for run in right]
        source = statistics.mean(map(statistics.median, before))
        target = statistics.mean(map(statistics.median, after))
        bootstrap = _bootstrap(after, generator) / _bootstrap(before, generator) - 1
        low, high = np.quantile(bootstrap, [0.025, 0.975]).tolist()
        status = "pass" if high <= limit else "regression" if low > limit else "inconclusive"
        item = {"source_ms": source, "target_ms": target, "change": target / source - 1,
                "confidence95": [low, high], "status": status}
        item.update(_memory_measurements(left, right, key))
        result[key] = item
    return {"passed": all(item["status"] == "pass" for item in result.values()),
            "regression_limit": limit, "processes_per_side": len(left), "results": result}


def main() -> None:
    """Write an auditable comparison and fail when parity has not been established.

    Returns:
        Writes JSON metrics and prints one status per measurement.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("tensors", "timings"), required=True)
    parser.add_argument("--reference", action="append", required=True, type=Path)
    parser.add_argument("--candidate", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-regression", type=float, default=0.02)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.kind == "tensors":
        assert len(args.reference) == len(args.candidate) == 1
        result = _tensors(args.reference[0], args.candidate[0])
    else:
        result = _timings(args.reference, args.candidate, args.max_regression)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "results"}, indent=2))
    for key, value in result["results"].items():
        if args.kind == "timings" or not value["equal"]:
            print(key, value)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
