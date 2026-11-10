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
"""Reject hidden slot/bit differences and statistically unproven performance parity."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch


_PATH = Path(__file__).resolve().parents[4] / "tests/torch/compare_mega_gate_migration.py"
_SPEC = importlib.util.spec_from_file_location("gate_migration_comparison", _PATH)
_COMPARE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_COMPARE)


def test_tensor_comparison_keeps_bits_and_slots(tmp_path):
    """Equal numeric zeros cannot hide signed-zero or expert-slot changes."""
    source, target = tmp_path / "source.pt", tmp_path / "target.pt"
    values = {"case": [torch.tensor([0.0, 1.0]), torch.tensor([1, 2]), None]}
    torch.save(values, source)
    torch.save(values, target)
    assert _COMPARE._tensors(source, target)["passed"]
    values["case"][0][0] = -0.0
    values["case"][1] = torch.tensor([2, 1])
    torch.save(values, target)
    result = _COMPARE._tensors(source, target)
    assert not result["passed"]
    assert not result["results"]["case/0"]["equal"]
    assert not result["results"]["case/1"]["equal"]


@pytest.mark.parametrize("latencies,status", [([1.005, 1.005], "pass"),
                                              ([1.05, 1.05], "regression"),
                                              ([1.0, 1.04], "inconclusive")])
def test_timing_acceptance_requires_evidence(tmp_path, latencies, status):
    """The 2% limit must reject regression and uncertainty separately."""
    sources, targets = [], []
    for index, latency in enumerate(latencies):
        for side, duration, paths in (("source", 1.0, sources), ("target", latency, targets)):
            path = tmp_path / f"{side}{index}.json"
            path.write_text(json.dumps({"fixture_sha256": "same-input", "timings": {
                "kernel": {"device_ms": [duration] * 9}}}), encoding="utf-8")
            paths.append(path)
    result = _COMPARE._timings(sources, targets, 0.02)
    assert result["results"]["kernel"]["status"] == status
    assert result["passed"] == (status == "pass")
