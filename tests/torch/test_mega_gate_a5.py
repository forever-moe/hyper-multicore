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
"""Launch single-card precision checks for the MegaGate module entry."""

from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.multicore._case_launcher import run_multicore_case
from tests.common.multicore._test_env import (
    multicore_adapter_is_available,
    prepare_multicore_test_environment,
    without_inherited_rank_environment,
)

_WORKER = str(Path(__file__).resolve().parent / "_test_mega_gate_a5.py")


def _run_worker(case: str, *, timeout: float = 420) -> None:
    """Run one MegaGate module worker on one NPU."""
    prepare_multicore_test_environment()
    if not multicore_adapter_is_available():
        raise RuntimeError("MegaGate ST requires a wheel or PYTHONPATH payload built with bash build.sh")
    with without_inherited_rank_environment():
        run_multicore_case(_WORKER, case, num_proc=1, timeout=timeout)



@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard",
          essential_mark="essential")
def test_mega_gate_sigmoid_parity() -> None:
    """Compare normalized and unnormalized sigmoid routes."""
    _run_worker("test_mega_gate_sigmoid_parity", timeout=840)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard",
          essential_mark="essential")
def test_mega_gate_sigmoid_direct_workspace() -> None:
    """Validate direct-route backward workspace bounds."""
    _run_worker("test_mega_gate_sigmoid_direct_workspace", timeout=840)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard",
          essential_mark="essential")
def test_mega_gate_projection_parity() -> None:
    """Compare fused projection outputs and gradients."""
    _run_worker("test_mega_gate_projection_parity", timeout=840)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard",
          essential_mark="essential")
def test_mega_gate_projection_streams_and_training() -> None:
    """Exercise streams, checkpointing and optimizer updates."""
    _run_worker("test_mega_gate_projection_streams_and_training", timeout=840)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard",
          essential_mark="essential")
def test_mega_gate_projection_profile() -> None:
    """Validate the repeated tile profile and backward captures."""
    _run_worker("test_mega_gate_projection_profile", timeout=840)
