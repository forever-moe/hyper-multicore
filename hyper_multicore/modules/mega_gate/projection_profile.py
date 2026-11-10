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
"""Bridge the repeated tile schedule into the pinned MegaKernel profiler ABI."""

from __future__ import annotations

import math
import torch
import torch.distributed as dist
from hyper_multicore.diagnostics.profiler.cycle_trace import (
    _PreparedProfileRuntime, _ProfileLayout, _ProfileMetadata, _TaskExecutionMode,
)
from hyper_multicore.modules.mega_gate.diagnostics import MEGA_GATE_SIGMOID_ROUTE_STAGE_NAMES
from hyper_multicore.scheduler.runtime import allocate_runtime_config, serialize_runtime_config


class _TileProfileRuntime(_PreparedProfileRuntime):
    """Adapt the repeated tile schedule to the existing profiler ABI and parser.

    The ordinary graph layout counts each descriptor once. This kernel repeats
    its descriptors for every token tile, so its capacity must include all tiles.
    Only the validated 32 AIC / 64 AIV scheduler ABI is supported.
    """

    def __init__(self, ordinary: torch.Tensor, tokens: int) -> None:
        """Size the cycle recorder for all repeated tiles on this device."""
        self._device = ordinary.device
        self._device_id = ordinary.device.index
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._normal_tensor = ordinary
        vectors = torch.npu.get_device_limit(self._device_id)["vector_core_num"]
        if vectors != 64:
            raise ValueError("The fused profiler requires 64 AIV workers")
        cubes = vectors // 2
        tiles = math.ceil(tokens / 32)
        max_tiles = math.ceil(tiles / cubes)
        self._layout = _ProfileLayout(max_tiles, max_tiles * 11,
                                     math.ceil(max_tiles / 16) * 16,
                                     math.ceil(max_tiles * 11 / 16) * 16, cubes)
        config_type = type(allocate_runtime_config(11, num_workers=vectors))
        config = config_type.from_buffer_copy(ordinary.cpu().numpy().tobytes())
        config.aic_profile_record_capacity = self._layout.aic_record_capacity
        config.aiv_profile_record_capacity = self._layout.aiv_record_capacity
        config.diagnostic_mode = 1
        for task_id in range(11):
            config.all_tasks[task_id].profile_desc_id = 0x30000 + task_id
        self._host_config = config
        self._profile_tensor = torch.frombuffer(
            bytearray(serialize_runtime_config(config)), dtype=torch.uint8,
        ).clone().to(self._device)
        names = ("MatMul", *MEGA_GATE_SIGMOID_ROUTE_STAGE_NAMES)
        stages = {0x30000 + index: name for index, name in enumerate(names)}
        stages[0x10000] = "WaitDependency"
        self._metadata = _ProfileMetadata(
            "HyperMegaGateProjection", "TokenTile", stages, dict(enumerate(names)),
            _TaskExecutionMode.EVENT_DRIVEN,
        )
        # Publish the cold profile config before another stream can reuse it.
        torch.npu.synchronize(self._device)

    @property
    def buffer_size(self) -> int:
        """Use the exact standard cycle ABI size, including small and partial tiles."""
        return self._layout.buffer_size
