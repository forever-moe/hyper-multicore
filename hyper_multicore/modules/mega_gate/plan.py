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
"""Build shared RuntimeConfigs for HyperMegaGate Route and RouteGrad."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
import torch.distributed as dist

from hyper_multicore.diagnostics import MegaKernelDiagnostics, prepare_mega_kernel_diagnostics
from hyper_multicore.scheduler.config import (
    EVENT_INVALID_ID,
    INVALID_PROFILE_OWNER_ID,
    TaskAiCoreType,
    TaskDescC,
    TaskType,
    RuntimeConfigC,
    NUM_WORKERS_VECTOR,
)
from hyper_multicore.scheduler.runtime import allocate_runtime_config, serialize_runtime_config

from .diagnostics import (
    MEGA_GATE_ROUTE_GRAD_K1_STAGE_NAMES,
    MEGA_GATE_ROUTE_GRAD_STAGE_NAMES,
    MEGA_GATE_ROUTE_STAGE_NAMES,
    MEGA_GATE_SIGMOID_ROUTE_STAGE_NAMES,
    bind_pipeline_graph,
)


MEGA_GATE_AIV_WORKER_CAPACITY = NUM_WORKERS_VECTOR
MEGA_GATE_ROUTE_TASK_TYPES = (
    TaskType.TASK_GATE_SOFTPLUS,
    TaskType.TASK_GATE_SQRT,
    TaskType.TASK_GATE_ADD_BIAS,
    TaskType.TASK_GATE_TOPK,
    TaskType.TASK_GATE_GATHER,
    TaskType.TASK_GATE_REDUCE_SUM,
    TaskType.TASK_GATE_ADD_EPSILON,
    TaskType.TASK_GATE_DIV,
    TaskType.TASK_GATE_MUL_SCALE,
    TaskType.TASK_GATE_CAST_INDEX,
)
MEGA_GATE_ROUTE_GRAD_TASK_TYPES = (
    TaskType.TASK_GATE_GRAD_MULS_SCALE,
    TaskType.TASK_GATE_GRAD_BROADCAST_DENOMINATOR,
    TaskType.TASK_GATE_GRAD_NEG,
    TaskType.TASK_GATE_GRAD_DIV_SELECTED,
    TaskType.TASK_GATE_GRAD_DIV_SELECTED_RATIO,
    TaskType.TASK_GATE_GRAD_MUL_CROSS,
    TaskType.TASK_GATE_GRAD_DIV_DIRECT,
    TaskType.TASK_GATE_GRAD_REDUCE_SUM,
    TaskType.TASK_GATE_GRAD_BROADCAST_ROW_SUM,
    TaskType.TASK_GATE_GRAD_ADD_SELECTED,
    TaskType.TASK_GATE_GRAD_ZEROS,
)
MEGA_GATE_ROUTE_GRAD_K1_TASK_TYPES = (
    TaskType.TASK_GATE_GRAD_MULS_SCALE,
    TaskType.TASK_GATE_GRAD_ZEROS,
)


@dataclass(frozen=True)
class MegaGatePlan:
    """Capacity-bounded RuntimeConfig shared by all input shapes on one NPU."""

    route_runtime: MegaKernelDiagnostics
    route_grad_runtime: MegaKernelDiagnostics
    route_grad_k1_runtime: MegaKernelDiagnostics
    vision_mask_placeholder: torch.Tensor
    use_sigmoid: bool = False


def _build_pipeline_runtime_config(
    task_types: tuple[TaskType, ...],
    stage_names: tuple[str, ...],
    *,
    kernel_name: str,
    worker_capacity: int = MEGA_GATE_AIV_WORKER_CAPACITY,
) -> RuntimeConfigC:
    """Create descriptors broadcast to every active token-row worker."""
    if len(task_types) != len(stage_names):
        raise RuntimeError("MegaGate task types and profile stages must have equal length")
    if not isinstance(kernel_name, str) or not kernel_name:
        raise ValueError("MegaGate kernel_name must be a non-empty string")
    if not isinstance(worker_capacity, int) or isinstance(worker_capacity, bool) or worker_capacity <= 0:
        raise ValueError("MegaGate worker_capacity must be a positive integer")
    runtime_config = allocate_runtime_config(len(task_types), num_workers=worker_capacity)
    runtime_config.task_num = len(task_types)
    runtime_config.num_workers = worker_capacity
    runtime_config.task_index_num[1] = len(task_types)
    for task_id, task_type in enumerate(task_types):
        task = TaskDescC()
        task.task_type = task_type
        task.task_aicore_type = TaskAiCoreType.TASK_AICORE_VECTOR
        task.trigger_event = EVENT_INVALID_ID
        task.dependent_event = EVENT_INVALID_ID
        task.task_index = 0
        task.task_split_num = worker_capacity
        task.tiling_data_offset = task_id
        task.profile_owner_id = INVALID_PROFILE_OWNER_ID
        runtime_config.all_tasks[task_id] = task
        runtime_config.vector_task_indices[task_id] = task_id
    return runtime_config


def build_mega_gate_plan(device: torch.device, *, use_sigmoid: bool = False) -> MegaGatePlan:
    """Build an eventless Route runtime for one NPU device.

    Args:
        device: NPU device receiving the serialized RuntimeConfig.
        use_sigmoid: Use sigmoid stage identities and tensor ports in diagnostic captures.

    Returns:
        Immutable plan with normal and profiled broadcast stage descriptors.
    """
    if device.type != "npu":
        raise ValueError(f"MegaGate plan requires an NPU device, got {device}")
    if device.index is None:
        device = torch.device("npu", torch.npu.current_device())
    rank = dist.get_rank() if dist.is_initialized() else 0
    available_workers = torch.npu.get_device_limit(device.index)["vector_core_num"]
    if not isinstance(available_workers, int) or isinstance(available_workers, bool) or available_workers <= 0:
        raise RuntimeError("MegaGate requires a positive vector_core_num device limit")

    def prepare(runtime_config: RuntimeConfigC, *, backward: bool = False) -> MegaKernelDiagnostics:
        """Prepare only ordinary storage; broadcast graph ports remain lazy.

        Args:
            runtime_config: Complete Host configuration describing native task queues.
            backward: Whether to prepare the RouteGrad pipeline instead of Route.
        """
        return prepare_mega_kernel_diagnostics(
            None, runtime_config, kernel_name="HyperMegaGateRouteGrad" if backward else "HyperMegaGateRoute",
            parameter_count=11, device=device, rank=rank, owner_label="TokenRowShard",
            execution_mode="aiv_pipeline", vector_worker_stride=1,
            graph_adapter=partial(bind_pipeline_graph, backward=backward, use_sigmoid=use_sigmoid),
        )

    result = MegaGatePlan(
        route_runtime=prepare(_build_pipeline_runtime_config(
            MEGA_GATE_ROUTE_TASK_TYPES,
            MEGA_GATE_SIGMOID_ROUTE_STAGE_NAMES if use_sigmoid else MEGA_GATE_ROUTE_STAGE_NAMES,
            kernel_name="HyperMegaGateRoute", worker_capacity=available_workers,
        )),
        route_grad_runtime=prepare(_build_pipeline_runtime_config(
            MEGA_GATE_ROUTE_GRAD_TASK_TYPES,
            MEGA_GATE_ROUTE_GRAD_STAGE_NAMES,
            kernel_name="HyperMegaGateRouteGrad", worker_capacity=available_workers,
        ), backward=True),
        route_grad_k1_runtime=prepare(_build_pipeline_runtime_config(
            MEGA_GATE_ROUTE_GRAD_K1_TASK_TYPES,
            MEGA_GATE_ROUTE_GRAD_K1_STAGE_NAMES,
            kernel_name="HyperMegaGateRouteGrad", worker_capacity=available_workers,
        ), backward=True),
        vision_mask_placeholder=torch.zeros((1,), device=device, dtype=torch.bool),
        use_sigmoid=use_sigmoid,
    )
    torch.npu.current_stream(device).synchronize()
    return result


def _build_projection_runtime(device: torch.device) -> torch.Tensor:
    """Serialize the pinned 32 AIC / 64 AIV projection and route tile schedule."""
    config = allocate_runtime_config(11, num_workers=64)
    config.task_num = 11
    config.num_workers = 64
    config.task_index_num[0] = 1
    config.task_index_num[1] = 10
    config.cube_task_indices[0] = 0
    for index, kind in enumerate((TaskType.TASK_MATMUL, *MEGA_GATE_ROUTE_TASK_TYPES)):
        task = TaskDescC()
        task.task_type = kind
        task.task_aicore_type = (TaskAiCoreType.TASK_AICORE_CUBE if index == 0
                                else TaskAiCoreType.TASK_AICORE_VECTOR)
        task.trigger_event = EVENT_INVALID_ID
        task.dependent_event = EVENT_INVALID_ID
        task.task_split_num = 32 if index == 0 else 64
        task.tiling_data_offset = 0 if index == 0 else index - 1
        task.profile_owner_id = INVALID_PROFILE_OWNER_ID
        config.all_tasks[index] = task
        if index:
            config.vector_task_indices[index - 1] = index
    return torch.frombuffer(bytearray(serialize_runtime_config(config)), dtype=torch.uint8).clone().to(device)
