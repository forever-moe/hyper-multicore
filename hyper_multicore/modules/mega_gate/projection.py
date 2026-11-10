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
"""Opt-in FP32 projection plus normalized TopK8 sigmoid routing on Ascend950."""

from __future__ import annotations

from functools import lru_cache
from threading import Lock
from weakref import WeakKeyDictionary

from typing import Any

import torch
from torch.autograd.function import once_differentiable

from hyper_multicore.diagnostics.common.session import _get_active_session
from .function import _native_ops
from .plan import _build_projection_runtime
from .projection_profile import _TileProfileRuntime
from .sigmoid import _sigmoid_plan_for_device, _sigmoid_route_backward, _zero_bias_for_device

_PROFILE_RUNTIMES = WeakKeyDictionary()
_PROFILE_LOCK = Lock()


@lru_cache(maxsize=None)
def _validate_projection_device(device: torch.device) -> None:
    if (not torch.npu.get_device_name(device).startswith('Ascend950') or
            torch.npu.get_device_limit(device.index)['vector_core_num'] != 64):
        raise ValueError('Projection fusion requires the validated Ascend950 with 32 AIC / 64 AIV workers')


def _projection_mm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Use per-operation FP32 math without changing the process HF32 policy."""
    return _native_ops()._mega_gate_projection_mm(left, right)


@lru_cache(maxsize=None)
def _projection_runtime(device: torch.device) -> torch.Tensor:
    """Publish a complete immutable plan once; the hot path does not synchronize."""
    runtime = _build_projection_runtime(device)
    torch.npu.current_stream(device).synchronize()
    return runtime


def _project(hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Use the ordinary adapter loader and optional standard MegaKernel profiling session."""
    native = _native_ops()
    runtime = _projection_runtime(hidden.device)
    profiler = _get_active_session("profiler")
    if profiler is None or not profiler.is_capture_enabled():
        return native._mega_gate_projection(hidden, weight, bias, runtime)
    key = (hidden.device, hidden.shape[0])

    def profile_runtime() -> _TileProfileRuntime:
        """Reuse capture-local metadata for this device and token count."""
        with _PROFILE_LOCK:
            cache = _PROFILE_RUNTIMES.setdefault(profiler, {})
            if key not in cache:
                if len(cache) >= 16:
                    cache.pop(next(iter(cache)))
                cache[key] = _TileProfileRuntime(runtime, hidden.shape[0])
            return cache[key]

    call = profiler.acquire_call(profile_runtime(), runtime, "forward")
    try:
        outputs = native._mega_gate_projection(
            hidden, weight, bias, call.runtime_config, call.diagnostic_buffer,
        )
        call.complete()
        return outputs
    except BaseException:
        call.cancel()
        raise


class _ProjectionRoute(torch.autograd.Function):
    """Combine logit-loss and route gradients before differentiating the projection."""

    @staticmethod
    def forward(ctx: Any, hidden: torch.Tensor, weight: torch.Tensor,
                bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Save invocation-local projection and routing state."""
        logits, weights, indices, scores, selected, denominator, _ = _project(hidden, weight, bias)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(hidden, weight, logits, scores, selected, denominator, indices)
        ctx.plan = _sigmoid_plan_for_device(hidden.device)
        ctx.mark_non_differentiable(indices)
        return logits, weights, indices

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_logits: torch.Tensor | None, grad_weights: torch.Tensor | None,
                 grad_indices: torch.Tensor | None) -> tuple[torch.Tensor | None, torch.Tensor | None, None]:
        """Merge both output gradients and differentiate the full projection."""
        del grad_indices
        hidden, weight, logits, scores, selected, denominator, indices = ctx.saved_tensors
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            if grad_weights is not None:
                route_grad = _sigmoid_route_backward(
                    logits, scores, selected, denominator, indices, grad_weights, ctx.plan, 8, True,
                )
                grad_logits = route_grad if grad_logits is None else grad_logits + route_grad
            if grad_logits is None:
                return None, None, None
            grad_weight = _projection_mm(grad_logits.t(), hidden) if ctx.needs_input_grad[1] else None
            grad_hidden = _projection_mm(grad_logits, weight) if ctx.needs_input_grad[0] else None
        return grad_hidden, grad_weight, None


def _validate_projection_tensors(hidden_states: torch.Tensor, weight: torch.Tensor,
                                 expert_bias: torch.Tensor | None) -> None:
    """Check the local full-weight shape/precision contract before touching an NPU."""
    tensors = (hidden_states, weight) if expert_bias is None else (hidden_states, weight, expert_bias)
    if any(type(tensor) not in (torch.Tensor, torch.nn.Parameter) for tensor in tensors):
        raise ValueError("Projection fusion requires local Torch tensors and a full projection weight")
    if hidden_states.ndim < 2 or weight.ndim != 2 or hidden_states.numel() == 0:
        raise ValueError("Expected nonempty hidden_states [..., hidden] and weight [experts, hidden]")
    experts, hidden = weight.shape
    if hidden <= 0 or hidden % 64 or hidden_states.shape[-1] != hidden or not 8 <= experts <= 384:
        raise ValueError("Projection fusion requires matching hidden dimensions divisible by 64 and 8..384 experts")
    if (hidden_states.dtype not in (torch.float32, torch.bfloat16) or
            weight.dtype not in (torch.float32, torch.bfloat16)):
        raise ValueError("Projection fusion requires FP32 or BF16 hidden states and weight")

def sigmoid_route_from_hidden_states(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    expert_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse an FP32 full-weight projection with normalized sigmoid TopK8 routing.

    Args:
        hidden_states: Nonempty local FP32/BF16 tensor ``[..., hidden]`` on Ascend950.
            The hidden dimension must be a positive multiple of 64.
        weight: Local FP32/BF16 full projection matrix ``[experts, hidden]``, with
            8..384 experts. DTensor and other dispatching subclasses are rejected.
        expert_bias: Optional local FP32 ``[experts]`` selection-only bias on the same device.

    Returns:
        FP32 logits ``[tokens, experts]``, unscaled FP32 normalized weights and
        ordered INT64 indices ``[tokens, 8]``. Casts remain differentiable. Only
        hidden states and weight receive first-order gradients. Full-block
        checkpoint recomputation is supported, including non-reentrant checkpoint.

    Raises:
        ValueError: Unsupported shape, dtype, placement or hardware.

    The caller owns routing scale/casts, auxiliary and z losses, fixed-router
    replacement, replay and expert-bias updates. Call after FSDP has materialized
    the full weight. TP-sharded or Partial weights require the original distributed
    projection followed by ``sigmoid_route_from_logits`` on complete local logits;
    this entry performs no collectives or implicit ``to_local`` conversion.
    FP32-cast-only memory recomputation is not implemented by this entry.
    """
    _validate_projection_tensors(hidden_states, weight, expert_bias)
    experts, hidden = weight.shape
    device = hidden_states.device
    if device.type != "npu" or weight.device != device:
        raise ValueError("Projection fusion requires hidden states and weight on the same Ascend950 NPU")
    _validate_projection_device(device)
    if expert_bias is not None and (
        expert_bias.shape != (experts,) or expert_bias.dtype != torch.float32 or expert_bias.device != device
    ):
        raise ValueError("expert_bias must be FP32 [experts] on the projection device")
    bias = _zero_bias_for_device(device, experts) if expert_bias is None else expert_bias.detach().contiguous()
    with torch.autocast(device_type='npu', enabled=False):
        return _ProjectionRoute.apply(
            hidden_states.reshape(-1, hidden).float().contiguous(), weight.float().contiguous(), bias,
        )
