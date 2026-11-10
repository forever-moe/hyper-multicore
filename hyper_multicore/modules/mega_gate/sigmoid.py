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
"""A5 logits-only sigmoid routing with native normalization and autograd."""

from __future__ import annotations

from functools import lru_cache
from threading import Lock

from typing import Any

import torch
from torch.autograd.function import once_differentiable

from .function import _native_ops
from .plan import MegaGatePlan, build_mega_gate_plan


_ZERO_BIAS_CACHE: dict[tuple[torch.device, int], torch.Tensor] = {}
_ZERO_BIAS_LOCK = Lock()


def _zero_bias_for_device(device: torch.device, experts: int) -> torch.Tensor:
    """Publish an initialized read-only selection bias once per device and width."""
    key = (device, experts)
    with _ZERO_BIAS_LOCK:
        bias = _ZERO_BIAS_CACHE.get(key)
        if bias is None:
            bias = torch.zeros(experts, dtype=torch.float32, device=device)
            # Complete the cold initialization before another stream can borrow
            # this constant. Cache hits never synchronize or share writable state.
            torch.npu.current_stream(device).synchronize()
            _ZERO_BIAS_CACHE[key] = bias
    return bias


@lru_cache(maxsize=None)
def _sigmoid_plan_for_device(device: torch.device) -> MegaGatePlan:
    """Keep sigmoid diagnostic metadata separate from sqrt-softplus plans."""
    return build_mega_gate_plan(device, use_sigmoid=True)


class _SigmoidRoute(torch.autograd.Function):
    """Keep projection, scaling, cast, auxiliary loss and bias updates with the caller."""

    @staticmethod
    def forward(ctx: Any, logits: torch.Tensor, bias: torch.Tensor, plan: MegaGatePlan,
                top_k: int, normalize: bool, use_bias: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Save the native sigmoid routing state for first-order backward."""
        ctx.set_materialize_grads(False)
        runtime = plan.route_runtime
        options = {"diagnostics": runtime} if runtime.is_active else {}
        weights, indices, scores, selected, denominator = _native_ops()._mega_gate_sigmoid_route(
            logits, bias, plan.vision_mask_placeholder, runtime.runtime_config, runtime.runtime_config,
            top_k=top_k, normalize=normalize, use_bias=use_bias, **options,
        )
        ctx.save_for_backward(logits, scores, selected, denominator, indices)
        ctx.plan = plan
        ctx.top_k = top_k
        ctx.normalize = normalize
        ctx.mark_non_differentiable(indices)
        return weights, indices

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_weights: torch.Tensor | None,
                 grad_indices: torch.Tensor | None) -> tuple[torch.Tensor | None, None, None, None, None, None]:
        """Return only the logits gradient; correction bias controls selection."""
        del grad_indices
        if grad_weights is None or not ctx.needs_input_grad[0]:
            return None, None, None, None, None, None
        logits, scores, selected, denominator, indices = ctx.saved_tensors
        gradient = _sigmoid_route_backward(logits, scores, selected, denominator, indices,
                                           grad_weights, ctx.plan, ctx.top_k, ctx.normalize)
        return gradient, None, None, None, None, None


def _sigmoid_route_backward(logits, scores, selected, denominator, indices, grad_weights,
                            plan: MegaGatePlan, top_k: int, normalize: bool) -> torch.Tensor:
    """Share the native route derivative and diagnostic launch across public entry points."""
    runtime = (plan.route_grad_runtime if top_k > 1 and normalize
               else plan.route_grad_k1_runtime)
    inputs = (logits, scores, selected, denominator, indices, grad_weights.contiguous())
    if not runtime.is_active:
        gradient = _native_ops()._mega_gate_sigmoid_route_grad(
            *inputs, runtime.runtime_config, runtime.runtime_config, top_k=top_k, normalize=normalize,
        )
    else:
        parameters = [selected, denominator, inputs[5], scores, indices,
                      runtime.runtime_config, runtime.runtime_config, None, None, None, logits]
        with runtime.launch(parameters, direction="backward") as call:
            gradient = _native_ops()._mega_gate_sigmoid_route_grad(
                *inputs, call.runtime_config, call.diagnostic_buffer, top_k=top_k, normalize=normalize,
            )
            parameters[9] = gradient
    return gradient


def _validate_sigmoid_shape(logits: torch.Tensor, top_k: int, normalize: bool) -> None:
    """Reject unsupported selection widths and dispatching tensors before initializing resources."""
    if type(logits) not in (torch.Tensor, torch.nn.Parameter):
        raise ValueError("sigmoid_route_from_logits requires a local Torch tensor")
    if logits.ndim != 2 or not all(logits.shape):
        raise ValueError(f"Expected nonempty logits shape [tokens, experts], got {tuple(logits.shape)}")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= logits.shape[1]:
        raise ValueError(f"top_k must be in [1, {logits.shape[1]}], got {top_k}")
    if normalize and top_k > 8 and top_k not in (16, 128, 384):
        raise ValueError(f"Normalized sigmoid routing is not calibrated for top_k={top_k}")
    if logits.shape[1] > 384:
        raise ValueError(f"At most 384 experts are supported, got {logits.shape[1]}")

def sigmoid_route_from_logits(
    logits: torch.Tensor,
    *,
    top_k: int,
    expert_bias: torch.Tensor | None = None,
    normalize: bool = True,
    plan: MegaGatePlan | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select ordered experts and return unscaled FP32 sigmoid route weights on A5.

    Args:
        logits: Local ``[tokens, experts]`` FP32, FP16 or BF16 tensor on Ascend950, with at most 384 experts.
        top_k: Number of selected experts in ``[1, experts]``.
        expert_bias: Optional FP32 ``[experts]`` bias affecting selection only.
        normalize: Divide selected scores by their row sum plus 1e-20 when K > 1.
            Normalized routing supports K=1..8, 16, 128 and 384.
        plan: Optional device-local plan built with ``use_sigmoid=True`` for sigmoid stage identities.
            The same plan supports normal execution, profiling and tensor dumps.

    Returns:
        Unscaled FP32 weights and ordered INT64 indices, both ``[tokens, top_k]``.
        Only logits receive gradients; first-order autograd is supported. The caller
        retains its original optional scaling and output-dtype conversion operations.

    Raises:
        ValueError: Inputs or the device are outside the supported contract.
    """
    _validate_sigmoid_shape(logits, top_k, normalize)
    if plan is not None and not plan.use_sigmoid:
        raise ValueError("Sigmoid routing requires a plan built with use_sigmoid=True")
    if logits.device.type != "npu":
        raise ValueError(f"sigmoid_route_from_logits requires an Ascend950 NPU, got {logits.device}")
    if not torch.npu.get_device_name(logits.device).startswith("Ascend950"):
        raise ValueError("sigmoid_route_from_logits is calibrated for Ascend950 only")
    if logits.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"Unsupported logits dtype: {logits.dtype}")
    if expert_bias is not None and (
        type(expert_bias) not in (torch.Tensor, torch.nn.Parameter)
        or expert_bias.shape != (logits.shape[1],) or expert_bias.dtype != torch.float32
        or expert_bias.device != logits.device
    ):
        raise ValueError("expert_bias must be FP32 [experts] on the logits device")
    resolved_plan = _sigmoid_plan_for_device(logits.device) if plan is None else plan
    if resolved_plan.vision_mask_placeholder.device != logits.device:
        raise ValueError("The sigmoid route plan must reside on the logits device")
    bias = (_zero_bias_for_device(logits.device, logits.shape[1]) if expert_bias is None
            else expert_bias.detach().contiguous())
    return _SigmoidRoute.apply(
        logits.float().contiguous(), bias, resolved_plan, top_k, normalize, expert_bias is not None,
    )
