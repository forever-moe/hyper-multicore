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
"""Autograd handoff between current MXFP8 root Parameters and physical execution."""

from dataclasses import dataclass
from typing import Any

import torch

from ...route import PreparedTopKRoute
from .weights import MXFP8ExpertWeightBinding


@dataclass(frozen=True)
class MXFP8ForwardState:
    """Owned output, ordinary saved activations and execution metadata for backward."""

    output: torch.Tensor
    saved_tensors: tuple[torch.Tensor, ...]
    metadata: Any
    workspace: Any = None


class _MXFP8ExpertsFunction(torch.autograd.Function):
    """Retain one MX invocation for autograd and resolve current weights by phase."""

    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        root1: torch.Tensor,
        root2: torch.Tensor,
        route: PreparedTopKRoute,
        probs: torch.Tensor,
        shapes: tuple[tuple[int, int, int], tuple[int, int, int]],
        execution: Any,
        grad_enabled: bool,
        shared_root1: torch.Tensor | None,
        shared_root2: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the MX graph and retain this invocation when gradients are required."""
        ctx.weight1 = MXFP8ExpertWeightBinding(root1, shapes[0])
        ctx.weight2 = MXFP8ExpertWeightBinding(root2, shapes[1])
        ctx.shared_weights = None
        shared_weights = None
        if shared_root1 is not None:
            shared_w1 = MXFP8ExpertWeightBinding(shared_root1, (1, *shared_root1.shape), forward_axis="row")
            shared_w2 = MXFP8ExpertWeightBinding(shared_root2, (1, *shared_root2.shape), forward_axis="row")
            ctx.shared_weights = shared_w1, shared_w2
            shared_weights = shared_w1.resolve('forward'), shared_w2.resolve('forward')
        retain_backward = grad_enabled and any(ctx.needs_input_grad)
        state = execution.forward(
            hidden, route, probs, ctx.weight1.resolve('forward'), ctx.weight2.resolve('forward'),
            retain_backward, shared_weights,
        )
        ctx.execution = execution
        ctx.execution_metadata = state.metadata
        ctx.save_for_backward(*state.saved_tensors)
        ctx.workspace = state.workspace if retain_backward else None
        if ctx.workspace is not None:
            ctx.workspace.track_graph(ctx)
        return state.output

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[Any, ...]:
        """Resolve backward weight storage and return logical gradients."""
        # Parameter materialization hooks replace physical fields on the retained roots.
        # Never reuse the forward-only physical views held by native launch code.
        weight1 = ctx.weight1.resolve('backward')
        weight2 = ctx.weight2.resolve('backward')
        shared_weights = None
        if ctx.shared_weights is not None:
            shared_w1, shared_w2 = ctx.shared_weights
            shared_weights = shared_w1.resolve('backward'), shared_w2.resolve('backward')
        dx, dp, dw1, dw2, shared_grads = ctx.execution.backward(
            gradient, ctx.saved_tensors, ctx.execution_metadata, weight1, weight2, shared_weights,
        )
        dw_shared1, dw_shared2 = None, None
        if shared_grads is not None:
            dw_shared1, dw_shared2 = (grad.reshape(binding.root.shape).to(dtype=binding.root.dtype)
                                      for binding, grad in zip(ctx.shared_weights, shared_grads))
        dw1 = dw1.reshape(ctx.weight1.root.shape).to(dtype=ctx.weight1.root.dtype)
        dw2 = dw2.reshape(ctx.weight2.root.shape).to(dtype=ctx.weight2.root.dtype)
        # The native autograd task owns the retain_graph flag.
        # pylint: disable-next=protected-access
        if ctx.workspace is not None and not torch._C._autograd._get_current_graph_task_keep_graph():
            ctx.workspace.release_graph(ctx)
        return dx, dw1, dw2, None, dp, None, None, None, dw_shared1, dw_shared2


def execute_mxfp8_experts(
    hidden: torch.Tensor,
    root1: torch.Tensor,
    root2: torch.Tensor,
    route: PreparedTopKRoute,
    probs: torch.Tensor,
    shapes: tuple[tuple[int, int, int], tuple[int, int, int]],
    execution: Any,
    *,
    shared_expert_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Keep root identity across phases and expose physical execution to autograd.

    Each phase resolves the axis consumed by its routed and shared GEMMs.
    Keep the trainable roots as explicit autograd inputs.
    """
    return _MXFP8ExpertsFunction.apply(
        hidden, root1, root2, route, probs, shapes, execution, torch.is_grad_enabled(),
        *(shared_expert_weights or (None, None)),
    )
