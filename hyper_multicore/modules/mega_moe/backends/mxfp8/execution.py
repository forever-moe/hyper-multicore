# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Orchestrate native MX expert graphs without retaining gathered weight views."""

from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch_npu

from hyper_multicore.torch import ops as multicore_ops

from .autograd import MXFP8ForwardState
from .diagnostics import current_mx_profile_scope, mx_profile_capture_enabled, profile_mx_launch
from .launch import launch_mxfp8_experts, prepare_mxfp8_launch
from .weights import MXFP8PhysicalWeight
from ...route import PreparedTopKRoute


@dataclass(frozen=True)
class _ExecutionMetadata:
    plan: Any
    hidden_shape: tuple[int, ...]
    source_rows: int
    probs_dtype: torch.dtype


def _quantized_buffers(rows: int, columns: int, experts: int, device: Any) -> tuple[torch.Tensor, ...]:
    """Allocate both data axes and preserve grouped packed-column padding."""
    return (
        torch.empty((rows, columns), dtype=torch.float8_e4m3fn, device=device),
        torch.empty((rows, columns // 64, 2), dtype=torch.uint8, device=device),
        torch.empty((rows, columns), dtype=torch.float8_e4m3fn, device=device),
        torch.empty(((rows + 63) // 64 + experts, columns, 2), dtype=torch.uint8, device=device),
    )


def _shared_quant_buffers(name, rows, columns, device):
    keys = (f'{name}_row', f'{name}_row_scale', f'{name}_col', f'{name}_col_scale')
    return dict(zip(keys, _quantized_buffers(rows, columns, 1, device)))


def _shared_forward_buffers(hidden, width, weight1, weight2):
    h, device = hidden.shape[-1], hidden.device
    source = hidden.reshape(-1, h).contiguous()
    rows = source.shape[0]
    return {
        'group': torch.full((1,), rows, dtype=torch.int64, device=device),
        'w1': weight1.row_data, 'w1_scale': weight1.row_scale,
        'w2': weight2.row_data, 'w2_scale': weight2.row_scale,
        'x': source, 'gate_up': torch.empty((rows, 2 * width), dtype=torch.bfloat16, device=device),
        'output': torch.empty_like(source),
        **_shared_quant_buffers('x', rows, h, device),
        **_shared_quant_buffers('activation', rows, width, device),
    }


def _shared_backward_buffers(dy, width, dtype, saved, weight1, weight2):
    rows, h = dy.shape
    device = dy.device
    group, x_col, x_scale, a_col, a_scale, gate_up = saved
    return {
        'group': group,
        'w1': weight1.col_data, 'w1_scale': weight1.col_scale,
        'w2': weight2.col_data, 'w2_scale': weight2.col_scale,
        'dy': dy, 'gate_up': gate_up,
        'x_col': x_col, 'x_col_scale': x_scale,
        'activation_col': a_col, 'activation_col_scale': a_scale,
        'da': torch.empty((rows, width), dtype=torch.bfloat16, device=device),
        'dx': torch.empty_like(dy),
        'dw1': torch.empty((2 * width, h), dtype=dtype, device=device),
        'dw2': torch.empty((h, width), dtype=dtype, device=device),
        **_shared_quant_buffers('dy', rows, h, device),
        **_shared_quant_buffers('du', rows, 2 * width, device),
    }


class MXFP8Execution:
    """Own one autograd invocation's orchestration while borrowing serial resources."""

    def __init__(self, resources: Any, *, nograd_no_clone: bool = True) -> None:
        self.resources = resources
        self.nograd_no_clone = nograd_no_clone
        self._profile_scope = None

    def _launch(self, buffers, runtime, events, plan, direction="forward"):
        """Submit the graph with exact buffer bounds and optional profiling."""
        if events.numel() < runtime.event_capacity * 4:
            raise ValueError("MX event storage is smaller than the graph")
        launch_graph = partial(launch_mxfp8_experts, direction=direction)
        if mx_profile_capture_enabled():
            scope = current_mx_profile_scope() or self._profile_scope
            if direction == 'forward':
                self._profile_scope = scope
            elif scope is not None:
                scope = {**scope, 'phase': 'backward'}
            profile_mx_launch(buffers, runtime, events, plan.spec, direction,
                              prepare_mxfp8_launch, launch_graph, scope=scope)
            return
        launch = prepare_mxfp8_launch(buffers, runtime.runtime, events, plan.spec)
        # Disabled diagnostics never read this binding, matching the BF16 adapter.
        launch_graph(buffers, runtime.runtime, events, events, launch)

    def forward(self, hidden: torch.Tensor, route: PreparedTopKRoute, probs: torch.Tensor,
                weight1: MXFP8PhysicalWeight, weight2: MXFP8PhysicalWeight,
                retain_backward: bool,
                shared_weights: tuple[MXFP8PhysicalWeight, MXFP8PhysicalWeight] | None = None) -> MXFP8ForwardState:
        """Prepare MX operands and restore the forward output to token order."""
        resources, workspace = self.resources, self.resources.workspace
        spec, device = resources.spec, hidden.device
        metadata = route.metadata
        pull = spec.dispatch_mode == "pull"
        capacity = metadata.expert_capacity if pull else workspace.expert_capacity
        shared_rows = hidden.numel() // spec.hidden_size if shared_weights is not None else None
        plan = resources.plan_for_capacity(capacity, device, tokens=metadata.plan_tokens, shared_rows=shared_rows)
        if pull:
            dispatch_buffer = torch.empty((capacity, spec.hidden_size), dtype=torch.bfloat16, device=device)
            source_buffer = workspace.source_buffer
        else:
            dispatch_buffer = workspace.expert_buffer
            source_buffer = route.routed_tokens
        x_row, x_scale, x_col, x_col_scale = _quantized_buffers(capacity, spec.hidden_size, spec.local_experts, device)
        a_row, a_scale, a_col, a_col_scale = _quantized_buffers(
            capacity, spec.intermediate_size, spec.local_experts, device)
        up = torch.empty((capacity, 2 * spec.intermediate_size), dtype=torch.bfloat16, device=device)
        down = torch.empty((capacity, spec.hidden_size), dtype=torch.bfloat16, device=device)
        if not pull:
            workspace.claim()
        try:
            buffers = [
                dispatch_buffer, metadata.dispatch_target_off * spec.hidden_size, source_buffer,
                metadata.dispatch_src_off * spec.hidden_size, metadata.dispatch_size * spec.hidden_size,
                weight1.col_data, metadata.group_list, up, a_row, weight2.col_data, metadata.group_list, down,
                workspace.routed_buffer, metadata.combine_target_off * spec.hidden_size,
                metadata.combine_src_off * spec.hidden_size, metadata.combine_size * spec.hidden_size,
                x_row, x_scale, x_col, x_col_scale, a_scale, a_col, a_col_scale,
                weight1.col_scale, weight2.col_scale, plan.forward.gmm, plan.forward.quant,
            ]
            shared = None
            if shared_weights is not None:
                shared = _shared_forward_buffers(hidden, spec.shared_intermediate_size, *shared_weights)
                shared['qbmm'] = plan.forward.qbmm
                buffers.extend(shared[name] for name in plan.forward.host.shared_buffer_names)
            events = workspace.prepare_event_counters(forward=True)
            self._launch(buffers, plan.forward, events, plan)
            expert_output = workspace.routed_buffer[:route.routed_tokens.shape[0]]
            if retain_backward or not self.nograd_no_clone:
                expert_output = expert_output.clone()
            probabilities = probs.float().contiguous()
            output = torch_npu.npu_moe_token_unpermute(expert_output, route.unpermute_mapping, probs=probabilities)
            if (self.nograd_no_clone and not retain_backward
                    and output.untyped_storage().data_ptr() == workspace.routed_buffer.untyped_storage().data_ptr()):
                raise RuntimeError('No-grad MX unpermute output must not alias shared routed storage.')
            saved = ((x_col, x_col_scale, a_col, a_col_scale, up, expert_output, probabilities,
                      route.unpermute_mapping, metadata.group_list, metadata.dispatch_target_off,
                      metadata.dispatch_src_off, metadata.dispatch_size, metadata.combine_target_off,
                      metadata.combine_src_off, metadata.combine_size) if retain_backward else ())
            if shared is not None:
                shared_output = shared['output']
                output = (output + shared_output if spec.shared_expert_add_order == 'routed_first'
                          else shared_output + output)
                if retain_backward:
                    saved += tuple(shared[name] for name in (
                        'group', 'x_col', 'x_col_scale', 'activation_col', 'activation_col_scale', 'gate_up',
                    ))
            return MXFP8ForwardState(
                output.reshape_as(hidden), saved,
                _ExecutionMetadata(plan, tuple(hidden.shape), route.routed_tokens.shape[0], probs.dtype), workspace,
            )
        finally:
            if not pull:
                workspace.release()

    def backward(self, gradient: torch.Tensor, saved: tuple[torch.Tensor, ...], metadata: _ExecutionMetadata,
                 weight1: MXFP8PhysicalWeight, weight2: MXFP8PhysicalWeight,
                 shared_weights: tuple[MXFP8PhysicalWeight, MXFP8PhysicalWeight] | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                            tuple[torch.Tensor, torch.Tensor] | None]:
        """Run the backward graph on the saved route and current weights."""
        workspace, plan = self.resources.workspace, metadata.plan
        spec, device = plan.spec, gradient.device
        (x_col, x_col_scale, a_col, a_col_scale, up, expert_output, probs, mapping, group_list,
         dispatch_target, dispatch_source, dispatch_size, combine_target, combine_source, combine_size,
         *shared_saved) = saved
        capacity = x_col.shape[0]
        dy_row, dy_scale, dy_col, dy_col_scale = _quantized_buffers(
            capacity, spec.hidden_size, spec.local_experts, device)
        du_row, du_scale, du_col, du_col_scale = _quantized_buffers(
            capacity, 2 * spec.intermediate_size, spec.local_experts, device)
        pull = spec.dispatch_mode == 'pull'
        source = (workspace.source_buffer[:metadata.source_rows] if pull else
                  torch.empty((metadata.source_rows, spec.hidden_size), dtype=torch.bfloat16, device=device))
        dispatch = (torch.empty((capacity, spec.hidden_size), dtype=torch.bfloat16, device=device)
                    if pull else workspace.expert_buffer)
        grad_probs = torch.empty_like(probs)
        da = torch.empty((capacity, spec.intermediate_size), dtype=torch.bfloat16, device=device)
        dx = torch.empty((capacity, spec.hidden_size), dtype=torch.bfloat16, device=device)
        partial_dtype = torch.float32 if spec.partial_weight_grad_dtype == "float32" else torch.bfloat16
        dw1 = torch.zeros(weight1.logical_shape, dtype=partial_dtype, device=device)
        dw2 = torch.zeros(weight2.logical_shape, dtype=partial_dtype, device=device)
        workspace.claim()
        try:
            dy = gradient.reshape(-1, spec.hidden_size).contiguous()
            multicore_ops.mega_moe_unpermute_grad_out(expert_output, dy, mapping, probs, source, grad_probs)
            buffers = [
                dispatch, dispatch_target * spec.hidden_size, workspace.source_buffer if pull else source,
                dispatch_source * spec.hidden_size, dispatch_size * spec.hidden_size,
                a_col, dw2, weight2.row_data, da, up, du_row, weight1.row_data, dx,
                workspace.routed_buffer, combine_target * spec.hidden_size, combine_source * spec.hidden_size,
                combine_size * spec.hidden_size, x_col, dw1, group_list, dy_row, dy_scale, dy_col, dy_col_scale,
                a_col_scale, weight2.row_scale, du_scale, du_col, du_col_scale, weight1.row_scale, x_col_scale,
                plan.backward.gmm, plan.backward.quant,
            ]
            shared = None
            if shared_weights is not None:
                shared = _shared_backward_buffers(
                    dy, spec.shared_intermediate_size, partial_dtype, shared_saved, *shared_weights,
                )
                shared['qbmm'] = plan.backward.qbmm
                buffers.extend(shared[name] for name in plan.backward.host.shared_buffer_names)
            events = workspace.prepare_event_counters(forward=False)
            self._launch(buffers, plan.backward, events, plan, direction="backward")
            grad_hidden = multicore_ops.moe_token_permute_grad(
                workspace.routed_buffer[:metadata.source_rows], mapping,
                gradient.numel() // spec.hidden_size, spec.top_k,
            )
            shared_grads = None
            if shared is not None:
                grad_hidden = grad_hidden + shared['dx']
                shared_grads = shared['dw1'], shared['dw2']
            return (grad_hidden.reshape(metadata.hidden_shape), grad_probs.to(metadata.probs_dtype),
                    dw1, dw2, shared_grads)
        finally:
            workspace.release()
