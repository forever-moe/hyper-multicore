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
"""Correctness reference for owner-sharded grouped experts."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


def _gather_rows(value, group, sizes):
    """Gather rank-ordered rows, padding only the collective transport buffers."""
    maximum = max(sizes)
    padded = value.new_zeros((maximum, *value.shape[1:]))
    padded[:value.shape[0]].copy_(value)
    shards = [torch.empty_like(padded) for _ in sizes]
    dist.all_gather(shards, padded, group=group)
    return torch.cat([shard[:size] for shard, size in zip(shards, sizes)])


class _ReplicatedRows(torch.autograd.Function):
    """Replicate a complete graph; each rank owns only its input/parameter rows.

    _LocalRows supplies all source gradients to each replica, so backward must
    select the owner rows without summing replicas or rounding BF16 partials.
    """

    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        size = torch.tensor([value.shape[0]], device=value.device, dtype=torch.int64)
        sizes = [torch.empty_like(size) for _ in range(dist.get_world_size(group))]
        dist.all_gather(sizes, size, group=group)
        ctx.sizes = [int(item.item()) for item in sizes]
        return _gather_rows(value, group, ctx.sizes)

    @staticmethod
    def backward(ctx, gradient):
        start = sum(ctx.sizes[:ctx.rank])
        return gradient.narrow(0, start, ctx.sizes[ctx.rank]).contiguous(), None


class _LocalRows(torch.autograd.Function):
    """Expose local outputs and gather every source's upstream gradient."""

    @staticmethod
    def forward(ctx, value, local_rows, group):
        ctx.group = group
        size = torch.tensor([local_rows], device=value.device, dtype=torch.int64)
        sizes = [torch.empty_like(size) for _ in range(dist.get_world_size(group))]
        dist.all_gather(sizes, size, group=group)
        ctx.sizes = [int(item.item()) for item in sizes]
        rank = dist.get_rank(group)
        return value.narrow(0, sum(ctx.sizes[:rank]), local_rows).contiguous()

    @staticmethod
    def backward(ctx, gradient):
        return _gather_rows(gradient, ctx.group, ctx.sizes), None, None


class _NpuGroupedMatmul(torch.autograd.Function):
    """PTA grouped-GEMM forward and backward for expert references."""

    @staticmethod
    def forward(ctx, tokens, weights, group_list):
        import torch_npu

        ctx.save_for_backward(tokens, weights, group_list)
        return torch_npu.npu_grouped_matmul(
            x=[tokens], weight=[weights], bias=[], group_list=group_list,
            split_item=3, group_type=0, group_list_type=0,
        )[0]

    @staticmethod
    def backward(ctx, gradient):
        import torch_npu

        tokens, weights, group_list = ctx.saved_tensors
        gradient = gradient.contiguous()
        dx = torch_npu.npu_grouped_matmul(
            x=[gradient], weight=[weights.transpose(-1, -2)], bias=[], group_list=group_list,
            split_item=3, group_type=0, group_list_type=0,
        )[0]
        dw = torch.zeros_like(weights) if tokens.numel() == 0 else torch_npu.npu_grouped_matmul(
            x=[tokens.transpose(0, 1)], weight=[gradient], bias=[], group_list=group_list,
            split_item=2, group_type=2, group_list_type=0,
        )[0]
        return dx, dw, None


class TorchExperts(nn.Module):
    """Owner-sharded parameters with a replicated grouped-expert reference.

    Every group rank must call forward/backward in the same order, including empty
    source ranks. Weights are gathered only for reference correctness. Parameters
    use [local_experts, output, input]; outputs preserve the local source token order.
    This deliberately simple implementation is not a throughput competitor.
    """

    def __init__(self, dim: int, hidden_dim: int, num_experts: int, *, group=None):
        """Create local expert shards; group is borrowed and never destroyed here."""
        super().__init__()
        self.group = group
        world = dist.get_world_size(group) if dist.is_initialized() else 1
        if num_experts % world:
            raise ValueError("Expert count must be divisible by group size")
        self.num_experts = num_experts
        local = num_experts // world
        self.w1 = nn.Parameter(torch.empty(local, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(local, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(local, hidden_dim, dim))
        rank = dist.get_rank(group) if dist.is_initialized() else 0
        # Original GroupedExperts initializes global w1/w2/w3 before EP sharding.
        # Preserve its RNG order, fan-in and owner slice without importing DTensor.
        with torch.no_grad():
            for parameter in (self.w1, self.w2, self.w3):
                global_weight = torch.empty(num_experts, *parameter.shape[1:])
                nn.init.kaiming_uniform_(global_weight.view(num_experts, -1), a=math.sqrt(5))
                parameter.copy_(global_weight.narrow(0, rank * local, local))

    def forward(self, hidden, ids, probabilities):
        """Return local routed outputs; gradients sum into the owning expert shard.

        Args:
            hidden: Local [tokens, dim] tensor, borrowed.
            ids: Global expert IDs [tokens, top_k].
            probabilities: Differentiable route weights of the same shape as ids.

        Returns:
            Local [tokens, dim] weighted output. Backward is collective when distributed.
        """
        weights = (self.w1, self.w3, self.w2)
        distributed = dist.is_initialized() and dist.get_world_size(self.group) > 1
        local_rows = hidden.shape[0]
        if distributed:
            weights = tuple(_ReplicatedRows.apply(weight, self.group) for weight in weights)
            hidden, ids, probabilities = (
                _ReplicatedRows.apply(value, self.group) for value in (hidden, ids, probabilities)
            )
        gate_weight, up_weight, down_weight = weights
        if hidden.device.type == "npu":
            import torch_npu

            packed = torch.cat((gate_weight, up_weight), dim=1).transpose(1, 2)
            routed, mapping = torch_npu.npu_moe_token_permute(hidden, ids.to(torch.int32))
            counts = torch.bincount(ids.reshape(-1).to(torch.int64), minlength=self.num_experts)
            group_list = counts.cumsum(0).to(torch.int64)
            gate_up = grouped_matmul(routed, packed, group_list=group_list)
            activation = torch_npu.npu_swiglu(gate_up, dim=-1)
            values = grouped_matmul(activation, down_weight.transpose(1, 2), group_list=group_list)
            output = torch_npu.npu_moe_token_unpermute(values, mapping, probabilities)
            return _LocalRows.apply(output, local_rows, self.group) if distributed else output
        # All parameters remain in every rank's graph, including experts with no tokens.
        output = hidden * 0 + sum(weight.sum() * 0 for weight in weights) + probabilities.sum() * 0
        for expert in range(self.num_experts):
            rows, slots = torch.where(ids == expert)
            source = hidden.index_select(0, rows)
            gate = F.linear(source, gate_weight[expert])
            up = F.linear(source, up_weight[expert])
            activation = (F.silu(gate.float()) * up.float()).to(hidden.dtype)
            value = F.linear(activation, down_weight[expert])
            weighted = (value.float() * probabilities[rows, slots, None]).to(hidden.dtype)
            output = output.index_add(0, rows, weighted)
        return _LocalRows.apply(output, local_rows, self.group) if distributed else output


def grouped_matmul(tokens: torch.Tensor, weights: torch.Tensor, *, group_list: torch.Tensor) -> torch.Tensor:
    """Multiply consecutive expert segments using the original stock PTA operation.

    Args:
        tokens: Borrowed [routed_tokens, input] tensor already grouped by expert.
        weights: Borrowed [experts, input, output] tensor.
        group_list: Cumulative segment ends, including repeated ends for empty experts.

    Returns:
        Concatenated [routed_tokens, output] results with input/weight gradients.
    """
    if tokens.device.type == "npu":
        return _NpuGroupedMatmul.apply(tokens, weights, group_list)
    starts = [0, *group_list.detach().cpu().tolist()]
    if len(starts) != weights.shape[0] + 1 or starts[-1] != tokens.shape[0]:
        raise ValueError("group_list must describe every expert and routed token")
    return torch.cat([tokens[starts[i]:starts[i + 1]] @ weights[i] for i in range(weights.shape[0])])
