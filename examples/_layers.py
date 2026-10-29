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
"""Ordinary layers needed by the standalone MoE and mHC examples."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F


def _rms_norm(x, weight, *, epsilon):
    """Use the same NPU RMSNorm, or its local CPU equation."""
    if x.device.type == "npu":
        import torch_npu
        return torch_npu.npu_rms_norm(x, weight, epsilon=epsilon)
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + epsilon)
    return (normalized * weight.float()).to(x.dtype), None


class RMSNorm(nn.Module):
    """Direct-scale RMSNorm used by the example architectures."""
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("RMSNorm hidden_size must be a positive integer")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states):
        return _rms_norm(hidden_states, self.weight, epsilon=self.eps)[0]


def _rotary_mul(hidden, cos, sin, *, rotary_mode):
    """Apply the example's half rotation without changing tensor layout."""
    if hidden.device.type == "npu":
        import torch_npu
        return torch_npu.npu_rotary_mul(hidden, cos, sin, rotary_mode=rotary_mode)
    if rotary_mode != "half":
        raise ValueError("Only half rotation is used by these examples")
    first, second = hidden.chunk(2, dim=-1)
    return hidden * cos + torch.cat((-second, first), dim=-1) * sin


def swiglu(hidden):
    """Run the original NPU SwiGLU operation."""
    import torch_npu
    return torch_npu.npu_swiglu(hidden, dim=-1)


def npu_fusion_attention_forward(module, query, key, value, attention_mask=None, dropout=0.0, scaling=None):
    """Run the causal, square BNSD attention path used by TransformerAttention."""
    import torch_npu
    if attention_mask is not None or query.shape[2] != key.shape[2] or not module.is_causal:
        raise ValueError("The examples use square causal attention without a supplied mask")
    mask = torch.ones((2048, 2048), dtype=torch.bool, device=query.device).triu(diagonal=1)
    output = torch_npu.npu_fusion_attention(
        query.contiguous(), key.contiguous(), value.contiguous(), query.shape[1], "BNSD",
        pse=None, padding_mask=None, atten_mask=mask,
        scale=query.shape[-1] ** -0.5 if scaling is None else scaling,
        pre_tockens=getattr(module, "pre_tockens", 1048576),
        next_tockens=getattr(module, "next_tockens", 0), keep_prob=1.0 - dropout,
        inner_precise=0, sparse_mode=3, actual_seq_qlen=None, actual_seq_kvlen=None,
    )[0]
    return output.transpose(1, 2), None


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply NPU RoPE using the Transformers ``apply_rotary_pos_emb`` contract.

    Args:
        q: Query tensor.
        k: Key tensor.
        cos: Cosine part of the rotary embedding.
        sin: Sine part of the rotary embedding.
        unsqueeze_dim: Dimension used to make ``cos`` and ``sin`` broadcastable
            to ``q`` and ``k``.

    Returns:
        Rotated query and key tensors with the same shapes as the inputs.

    Raises:
        ValueError: If the rotary dimension does not cover the complete Q/K
            head dimension supported by the NPU operator interface.
    """
    if cos.ndim == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    if cos.shape[0] == 1 and q.shape[0] > 1:
        cos = cos.expand(q.shape[0], -1, -1)
        sin = sin.expand(q.shape[0], -1, -1)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    if rotary_dim != q.shape[-1] or rotary_dim != k.shape[-1]:
        raise ValueError(
            "NPU RoPE requires the rotary dimension to equal the Q/K head dimension"
        )
    q_embed = _rotary_mul(q.clone(), cos, sin, rotary_mode="half")
    k_embed = _rotary_mul(k.clone(), cos, sin, rotary_mode="half")
    return q_embed, k_embed


class SwiGLUMLP(nn.Module):
    """Transformers-compatible SwiGLU MLP using one fused Gate/Up matmul."""

    @staticmethod
    def _source_projections(module: nn.Module) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
        """Validate and return the source Gate, Up, and Down projections."""
        required = ('gate_proj', 'up_proj', 'down_proj')
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise TypeError(f'SwiGLUMLP source module is missing: {missing}')
        projections = (module.gate_proj, module.up_proj, module.down_proj)
        if not all((isinstance(projection, nn.Linear) for projection in projections)):
            raise TypeError('SwiGLUMLP requires Gate, Up, and Down projections to be nn.Linear')
        return projections

    @staticmethod
    def _validate_projection_contract(gate_proj: nn.Linear, up_proj: nn.Linear, down_proj: nn.Linear) -> None:
        """Validate source projection shapes, placement, and training policy."""
        if gate_proj.in_features != up_proj.in_features:
            raise ValueError('Gate and Up projections must have the same input size')
        if gate_proj.out_features != up_proj.out_features:
            raise ValueError('Gate and Up projections must have the same output size')
        if down_proj.in_features != gate_proj.out_features:
            raise ValueError('Down projection input size must equal the SwiGLU intermediate size')
        if down_proj.out_features != gate_proj.in_features:
            raise ValueError('Down projection output size must equal the MLP hidden size')
        if gate_proj.weight.device != up_proj.weight.device or gate_proj.weight.dtype != up_proj.weight.dtype:
            raise ValueError('Gate and Up projection weights must share device and dtype')
        if gate_proj.weight.device != down_proj.weight.device or gate_proj.weight.dtype != down_proj.weight.dtype:
            raise ValueError('Gate, Up, and Down projection weights must share device and dtype')
        if gate_proj.weight.requires_grad != up_proj.weight.requires_grad:
            raise ValueError('Gate and Up projection weights must share a training policy')
        if (gate_proj.bias is None) != (up_proj.bias is None):
            raise ValueError('Gate and Up projections must use the same bias policy')
        if gate_proj.bias is not None and gate_proj.bias.requires_grad != up_proj.bias.requires_grad:
            raise ValueError('Gate and Up projection biases must share a training policy')

    def _initialize_fc1(self, gate_proj: nn.Linear, up_proj: nn.Linear) -> None:
        """Create the packed Gate/Up projection with source parameter values."""

        packed_weight = torch.cat((gate_proj.weight.detach(), up_proj.weight.detach()), dim=0)
        self.linear_fc1 = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=gate_proj.bias is not None, device=gate_proj.weight.device, dtype=gate_proj.weight.dtype)
        self.linear_fc1.weight = nn.Parameter(packed_weight, requires_grad=gate_proj.weight.requires_grad)
        if gate_proj.bias is not None:
            packed_bias = torch.cat((gate_proj.bias.detach(), up_proj.bias.detach()), dim=0)
            self.linear_fc1.bias = nn.Parameter(packed_bias, requires_grad=gate_proj.bias.requires_grad)

    def _initialize_fc2(self, down_proj: nn.Linear) -> None:
        """Create the Down projection and reuse its source parameters."""
        self.linear_fc2 = nn.Linear(down_proj.in_features, down_proj.out_features, bias=down_proj.bias is not None, device=down_proj.weight.device, dtype=down_proj.weight.dtype)
        self.linear_fc2.weight = down_proj.weight
        self.linear_fc2.bias = down_proj.bias

    def __init__(self, *, module: nn.Module, module_fqn: str='', context: Mapping[str, Any] | None=None) -> None:
        """Build the high-performance MLP from separate source projections.

        Args:
            module: Source MLP exposing ``gate_proj``, ``up_proj``, and ``down_proj``.
            module_fqn: Fully qualified source-module name supplied by replacement.
            context: Replacement context supplied by Trainer.

        Raises:
            TypeError: If the source projection modules are missing or unsupported.
            ValueError: If projection layouts, training policies, or activation differ.
        """
        super().__init__()
        del module_fqn, context
        (gate_proj, up_proj, down_proj) = self._source_projections(module)
        self._validate_projection_contract(gate_proj, up_proj, down_proj)
        config = getattr(module, 'config', None)
        hidden_act = getattr(config, 'hidden_act', getattr(config, 'hidden_activation', None))
        if hidden_act is not None and hidden_act not in ('silu', 'swiglu'):
            raise ValueError(f'SwiGLUMLP requires a SiLU activation, but got {hidden_act}')
        self.config = config
        self.hidden_size = gate_proj.in_features
        self.intermediate_size = gate_proj.out_features
        self._initialize_fc1(gate_proj, up_proj)
        self._initialize_fc2(down_proj)
        self.train(module.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the fused Gate/Up projection, SwiGLU, and Down projection."""
        intermediate_parallel = self.linear_fc1(x)
        if intermediate_parallel.device.type == 'npu':
            intermediate_parallel = swiglu(intermediate_parallel)
        else:
            (gate, up) = intermediate_parallel.chunk(2, dim=-1)
            intermediate_parallel = F.silu(gate) * up
        return self.linear_fc2(intermediate_parallel)
