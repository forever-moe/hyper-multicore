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
"""Local-Tensor AdamW and FP32 master parameters for the integration examples."""

from __future__ import annotations
import logging
from typing import Any, List
import torch
from torch import nn

MIXED_PRECISION_OPTIMIZER_STATE_KEY = "_mixed_precision_optimizer"
FP32_MAIN_PARAM_STATE_KEY = "fp32_from_fp16_params"
logger = logging.getLogger(__name__)


def _copy_tensor(destination: torch.Tensor, source: torch.Tensor) -> None:
    """Copy a local parameter shard with the destination dtype and device."""
    destination.copy_(source.to(device=destination.device, dtype=destination.dtype))


def _gradient_for_param(optimizer_param: nn.Parameter, gradient: torch.Tensor) -> torch.Tensor:
    """Validate a local gradient and cast it without changing its device."""
    if gradient.device != optimizer_param.device:
        raise ValueError("Optimizer gradient must already be on its parameter's device")
    if gradient.shape != optimizer_param.shape:
        raise ValueError("Optimizer gradient local shape must match its parameter shard")
    return gradient.detach().to(dtype=optimizer_param.dtype)


def adamw(
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        max_exp_avg_sqs: List[torch.Tensor],
        step: int,
        *,
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        eps: float,
        maximize: bool
) -> None:
    r"""Functional API that performs AdamW algorithm computation.
    See :class:`~torch.optim.AdamW` for details.
    """
    step_tensor = torch.tensor(step, dtype=torch.int64, device=params[0].device)
    state_steps = [step_tensor] * len(params)

    torch._fused_adamw_(  # pylint: disable=protected-access
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs if amsgrad else [],
        state_steps,
        amsgrad=amsgrad,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize
    )


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer implementation."""

    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            amsgrad=False,
            maximize=False
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
            "maximize": maximize
        }
        super().__init__(params, defaults)

    def __setstate__(self, state):
        """Set optimizer state."""
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)

    def __str__(self):
        return super().__repr__()

    __repr__ = __str__

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []

            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            group['step'] = (group.get('step') or 0) + 1

            current_rank_params = group['params']
            for p in current_rank_params:
                if p.grad is None:
                    continue

                if p.grad.data.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)

                params_with_grad.append(p)
                grads.append(p.grad)
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

            if params_with_grad:
                adamw(
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    group['step'],
                    amsgrad=amsgrad,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    eps=group['eps'],
                    maximize=group['maximize']
                )

        return loss


class MixedPrecisionOptimizer:
    """Base composition for an optimizer that owns precision-conversion state."""

    def __init__(self, optimizer: AdamW, model: nn.Module) -> None:
        """Bind the inner optimizer and expose its scheduler-facing state.

        Args:
            optimizer: Chained leaf-optimizer composition.
            model: Module containing the computation parameters.
        """
        self.optimizer = optimizer
        self.model = model
        self.optimizers_dict: dict[str, Any] = {"adamw": optimizer}
        self.chained_optimizers: list[Any] = [optimizer]
        self.param_groups: list[dict[str, Any]] = optimizer.param_groups

    def prepare_grads(self) -> None:
        """Prepare model gradients for the inner optimizer."""
        raise NotImplementedError

    def step_with_ready_grads(self, closure: Any = None) -> Any:
        """Update parameters after :meth:`prepare_grads` completed.

        Args:
            closure: Optional callable that reevaluates the model and returns loss.
        """
        raise NotImplementedError

    def reload_model_params(self) -> None:
        """Reload optimizer-owned parameters from the registered model."""
        raise NotImplementedError


class Float16OptimizerWithFloat16Params(MixedPrecisionOptimizer):
    """Update fp32 main params and copy them back to fp16/bf16 model params."""

    def __init__(self, optimizer: AdamW, model: nn.Module) -> None:
        """Replace leaf optimizer params with fp32 main parameters.

        Args:
            optimizer: Chained AdamW/Muon optimizer over model parameters.
            model: Module containing the computation parameters.
        """
        super().__init__(optimizer, model)
        self.float16_groups: list[list[nn.Parameter]] = []
        self.fp32_from_float16_groups: list[list[nn.Parameter]] = []
        self.fp32_from_fp32_groups: list[list[nn.Parameter]] = []
        self.model_param_by_optimizer_param: dict[nn.Parameter, nn.Parameter] = {}
        self._build_model_and_main_param_groups()
        self.optimizer_param_by_model_param = {
            model_param: optimizer_param
            for optimizer_param, model_param in self.model_param_by_optimizer_param.items()
        }
        self.param_groups = self.optimizer.param_groups

        parameter_names = {
            parameter: name for name, parameter in model.named_parameters()
        }
        self._main_param_by_fqn = {
            parameter_names[model_param]: optimizer_param
            for optimizer_param, model_param in self.model_param_by_optimizer_param.items()
        }

    def _build_model_and_main_param_groups(self) -> None:
        """Build Megatron-compatible model and main-parameter group views."""
        parameter_names = {
            parameter: name for name, parameter in self.model.named_parameters()
        }
        for leaf_optimizer in self.chained_optimizers:
            for optimizer_group in leaf_optimizer.param_groups:
                float16_group = []
                fp32_from_float16_group = []
                fp32_from_fp32_group = []
                for param_index, model_param in enumerate(optimizer_group["params"]):
                    model_param.main_grad = None
                    if model_param.dtype == torch.float32:
                        model_param.main_param = model_param
                        fp32_from_fp32_group.append(model_param)
                        continue
                    if model_param.dtype not in (torch.float16, torch.bfloat16):
                        raise TypeError(
                            "Float16OptimizerWithFloat16Params supports trainable "
                            "float16, bfloat16, or float32 parameters; "
                            f"{parameter_names[model_param]} has dtype {model_param.dtype}"
                        )

                    main_param = nn.Parameter(
                        model_param.detach().to(dtype=torch.float32),
                        requires_grad=model_param.requires_grad,
                    )
                    main_param.model_name = parameter_names[model_param]
                    model_param.main_param = main_param
                    optimizer_group["params"][param_index] = main_param
                    if model_param in leaf_optimizer.state:
                        leaf_optimizer.state[main_param] = leaf_optimizer.state.pop(model_param)

                    float16_group.append(model_param)
                    fp32_from_float16_group.append(main_param)
                    self.model_param_by_optimizer_param[main_param] = model_param

                self.float16_groups.append(float16_group)
                self.fp32_from_float16_groups.append(fp32_from_float16_group)
                self.fp32_from_fp32_groups.append(fp32_from_fp32_group)

    def _copy_model_grads_to_main_grads(self) -> None:
        """Transfer model gradients to fp32 optimizer params, casting dtype only.

        The producer must supply the optimizer parameter's device and layout.
        Rebind existing ``main_grad`` buffers to the prepared gradient and clear
        model ``grad``. The local source reference is overwritten before the
        next conversion or released on return, so reference counting reclaims
        converted source buffers before the optimizer step.
        External owners of shared gradient storage remain responsible for it.
        """
        for model_group, main_group in zip(
                self.float16_groups,
                self.fp32_from_float16_groups,
        ):
            for model_param, main_param in zip(model_group, main_group):
                model_gradient = model_param.main_grad
                if model_gradient is None:
                    model_gradient = model_param.grad
                main_param.grad = (
                    _gradient_for_param(main_param, model_gradient)
                    if model_gradient is not None else None
                )
                if model_param.main_grad is not None:
                    model_param.main_grad = main_param.grad
                model_param.grad = None

        for fp32_group in self.fp32_from_fp32_groups:
            for model_param in fp32_group:
                model_gradient = model_param.main_grad
                if model_gradient is not None:
                    model_param.grad = _gradient_for_param(model_param, model_gradient)
                    model_param.main_grad = model_param.grad

    @torch.no_grad()
    def _copy_main_params_to_model_params(self) -> None:
        """Copy updated fp32 main params to fp16/bf16 model params."""
        for model_group, main_group in zip(
                self.float16_groups,
                self.fp32_from_float16_groups,
        ):
            for model_param, main_param in zip(model_group, main_group):
                _copy_tensor(model_param, main_param)

    @torch.no_grad()
    def _copy_model_params_to_main_params(self) -> None:
        """Refresh fp32 main params from fp16/bf16 model params."""
        for model_group, main_group in zip(
                self.float16_groups,
                self.fp32_from_float16_groups,
        ):
            for model_param, main_param in zip(model_group, main_group):
                _copy_tensor(main_param, model_param)

    def prepare_grads(self) -> None:
        """Make fp32 main gradients ready for AdamW."""
        self._copy_model_grads_to_main_grads()

    def step_with_ready_grads(self, closure: Any = None) -> Any:
        """Update main params, then copy them back to model params.

        Args:
            closure: Optional callable that reevaluates the model and returns loss.

        Returns:
            The inner optimizer step result.
        """
        loss = self.optimizer.step(closure=closure)
        self._copy_main_params_to_model_params()
        return loss

    def step(self, closure: Any = None) -> Any:
        """Prepare gradients and execute one mixed-precision optimizer step.

        Args:
            closure: Optional callable that reevaluates the model and returns loss.

        Returns:
            The inner optimizer step result.
        """
        self.prepare_grads()
        return self.step_with_ready_grads(closure=closure)

    def reload_model_params(self) -> None:
        """Initialize main params from the current model parameter values."""
        self._copy_model_params_to_main_params()

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear inner grads plus runtime-only model grad/main_grad state.

        Args:
            set_to_none: Whether inner gradients are reset to ``None``.
        """
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for model_param in self.optimizer_param_by_model_param:
            model_param.grad = None
            model_param.main_grad = None
            if hasattr(model_param, "grad_added_to_main_grad"):
                model_param.grad_added_to_main_grad = False
        for fp32_group in self.fp32_from_fp32_groups:
            for model_param in fp32_group:
                model_param.grad = None
                model_param.main_grad = None
                if hasattr(model_param, "grad_added_to_main_grad"):
                    model_param.grad_added_to_main_grad = False

    def state_dict(self) -> dict[str, Any]:
        """Append fp32 main weights to the existing optimizer FQN state."""
        state_dict = self.optimizer.state_dict()
        state_dict[MIXED_PRECISION_OPTIMIZER_STATE_KEY] = {
            FP32_MAIN_PARAM_STATE_KEY: dict(self._main_param_by_fqn),
        }
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore optimizer moments and explicit fp32 main weights.

        Args:
            state_dict: Optimizer state containing leaf and main-param entries.
        """
        optimizer_state = dict(state_dict)
        if MIXED_PRECISION_OPTIMIZER_STATE_KEY not in optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)
            self.param_groups = self.optimizer.param_groups
            self.reload_model_params()
            logger.warning(
                "Checkpoint has no %s state; initialized fp32 main parameters "
                "from the loaded model parameters. Values not representable in "
                "the model precision cannot be recovered.",
                MIXED_PRECISION_OPTIMIZER_STATE_KEY,
            )
            return

        mixed_precision_state = optimizer_state.pop(
            MIXED_PRECISION_OPTIMIZER_STATE_KEY
        )
        if not isinstance(mixed_precision_state, dict):
            raise RuntimeError(
                "Mixed-precision optimizer checkpoint state must be a mapping"
            )
        checkpoint_main_params = mixed_precision_state.get(
            FP32_MAIN_PARAM_STATE_KEY
        )
        if not isinstance(checkpoint_main_params, dict):
            raise RuntimeError(
                "Mixed-precision optimizer checkpoint is missing "
                f"{FP32_MAIN_PARAM_STATE_KEY}"
            )
        missing_fqns = sorted(
            set(self._main_param_by_fqn) - set(checkpoint_main_params)
        )
        if missing_fqns:
            raise RuntimeError(
                "Mixed-precision optimizer checkpoint is missing fp32 main "
                f"parameters: {', '.join(missing_fqns)}"
            )

        self.optimizer.load_state_dict(optimizer_state)
        self.param_groups = self.optimizer.param_groups
        with torch.no_grad():
            for parameter_fqn, main_param in self._main_param_by_fqn.items():
                _copy_tensor(main_param, checkpoint_main_params[parameter_fqn])
