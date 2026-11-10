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
"""Projection-fusion boundary, repeated profiling capacity and autograd contracts."""

from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

from hyper_multicore.modules.mega_gate import sigmoid_route_from_hidden_states
from hyper_multicore.modules.mega_gate import projection
from hyper_multicore.modules.mega_gate import plan as gate_plan
from hyper_multicore.modules.mega_gate.projection_profile import _TileProfileRuntime


class TestProjectionRoute(unittest.TestCase):
    """Keep unsupported dispatch/precision out and preserve every gradient source."""

    def test_unsupported_inputs_do_not_load_native(self) -> None:
        """Reject malformed, sharded and unsupported inputs at the public boundary."""
        class DispatchTensor(torch.Tensor):
            """Stand in for an external tensor dispatch implementation."""

        cases = [(torch.empty(2, 64).as_subclass(DispatchTensor), torch.empty(12, 64), "local"),
                 (torch.empty(0, 64), torch.empty(12, 64), "nonempty"),
                 (torch.empty(2, 128), torch.empty(12, 64), "matching"),
                 (torch.empty(2, 64), torch.empty(7, 64), "8..384"),
                 (torch.empty(2, 64, dtype=torch.float16), torch.empty(12, 64), "FP32 or BF16"),
                 (torch.empty(2, 64), torch.empty(12, 64), "Ascend950")]
        with patch.object(projection, "_native_ops") as native:
            for hidden, weight, error in cases:
                with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                    sigmoid_route_from_hidden_states(hidden, weight)
            native.assert_not_called()

    def test_projection_merges_logit_and_route_gradients(self) -> None:
        """Cover unused outputs, frozen input/weight and simultaneous auxiliary losses."""
        def _project(hidden, weight, bias):
            logits = hidden @ weight.t()
            scores = logits.sigmoid()
            indices = (scores + bias).topk(8).indices
            selected = scores.gather(1, indices)
            denominator = selected.sum(-1, keepdim=True) + 1.0e-20
            return logits, selected / denominator, indices, scores, selected, denominator, None

        def _route_backward(logits, scores, selected, denominator, indices, grad, *args):
            del scores, selected, denominator, args
            with torch.enable_grad():
                detached = logits.detach().requires_grad_()
                selected_scores = detached.sigmoid().gather(1, indices)
                weights = selected_scores / (selected_scores.sum(-1, keepdim=True) + 1.0e-20)
                return torch.autograd.grad(weights, detached, grad)[0]

        for input_grad, weight_grad in ((True, True), (True, False), (False, True)):
            for use_logits, use_weights in ((True, True), (True, False), (False, True)):
                with self.subTest(input_grad=input_grad, weight_grad=weight_grad,
                                  use_logits=use_logits, use_weights=use_weights):
                    hidden = torch.randn(3, 64, requires_grad=input_grad)
                    weight = torch.randn(12, 64, requires_grad=weight_grad)
                    bias = torch.randn(12)
                    with patch.object(projection, "_project", side_effect=_project), \
                            patch.object(projection, "_sigmoid_plan_for_device", return_value=None), \
                            patch.object(projection, "_projection_mm", side_effect=torch.mm), \
                            patch.object(projection, "_sigmoid_route_backward", side_effect=_route_backward):
                        actual = projection._ProjectionRoute.apply(hidden, weight, bias)
                        expected = _project(hidden, weight, bias)
                        coefficient = torch.randn(3, 8)
                        def _loss(outputs):
                            return ((outputs[0].square().sum() if use_logits else 0) +
                                    ((outputs[1] * coefficient).sum() if use_weights else 0))
                        inputs = tuple(tensor for tensor in (hidden, weight) if tensor.requires_grad)
                        gradients = torch.autograd.grad(_loss(actual), inputs)
                        reference = torch.autograd.grad(_loss(expected), inputs)
                    for value, golden in zip(gradients, reference):
                        torch.testing.assert_close(value, golden)

    def test_repeated_tile_profile_capacity(self) -> None:
        """Tail and large T must reserve capacity for every repeated task and dependency wait."""
        with patch.object(torch, "npu", create=True) as npu:
            npu.get_device_limit.return_value = {"vector_core_num": 64}
            runtime = gate_plan._build_projection_runtime(torch.device("cpu"))
            for tokens in (1, 31, 32, 33, 4096, 12288):
                with self.subTest(tokens=tokens):
                    prepared = _TileProfileRuntime(runtime, tokens)
                    per_core = ((tokens + 31) // 32 + 31) // 32
                    self.assertGreaterEqual(prepared._layout.aic_record_capacity, per_core)
                    self.assertGreaterEqual(prepared._layout.aiv_record_capacity, 11 * per_core)
                    self.assertEqual(prepared.buffer_size,
                                     32 * (64 + 32 * prepared._layout.aic_record_capacity) +
                                     64 * (64 + 32 * prepared._layout.aiv_record_capacity))

    def test_index_free_plan_resolves_current_device_once(self) -> None:
        """Use one concrete device for both hardware metadata and every allocation."""
        requested = SimpleNamespace(type="npu", index=None)
        resolved = SimpleNamespace(type="npu", index=3)
        with patch.object(gate_plan.torch, "npu", create=True) as npu, \
                patch.object(gate_plan.torch, "device", return_value=resolved) as device, \
                patch.object(gate_plan.torch, "zeros", return_value=object()) as zeros, \
                patch.object(gate_plan, "prepare_mega_kernel_diagnostics", return_value=MagicMock()) as prepare:
            npu.current_device.return_value = 3
            npu.get_device_limit.return_value = {"vector_core_num": 64}
            gate_plan.build_mega_gate_plan(requested)
            device.assert_called_once_with("npu", 3)
            npu.get_device_limit.assert_called_once_with(3)
            npu.current_stream.assert_called_once_with(resolved)
            self.assertIs(zeros.call_args.kwargs["device"], resolved)
            self.assertTrue(all(call.kwargs["device"] is resolved for call in prepare.call_args_list))
