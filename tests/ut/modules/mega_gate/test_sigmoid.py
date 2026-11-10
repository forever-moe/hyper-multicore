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
"""Boundary validation for the logits-only sigmoid route API."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from hyper_multicore.scheduler.runtime import serialize_runtime_config
from hyper_multicore.modules.mega_gate import sigmoid_route_from_logits
from hyper_multicore.modules.mega_gate import sigmoid
from hyper_multicore.modules.mega_gate import plan as gate_plan
from hyper_multicore.modules.mega_gate.diagnostics import bind_pipeline_graph
from hyper_multicore.modules.mega_gate.plan import (
    MEGA_GATE_ROUTE_TASK_TYPES, MEGA_GATE_ROUTE_STAGE_NAMES,
    MEGA_GATE_ROUTE_GRAD_K1_TASK_TYPES, MEGA_GATE_ROUTE_GRAD_K1_STAGE_NAMES,
    _build_pipeline_runtime_config,
)


class _TensorSubclass(torch.Tensor):
    """Represent a dispatching tensor outside the native local-tensor contract."""


class TestZeroBiasCache(unittest.TestCase):
    """Only publish complete immutable constants, including concurrent first use."""

    def test_reuse_without_repeated_initialization(self) -> None:
        """Cache hits reuse the constant without scheduling or synchronizing work."""
        value = torch.zeros(8)
        with patch.dict(sigmoid._ZERO_BIAS_CACHE, clear=True), \
                patch.object(sigmoid.torch, "zeros", return_value=value) as allocate, \
                patch.object(sigmoid.torch, "npu", create=True) as npu:
            first = sigmoid._zero_bias_for_device(torch.device("cpu"), 8)
            second = sigmoid._zero_bias_for_device(torch.device("cpu"), 8)
            self.assertIs(first, value)
            self.assertIs(second, first)
            allocate.assert_called_once_with(8, dtype=torch.float32, device=torch.device("cpu"))
            npu.current_stream.assert_called_once_with(torch.device("cpu"))
            npu.current_stream.return_value.synchronize.assert_called_once()

    def test_device_and_width_have_independent_constants(self) -> None:
        """Input widths and devices cannot borrow incompatible constant storage."""
        values = [torch.zeros(8), torch.zeros(7), torch.zeros(8)]
        with patch.dict(sigmoid._ZERO_BIAS_CACHE, clear=True), \
                patch.object(sigmoid.torch, "zeros", side_effect=values), \
                patch.object(sigmoid.torch, "npu", create=True):
            actual = [sigmoid._zero_bias_for_device(device, experts) for device, experts in
                      ((torch.device("cpu"), 8), (torch.device("cpu"), 7), (torch.device("meta"), 8))]
            for result, expected in zip(actual, values):
                self.assertIs(result, expected)

    def test_failed_initialization_is_not_published(self) -> None:
        """A failed producer synchronization must leave no reusable incomplete tensor."""
        with patch.dict(sigmoid._ZERO_BIAS_CACHE, clear=True), \
                patch.object(sigmoid.torch, "zeros", return_value=torch.zeros(8)), \
                patch.object(sigmoid.torch, "npu", create=True) as npu:
            npu.current_stream.return_value.synchronize.side_effect = RuntimeError("initialization failed")
            with self.assertRaisesRegex(RuntimeError, "initialization failed"):
                sigmoid._zero_bias_for_device(torch.device("cpu"), 8)
            self.assertEqual(sigmoid._ZERO_BIAS_CACHE, {})

    def test_concurrent_first_use_initializes_once(self) -> None:
        """Concurrent callers receive one fully initialized constant."""
        value = torch.zeros(8)
        with patch.dict(sigmoid._ZERO_BIAS_CACHE, clear=True), \
                patch.object(sigmoid.torch, "zeros", return_value=value) as allocate, \
                patch.object(sigmoid.torch, "npu", create=True) as npu, \
                ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: sigmoid._zero_bias_for_device(torch.device("cpu"), 8), range(16)))
            self.assertTrue(all(result is value for result in results))
            allocate.assert_called_once()
            npu.current_stream.return_value.synchronize.assert_called_once()


class TestSigmoidRoute(unittest.TestCase):
    """Reject unsupported inputs before initializing native resources."""

    def test_cpu_requires_explicit_npu_input(self) -> None:
        """A low-level native entry must never silently execute Torch instead."""
        with self.assertRaisesRegex(ValueError, "NPU"):
            sigmoid_route_from_logits(torch.randn(3, 8), top_k=2)

    def test_invalid_top_k(self) -> None:
        """Invalid selections fail at the API boundary."""
        for top_k in (0, -1, 9):
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(ValueError, "top_k"):
                    sigmoid_route_from_logits(torch.randn(3, 8), top_k=top_k)

    def test_invalid_shape(self) -> None:
        """Only nonempty matrices form a supported token-row shard."""
        for shape in ((8,), (0, 8), (3, 0)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "shape"):
                    sigmoid_route_from_logits(torch.empty(shape), top_k=1)

    def test_expert_capacity(self) -> None:
        """The native entry bounds the expert count before allocating its plan."""
        with self.assertRaisesRegex(ValueError, "384"):
            sigmoid_route_from_logits(torch.empty(3, 385), top_k=1)

    def test_tensor_subclasses_require_their_original_dispatch(self) -> None:
        """Distributed tensor dispatch must not enter an unregistered native op."""
        with self.assertRaisesRegex(ValueError, "local Torch tensor"):
            sigmoid_route_from_logits(torch.empty(3, 8).as_subclass(_TensorSubclass), top_k=1)

    def test_legacy_plan_is_rejected_before_native_launch(self) -> None:
        """Legacy row ownership cannot describe the sigmoid forward worker count."""
        with self.assertRaisesRegex(ValueError, "use_sigmoid=True"):
            sigmoid_route_from_logits(torch.empty(65, 64), top_k=8, plan=SimpleNamespace(use_sigmoid=False))

    def test_uncalibrated_normalized_width_is_rejected(self) -> None:
        """A native caller must not silently accept widths with different CANN reduction trees."""
        for top_k in (9, 15, 17, 127, 129, 225):
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(ValueError, "not calibrated"):
                    sigmoid_route_from_logits(torch.empty(65, 384), top_k=top_k)


class TestSigmoidAutograd(unittest.TestCase):
    """Check saved state and selection of the native backward schedule on CPU."""

    def test_backward_schedule_and_bias_ownership(self) -> None:
        """No-norm and K=1 use the direct schedule, with no correction-bias gradient."""
        for top_k, normalize, use_bias in (
            (1, True, False), (1, True, True), (3, False, False),
            (3, False, True), (3, True, False), (3, True, True),
        ):
            with self.subTest(top_k=top_k, normalize=normalize, use_bias=use_bias):
                logits = torch.randn(2, 8, requires_grad=True)
                bias = torch.randn(8, requires_grad=True)
                ids = torch.zeros(2, top_k, dtype=torch.int64)
                scores = torch.rand_like(logits)
                selected = torch.rand(2, top_k)
                denominator = torch.ones(2, 1)
                plan = SimpleNamespace(
                    route_runtime=SimpleNamespace(is_active=False, runtime_config=torch.tensor([1])),
                    route_grad_runtime=SimpleNamespace(is_active=False, runtime_config=torch.tensor([2])),
                    route_grad_k1_runtime=SimpleNamespace(is_active=False, runtime_config=torch.tensor([3])),
                    vision_mask_placeholder=torch.zeros(1, dtype=torch.bool),
                )
                ops = MagicMock()
                ops._mega_gate_sigmoid_route.return_value = (
                    selected.clone(), ids, scores, selected, denominator,
                )
                expected = torch.randn_like(logits)
                ops._mega_gate_sigmoid_route_grad.return_value = expected
                with patch.object(sigmoid, "_native_ops", return_value=ops):
                    weights, output_ids = sigmoid._SigmoidRoute.apply(
                        logits, bias, plan, top_k, normalize, use_bias,
                    )
                    grad, bias_grad = torch.autograd.grad(weights.sum(), (logits, bias), allow_unused=True)
                self.assertIs(grad, expected)
                self.assertIsNone(bias_grad)
                self.assertFalse(output_ids.requires_grad)
                self.assertIs(ops._mega_gate_sigmoid_route.call_args.kwargs["use_bias"], use_bias)
                args = ops._mega_gate_sigmoid_route_grad.call_args.args
                self.assertIs(args[2], selected)
                self.assertTrue(args[5].is_contiguous())
                self.assertIs(args[6], plan.route_grad_runtime.runtime_config if top_k > 1 and normalize
                              else plan.route_grad_k1_runtime.runtime_config)


class TestSigmoidProfilingGraph(unittest.TestCase):
    """The profile must describe the executed sigmoid stages and operand positions."""

    def test_forward_uses_sigmoid_names_and_score_writer(self) -> None:
        """Keep the existing ABI while assigning scores to the Sigmoid producer."""
        config = _build_pipeline_runtime_config(
            MEGA_GATE_ROUTE_TASK_TYPES, MEGA_GATE_ROUTE_STAGE_NAMES, kernel_name="HyperMegaGateRoute",
        )
        graph = bind_pipeline_graph(None, config, use_sigmoid=True)
        nodes = graph.topological_sort()
        self.assertEqual([node.name for node in nodes], [
            "Sigmoid", "SqrtBypass", "AddBias", "TopK", "Gather", "ReduceSum", "AddEpsilon", "Div",
            "CopyWeights", "CastIndex",
        ])
        self.assertEqual([(port.name, port.param_position) for port in nodes[0].outputs], [("route_scores", 8)])
        self.assertEqual(nodes[1].outputs, [])

    def test_backward_has_a_zero_task_sigmoid_tail(self) -> None:
        """The CANN post-processing tail is outside the measured MegaKernel stages."""
        config = _build_pipeline_runtime_config(
            MEGA_GATE_ROUTE_GRAD_K1_TASK_TYPES, MEGA_GATE_ROUTE_GRAD_K1_STAGE_NAMES,
            kernel_name="HyperMegaGateRouteGrad",
        )
        graph = bind_pipeline_graph(None, config, backward=True, use_sigmoid=True)
        nodes = graph.topological_sort()
        self.assertEqual(nodes[-1].name, "SigmoidGradPost")
        self.assertEqual(nodes[-1].task_num, 0)
        self.assertEqual(nodes[-1].outputs[0].param_position, 9)


class TestSigmoidWorkerCapacity(unittest.TestCase):
    """A5 runtime storage and row ownership must cover every selected AIV."""

    def test_full_a5_capacity_survives_serialization(self) -> None:
        """Dynamic runtime storage must reserve 36 Cube slots for 72 AIVs."""
        config = _build_pipeline_runtime_config(
            MEGA_GATE_ROUTE_TASK_TYPES, MEGA_GATE_ROUTE_STAGE_NAMES,
            kernel_name="HyperMegaGateRoute", worker_capacity=72,
        )
        self.assertEqual(config.num_workers, 72)
        self.assertTrue(all(config.all_tasks[i].task_split_num == 72 for i in range(config.task_num)))
        image = serialize_runtime_config(config)
        self.assertGreater(len(image), 0)


class TestMegaGateDeviceCapacity(unittest.TestCase):
    """All forward and backward runtime images must match the device capacity."""

    def test_device_capacity_applies_to_every_pipeline(self) -> None:
        """A5 and 910 device plans use the same queried row-worker capacity."""
        for workers, use_sigmoid in ((48, False), (64, False), (64, True), (72, True)):
            with self.subTest(workers=workers, use_sigmoid=use_sigmoid):
                prepared = []

                def _prepare(graph, runtime_config, **options):
                    del graph, options
                    prepared.append(runtime_config)
                    return SimpleNamespace(runtime_config=None)

                with patch.object(gate_plan.torch, "npu", create=True) as npu, \
                        patch.object(gate_plan, "prepare_mega_kernel_diagnostics", side_effect=_prepare), \
                        patch.object(gate_plan.torch, "zeros", return_value=torch.empty(1)), \
                        patch.object(gate_plan.dist, "is_initialized", return_value=False):
                    npu.get_device_limit.return_value = {"vector_core_num": workers}
                    gate_plan.build_mega_gate_plan(SimpleNamespace(type="npu", index=3), use_sigmoid=use_sigmoid)
                self.assertEqual([config.num_workers for config in prepared], [workers] * 3)
                for config in prepared:
                    self.assertTrue(all(config.all_tasks[i].task_split_num == workers
                                        for i in range(config.task_num)))
                    self.assertGreater(len(serialize_runtime_config(config)), 0)
