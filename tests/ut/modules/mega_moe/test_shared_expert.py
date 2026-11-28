# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Verify local shared-expert task edges and the public weight/gradient contract."""

import struct
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from hyper_multicore import MegaMoeExperts, MegaMoeNumericsOptions
from hyper_multicore.modules.mega_moe import invocation
from hyper_multicore.modules.mega_moe.backends.mxfp8 import backend, execution, shared_expert
from hyper_multicore.modules.mega_moe.backends.mxfp8.autograd import execute_mxfp8_experts
from hyper_multicore.modules.mega_moe.backends.mxfp8.graph import build_mxfp8_runtime
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec, SHARED_EXPERT_EVENTS
from hyper_multicore.scheduler.config import ATOMIC_ADD_VALUE_LEN, TaskType, event_workspace_bytes
from hyper_multicore.scheduler.runtime import serialize_runtime_config
from hyper_multicore.tasks.qbmm_mxfp8 import qbmm_tiling


@pytest.fixture
def dense_tiler(monkeypatch):
    calls = []

    def tiling(layout, m, n, k, output_bytes, cubes, *, device):
        calls.append((layout, m, n, k, output_bytes))
        return 0x20010, cubes // 2, struct.pack("<3I52x", m, n, k)

    monkeypatch.setattr(shared_expert, "qbmm_tiling", tiling)
    return calls


@pytest.mark.parametrize(
    "layout,dims,output_bytes,key,shapes",
    [
        ("NT", (4096, 2560, 3584), 2, 0x20914, ((4096, 3584), (2560, 3584), (2560, 56, 2), (4096, 56, 2))),
        ("NN", (4096, 1280, 3584), 2, 0x20910, ((4096, 3584), (3584, 1280), (56, 1280, 2), (4096, 56, 2))),
        ("TN", (3584, 1280, 4096), 4, 0x20011, ((4096, 3584), (4096, 1280), (64, 1280, 2), (64, 3584, 2))),
    ],
)
def test_qbmm_tiler_consumes_the_native_storage_shapes(monkeypatch, layout, dims, output_bytes, key, shapes):
    active_device = torch.device("npu", 0)
    target_device = torch.device("npu", 1)

    @contextmanager
    def use_device(device):
        nonlocal active_device
        previous, active_device = active_device, device
        try:
            yield
        finally:
            active_device = previous

    def cann_tiler(op, compile_info, inputs, outputs, *, attrs):
        del compile_info
        assert active_device == target_device
        assert op == "QuantBatchMatmulV3"
        assert tuple(desc["shape"] for desc in inputs if desc is not None) == shapes
        assert outputs[0]["dtype"] == ("float32" if output_bytes == 4 else "bfloat16")
        params = {attr["name"]: attr["value"] for attr in attrs}
        assert params["transpose_x1"] == (layout[0] == "T")
        assert params["transpose_x2"] == (layout[1] == "T")
        assert params["group_size"] == 4295032864
        window_len = 4 if target_device.index == 1 else 0
        data = struct.pack("<3I44xI4x", *dims, window_len)
        return {"tiling_key": key, "block_dim": 32, "tiling_data": data}

    monkeypatch.setitem(sys.modules, "tbe.common.utils.op_tiling", SimpleNamespace(do_op_tiling=cann_tiler))
    monkeypatch.setattr(torch.npu, "device", use_device)
    qbmm_tiling.cache_clear()
    try:
        newer = qbmm_tiling(layout, *dims, output_bytes, 32, device=target_device)
        assert newer[:2] == (key, 32)
        assert active_device.index == 0
        target_device = torch.device("npu", 2)
        older = qbmm_tiling(layout, *dims, output_bytes, 32, device=target_device)
        assert older[:2] == (key, 32)
        assert newer[2] != older[2]
        assert active_device.index == 0
    finally:
        qbmm_tiling.cache_clear()


@pytest.mark.parametrize("window_fields", [(8, 0, 0), (4, 128, 128)])
def test_qbmm_rejects_tiling_the_worker_cannot_execute(monkeypatch, window_fields):
    dims = (4096, 2560, 3584)
    data = struct.pack("<3I44xIHH", *dims, *window_fields)
    cann_tiler = Mock(return_value={"tiling_key": 0x20914, "block_dim": 32, "tiling_data": data})
    monkeypatch.setitem(sys.modules, "tbe.common.utils.op_tiling", SimpleNamespace(do_op_tiling=cann_tiler))
    monkeypatch.setattr(torch.npu, "device", nullcontext)
    qbmm_tiling.cache_clear()
    with pytest.raises(ValueError, match=r"QBMM tiling.*key=0x20914, windowLen=\d+, mL0=\d+, nL0=\d+"):
        qbmm_tiling("NT", *dims, 2, 32, device=torch.device("npu", 1))


@pytest.mark.parametrize("failure", [ValueError("Incompatible QBMM tiling"), OSError("Unable to load CANN tiler")])
def test_shared_tiling_failure_keeps_the_rank_local_cause(caplog, failure):
    hidden = torch.empty((2, 64), dtype=torch.bfloat16)
    resources = SimpleNamespace(resources=object(), specification={"partial_weight_grad_dtype": "float32"})
    backend_stub = SimpleNamespace(validate_shared_weights=Mock(), validate_shared_tiling=Mock(side_effect=failure))
    module = SimpleNamespace(shared_intermediate_size=128, ep_size=2, hidden_size=64,
                             _resource_group=resources, _backend=backend_stub,
                             _validate_forward_inputs=Mock(return_value=hidden))

    _, weight_error = MegaMoeExperts._prepare_forward_inputs(module, hidden, None, None, None, (), ())

    assert weight_error
    assert caplog.records[-1].exc_info[1] is failure


@pytest.mark.parametrize('ep, experts', [(1, 144), (4, 12)])
@pytest.mark.parametrize('forward', [True, False])
def test_shared_graph_waits_for_routed_math_and_preserves_combine(ep, experts, forward, dense_tiler):
    spec = MegaMoeSpec(128, 512, 256, experts, 8, None, 1024, ep, None, 0, 32,
                       dispatch_mode='pull', precision='mxfp8', architecture=3510,
                       shared_intermediate_size=128, partial_weight_grad_dtype='float32')
    program = build_mxfp8_runtime(spec, forward=forward, shared_rows=65)
    routed = build_mxfp8_runtime(replace(spec, shared_intermediate_size=None), forward=forward)
    cfg = program.config
    assert cfg.event_capacity * 4 + (2 * (ep + 1) * 64 if ep > 1 else 0) == event_workspace_bytes(
        ep, experts, compute_stages=6, local_events=SHARED_EXPERT_EVENTS,
    )
    assert serialize_runtime_config(cfg)
    assert program.buffer_count <= 64
    assert program.gmm_descriptors == routed.gmm_descriptors
    assert all(bytes(cfg.all_tasks[i]) == bytes(routed.config.all_tasks[i])
               for i in range(routed.config.task_num))
    offset = 0
    stages = {}
    for op in program.graph.topological_sort():
        stages[op.name] = list(cfg.all_tasks[offset:offset + op.task_num])
        offset += op.task_num
    if forward:
        assert dense_tiler == [('NT', 65, 256, 512, 2), ('NT', 65, 512, 128, 2)]
        dense_stages = ('shared_up', 'shared_down')
    else:
        assert dense_tiler == [('TN', 512, 128, 65, 4), ('NN', 65, 128, 512, 2),
                               ('TN', 256, 512, 65, 4), ('NN', 65, 512, 256, 2)]
        dense_stages = ('shared_dw2', 'shared_da', 'shared_dw1', 'shared_dx')
    for index, name in enumerate(dense_stages):
        tasks = stages[name]
        assert len(tasks) == spec.num_cube_cores // 2
        assert all(task.task_type == TaskType.TASK_MX_QBMM and task.extra_value_0 == index for task in tasks)
        assert cfg.all_event_num_triggers[tasks[0].trigger_event] == len(tasks)
    assert len(program.qbmm_descriptors) == 88 * len(dense_stages)
    last_routed = stages['down' if forward else 'dx']
    join = stages['shared_ready']
    assert {task.dependent_event for task in join} == {task.trigger_event for task in last_routed}
    event = join[0].trigger_event
    assert event >= cfg.completion_event + ATOMIC_ADD_VALUE_LEN
    assert cfg.all_event_num_triggers[event] == len(join) == spec.local_experts
    shared_ids = set(range(routed.config.task_num, cfg.task_num))
    for worker, queue in enumerate(cfg.worker_queues):
        kind = 0 if worker < spec.num_cube_cores else 1
        indices = cfg.cube_task_indices if kind == 0 else cfg.vector_task_indices
        tasks = list(indices[queue.begin:queue.end])
        shared_tasks = shared_ids.intersection(tasks)
        if shared_tasks:
            assert kind == 0 or (worker - spec.num_cube_cores) % 2 == 0
            assert set(tasks[-len(shared_tasks):]) == shared_tasks


def test_shared_weight_gradients_participate_in_existing_route_agreement():
    """A rank with only a shared gradient must still participate in backward."""
    weights = (torch.ones(1), torch.ones(1))
    shared = (torch.ones(1, requires_grad=True), torch.ones(1))
    resources = SimpleNamespace(spec=SimpleNamespace(dispatch_mode='push', auto_tokens=True),
                                workspace=Mock(), placement=None)
    with patch.object(invocation, 'prepare_topk_route', return_value=object()) as route:
        with invocation.prepare_expert_invocation(
            weights[0], weights[0], weights[0], None, weights, resources, shared_expert_weights=shared,
        ):
            assert route.call_args.kwargs['autograd_mask'] == (1 | 1 << 6)


def _root(shape):
    root = torch.nn.Parameter(torch.zeros(shape, dtype=torch.bfloat16))
    m, n = shape[-2:]
    root.row_data = torch.empty(shape, dtype=torch.float8_e4m3fn)
    root.col_data = torch.empty(shape, dtype=torch.float8_e4m3fn)
    root.row_scale = torch.empty((*shape[:-2], m, n // 64, 2), dtype=torch.uint8)
    root.col_scale = torch.empty((*shape[:-2], m // 64, n, 2), dtype=torch.uint8)
    return root


@pytest.mark.parametrize('order', ['routed_first', 'shared_first'])
@pytest.mark.parametrize('rows', [0, 3])
def test_shared_forward_and_backward_bind_current_weights_and_actual_rows(order, rows, monkeypatch, dense_tiler):
    spec = MegaMoeSpec(128, 64, 64, 1, 1, None, 128, 1, None, 0, 32,
                       dispatch_mode='pull', precision='mxfp8', architecture=3510,
                       shared_intermediate_size=128, shared_expert_add_order=order, partial_weight_grad_dtype='float32')
    forward = backend._device_runtime(build_mxfp8_runtime(spec, forward=True, shared_rows=rows), 'cpu', spec)
    backward = backend._device_runtime(build_mxfp8_runtime(spec, forward=False, shared_rows=rows), 'cpu', spec)
    plan = SimpleNamespace(spec=spec, forward=forward, backward=backward)
    workspace = Mock(source_buffer=torch.zeros((rows, 64), dtype=torch.bfloat16),
                     routed_buffer=torch.zeros((rows, 64), dtype=torch.bfloat16))
    resources = SimpleNamespace(spec=spec, workspace=workspace, plan_for_capacity=Mock(return_value=plan))
    offsets = torch.zeros(1, dtype=torch.long)
    metadata = SimpleNamespace(expert_capacity=max(1, rows), plan_tokens=128, group_list=torch.tensor([rows]),
                               dispatch_target_off=offsets, dispatch_src_off=offsets, dispatch_size=offsets,
                               combine_target_off=offsets, combine_src_off=offsets, combine_size=offsets)
    route = SimpleNamespace(metadata=metadata, routed_tokens=workspace.source_buffer,
                            unpermute_mapping=torch.arange(rows, dtype=torch.int32))
    h = torch.full((rows, 64), 1, dtype=torch.bfloat16, requires_grad=True)
    probs = torch.ones(rows, 1, requires_grad=True)
    shapes = ((1, 64, 128), (1, 64, 64))
    routed = tuple(_root(shape) for shape in shapes)
    shared = _root((256, 64)), _root((64, 128))
    fresh_col_data = [torch.zeros_like(root.col_data) for root in shared]

    def launch(buffers, runtime, events, plan, direction='forward'):
        del events, plan
        names = runtime.host.shared_buffer_names
        assert len(buffers) == runtime.host.buffer_count
        binds = dict(zip(names, buffers[-len(names):]))
        assert binds['group'].tolist() == [rows]
        assert binds['qbmm'] is runtime.qbmm
        axis = 'row' if direction == 'forward' else 'col'
        assert binds['w1'].data_ptr() == getattr(shared[0], f'{axis}_data').data_ptr()
        assert binds['w2_scale'].data_ptr() == getattr(shared[1], f'{axis}_scale').data_ptr()
        if direction == 'forward':
            assert binds['x'].data_ptr() == h.data_ptr()
            binds['output'].fill_(5)
            workspace.routed_buffer.fill_(2)
        else:
            assert torch.equal(binds['dy'], torch.full_like(h, 11))
            assert binds['dw1'].dtype == binds['dw2'].dtype == torch.float32
            binds['dx'].fill_(3)
            binds['dw1'].fill_(6)
            binds['dw2'].fill_(7)
            buffers[6].fill_(8)
            buffers[18].fill_(9)
            workspace.routed_buffer.fill_(4)

    runner = execution.MXFP8Execution(resources)
    monkeypatch.setattr(runner, '_launch', launch)
    monkeypatch.setattr(execution.torch_npu, 'npu_moe_token_unpermute',
                        lambda output, mapping, probs: output.clone())
    monkeypatch.setattr(execution.multicore_ops, 'mega_moe_unpermute_grad_out',
                        lambda output, dy, mapping, probs, source, dp: dp.fill_(10))
    monkeypatch.setattr(execution.multicore_ops, 'moe_token_permute_grad',
                        lambda output, mapping, rows, top_k: output.clone())

    y = execute_mxfp8_experts(h, *routed, route, probs, shapes, runner, shared_expert_weights=shared)
    torch.testing.assert_close(y, torch.full_like(h, 7))
    for root, col_data in zip(shared, fresh_col_data):
        root.row_data = root.row_scale = None
        root.col_data = col_data
    grads = torch.autograd.grad(y, (h, probs, *routed, *shared), grad_outputs=torch.full_like(h, 11))
    for grad, value in zip(grads, (7, 10, 9, 8, 6, 7)):
        torch.testing.assert_close(grad, torch.full_like(grad, value))
    assert workspace.track_graph.call_count == 1
    assert workspace.release_graph.call_count == 1
    assert resources.plan_for_capacity.call_args.kwargs["shared_rows"] == rows


def test_shared_expert_public_configuration_and_weight_shapes():
    sizes = dict(hidden_size=64, intermediate_size=128, num_experts=1, top_k=1)
    with pytest.raises(ValueError, match='requires MXFP8'):
        MegaMoeExperts(**sizes, shared_intermediate_size=64)
    with pytest.raises(ValueError, match='positive multiple of 64'):
        MegaMoeExperts(**sizes, precision='mxfp8', shared_intermediate_size=63)
    with pytest.raises(ValueError, match='shared_expert_add_order'):
        MegaMoeExperts(**sizes, numerics_options=MegaMoeNumericsOptions(shared_expert_add_order='invalid'))
    layer = MegaMoeExperts(**sizes, precision='mxfp8', shared_intermediate_size=256)
    try:
        hidden = torch.empty(1, 64, dtype=torch.bfloat16)
        backend.validate_shared_weights(layer, (_root((512, 64)), _root((64, 256))), hidden)
        with pytest.raises(ValueError, match='Dense roots'):
            backend.validate_shared_weights(layer, (_root((64, 512)), _root((64, 256))), hidden)
    finally:
        layer.close()
