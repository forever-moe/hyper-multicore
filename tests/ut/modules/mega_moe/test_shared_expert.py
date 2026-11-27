# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Verify local shared-expert task edges and the public weight/gradient contract."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from hyper_multicore import MegaMoeExperts, MegaMoeNumericsOptions
from hyper_multicore.modules.mega_moe import invocation
from hyper_multicore.modules.mega_moe.backends.mxfp8 import backend, execution
from hyper_multicore.modules.mega_moe.backends.mxfp8.autograd import execute_mxfp8_experts
from hyper_multicore.modules.mega_moe.backends.mxfp8.graph import build_mxfp8_runtime
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec, SHARED_EXPERT_EVENTS
from hyper_multicore.scheduler.config import ATOMIC_ADD_VALUE_LEN, event_workspace_bytes
from hyper_multicore.scheduler.runtime import serialize_runtime_config


@pytest.mark.parametrize('ep, experts', [(1, 144), (4, 12)])
@pytest.mark.parametrize('forward', [True, False])
def test_shared_graph_waits_for_routed_math_and_preserves_combine(ep, experts, forward):
    spec = MegaMoeSpec(128, 512, 256, experts, 8, None, 1024, ep, None, 0, 32,
                       dispatch_mode='pull', precision='mxfp8', architecture=3510, shared_intermediate_size=128)
    program = build_mxfp8_runtime(spec, forward=forward)
    routed = build_mxfp8_runtime(replace(spec, shared_intermediate_size=None), forward=forward)
    cfg = program.config
    assert cfg.event_capacity * 4 + (2 * (ep + 1) * 64 if ep > 1 else 0) == event_workspace_bytes(
        ep, experts, compute_stages=6, local_events=SHARED_EXPERT_EVENTS,
    )
    assert serialize_runtime_config(cfg)
    assert program.buffer_count <= 64
    assert all(bytes(cfg.all_tasks[i]) == bytes(routed.config.all_tasks[i])
               for i in range(routed.config.task_num))
    offset = 0
    stages = {}
    for op in program.graph.topological_sort():
        stages[op.name] = list(cfg.all_tasks[offset:offset + op.task_num])
        offset += op.task_num
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
def test_shared_forward_and_backward_bind_current_weights_and_actual_rows(order, rows, monkeypatch):
    spec = MegaMoeSpec(128, 64, 64, 1, 1, None, 128, 1, None, 0, 32,
                       dispatch_mode='pull', precision='mxfp8', architecture=3510,
                       shared_intermediate_size=128, shared_expert_add_order=order, partial_weight_grad_dtype='float32')
    forward = backend._device_runtime(build_mxfp8_runtime(spec, forward=True), 'cpu', spec)
    backward = backend._device_runtime(build_mxfp8_runtime(spec, forward=False), 'cpu', spec)
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
