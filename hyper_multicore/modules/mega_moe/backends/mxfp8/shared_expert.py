# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Lower the local dense expert after routed computation on the same workers."""

from dataclasses import replace

from hyper_multicore.scheduler.config import ATOMIC_ADD_VALUE_LEN, TaskAiCoreType, TaskDescC, TaskType
from hyper_multicore.scheduler.graph import OperatorNode, OpType, SplitSpec, TensorSpec
from hyper_multicore.tasks.gmm_mxfp8 import Mxfp8GmmFillConfig
from hyper_multicore.tasks.quantization_mxfp8 import (
    MX_QUANT_TASK, MX_SWIGLU_GRAD_TASK, MX_SWIGLU_TASK, Mxfp8QuantFillConfig,
)
from hyper_multicore.tasks.swiglu_mxfp8 import Mxfp8SwiGLUFillConfig
from hyper_multicore.modules.mega_moe.spec import SHARED_EXPERT_EVENTS


def add_shared_expert(graph, spec, forward, specs, gmm_table, quant_table, gmms, quants, clamp_bits):
    """Append one local group using the existing MXFP8 compute operand contracts."""
    h, width, rows, cubes = spec.hidden_size, spec.shared_intermediate_size, spec.local_num_tokens, spec.num_cube_cores
    buffers = {}

    def buffer(name, shape, dtype_size=2):
        operand = TensorSpec(f'shared_{name}', list(shape), len(specs) + len(buffers), dtype_size=dtype_size)
        buffers[name] = operand
        return operand

    def quantized(name, columns):
        return (
            buffer(f'{name}_row', (rows, columns), 1),
            buffer(f'{name}_row_scale', (rows, columns // 64, 2), 1),
            buffer(f'{name}_col', (rows, columns), 1),
            buffer(f'{name}_col_scale', ((rows + 63) // 64 + 1, columns, 2), 1),
        )

    def saved_columns(name, columns):
        return (buffer(f'{name}_col', (rows, columns), 1),
                buffer(f'{name}_col_scale', ((rows + 63) // 64 + 1, columns, 2), 1))

    ready = OperatorNode('shared_ready', OpType.BARRIER, [], [], [], 1,
                         SplitSpec(None, lambda values: values.single_rank_expert_num), -1, None)
    graph.add_op(ready)
    graph.add_edge('down' if forward else 'dx', ready)
    ops = [ready]
    group = buffer('group', (1,), 8)

    def stage(name, kind, inputs, outputs, fill):
        op = OperatorNode(f'shared_{name}', kind, list(inputs), list(outputs),
                          [operand.param_position for operand in (*inputs, *outputs)],
                          1, SplitSpec(None, lambda _: cubes), -1, fill)
        graph.add_op(op)
        graph.add_edge(ops[-1], op)
        ops.append(op)

    def quant(name, kind, columns, inputs, outputs):
        fill = (Mxfp8QuantFillConfig(kind, columns, cubes, False, quants) if kind == MX_QUANT_TASK
                else Mxfp8SwiGLUFillConfig(kind, columns, cubes, False, quants, clamp_bits))
        op_type = {MX_QUANT_TASK: OpType.QUANTIZE, MX_SWIGLU_TASK: OpType.SWIGLU_QUANT,
                   MX_SWIGLU_GRAD_TASK: OpType.SWIGLU_GRAD_QUANT}[kind]
        stage(name, op_type, (*inputs, group, quant_table), outputs, fill)

    def gmm(name, layout, dims, lhs, rhs, out):
        x, x_scale = lhs
        w, w_scale = rhs
        fill = Mxfp8GmmFillConfig(layout, *dims, x_scale.param_position, w_scale.param_position, cubes, gmms)
        stage(name, OpType.GMM, (x, w, group, gmm_table), (out,), fill)

    def weight(name, out_channels, in_channels):
        shape = (out_channels, in_channels)
        scale = ((out_channels, in_channels // 64, 2) if forward
                 else (out_channels // 64, in_channels, 2))
        return buffer(name, shape, 1), buffer(f'{name}_scale', scale, 1)

    w1, w2 = weight('w1', 2 * width, h), weight('w2', h, width)
    gate_up = buffer('gate_up', (rows, 2 * width))
    if forward:
        x, activation = quantized('x', h), quantized('activation', width)
        source = buffer('x', (rows, h))
        output = buffer('output', (rows, h))
        quant('quant_x', MX_QUANT_TASK, h, (source,), x)
        gmm('up', 'NT', (rows, 2 * width, h), x[:2], w1, gate_up)
        quant('swiglu_quant', MX_SWIGLU_TASK, width, (gate_up,), activation)
        gmm('down', 'NT', (rows, h, width), activation[:2], w2, output)
    else:
        dy, du = quantized('dy', h), quantized('du', 2 * width)
        x_col = saved_columns('x', h)
        a_col = saved_columns('activation', width)
        source = buffer('dy', (rows, h))
        da = buffer('da', (rows, width))
        dx = buffer('dx', (rows, h))
        grad_bytes = 4 if spec.partial_weight_grad_dtype == 'float32' else 2
        dw1 = buffer('dw1', (2 * width, h), grad_bytes)
        dw2 = buffer('dw2', (h, width), grad_bytes)
        quant('quant_dy', MX_QUANT_TASK, h, (source,), dy)
        gmm('dw2', 'TN', (h, width, rows), dy[2:], a_col, dw2)
        gmm('da', 'NN', (rows, width, h), dy[:2], w2, da)
        quant('swiglu_grad_quant', MX_SWIGLU_GRAD_TASK, 2 * width, (gate_up, da), du)
        gmm('dw1', 'TN', (2 * width, h, rows), du[2:], x_col, dw1)
        gmm('dx', 'NN', (rows, h, 2 * width), du[:2], w1, dx)
    return ops, tuple(buffers)


def fill_shared_expert(cfg, values, ops, routed_events):
    """Join the last routed math event of every expert before any shared work."""
    event = cfg.event_capacity - ATOMIC_ADD_VALUE_LEN - SHARED_EXPERT_EVENTS + 1
    start = values.pre_task_num
    for expert, routed_event in enumerate(routed_events):
        task = TaskDescC()
        task.task_type = TaskType.TASK_BEGIN_TASK_GRAPH
        task.task_aicore_type = TaskAiCoreType.TASK_AICORE_VECTOR
        task.task_index, task.task_split_num, task.task_split_value = expert, len(routed_events), 1
        task.dependent_event, task.trigger_event = routed_event, event
        cfg.all_tasks[start + expert] = task
        cfg.vector_task_indices[values.pre_vector_task_num + expert] = start + expert
    cfg.task_index_num[1] += len(routed_events)
    cfg.all_event_num_triggers[event] = len(routed_events)
    local = replace(
        values, ep=1, all_expert_num=1, top_k=1, rank_id=0,
        pre_task_num=start + len(routed_events),
        pre_vector_task_num=values.pre_vector_task_num + len(routed_events),
        pre_pre_event_num=event - 1, pre_event_num=event,
    )
    for op in ops[1:]:
        op.fill_config.fill(cfg, op, local)
    return range(start, local.pre_task_num), local.pre_vector_task_num
