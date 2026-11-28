# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Plan a task-local Dense MXFP8 QuantBatchMatmulV3 using the CANN tiler."""

from dataclasses import dataclass
from functools import lru_cache
import struct

from hyper_multicore.scheduler.config import TaskType
from hyper_multicore.tasks.utils import fill_expert_compute_tasks

_TILING_BYTES = 64
_TILING_WINDOW_OFFSET = 56
_ASW_WINDOW_LEN = 4
_RECORD = struct.Struct("<QIII4x")
_TRANSPOSE = {"NN": 0, "NT": 4, "TN": 1}
_KERNEL_TYPES = (0, 1, 9, 10)
_BLAZE_WITHOUT_BATCH = (2 << 16) | (1 << 4)
_GROUP_SIZE_MX = (1 << 32) | (1 << 16) | 32


def _tensor(shape, dtype):
    return {"shape": shape, "ori_shape": shape, "format": "ND", "ori_format": "ND", "dtype": dtype}


@lru_cache(maxsize=64)
def qbmm_tiling(layout, m, n, k, output_bytes, cubes, *, device):
    """Return (key, block count, raw tiling) for one unbatched ND QBMM."""
    transpose = _TRANSPOSE[layout]
    if m == 0 or k == 0:
        # Keep a Cube completion for an empty output or a zero-K weight gradient.
        return _BLAZE_WITHOUT_BATCH | transpose, cubes, struct.pack("<3I52x", m, n, k)

    import torch
    from tbe.common.utils.op_tiling import do_op_tiling

    trans_a, trans_b = layout[0] == "T", layout[1] == "T"
    a_shape, b_shape = ((k, m) if trans_a else (m, k)), ((n, k) if trans_b else (k, n))
    scale_k = (k + 63) // 64
    a_scale = (scale_k, m, 2) if trans_a else (m, scale_k, 2)
    b_scale = (n, scale_k, 2) if trans_b else (scale_k, n, 2)
    dtype = "float32" if output_bytes == 4 else "bfloat16"
    attrs = [
        {"name": "dtype", "dtype": "int", "value": 0 if output_bytes == 4 else 27},
        {"name": "transpose_x1", "dtype": "bool", "value": trans_a},
        {"name": "transpose_x2", "dtype": "bool", "value": trans_b},
        {"name": "group_size", "dtype": "int", "value": _GROUP_SIZE_MX},
    ]
    # The installed OPP tiler queries the current runtime device's platform.
    with torch.npu.device(device):
        info = do_op_tiling(
            "QuantBatchMatmulV3", {},
            [_tensor(a_shape, "float8_e4m3fn"), _tensor(b_shape, "float8_e4m3fn"),
             _tensor(b_scale, "float8_e8m0"), None, None, _tensor(a_scale, "float8_e8m0")],
            [_tensor((m, n), dtype)], attrs=attrs,
        )
    key, workers, data = info["tiling_key"], info["block_dim"], info["tiling_data"]
    kernel_types = (0, 1) if output_bytes == 4 else _KERNEL_TYPES
    allowed_keys = {_BLAZE_WITHOUT_BATCH | (kind << 8) | _TRANSPOSE[layout] for kind in kernel_types}
    if (key not in allowed_keys or not 0 < workers <= cubes or len(data) != _TILING_BYTES
            or struct.unpack_from("<3I", data) != (m, n, k)):
        raise ValueError(f"Unsupported Dense MXFP8 QBMM tiling on {device} for {layout} {(m, n, k)} -> {dtype}: "
                         f"key={key}, blocks={workers}, bytes={len(data)}")
    # The pinned worker uses a four-tile ASW window and derives its own L0 shape.
    # Older CANN tilers leave these fields reserved and zeroed.
    window_len, m_l0, n_l0 = struct.unpack_from("<IHH", data, _TILING_WINDOW_OFFSET)
    if window_len not in (0, _ASW_WINDOW_LEN) or m_l0 or n_l0:
        raise ValueError(f"Incompatible Dense MXFP8 QBMM tiling on {device} for {layout} {(m, n, k)} -> {dtype}: "
                         f"key={key:#x}, windowLen={window_len}, mL0={m_l0}, nL0={n_l0}")
    return key, workers, data


@dataclass
class Mxfp8QbmmFillConfig:
    key: int
    workers: int
    tiling: bytes
    x_scale: int
    w_scale: int
    records: list

    def fill(self, cfg, op, values):
        index = len(self.records)
        self.records.append(_RECORD.pack(self.key, self.x_scale, self.w_scale, op.outputs[0].dtype_size)
                            + self.tiling)
        fill_expert_compute_tasks(cfg, op, values, TaskType.TASK_MX_QBMM, self.workers, False,
                                  [index] * self.workers, cube=True)
