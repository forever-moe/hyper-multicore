# A5 sigmoid MegaGate

Migration evidence and acceptance results are recorded in the
[2026-11-09 migration report](mega_gate_migration_20261109.md).

The functional APIs accept caller-owned tensors and parameters. They perform no
collectives and do not initialize HCCL or SHMEM communication. Build the unified
payload, activate it before starting Python, and use one vendor per process:

```bash
source /path/to/cann/set_env.sh
GIT_LFS_SKIP_SMUDGE=1 bash build.sh --soc-list ascend950 --jobs 8 --native-only
source activate.bash
ASCEND_RT_VISIBLE_DEVICES=0 python examples/mega_gate_minimal.py --mode projection
```

The build verifies the locked upstream source commits and patches. Skipping LFS
smudge avoids downloading documentation images; required kernel sources are
exported and patched in the isolated build tree. No old binaries or PanGu source
checkout is needed to build or execute the component.

## Interfaces

```python
from hyper_multicore.modules.mega_gate import (
    sigmoid_route_from_logits,
    sigmoid_route_from_hidden_states,
)

weights, indices = sigmoid_route_from_logits(
    logits, top_k=8, expert_bias=expert_bias, normalize=True,
)
logits, weights, indices = sigmoid_route_from_hidden_states(
    hidden_states, weight, expert_bias=expert_bias,
)
```

| Entry | Contract |
| --- | --- |
| Logits route | Nonempty local `[T,E]` FP32/FP16/BF16 logits on Ascend950; `E<=384`; `1<=K<=E` |
| Normalized route | Supported `K=1..8,16,128,384`, subject to `K<=E`; K1 returns its score directly |
| Projection | Nonempty local `[...,H]` hidden states and full `[E,H]` weight, FP32/BF16; `H%64==0`, `8<=E<=384` |
| Projection hardware | Validated Ascend950 with 32 AIC and 64 AIV; token tile remains 32 |
| Projection routing | Normalized sigmoid TopK8; FP32 logits `[T,E]` and weights `[T,8]`, INT64 ordered indices `[T,8]` |
| Bias | Optional FP32 `[E]` on the same device; changes selection only and receives no gradient |
| Backward | First-order autograd; direct logits gradients and routed gradients are combined before dW/dX |

The APIs return unscaled routing weights. The caller owns scaling, output casts,
auxiliary/z losses, fixed-router index replacement, bias updates and statistics.
Projection disables autocast locally and uses FP32 KEEP_DTYPE derivative MatMuls,
without changing the process HF32 policy. Full-block checkpointing is supported;
FP32-cast-only recomputation is not implemented by the projection entry.

FSDP must materialize the full weight before projection. Distributed tensor
subclasses and partial/sharded weights are rejected. A caller with distributed
projection should retain that projection and call the logits-only API on its
complete local logits. The functions implement no group-limited routing or replay.

The existing `MegaGate` class retains its text/vision and Torch scoring behavior.
Using these functions explicitly does not replace its Parameter management or
silently redirect all existing sigmoid calls to a different projection policy.

## Validation

CPU contracts and independent NPU tests:

```bash
python -m pytest tests/ut/modules/mega_gate -q
ASCEND_RT_VISIBLE_DEVICES=0 python -m pytest tests/torch/test_mega_gate_a5.py -q
ASCEND_RT_VISIBLE_DEVICES=0 python -m pytest tests/torch/test_mega_gate.py -q
```

NPU tests cover route normalization and direct gradients, projection tails,
FP32/BF16, reentrant/non-reentrant checkpointing, optimizer updates, streams and
complete profiler traces. The original Gate suite protects sqrtsoftplus,
text/vision, precision order and concurrent symbol resolution.

`tests/torch/benchmark_mega_gate_migration.py` generates shared CPU fixtures and
runs one isolated source or target process. Its optional `--source-root` is a
qualification-tool input only; normal tests and the runtime have no PanGu
dependency. Activate the matching vendor separately for each side:

```bash
python tests/torch/benchmark_mega_gate_migration.py \
  --mode generate --fixture /tmp/gate-fixture.pt
python tests/torch/benchmark_mega_gate_migration.py \
  --mode qualify --fixture /tmp/gate-fixture.pt --output /tmp/gate-target.pt
python tests/torch/benchmark_mega_gate_migration.py \
  --mode benchmark --fixture /tmp/gate-fixture.pt --output /tmp/gate-target-timing.json
```

Run the same commands under the source environment with `--source-root` pointing
to the reference repository, using different output paths. Each output records
the shared fixture hash, harness hash, implementation path and framework/device
identity. Compare every tensor in its original slot order; do not sort expert
indices to hide differences. Qualification includes both gradient branches,
frozen inputs/weights, HF32/autocast state, streams, and twelve-step training with
momentum, bias updates and serialized optimizer restore.

Normal timings disable diagnostics and report kernel, public forward,
forward/backward and checkpoint costs separately at T4096/T6144, H5120, E384.
Defaults are twenty warmups and nine rounds of one hundred iterations, using
NPU events and synchronized wall time. Alternate source and target processes;
judge the agreed 2% regression limit against round/process variability. Peak and
retained allocated memory are recorded separately.

The CPU comparator checks tensor bytes, including signed zeros and original
expert slots. For timing acceptance it resamples independent processes and
rounds, and reports pass, regression or inconclusive against the 2% limit:

```bash
python tests/torch/compare_mega_gate_migration.py --kind tensors \
  --reference /tmp/gate-source.pt --candidate /tmp/gate-target.pt \
  --output /tmp/gate-parity.json
python tests/torch/compare_mega_gate_migration.py --kind timings \
  --reference /tmp/source1.json --reference /tmp/source2.json \
  --candidate /tmp/target1.json --candidate /tmp/target2.json \
  --output /tmp/gate-performance.json
```

Migration parity does not establish original Router convergence equivalence.
The reference projection changes FP32 reduction order and has known multi-step
trajectory drift; near-tie TopK decisions require explicit validation. Neither
standalone component timings nor these tests establish cluster MFU gains.
