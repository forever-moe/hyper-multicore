# MegaGate A5 迁移验收记录（2026-11-09）

当前 MegaGate 适配已迁入本仓库的统一构建、加载、模块和测试体系。独立运行不需要 PanGu 源码或旧二进制。迁移参考是原仓库的实际工作区，不能只用其 HEAD 代表未提交的适配；参考文件及二进制哈希见 [来源清单](mega_gate_migration_sources.json)。

## 交付范围

- 新增 `sigmoid_route_from_logits` 与 `sigmoid_route_from_hidden_states`，支持 sigmoid 路由、融合投影、autograd、诊断和性能采样。
- 迁入路由正反向内核、A5 投影内核及运行时依赖，接入统一 Host/schema/ACLNN、Torch 注册和产物完整性校验。
- 锁定投影所需的 ops-nn MatMul 补丁及 SHA256；在隔离构建树应用补丁。
- 保留本仓库 `MegaGate` 类的参数管理和 text/vision 行为。融合 sigmoid 投影通过新函数显式调用。
- 迁入原实现的符号和上下文回调缓存，封装为 Gate 私有 `mega_gate_op_api.h`。其他组件的通用桥接未改变。
- 提供独立示例、CPU/NPU 测试、公共输入生成器及字节/性能比较工具。

原 PanGu 仓库未修改，其正式训练入口仍引用原组件。验收时仅在独立测试进程中替换 Router 的两个函数入口，使用真实 Router、HCCL 和反向计算。

## 构建身份

| 项目 | 本次记录 |
| --- | --- |
| 目标基线 | `fd8131f24512ccf0764b277460630e7d2b8cb0fd`，分支 `megagate_a5_master` |
| 来源 HEAD | `5ed7d9dd575b5fee2631c8bad39a0ff4c95676dd`，另冻结实际工作区文件 |
| 设备 | Ascend950PR_9589，32 AIC / 64 AIV |
| CANN | 9.2.0 正式版，20261008 |
| Torch / torch_npu | 2.12.0+cpu / 2.12.0.post3，cp311，aarch64 |
| 最终组件源码 SHA256 | `30a3b1fb9cb5f92f2b5cdf2f64a07445aeb73a1a6dd6552b1f0ecaba9ab67aed` |

统一 native-local 构建成功，发布目录为 `build/native/payload/hyper_multicore`。原生库哈希、适配器指纹和当前源码一致性见 [构建身份](validation/mega_gate_20261109/final_identity.json)。来源清单的 67 个文件验后未变化；通用 `cached_op_api.h` 与目标迁移前一致。

本机复现环境：

```bash
cd /home/w00504392/code/hyper-multicore
source /home/xh/CANN/cann-1008/Ascend/cann-9.2.0/set_env.sh
export PATH=/home/w00504392/workspace/mega_gate_migration_20261109/tools-env/bin:$PATH
GIT_LFS_SKIP_SMUDGE=1 bash build.sh --soc-list ascend950 --jobs 8 --native-only
source activate.bash
ASCEND_RT_VISIBLE_DEVICES=0 python examples/mega_gate_minimal.py --mode projection
```

本机 `/usr/local/Ascend/cann-9.2.0` 实为较旧 beta.2，不能替代上述正式版完成现有 A5 全量组件构建。新环境按 [接口与构建说明](mega_gate_a5.md) 使用匹配的正式工具链和锁定依赖。

## 功能与数值验收

| 检查 | 最终结果 |
| --- | --- |
| 完整 CPU 单元测试 | 349 passed，1943 subtests passed |
| 原 MegaGate + 新 A5 NPU 测试 | 11 passed |
| 独立组件：同一输入、同一参考权重 | 25 组、106 个张量逐字节相同 |
| 实际 PanGu Router：同一输入和参数 | 24 组、104 个张量逐字节相同 |
| TP2 + sequence parallel | 双 rank 通过，覆盖梯度归约 |
| Hyper FSDP2 | 双 rank 通过，覆盖权重 materialize 和 reduce-scatter |
| 搬移目录后运行 | 在 `/tmp` 启动搬移后的示例，真实前反向通过 |
| 变更文件复杂度检查 | lizard 通过，CCN ≤ 15、NLOC ≤ 100 |

字节比较保留原始专家槽位、INT64 索引及浮点 signed zero，不通过排序消除差异。两份详情分别为 [组件对比](validation/mega_gate_20261109/parity.json) 和 [Router 对比](validation/mega_gate_20261109/router_parity.json)。

组件覆盖 FP32/BF16/FP16 路由、归一化与非归一化、可选 bias、尾块、不同 K/E、投影 T=1/31/32/33/129/4096/6144、H=64/128/5120、E=8/12/16/33/384；比较 logits、路由权重、专家索引、dX/dW，检查 bias 无梯度。另覆盖仅 logits/仅 weights 梯度、冻结输入/权重、HF32/autocast 状态、双流与串行一致性、checkpoint，以及 12 步带 momentum 的更新、bias 更新和优化器序列化恢复。

实际 Router 覆盖 T4096/T6144、route/projection、fix-router 开关、辅助/z loss、专家计数、前向、训练和 PanGu checkpoint。双卡结果保留在本目录下的 [验证数据](validation/mega_gate_20261109)，分布式对原生 Router 的比较使用来源测试已有的误差阈值，并非逐字节一致承诺。

CPU 套件报告的 5 个 warning 是现有 pytest mark 未注册；NPU 套件报告框架弃用及 mark warning，无失败用例。

## 性能验收方法

比较迁移前后的同一适配实现。固定 H5120/E384/K8，分别测 T4096/T6144 的内核、公共前向、前反向、checkpoint，以及真实 Router 的 route/projection 前向、训练、checkpoint，共 20 个测量项。

每项预热 20 次，9 轮，每轮 100 次；记录 NPU event 与同步 wall time。使用同一张物理 NPU0、固定 CPU 核 `68,70,72,74,76,78,80,82`，先完成其他测试再计时。顺序为来源1、迁移1、迁移2、来源2，各次均为全新进程，每侧两个独立进程。

首轮 20 项中 19 项通过。T4096 route checkpoint 的迁移1进程存在一轮 16.253ms（其他轮次主要约 3.326ms），区间上界 5.329%，判为证据不足，见 [首轮判定](validation/mega_gate_20261109/final_first_abba/router_performance.json)。保留全部原始轮次，追加真实 Router 的来源3、迁移3、迁移4、来源4；Router 最终每侧四个独立进程，组件仍为每侧两个。未剔除异常轮次或放宽门槛。

四进程复查时该项区间上界仍为 2.678%，见 [四进程结果](validation/mega_gate_20261109/router_four_processes.json)。随后仅对 T4096 route 链路追加来源5、迁移5、迁移6、来源6；使用相同计时代码和预热/轮次设置，仅限制 fixture 为 T4096、mode 为 route。最终这三个测量项每侧六个进程，其余 Router 项每侧四个，组件每侧两个；所有原始样本均保留。

数值是各进程轮次中位数的平均值。对进程和轮次做 10000 次分层 bootstrap，报告回退比例的 95% 区间；仅当上界不超过 2% 时判定通过。区间跨过 2% 为证据不足，不按通过处理。

最终 20 项均通过 2% 门槛。组件 8 项显存峰值在所有来源/迁移进程中逐项相同，测量后额外常驻显存均为 0。正变化表示迁移后变慢。

### 独立组件（每侧两个进程）

| 测量项 | 来源 ms | 迁移 ms | 变化 | 95% 区间 |
| --- | ---: | ---: | ---: | ---: |
| T4096/kernel | 0.624113 | 0.624013 | -0.016% | [-0.037%, +0.041%] |
| T4096/forward | 0.683500 | 0.683310 | -0.028% | [-0.231%, +0.105%] |
| T4096/forward_backward | 2.144196 | 2.142914 | -0.060% | [-0.099%, +0.020%] |
| T4096/checkpoint | 2.843625 | 2.842494 | -0.040% | [-0.099%, +0.040%] |
| T6144/kernel | 0.927460 | 0.927463 | +0.000% | [-0.034%, +0.027%] |
| T6144/forward | 1.027025 | 1.027657 | +0.062% | [-0.072%, +0.178%] |
| T6144/forward_backward | 3.204124 | 3.202570 | -0.048% | [-0.075%, -0.017%] |
| T6144/checkpoint | 4.255924 | 4.255332 | -0.014% | [-0.038%, +0.016%] |

### 实际 Router

| 测量项 | 每侧进程 | 来源 ms | 迁移 ms | 变化 | 95% 区间 |
| --- | ---: | ---: | ---: | ---: | ---: |
| T4096/route/forward | 6 | 0.863626 | 0.864600 | +0.113% | [-0.062%, +0.348%] |
| T4096/route/training | 6 | 2.458143 | 2.457349 | -0.032% | [-0.074%, +0.030%] |
| T4096/route/checkpoint | 6 | 3.325679 | 3.326486 | +0.024% | [-0.031%, +1.785%] |
| T4096/projection/forward | 4 | 0.785696 | 0.786281 | +0.074% | [-0.064%, +0.210%] |
| T4096/projection/training | 4 | 2.348760 | 2.348255 | -0.022% | [-0.086%, +0.045%] |
| T4096/projection/checkpoint | 4 | 3.136027 | 3.137711 | +0.054% | [-0.095%, +0.222%] |
| T6144/route/forward | 4 | 1.277111 | 1.277570 | +0.036% | [-0.041%, +0.116%] |
| T6144/route/training | 4 | 3.568150 | 3.568580 | +0.012% | [-0.045%, +0.051%] |
| T6144/route/checkpoint | 4 | 4.845147 | 4.845888 | +0.015% | [-0.033%, +0.066%] |
| T6144/projection/forward | 4 | 1.150908 | 1.151573 | +0.058% | [-0.026%, +0.133%] |
| T6144/projection/training | 4 | 3.416013 | 3.414227 | -0.052% | [-0.103%, +0.111%] |
| T6144/projection/checkpoint | 4 | 4.564244 | 4.567925 | +0.081% | [-0.046%, +0.232%] |

完整数据：[组件性能](validation/mega_gate_20261109/performance.json)、[Router 合并结果](validation/mega_gate_20261109/router_performance.json)、[T4096 route 六进程复查](validation/mega_gate_20261109/router_t4096_route_performance.json)。同目录的 `sourceN*.json` / `targetN*.json` 保留全部原始轮次和 wall time。

## 发现并修复的迁移问题

初版复用目标通用桥接，单独内核计时接近原实现，但 T4096 实际 Router 的 projection 训练和 checkpoint 分别回退约 11.44% / 15.72%。主机侧诊断发现符号及上下文回调解析开销：投影调用约 83μs → 264μs，路由反向约 41μs → 143μs。

最终将来源的缓存桥接按原调用语义迁入 Gate 私有头文件，仅缓存进程生命周期的函数符号与回调；executor、workspace 和执行上下文仍按调用/线程处理。修复后诊断值约为投影 87μs、路由反向 42μs。这些主机诊断值只用于定位，正式验收依据上面的独立进程性能数据。保留 [初版失败数据](validation/mega_gate_20261109/initial_bridge/router_performance.json) 便于审查。

另将源 profiling 包装适配为本仓库现有 `acquire_call` 接口；沿用实际设备 AIV 容量，更新旧测试中写死的 48 核预期。CPU 旧架构断言同步限定为使用通用桥接的组件。

## 使用及边界

完整参数约束和命令见 [MegaGate A5 接口说明](mega_gate_a5.md)，最小运行示例为 `examples/mega_gate_minimal.py`。组件不初始化通信、不管理消费者参数、不包办 Router 的缩放、辅助 loss、bias 更新或 fixed-router 替换。FSDP 调用前须 materialize 完整权重；分布式投影可使用 logits 路由入口。

本次验证证明迁移与来源适配在已测范围内一致。来源融合投影相对原始非融合 Router 存在 FP32 归约顺序变化和已知多步轨迹漂移；迁移验收不等同于证明原始 Router 收敛等价。未执行 512 卡长训或集群 MFU 验证，也未外推集群性能收益。

## 数据保留与复查

仓库保留来源清单、完整比较 JSON、构建身份、独立进程计时数据及双卡摘要；大型 `.pt` 和详细日志保存在本机工作目录 `/home/w00504392/workspace/mega_gate_migration_20261109`。

仓库内可复用工具：

- `tests/torch/benchmark_mega_gate_migration.py`：生成共享 fixture、组件资格验证与计时，可通过 `--source-root` 单独加载来源适配。
- `tests/torch/compare_mega_gate_migration.py`：字节比较、2% 性能判定和置信区间。
- `tests/torch/test_mega_gate_a5.py`：独立 NPU 门禁，不依赖 PanGu。

真实 PanGu 集成复查脚本 `router_compare.py`、`distributed_check.py`、ABBA 驱动 `run_timings.bash` 及两轮补测驱动保存在上述本机工作目录。它们只在验收进程重接函数引用，未修改来源代码。最终结果属于当前工作区，尚未 commit 或 push。
