# 原仓 PR 迁移映射表（临时）

本表供过渡期将 HyperParallel 的 multicore PR 移植到 HyperMulticore 时使用。
来源核对基准为原仓 `5061284e3a9082010e733b38c002a7bf0152d1fc`，
目标结构核对基点为本仓 `5d7d98ac04c255f4636e2a0f56e3ffef368dd486`。
它记录路径与结构对应关系；后续同步来源包括 #1300（`bcffec39`）、#1343（`5c4caa16`）、
#1256（`42c2f4d9`）。具体实现以目标分支当前代码为准，验证结果保存在仓库外同步记录。
过渡结束后删除本表；不将其加入正式产品指南或制品清单。

## 移植一个 PR

1. 读取原 PR 的完整 diff、父提交及关联测试，逐文件确定属于组件、下游集成还是共享基础设施。
   对混合 PR 分别处理：组件变更移入本仓，模型及并行框架接入留在 HyperParallel。
2. 按下面的路径表定位；具体例外优先于目录前缀规则。对新增文件按职责确定位置。
   核对目标分支是否已含等价修改，按变更语义移植，避免整文件覆盖或直接套用原仓提交。
3. 按“结构差异”检查导入、算子注册、参数布局、依赖补丁与测试 oracle。
   保留当前独立构建和延迟加载入口；每个原 PR 文件都应有目标位置或留在原仓的理由。
4. 迁移相关测试并保持原断言与数值阈值；按影响范围选择现有用例，执行方法见
   [测试指南](tests/README.md)，构建方法见[构建指南](docs/build.md)。
   仅文档变更做静态检查；生产或构建输入变更使用对应新制品验证。
5. 在目标 PR 描述中记录原 PR 链接、源提交、移入/留在原仓的范围、必要适配及验证结果。
   对未验证部分写明原因；源码已移入与验证通过分别记录。

## 路径映射

表中路径相对于各自仓库根目录；`<相对路径>` 表示保留剩余目录和文件名。
为缩短例外表，`C/` 代表原仓 `hyper_parallel/core/multicore/`，
`N/` 代表本仓 `hyper_multicore/`。

| 原仓位置 | 本仓位置 | 处理方式 |
| --- | --- | --- |
| `hyper_parallel/core/multicore/<相对路径>` | `hyper_multicore/<相对路径>` | 默认规则，以下例外优先 |
| `C/modules/` | `N/modules/` | MoE、Gate、MHC 正反向及资源管理；包含 `mega_mhc_grad/`、`mega_mhc_common.py` |
| `C/scheduler/`、`C/tasks/` | `N/scheduler/`、`N/tasks/` | 保留内部相对结构，与 Device runtime 协议配套 |
| `C/ops/hyper_mega_*/`、`C/ops/runtime/` | `N/ops/hyper_mega_*/`、`N/ops/runtime/` | Host、Kernel、ACLNN、运行时头文件；下述已删除模板除外 |
| `C/torch/csrc/` | `N/torch/csrc/` | 保留源文件名，注册命名空间见下文 |
| `C/diagnostics/`、`C/shmem/` | `N/diagnostics/`、`N/shmem/` | 保留内部相对结构，SHMEM 包含 native、补丁和测试依赖的生命周期接口 |
| `C/cmake/`、`C/_build/`、`C/build.sh` | `N/cmake/`、`N/_build/`、`N/build.sh` | 路径对应；保留本仓构建结构，不能整文件替换 |
| `C/_loader.py`、`C/_automatic.py`、`C/_collective.py` | `N/_loader.py`、`N/_automatic.py`、`N/_collective.py` | 加载、自动清理与 collective 生命周期 |
| `C/ops/<文件名>.patch` | `N/ops/patches/<文件名>.patch` | 8 份算子补丁集中存放；SHMEM 的 2 份补丁仍在 `N/shmem/_build/patches/` |
| `C/torch/ops.py`、`C/torch/mega_gate_ops.py` | `N/torch/ops.py` | 已合并；Gate 的 `_mega_gate_route`、`_mega_gate_route_grad` 位于统一入口 |
| `C/scripts/hyper_parallel_multicore_set_env.bash` | `scripts/hyper_multicore_set_env.bash` | wheel 安装后的定位脚本；包内 `C/set_env.bash` 对应 `N/set_env.bash` |
| `C/docs/architecture.md`、`build.md`、`shmem.md` | `docs/` 下同名文件 | 文档已移到仓库级，内容按独立包使用方式维护 |
| `C/docs/mega_kernel_profiling.md` | `docs/profiling.md` | 已改名 |
| `C/README.md` | `N/README.md` | 组件指南；本仓根中英 README 维护独立包安装与开发入口 |
| `C/examples/` 的独立 MoE/mHC 小网络及接入指南 | `examples/` | 普通层、all-to-all EP、FP32 master AdamW 做独立适配；真实模型训练配方及完整并行框架留在原仓 |
| `tests/ut/core/multicore/<相对路径>` | `tests/ut/<相对路径>` | CPU 用例保留后缀路径；不能重新引入旧仓 import 或 fixture |
| `tests/torch/multicore/<相对路径>` | `tests/torch/<相对路径>` | NPU launcher、worker、SHMEM 和两份 MHC 精度脚本，以下例外优先 |
| `tests/torch/multicore/_case_launcher.py`、`_evidence_helpers.py`、`_mega_moe_utils.py`、`_test_env.py` | `tests/common/multicore/` 下同名文件 | 4 份 helper 已集中，引用路径同步调整 |
| `tests/torch/multicore/README.md` | `tests/README.md` | 已重写为 CPU/NPU 总测试指南，按当前使用方式更新 |
| `tests/common/multicore/mega_mhc_golden.py`、`mega_mhc_grad_golden.py` | 原相对路径 | 公共 MHC golden 保留 |
| `tests/common/distributed_launcher.py`、`port_utils.py`、`random_generator.py`、`mark_utils.py` | 原相对路径 | 公共测试辅助代码，只保留本仓所需能力 |
| `scripts/check_gcc_version.sh`、`scripts/native/cann_version.sh` | 原相对路径 | 构建环境检查 |
| `scripts/pre-commit/run_pylint.py`、`run_lizard.py` | 原相对路径 | 使用原仓包装器与策略 |
| `scripts/pylint_hyperparallel.py` | `scripts/pylint_hypermulticore.py` | 插件更名，移除不适用 Torch-only 的后端导入禁令，其余规则保留 |
| 根 `build.sh`、`setup.py`、`MANIFEST.in`、`requirements.txt` | 根同名文件 | 职责对应，实现按独立包维护；不复制原仓全项目依赖与打包逻辑 |

## 结构差异：迁移时必须保留

| 变更涉及内容 | 本仓处理方式 |
| --- | --- |
| Python 导入 | `hyper_parallel.core.multicore` → `hyper_multicore`。同步 #1256 后公共 MHC 类名为 `MegaMhc`，MoE 自动推导 token 容量，不再接收 `local_num_tokens` / `max_local_num_tokens` 构造参数；本仓不提供 `hyper_parallel` 兼容包，也不依赖原仓运行或构建 |
| C++ 命名空间 | 原 `hyper_parallel::multicore::...` → `hyper_multicore::...`，同步迁移 namespace、using 和限定名；例如 SHMEM runtime 的头文件与实现 |
| Torch dispatcher | `torch.ops.hyper_parallel.*` → `torch.ops.hyper_multicore.*`，C++ 的注册命名空间对应调整；同名算子的 schema 保持兼容 |
| Torch SO 与加载 | 原 `libhyper_parallel_mega_moe_torch.so`、`libhyper_parallel_mega_gate_torch.so`，以及新仓曾使用的两个组件 SO，均由单个 `libhyper_multicore_torch.so` 承接；目标为 `hyper_multicore_torch`，安装于包的 `lib/framework/torch/` |
| Python 算子包装 | 三类组件都经 `N/torch/ops.py` 延迟加载；保留 `_loader.py` 的环境、ABI 校验、vendor 预加载，以及 transport/梯度 alias 检查。新增注册源加入统一 CMake 目标，不恢复 `mega_gate_ops.py` |
| Vendor 与 SHMEM 名称 | vendor 为 `hyper_multicore_nn`；私有库使用 `hyper_multicore` 项目名。具体制品路径以 `N/_build/artifacts.py` 为准；ACLNN/`HyperMega*` 算子名、上游 CANN/SHMEM 符号和 bootstrap 名称保留 |
| SHMEM 用户配置 | 原 `HYPER_PARALLEL_SHMEM_*` 对应 `HYPER_MULTICORE_SHMEM_*` 的 `HEAP_SIZE`、`TIMEOUT_SEC`、`DATA_ENGINE`、`BOOTSTRAP_ENDPOINT` 四项；旧前缀不作兼容读取 |
| 内部标识 | 既有 `HP_*`/`hp_*` 编译、诊断和测试控制标识按实际用途保留，例如 `[HP-NATIVE-*]`。不能全局替换 `HP`、上游名称、版权或合法系统符号 |
| Transformer 依赖 | `dependencies.lock.json` 使用 schema 2。单个 `components.multicore.ops_transformer` 仓库下有 `revisions.gmm` 与 `revisions.mhc`；旧 `ops_transformer_mhc` 合入后者。单仓库缓存 `deps/ops_transformer/src`，按显式 commit 导出；重复 repository 顶层声明会被拒绝 |
| 补丁修改 | 修改 `N/ops/patches/` 或 SHMEM 补丁时，同步更新锁文件中对应 revision 的 hash、顺序等受影响信息；只更新 PR 实际变更的依赖版本，不将缓存源码作为补丁来源 |
| 构建与打包 | 本仓根构建、`setup.py`、`N/_build/artifacts.py` 和 `scripts/check_repository.py` 是独立工程入口；native 不在 pip 安装时编译。组件版本由 `artifacts.py` 定义，native wheel 按实际 Torch 构建身份生成元数据 |
| 调度 ABI | `scheduler/config.py`、`scheduler/runtime.py`、`tasks/`、`ops/runtime/runtime_config.hpp` 及 Torch 参数布局按变更范围配套核对，保留已有 C++ reader 检查 |
| 测试入口 | launcher 保持轻量，Torch/NPU 导入及执行在 worker 内。安装态测试在源码外运行。CPU 为 `python -m pytest tests/ut`，NPU 为 `python -m pytest tests/torch` |
| MoE reference | `tests/common/multicore/moe_reference.py` 是独立测试辅助实现，不是原仓同名文件搬迁。`TorchExperts` 对应原测试依赖的 `hyper_parallel/components/modules/moe.py::GroupedExperts` 与 `hyper_parallel/core/expert_parallel/expert_parallel.py::ExpertParallel` 的计算/分片语义；`grouped_matmul` 对照 `hyper_parallel/components/functional/grouped_matmul.py`。不移入 DTensor/EP 框架；PR 若改变路由、布局或梯度语义，需用同一输入、权重、上游梯度重新核对 reference，保持原阈值 |
| 测试 reference 的性能 | reference 用于正确性，不能沿用原仓 `GroupedExperts` 与 Mega 的速度比作为它的验收结论；涉及性能的 PR 使用真实原生路径对照，范围按该 PR 约定 |
| 代码检查配置 | 保留原仓三个 hook；Pylint 3.3.7，Python 目标 3.10，Lizard CCN≤15/NLOC≤100。其他工具延续未固定版本策略；不通过本表引入新工具版本 |

## 留在原仓或已移除的内容

| 原仓内容 | 归属或替代位置 |
| --- | --- |
| `tests/torch/multicore/test_deepseek_v41_megamoe.py`、`_test_deepseek_v41_parallel.py`、`_test_deepseek_v41_dynamic_tokens.py` | DeepSeek 模型/并行集成留在原仓 |
| `tests/torch/multicore/test_mega_gate_deepseek_v41.py`、`_test_mega_gate_deepseek_v41.py` | Gate 模型接入测试留在原仓 |
| `hyper_parallel/models/`、`tests/ut/auto_models/`、`tests/torch/auto_models/`、DTensor/EP/FSDP/分布式 checkpoint 及相应集成测试 | 留在原仓；若 PR 同时修改组件与这些模块，拆分各仓变更 |
| MoE、MoEGrad、Gate、GateGrad 各自根、`op_host/`、`op_graph/` 的 12 份 `CMakeLists.txt` | 已移除未参与统一构建的模板；实际构建入口为 `N/cmake/vendor/`，原 PR 的有效构建变更合入这里 |
| MoE、MoEGrad 的两份 `op_graph/fusion_pass/.gitkeep` | 已移除空目录占位；仅在有实际源码需求时增加对应文件 |
| 原仓全项目工具安装脚本及依赖配置 | 本仓按根 README 使用工具；本仓未保留 `pyproject.toml`、`requirements-dev.txt` 或独立 pytest 配置，不为套用原 PR 恢复它们 |
| 预编译包、`build/`、`dist/`、个人环境与迁移日志 | 不随 PR 移入源码；构建缓存和过程证据放在仓库外工作目录 |

路径表列出的文件名是定位依据；新增目录或接口时以当前实现和 PR 需求为准。
本表不授权顺带修复原仓已有问题，也不要求每个 PR 重跑迁移阶段的全量复建或性能验收。

## 独立网络示例的适配边界

`C/examples/` → `examples/`，采用 `examples.*` 模块入口。`_layers.py` 抽取普通层，
`_native.py` 保留 all-to-all EP 与本地专家参数，`_optimizer.py` 保留 FP32 master AdamW；
不迁入 DTensor/DeviceMesh/ParallelStyle 或通用优化器工厂。后续 PR 应分别核对这些抽取路径，
不能整文件覆盖回原仓框架依赖，也不能用 all-gather correctness reference 替换普通 EP benchmark。
现有 `mega_moe_minimal.py` 是独立化时新增的快速入口；重复的独立计时包装已移除，采样使用 minimal 参数。
