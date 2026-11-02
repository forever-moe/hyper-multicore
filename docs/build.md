# 构建与安装

从源码构建 native wheel，或将已有 wheel 安装到配套环境。CPU/NPU 用例的添加、执行和排错见
[测试指南](../tests/README.md)。

## 环境要求

| 项目 | 要求 |
| --- | --- |
| 系统 | Linux；首轮框架矩阵为 aarch64，其他平台须登记并验证 |
| Python | 包元数据为 >=3.10,<3.13；首轮框架矩阵为 cp310，构建需要开发头文件 |
| 框架 | 相互配套的 Torch、torch-npu/PTA，NumPy >=1.26,<2 |
| Ascend | 构建及首轮验收使用 CANN toolkit/ops 9.2.0；运行环境的 PTA/CANN/driver 须彼此兼容 |
| 构建工具 | GCC/G++、CMake >=3.18、GNU Make、Git、pybind11 和 CANN 编译工具 |

先激活所选 Python 环境并安装配套框架，然后从源码根目录运行：

```bash
source /path/to/cann/set_env.sh
python -m pip install -r requirements.txt setuptools wheel "packaging>=22" "pybind11>=2.11.1" decorator attrs psutil
```

CANN 路径按实际安装位置填写；GCC 版本由
[工具链检查脚本](../scripts/check_gcc_version.sh)检查。
Torch adapter 和 SHMEM binding 的 C++ 标准由目标 TorchConfig 决定。完整矩阵中的 Torch 2.14
要求正式 C++20；仅支持早期 C++20 草案的编译器不能满足其头文件检查。
需要时在构建前显式设置 `HYPER_MULTICORE_FRAMEWORK_CXX` 为兼容目标 op-plugin 头文件、且支持该标准的 G++。
例如 `HYPER_MULTICORE_FRAMEWORK_CXX=/path/to/g++ bash build.sh --all --soc-list ascend910b --jobs 24`。
该变量只影响框架适配对及其预检，公共 CANN vendor/SDK/ops 继续使用原有工具链；
未设置时沿用普通 `CXX` 或系统默认编译器。框架与公共资源分别记录工具链并独立计算缓存身份。
仅满足语言标准仍不足以保证兼容；实际执行桥编译和链接预检是进入 native 构建的必要条件。
构建记录实际编译器路径和版本，不自动更换或安装工具链；默认构建仍使用当前选择的编译器。

## 编译

```bash
# 默认：只为当前安装的 Torch/PTA 构建一套适配对。
bash build.sh --soc-list ascend910b --jobs 24

# 发布：为框架锁中的全部组合构建适配对，汇总一个 wheel。
bash build.sh --all --soc-list ascend910b --jobs 24

# 为当前框架构建双 SoC 产物。
bash build.sh --soc-list ascend910b,ascend910_93 --jobs 24

# 仅构建 native payload，不生成 wheel。
bash hyper_multicore/build.sh --soc-list ascend910b --jobs 24
```

`--soc-list` 选择编译目标，`--jobs` 设置并行度。`--clean` 清理 native 工作及安装输出，
保留 Git/框架下载缓存和已准备的框架环境。`--all` 只扩大框架矩阵，
不隐含清缓存或增加 SoC。默认路径按实际头文件、执行桥和编译能力预检，不要求达到全量矩阵的 post 版本；
缺少能力时明确失败，不自动替换本机框架。不支持 `--multicore` / `--strict`；本仓只构建本组件，
SHMEM 为必需内部依赖。任一 native 构建或校验失败都会非零退出并停止组包。

| 输出 | 位置 |
| --- | --- |
| 完整 wheel | `dist/`，实际文件路径由根构建命令打印 |
| 公共资源及逐组合输出 | `build/native/common/<fingerprint>/`、`build/native/variants/<fingerprint>/` |
| 可运行 native payload | `build/native/payload/hyper_multicore/` |
| 构建及逐阶段日志 | `build/native/logs/<build-id>/`，包含 `stages.json` |

失败时先查看控制器打印的阶段、组合及日志路径；底层脚本同时提供 `HP_NATIVE_REASON_CODE`。
日志包含依赖获取、源码组装、SHMEM、vendor、Torch adapter 和制品校验阶段。
源码及生成物分开保存，构建不会把 native 文件写回生产源码目录。

## 安装与激活

### 使用 wheel

按 Python ABI、CPU 架构和 Torch/PTA 分支选择制品，安装构建命令打印的唯一 wheel。
运行 PTA 可以早于构建 post，CANN 不必与构建版本相同；本包不强制该版本配对，
使用方须准备彼此兼容的框架、CANN 和驱动。
替换下面示意路径，不要用通配符一次安装多个制品：

```bash
python -m pip install --no-deps "/path/to/hyper_multicore-<version>-<python>-<abi>-<platform>.whl"
source "$(command -v hyper_multicore_set_env.bash)"
```

`--no-deps` 使用当前环境预先配置的框架。激活脚本设置随包 vendor 和 SHMEM 库的路径；
必须在导入 Torch/torch-npu 之前激活。更换制品或环境后应启动新的 Python 进程。

检查安装来源时，在源码目录外运行，并移除指向源码的 `PYTHONPATH`：

```bash
python -c 'import hyper_multicore; print(hyper_multicore.__file__)'
```

输出应指向安装位置。然后按[快速开始](../README.md#快速开始)运行现有示例，
或按[测试指南](../tests/README.md)运行指定用例。

### 使用源码 payload

从源码根目录完成 native 构建后，在当前源码目录中激活本地 payload：

```bash
source /path/to/cann/set_env.sh
source build/native/payload/hyper_multicore/lib/set_env.bash
```

此模式从当前源码导入 Python 实现、从 `build/native/payload/` 定位 native 库。
不要同时激活另一安装版本的 payload。

## 制品内容与兼容性

默认 wheel 版本包含实际 `torch-npu` 发行版本，例如 `0.1.0.dev0+pt.2.12.0`。
`--all` 全量 wheel 使用组件版本，不带单一 PTA 后缀，例如
`hyper_multicore-0.1.0.dev0-cp310-cp310-linux_aarch64.whl`。
首轮构建目标为 PTA 2.7.1.post11、2.9.0.post9、2.10.0.post7、2.11.0.post3、2.12.0.post3、2.13.0 和 2.14.0，
实际精确 wheel、依赖和哈希见[框架锁](../hyper_multicore/_build/frameworks.lock.json)。构建目标不是全部运行环境已验收的声明。
完整身份保存在生成的 `build_info.json`，包括 Python/C++ ABI、Torch/PTA、CANN、
op-plugin 头文件、源码和 native 文件摘要。源码摘要包含算子 `.ini` 构建配置。
`native_sha256` 保留共享库摘要，`payload_sha256` 记录除 `build_info.json` 自身外的完整
payload 文件及 SHA256，包括 kernel、元数据、索引、配置和激活脚本。
打包前检查所选 SoC 的必需运行资源、文件集合及摘要；旧 payload 缺少完整清单时会被拒绝，
应从可信构建输入重新生成并校验清单，不能通过跳过校验打包。
加载按 Torch/PTA 分支、Python/CPU/C++ ABI 选定唯一适配对，不比较运行 PTA post 与构建版本的大小，
也不以外部 PTA 库 SHA256、Torch commit、CANN 版本或版本文件哈希拦截。
缺少 PTA、缺库/符号或初始化失败时保留原始原因和选中变体；一个进程不换用其他适配对重试。
不同内容的同名 wheel 不覆盖已有文件，新文件进入 `dist/<build-id>/`，以命令打印的精确路径为准。

每个变体在 `adapters/<variant-id>/` 中携带 `libhyper_multicore_torch.so` 与
`hyper_multicore_shmem_torch.so`；MegaMoE、MegaGate 和 MegaMHC 共用该变体的 Torch adapter，
通过统一入口在首次调用时加载。vendor 位于
`lib/vendors/hyper_multicore_nn/`，私有通信库位于 `shmem/lib/`。
wheel 包含运行所需 native payload 和 runtime 头文件，不包含完整 Torch、CANN 或 driver。
没有 `hyper_parallel` 兼容包，也不支持在同一进程中混用两套 native runtime。

查看当前安装包的实际构建矩阵，无需导入 Torch/PTA 或加载 SO：

```bash
hyper_multicore_build_info
hyper_multicore_build_info --json
python -m pip show -f hyper-multicore
```

矩阵命令读取当前安装位置的 `build_info.json` 和 `adapters/manifest.json`，展示构建模式、
框架版本、C++ ABI、CANN/SoC 及 SO 路径。默认包只显示一组；未安装 PTA 也能查看。
`pip show` 保留标准元数据输出，`-f` 可定位包内清单文件。
构建矩阵用于追溯；允许加载或加载成功均不代替当前环境的功能验收。

生成源码包：

```bash
python setup.py sdist
```

sdist 使用通用源码版本，包含源码、CMake、锁文件、补丁及构建所需测试资源。
安装 wheel 和导入 Python 包均不会自动编译算子。

## 依赖与缓存

依赖固定在[锁文件](../hyper_multicore/_build/dependencies.lock.json)。一个顶层条目对应一个
Git 仓库；同仓库的多个版本放在 `revisions` 中。`ops_transformer` 的 `gmm` 和 `mhc`
共用 `build/native/deps/ops_transformer/src`，按锁定 commit 分别导出源码片段。
`ops_nn` 和 `shmem` 使用各自的缓存仓库，工作目录的 HEAD 不参与版本选择。

准备器只 fetch 缺失提交，完整热缓存不访问远端。提交、tree、archive 和补丁均按锁文件校验，
补丁只应用于隔离源码副本。依赖校验失败直接报错，不删除源码或重新 clone 掩盖失败。
构建缓存按输入指纹区分。`--source-dir` 校验已有依赖源码；`--verify-only` 不获取提交或修改缓存。

全量构建顺序使用项目内的 `build/native/framework-envs/`，通过各环境的绝对 Python 路径执行，
不激活或覆盖调用者环境。包哈希或基础解释器变化生成新环境；成功全量构建后保留当前环境及每条维护分支
最近一套成功回退环境，回收更早闲置环境。失败构建和默认路径不触发该回收。

离线用户预先准备运行依赖后安装本地全量 wheel，安装、导入和首次调用不会下载框架、获取 native 源码或执行编译。
本仓不提供离线源码构建流程。

## CPU 开发

```bash
python -m pip install -r requirements.txt setuptools wheel "packaging>=22"
python -m pip install pytest
python -m pip install --no-deps --no-build-isolation -e .
python scripts/check_repository.py
python -m pytest tests/ut -q
```

CPU 单测包含 C++17 runtime ABI reader，需要 G++。没有 native 的 editable 安装仅用于
Host 开发；执行 NPU 用例仍需先构建并激活 native payload。
