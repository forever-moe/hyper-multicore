# 构建与安装

从源码构建 native wheel，或将已有 wheel 安装到配套环境。CPU/NPU 用例的添加、执行和排错见
[测试指南](../tests/README.md)。

## 环境要求

| 项目 | 要求 |
| --- | --- |
| 系统 | Linux x86_64/aarch64 |
| Python | >=3.10,<3.13，包含开发头文件 |
| 框架 | 相互配套的 Torch、torch-npu/PTA，NumPy >=1.26,<2 |
| Ascend | 配套的 driver、CANN toolkit/ops >=9.1.0 |
| 构建工具 | GCC/G++、CMake >=3.18、GNU Make、Git、pybind11 和 CANN 编译工具 |

先激活所选 Python 环境并安装配套框架，然后从源码根目录运行：

```bash
source /path/to/cann/set_env.sh
python -m pip install -r requirements.txt setuptools wheel "packaging>=22" "pybind11>=2.11.1"
```

CANN 路径按实际安装位置填写；GCC 版本由
[工具链检查脚本](../scripts/check_gcc_version.sh)检查。

## 编译

```bash
# 完整构建并生成 wheel。
bash build.sh --soc-list ascend910b --jobs 24

# 构建双 SoC 产物。
bash build.sh --soc-list ascend910b,ascend910_93 --jobs 24

# 仅构建 native payload，不生成 wheel。
bash hyper_multicore/build.sh --soc-list ascend910b --jobs 24
```

`--soc-list` 选择编译目标，`--jobs` 设置并行度。`--clean` 清理 native 工作及安装输出，
保留下载的 Git 依赖缓存。不支持 `--multicore` / `--strict`；本仓只构建本组件，
SHMEM 为必需内部依赖。任一 native 构建或校验失败都会非零退出并停止组包。

| 输出 | 位置 |
| --- | --- |
| 完整 wheel | `dist/`，实际文件路径由根构建命令打印 |
| 组件 native 输出 | `build/native/components/multicore/hyper_multicore/` |
| 可运行 native payload | `build/native/payload/hyper_multicore/` |
| 完整组件构建日志 | `build/native/logs/multicore.log` |

失败时先查看构建输出中的 `HP_NATIVE_REASON_CODE` 和错误指向的日志。
日志包含依赖获取、源码组装、SHMEM、vendor、Torch adapter 和制品校验阶段。
源码及生成物分开保存，构建不会把 native 文件写回生产源码目录。

## 安装与激活

### 使用 wheel

在与构建身份一致的 Python/Torch/PTA/CANN 环境中，安装构建命令打印的唯一 wheel。
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

wheel 版本自动使用实际 `torch-npu` 发行版本，例如 `0.1.0.dev0+pt.2.9.0.post4`。
完整身份保存在生成的 `build_info.json`，包括 Python/C++ ABI、Torch/PTA、CANN、
op-plugin 头文件、源码和 native 文件摘要。源码摘要包含算子 `.ini` 构建配置。
`native_sha256` 保留共享库摘要，`payload_sha256` 记录除 `build_info.json` 自身外的完整
payload 文件及 SHA256，包括 kernel、元数据、索引、配置和激活脚本。
打包前检查所选 SoC 的必需运行资源、文件集合及摘要；旧 payload 缺少完整清单时会被拒绝，
应从可信构建输入重新生成并校验清单，不能通过跳过校验打包。
加载前校验当前环境身份，不匹配即报错。
不同内容的同名 wheel 不会被构建脚本覆盖。

MegaMoE、MegaGate 和 MegaMHC 共用 `lib/framework/torch/libhyper_multicore_torch.so`，
通过 `hyper_multicore.torch.ops` 在首次调用时加载。vendor 位于
`lib/vendors/hyper_multicore_nn/`，私有通信库位于 `shmem/lib/`。
wheel 包含运行所需 native payload 和 runtime 头文件，不包含完整 Torch、CANN 或 driver。
没有 `hyper_parallel` 兼容包，也不支持在同一进程中混用两套 native runtime。

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
