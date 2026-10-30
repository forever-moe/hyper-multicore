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
"""Locate and load the component-owned multicore payload."""

from __future__ import annotations

import ctypes
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import platform
import sys

_VENDOR_NAME = "hyper_multicore_nn"


class NativeComponentUnavailableError(RuntimeError):
    """Report an unavailable or incorrectly activated multicore component."""


def require_multicore_environment() -> Path:
    """Require the packaged custom OPP environment prepared by ``set_env.bash``."""
    component_root = _component_root()
    vendor_root = component_root / "vendors" / _VENDOR_NAME
    if not vendor_root.is_dir():
        raise NativeComponentUnavailableError(
            f"[HP-NATIVE-PAYLOAD-MISSING] component=multicore vendor={vendor_root}. "
            "Inspect the current build log; for source/PYTHONPATH development, run "
            "./build.sh."
        )
    vendor_root = vendor_root.resolve()
    op_api_root = (vendor_root / "op_api" / "lib").resolve()
    shmem_root = (component_root.parent / "shmem" / "lib").resolve()
    required_library_roots = (op_api_root, shmem_root, shmem_root / "shmem")
    missing_variables = []
    if not _environment_contains_path("ASCEND_CUSTOM_OPP_PATH", vendor_root):
        missing_variables.append("ASCEND_CUSTOM_OPP_PATH")
    if not all(_environment_contains_path("LD_LIBRARY_PATH", path) for path in required_library_roots):
        missing_variables.append("LD_LIBRARY_PATH")
    if missing_variables:
        set_env_script = component_root / "set_env.bash"
        missing_text = ",".join(missing_variables)
        if any(module in sys.modules for module in ("torch", "torch_npu")):
            raise NativeComponentUnavailableError(
                "[HP-NATIVE-OPP-ACTIVATION-TOO-LATE] component=multicore "
                f"missing={missing_text}. Torch/torch_npu has already been imported; "
                f"exit the current Python process, run source {set_env_script}, and start a new process."
            )
        raise NativeComponentUnavailableError(
            "[HP-NATIVE-OPP-NOT-ACTIVATED] component=multicore "
            f"missing={missing_text}. Run source {set_env_script} before starting Python with multicore."
        )
    return vendor_root.resolve()


def get_multicore_paths() -> tuple[Path, Path]:
    """Return the vendor root and shared Torch adapter after ABI validation."""
    vendor_root = require_multicore_environment()
    adapter = _component_root() / "framework" / "torch" / "libhyper_multicore_torch.so"
    vendor_library = vendor_root / "op_api" / "lib" / "libcust_opapi.so"
    if not vendor_library.is_file() or not adapter.is_file():
        raise NativeComponentUnavailableError(
            "[HP-NATIVE-FRAMEWORK-TARGET-UNAVAILABLE] component=multicore framework=torch "
            f"root={_component_root()}. The current wheel/PYTHONPATH payload does not include an adapter for this "
            "framework; rebuild with bash build.sh and inspect the build log."
        )
    validate_native_compatibility()
    return vendor_root, adapter.resolve()


def validate_native_compatibility() -> None:
    """Reject a native build made for a different Torch/PTA, Python or C++ ABI.

    Reads the selected payload's generated identity without modifying the environment.
    Called before adapter or SHMEM binding loading, not at plain package import.

    Raises:
        NativeComponentUnavailableError: If identity is missing or incompatible.
    """
    info_path = _component_root().parent / "build_info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        actual = {"torch_version": metadata.version("torch"), "pta_version": metadata.version("torch-npu"),
                  "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
                  "host_arch": platform.machine()}
        import torch  # Native boundary only; interrogate the framework selected for this process.

        actual["torch_cxx11_abi"] = bool(torch._C._GLIBCXX_USE_CXX11_ABI)
        actual["torch_commit"] = getattr(torch.version, "git_version", None)
        spec = util.find_spec("torch_npu")
        if spec is None or spec.origin is None:
            raise RuntimeError("torch_npu package is unavailable")
        library = Path(spec.origin).resolve().parent / "lib/libtorch_npu.so"
        actual["libtorch_npu_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
        version_file = Path(os.environ["ASCEND_HOME_PATH"]) / "opp/version.info"
        actual["cann_version_sha256"] = hashlib.sha256(version_file.read_bytes()).hexdigest()
        mismatches = {key: {"built": info.get(key), "actual": value}
                      for key, value in actual.items() if key not in info or info[key] != value}
        if mismatches:
            raise RuntimeError(f"Native environment identity mismatch: {mismatches}")
    except (OSError, ValueError, KeyError, RuntimeError, metadata.PackageNotFoundError) as error:
        raise NativeComponentUnavailableError(f"[HYPER-MULTICORE-ABI-MISMATCH] {info_path}: {error}") from error


def preload_vendor_library(vendor_root: Path) -> None:
    """Load the exact ACLNN library without exposing its dependency symbols globally."""
    library_path = vendor_root / "op_api" / "lib" / "libcust_opapi.so"
    try:
        # CANN libraries can export identically named globals. Global promotion
        # can bind their independent destructors to the same object at exit.
        # ACLNN entry points are resolved through the vendor library handle.
        ctypes.CDLL(str(library_path), mode=ctypes.RTLD_LOCAL)
    except OSError as error:
        raise NativeComponentUnavailableError(
            f"[HP-NATIVE-VENDOR-LOAD-FAILED] library={library_path} error={error}."
        ) from error


def _component_root() -> Path:
    """Return the installed or source-build multicore payload root."""
    packaged = Path(__file__).resolve().parent / "lib"
    if packaged.is_dir():
        return packaged
    repository_root = Path(__file__).resolve().parents[1]
    if (repository_root / "setup.py").is_file():
        source_build = (
            repository_root / "build" / "native" / "payload" / "hyper_multicore" / "lib"
        )
        if source_build.is_dir():
            return source_build
    return packaged


def _environment_contains_path(variable: str, expected: Path) -> bool:
    """Return whether an environment path list contains ``expected``."""
    configured = [value for value in os.environ.get(variable, "").split(os.pathsep) if value]
    return str(expected.resolve()) in {str(Path(value).resolve()) for value in configured}
