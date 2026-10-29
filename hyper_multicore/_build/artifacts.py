# Copyright 2026 Huawei Technologies Co., Ltd.
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
"""Build-time artifact identity and payload validation; no runtime package import."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from packaging.version import Version

COMPONENT_VERSION = "0.1.0.dev0"
REQUIRED_LIBRARIES = (
    "lib/framework/torch/libhyper_multicore_mega_moe_torch.so",
    "lib/framework/torch/libhyper_multicore_mega_gate_torch.so",
    "lib/vendors/hyper_multicore_nn/op_api/lib/libcust_opapi.so",
    "shmem/lib/framework/torch/hyper_multicore_shmem_torch.so",
    "shmem/lib/libhyper_multicore_shmem_ops.so",
    "shmem/lib/shmem/libhyper_multicore_shmem.so",
    "shmem/lib/shmem/libhyper_multicore_shmem_utils.so",
    "shmem/lib/shmem/aclshmem_bootstrap_uid.so",
    "shmem/lib/shmem/aclshmem_bootstrap_config_store.so",
)


def artifact_version(pta_version: str) -> str:
    """Return the component version tagged with the actual PTA distribution version.

    Args:
        pta_version: Full installed torch-npu version, preserving post releases.

    Returns:
        PEP 440 artifact version. A PTA local version is flattened into local-label segments.

    Raises:
        ValueError: If the version is empty or cannot be represented.
    """
    if not pta_version.strip():
        raise ValueError("torch-npu version is required for native artifacts")
    parsed = Version(pta_version)
    label = str(parsed).replace("+", ".").replace("!", ".")
    return str(Version(f"{COMPONENT_VERSION}+pt.{label}"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_identity(root: Path) -> dict:
    inputs = {}
    for directory in ("hyper_multicore", "scripts"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".cpp", ".cc", ".h", ".hpp", ".sh", ".bash",
                                                ".patch", ".json", ".cmake", ".txt"}:
                if any(part in {"lib", "__pycache__"} for part in path.relative_to(root).parts):
                    continue
                inputs[path.relative_to(root).as_posix()] = _sha256(path)
    for name in ("setup.py", "MANIFEST.in", "build.sh", "requirements.txt"):
        inputs[name] = _sha256(root / name)
    build_commit = None
    if (root / ".git").exists():
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=False, capture_output=True)
        if result.returncode == 0:
            build_commit = result.stdout.decode().strip()
    return {"build_commit": build_commit,
            "source_sha256": hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()}


def build_identity(root: Path, soc_list: str) -> dict:
    """Read the active native toolchain identity without downloading or compiling.

    Args:
        root: Standalone repository or unpacked sdist root.
        soc_list: Actual selected CANN SoC list.

    Returns:
        Framework, toolchain and source identity for this build.

    Raises:
        RuntimeError: If Python/PTA/CANN or required op-plugin inputs are missing.
    """
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        raise RuntimeError("Native builds require Python >=3.10,<3.13")
    pta_version = metadata.version("torch-npu")
    version = artifact_version(pta_version)
    import torch  # Build-time ABI interrogation; do not import the component runtime.

    spec = util.find_spec("torch_npu")
    if spec is None or spec.origin is None:
        raise RuntimeError("torch_npu package is required")
    pta_root = Path(spec.origin).resolve().parent
    headers_root = pta_root / "include" / "third_party" / "op-plugin"
    if not (headers_root / "op_plugin/include/npu_cpp_extension.h").is_file():
        raise RuntimeError(f"Missing installed op-plugin header: {headers_root}")
    headers = {p.relative_to(headers_root).as_posix(): _sha256(p)
               for p in sorted(headers_root.rglob("*")) if p.is_file()}
    cann_root = Path(os.environ["ASCEND_HOME_PATH"])
    cann_version = cann_root / "opp/version.info"
    info = {
        "schema_version": 1, "component_version": COMPONENT_VERSION, "artifact_version": version,
        "pta_version": pta_version, "torch_version": metadata.version("torch"),
        "torch_commit": getattr(torch.version, "git_version", None),
        "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "host_arch": platform.machine(), "soc_list": soc_list.split(","),
        "cann_root": str(cann_root.resolve()), "cann_version_info": cann_version.read_text(),
        "cann_version_sha256": _sha256(cann_version),
        "op_plugin_headers_sha256": hashlib.sha256(json.dumps(headers, sort_keys=True).encode()).hexdigest(),
        "libtorch_npu_sha256": _sha256(pta_root / "lib/libtorch_npu.so"),
        "dependency_lock_sha256": _sha256(root / "hyper_multicore/_build/dependencies.lock.json"),
        **_source_identity(root),
    }
    info["build_fingerprint"] = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()
    return info


def validate_payload(root: Path, info: dict) -> None:
    """Require the complete native payload and its recorded file hashes.

    Args:
        root: Prepared hyper_multicore payload directory.
        info: Its generated build_info object.

    Raises:
        RuntimeError: If identity, required files or native checksums are invalid.
    """
    if info["artifact_version"] != artifact_version(info["pta_version"]):
        raise RuntimeError("Payload version does not match its PTA identity")
    for relative in (*REQUIRED_LIBRARIES, "lib/set_env.bash"):
        path = root / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing native payload file: {relative}")
    hashes = info.get("native_sha256", {})
    for path in root.rglob("*.so"):
        relative = path.relative_to(root).as_posix()
        if hashes.get(relative) != _sha256(path):
            raise RuntimeError(f"Unrecorded or changed native library: {relative}")
    for relative in REQUIRED_LIBRARIES:
        if relative not in hashes:
            raise RuntimeError(f"Missing native checksum: {relative}")


def main() -> None:
    """Prepare identity before compilation, finalize its libraries, or validate a payload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--soc-list", default="ascend910b,ascend910_93")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--finalize", type=Path, metavar="IDENTITY_JSON")
    action.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        info = build_identity(Path(__file__).resolve().parents[2], args.soc_list)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        print(info["build_fingerprint"])
    elif args.finalize:
        info = json.loads(args.finalize.read_text(encoding="utf-8"))
        info["native_sha256"] = {p.relative_to(args.output).as_posix(): _sha256(p)
                                 for p in args.output.rglob("*.so")}
        validate_payload(args.output, info)
        (args.output / "build_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    else:
        validate_payload(args.output, json.loads((args.output / "build_info.json").read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
