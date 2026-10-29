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
    "lib/framework/torch/libhyper_multicore_torch.so",
    "lib/vendors/hyper_multicore_nn/op_api/lib/libcust_opapi.so",
    "shmem/lib/framework/torch/hyper_multicore_shmem_torch.so",
    "shmem/lib/libhyper_multicore_shmem_ops.so",
    "shmem/lib/shmem/libhyper_multicore_shmem.so",
    "shmem/lib/shmem/libhyper_multicore_shmem_utils.so",
    "shmem/lib/shmem/aclshmem_bootstrap_uid.so",
    "shmem/lib/shmem/aclshmem_bootstrap_config_store.so",
)
_VENDOR_OPERATORS = (
    "hyper_mega_moe", "hyper_mega_moe_grad", "hyper_mega_mhc", "hyper_mega_mhc_grad",
    "hyper_mega_gate_route", "hyper_mega_gate_route_grad",
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
                                                ".patch", ".json", ".ini", ".cmake", ".txt"}:
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


def _payload_hashes(root: Path) -> dict[str, str]:
    """Hash every packaged resource except the identity manifest itself."""
    return {path.relative_to(root).as_posix(): _sha256(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and path != root / "build_info.json"}


def _kernel_references(value):
    """Read CANN binary and per-operator index references, including nested binInfo."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"binPath", "jsonPath", "jsonFilePath"}:
                if not isinstance(item, str) or not item:
                    raise RuntimeError(f"Invalid kernel index reference: {key}={item!r}")
                yield item
            else:
                yield from _kernel_references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _kernel_references(item)


def _kernel_index_resources(kernel_root: Path, index: Path) -> list[Path]:
    """Require every indexed kernel variant and its metadata/object pair."""
    if not index.is_file():
        raise RuntimeError(f"Missing native payload resource: {index}")
    try:
        contents = json.loads(index.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise RuntimeError(f"Invalid kernel index: {index}") from error
    references = list(_kernel_references(contents))
    if not references:
        raise RuntimeError(f"Missing kernel index references: {index}")
    required = []
    for relative in references:
        path = kernel_root / relative
        if not path.resolve().is_relative_to(kernel_root.resolve()):
            raise RuntimeError(f"Kernel index reference outside payload: {relative}")
        required.append(path)
        if path.suffix == ".json":
            required.append(path.with_suffix(".o"))
    return required


def _validate_vendor_resources(root: Path, soc_list: list[str], host_arch: str) -> None:
    """Apply the vendor build's required runtime-resource checks at packaging time."""
    if not isinstance(soc_list, list) or not soc_list or any(
        soc not in {"ascend910b", "ascend910_93"} for soc in soc_list
    ):
        raise RuntimeError(f"Missing or unsupported payload SoC list: {soc_list}")
    if host_arch not in {"aarch64", "x86_64"}:
        raise RuntimeError(f"Missing or unsupported payload Host architecture: {host_arch}")
    vendor = root / "lib/vendors/hyper_multicore_nn/op_impl/ai_core/tbe"
    host_libraries = [vendor / "op_tiling/liboptiling.so",
                      vendor / f"op_tiling/lib/linux/{host_arch}/libcust_opmaster_rt2.0.so",
                      root / f"lib/vendors/hyper_multicore_nn/op_proto/lib/linux/{host_arch}/libcust_opsproto_rt2.0.so"]
    for soc in soc_list:
        indexes = [vendor / f"kernel/config/{soc}/binary_info_config.json"]
        required = [*host_libraries, vendor / f"config/{soc}/aic-{soc}-ops-info.json", *indexes]
        for operator in _VENDOR_OPERATORS:
            index = vendor / f"kernel/config/{soc}/{operator}.json"
            required.append(index)
            indexes.append(index)
            for suffix in (".o", ".json"):
                matches = sorted((vendor / f"kernel/{soc}/{operator}").glob(f"*{suffix}"))
                if not matches:
                    raise RuntimeError(f"Missing native payload resource: {soc}/{operator}/*{suffix}")
                required.extend(matches)
        for index in indexes:
            required.extend(_kernel_index_resources(vendor / "kernel", index))
        for path in required:
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing or empty native payload resource: {path.relative_to(root)}")


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
    adapters = sorted(path.name for path in (root / "lib/framework/torch").glob("*.so"))
    if adapters != ["libhyper_multicore_torch.so"]:
        raise RuntimeError(f"Expected only the shared Torch adapter, found: {adapters}")
    hashes = info.get("native_sha256", {})
    for path in root.rglob("*.so"):
        relative = path.relative_to(root).as_posix()
        if hashes.get(relative) != _sha256(path):
            raise RuntimeError(f"Unrecorded or changed native library: {relative}")
    for relative in REQUIRED_LIBRARIES:
        if relative not in hashes:
            raise RuntimeError(f"Missing native checksum: {relative}")
    _validate_vendor_resources(root, info.get("soc_list"), info.get("host_arch"))
    recorded = info.get("payload_sha256")
    if not isinstance(recorded, dict) or not recorded:
        raise RuntimeError("Missing complete payload checksum manifest: payload_sha256")
    actual = _payload_hashes(root)
    missing = sorted(recorded.keys() - actual.keys())
    unrecorded = sorted(actual.keys() - recorded.keys())
    if missing or unrecorded:
        raise RuntimeError(f"Payload file set mismatch: missing={missing}, unrecorded={unrecorded}")
    for relative, digest in actual.items():
        if recorded[relative] != digest:
            raise RuntimeError(f"Changed payload resource: {relative}")


def main() -> None:
    """Prepare identity before compilation, finalize its resources, or validate a payload."""
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
        info["payload_sha256"] = _payload_hashes(args.output)
        validate_payload(args.output, info)
        (args.output / "build_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    else:
        validate_payload(args.output, json.loads((args.output / "build_info.json").read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
