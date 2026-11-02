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
"""Build common native resources once and package explicit framework adapter pairs."""

from __future__ import annotations

import argparse
import base64
import csv
import ctypes
from email.parser import BytesParser
import io
import re
import zipfile
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import shutil
import shlex
import signal
import subprocess
import sys
import time
import tempfile

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "build/native"
HELPERS = runpy.run_path(str(ROOT / "hyper_multicore/_build/artifacts.py"))
FRAMEWORK = runpy.run_path(str(ROOT / "hyper_multicore/_build/framework_environments.py"))
MANIFEST = runpy.run_path(str(ROOT / "hyper_multicore/_native_manifest.py"))


CACHE_SCHEMA_VERSION = 2
COMMON_DIRECTORIES = (
    "hyper_multicore/ops", "hyper_multicore/cmake/vendor", "hyper_multicore/shmem/ccsrc/ops",
    "hyper_multicore/shmem/ccsrc/data_plane", "hyper_multicore/shmem/cmake",
)
COMMON_EXTRA_INPUTS = (
    "hyper_multicore/shmem/ccsrc/cann/device.h", "hyper_multicore/shmem/ccsrc/cann/signal.h",
    "hyper_multicore/build.sh", "hyper_multicore/set_env.bash", "hyper_multicore/shmem/build.sh",
    "hyper_multicore/shmem/_build/shmem_sdk.sh", "hyper_multicore/_build/assemble_multicore_source.py",
    "hyper_multicore/_build/merge_multicore_vendors.py", "hyper_multicore/_build/prepare_dependencies.py",
    "hyper_multicore/_build/dependencies.lock.json", "hyper_multicore/cmake/hardening.cmake",
    "scripts/native/cann_version.sh",
)


def common_source_fingerprint() -> str:
    """Include every consumed common source, including the packaged activation script."""
    return fingerprint_files(COMMON_DIRECTORIES, COMMON_EXTRA_INPUTS)


def source_observation() -> dict:
    """Freeze source identity and file generations; restoring old bytes is still a change."""
    generations = {}
    for directory in ("hyper_multicore", "scripts"):
        for path in (ROOT / directory).rglob("*"):
            relative = path.relative_to(ROOT)
            if "lib" in relative.parts or "__pycache__" in relative.parts or not path.is_file():
                continue
            stat = path.stat()
            generations[relative.as_posix()] = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    for name in ("setup.py", "MANIFEST.in", "build.sh", "requirements.txt", "tests/ut/shmem/native_log_test.cpp"):
        path = ROOT / name
        if path.is_file():
            stat = path.stat()
            generations[name] = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    return {"identity": HELPERS["_source_identity"](ROOT), "generations": generations}


class CacheTransaction:
    """Commit only unchanged stages and revoke this run's new markers on failure."""

    def __init__(self):
        self.frozen = source_observation()
        self.created = set()

    def __enter__(self):
        self.check()
        return self

    def check(self):
        """Reject source or file-generation changes before certifying an output."""
        if source_observation() != self.frozen:
            raise RuntimeError("Source inputs changed during build; refusing mixed-source cache or payload")

    def prepare(self, directory: Path):
        """Discard abandoned task outputs without touching any reusable stage."""
        self.check()
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        self.created.add(directory)

    def complete(self, directory: Path, record: dict):
        """Publish a stage marker only after output validation and a fresh input check."""
        self.check()
        write_json(directory / "complete.json", {**record, "cache_schema": CACHE_SCHEMA_VERSION})
        self.check()

    def __exit__(self, exception_type, _exception, _traceback):
        try:
            if exception_type is None:
                self.check()
        except BaseException:
            self.revoke()
            raise
        if exception_type is not None:
            self.revoke()

    def revoke(self):
        """Keep successful pre-existing caches; remove only this run's completion claims."""
        for directory in self.created:
            (directory / "complete.json").unlink(missing_ok=True)


def write_json(path: Path, value) -> None:
    """Write task records atomically in their existing parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def publish_wheel(source: Path, destination: Path) -> None:
    """Expose a complete wheel atomically, preserving any prior successful artifact."""
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        shutil.copy2(source, temporary)
        with temporary.open("rb") as file:
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def exchange_directories(first: Path, second: Path) -> None:
    """Exchange Linux directory entries without a missing-payload interval."""
    library = ctypes.CDLL(None, use_errno=True)
    rename = getattr(library, "renameat2", None)
    if rename is None:
        raise RuntimeError("Atomic native payload publication requires Linux renameat2")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(first), -100, os.fsencode(second), 2) != 0:  # AT_FDCWD, RENAME_EXCHANGE
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(second))


def publish_payload(candidate: Path, selected: Path) -> None:
    """Keep the visible payload complete even if interrupted immediately after publication."""
    try:
        if selected.is_symlink():
            raise ValueError(f"Refusing to replace an external payload symlink: {selected}")
        if selected.exists() and not selected.is_dir():
            raise ValueError(f"Native payload destination is not a directory: {selected}")
        if selected.exists():
            exchange_directories(candidate, selected)
        else:
            os.replace(candidate, selected)
    finally:
        if candidate.exists():
            shutil.rmtree(candidate)


def fingerprint_files(directories, extra=()) -> str:
    """Hash only consumed inputs rather than the entire framework matrix/source tree."""
    paths = set()
    for relative in directories:
        path = ROOT / relative
        paths.update(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    paths.update(ROOT / p for p in extra)
    return FRAMEWORK["digest_json"]({p.relative_to(ROOT).as_posix(): MANIFEST["file_sha256"](p)
                                     for p in sorted(paths)})


def compiler_metrics(output: str) -> dict:
    """Count logged compiler commands and separately expose observable CMake/OPC actions."""
    commands = 0
    for line in output.splitlines():
        if not re.search(r'(?:^|\s)-c(?:\s|$)', line):
            continue
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        if '-c' in tokens and any(re.search(r'(?:^|-)(?:g\+\+|c\+\+|clang\+\+|bisheng)(?:-\d+(?:\.\d+)*)?$', Path(token).name) for token in tokens):
            commands += 1
    object_actions = len(re.findall(r'Building (?:C|CXX) object ', output))
    return {'compiler_invocations': max(commands, object_actions),
            'compiler_commands_logged': commands,
            'host_object_compile_actions': object_actions,
            'kernel_compile_jobs': output.count('Opc tool start working now')}


class Commands:
    """Capture subprocess evidence and terminate the complete task group on interruption."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.events = []

    def __call__(self, command, *, label: str, env=None, capture=False):
        started = time.monotonic()
        log = self.directory / f"{len(self.events):03d}-{label}.log"
        print(f"[{label}] log={log}", flush=True)
        with log.open("w") as output:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE if capture else output,
                                       stderr=output, text=True, start_new_session=True)
            try:
                stdout, _ = process.communicate()
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
                # The leader can exit while a grandchild still holds the process group.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    remaining = []
                    for stat in Path("/proc").glob("[0-9]*/stat"):
                        try:
                            fields = stat.read_text().rsplit(")", 1)[1].split()
                            if fields[0] != "Z" and int(fields[2]) == process.pid:
                                remaining.append(stat)
                        except (OSError, ValueError, IndexError):
                            continue
                    if not remaining:
                        break
                    time.sleep(0.05)
                raise
        record = {"stage": label, "elapsed_seconds": time.monotonic() - started,
                  "exit_code": process.returncode, "log": str(log), "command": list(command),
                  "disk_free_bytes": shutil.disk_usage(NATIVE).free}
        record.update(compiler_metrics(log.read_text()))
        if capture:
            log.write_text((stdout or "") + log.read_text())
        self.events.append(record)
        write_json(self.directory / "stages.json", self.events)
        if process.returncode:
            raise RuntimeError(f"Stage {label} failed ({process.returncode}); inspect {log}")
        return stdout


def probe_identity(python: Path, socs: str, run: Commands) -> dict:
    """Collect actual target metadata in a separate interpreter; controller imports no Torch."""
    code = "import json,runpy,sys; from pathlib import Path; h=runpy.run_path(sys.argv[1]); print(json.dumps(h['build_identity'](Path(sys.argv[2]),sys.argv[3])))"
    result = run([str(python), "-c", code, str(ROOT / "hyper_multicore/_build/artifacts.py"), str(ROOT), socs],
                 label=f"probe-{python.parent.parent.name}", env=FRAMEWORK["isolated_environment"](python), capture=True)
    identity = json.loads(result)
    identity["torch_release"] = MANIFEST["release"](identity["torch_version"])
    identity["pta_release"] = MANIFEST["release"](identity["pta_version"])
    versions = [line.partition("=")[2].strip() for line in identity["cann_version_info"].splitlines()
                if line.partition("=")[0].strip() == "Version"]
    if versions != ["9.2.0"]:
        raise ValueError(f"Builds require CANN 9.2.0, found {versions}")
    identity["cann_version"] = versions[0]
    return identity


def hashes(root: Path) -> dict:
    """Hash every resource in a completed stage, excluding its external completion record."""
    return {p.relative_to(root).as_posix(): MANIFEST["file_sha256"](p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def compiler_identity(value: str, run: Commands, label: str) -> dict:
    """Record an explicitly selected compiler without changing the caller environment."""
    command = shlex.split(value)
    if not command or shutil.which(command[0]) is None:
        raise ValueError(f"Missing selected C++ compiler: {command}")
    command[0] = str(Path(shutil.which(command[0])).resolve())
    return {"command": command, "version": run([*command, "--version"], label=label, capture=True)}


def adapter_environment(python: Path) -> dict:
    """Allow newer framework compilers without changing the CANN vendor toolchain."""
    env = FRAMEWORK["isolated_environment"](python)
    if "HYPER_MULTICORE_FRAMEWORK_CXX" in env:
        env["CXX"] = env["HYPER_MULTICORE_FRAMEWORK_CXX"]
    return env


def compiler_inputs(env: dict) -> dict:
    """Include effective compile, header, library and tool-Python search settings in task keys."""
    return {key: env.get(key) for key in ("CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS",
        "CMAKE_BUILD_TYPE", "CMAKE_PREFIX_PATH", "CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH", "PYTHONPATH")}


def preflight_fingerprint(identity: dict, python: Path, compiler: dict, env: dict) -> str:
    """Reuse interface probes when their actual inputs, rather than unrelated commits, match."""
    return FRAMEWORK["digest_json"]({
        "framework": {k: identity[k] for k in ("torch_version", "pta_version", "torch_cxx11_abi", "python_abi",
            "host_arch", "torch_headers_sha256", "pta_headers_sha256", "torch_libraries_sha256",
            "libtorch_npu_sha256", "cann_root", "cann_version_sha256")},
        "source": fingerprint_files(("hyper_multicore/cmake/framework_probe",),
            ("hyper_multicore/cmake/torch_compat.cmake", "hyper_multicore/torch/csrc/cached_op_api.h")),
        "python": str(python), "compiler": compiler,
        "flags": {k: env.get(k) for k in ("CXX", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "CMAKE_PREFIX_PATH",
                                         "CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH")}})


def cache_valid(directory: Path, key: str) -> bool:
    """Never reuse an incomplete or changed native stage."""
    marker = directory / "complete.json"
    if not marker.exists():
        return False
    record = json.loads(marker.read_text())
    if record.get("cache_schema") != CACHE_SCHEMA_VERSION:
        return False
    if record.get("fingerprint") != key or record.get("hashes") != hashes(directory / "payload"):
        raise ValueError(f"Native cache integrity mismatch: {directory}")
    if "sdk_root" in record:
        sdk = Path(record["sdk_root"])
        handoff = directory / "sdk.json"
        if not handoff.is_file() or MANIFEST["file_sha256"](handoff) != record.get("sdk_manifest_sha256"):
            raise ValueError(f"Changed SDK handoff: {directory}")
        if not sdk.is_dir() or hashes(sdk) != record.get("sdk_hashes"):
            raise ValueError(f"Missing or changed SDK inputs: {sdk}")
    return True


def cmake_pair(python: Path, identity: dict, common: Path, sdk: Path, key: str, args, run: Commands,
               transaction=None) -> tuple[Path, dict]:
    """Guard both standalone stage calls and the encompassing build transaction."""
    with transaction or CacheTransaction() as guard:
        return _cmake_pair(python, identity, common, sdk, key, args, run, guard)


def _cmake_pair(python: Path, identity: dict, common: Path, sdk: Path, key: str, args,
                run: Commands, transaction: CacheTransaction) -> tuple[Path, dict]:
    """Build one binding/runtime and one unified adapter with exactly the target framework."""
    directory = NATIVE / "variants" / key
    if args.clean and directory.exists():
        shutil.rmtree(directory)
    identifier = "pt" + identity["pta_version"].replace("+", ".") + "-" + key[:12]
    paths = {"torch_so": f"adapters/{identifier}/libhyper_multicore_torch.so",
             "shmem_so": f"adapters/{identifier}/hyper_multicore_shmem_torch.so"}
    cached = cache_valid(directory, key)
    run.events.append({"stage": identifier, "cache_hit": cached, "fingerprint": key})
    if cached:
        previous = json.loads((directory / "identity.json").read_text())
        if previous.get("task_fingerprint") != key or previous.get("id") != identifier:
            raise ValueError(f"Changed cached adapter identity: {directory}")
        return directory / "payload", {**identity, **paths, "id": identifier, "task_fingerprint": key,
                                       "native_build_identity": previous["native_build_identity"]}
    transaction.prepare(directory)
    payload = directory / "payload"
    payload.mkdir(parents=True, exist_ok=True)
    env = adapter_environment(python)
    env.update(SHMEM_HOME_PATH=str(sdk), SHMEM_SOURCE_DIR=str(sdk / "shmem"))
    env["LD_LIBRARY_PATH"] = os.pathsep.join([str(common / "shmem/lib"), str(common / "shmem/lib/shmem"),
                                             env.get("LD_LIBRARY_PATH", "")])
    for kind, source, extra in (
        ("shmem", "hyper_multicore/shmem/ccsrc", [f"-DSHMEM_OPS_LIB={common / 'shmem/lib/libhyper_multicore_shmem_ops.so'}"]),
        ("torch", "hyper_multicore/torch", [f"-DHP_MULTICORE_VENDOR_ROOT={common / 'lib/vendors/hyper_multicore_nn'}"]),
    ):
        work = directory / kind
        run(["cmake", "-S", str(ROOT / source), "-B", str(work), "-DCMAKE_BUILD_TYPE=Release",
             "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", f"-DPython3_EXECUTABLE={python}", f"-DCMAKE_INSTALL_PREFIX={payload}", *extra],
            label=f"configure-{identifier}-{kind}", env=env)
        run(["cmake", "--build", str(work), "--parallel", str(args.jobs), "--verbose"], label=f"compile-{identifier}-{kind}", env=env)
        if kind == "shmem":
            run(["ctest", "--test-dir", str(work), "--output-on-failure"], label=f"ctest-{identifier}", env=env)
        run(["cmake", "--install", str(work)], label=f"install-{identifier}-{kind}", env=env)
    for name in ("libhyper_multicore_torch.so", "hyper_multicore_shmem_torch.so"):
        validate_elf(payload / name, identity["host_arch"], run, env)
    record = {**identity, **paths, "id": identifier, "task_fingerprint": key,
              "native_build_identity": {k: identity[k] for k in ("build_commit", "source_sha256", "build_fingerprint")}}
    write_json(directory / "identity.json", record)
    transaction.complete(directory, {"fingerprint": key, "hashes": hashes(payload)})
    return payload, record


def validate_elf(path: Path, arch: str, run: Commands, env: dict) -> None:
    """Keep target architecture, RPATH, private dependencies and ELF hardening explicit."""
    data = run(["readelf", "-hW", "-lW", "-dW", str(path)], label=f"elf-{path.name}", env=env, capture=True)
    expected = "AArch64" if arch == "aarch64" else "Advanced Micro Devices X86-64"
    if expected not in data or "GNU_RELRO" not in data or "BIND_NOW" not in data:
        raise ValueError(f"Missing architecture or ELF hardening: {path}")
    if re.search(r"GNU_STACK[^\n]*RWE", data):
        raise ValueError(f"Executable stack is forbidden: {path}")
    if "(RPATH)" in data or "(RUNPATH)" in data or "Shared library: [libcust_opapi.so]" in data:
        raise ValueError(f"Forbidden adapter runtime dependency/path: {path}")
    if "Shared library: [libshmem" in data:
        raise ValueError(f"Generic SHMEM dependency: {path}")
    linked = run(["ldd", str(path)], label=f"dependencies-{path.name}", env=env, capture=True)
    missing = [line.strip().split()[0] for line in linked.splitlines() if "not found" in line]
    if any(not name.startswith(("libtorch", "libc10")) for name in missing):
        raise ValueError(f"Missing adapter dependencies: {missing}")


def validate_wheel(wheel: Path, payload: Path, info: dict) -> None:
    """Verify archive membership, RECORD, identities and native payload before publication."""
    with zipfile.ZipFile(wheel) as archive:
        names, record_path, metadata_path = wheel_identity_members(archive)
        validate_wheel_record(archive, record_path, names)
        validate_wheel_metadata(archive, metadata_path, info)
        validate_wheel_payload(archive, names, payload, info)


def wheel_identity_members(archive):
    """Require unique safe entries and exactly one distribution identity."""
    names = archive.namelist()
    if len(names) != len(set(names)) or any(Path(n).is_absolute() or ".." in Path(n).parts for n in names):
        raise ValueError("Wheel contains duplicate or unsafe entries")
    records = [n for n in names if len(Path(n).parts) == 2 and n.endswith(".dist-info/RECORD")]
    metas = [n for n in names if len(Path(n).parts) == 2 and n.endswith(".dist-info/METADATA")]
    if len(records) != 1 or len(metas) != 1:
        raise ValueError("Wheel requires unique RECORD and METADATA")
    return names, records[0], metas[0]


def validate_wheel_record(archive, record_path, names):
    """Require RECORD to cover every archive member with its correct size and digest."""
    entries = list(csv.reader(io.StringIO(archive.read(record_path).decode())))
    if len(entries) != len({row[0] for row in entries}) or {row[0] for row in entries} != set(names):
        raise ValueError("Wheel RECORD membership mismatch")
    for name, checksum, size in entries:
        if name == record_path:
            continue
        contents = archive.read(name)
        expected = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(contents).digest()).rstrip(b"=").decode()
        if checksum != expected or size != str(len(contents)):
            raise ValueError(f"Wheel RECORD mismatch: {name}")


def validate_wheel_metadata(archive, metadata_path, info):
    """Keep the component version and default/full Torch dependency rules unchanged."""
    metadata = BytesParser().parsebytes(archive.read(metadata_path))
    if metadata["Name"].replace("_", "-") != "hyper-multicore" or metadata["Version"] != info["artifact_version"]:
        raise ValueError("Wheel distribution identity mismatch")
    from packaging.requirements import Requirement

    torch = [Requirement(v) for v in metadata.get_all("Requires-Dist", []) if Requirement(v).name.lower() == "torch"]
    if len(torch) != 1 or (info["build_mode"] == "all" and str(torch[0].specifier)):
        raise ValueError("Incorrect full-matrix Torch dependency")
    if info["build_mode"] == "current" and str(torch[0].specifier) != "==" + info["torch_version"]:
        raise ValueError("Incorrect default-wheel Torch dependency")


def validate_wheel_payload(archive, names, payload, info):
    """Check complete native membership, resource hashes and assembled identity bytes."""
    expected = {"hyper_multicore/" + n for n in info["payload_sha256"]}
    actual = {n for n in names if any(n.startswith("hyper_multicore/" + prefix + "/") for prefix in ("lib", "shmem/lib", "adapters"))}
    if actual != expected:
        raise ValueError("Wheel native payload membership mismatch")
    for name, checksum in info["payload_sha256"].items():
        if hashlib.sha256(archive.read("hyper_multicore/" + name)).hexdigest() != checksum:
            raise ValueError(f"Wheel payload mismatch: {name}")
    for name in ("build_info.json", "adapters/manifest.json"):
        if archive.read("hyper_multicore/" + name) != (payload / name).read_bytes():
            raise ValueError(f"Wheel identity mismatch: {name}")


def validate_toolchain() -> None:
    """Reject unavailable common build inputs before preparing any framework environment."""
    cann = os.environ.get("ASCEND_HOME_PATH", "")
    if not cann:
        raise ValueError("ASCEND_HOME_PATH is missing; source the CANN 9.2.0 set_env.sh before building")
    version_file = Path(cann) / "opp/version.info"
    versions = [line.partition("=")[2].strip() for line in version_file.read_text().splitlines()
                if line.partition("=")[0].strip() == "Version"]
    if versions != ["9.2.0"]:
        raise ValueError(f"Builds require CANN 9.2.0, found {versions}")
    required = ("cmake", "ctest", "gcc", "g++", "make", "git", "bisheng", "asc_opc", "readelf", "ldd", "ld")
    missing = [name for name in required if shutil.which(name) is None]
    if missing:
        raise ValueError(f"Missing native build tools: {missing}")


def build(args) -> Path:
    """Keep every new stage provisional until the complete build returns successfully."""
    with CacheTransaction() as transaction:
        return _build(args, transaction)


def _build(args, transaction: CacheTransaction) -> Path:
    """Coordinate task phases without duplicating cache, validation or publication policy."""
    validate_toolchain()
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    run = Commands(NATIVE / "logs" / run_id)
    targets, pythons, identities, host_cxx = prepare_frameworks(args, run)
    common_key, common_env = common_inputs(args, run, pythons[0], identities[0])
    common, sdk = build_common(args, run, pythons[0], common_key, common_env, transaction)
    outputs, records = build_adapters(args, run, pythons, identities, host_cxx, common_key, common, sdk, transaction)
    payload, info = assemble_payload(args, run_id, identities, outputs, records, common, common_key)
    transaction.check()
    if not args.native_only:
        package_wheel(pythons[0], run, payload, info, transaction, run_id)
    selected = publish_build_payload(payload, run_id, transaction)
    if args.all:
        removed = FRAMEWORK["prune_environments"](NATIVE, pythons, {t["pta_release"] for t in targets})
        run.events.append({"stage": "prune-framework-environments", "removed": removed})
    write_json(run.directory / "stages.json", run.events)
    print(f"Mode: {info['build_mode']}; adapters: {[r['pta_version'] for r in records]}; payload: {selected}", flush=True)
    return selected


def prepare_frameworks(args, run: Commands):
    """Resolve the target interpreters and validate their actual identities and interfaces."""
    targets = FRAMEWORK["load_targets"](ROOT / "hyper_multicore/_build/frameworks.lock.json")
    base = Path(sys.executable).absolute()
    if args.all:
        for target in targets:
            FRAMEWORK["validate_target"](target)
        pythons = [FRAMEWORK["prepare_environment"](t, NATIVE, run, base) for t in targets]
    else:
        pythons = [base]
    identities = [probe_identity(p, args.soc_list, run) for p in pythons]
    host_cxx = compiler_identity(os.environ.get("HYPER_MULTICORE_FRAMEWORK_CXX", os.environ.get("CXX", "c++")),
                                 run, "selected-framework-cxx-version")
    for identity in identities:
        identity["host_cxx"] = host_cxx
    for identity in identities:
        if not any(t["torch_release"] == identity["torch_release"] and t["pta_release"] == identity["pta_release"] for t in targets):
            raise ValueError(f"No supported build branch: {identity['torch_version']}/{identity['pta_version']}")
    if args.all:
        for target, identity in zip(targets, identities):
            if identity["torch_version"] != target["torch_version"] or identity["pta_version"] != target["pta_build_version"]:
                raise ValueError(f"Target framework identity changed: {target['id']}")
    preflight_frameworks(args, run, pythons, identities, host_cxx)
    return targets, pythons, identities, host_cxx


def preflight_frameworks(args, run: Commands, pythons, identities, host_cxx):
    """Run real target interface probes before expensive common-kernel work."""
    for python, identity in zip(pythons, identities):
        env = adapter_environment(python)
        preflight_key = preflight_fingerprint(identity, python, host_cxx, env)
        work = NATIVE / "preflight" / preflight_key
        run(["cmake", "-S", str(ROOT / "hyper_multicore/cmake/framework_probe"), "-B", str(work),
             f"-DPython3_EXECUTABLE={python}", "-DCMAKE_BUILD_TYPE=Release"], label=f"preflight-configure-{identity['pta_version']}", env=env)
        run(["cmake", "--build", str(work), "--parallel", str(args.jobs)], label=f"preflight-compile-{identity['pta_version']}", env=env)


def common_inputs(args, run: Commands, python: Path, identity: dict):
    """Collect only source, toolchain and compiler-Python inputs consumed by common tasks."""
    common_source = common_source_fingerprint()
    compiler = {name: {"path": str(Path(shutil.which(name)).resolve()), "version": run([name, "--version"], label=f"toolchain-{name}", capture=True)}
                for name in ("gcc", "g++", "bisheng", "cmake", "ld", "make")}
    compiler["selected_cxx"] = compiler_identity(os.environ.get("CXX", "c++"), run, "selected-common-cxx-version")
    common_env = FRAMEWORK["cann_tool_environment"](python)
    tool_packages = run([str(python), "-c", "from importlib.metadata import version; import json; print(json.dumps({n:version(n) for n in ('numpy','decorator','attrs','psutil')}))"],
                        label="cann-python-packages", env=FRAMEWORK["isolated_environment"](python), capture=True)
    cann = Path(common_env["ASCEND_HOME_PATH"])
    probe = cann / "compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/util/ascendc_compile_kernel.py"
    if not probe.is_file():
        probe = cann / (platform.machine() + "-linux/tikcpp/ascendc_kernel_cmake/fwk_modules/util/ascendc_compile_kernel.py")
    if not probe.is_file():
        raise ValueError(f"Missing CANN ASC compilation entry point: {probe}")
    run([str(python), str(probe), "--help"], label="cann-python-import-probe", env=common_env)
    common_key = FRAMEWORK["digest_json"]({"cache_schema": CACHE_SCHEMA_VERSION, "source": common_source, "socs": args.soc_list, "compiler_python_packages": json.loads(tool_packages),
        "cann_root": identity["cann_root"], "cann_sha": identity["cann_version_sha256"], "compiler": compiler, "asc_opc": MANIFEST["file_sha256"](Path(shutil.which("asc_opc")).resolve()),
        "flags": compiler_inputs(common_env)})
    return common_key, common_env


def build_common(args, run: Commands, python: Path, common_key: str, common_env: dict,
                 transaction: CacheTransaction):
    """Prepare one validated common payload and its immutable SDK handoff."""
    common_dir = NATIVE / "common" / common_key
    if args.clean and common_dir.exists():
        shutil.rmtree(common_dir)
    common = common_dir / "payload"
    hit = cache_valid(common_dir, common_key)
    run.events.append({"stage": "common", "cache_hit": hit, "fingerprint": common_key})
    if not hit:
        transaction.prepare(common_dir)
        env = dict(common_env)
        env.update(PYTHON_BIN=str(python), HP_NATIVE_STAGE="common", HP_COMPONENT_ROOT=str(common),
                   HP_NATIVE_WORK_ROOT=str(common_dir / "work"), HP_SHMEM_WORK_ROOT=str(common_dir / "shmem-work"),
                   HP_SHMEM_HANDOFF=str(common_dir / "sdk.json"))
        run(["bash", str(ROOT / "hyper_multicore/build.sh"), "--stage-only", "--soc-list", args.soc_list,
             "--jobs", str(args.jobs), *(["--clean"] if args.clean else [])], label="common-resources", env=env)
        sdk_record = json.loads((common_dir / "sdk.json").read_text())
        transaction.complete(common_dir, {"fingerprint": common_key, "hashes": hashes(common),
                   "sdk_root": sdk_record["install_root"], "sdk_hashes": hashes(Path(sdk_record["install_root"])),
                   "sdk_manifest_sha256": MANIFEST["file_sha256"](common_dir / "sdk.json")})
    sdk = Path(json.loads((common_dir / "sdk.json").read_text())["install_root"])
    return common, sdk


def build_adapters(args, run: Commands, pythons, identities, host_cxx, common_key, common, sdk,
                   transaction: CacheTransaction):
    """Build or reuse exactly the requested framework pairs."""
    adapter_source = fingerprint_files(("hyper_multicore/torch/csrc", "hyper_multicore/shmem/ccsrc/runtime",
        "hyper_multicore/shmem/ccsrc/cann", "hyper_multicore/shmem/ccsrc/bindings", "hyper_multicore/shmem/ccsrc/include", "hyper_multicore/shmem/ccsrc/device"),
        ("hyper_multicore/torch/CMakeLists.txt", "hyper_multicore/shmem/ccsrc/CMakeLists.txt", "hyper_multicore/cmake/torch_compat.cmake",
         "hyper_multicore/cmake/hardening.cmake", "tests/ut/shmem/native_log_test.cpp"))
    records, outputs = [], []
    for python, identity in zip(pythons, identities):
        transaction.check()
        key = FRAMEWORK["digest_json"]({"cache_schema": CACHE_SCHEMA_VERSION, "common": common_key, "source": adapter_source, "python": str(python),
            "compiler": host_cxx,
            "common_path": str(common),
            "flags": compiler_inputs(adapter_environment(python)),
            "framework": {k: identity[k] for k in ("torch_version", "pta_version", "torch_commit", "torch_cxx11_abi", "python_abi", "host_arch", "op_plugin_headers_sha256", "libtorch_npu_sha256", "torch_headers_sha256", "pta_headers_sha256", "torch_libraries_sha256")}})
        output, record = cmake_pair(python, identity, common, sdk, key, args, run, transaction)
        outputs.append(output)
        records.append(record)
    return outputs, records


def assemble_payload(args, run_id, identities, outputs, records, common, common_key):
    """Assemble and validate one selected payload without changing runtime registration policy."""
    staging = NATIVE / "staging" / run_id
    payload = staging / "hyper_multicore"
    shutil.copytree(common, payload)
    for output, record in zip(outputs, records):
        shutil.copytree(output, payload / "adapters" / record["id"])
    manifest = {"schema_version": 1, "python_abi": identities[0]["python_abi"], "host_arch": identities[0]["host_arch"],
                "variants": [{k: record[k] for k in ("id", "torch_release", "pta_release", "torch_cxx11_abi", "torch_so", "shmem_so")} for record in records],
                "validated_runtime_pairs": [], "known_runtime_issues": []}
    write_json(payload / "adapters/manifest.json", manifest)
    info = {**identities[0], "schema_version": 2, "build_mode": "all" if args.all else "current", "variants": records,
            "artifact_version": HELPERS["COMPONENT_VERSION"] if args.all else HELPERS["artifact_version"](identities[0]["pta_version"]),
            "requested_target_ids": [v["id"] for v in records], "common_fingerprint": common_key,
            "framework_lock_sha256": MANIFEST["file_sha256"](ROOT / "hyper_multicore/_build/frameworks.lock.json") if args.all else None}
    info["native_sha256"] = {p.relative_to(payload).as_posix(): MANIFEST["file_sha256"](p) for p in payload.rglob("*.so")}
    info["payload_sha256"] = HELPERS["_payload_hashes"](payload)
    HELPERS["validate_payload"](payload, info)
    write_json(payload / "build_info.json", info)
    return payload, info


def package_wheel(python: Path, run: Commands, payload: Path, info: dict,
                  transaction: CacheTransaction, run_id: str):
    """Validate the complete wheel and source state before atomically publishing its path."""
    env = FRAMEWORK["isolated_environment"](python)
    env["HYPER_MULTICORE_NATIVE_OUTPUT_ROOT"] = str(payload)
    wheel_dir = payload.parent / "wheel"
    run([str(python), "setup.py", "-q", "build", "--build-base", str(payload.parent / "wheel-build"),
         "bdist_wheel", "--bdist-dir", str(payload.parent / "bdist"), "--dist-dir", str(wheel_dir)], label="package-wheel", env=env)
    wheels = list(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Expected exactly one native wheel")
    validate_wheel(wheels[0], payload, info)
    transaction.check()
    destination = ROOT / "dist" / wheels[0].name
    destination.parent.mkdir(exist_ok=True)
    if destination.exists() and MANIFEST["file_sha256"](destination) != MANIFEST["file_sha256"](wheels[0]):
        destination = destination.parent / run_id / destination.name
        destination.parent.mkdir()
    publish_wheel(wheels[0], destination)
    print(f"Wheel: {destination}\nSHA256: {MANIFEST['file_sha256'](destination)}", flush=True)


def publish_build_payload(payload: Path, run_id: str, transaction: CacheTransaction) -> Path:
    """Expose a complete selected payload after the final source check."""
    selected = NATIVE / "payload/hyper_multicore"
    selected.parent.mkdir(parents=True, exist_ok=True)
    candidate = selected.with_name(f".payload-{run_id}")
    shutil.copytree(payload, candidate)
    transaction.check()
    publish_payload(candidate, selected)
    return selected


def main() -> None:
    """Expose default local-framework builds and the explicit complete release matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Build every locked framework target; keep native caches")
    parser.add_argument("--soc-list", default="ascend910b,ascend910_93")
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--native-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    socs = args.soc_list.split(",")
    if args.jobs < 1 or not socs or len(set(socs)) != len(socs) or any(s not in ("ascend910b", "ascend910_93") for s in socs):
        parser.error("--jobs must be positive and --soc-list must contain unique supported CANN SoCs")
    if platform.system() != "Linux":
        parser.error("Native builds require Linux/CANN/PTA")
    NATIVE.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with (NATIVE / ".build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            build(args)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            parser.exit(1, f"Native build failed: {error}\n")


if __name__ == "__main__":
    main()
