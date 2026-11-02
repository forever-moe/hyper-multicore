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
"""Locked wheel preparation and project-owned, content-addressed framework environments."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import urllib.parse
import zipfile
from email.parser import BytesParser


def digest_json(value) -> str:
    """Return a deterministic input identity."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_targets(path: Path) -> list[dict]:
    """Read the exact current-platform target list; never filter unavailable targets."""
    lock = json.loads(path.read_text())
    if lock.get("schema_version") != 1:
        raise ValueError("Unsupported framework lock schema")
    abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
    groups = [p for p in lock["platforms"] if p["python_abi"] == abi and p["host_arch"] == platform.machine()]
    if len(groups) != 1 or not groups[0]["targets"]:
        raise ValueError(f"No unique framework matrix for {abi}/{platform.machine()}")
    targets = groups[0]["targets"]
    ids, branches = [t["id"] for t in targets], [t["pta_release"] for t in targets]
    if len(set(ids)) != len(ids) or len(set(branches)) != len(branches):
        raise ValueError("Duplicate framework target ID or branch")
    return targets


def validate_target(target: dict) -> None:
    """Require exact binary inputs, complete identities and safe local cache filenames."""
    if target.get("build_cann_versions") != ["9.2.0"]:
        raise ValueError(f"Invalid build CANN baseline: {target['id']}")
    for field in ("id", "torch_release", "pta_release", "torch_version", "pta_build_version"):
        if not isinstance(target.get(field), str) or not re.fullmatch(r"[a-zA-Z0-9.+_-]+", target[field]):
            raise ValueError(f"Unsafe framework target field: {field}")
    packages = target.get("packages", [])
    names = set()
    for package in packages:
        name = package["name"].lower().replace("_", "-")
        if name in names:
            raise ValueError(f"Duplicate locked distribution: {name}")
        names.add(name)
        validate_locked_package(package, target, name)
    if not {"torch", "torch-npu", "numpy", "setuptools", "wheel", "packaging", "pybind11", "pytest", "decorator", "attrs", "psutil"} <= names:
        raise ValueError(f"Incomplete framework/build/test package closure: {target['id']}")


def validate_locked_package(package: dict, target: dict, name: str):
    """Keep each package URL, version, filename and SHA256 fully locked."""
    filename = package["filename"]
    if Path(filename).name != filename or not filename.endswith(".whl"):
        raise ValueError(f"Unsafe wheel filename: {filename}")
    url = urllib.parse.urlparse(package["url"])
    if url.scheme != "https" or url.username or url.password or url.query:
        raise ValueError("Framework wheels require an HTTPS URL without embedded credentials/query")
    if urllib.parse.unquote(Path(url.path).name) != filename:
        raise ValueError(f"Wheel URL basename differs from locked filename: {filename}")
    if name in ("torch", "torch-npu"):
        expected = target["torch_version" if name == "torch" else "pta_build_version"]
        if package["version"] != expected:
            raise ValueError(f"Target build version differs from locked wheel: {name}")
    if not re.fullmatch(r"[0-9a-f]{64}", package["sha256"]):
        raise ValueError("Missing locked wheel SHA256")


def isolated_environment(python: Path) -> dict:
    """Preserve CANN/toolchain settings but remove inherited framework configuration."""
    env = dict(os.environ)
    for key in ("PYTHONHOME", "PYTHONPATH", "TORCH_INSTALL_PREFIX", "Torch_DIR", "Caffe2_DIR"):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    # Framework-owned paths are chosen from the target interpreter, never inherited from another venv.
    for key in ("LD_LIBRARY_PATH", "LIBRARY_PATH", "CPATH", "CPLUS_INCLUDE_PATH", "CMAKE_PREFIX_PATH"):
        if key in env:
            env[key] = os.pathsep.join(p for p in env[key].split(os.pathsep)
                                      if "site-packages/torch" not in p and "framework-envs/" not in p
                                      and "site-packages/hyper_multicore" not in p
                                      and "/build/native/payload/" not in p and "/build/native/components/" not in p)
    return env


def cann_tool_environment(python: Path) -> dict:
    """Expose only the selected CANN tool modules to native compiler workers, never to pip."""
    env = isolated_environment(python)
    cann = Path(env["ASCEND_HOME_PATH"]).resolve()
    paths = [cann / "python/site-packages", cann / "opp/built-in/op_impl/ai_core/tbe"]
    if not all(path.is_dir() for path in paths):
        raise ValueError(f"Missing installed CANN compiler Python paths: {paths}")
    env["PYTHONPATH"] = os.pathsep.join(map(str, paths))
    env["HI_PYTHON"] = str(python)
    return env


def verify_wheel(path: Path, package: dict) -> None:
    """Validate locked bytes, current platform tags and actual distribution metadata."""
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename, canonicalize_name

    with path.open("rb") as source:
        checksum = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    if checksum.hexdigest() != package["sha256"]:
        raise ValueError(f"Wheel SHA256 mismatch: {path}")
    _, _, _, tags = parse_wheel_filename(path.name)
    if not tags.intersection(sys_tags()):
        raise ValueError(f"Wheel tags do not match build platform: {path.name}")
    with zipfile.ZipFile(path) as archive:
        entries = [n for n in archive.namelist() if len(Path(n).parts) == 2 and n.endswith(".dist-info/METADATA")]
        if len(entries) != 1:
            raise ValueError(f"Ambiguous wheel metadata: {path}")
        meta = BytesParser().parsebytes(archive.read(entries[0]))
    if canonicalize_name(meta["Name"]) != canonicalize_name(package["name"]) or meta["Version"] != package["version"]:
        raise ValueError(f"Locked wheel distribution mismatch: {path}")


def prepare_environment(target: dict, native: Path, run, base_python: Path) -> Path:
    """Install only verified local wheels into a deterministic, non-relocated venv."""
    validate_target(target)
    key = digest_json({"base_python": str(base_python.resolve()), "base_version": sys.version,
                       "abi": sys.implementation.cache_tag, "arch": platform.machine(), "packages": target["packages"]})
    prefix = f"cp{sys.version_info.major}{sys.version_info.minor}-{platform.machine()}-torch{target['torch_release']}-pta{target['pta_build_version']}"
    directory = native / "framework-envs" / f"{prefix}-{key[:20]}"
    python = directory / "bin/python"
    ready = directory / "ready.json"
    if ready.exists():
        record = json.loads(ready.read_text())
        if record.get("environment_key") != key:
            raise ValueError(f"Corrupt environment completion marker: {directory}")
        verify_environment(python, target, run)
        return python
    if directory.exists():
        import shutil
        shutil.rmtree(directory)  # Only our incomplete, key-addressed environment under the build lock.
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=True)
    pending = directory / "pending.json"
    pending.write_text(json.dumps({"owner": "hyper-multicore-framework-env", "environment_key": key,
                                  "target": target["id"], "pta_release": target["pta_release"], "created_at": time.time()}))
    run([str(base_python), "-m", "venv", str(directory)], label=f"venv-{target['id']}")
    cache = native / "framework-packages"
    cache.mkdir(parents=True, exist_ok=True)
    download = directory / "download-requirements.txt"
    install = directory / "install-requirements.txt"
    missing = [p for p in target["packages"] if not (cache / p["filename"]).exists()]
    download.write_text("".join(f"{p['url']} --hash=sha256:{p['sha256']}\n" for p in missing))
    if missing:
        run([str(python), "-m", "pip", "download", "--no-deps", "--require-hashes", "--only-binary=:all:",
             "--dest", str(cache), "-r", str(download)], label=f"download-{target['id']}", env=isolated_environment(python))
    for package in target["packages"]:
        verify_wheel(cache / package["filename"], package)
    install.write_text("".join(f"{cache / p['filename']} --hash=sha256:{p['sha256']}\n" for p in target["packages"]))
    run([str(python), "-m", "pip", "install", "--no-index", "--no-cache-dir", "--no-deps", "--require-hashes",
         "--only-binary=:all:", "-r", str(install)], label=f"install-{target['id']}", env=isolated_environment(python))
    verify_environment(python, target, run)
    ready.write_text(json.dumps({"environment_key": key, "target": target["id"], "pta_release": target["pta_release"], "created_at": time.time(), "packages": target["packages"]}, indent=2))
    pending.unlink(missing_ok=True)
    return python


def verify_environment(python: Path, target: dict, run) -> None:
    """Detect changed packages or dependency conflicts before reusing a cached environment."""
    expected = {p["name"]: p["version"] for p in target["packages"]}
    code = 'from importlib.metadata import version\nimport json,sys\nexpected=json.loads(sys.argv[1])\nactual={n:version(n) for n in expected}\nif actual != expected:\n    raise RuntimeError(f"Framework environment changed: {actual!r} != {expected!r}")'
    run([str(python), "-c", code, json.dumps(expected)], label=f"identity-{target['id']}", env=isolated_environment(python))
    run([str(python), "-m", "pip", "check"], label=f"pip-check-{target['id']}", env=isolated_environment(python))


def environment_record(directory: Path):
    """Read successful environments or explicitly owned abandoned installs only."""
    marker = directory / "ready.json"
    if marker.is_file():
        return json.loads(marker.read_text())
    try:
        record = json.loads((directory / "pending.json").read_text())
    except (OSError, ValueError):
        return None
    key = record.get("environment_key", "")
    if (record.get("owner") == "hyper-multicore-framework-env"
            and re.fullmatch(r"[0-9a-f]{64}", key) and directory.name.endswith(key[:20])):
        return record
    return None


def rollback_environments(candidates, branches):
    """Retain exactly the newest previously successful rollback for each active branch."""
    keep = set()
    for branch in branches:
        successful = [(d, m) for d, m in candidates if m.get("pta_release") == branch and m.get("last_successful_build")]
        if successful:
            keep.add(max(successful, key=lambda item: item[1]["last_successful_build"])[0])
    return keep


def prune_environments(native: Path, selected: list[Path], branches: set[str]) -> list[str]:
    """After full success retain selected environments and one successful rollback per branch."""
    import shutil

    current = {p.parent.parent.resolve() for p in selected}
    candidates = []
    for directory in (native / "framework-envs").iterdir():
        if directory.is_symlink() or not directory.is_dir():
            continue
        record = environment_record(directory)
        if record is None:
            continue
        if directory.resolve() in current:
            record["last_successful_build"] = time.time()
            (directory / "ready.json").write_text(json.dumps(record, indent=2))
        else:
            candidates.append((directory, record))
    keep = rollback_environments(candidates, branches)
    removed = []
    for directory, _ in candidates:
        if directory not in keep:
            shutil.rmtree(directory)
            removed.append(str(directory))
    return removed
