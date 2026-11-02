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
"""Installed matrix inspection remains available without importing Torch or PTA."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from tests.ut.test_artifacts import _payload_fixture

ROOT = Path(__file__).resolve().parents[2]


def test_matrix_json_works_without_framework_import(tmp_path):
    """Use an isolated fake installation to prove the CLI reads records, not native imports."""
    package = tmp_path / "hyper_multicore"
    info = _payload_fixture(package)
    (package / "build_info.json").write_text(json.dumps(info))
    shutil.copy2(ROOT / "hyper_multicore/_native_manifest.py", package / "_native_manifest.py")
    (package / "__init__.py").write_text("raise RuntimeError('must not import package')")
    for name in ("torch", "torch_npu"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "__init__.py").write_text("raise RuntimeError('must not import framework')")
    metadata = tmp_path / "hyper_multicore-0.1.0.dev0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: hyper-multicore\nVersion: 0.1.0.dev0\n")
    command = [sys.executable, str(ROOT / "scripts/hyper_multicore_build_info"), "--json"]
    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    result = subprocess.run(command, env=env, cwd=tmp_path, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["package_root"] == str(package)
    assert report["build_info"]["variants"][0]["id"] == "fixture"
    # Consistent IDs cannot hide changed pair fields or a corrupted packaged resource.
    (package / "adapters/fixture/libhyper_multicore_torch.so").write_bytes(b"corrupted")
    failed = subprocess.run(command, env=env, cwd=tmp_path, text=True, capture_output=True, check=False)
    assert failed.returncode != 0 and "Changed payload resource" in failed.stderr
