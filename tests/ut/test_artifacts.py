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
"""Artifact-label and fail-closed payload tests without the component runtime."""

from importlib import metadata
import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

HELPER = runpy.run_path(str(Path(__file__).absolute().parents[2] / "hyper_multicore/_build/artifacts.py"))
LOADER = runpy.run_path(str(Path(__file__).absolute().parents[2] / "hyper_multicore/_loader.py"))


class TestArtifacts(unittest.TestCase):
    """Keep PTA labels and required native inputs explicit."""

    def test_pta_version_and_post_release(self):
        """PTA supplies the complete public label, not the Torch version."""
        self.assertEqual(HELPER["artifact_version"]("2.9.0"), "0.1.0.dev0+pt.2.9.0")
        self.assertEqual(HELPER["artifact_version"]("2.9.0.post4"), "0.1.0.dev0+pt.2.9.0.post4")
        self.assertEqual(HELPER["artifact_version"]("2.9.0.post4+custom.1"),
                         "0.1.0.dev0+pt.2.9.0.post4.custom.1")
        for value in ("", "unknown"):
            with self.assertRaises(ValueError):
                HELPER["artifact_version"](value)

    def test_missing_pta_is_not_guessed(self):
        """Native identity creation must fail when torch-npu is not installed."""
        with patch.object(HELPER["sys"], "version_info", (3, 11)), patch.object(
            metadata, "version", side_effect=metadata.PackageNotFoundError("torch-npu")
        ), self.assertRaises(metadata.PackageNotFoundError):
            HELPER["build_identity"](Path("."), "ascend910b")

    def test_missing_native_payload_is_rejected(self):
        """A pure-Python tree is never accepted as a complete native wheel payload."""
        with tempfile.TemporaryDirectory() as directory:
            info = {"pta_version": "2.9.0", "artifact_version": "0.1.0.dev0+pt.2.9.0"}
            with self.assertRaisesRegex(RuntimeError, "Missing native payload"):
                HELPER["validate_payload"](Path(directory), info)

    def test_sdist_does_not_inherit_an_enclosing_repository_commit(self):
        """Unpacked sources use their digest, never a parent directory's unrelated Git HEAD."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "hyper_multicore").mkdir()
            (root / "scripts").mkdir()
            for name in ("setup.py", "MANIFEST.in", "build.sh", "requirements.txt"):
                (root / name).write_text("fixture\n", encoding="utf-8")
            with patch.object(HELPER["subprocess"], "run", side_effect=AssertionError("unexpected Git lookup")):
                identity = HELPER["_source_identity"](root)
            self.assertIsNone(identity["build_commit"])
            self.assertEqual(len(identity["source_sha256"]), 64)

    def test_native_environment_mismatch_is_rejected(self):
        """Wrong Torch, PTA, Python, ABI or custom PTA binary must fail before loading."""
        import sys
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lib").mkdir()
            (root / "lib/libtorch_npu.so").write_bytes(b"mock PTA binary")
            (root / "opp").mkdir()
            (root / "opp/version.info").write_bytes(b"mock CANN version")
            actual = {
                "torch_version": "2.9.0", "pta_version": "2.9.0.post4",
                "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
                "host_arch": LOADER["platform"].machine(), "torch_cxx11_abi": True, "torch_commit": "mock",
                "libtorch_npu_sha256": hashlib.sha256(b"mock PTA binary").hexdigest(),
                "cann_version_sha256": hashlib.sha256(b"mock CANN version").hexdigest(),
            }
            validate = LOADER["validate_native_compatibility"]
            with patch.dict(validate.__globals__, {"_component_root": lambda: root / "lib"}), patch.object(
                metadata, "version", side_effect=lambda name: actual["torch_version" if name == "torch"
                                                                      else "pta_version"]
            ), patch.object(LOADER["util"], "find_spec", return_value=SimpleNamespace(origin=str(root / "__init__.py"))), \
                    patch.dict(sys.modules, {"torch": SimpleNamespace(
                        _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True), version=SimpleNamespace(git_version="mock"))}), \
                    patch.dict(LOADER["os"].environ, {"ASCEND_HOME_PATH": str(root)}):
                (root / "build_info.json").write_text(json.dumps(actual), encoding="utf-8")
                validate()
                for key in actual:
                    changed = dict(actual)
                    changed[key] = "incompatible"
                    (root / "build_info.json").write_text(json.dumps(changed), encoding="utf-8")
                    with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                        validate()


if __name__ == "__main__":
    unittest.main()
