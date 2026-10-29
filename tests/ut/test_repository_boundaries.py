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
"""Whole-tree static boundary checks, independent of the installed NPU runtime."""

from pathlib import Path
import runpy
import unittest


class TestRepositoryBoundaries(unittest.TestCase):
    """Enforce the no-downstream rule and unchanged third-party patch hashes."""

    def test_standalone_repository_boundaries(self):
        """Parse production, build, test and example files, not just the default UT tree."""
        root = Path(__file__).absolute().parents[2]
        runpy.run_path(str(root / "scripts/check_repository.py"))["main"]()


if __name__ == "__main__":
    unittest.main()
