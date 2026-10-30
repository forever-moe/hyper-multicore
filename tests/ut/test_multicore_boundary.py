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

"""Verify component exports and absence of framework dispatch."""

import ast
import sys
from pathlib import Path
import unittest

import hyper_multicore as multicore
from tests.common.mark_utils import arg_mark


class TestMulticoreBoundary(unittest.TestCase):
    """Check the component contract without initializing a native runtime."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_business_symbols_are_exported_by_standalone_package(self):
        """Verify the Multicore public export boundary.

        Feature: Multicore public exports.
        Description: Inspect the HyperMulticore public package symbols.
        Expectation: The package exports its business modules and diagnostic helpers.
        """
        self.assertEqual(multicore.__all__, [
            "HyperMegaMhc", "MegaGate", "MegaMoeExperts", "MegaKernelDiagnostics",
            "prepare_mega_kernel_diagnostics", "validate_mega_kernel_diagnostic_contract",
        ])
        self.assertTrue(callable(multicore.prepare_mega_kernel_diagnostics))
        for name in ("HyperMegaMhc", "MegaGate", "MegaMoeExperts"):
            self.assertTrue(callable(getattr(multicore, name)))
        self.assertNotIn("hyper_multicore_shmem_torch", sys.modules)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_runtime_sources_import_no_platform_layer(self):
        """Verify production sources stay Torch-only.

        Feature: Multicore runtime convergence.
        Description: Parse imports in the converged production Python tree.
        Expectation: Runtime sources do not depend on HyperParallel.
        """
        root = Path(multicore.__file__).parent
        for path in root.rglob("*.py"):
            if "lib" in path.relative_to(root).parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    continue
                for module in modules:
                    with self.subTest(path=path, module=module):
                        self.assertFalse(module == "hyper_parallel" or module.startswith("hyper_parallel."))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_diagnostics_dependency_boundaries(self):
        """Feature: MegaKernel diagnostics.
        Description: Keep common code tool-neutral and both tools independent of business adapters.
        Expectation: Diagnostics dependency boundaries.
        """
        root = Path(multicore.__file__).parent / "diagnostics"
        for path in root.rglob("*.py"):
            relative = path.relative_to(root)
            forbidden = ["hyper_multicore.modules"]
            if relative.parts[0] in ("common", "api.py", "__init__.py"):
                forbidden.extend(("diagnostics.profiler", "diagnostics.dump"))
            elif relative.parts[0] == "dump":
                forbidden.append("diagnostics.profiler")
            elif relative.parts[0] == "profiler":
                forbidden.append("diagnostics.dump")
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                modules = ([node.module or ""] if isinstance(node, ast.ImportFrom)
                           else [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                for module in modules:
                    with self.subTest(path=path, module=module):
                        self.assertFalse(any(item in module for item in forbidden))
