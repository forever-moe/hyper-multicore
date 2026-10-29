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
"""Package standalone Python sources and an explicitly built native payload."""

import json
import os
from pathlib import Path
import runpy
import shutil
import sys

from setuptools import Distribution, find_packages, setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
COMPONENT_VERSION = "0.1.0.dev0"
payload = os.environ.get("HYPER_MULTICORE_NATIVE_OUTPUT_ROOT", "").strip()
info = (json.loads((Path(payload) / "build_info.json").read_text(encoding="utf-8"))
        if payload and "sdist" not in sys.argv else None)


class BuildPy(build_py):
    """Copy only a validated, explicitly selected native payload into a wheel."""

    def run(self) -> None:
        """Reject release wheels without native files; permit editable Host development."""
        if info is None and "editable_wheel" not in sys.argv:
            raise RuntimeError("Native payload required for wheels; run bash build.sh. Use editable for Host UT.")
        if info is not None:
            helper = runpy.run_path(str(ROOT / "hyper_multicore/_build/artifacts.py"))
            helper["validate_payload"](Path(payload), info)
            # Removed modules and native libraries must not survive an incremental wheel build.
            package_build = Path(self.build_lib) / "hyper_multicore"
            if package_build.exists():
                shutil.rmtree(package_build)
        super().run()
        if info is not None:
            shutil.copytree(payload, Path(self.build_lib) / "hyper_multicore", dirs_exist_ok=True)


class BinaryDistribution(Distribution):
    """Keep native wheel tags specific to the actual Python ABI and Host platform."""

    def has_ext_modules(self) -> bool:
        """Return true because published wheels contain native extensions."""
        return True


setup(
    name="hyper-multicore",
    version=info["artifact_version"] if info else COMPONENT_VERSION,
    description="Standalone Ascend multicore operators and PyTorch runtime",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/forever-moe/hyper-multicore",
    license="Apache-2.0",
    python_requires=">=3.10,<3.13",
    packages=find_packages(include=["hyper_multicore", "hyper_multicore.*"]),
    install_requires=["numpy>=1.26,<2", f"torch=={info['torch_version']}" if info else "torch"],
    scripts=["scripts/hyper_multicore_set_env.bash"],
    include_package_data=False,
    package_data={"hyper_multicore": ["ops/runtime/*.h", "ops/runtime/*.hpp"]},
    cmdclass={"build_py": BuildPy},
    distclass=BinaryDistribution,
)
