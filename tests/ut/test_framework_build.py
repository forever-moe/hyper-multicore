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
"""Controller and environment invariants without downloading frameworks or compiling kernels."""

import hashlib
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENV = runpy.run_path(str(ROOT / "hyper_multicore/_build/framework_environments.py"))
BUILD = runpy.run_path(str(ROOT / "hyper_multicore/_build/build_native.py"))


def make_target(cache):
    """Create a small wheel-only dependency closure with real bytes and metadata."""
    packages = []
    for name in ("torch", "torch-npu", "numpy", "setuptools", "wheel", "packaging", "pybind11", "pytest", "decorator", "attrs", "psutil"):
        filename = name.replace("-", "_") + "-1.0.0-py3-none-any.whl"
        path = cache / filename
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(name.replace("-", "_") + "-1.0.0.dist-info/METADATA",
                             f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0.0\n")
        packages.append({"name": name, "version": "1.0.0", "filename": filename,
                         "url": "https://example.invalid/" + filename,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return {"id": "fixture", "torch_version": "1.0.0", "torch_release": "1.0.0", "pta_release": "1.0.0", "pta_build_version": "1.0.0",
            "build_cann_versions": ["9.2.0"], "packages": packages}


def test_install_uses_verified_local_list_and_no_dependency_download(tmp_path):
    """Generated download URLs never appear in the installer list or installer arguments."""
    cache = tmp_path / "framework-packages"
    cache.mkdir()
    target = make_target(cache)
    commands = []
    def run(command, **_kwargs):
        commands.append(command)
        if command[1:3] == ["-m", "venv"]:
            (Path(command[3]) / "bin").mkdir(parents=True)
            (Path(command[3]) / "bin/python").write_text("fixture")
    python = ENV["prepare_environment"](target, tmp_path, run, Path(sys.executable))
    installation = next(c for c in commands if c[1:4] == ["-m", "pip", "install"])
    assert {"--no-index", "--no-cache-dir", "--no-deps", "--require-hashes", "--only-binary=:all:"} <= set(installation)
    listing = Path(installation[-1]).read_text()
    assert "https://" not in listing and "http://" not in listing
    assert all(str(cache / p["filename"]) in listing for p in target["packages"])
    assert not any(c[1:4] == ["-m", "pip", "download"] for c in commands)
    assert python.parent.parent.joinpath("ready.json").exists()
    target["packages"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        ENV["prepare_environment"](target, tmp_path, run, Path(sys.executable))


def test_environment_creation_failure_never_marks_ready(tmp_path):
    """An installer failure cannot become a reusable cached environment."""
    cache = tmp_path / "framework-packages"
    cache.mkdir()
    target = make_target(cache)
    def fail(command, **_kwargs):
        if command[1:3] == ["-m", "venv"]:
            (Path(command[3]) / "bin").mkdir(parents=True)
        else:
            raise RuntimeError("offline missing dependency")
    with pytest.raises(RuntimeError, match="offline"):
        ENV["prepare_environment"](target, tmp_path, fail, Path(sys.executable))
    assert not list(tmp_path.rglob("ready.json"))


def test_environment_isolation_preserves_cann_but_removes_framework_paths():
    """The selected interpreter cannot inherit another framework's CMake or SO paths."""
    with patch.dict(ENV["os"].environ, {"PYTHONPATH": "wrong", "Torch_DIR": "wrong",
                    "LD_LIBRARY_PATH": "/cann/lib64:/old/site-packages/torch/lib:/old/build/native/payload/hyper_multicore/shmem/lib",
                    "ASCEND_HOME_PATH": "/cann", "CMAKE_PREFIX_PATH": "/cann/compiler:/old/site-packages/torch/share/cmake"}, clear=True):
        result = ENV["isolated_environment"](Path("/owned/venv/bin/python"))
    assert "PYTHONPATH" not in result and "Torch_DIR" not in result
    assert result["LD_LIBRARY_PATH"] == "/cann/lib64"
    assert result["ASCEND_HOME_PATH"] == "/cann"
    assert result["CMAKE_PREFIX_PATH"] == "/cann/compiler"
    assert result["PATH"].startswith("/owned/venv/bin")


def test_framework_compiler_override_does_not_change_common_toolchain():
    """A framework requiring newer C++ cannot redirect the CANN vendor compiler."""
    with patch.dict(ENV["os"].environ, {"CXX": "/system/g++", "HYPER_MULTICORE_FRAMEWORK_CXX": "/framework/g++"}):
        common = ENV["isolated_environment"](Path("/owned/venv/bin/python"))
        adapter = BUILD["adapter_environment"](Path("/owned/venv/bin/python"))
        assert common["CXX"] == "/system/g++"
        assert adapter["CXX"] == "/framework/g++"
        assert ENV["os"].environ["CXX"] == "/system/g++"


def test_interrupted_wheel_copy_preserves_previous_published_bytes(tmp_path):
    """A partial copy must never occupy the final installable wheel name."""
    source, destination = tmp_path / "source.whl", tmp_path / "published.whl"
    source.write_bytes(b"complete new wheel")
    destination.write_bytes(b"complete previous wheel")
    def interrupt(_source, temporary):
        Path(temporary).write_bytes(b"partial")
        raise KeyboardInterrupt()
    with patch.object(BUILD["shutil"], "copy2", side_effect=interrupt), pytest.raises(KeyboardInterrupt):
        BUILD["publish_wheel"](source, destination)
    assert destination.read_bytes() == b"complete previous wheel"
    assert not list(tmp_path.glob("*.tmp"))


def test_interrupted_payload_exchange_keeps_a_complete_visible_directory(tmp_path):
    """Interrupt immediately after the real atomic exchange, before old-output cleanup."""
    candidate, selected = tmp_path / "candidate", tmp_path / "selected"
    candidate.mkdir()
    selected.mkdir()
    (candidate / "identity").write_text("complete new payload")
    (selected / "identity").write_text("complete previous payload")
    exchange = BUILD["exchange_directories"]
    def interrupt(first, second):
        exchange(first, second)
        raise KeyboardInterrupt()
    function = BUILD["publish_payload"]
    with patch.dict(function.__globals__, {"exchange_directories": interrupt}), pytest.raises(KeyboardInterrupt):
        function(candidate, selected)
    assert (selected / "identity").read_text() == "complete new payload"
    assert not candidate.exists()


def test_failed_payload_exchange_preserves_previous_directory(tmp_path):
    """A filesystem publication failure cannot remove a prior successful payload."""
    candidate, selected = tmp_path / "candidate", tmp_path / "selected"
    candidate.mkdir()
    selected.mkdir()
    (selected / "identity").write_text("complete previous payload")
    function = BUILD["publish_payload"]
    with patch.dict(function.__globals__, {"exchange_directories": lambda *_: (_ for _ in ()).throw(OSError("exchange failed"))}):
        with pytest.raises(OSError, match="exchange failed"):
            function(candidate, selected)
    assert (selected / "identity").read_text() == "complete previous payload"
    assert not candidate.exists()


def test_prune_reclaims_only_marked_abandoned_framework_environments(tmp_path):
    """Successful builds may reclaim failed owned installs while preserving unowned directories."""
    base = tmp_path / "framework-envs"
    key = "1" * 64
    owned = base / ("cp310-aarch64-torch2.7.1-pta2.7.1.post11-" + key[:20])
    owned.mkdir(parents=True)
    (owned / "pending.json").write_text(json.dumps({"owner": "hyper-multicore-framework-env", "environment_key": key,
                                                   "pta_release": "2.7.1"}))
    unowned = base / "unrecognized-user-directory"
    unowned.mkdir()
    removed = ENV["prune_environments"](tmp_path, [], {"2.7.1"})
    assert removed == [str(owned)]
    assert unowned.exists()


@pytest.mark.parametrize("variable", ["CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH", "CMAKE_PREFIX_PATH", "PYTHONPATH"])
def test_task_compiler_inputs_include_consumed_search_paths(variable):
    """Common and adapter keys record effective include/link settings while ignoring scheduling."""
    original = BUILD["compiler_inputs"]({variable: "/original"})
    changed = BUILD["compiler_inputs"]({variable: "/replacement"})
    assert original != changed
    assert BUILD["compiler_inputs"]({variable: "/original", "JOBS": "24"}) == original


@pytest.mark.parametrize("compiler", ["/usr/bin/g++", "/tools/aarch64-conda-linux-gnu-g++", "/tool chain/g++-12",
                                     "/usr/bin/clang++-15", "/cann/bin/bisheng", "c++"])
def test_compiler_metrics_recognize_prefixed_and_versioned_compilers(compiler):
    """Real cross-toolchain compile commands must not be reported as cache hits or zero compiles."""
    output = f'cd /work && "{compiler}" -DNAME="quoted value" -o source.o -c source.cpp\n"{compiler}" -shared source.o -o lib.so\n'
    metrics = BUILD["compiler_metrics"](output)
    assert metrics["compiler_invocations"] == metrics["compiler_commands_logged"] == 1
    assert metrics["host_object_compile_actions"] == metrics["kernel_compile_jobs"] == 0


def test_compiler_metrics_keep_cmake_actions_and_kernel_jobs_separate():
    """CMake progress is an observable fallback; OPC jobs do not count hidden compiler subprocesses."""
    metrics = BUILD["compiler_metrics"]('[ 50%] Building CXX object source.cpp.o\nOpc tool start working now\n')
    assert metrics == {"compiler_invocations": 1, "compiler_commands_logged": 0,
                       "host_object_compile_actions": 1, "kernel_compile_jobs": 1}
    assert not any(BUILD["compiler_metrics"]('[100%] Built target cached\n').values())


def test_preflight_reuses_interfaces_across_unrelated_source_identity_changes(tmp_path):
    """An assembly commit cannot force recompilation of unchanged framework interfaces."""
    for name in ("hyper_multicore/cmake/framework_probe/probe.cpp", "hyper_multicore/cmake/torch_compat.cmake",
                 "hyper_multicore/torch/csrc/cached_op_api.h"):
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("probe input")
    identity = {k: "fixture" for k in ("torch_version", "pta_version", "torch_cxx11_abi", "python_abi", "host_arch",
        "torch_headers_sha256", "pta_headers_sha256", "torch_libraries_sha256", "libtorch_npu_sha256", "cann_root", "cann_version_sha256")}
    function = BUILD["preflight_fingerprint"]
    with patch.dict(function.__globals__, {"ROOT": tmp_path}):
        first = function(identity, Path("/venv/bin/python"), {"version": "GCC12"}, {})
        identity.update(build_commit="later assembly commit", source_sha256="unrelated packaging input")
        assert function(identity, Path("/venv/bin/python"), {"version": "GCC12"}, {}) == first
        assert function(identity, Path("/venv/bin/python"), {"version": "GCC10"}, {}) != first
        (tmp_path / "hyper_multicore/torch/csrc/cached_op_api.h").write_text("changed consumed header")
        assert function(identity, Path("/venv/bin/python"), {"version": "GCC12"}, {}) != first


def test_native_cache_checks_resources_and_sdk_handoff(tmp_path):
    """A valid marker does not mask modified shared resources or missing SDK headers."""
    (tmp_path / "payload").mkdir()
    (tmp_path / "payload/lib.so").write_bytes(b"native")
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    (sdk / "shmem.h").write_text("header")
    (tmp_path / "sdk.json").write_text(json.dumps({"install_root": str(sdk)}))
    data = {"cache_schema": BUILD["CACHE_SCHEMA_VERSION"], "fingerprint": "key", "hashes": BUILD["hashes"](tmp_path / "payload"), "sdk_root": str(sdk),
            "sdk_hashes": BUILD["hashes"](sdk), "sdk_manifest_sha256": hashlib.sha256((tmp_path / "sdk.json").read_bytes()).hexdigest()}
    (tmp_path / "complete.json").write_text(json.dumps(data))
    assert BUILD["cache_valid"](tmp_path, "key")
    (sdk / "shmem.h").write_text("changed")
    with pytest.raises(ValueError, match="SDK inputs"):
        BUILD["cache_valid"](tmp_path, "key")


def test_scoped_fingerprint_ignores_unconsumed_framework_config(tmp_path):
    """Only the dependency set passed by a stage can invalidate that stage."""
    (tmp_path / "common").mkdir()
    (tmp_path / "common/kernel.h").write_text("kernel")
    lock = tmp_path / "frameworks.json"
    lock.write_text("first matrix")
    function = BUILD["fingerprint_files"]
    with patch.dict(function.__globals__, {"ROOT": tmp_path}):
        first = function(("common",))
        lock.write_text("different matrix")
        assert function(("common",)) == first
        (tmp_path / "common/kernel.h").write_text("changed kernel")
        assert function(("common",)) != first


def test_success_prunes_only_owned_old_environments(tmp_path):
    """Retain current and latest successful rollback; never touch external symlinks."""
    base = tmp_path / "framework-envs"
    base.mkdir()
    current = base / "current"
    for name, timestamp in (("current", 3), ("rollback", 2), ("older", 1), ("removed-branch", 5)):
        directory = base / name
        (directory / "bin").mkdir(parents=True)
        (directory / "ready.json").write_text(json.dumps({"pta_release": "2.8.0" if name == "removed-branch" else "2.12.0",
                                                         "last_successful_build": timestamp}))
    external = tmp_path / "external"
    external.mkdir()
    (base / "external-link").symlink_to(external, target_is_directory=True)
    removed = ENV["prune_environments"](tmp_path, [current / "bin/python"], {"2.12.0"})
    assert {Path(p).name for p in removed} == {"older", "removed-branch"}
    assert current.exists() and (base / "rollback").exists() and external.exists()


def test_interruption_kills_grandchild_before_controller_returns(tmp_path):
    """TERM-resistant descendants must not outlive their interrupted build controller."""
    import os
    import signal
    import subprocess
    import time

    child_pid = tmp_path / "grandchild.pid"
    group_pid = tmp_path / "child-group.pid"
    grandchild = "import os,signal,time; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); Path(%r).write_text(str(os.getpid())); time.sleep(120)" % str(child_pid)
    child = "import os,subprocess,sys,time; from pathlib import Path; Path(%r).write_text(str(os.getpid())); subprocess.Popen([sys.executable,'-c',%r]); time.sleep(120)" % (str(group_pid), grandchild)
    wrapper = tmp_path / "controller.py"
    wrapper.write_text("import runpy,sys; from pathlib import Path\nh=runpy.run_path(%r)\nh['Commands'](Path(%r))([sys.executable,'-c',%r],label='child')\n" %
                       (str(ROOT / "hyper_multicore/_build/build_native.py"), str(tmp_path / "logs"), child))
    with subprocess.Popen([sys.executable, str(wrapper)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as controller:
        try:
            deadline = time.monotonic() + 20
            while not child_pid.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert child_pid.exists(), "test grandchild did not start"
            controller.send_signal(signal.SIGINT)
            controller.wait(timeout=25)
            stat = Path("/proc") / child_pid.read_text() / "stat"
            assert not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
        finally:
            if controller.poll() is None:
                controller.kill()
                controller.wait()
            if group_pid.exists():
                try:
                    os.killpg(int(group_pid.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if child_pid.exists():
                try:
                    os.kill(int(child_pid.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass


def test_cann_python_paths_are_scoped_to_compiler_workers(tmp_path):
    """CANN tools need their imports, while pip must not inspect unrelated CANN distributions."""
    cann = tmp_path / "cann"
    paths = [cann / "python/site-packages", cann / "opp/built-in/op_impl/ai_core/tbe"]
    for path in paths:
        path.mkdir(parents=True)
    python = Path("/owned/venv/bin/python")
    with patch.dict(ENV["os"].environ, {"ASCEND_HOME_PATH": str(cann), "PYTHONPATH": "/old/framework:/user/source"}):
        installed = ENV["isolated_environment"](python)
        compiler = ENV["cann_tool_environment"](python)
    assert "PYTHONPATH" not in installed
    assert compiler["PYTHONPATH"].split(ENV["os"].pathsep) == list(map(str, paths))
    assert compiler["HI_PYTHON"] == str(python)


def test_cann_precheck_fails_before_framework_installation(tmp_path):
    """A wrong CANN installation must be rejected before preparing the release matrix."""
    function = BUILD["validate_toolchain"]
    with patch.dict(BUILD["os"].environ, {}, clear=True), pytest.raises(ValueError, match="source the CANN"):
        function()
    (tmp_path / "opp").mkdir()
    (tmp_path / "opp/version.info").write_text("Version=9.1.0\n")
    with patch.dict(BUILD["os"].environ, {"ASCEND_HOME_PATH": str(tmp_path)}), pytest.raises(ValueError, match="9.2.0"):
        function()


def test_vendored_distribution_metadata_is_not_the_wheel_identity(tmp_path):
    """Setuptools-like nested metadata must not create a false duplicate primary distribution."""
    target = make_target(tmp_path)
    package = next(p for p in target["packages"] if p["name"] == "setuptools")
    wheel = tmp_path / package["filename"]
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("setuptools/_vendor/example-2.0.dist-info/METADATA", "Name: example\nVersion: 2.0\n")
    package["sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
    ENV["verify_wheel"](wheel, package)
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("unexpected-2.0.dist-info/METADATA", "Name: unexpected\nVersion: 2.0\n")
    package["sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Ambiguous"):
        ENV["verify_wheel"](wheel, package)


@pytest.mark.parametrize("minor", [7, 12, 14])
@pytest.mark.parametrize("forwarded", [False, True])
def test_op_plugin_compilation_uses_actual_torch_branch(tmp_path, minor, forwarded):
    """Newer op-plugin headers must not select removed APIs for an older Torch build."""
    import subprocess

    torch = tmp_path / "torch"
    pta = tmp_path / "torch_npu"
    header = torch / "include/torch/csrc/api/include/torch/version.h"
    plugin = pta / "include/third_party/op-plugin/op_plugin/utils/Version.h"
    header.parent.mkdir(parents=True)
    plugin.parent.mkdir(parents=True)
    definitions = f"#define TORCH_VERSION_MAJOR 2\n#define TORCH_VERSION_MINOR {minor}\n"
    if forwarded:
        header.write_text('#include <torch/headeronly/version.h>\n')
        header = torch / "include/torch/headeronly/version.h"
        header.parent.mkdir(parents=True)
    header.write_text(definitions)
    plugin.write_text("#define V2R7 6\n#define V2R12 11\n#define V2R14 13\n#define VERSION_NEWEST V2R14\n")
    script = tmp_path / "check.cmake"
    script.write_text('include("%s")\nhp_detect_op_plugin_version("%s" "%s" selected)\nif(NOT selected STREQUAL "V2R%d")\nmessage(FATAL_ERROR "wrong branch")\nendif()\n' %
                      (ROOT / "hyper_multicore/cmake/torch_compat.cmake", torch, pta, minor))
    completed = subprocess.run(["cmake", "-P", str(script)], text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    plugin.write_text("#define VERSION_NEWEST 99\n")
    completed = subprocess.run(["cmake", "-P", str(script)], text=True, capture_output=True, check=False)
    assert completed.returncode != 0 and "no version selector" in completed.stderr


@pytest.mark.parametrize("kind", ["common", "variants"])
def test_changed_stage_never_leaves_reusable_completion(tmp_path, kind):
    """Reject changed inputs before cache commit and rebuild after restoring A."""
    identity = {"source_sha256": "A"}
    directory = tmp_path / kind / "key"
    cls = BUILD["CacheTransaction"]
    with patch.dict(cls.check.__globals__, {"source_observation": lambda: dict(identity)}):
        with pytest.raises(RuntimeError, match="changed"):
            with cls() as transaction:
                transaction.prepare(directory)
                (directory / "payload").mkdir()
                (directory / "payload/lib.so").write_bytes(b"B output")
                identity["source_sha256"] = "B"
                transaction.complete(directory, {"fingerprint": "key", "hashes": BUILD["hashes"](directory / "payload")})
        assert not (directory / "complete.json").exists()
        identity["source_sha256"] = "A"
        assert not BUILD["cache_valid"](directory, "key")
        with cls() as transaction:
            transaction.prepare(directory)
            assert not (directory / "payload/lib.so").exists()
            (directory / "payload").mkdir()
            (directory / "payload/lib.so").write_bytes(b"A output")
            transaction.complete(directory, {"fingerprint": "key", "hashes": BUILD["hashes"](directory / "payload")})
        assert BUILD["cache_valid"](directory, "key")


@pytest.mark.parametrize("error", [RuntimeError("worker failed"), KeyboardInterrupt()])
def test_failed_run_revokes_new_markers_but_preserves_existing_cache(tmp_path, error):
    """A later failure cannot leave a new completion; previous successful caches survive."""
    cls = BUILD["CacheTransaction"]
    identity = {"source_sha256": "A"}
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "complete.json").write_text("previous success")
    directory = tmp_path / "new"
    with patch.dict(cls.check.__globals__, {"source_observation": lambda: dict(identity)}):
        with pytest.raises(type(error)):
            with cls() as transaction:
                transaction.prepare(directory)
                (directory / "payload").mkdir()
                transaction.complete(directory, {"fingerprint": "key", "hashes": {}})
                raise error
    assert not (directory / "complete.json").exists()
    assert (existing / "complete.json").read_text() == "previous success"


def test_legacy_cache_markers_are_not_reusable(tmp_path):
    """Old cache records cannot certify artifacts produced before the transaction fix."""
    (tmp_path / "payload").mkdir()
    (tmp_path / "complete.json").write_text(json.dumps({"fingerprint": "key", "hashes": {}}))
    assert not BUILD["cache_valid"](tmp_path, "key")


def test_activation_script_is_a_consumed_common_input(tmp_path):
    """Changing activation bytes changes the public common-source fingerprint."""
    function = BUILD["common_source_fingerprint"]
    def digest(_directories, extra):
        return hashlib.sha256(b"".join((tmp_path / name).read_bytes() for name in extra)).hexdigest()
    for name in BUILD["COMMON_EXTRA_INPUTS"]:
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("input")
    with patch.dict(function.__globals__, {"fingerprint_files": digest}):
        before = function()
        (tmp_path / "hyper_multicore/set_env.bash").write_text("new activation")
        assert function() != before


def test_adapter_worker_input_change_is_rejected_before_cache_commit(tmp_path):
    """Exercise the actual adapter stage with a worker that changes A into B."""
    from types import SimpleNamespace

    identity = {"pta_version": "2.12.0", "host_arch": "aarch64", "build_commit": "fixture",
                "source_sha256": "A", "build_fingerprint": "fixture"}
    observation = {"source": "A"}
    native = tmp_path / "native"
    payload = native / "variants/key/payload"
    class Worker:
        events = []
        def __call__(self, command, **_kwargs):
            if command[:2] == ["cmake", "--install"]:
                payload.mkdir(parents=True, exist_ok=True)
                (payload / "libhyper_multicore_torch.so").write_bytes(b"B adapter")
                (payload / "hyper_multicore_shmem_torch.so").write_bytes(b"B binding")
                observation["source"] = "B"
    function = BUILD["cmake_pair"]
    with patch.dict(function.__globals__, {"NATIVE": native, "source_observation": lambda: dict(observation),
                    "adapter_environment": lambda _python: {}, "validate_elf": lambda *_args: None}):
        with pytest.raises(RuntimeError, match="changed"):
            function(Path(sys.executable), identity, tmp_path / "common", tmp_path / "sdk", "key",
                     SimpleNamespace(clean=False, jobs=1), Worker())
    assert not (payload.parent / "complete.json").exists()


@pytest.mark.parametrize("relative", ["hyper_multicore/input.cpp", "setup.py", "MANIFEST.in", "build.sh",
                                      "requirements.txt", "tests/ut/shmem/native_log_test.cpp"])
def test_restoring_source_bytes_during_build_is_still_rejected(tmp_path, relative):
    """A write-and-restore changes the file generation even if its digest is unchanged."""
    source = tmp_path / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("A")
    cls = BUILD["CacheTransaction"]
    helpers = {"_source_identity": lambda _root: {"source_sha256": "A"}}
    with patch.dict(cls.check.__globals__, {"ROOT": tmp_path, "HELPERS": helpers}):
        transaction = cls()
        # Linux may reuse a coarse filesystem timestamp for writes in the same tick.
        import time
        time.sleep(0.02)
        source.write_text("B")
        source.write_text("A")
        with pytest.raises(RuntimeError, match="changed"):
            transaction.check()


def test_common_worker_input_change_is_rejected_before_cache_commit(tmp_path):
    """Exercise common SDK/payload certification with a source-changing worker."""
    from types import SimpleNamespace

    native = tmp_path / "native"
    directory = native / "common/key"
    observation = {"source": "A"}
    class Worker:
        events = []
        def __call__(self, _command, **_kwargs):
            (directory / "payload").mkdir()
            (directory / "payload/lib.so").write_bytes(b"B common output")
            sdk = directory / "sdk"
            sdk.mkdir()
            (sdk / "header.h").write_text("B SDK")
            (directory / "sdk.json").write_text(json.dumps({"install_root": str(sdk)}))
            observation["source"] = "B"
    function = BUILD["build_common"]
    with patch.dict(function.__globals__, {"NATIVE": native, "source_observation": lambda: dict(observation)}):
        with pytest.raises(RuntimeError, match="changed"), BUILD["CacheTransaction"]() as transaction:
            function(SimpleNamespace(clean=False, jobs=1, soc_list="ascend910b"), Worker(),
                     Path(sys.executable), "key", {}, transaction)
    assert not (directory / "complete.json").exists()


def test_component_shell_dispatch_preserves_arguments_and_failure(tmp_path):
    """The surviving public shell path forwards selection and returns the controller's failure."""
    import os
    import subprocess

    output = tmp_path / "arguments.json"
    python = tmp_path / "python"
    python.write_text(f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\n"
                      f"Path({str(output)!r}).write_text(json.dumps(sys.argv[1:]))\nraise SystemExit(23)\n")
    python.chmod(0o755)
    result = subprocess.run(["bash", str(ROOT / "hyper_multicore/build.sh"), "--soc-list", "ascend910b",
                             "--jobs", "2"], env={**os.environ, "PYTHON_BIN": str(python), "HP_NATIVE_STAGE": ""},
                            text=True, capture_output=True, check=False)
    assert result.returncode == 23, result.stderr
    arguments = json.loads(output.read_text())
    assert arguments[:2] == [str(ROOT / "hyper_multicore/_build/build_native.py"), "--native-only"]
    assert arguments[arguments.index("--soc-list") + 1] == "ascend910b"
    assert arguments[arguments.index("--jobs") + 1] == "2"
