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
"""Observe the production forward division overload without emulating NPU arithmetic."""

from pathlib import Path
import subprocess
import tempfile
import unittest

import hyper_multicore


_ROOT = Path(hyper_multicore.__file__).resolve().parent.parent
_MODEL = r'''
#include <cstdint>
#include <initializer_list>
int observed_mode = -1;
int observed_count = -1;
enum { PIPE_V };
template<int> void PipeBarrier() {}
template<typename T> void Div(T* output, T* left, T* right, int count) {
    observed_mode = 0;
    observed_count = count;
    for (int index = 0; index < count; ++index) { output[index] = left[index] / right[index]; }
}
#if __NPU_ARCH__ == 3510
enum class DivAlgo { DIFF_COMPENSATION = 1 };
struct DivConfig { DivAlgo algo; };
template<typename T, const DivConfig& config> void Div(T* output, T* left, T* right, int count) {
    Div(output, left, right, count);
    observed_mode = static_cast<int>(config.algo);
}
#endif
int main() {
    for (int rows : {1, 3}) {
        const int k_align_ = 8;
        float output[32], weights[32], broadcast[32];
        for (int index = 0; index < 32; ++index) {
            output[index] = -7;
            weights[index] = (index + 1) * 0.125f;
            broadcast[index] = (index + 1) * 0.25f;
        }
HP_FORWARD_DIV
        if (observed_mode != (__NPU_ARCH__ == 3510 ? 1 : 0) || observed_count != rows * k_align_) { return 1; }
        for (int index = 0; index < 32; ++index) {
            const float expected = index < rows * k_align_ ? weights[index] / broadcast[index] : -7;
            if (output[index] != expected) { return 2; }
        }
    }
    return 0;
}
'''


class TestRouteForwardDiv(unittest.TestCase):
    """Keep A2/A3 dispatch while matching A5 RealDiv precision."""

    def test_forward_precision_and_tail_bounds(self) -> None:
        """Compile the actual dispatch and observe its SDK mode/count on three architectures."""
        header = _ROOT / "hyper_multicore/ops/hyper_mega_gate/op_kernel/route_pipeline.h"
        source = header.read_text(encoding="utf-8")
        method = source[source.index("void DivideSelected()"):source.index("void ScaleSelected()")]
        start = method.index("PipeBarrier<PIPE_V>();", method.index("BroadCast<float, 2, 1>"))
        dispatch = method[start:method.index("output_queue.EnQue(output)")].rsplit("      }", 1)[0]
        model = _MODEL.replace("HP_FORWARD_DIV", dispatch)
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "gate-forward-div"
            for architecture in (2201, 2203, 3510):
                with self.subTest(architecture=architecture):
                    result = subprocess.run(
                        ["c++", "-std=c++17", f"-D__NPU_ARCH__={architecture}", "-x", "c++", "-",
                         "-o", str(executable)], input=model, capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(result.returncode, 0, f"expected compile=0 actual={result.stderr}")
                    result = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
                    self.assertEqual(result.returncode, 0,
                                     f"expected mode/count and intact tail actual exit={result.returncode}")
