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
"""Exercise the A5 functional Gate APIs without a distributed process group."""

import argparse
import json

import torch
import torch_npu  # pylint: disable=unused-import

from hyper_multicore.modules.mega_gate import sigmoid_route_from_hidden_states, sigmoid_route_from_logits


def main() -> None:
    """Run forward and backward using caller-owned Gate parameters.

    Returns:
        Prints output shapes and input/weight gradient norms.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("route", "projection"), default="projection")
    parser.add_argument("--tokens", type=int, default=4096)
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(71)
    hidden = torch.randn(args.tokens, 5120, device="npu", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(384, 5120, device="npu", dtype=torch.bfloat16) * 0.001383)
    bias = torch.zeros(384, device="npu", dtype=torch.float32)
    upstream = torch.randn(args.tokens, 8, device="npu")
    if args.mode == "projection":
        logits, weights, indices = sigmoid_route_from_hidden_states(hidden, weight, expert_bias=bias)
    else:
        with torch.autocast("npu", enabled=False):
            logits = torch.nn.functional.linear(hidden.float(), weight.float())  # pylint: disable=not-callable
        weights, indices = sigmoid_route_from_logits(logits, top_k=8, expert_bias=bias)
    loss = (weights * upstream).mean() + logits.square().mean() * 1e-3
    loss.backward()
    torch.npu.synchronize()
    print(json.dumps({"mode": args.mode, "logits": list(logits.shape), "weights": list(weights.shape),
                      "indices": list(indices.shape), "dx_norm": hidden.grad.float().norm().item(),
                      "dw_norm": weight.grad.float().norm().item()}))


if __name__ == "__main__":
    main()
