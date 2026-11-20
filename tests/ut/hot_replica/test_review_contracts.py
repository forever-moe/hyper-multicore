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
"""Exercise the placement and resource cleanup without starting a device runtime."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from hyper_multicore import _collective
from hyper_multicore.modules.hot_replica import mega_moe_adapter, shmem_transfer
from hyper_multicore.modules.mega_moe.heap_manager import _HeapManager
from hyper_multicore.shmem import _lifecycle


def test_retry_free_skips_an_already_released_symmetric_allocation(monkeypatch):
    pool = SimpleNamespace(weights=("first", "second"))
    free = Mock(side_effect=[None, RuntimeError("busy"), None])
    monkeypatch.setattr(shmem_transfer.shmem, "free", free)
    with pytest.raises(RuntimeError, match="busy"):
        shmem_transfer.SharedExpertPool._free(pool)  # pylint: disable=protected-access
    assert pool.weights == ("second",)
    shmem_transfer.SharedExpertPool._free(pool)  # pylint: disable=protected-access
    assert [c.args for c in free.call_args_list] == [("first",), ("second",), ("second",)]
    assert not pool.weights


@pytest.mark.parametrize("failure", ["local", "remote", "diagnostic", "exchange"])
def test_terminal_bank_allocation_skips_owner_cleanup_on_all_peers(monkeypatch, failure):
    """Reuse the existing result exchange to stop cleanup on failing and healthy ranks."""
    monkeypatch.setattr(_lifecycle, "_runtime_invalidated", False)
    hidden = torch.empty((0, 4))
    manager = _HeapManager(None, hidden)
    monkeypatch.setattr(manager, "reserve", Mock())
    monkeypatch.setattr(manager, "bind", Mock())
    close = Mock()
    monkeypatch.setattr(manager, "close_workspace", close)
    monkeypatch.setattr(torch.distributed, "get_rank", Mock(return_value=0))
    monkeypatch.setattr(torch.distributed, "get_world_size", Mock(return_value=2))
    original = MemoryError("Tensor publication failed")
    def allocate() -> None:
        """Model a raw native error after the Python lifecycle latched its failure."""
        if failure == "local":
            monkeypatch.setattr(_lifecycle, "_runtime_invalidated", True)
            raise original
    monkeypatch.setattr(shmem_transfer.SharedExpertPool, "_allocate", Mock(side_effect=allocate))
    def peer_results(value: object, **_kwargs: object) -> list[object]:
        """Supply one existing stage exchange with the selected peer outcome."""
        report = ("MemoryError", "Tensor publication failed")
        return [value] if failure == "local" else [value, (None, report, True)]
    exchange = Mock(side_effect=peer_results)
    if failure == "exchange":
        exchange.side_effect = _collective.CollectiveConvergenceError("Root exchange failed")
    if failure == "diagnostic":
        monkeypatch.setattr(_collective, "format_collective_failures", Mock(side_effect=MemoryError("diagnostic")))
    monkeypatch.setattr(_collective, "exchange", exchange)
    spec = SimpleNamespace(local_experts=1, hidden_size=4, intermediate_size=2,
                           num_cube_cores=1, expert_weight_layout="EIO", ep_group=object())
    expected = {"diagnostic": MemoryError, "exchange": _collective.CollectiveConvergenceError}.get(
        failure, _lifecycle._RuntimeUnavailableError)  # pylint: disable=protected-access
    with pytest.raises(expected) as raised:
        mega_moe_adapter._get_mega_bank(hidden, SimpleNamespace(spec=spec, heap_manager=manager), Mock())
    close.assert_not_called()
    assert manager.state == "failed"
    assert _lifecycle._runtime_invalidated  # pylint: disable=protected-access
    assert exchange.call_count == 1
    if failure == "local":
        assert raised.value.__cause__ is original
