# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio

import pytest

from vllm_omni.engine.stage_pool import StagePool

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Replica:
    def __init__(self, *, hang: bool = False, fail: bool = False):
        self.calls: list[list[str]] = []
        self.started = asyncio.Event()
        self.gate = asyncio.Event()
        if not hang:
            self.gate.set()
        self.fail = fail

    async def call_utility_async(self, method, request_ids):
        assert method == "omni_release_request_resources"
        self.calls.append(list(request_ids))
        self.started.set()
        await self.gate.wait()
        if self.fail:
            raise RuntimeError("replica gone")


@pytest.fixture
def make_pool(mocker):
    config = mocker.Mock(model_config=mocker.Mock(async_chunk=True))
    return lambda *replicas: StagePool(0, list(replicas), stage_vllm_config=config)


async def _settle(pool: StagePool) -> None:
    await asyncio.wait_for(asyncio.gather(*pool._release_flushers.values()), timeout=2.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "async_chunk,use_v2,native,expected",
    [(False, False, False, False), (True, False, False, True), (False, True, True, True), (False, True, False, False)],
)
async def test_release_includes_native_full_payload_but_preserves_legacy_gate(
    mocker, async_chunk, use_v2, native, expected
) -> None:
    config = mocker.Mock(
        model_config=mocker.Mock(
            async_chunk=async_chunk, use_v2_model_runner=use_v2, supports_native_mrv2_data_plane=native
        )
    )
    replica = _Replica()
    pool = StagePool(0, [replica], stage_vllm_config=config)

    pool.schedule_release_request_resources(["request"])
    await _settle(pool)

    assert replica.calls == ([["request"]] if expected else [])
    assert not pool._pending_releases and not pool._release_flushers


@pytest.mark.asyncio
async def test_hung_replica_does_not_delay_healthy_replica_batches(monkeypatch, make_pool) -> None:
    hung, healthy = _Replica(hang=True), _Replica()
    pool = make_pool(hung, healthy)
    monkeypatch.setattr(pool, "RELEASE_RPC_TIMEOUT_S", 0.2)
    warn = []
    monkeypatch.setattr("vllm_omni.engine.stage_pool.logger.warning", lambda *a: warn.append(a))

    pool.schedule_release_request_resources(["a"])
    pool.schedule_release_request_resources(["a"])
    await asyncio.wait_for(healthy.started.wait(), timeout=2.0)
    healthy.started.clear()
    pool.schedule_release_request_resources(["b"])
    pool.schedule_release_request_resources(["c", "b"])
    await asyncio.wait_for(healthy.started.wait(), timeout=2.0)

    assert healthy.calls == [["a"], ["b", "c"]]
    assert hung.calls == [["a"]]

    await _settle(pool)
    assert hung.calls == [["a"], ["b", "c"]]
    assert len(warn) == 2  # both of the hung replica's RPCs timed out


@pytest.mark.asyncio
@pytest.mark.parametrize("removed", [False, True])
async def test_replica_exit_clears_pending_releases(monkeypatch, make_pool, removed) -> None:
    replica = _Replica(hang=True, fail=not removed)
    pool = make_pool(replica)

    pool.schedule_release_request_resources(["a"])
    await asyncio.wait_for(replica.started.wait(), timeout=2.0)
    pool.schedule_release_request_resources(["b"])
    warn = []
    monkeypatch.setattr("vllm_omni.engine.stage_pool.logger.warning", lambda *a: warn.append(a))
    if removed:
        pool.clients[0] = None
    replica.gate.set()
    await _settle(pool)

    assert replica.calls == ([["a"]] if removed else [["a"], ["b"]])
    assert not pool._pending_releases
    assert not pool._release_flushers
    replica.fail = False
    pool.schedule_release_request_resources(["c"])
    await _settle(pool)
    assert len(warn) == (0 if removed else 2)
    if not removed:
        assert replica.calls == [["a"], ["b"], ["c"]]
