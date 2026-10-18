# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request admission, abort ownership and cleanup retries on the default route."""

import asyncio
from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams

from tests.engine.test_orchestrator import (
    FakePromptRequest,
    FakeStageClient,
    _build_request_output,
    _build_stage_pools,
)
from vllm_omni.engine.errors import ResourceReleaseError
from vllm_omni.engine.messages import ErrorMessage, OutputMessage, StageSubmissionMessage
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState
from vllm_omni.engine.stage_pool import RequestClosedError, StageUnavailableError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _direct_orchestrator(*, async_chunk=False):
    final_stage_id = 2
    clients = [FakeStageClient(final_output=(sid == final_stage_id)) for sid in range(3)]
    pools = _build_stage_pools([[client] for client in clients])
    orch = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=asyncio.Queue(),
        stage_pools=pools,
        async_chunk=async_chunk,
    )
    state = OrchestratorRequestState(
        request_id="r",
        sampling_params_list=[SamplingParams(max_tokens=1) for _ in pools],
        final_stage_id=final_stage_id,
        final_output_stage_ids={final_stage_id},
        prompt={"prompt_token_ids": [1, 2]},
    )
    orch.request_states["r"] = state
    return orch, state, clients


def _submission(**overrides):
    defaults = dict(
        type="add_request",
        request_id="r",
        prompt=FakePromptRequest(request_id="r", prompt_token_ids=[1], resumable=False),
        original_prompt={},
        output_prompt_text=None,
        request_timestamp=1.0,
        preprocess_ms=0.0,
        enqueue_ts=0.0,
        sampling_params_list=[SamplingParams(max_tokens=1) for _ in range(3)],
        final_stage_id=2,
        final_output_stage_ids=[2],
    )
    return StageSubmissionMessage(**(defaults | overrides))


@pytest.mark.parametrize("selected", [set(), {0, 1}])
def test_completion_derives_selected_outputs_from_stage_termination(selected):
    state = OrchestratorRequestState(
        request_id="r", final_stage_id=1, final_output_stage_ids=selected, finished_stage_ids={2}
    )
    assert not state.final_outputs_finished
    state.finished_stage_ids.add(0)
    assert not state.final_outputs_finished
    state.finished_stage_ids.add(1)
    assert state.final_outputs_finished


@pytest.mark.asyncio
async def test_cancel_while_waiting_for_replica_stops_old_request_admission(mocker):
    orch, state, clients = _direct_orchestrator(async_chunk=True)
    pool = orch.stage_pools[2]
    pool.add_client("tcp://127.0.0.1:5555", clients[2], replica_id=0)
    polled = asyncio.Event()

    def no_replicas(_stage_id):
        polled.set()
        return SimpleNamespace(replicas=[])

    pool.attach_hub(SimpleNamespace(get_replicas_for_stage=no_replicas))
    pool.attach_load_balancer(SimpleNamespace(select=lambda task, replicas: 0))
    task = asyncio.create_task(
        orch._prewarm_async_chunk_stages(
            "r", FakePromptRequest(request_id="r", prompt_token_ids=[1], resumable=False), state
        )
    )
    await asyncio.wait_for(polled.wait(), timeout=1)
    await orch._cleanup_request_ids(["r"], abort=True)
    new_state = OrchestratorRequestState(request_id="r")
    orch.request_states["r"] = new_state
    pool.bind("r", "tcp://127.0.0.1:5555")
    assert not await asyncio.wait_for(task, timeout=1)
    assert orch.request_states["r"] is new_state
    assert pool.get_bound_replica_id("r") == 0
    assert not clients[2].add_request_calls
    await orch._cleanup_request_ids(["r"])


@pytest.mark.asyncio
async def test_duplicate_cleanup_waiter_does_not_cancel_abort_owner(mocker):
    orch, state, clients = _direct_orchestrator(async_chunk=True)
    orch.stage_pools[0].select_replica_id("r")
    started, finish = asyncio.Event(), asyncio.Event()

    async def abort(_ids):
        started.set()
        await finish.wait()

    abort_call = mocker.patch.object(clients[0], "abort_requests_async", side_effect=abort)
    owner = asyncio.create_task(orch._cleanup_request_ids(["r"], abort=True))
    await asyncio.wait_for(started.wait(), timeout=1)
    waiter = asyncio.create_task(orch._cleanup_request_ids(["r"], abort=True))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert state.closing and not state.cleanup_future.done()
    with pytest.raises(RequestClosedError):
        await orch.stage_pools[0].submit_update("r", state, SimpleNamespace())
    await orch._handle_add_request(_submission())
    error = orch.output_async_queue.get_nowait()
    assert isinstance(error, ErrorMessage) and error.status_code == 409
    assert orch.request_states["r"] is state
    finish.set()
    await asyncio.wait_for(owner, timeout=1)
    assert abort_call.await_count == 1
    assert "r" not in orch.request_states


@pytest.mark.asyncio
@pytest.mark.parametrize("own_release_fails", [False, True])
async def test_automatic_cleanup_joins_failed_owner_and_reports_only_failed_requests(mocker, own_release_fails):
    orch, state, _ = _direct_orchestrator()
    other_state = OrchestratorRequestState(request_id="s")
    orch.request_states["s"] = other_state
    entered, second_entered, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    failures = {"r", "s"} if own_release_fails else {"r"}

    async def release(ids):
        if ids == ["r"]:
            entered.set()
            await finish.wait()
        else:
            second_entered.set()
        if failures.intersection(ids):
            raise ResourceReleaseError("READ ownership remains active")

    mocker.patch.object(orch.stage_pools[0], "release_request_resources", side_effect=release)
    owner = asyncio.create_task(orch._cleanup_request_ids(["r"], abort=True))
    await asyncio.wait_for(entered.wait(), timeout=1)
    automatic = asyncio.create_task(orch._cleanup_request_ids(["r", "s"]))
    await asyncio.wait_for(second_entered.wait(), timeout=1)
    await asyncio.sleep(0)
    assert not automatic.done()
    finish.set()
    with pytest.raises(ResourceReleaseError):
        await asyncio.wait_for(owner, timeout=1)
    await asyncio.wait_for(automatic, timeout=1)
    errors = []
    while not orch.output_async_queue.empty():
        errors.append(orch.output_async_queue.get_nowait())
    assert {error.request_id for error in errors} == failures
    assert all(isinstance(error, ErrorMessage) and not error.fatal for error in errors)
    assert state.closing and orch.request_states["r"] is state
    assert ("s" in orch.request_states) is own_release_fails
    failures.clear()
    await orch._cleanup_request_ids(["r", "s"], abort=True)
    assert not orch.request_states


@pytest.mark.asyncio
@pytest.mark.parametrize("release_fails", [False, True])
async def test_parent_cleanup_joins_completed_companion_cleanup(mocker, release_fails):
    orch, state, _ = _direct_orchestrator()
    companion_id = "r__neg"
    companion = OrchestratorRequestState(request_id=companion_id)
    orch.request_states[companion_id] = companion
    orch._cfg_tracker.register_companion("r", "neg", companion_id)
    orch._cfg_tracker.set_companion_output(companion_id, {"conditioning": 1})
    orch._cfg_tracker.on_companion_completed(companion_id)
    entered, parent_entered, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def release(ids):
        if ids == [companion_id]:
            entered.set()
            await finish.wait()
            if release_fails:
                raise ResourceReleaseError("READ ownership remains active")
        else:
            parent_entered.set()

    release_call = mocker.patch.object(orch.stage_pools[0], "release_request_resources", side_effect=release)
    owner = asyncio.create_task(orch._cleanup_request_ids([companion_id]))
    await asyncio.wait_for(entered.wait(), timeout=1)
    companion_future = companion.cleanup_future
    parent = asyncio.create_task(orch._cleanup_request_ids(["r"], abort=True))
    try:
        await asyncio.wait_for(parent_entered.wait(), timeout=1)
        assert companion.cleanup_future is companion_future
        assert orch.request_states["r"] is state and state.closing
        assert not parent.done()
    finally:
        finish.set()
        results = await asyncio.wait_for(asyncio.gather(owner, parent, return_exceptions=True), timeout=1)
    assert [call.args[0] for call in release_call.await_args_list] == [[companion_id], ["r"]]
    assert results[0] == []
    if release_fails:
        assert isinstance(results[1], ResourceReleaseError)
        assert orch.request_states["r"] is state and state.closing
        assert companion.closing and orch._cfg_tracker.has_companions("r")
        release_fails = False
        await orch._cleanup_request_ids(["r"], abort=True)
        assert set(release_call.await_args_list[-1].args[0]) == {"r", companion_id}
    else:
        assert results[1] == []
    assert not orch.request_states


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_id", ["r", "r__neg"])
async def test_failed_cfg_cleanup_retry_releases_the_complete_owner_batch(mocker, retry_id):
    orch, state, _ = _direct_orchestrator()
    companion_id = "r__neg"
    companion = OrchestratorRequestState(request_id=companion_id)
    orch.request_states[companion_id] = companion
    orch._cfg_tracker.register_companion("r", "neg", companion_id)
    release_call = mocker.patch.object(
        orch.stage_pools[0],
        "release_request_resources",
        side_effect=ResourceReleaseError("READ ownership remains active"),
    )
    with pytest.raises(ResourceReleaseError):
        await orch._cleanup_request_ids(["r"], abort=True)
    assert state.closing and companion.closing
    assert state.cleanup_future is companion.cleanup_future
    assert orch._cfg_tracker.has_companions("r")
    release_call.side_effect = None
    await orch._cleanup_request_ids([retry_id], abort=True)
    assert not orch.request_states
    assert all(set(call.args[0]) == {"r", companion_id} for call in release_call.await_args_list)


@pytest.mark.asyncio
async def test_failed_abort_can_be_retried_without_reopening_admission(mocker):
    orch, state, clients = _direct_orchestrator()
    orch.stage_pools[0].select_replica_id("r")
    abort = mocker.patch.object(clients[0], "abort_requests_async", side_effect=[RuntimeError("RPC unavailable"), None])
    with pytest.raises(RuntimeError, match="RPC unavailable"):
        await orch._cleanup_request_ids(["r"], abort=True)
    assert state.closing and orch.request_states["r"] is state
    with pytest.raises(RequestClosedError):
        await orch.stage_pools[0].pick("r", request_state=state)
    await orch._cleanup_request_ids(["r"], abort=True)
    assert abort.await_count == 2
    assert "r" not in orch.request_states


@pytest.mark.asyncio
async def test_lost_preselected_replica_cannot_be_reselected(mocker):
    orch, state, clients = _direct_orchestrator(async_chunk=True)
    pool = orch.stage_pools[2]
    pool.add_client("replica-1", FakeStageClient(final_output=True), replica_id=1)
    assert await pool.pick("r", request_state=state) == 0
    pool.evict_replica(0)
    with pytest.raises(StageUnavailableError):
        await pool.submit_initial("r", state, SimpleNamespace())
    await orch._handle_dead_replica(2, 0, RuntimeError("replica exited"))
    assert isinstance(orch.output_async_queue.get_nowait(), ErrorMessage)
    assert "r" not in orch.request_states
    assert not pool.clients[1].add_request_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("release_fails", [False, True])
async def test_empty_bridge_waits_for_cleanup_before_publishing_success(mocker, release_fails):
    orch, state, _ = _direct_orchestrator(async_chunk=True)
    entered, resume = asyncio.Event(), asyncio.Event()

    async def release(_ids):
        entered.set()
        await resume.wait()
        if release_fails:
            raise ResourceReleaseError("receiver still owns the payload")

    mocker.patch.object(orch.stage_pools[0], "release_request_resources", side_effect=release)
    task = asyncio.create_task(orch._forward_to_next_stage("r", 0, _build_request_output("r"), state))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert orch.output_async_queue.empty()
        assert state.closing and not state.cleanup_future.done()
    finally:
        resume.set()
        await asyncio.wait_for(task, timeout=1)
    result = orch.output_async_queue.get_nowait()
    if release_fails:
        assert isinstance(result, ErrorMessage)
        assert orch.request_states["r"] is state
    else:
        assert isinstance(result, OutputMessage) and result.finished
        assert "r" not in orch.request_states
    assert orch.output_async_queue.empty()


@pytest.mark.asyncio
async def test_abort_retry_preserves_tokens_after_transfer_release_failure(mocker):
    orch, state, clients = _direct_orchestrator()
    pool = orch.stage_pools[2]
    pool.select_replica_id("r")
    pool.output_processor.request_outputs = [_build_request_output("r", token_ids=[7, 8, 9], finished=False)]
    release = mocker.patch.object(
        orch.stage_pools[0],
        "release_request_resources",
        side_effect=[ResourceReleaseError("receiver still owns the payload"), None],
    )
    with pytest.raises(ResourceReleaseError):
        await orch._cleanup_request_ids(["r"], abort=True)
    assert orch.request_states["r"] is state and state.closing
    assert pool.get_bound_replica_id("r") is None
    outputs = await orch._cleanup_request_ids(["r"], abort=True)
    assert len(outputs) == 1 and outputs[0].finished
    assert list(outputs[0].engine_outputs.outputs[0].token_ids) == [7, 8, 9]
    assert outputs[0].replica_id == 0
    assert clients[2].abort_calls == [["r"]]
    assert release.await_count == 2 and "r" not in orch.request_states


@pytest.mark.asyncio
async def test_cancelled_prewarm_releases_bindings_without_resurrection():
    orch, state, clients = _direct_orchestrator(async_chunk=True)
    await orch._prewarm_async_chunk_stages(
        "r", FakePromptRequest(request_id="r", prompt_token_ids=[1, 2], resumable=False), state
    )
    await orch._cleanup_request_ids(["r"], abort=True)
    assert "r" not in orch.request_states
    assert all(pool.get_bound_replica_id("r") is None for pool in orch.stage_pools)
    assert not await orch._prewarm_async_chunk_stages(
        "r", FakePromptRequest(request_id="r", prompt_token_ids=[1, 2], resumable=False), state
    )
    assert len(clients[2].add_request_calls) == 1
