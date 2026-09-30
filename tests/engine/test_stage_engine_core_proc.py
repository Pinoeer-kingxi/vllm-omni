# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
from vllm.v1.engine.core import EngineCoreProc

from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_preprocess_add_request_preserves_omni_fields(mocker):
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    request = SimpleNamespace(
        request_id="internal",
        external_req_id="external",
        additional_information={"conditioning": "payload"},
    )
    scheduler_request = SimpleNamespace()

    mocker.patch.object(
        EngineCoreProc,
        "preprocess_add_request",
        return_value=(scheduler_request, 3),
    )
    result, current_wave = engine.preprocess_add_request(request)

    assert result is scheduler_request
    assert current_wave == 3
    assert result.external_req_id == "external"
    assert result.additional_information == {"conditioning": "payload"}


def test_talker_hasher_uses_resolved_scheduler_hash_size_and_configured_algorithm():
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256_cbor
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash

    from vllm_omni.request import OmniRequest

    init_none_hash(sha256_cbor)
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.scheduler = SimpleNamespace(hash_block_size=2)
    engine.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16, prefix_caching_hash_algo="sha256_cbor", enable_prefix_caching=True),
        model_config=SimpleNamespace(model_stage="talker", model_arch="Qwen3OmniMoeForConditionalGeneration"),
    )
    engine._init_talker_request_hasher()
    request = OmniRequest(
        request_id="resolved-hash",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        block_hasher=engine.request_block_hasher,
    )
    assert len(request.block_hashes) == 1
    hashes = request.block_hashes
    request.block_hashes = []
    assert hashes == get_request_block_hasher(2, sha256_cbor)(request)


def test_non_talker_stage_keeps_native_request_hasher():
    engine = StageEngineCoreProc.__new__(StageEngineCoreProc)
    engine.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        model_config=SimpleNamespace(model_stage="thinker"),
    )
    engine.request_block_hasher = original = object()
    engine._init_talker_request_hasher()
    assert engine.request_block_hasher is original
