# RFC: Pipeline-configured static stage transitions

Status: Proposed; scope and API pending community review.

This document proposes an API and runtime contract. The implementation described
below is a local prototype, separate from this documentation PR.

## Motivation

[Roadmap #8264, item 14](https://github.com/vllm-project/vllm-omni/issues/8264#event-32671404704)
calls for configured stage transitions with a sequential default, applied consistently
to forwarding, async-chunk prewarming, completion and cleanup.

Changing only the forwarding target leaves other paths inconsistent. For example,
a configured 0 -> 2 handoff fails if the receiving worker still constructs keys
for stage 1. Reordering also makes the largest stage ID an invalid completion rule.
The frontend, orchestrator and transport endpoints need one control topology.

## Proposed Change

### 1. Overview

Add `PipelineConfig.stage_transitions` and resolve it into an immutable
`StageRouting` chain starting at stage 0. Existing frontend, Orchestrator,
StagePool and worker paths use that chain. Stage IDs retain their configuration
and sampling-array indices.

The first release supports default/explicit sequential, skip, reorder and
single-stage routes, subject to registered model dependencies. Omitted stages
remain initialized and subject to existing startup checks, but receive no
inference submission or prewarming.

A chain preserves the existing single-successor request lifecycle. Branches,
joins, cycles, token-dependent routing, initialization pruning and non-sequential
duplex are outside this release. Sequential duplex keeps its
session contract; non-sequential duplex fails at startup. Existing replica
selection and scheduling remain unchanged.

This proposal changes the configured chain. [#6302](https://github.com/vllm-project/vllm-omni/issues/6302)
separately proposes output-contract execution planning and model-local pruning;
[#504](https://github.com/vllm-project/vllm-omni/issues/504) proposes model-selected
dynamic transitions. Multi-replica scheduling and queue blocking remain with
[#3481](https://github.com/vllm-project/vllm-omni/issues/3481) and
[#4680](https://github.com/vllm-project/vllm-omni/issues/4680).

### 2. Acceptance criteria

- R1 — Compatibility: `None` preserves sequential routing and existing supported
  success/cancellation behavior. Explicit edges replace the sequence completely.
- R2 — Admission: Reject invalid topology and unsatisfied active dependencies at
  construction/startup. Reject unreachable endpoints, invalid output-stage selections
  and insufficient sampling arrays before dispatch.
- R3 — Forwarding: Ordinary, segment-final, CFG-ready and KV-ready handoffs use
  the same successor. KV-ready cannot submit twice; first audio reaches the
  configured downstream accumulator.
- R4 — Prewarming: Follow the admitted prefix and advertise the bound producer
  replica. Bridge-fed stages bind first and submit when conditioning arrives.
  Stages outside the prefix receive no inference submission or prewarming.
- R5 — Ordering: Use actual route endpoints while retaining per-request chunk
  counters, segment boundaries, terminal markers and generation isolation.
- R6 — Completion: Processed and empty/raw terminal paths share the completion
  predicate. Non-streaming async-chunk requests also await submitted upstream
  stages; selected outputs and endpoints follow route order.
- R7 — Cancellation: Prevent submission after cancellation, including across
  replica awaits and ID reuse. Repeated cleanup shares its owner operation;
  preserve CFG ownership and state needed to complete or retry failed release.

### 3. Design

#### Configuration

Append one field to the registered `PipelineConfig`:

```python
stage_transitions: tuple[tuple[int, int], ...] | None = None
```

A skip-route example:

```python
from vllm_omni.config.stage_config import PipelineConfig, StagePipelineConfig

pipeline = PipelineConfig(
    model_type="example",
    stages=(
        StagePipelineConfig(stage_id=0, model_stage="encoder"),
        StagePipelineConfig(stage_id=1, model_stage="unused"),
        StagePipelineConfig(
            stage_id=2,
            model_stage="decoder",
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
        ),
    ),
    stage_transitions=((0, 2),),
)
```

The following values assume matching stage declarations and dependencies:

| Value | Execution |
| --- | --- |
| `None` | Existing 0 -> 1 -> 2 sequence |
| `((0, 2),)` | 0 -> 2 |
| `((2, 1), (0, 2))` | 0 -> 2 -> 1; stage 1 declares final output; edge order is irrelevant |
| `()` | Stage 0 only; stage 0 must declare a final output |

Explicit input must be a tuple of two-integer tuples, excluding bool. Stage IDs
must be ordered contiguous indices 0..N-1. Reject invalid IDs, self/duplicate
edges, branches, joins, cycles and edges unreachable from stage 0. The chain
must end at a declared final-output stage. Each active stage's `input_sources`
must occur earlier on the chain. Registered processors still determine payload
compatibility. `None` keeps the existing topology validation.

Model developers declare the route in the registry. Deployment devices, replicas
and connector `edges` remain deployment concerns, following the existing
[configuration ownership](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/architecture_overview/#configuration-and-runtime-resolution).
Control transitions and data dependencies describe separate facts.

Configuration resolution must also compose injected stages with the route.
For `--forced-aligner`, append the aligner after the actual route tail and use
that tail as its input source; extend explicit transitions, while keeping `None`
as `None`. Typed and compatibility launch representations share this effective
pipeline before connector projection and engine route construction. The
aligner processor still requires a compatible audio payload from that tail.

The route is fixed for an engine's lifetime. A topology change requires draining
or cancelling requests and restarting participants with matching topology and
deployment. No CLI/YAML route override, hot reload or version negotiation is added.
Headless registration does not compare complete resolved pipelines; matching
participant configuration remains a deployment prerequisite.
A rollback restores the original pipeline stage definitions, dependencies and
compatible deployment, then restarts participants. Setting `None` alone changes
control order; it cannot make incompatible model or connector contracts valid.

#### Request and runtime behavior

`StageRouting.path_to(endpoint)` and `next_stage(stage, endpoint)` are pure queries.
Validate configuration and public/IPC input at admission; internal paths use the
validated chain. Keep validation maps temporary and derive request prefixes.

A client request endpoint must be reachable and declare a final output. Selected outputs
must belong to its prefix and include the endpoint. Modalities select the last
matching output in route order; an explicit modality available only on an inactive
stage is rejected. Streaming updates retain their admitted endpoint and outputs.

For 0 -> 2 -> 1, a request ending at output stage 2 executes only 0 -> 2. Its
public sampling array still has one slot per registered stage, including stage 1.
The IPC admission check separately requires coverage through the highest stage ID
in the prefix. Stage 1 receives no submission or prewarming for this request.
Worker connector endpoints remain fixed for the full chain. A producer may publish
unused payloads past a request's endpoint; cleanup follows the backend contract
below. This proposal does not add per-request data-plane pruning.

All handoff paths use the successor query, preserving existing CFG/KV prerequisites
and KV-ready deduplication. First-audio delivery targets that successor. Async
prewarming walks adjacent edges of the same request prefix. Bind the producer
through StagePool before advertising its address; bridge-fed producers receive
real conditioning before their first submission.
Prewarming a downstream receiver requires stage-0 token IDs; embeds-only input
fails that path. Empty bridge input must abort receivers already prewarmed. A diffusion
processor returning `None` or an empty list fails its request and uses the same
automatic cleanup, including request-scoped reporting of release failure.

Dispatch checks request identity and closing state before and after replica
awaits. A lost bound producer fails its request; changing replicas after receiver
prewarming would leave the receiver pointing to the old producer. Chunk/segment
ordering remains owned by the existing per-request protocol.

Project source/target IDs into copied connector specifications, including local,
distributed, multi-API and headless startup, V1 adapters and MRv2 workers. Preserve
backend extras. An explicit `None` endpoint means no peer; missing legacy endpoint
fields retain their old defaults. Route projection must not create omitted edges.
Existing local IPC/SHM defaults remain. A deployment selecting a payload backend
must map the active pairs explicitly or through `default_connector`; old numeric
edges are not automatically remapped. A middle worker using one connector
requires the same backend in both directions.
Streaming-input requests with a native MRv2 downstream receiver remain unsupported;
native AR-to-DiT KV deployment restrictions also apply. V1 async transport stays with the
scheduler adapter; native MRv2 transport stays with the worker data plane.

#### Completion and cancellation

Use one completion predicate for processed and raw/empty terminal outputs:

```text
all selected final outputs finished
AND
(not async-chunk OR streaming input OR all submitted stages finished)
```

Hold the endpoint terminal output until the predicate is true, so late upstream
usage and stop reasons arrive. Segment boundaries do not terminate the request;
duplex session ownership continues to determine its lifetime.

The existing cleanup Future represents both the operation and closing state.
Expand CFG parent/companion IDs before claiming cleanup; join an existing companion
cleanup without replacing its Future. Cancelling a duplicate waiter leaves the
owner running. Broadcast cleanup to potentially owning pools: abort may already
have removed bindings, and CFG/transport resources may outlive the visible prefix.
Inactive stages can receive cleanup control calls.

Respect each backend's cleanup contract before reporting successful termination
or abort. Queue publication fences behind pending puts. For tested SHM/NIXL paths,
wait for confirmed release; legacy backends retain per-payload lifecycle after the
local fence. Their ACK does not certify physical release. A release failure retains
closed state and CFG relationships, reports a request-scoped error, and makes
the internal abort RPC report `success=False`; the public async abort API raises
on that failure and returns `None` on success. The internal ID remains unavailable
until cleanup succeeds.

Derive selected-output completion from `finished_stage_ids`, submitted membership
from `stage_submit_ts`, replica destinations from existing bindings, and closing
state from the cleanup Future. Configuration/query code lives in
`config/stage_config.py` and `config/stage_routing.py`; lifecycle and I/O effects
remain with existing orchestrator, pools, adapters and workers.

### 4. Validation

The frozen prototype passed 2014 CPU tests and 3 subtests across 32 modules;
one HF-config-dependent test was skipped. Applicable changed-file hooks, including
mypy, passed. A focused command is:

```bash
python -m pytest tests/config/test_stage_routing.py \
  tests/engine/test_configured_stage_routing.py \
  tests/worker/test_configured_connector_routing.py \
  tests/config/test_forced_aligner_injection.py \
  -m 'core_model and cpu' --run-level=core_model -q
```

Use Python 3.12, vLLM 0.31.0 and matching dependencies; parent-death checks require
Linux. The [prototype record](#prototype-validation-record) identifies the frozen source
and the scope of these local results. The command above requires that prototype;
the proposed field and its tests are not introduced by this documentation PR.

| Coverage | Evidence |
| --- | --- |
| R1-R2 | Default/explicit/skip/reorder/single-stage, invalid input, dependencies, output selection and legacy-backend CPU regressions |
| R3-R4 | Ordinary/CFG/KV handoff, first audio, bound producer and prewarming regressions; actual skip/reorder execution |
| R5 | Cross-node native NIXL exact GPU values, ordered chunks, terminal markers and key isolation |
| R6-R7 | Completion, late upstream, empty diffusion input/release failure, cleanup ownership/retry and ID-reuse regressions |
| Injected stages | Actual Qwen3-TTS + ForcedAligner: default/skip in V1 sync and MRv2 async; finite audio and nine ordered word intervals in each case |
| Distributed lifecycle | Fresh Qwen3-TTS V1/MRv2 0 -> 2: public abort, both-node resource checks, ID reuse, bound-replica failure and surviving requests |
| Deployment consistency | Test preflight compares all 27 production hashes and complete resolved pipelines on both nodes; deliberately mismatched model contract rejected |
| Historical model coverage | Qwen2.5-Omni 0 -> 2 -> 1 and Breeze-TTS-2 CFG/SHM retain their original recorded source revisions |

Frozen baseline: `c548a110a5a9bc278c39cfa81e673a8d505a24dd`.
Current patch SHA256: `c6442e41999da324ae44b9538cac2fcb1f71c6c1a7cd8044975baf720b239b4e`.
All 27 production files matched on both nodes for the recorded hardware tests.
Hardware harnesses and raw logs remain local acceptance artifacts. The prototype
CPU regressions use the existing CI sweep; no new GPU CI job is proposed.

NIXL hardware coverage is SDK 1.5/UCX TCP/cuda_copy, not RDMA or other backends.
Native tests include 512 MiB reader-exit and live-READ cancellation, followed by
independent exact reads. Configuration preflight is a test/deployment check,
not runtime negotiation. Finite audio and ordered timestamps do not certify
perceptual quality or alignment accuracy.

### 5. Review questions and rollout

1. Agree static-chain scope, stage-0 entry and the registered `stage_transitions`
   interface for the two claimed bullets, including its boundary with #6302.
2. Agree whether the additional transport changes below are prerequisites or
   separate submissions. Preserve default-backend compatibility in either case.
3. Agree merge coverage and acceptable transport cost for target deployments.

Keep R1-R7 coherent in one routing change, including startup endpoint projection.
Review independent startup/process ownership fixes separately, even when they
share files with routing. Required fixes must precede the route change or have
an explicit dependency. Review L2 separately: it strengthens the NIXL ownership
contract and changes its resource/performance policy. The roadmap claim does not
require this stronger guarantee for every backend. Retest each resulting split;
the current combined patch does not certify a standalone routing patch.

The prototype currently bundles two additional transport goals:

| Goal | Implementation and limit |
| --- | --- |
| L1 — Confirmed release | Queue/RPC acknowledgement waits for live SHM/NIXL owners; legacy full-payload, native KV and other backend protocols retain their contracts |
| L2 — NIXL remote-owner loss | Private producer contexts isolate request prefixes; source tensors remain held until native teardown returns. Normal completed transfers can reuse one unregistered idle context per connector |

L2 tests invoked lifecycle cleanup after reader exit and live-READ cancellation,
reclaimed 512 MiB, then verified key reuse and independent reads. Claimed payloads
are not reclaimed by lease expiry; request cleanup must retire their contexts.
Synchronous SDK teardown has no universal hard deadline. Source publication waits
for CUDA readiness with a device fence.

The local native fixture uses 32 KiB chunks and eight prefixes, publishing all
chunks before launching receiver READs. Private contexts achieved 2.09 MiB/s versus
11.31 for a shared-context normal-completion reference; put p95 was
105.5 versus 0.52 ms. The private fixture observed eight producer contexts and
a process increase of 64 threads and 52.8 MiB RSS from its own baseline; the
shared fixture observed 24 threads and 14.8 MiB RSS growth. These process deltas
include fixture threads and buffers, not just SDK contexts. This burst-transfer
fixture is not inference throughput.

Real-model measurements use one MRv2 Qwen3-TTS instance on exclusive A6000 GPU 5,
one decoder replica and `max_num_seqs=2`. Frontend concurrency 8 does not mean
eight active source contexts; the observed production peak was two. Requests use
the same prompt, `max_tokens=64` and 122880 observed audio samples each. Production
is measured before the shared reference, with one discarded warm-up and two measured batches
per policy/concurrency. Cells below show production / shared-context reference:

| Concurrent requests | Requests/s | Mean time to first audio (s) |
| --- | --- | --- |
| 1 | 0.271 / 0.273 | 0.234 / 0.225 |
| 4 | 0.502 / 0.515 | 2.392 / 2.326 |
| 8 | 0.500 / 0.495 | 6.487 / 6.516 |

The reference runs normal completion only and lacks L2's loss isolation. Both
policies keep the production CUDA fence; this comparison measures context policy,
not the fence independently or the entire patch against unmodified upstream.
Policy order and sample count limit conclusions; no general performance-parity
claim is made. The [prototype record](#prototype-validation-record) summarizes measurement
settings and evidence boundaries. Raw traces are local artifacts, rather than
reviewable tests submitted in this PR.

### 6. References

- [Roadmap item 14 and claim](https://github.com/vllm-project/vllm-omni/issues/8264#event-32671404704)
- [Architecture and configuration ownership](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/architecture_overview/)
- [Async chunk and prewarming](https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/async_chunk/)
- [Configuration refactor #6500](https://github.com/vllm-project/vllm-omni/issues/6500)
- [Request-selected execution #6302](https://github.com/vllm-project/vllm-omni/issues/6302), [model-selected transitions #504](https://github.com/vllm-project/vllm-omni/issues/504)
- [Orchestrator refactor #3481](https://github.com/vllm-project/vllm-omni/issues/3481), [queue blocking #4680](https://github.com/vllm-project/vllm-omni/issues/4680)
- [Standalone NIXL connector RFC #6160](https://github.com/vllm-project/vllm-omni/issues/6160)
- [RFC issue template](https://github.com/vllm-project/vllm-omni/blob/main/.github/ISSUE_TEMPLATE/750-RFC.yml), [linked design template](https://docs.google.com/document/d/1jcgR3cDaUQH3VczD4ZcKaJAoYWHjCmnYzHkCNyz-9fk/edit)

## Feedback Period

At least one week after publication.

## Prototype validation record

The reported implementation results belong to a frozen local prototype based on
`c548a110a5a9bc278c39cfa81e673a8d505a24dd`, with patch SHA256
`c6442e41999da324ae44b9538cac2fcb1f71c6c1a7cd8044975baf720b239b4e`.
It contains 54 changed/new files, including 27 production files. The two nodes
used identical production hashes and resolved pipelines for the recorded tests.
These results describe that combined prototype; they are not test results for
this documentation PR or for a future split implementation.

- CPU: 2014 passed, one skipped, three subtests passed across 32 modules, using
  Python 3.12 and vLLM 0.31.0. The skipped case requires an unavailable HF model
  config. Applicable changed-file pre-commit hooks, including mypy, passed.
- Configuration examples: default, skip, reorder and single-stage routes passed
  actual `PipelineConfig`/`StageRouting` validation with CUDA devices hidden.
- ForcedAligner: Qwen3-TTS default/skip routes ran in V1 sync and MRv2 async modes;
  each returned finite audio and nine ordered word intervals. This verifies
  execution and output structure, not perceptual quality or alignment accuracy.
- Distributed lifecycle: Qwen3-TTS V1/MRv2 `0 -> 2` requests covered cancellation,
  external-ID reuse, bound-replica loss and continuing service on a surviving
  replica. Complete resolved-pipeline preflight was a test/deployment check,
  rather than a production handshake.
- Native transport: A6000/L20 nodes, NIXL 1.5 and UCX `tcp,cuda_copy,self`.
  Exact tensor/chunk tests covered ordered delivery, empty terminal markers and
  generation isolation. Reader-exit and live-READ cancellation tests used
  512 MiB payloads and explicitly invoked lifecycle cleanup before testing
  key reuse and independent reads. RDMA and other SDK/backend versions were
  not validated.
- Model performance: `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, MRv2, NIXL async chunk,
  one exclusive A6000, one decoder replica and `max_num_seqs=2`. The same
  prompt used `temperature=0`, `seed=42`, `max_tokens=64` and codec-EOS
  logit bias `{2150: -100.0}`; each request returned 122880 finite audio samples.
  Each policy/concurrency had one warm-up and two measured batches: 26 warm-up
  and 52 measured requests in total. 116 process-ownership samples observed
  only the test's GPU processes. Production preceded the shared reference;
  both retained the production CUDA fence.

The hardware harnesses, raw traces and combined source patch are not published
in this PR. An implementation submission must provide its regression tests,
reproduction setup and evidence for its exact source revision, and repeat the
required checks after splitting the prototype. The numbers above do not
establish whole-patch performance parity with upstream.

## Authorship

AI assistance: Codex assisted implementation, testing and drafting. The
contributor reviewed the RFC before publication.
