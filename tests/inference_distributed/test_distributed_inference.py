# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for marin.inference.distributed.

Three groups:

- Pure config / data-class tests (no I/O).
- Compile-cache resolution (no I/O).
- End-to-end smoke test against a stubbed engine + LocalClient, exercising
  the full pipeline build → execute → result iteration path.
"""
from __future__ import annotations

import gzip
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fray.local_backend import LocalClient
from marin.inference.distributed import (
    InferenceConfig,
    InferenceResult,
    ModelSpec,
    ResponseRecord,
    SamplingParams,
    compile_cache,
    inference,
    vllm_worker,
)
from marin.inference.distributed import input as input_module
from marin.inference.distributed.pipeline import assign_shard_ids, rotate_for_region

# ---------------------------------------------------------------------------
# ModelSpec.resolve_for_region
# ---------------------------------------------------------------------------


def test_model_spec_hf_id_passes_through():
    spec = ModelSpec(model="meta-llama/Llama-3-8B")
    assert spec.resolve_for_region("us-central1") == "meta-llama/Llama-3-8B"


def test_model_spec_marin_uri_resolves_per_region():
    spec = ModelSpec(model="marin://checkpoints/qwen3-8b/hf/step-1318")
    us = spec.resolve_for_region("us-central1")
    eu = spec.resolve_for_region("europe-west4")
    assert us == "gs://marin-us-central1/checkpoints/qwen3-8b/hf/step-1318"
    assert eu == "gs://marin-eu-west4/checkpoints/qwen3-8b/hf/step-1318"


def test_model_spec_gs_uri_matching_region_passes():
    spec = ModelSpec(model="gs://marin-us-central1/x/y/z")
    assert spec.resolve_for_region("us-central1") == "gs://marin-us-central1/x/y/z"


def test_model_spec_gs_uri_wrong_region_raises():
    spec = ModelSpec(model="gs://marin-us-central1/x/y/z")
    with pytest.raises(ValueError, match="bucket 'marin-us-central1'"):
        spec.resolve_for_region("europe-west4")


# ---------------------------------------------------------------------------
# InferenceConfig validation
# ---------------------------------------------------------------------------


def test_inference_config_rejects_empty_regions():
    with pytest.raises(ValueError, match="regions must be non-empty"):
        InferenceConfig(regions=[])


def test_inference_config_rejects_unknown_region():
    with pytest.raises(ValueError, match="Unknown region"):
        InferenceConfig(regions=["mars-central1"])


def test_inference_config_rejects_unknown_results_region():
    with pytest.raises(ValueError, match="Unknown results_region"):
        InferenceConfig(regions=["us-central1"], results_region="mars-central1")


def test_inference_config_defaults_match_zephyr_upstream():
    """The library defaults are intentionally pinned to upstream Zephyr's defaults."""
    cfg = InferenceConfig(regions=["us-central1"])
    assert cfg.heartbeat_timeout == 120.0
    assert cfg.max_shard_failures == 3
    assert cfg.max_shard_infra_failures == 20


def test_inference_config_default_worker_extras_includes_vllm_and_tpu():
    """Regression test: a 2026-05-20 real-cluster run failed with
    ``ModuleNotFoundError: No module named 'vllm'`` because the library
    submitted Zephyr worker jobs with no extras installed. The fix wires
    ``InferenceConfig.worker_extras`` → Fray ``EnvironmentConfig.extras`` →
    Iris ``EnvironmentSpec.extras`` so workers get vLLM installed at startup.
    Defaults must include both ``marin:vllm`` (the engine) and ``marin:tpu``
    (the JAX/libtpu stack); changing them silently would re-introduce the bug.
    """
    cfg = InferenceConfig(regions=["us-central1"])
    assert "marin:vllm" in cfg.worker_extras
    assert "marin:tpu" in cfg.worker_extras


def test_regional_job_spec_propagates_worker_extras():
    """The regional spec carries worker_extras forward so the in-process
    builder in ``regional_job._build_context`` can construct a Fray
    ``EnvironmentConfig`` for the Zephyr worker group.
    """
    from marin.inference.distributed.meta_coordinator import _make_regional_spec

    cfg = InferenceConfig(
        regions=["us-central1"],
        results_region="us-central1",
        worker_extras=("marin:vllm", "marin:tpu", "marin:custom"),
    )
    spec = _make_regional_spec(
        config=cfg,
        region="us-central1",
        input_files=["/tmp/a.jsonl.gz"],
        model_spec=ModelSpec(model="hf/test"),
        results_uri="file:///tmp/results",
        run_id="testrun",
    )
    assert spec.worker_extras == ("marin:vllm", "marin:tpu", "marin:custom")


def test_build_context_threads_worker_environment():
    """``regional_job._build_context`` must produce a ``ZephyrContext`` whose
    ``worker_environment.extras`` exactly matches ``RegionalJobSpec.worker_extras``.
    This is the boundary where the missing-vllm bug actually manifested.
    """
    from fray.local_backend import LocalClient
    from marin.inference.distributed.regional_job import RegionalJobSpec, _build_context

    spec = RegionalJobSpec(
        region="us-central1",
        results_uri="file:///tmp/results",
        input_files=("/tmp/a.jsonl.gz",),
        model_spec=ModelSpec(model="hf/test"),
        sampling=SamplingParams(),
        job_name="job",
        run_id="testrun",
        tpu_shapes=("v5p-8",),
        max_workers=1,
        worker_preemptible=True,
        heartbeat_timeout=120.0,
        max_shard_failures=3,
        max_shard_infra_failures=20,
        chunk_size=2000,
        compile_cache_uri_template=None,
        worker_extras=("marin:vllm", "marin:tpu"),
    )

    from fray import set_current_client

    client = LocalClient()
    try:
        with set_current_client(client):
            ctx = _build_context(spec)
    finally:
        client.shutdown(wait=True)

    assert (
        ctx.worker_environment is not None
    ), "ZephyrContext.worker_environment must be set so Zephyr propagates extras to worker jobs."
    assert list(ctx.worker_environment.extras) == ["marin:vllm", "marin:tpu"]


def test_build_context_omits_worker_environment_when_extras_empty_and_cache_disabled():
    """Empty extras + explicitly disabled compile cache (template="") produces
    ``worker_environment=None`` so Zephyr falls back to the cluster-default
    environment. The compile-cache env vars no longer get injected when the
    cache is opted out.
    """
    from fray.local_backend import LocalClient
    from marin.inference.distributed.regional_job import RegionalJobSpec, _build_context

    spec = RegionalJobSpec(
        region="us-central1",
        results_uri="file:///tmp/results",
        input_files=("/tmp/a.jsonl.gz",),
        model_spec=ModelSpec(model="hf/test"),
        sampling=SamplingParams(),
        job_name="job",
        run_id="testrun",
        tpu_shapes=("v5p-8",),
        max_workers=1,
        worker_preemptible=True,
        heartbeat_timeout=120.0,
        max_shard_failures=3,
        max_shard_infra_failures=20,
        chunk_size=2000,
        compile_cache_uri_template="",  # explicit opt-out
        worker_extras=(),
    )

    from fray import set_current_client

    client = LocalClient()
    try:
        with set_current_client(client):
            ctx = _build_context(spec)
    finally:
        client.shutdown(wait=True)

    assert ctx.worker_environment is None


def test_build_context_threads_compile_cache_env_vars_to_worker_environment():
    """The XLA compile-cache env vars must land on ``worker_environment.env_vars``
    so they reach the TPU worker process (which is where vLLM actually compiles).

    Regression test for a 2026-05-21 real-cluster integration-test failure
    (``infinttest-6``) where the cache directory remained empty after a full
    run: the env vars were being set on the regional CPU coordinator's
    ``os.environ`` rather than threaded through Fray's ``EnvironmentConfig``
    to the worker. Without the worker_environment carrying these vars, JAX
    and vLLM compile from scratch on every run.
    """
    from fray.local_backend import LocalClient
    from marin.inference.distributed.regional_job import RegionalJobSpec, _build_context

    spec = RegionalJobSpec(
        region="us-central1",
        results_uri="file:///tmp/results",
        input_files=("/tmp/a.jsonl.gz",),
        model_spec=ModelSpec(model="hf/test", engine_kwargs={"tensor_parallel_size": 4}),
        sampling=SamplingParams(),
        job_name="job",
        run_id="testrun",
        tpu_shapes=("v5p-8",),
        max_workers=1,
        worker_preemptible=True,
        heartbeat_timeout=120.0,
        max_shard_failures=3,
        max_shard_infra_failures=20,
        chunk_size=2000,
        compile_cache_uri_template=None,  # default cache prefix
        worker_extras=("marin:vllm",),
    )

    from fray import set_current_client

    client = LocalClient()
    try:
        with set_current_client(client):
            ctx = _build_context(spec)
    finally:
        client.shutdown(wait=True)

    assert ctx.worker_environment is not None
    env_vars = ctx.worker_environment.env_vars
    assert "JAX_COMPILATION_CACHE_DIR" in env_vars
    assert "VLLM_XLA_CACHE_PATH" in env_vars
    assert env_vars["JAX_COMPILATION_CACHE_DIR"] == env_vars["VLLM_XLA_CACHE_PATH"]
    assert env_vars["JAX_COMPILATION_CACHE_DIR"].startswith("gs://marin-us-central1/tmp/ttl=30d/vllm-cache/")
    assert env_vars.get("JAX_ENABLE_COMPILATION_CACHE") == "1"
    # Extras still threaded through alongside env vars.
    assert list(ctx.worker_environment.extras) == ["marin:vllm"]


def test_build_context_includes_compile_cache_env_vars_even_when_no_extras():
    """When extras are empty but the compile cache is on (default), the worker
    still needs ``worker_environment`` so the cache env vars reach it.
    """
    from fray.local_backend import LocalClient
    from marin.inference.distributed.regional_job import RegionalJobSpec, _build_context

    spec = RegionalJobSpec(
        region="us-central1",
        results_uri="file:///tmp/results",
        input_files=("/tmp/a.jsonl.gz",),
        model_spec=ModelSpec(model="hf/test"),
        sampling=SamplingParams(),
        job_name="job",
        run_id="testrun",
        tpu_shapes=("v5p-8",),
        max_workers=1,
        worker_preemptible=True,
        heartbeat_timeout=120.0,
        max_shard_failures=3,
        max_shard_infra_failures=20,
        chunk_size=2000,
        compile_cache_uri_template=None,
        worker_extras=(),
    )

    from fray import set_current_client

    client = LocalClient()
    try:
        with set_current_client(client):
            ctx = _build_context(spec)
    finally:
        client.shutdown(wait=True)

    assert ctx.worker_environment is not None
    assert "JAX_COMPILATION_CACHE_DIR" in ctx.worker_environment.env_vars
    assert list(ctx.worker_environment.extras) == []


def test_resolve_cache_uri_empty_template_disables_cache():
    """An empty-string template explicitly disables the compile cache."""
    from marin.inference.distributed import compile_cache

    spec = ModelSpec(model="hf/test")
    assert compile_cache.resolve_cache_uri(spec, "us-central1", template="") is None


# ---------------------------------------------------------------------------
# Input record validation
# ---------------------------------------------------------------------------


def test_input_validate_text_payload_ok():
    input_module.validate_record({"id": "1", "payload": {"kind": "text", "prompt": "hi"}})


def test_input_validate_messages_payload_ok():
    input_module.validate_record(
        {"id": "1", "payload": {"kind": "messages", "messages": [{"role": "user", "content": "hi"}]}}
    )


def test_input_validate_missing_id():
    with pytest.raises(ValueError, match="missing 'id'"):
        input_module.validate_record({"payload": {"kind": "text", "prompt": "hi"}})


def test_input_validate_unknown_kind():
    with pytest.raises(ValueError, match="unknown kind"):
        input_module.validate_record({"id": "1", "payload": {"kind": "image", "url": "x"}})


def test_input_validate_text_missing_prompt():
    with pytest.raises(ValueError, match="text payload missing 'prompt'"):
        input_module.validate_record({"id": "1", "payload": {"kind": "text"}})


def test_materialize_inline_input_records_per_file_controls_file_count(tmp_path):
    """`records_per_file` must determine the number of materialized files.

    Regression for the bug where the input module hard-coded the file chunk
    size to 5000 records, making ``InferenceConfig.shard_size`` a no-op for
    inline inputs. See also
    ``test_inference_inline_input_shard_size_controls_output_shard_count``.
    """
    output_dir = f"file://{tmp_path}/inputs"
    records = [{"id": f"r{i}", "payload": {"kind": "text", "prompt": str(i)}} for i in range(20)]
    paths = input_module.materialize_inline_input(records, output_dir=output_dir, records_per_file=5)
    assert len(paths) == 4, paths
    paths_one = input_module.materialize_inline_input(
        records, output_dir=f"file://{tmp_path}/inputs-one", records_per_file=1
    )
    assert len(paths_one) == 20


# ---------------------------------------------------------------------------
# Compile-cache resolution
# ---------------------------------------------------------------------------


def test_compile_cache_default_template():
    spec = ModelSpec(model="marin://checkpoints/m1")
    uri = compile_cache.resolve_cache_uri(spec, "us-central1", template=None)
    assert uri.startswith("gs://marin-us-central1/tmp/ttl=30d/vllm-cache/")


def test_compile_cache_hash_changes_with_engine_kwargs():
    a = ModelSpec(model="marin://m", engine_kwargs={"tensor_parallel_size": 4})
    b = ModelSpec(model="marin://m", engine_kwargs={"tensor_parallel_size": 8})
    assert compile_cache.model_cache_hash(a, "us-central1") != compile_cache.model_cache_hash(b, "us-central1")


def test_compile_cache_configure_env_no_op_when_uri_none(monkeypatch):
    env: dict[str, str] = {}
    compile_cache.configure_env(None, env=env)
    assert env == {}


def test_compile_cache_configure_env_sets_jax_and_vllm():
    env: dict[str, str] = {}
    compile_cache.configure_env("gs://marin-us-central1/tmp/ttl=30d/vllm-cache/xyz", env=env)
    assert env["JAX_COMPILATION_CACHE_DIR"] == "gs://marin-us-central1/tmp/ttl=30d/vllm-cache/xyz"
    assert env["VLLM_XLA_CACHE_PATH"] == "gs://marin-us-central1/tmp/ttl=30d/vllm-cache/xyz"
    assert env["JAX_ENABLE_COMPILATION_CACHE"] == "1"


# ---------------------------------------------------------------------------
# Per-region shuffle (rotation)
# ---------------------------------------------------------------------------


def test_assign_shard_ids_uses_sorted_position():
    files = ["b", "a", "c"]
    pairs = assign_shard_ids(files)
    # Sorted: ['a', 'b', 'c'] → shard IDs 0, 1, 2
    assert pairs == [(0, "a"), (1, "b"), (2, "c")]


def test_rotate_for_region_is_deterministic():
    items = [(i, f"shard-{i:02d}") for i in range(6)]
    a = rotate_for_region(items, "us-central1")
    b = rotate_for_region(items, "us-central1")
    assert a == b


def test_rotate_for_region_differs_across_regions():
    # 100-item list makes hash-collision-mod-N exceedingly unlikely between
    # two distinct region names. (Python's PYTHONHASHSEED randomization can
    # collide small modulos across regions in any given process.)
    items = [(i, f"shard-{i:03d}") for i in range(100)]
    a = rotate_for_region(items, "us-central1")
    b = rotate_for_region(items, "europe-west4")
    assert sorted(a) == sorted(b)  # same content
    assert a != b  # different order


def test_rotate_for_region_preserves_membership():
    items = [(i, f"shard-{i:02d}") for i in range(6)]
    rotated = rotate_for_region(items, "us-east1")
    assert set(rotated) == set(items)
    assert len(rotated) == len(items)


# ---------------------------------------------------------------------------
# Response extraction
# ---------------------------------------------------------------------------


@dataclass
class _FakeCompletion:
    text: str
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    cumulative_logprob: float | None = None
    logprobs: Any = None


@dataclass
class _FakeRequestOutput:
    outputs: list[_FakeCompletion]
    prompt_logprobs: Any = None
    num_cached_tokens: int | None = None


def test_extract_response_preserves_think_blocks_verbatim():
    """Reasoning markers must pass through; the worker MUST NOT strip them."""
    raw_text = "<think>plan: x then y</think>\nfinal answer"
    output = _FakeRequestOutput(outputs=[_FakeCompletion(text=raw_text, finish_reason="stop")])
    text, extras = vllm_worker._extract_response(output)
    assert text == raw_text
    assert extras["finish_reason"] == "stop"


def test_extract_response_omits_none_fields():
    output = _FakeRequestOutput(outputs=[_FakeCompletion(text="hello")])
    _, extras = vllm_worker._extract_response(output)
    # None-valued vLLM fields should not appear in extras at all.
    assert extras == {}


def test_extract_response_captures_request_level_fields():
    output = _FakeRequestOutput(
        outputs=[_FakeCompletion(text="hi", cumulative_logprob=-3.14)],
        num_cached_tokens=42,
    )
    _, extras = vllm_worker._extract_response(output)
    assert extras["num_cached_tokens"] == 42
    assert extras["cumulative_logprob"] == pytest.approx(-3.14)


# ---------------------------------------------------------------------------
# Stub engine for end-to-end testing
# ---------------------------------------------------------------------------


class _StubEngine:
    """Minimal engine satisfying `InferenceEngine` protocol.

    Echoes prompts back as ``"echo:<prompt>"`` for text mode, and for messages
    mode echoes the last user content. Returns objects shaped like vLLM's
    `RequestOutput`.
    """

    def __init__(self) -> None:
        self.generate_calls: list[Sequence[str]] = []
        self.chat_calls: list[Sequence[Sequence[dict[str, Any]]]] = []

    def generate(self, prompts: Sequence[str], sampling_params: Any) -> list[_FakeRequestOutput]:
        self.generate_calls.append(list(prompts))
        return [_FakeRequestOutput(outputs=[_FakeCompletion(text=f"echo:{p}", finish_reason="stop")]) for p in prompts]

    def chat(
        self,
        conversations: Sequence[Sequence[dict[str, Any]]],
        sampling_params: Any,
    ) -> list[_FakeRequestOutput]:
        self.chat_calls.append([list(c) for c in conversations])
        outputs = []
        for convo in conversations:
            last = next((m["content"] for m in reversed(list(convo)) if m["role"] == "user"), "")
            outputs.append(_FakeRequestOutput(outputs=[_FakeCompletion(text=f"chat-echo:{last}", finish_reason="stop")]))
        return outputs


@pytest.fixture
def stub_engine(monkeypatch):
    """Install a stub engine + identity sampling factory before each test."""
    engine = _StubEngine()
    vllm_worker.set_engine_factory(lambda model, kwargs: engine)
    vllm_worker.set_sampling_factory(lambda sp: sp)  # don't try to build a real vllm.SamplingParams
    yield engine
    vllm_worker.set_engine_factory(None)
    vllm_worker.set_sampling_factory(None)
    vllm_worker.reset_engine_cache()


@pytest.fixture
def local_fray_client():
    client = LocalClient()
    yield client
    client.shutdown(wait=True)


def _local_results_root(tmp_path: Path, monkeypatch) -> Path:
    """Redirect `marin_prefix_for_region` to a local file:// path under tmp_path.

    Lets the test exercise the real materialize / pipeline / read paths against
    a real filesystem (via fsspec's local backend) without touching GCS.
    """
    local_root = tmp_path / "marin-buckets"
    local_root.mkdir(exist_ok=True)
    monkeypatch.setattr(
        "marin.inference.distributed.api.marin_prefix_for_region",
        lambda region: f"file://{local_root}/marin-{region}",
    )
    monkeypatch.setattr(
        "marin.inference.distributed.compile_cache.marin_prefix_for_region",
        lambda region: f"file://{local_root}/marin-{region}",
    )
    return local_root


def test_inference_smoke_single_region(tmp_path, monkeypatch, stub_engine, local_fray_client):
    """End-to-end: single region, in-memory text input, stub engine, local fs."""
    from fray import set_current_client

    with set_current_client(local_fray_client):
        _local_results_root(tmp_path, monkeypatch)
        prompts = [{"id": f"p{i}", "payload": {"kind": "text", "prompt": f"hello-{i}"}} for i in range(8)]
        cfg = InferenceConfig(
            regions=["us-central1"],
            results_region="us-central1",
            max_workers_per_region=2,
            shard_size=4,
            job_name="smoketest",
            sampling=SamplingParams(max_tokens=4),
        )
        result = inference(model=ModelSpec(model="meta-llama/Llama-3-8B"), dataset=prompts, config=cfg)

    assert isinstance(result, InferenceResult)
    assert result.is_complete, f"missing shards: {result.missing_shards}"
    records = result.to_list()
    assert len(records) == len(prompts)
    by_id = {r.id: r for r in records}
    for prompt in prompts:
        assert by_id[prompt["id"]].response == f"echo:{prompt['payload']['prompt']}"
        assert by_id[prompt["id"]].extra.get("finish_reason") == "stop"


def test_inference_messages_payload_dispatch(tmp_path, monkeypatch, stub_engine, local_fray_client):
    """Verify the messages-kind dispatch path hits ``engine.chat``."""
    from fray import set_current_client

    with set_current_client(local_fray_client):
        _local_results_root(tmp_path, monkeypatch)
        prompts = [
            {
                "id": f"m{i}",
                "payload": {
                    "kind": "messages",
                    "messages": [{"role": "user", "content": f"hi-{i}"}],
                },
            }
            for i in range(4)
        ]
        cfg = InferenceConfig(
            regions=["us-central1"],
            results_region="us-central1",
            max_workers_per_region=1,
            shard_size=2,
            job_name="msgtest",
            sampling=SamplingParams(max_tokens=4),
        )
        result = inference(model=ModelSpec(model="meta-llama/Llama-3-8B"), dataset=prompts, config=cfg)

    assert result.is_complete
    assert stub_engine.chat_calls, "chat path was not exercised"
    assert not stub_engine.generate_calls, "generate path should not be used for messages"


def test_inference_mixed_payload_kinds_in_shard_yields_incomplete_result(
    tmp_path, monkeypatch, stub_engine, local_fray_client, caplog
):
    """Mixed payload kinds in one shard must fail the shard (not the whole run).

    Per decision #11, regional failures are non-fatal: the meta-coordinator
    logs the failure and returns an `InferenceResult` with ``missing_shards``
    populated. The run as a whole only fails when every region failed the
    same shard.
    """
    import logging

    from fray import set_current_client

    with set_current_client(local_fray_client), caplog.at_level(logging.WARNING):
        _local_results_root(tmp_path, monkeypatch)
        prompts = [
            {"id": "t1", "payload": {"kind": "text", "prompt": "x"}},
            {"id": "m1", "payload": {"kind": "messages", "messages": [{"role": "user", "content": "y"}]}},
        ]
        cfg = InferenceConfig(
            regions=["us-central1"],
            results_region="us-central1",
            max_workers_per_region=1,
            shard_size=2,
            job_name="mixedtest",
            sampling=SamplingParams(max_tokens=4),
        )
        result = inference(model=ModelSpec(model="meta-llama/Llama-3-8B"), dataset=prompts, config=cfg)

    assert not result.is_complete
    assert result.missing_shards == (0,)
    # The shard failure should surface in logs as a ValueError about mixed kinds.
    assert any(
        "mixed payload kinds" in record.getMessage() for record in caplog.records
    ), "Expected the shard failure reason to be logged."


def test_inference_skip_existing_does_not_recompute(tmp_path, monkeypatch, stub_engine, local_fray_client):
    """A second run over the same input should skip everything via output existence."""
    from fray import set_current_client

    with set_current_client(local_fray_client):
        _local_results_root(tmp_path, monkeypatch)
        prompts = [{"id": f"p{i}", "payload": {"kind": "text", "prompt": f"hi-{i}"}} for i in range(4)]
        cfg = InferenceConfig(
            regions=["us-central1"],
            results_region="us-central1",
            max_workers_per_region=1,
            shard_size=2,
            job_name="skiptest",
            sampling=SamplingParams(max_tokens=4),
        )

        result1 = inference(model=ModelSpec(model="meta-llama/Llama-3-8B"), dataset=prompts, config=cfg)
        assert len(stub_engine.generate_calls) > 0
        assert result1.is_complete

        # Re-running materializes a fresh inputs dir, so to actually test
        # skip_existing we need to point the second run at the same
        # results_uri. We do this by pinning the run_id via uuid mocking.
        # Reuse the same materialized input files for a second run targeting
        # the same results_uri by also pinning the job_name + manually crafting
        # the run prefix. Simpler: just run twice in a row with the same prompts
        # and observe that the second run sees the existing outputs. We do this
        # by monkeypatching uuid to give a stable run_id.
        import uuid as _uuid

        monkeypatch.setattr(_uuid, "uuid4", lambda: _StableUUID(result1.results_uri))

        # Re-running with the stable run_id should hit skip_existing on every shard.
        stub_engine.generate_calls.clear()
        result2 = inference(model=ModelSpec(model="meta-llama/Llama-3-8B"), dataset=prompts, config=cfg)

    assert result2.is_complete
    assert (
        not stub_engine.generate_calls
    ), "Second run should have hit skip_existing on every shard and made zero engine calls."


def test_inference_inline_input_shard_size_controls_output_shard_count(
    tmp_path, monkeypatch, stub_engine, local_fray_client
):
    """`cfg.shard_size` must drive the number of output shards for inline input.

    Regression for the failing real-cluster multi-region integration test, where
    32 prompts at ``shard_size=8`` produced only **one** output shard because
    the input materializer chunked by a hard-coded internal constant (5000)
    instead of honoring ``shard_size``. Per-region rotation and ``skip_existing``
    arbitration only matter when there is real shard granularity to rotate over.
    """
    from fray import set_current_client

    with set_current_client(local_fray_client):
        _local_results_root(tmp_path, monkeypatch)
        prompts = [{"id": f"p{i:03d}", "payload": {"kind": "text", "prompt": f"x-{i}"}} for i in range(32)]
        cfg = InferenceConfig(
            regions=["us-central1"],
            results_region="us-central1",
            max_workers_per_region=1,
            shard_size=8,
            job_name="shardsize-regression",
            sampling=SamplingParams(max_tokens=4),
        )
        result = inference(model=ModelSpec(model="meta-llama/Llama-3-8B"), dataset=prompts, config=cfg)

    assert result.is_complete, f"missing: {result.missing_shards}"
    output_files = result.list_output_files()
    assert len(output_files) == 4, (
        f"expected 4 output shards for 32 prompts / shard_size=8, " f"got {len(output_files)}: {output_files}"
    )
    records = result.to_list()
    assert len(records) == 32
    shard_ids = {r.shard for r in records}
    assert shard_ids == {0, 1, 2, 3}, shard_ids


def test_inference_inline_input_uneven_shard_size_rounds_up(tmp_path, monkeypatch, stub_engine, local_fray_client):
    """Non-divisible N: 10 prompts at shard_size=3 -> 4 shards (3,3,3,1)."""
    from fray import set_current_client

    with set_current_client(local_fray_client):
        _local_results_root(tmp_path, monkeypatch)
        prompts = [{"id": f"u{i}", "payload": {"kind": "text", "prompt": str(i)}} for i in range(10)]
        cfg = InferenceConfig(
            regions=["us-central1"],
            results_region="us-central1",
            max_workers_per_region=1,
            shard_size=3,
            job_name="shardsize-uneven",
            sampling=SamplingParams(max_tokens=4),
        )
        result = inference(model=ModelSpec(model="meta-llama/Llama-3-8B"), dataset=prompts, config=cfg)

    assert result.is_complete
    assert len(result.list_output_files()) == 4


class _StableUUID:
    """Helper that produces a uuid whose .hex[:12] matches the run_id encoded in
    a previously-built results_uri. Lets us re-run inference() against the same
    output directory."""

    def __init__(self, prior_results_uri: str) -> None:
        # results_uri = <prefix>/<job_name>/<run_id>/outputs
        parts = prior_results_uri.split("/")
        self._hex = parts[-2] + "0" * 32  # pad to satisfy slice [:12]

    @property
    def hex(self) -> str:
        return self._hex


# ---------------------------------------------------------------------------
# Helpers used by the smoke test to read raw output files (when needed).
# ---------------------------------------------------------------------------


def _read_output_records(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_response_record_roundtrip():
    rec = ResponseRecord(id="x", shard=3, response="hi", extra={"finish_reason": "stop"})
    revived = ResponseRecord.from_dict(rec.to_dict())
    assert revived == rec
