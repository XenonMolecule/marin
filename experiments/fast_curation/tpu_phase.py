# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 2 (TPU): score the CPU phase's survivors with JAX/Levanter ModernBERT.

A standalone TPU worker that:

1. Loads the ModernBERT useful-classifier ONCE via ``load_hf_sequence_classifier`` (the
   path that correctly round-trips the fine-tuned HF head — NOT ``converter.load_pretrained``,
   which silently gives a random classifier). Data-parallel mesh: model replicated (~150M
   fits one chip), batch sharded; splash attention at 8192.
2. Iterates the WARC manifest in a shuffled order; for each WARC whose CPU survivor parquet
   exists and is not yet in the central completed registry, atomically claims it, reads the
   PRE-TOKENIZED survivors, scores ``P(useful)``, and writes back:
     * ``kept/data-{h}.parquet``      — survivors with ``modernbert_prob >= threshold``,
     * ``tombstones/data-{h}.jsonl.gz`` — ``{doc_id, modernbert_prob}`` for ALL dropped
       survivors (so re-thresholding is a cheap offline re-filter, no TPU re-run),
     * ``timing_tpu/data-{h}.json``,
   then registers the WARC as completed.
3. Polls for new survivors while the CPU phase is still running; exits when everything is done.

Run as a standalone TPU Iris job (one per slice; launch many via ``launch_tpu.py``)::

    uv run iris --cluster marin job run --region us-east5 \\
      --tpu v6e-4 --enable-extra-resources --extra tpu --memory 64GB \\
      --priority batch --preemptible --no-wait --job-name fastcur-tpu-0 \\
      -e HF_TOKEN hf_... -- \\
      python -m experiments.fast_curation.tpu_phase --spec fastpipe_v1 --shuffle-seed 0
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import random
import time
from collections.abc import Callable

import fsspec
import numpy as np

from experiments.baseline_collection.decode_warcs_clean import _load_manifest, _warc_path_hash
from experiments.baseline_collection.run_extract_standalone import (
    _claim_warc_atomic,
    _list_fresh_claims,
    _load_completed_registry,
    _refresh_claim,
    _register_completed_warc,
)
from experiments.fast_curation import batch_format
from experiments.fast_curation.cpu_phase import _phase_is_complete
from experiments.fast_curation.spec import PipelineSpec, get_spec
from experiments.fast_curation.telemetry import Heartbeat, region_from_bucket

logger = logging.getLogger(__name__)

PAD_TOKEN_ID = 50283
# Training rounds Axis("vocab", len(tokenizer)) for partitioning; the pooled checkpoints were written
# with ModernBERT's tokenizer at this padded size.
POOLED_VOCAB_SIZE = 50368
# Give up after this many idle passes even if _phase_a_end was never written — a stuck upstream
# straggler must not leave a TPU idling forever (see cpu_phase.HARD_IDLE_PASSES). ~30 min at poll=30.
HARD_IDLE_PASSES = 60
# Refresh an in-flight WARC claim at most this often. Phase B previously never refreshed, so its
# claim timestamp meant "when work started", not "worker is alive" — a preempted holder and a
# legitimately slow WARC looked identical, and the only safe stale window was hours. Keyed to
# completed batches (see score_survivors) so a wedged worker stops refreshing and is reclaimed.
CLAIM_REFRESH_SECONDS = 60.0


def _throttled(fn: Callable[[], None], min_interval: float) -> Callable[[], None]:
    """Wrap ``fn`` so it runs at most once per ``min_interval`` seconds (first call runs)."""
    last = 0.0

    def call() -> None:
        nonlocal last
        now = time.monotonic()
        if last and now - last < min_interval:
            return
        last = now
        fn()

    return call


def _assert_ckpt_in_region(ckpt: str, bucket: str) -> None:
    """Fail fast if the checkpoint is NOT under the worker's own regional bucket.

    The checkpoint is read by every worker at startup; a cross-region read re-triggers egress on
    every worker and every preemption-retry. ``spec.modernbert_ckpt_for(bucket)`` rebuckets the
    canonical path into the worker's ``bucket`` by construction, so this asserts that invariant
    holds (and is naming-agnostic — e.g. region ``europe-west4`` correctly uses bucket
    ``marin-eu-west4``). Mirror the checkpoint into ``bucket`` with a one-time ``gcloud storage cp``
    BEFORE launch; the load fails clearly if it isn't there.
    """
    if not ckpt.startswith(bucket.rstrip("/") + "/"):
        raise RuntimeError(
            f"Checkpoint {ckpt} is not under this worker's regional bucket {bucket}. Mirror it "
            f"into {bucket} BEFORE launch (one-time `gcloud storage cp`) — do NOT pull cross-region."
        )


def _gcs_exists(path: str) -> bool:
    fs = fsspec.filesystem("gcs")
    return fs.exists(path.replace("gs://", ""))


def _list_input_hashes(input_prefix: str) -> set[str] | None:
    """Hashes with a ``data-{h}.parquet`` under ``input_prefix``, from one fresh listing.

    Replaces a per-WARC ``exists()`` probe per manifest entry per pass.
    ``refresh=True`` bypasses gcsfs's process-lifetime dircache so newly-produced
    inputs are visible. Returns None on listing failure (caller falls back to the
    per-WARC probes).
    """
    fs = fsspec.filesystem("gcs")
    try:
        names = fs.ls(input_prefix.replace("gs://", ""), refresh=True)
    except FileNotFoundError:
        return set()  # upstream dir not created yet — nothing present
    except Exception as e:
        logger.warning("input listing failed for %s: %s", input_prefix, e)
        return None
    out = set()
    for p in names:
        base = p.rsplit("/", 1)[-1]
        if base.startswith("data-") and base.endswith(".parquet"):
            out.add(base[len("data-") : -len(".parquet")])
    return out


def _setup_compile_cache(cache_dir: str) -> None:
    """Mount a persistent (GCS) XLA compile cache so the ~15-20 min splash@8192 compile is
    paid once for the whole fleet (cache key = model + static-shape program)."""
    import jax

    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    logger.info("XLA persistent compile cache -> %s", cache_dir)


def load_model(spec: PipelineSpec, mesh, bucket: str):
    """Load the ModernBERT seq-classifier onto ``mesh`` (bf16). Returns ``(model, config)``.

    Reads the checkpoint from the WORKER's region mirror (``bucket``), not the canonical us-east5
    path, so a job in region R never reads the checkpoint cross-region."""
    import jax
    import jax.numpy as jnp
    from haliax.partitioning import set_mesh
    from levanter.models.modernbert import ModernBertConfig, load_hf_sequence_classifier
    from levanter.utils.tree_utils import inference_mode

    hf_ref = spec.modernbert_ckpt_for(bucket)
    backend = "splash" if spec.max_length >= 8192 else "vanilla"
    # Derive architecture (hidden/layers/heads) from the checkpoint's own config.json so base
    # AND large checkpoints load correctly; override only eval context, attn backend, labels, pad.
    converter = ModernBertConfig().hf_checkpoint_converter(ref_checkpoint=hf_ref)
    hf_config = converter.hf_config_from_hf_checkpoint(hf_ref)
    config = dataclasses.replace(
        ModernBertConfig.from_hf_config(hf_config),
        max_seq_len=spec.max_length,
        attn_backend=backend,
        num_labels=2,
        pad_token_id=spec.pad_token_id,
    )
    logger.info(
        "loading ModernBERT %s (hidden=%d layers=%d heads=%d ctx=%d backend=%s)",
        hf_ref,
        config.hidden_dim,
        config.num_layers,
        config.num_heads,
        config.max_seq_len,
        backend,
    )
    with set_mesh(mesh):
        model = load_hf_sequence_classifier(config, hf_ref, axis_mapping=None, dtype=jnp.bfloat16)
    model = inference_mode(model, True)

    # Head-sanity guard: a failed/random-head load (the `converter.load_pretrained` trap) leaves
    # the classifier near-constant. A genuinely loaded head has real spread.
    leaves = [x for x in jax.tree_util.tree_leaves(model.classifier) if hasattr(x, "size") and x.size > 1]
    stds = [float(jnp.std(x.astype(jnp.float32))) for x in leaves]
    if stds and max(stds) < 1e-6:
        raise RuntimeError(
            "ModernBERT classifier head appears degenerate (all ~constant) — the HF load likely "
            "failed to restore the fine-tuned head. Check load_hf_sequence_classifier."
        )
    logger.info("classifier head std=%.4g (non-degenerate)", max(stds) if stds else float("nan"))
    return model, config


def load_pooled_model(spec: PipelineSpec, mesh, bucket: str):
    """Load the pooled-transformer pre-filter onto ``mesh``. Returns ``(model, config)``.

    Equinox format (``model.eqx`` + ``config.json``) rather than HF safetensors, so it loads via
    ``load_pooled_transformer_classifier`` — but its forward is ``m(tokens, mask)`` returning label
    logits exactly like ModernBERT's, so ``score_survivors`` scores it unchanged.
    """
    import dataclasses as _dc
    import json

    from haliax.partitioning import set_mesh
    from levanter.models.pooled_transformer import PooledTransformerConfig, load_pooled_transformer_classifier
    from levanter.utils.tree_utils import inference_mode

    ckpt = spec.pooled_ckpt_for(bucket)
    _assert_ckpt_in_region(ckpt, bucket)
    with fsspec.open(f"{ckpt}/config.json", "rt", encoding="utf-8") as f:
        raw = json.load(f)
    config_class = raw.pop("config_class", None)
    if config_class != "PooledTransformerConfig":
        raise RuntimeError(f"{ckpt}/config.json is a {config_class!r}, not a PooledTransformerConfig")
    config = _dc.replace(PooledTransformerConfig(**raw), max_seq_len=spec.max_length)
    if config.pad_token_id != spec.pad_token_id:
        raise RuntimeError(
            f"pooled ckpt pad_token_id={config.pad_token_id} != spec {spec.pad_token_id}; it must share "
            "ModernBERT's tokenization or Phase A's tokens are invalid for it"
        )
    logger.info(
        "loading pooled %s (hidden=%d layers=%d heads=%d ctx=%d)",
        ckpt,
        config.hidden_dim,
        config.num_layers,
        config.num_heads,
        config.max_seq_len,
    )
    with set_mesh(mesh):
        model = load_pooled_transformer_classifier(config, ckpt, vocab_size=POOLED_VOCAB_SIZE)
    return inference_mode(model, True), config


def make_score_fn():
    """Build the jitted P(useful) callable ONCE (reused across all WARCs so XLA caches by shape)."""
    import haliax as hax
    import jax.numpy as jnp

    def _probs_impl(m, tokens, mask):
        logits = m(tokens, mask).astype(jnp.float32)
        return hax.nn.softmax(logits, axis="label")["label", 1]

    return hax.named_jit(_probs_impl, axis_resources={"batch": "data"})


def score_survivors(
    score_fn,
    model,
    config,
    id_lists: list[list[int]],
    *,
    batch_size: int,
    pad_token_id: int,
    bucket_tokens: bool,
    on_batch: Callable[[], None] | None = None,
) -> np.ndarray:
    """Return ``P(useful)`` per survivor. Batch sharded over 'data'; static ``[batch_size, ctx]``.

    ``bucket_tokens``: pad each (length-sorted) batch up to the smallest length bucket that
    fits it (≪ 8192 for short docs) instead of always 8192. Each distinct ctx compiles once
    (cached). When False, always pad to ``config.max_seq_len`` (exact parity with
    ``score_modernbert_useful._score``).

    ``on_batch`` fires after each completed batch — a liveness signal keyed to real forward
    progress, so a wedged worker stops emitting it and its claim correctly goes stale.
    """
    import haliax as hax
    from haliax import Axis
    from levanter.layers.attention import AttentionMask

    n = len(id_lists)
    out = np.zeros((n,), dtype=np.float32)
    if n == 0:
        return out

    order = np.argsort([len(x) for x in id_lists], kind="stable")
    Batch = Axis("batch", batch_size)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        chunk = [id_lists[i] for i in idx]
        bs = len(chunk)
        max_len = max((len(c) for c in chunk), default=1)
        target = batch_format.bucket_for(max_len, config.max_seq_len) if bucket_tokens else config.max_seq_len
        # Always allocate ``batch_size`` rows (pad the tail with empty rows) for a static shape.
        padded_rows = list(chunk) + [[]] * (batch_size - bs)
        ids, seg = batch_format.pad_batch(padded_rows, target, pad_token_id)
        Pos = Axis("position", target)
        tokens = hax.named(ids, (Batch, Pos))
        seg_named = hax.named(seg, (Batch, Pos))
        mask = AttentionMask(is_causal=False).with_segment_ids(seg_named, seg_named)
        probs = np.asarray(score_fn(model, tokens, mask).array)[:bs]
        out[idx] = probs
        if on_batch is not None:
            on_batch()
    return out


def process_warc(
    spec: PipelineSpec,
    warc_hash: str,
    score_fn,
    model,
    config,
    *,
    bucket: str,
    batch_size: int,
    bucket_tokens: bool,
    registry_prefix: str,
    refresh: Callable[[], None] | None = None,
) -> dict:
    """Score one WARC's survivors and write kept/tombstone/timing; register on success."""
    import pyarrow as pa

    survivor_path = f"{spec.survivors_prefix(bucket)}/data-{warc_hash}.parquet"
    kept_path = f"{spec.kept_prefix(bucket)}/data-{warc_hash}.parquet"
    tomb_path = f"{spec.tombstones_prefix(bucket)}/data-{warc_hash}.jsonl.gz"
    timing_path = f"{spec.namespace(bucket)}/timing_tpu/data-{warc_hash}.json"

    t0 = time.monotonic()
    table = batch_format.read_table(survivor_path)
    n = table.num_rows

    id_lists = table.column("input_ids").to_pylist() if n else []
    t_read = time.monotonic() - t0

    t1 = time.monotonic()
    probs = score_survivors(
        score_fn,
        model,
        config,
        id_lists,
        batch_size=batch_size,
        pad_token_id=spec.pad_token_id,
        bucket_tokens=bucket_tokens,
        on_batch=refresh,
    )
    t_score = time.monotonic() - t1

    keep_mask = probs >= spec.modernbert_threshold
    n_kept = int(keep_mask.sum())

    kept_table = table.filter(pa.array(keep_mask)) if n else table
    kept_table = kept_table.append_column("modernbert_prob", pa.array(probs[keep_mask], type=pa.float32()))
    batch_format.write_kept(kept_path, kept_table)

    if n:
        doc_ids = table.column("doc_id").to_pylist()
        dropped = [(doc_ids[i], float(probs[i])) for i in range(n) if not keep_mask[i]]
        if dropped:
            batch_format.write_tombstones(tomb_path, dropped)

    payload = {
        "warc_hash": warc_hash,
        "n_survivors": n,
        "n_kept": n_kept,
        "n_dropped": n - n_kept,
        "read_s": round(t_read, 3),
        "score_s": round(t_score, 3),
        "wall_s": round(time.monotonic() - t0, 3),
    }
    try:
        with fsspec.open(timing_path, "w") as f:
            import json

            json.dump(payload, f)
    except Exception as e:
        logger.warning("failed to write tpu timing for %s: %s", warc_hash, e)

    _register_completed_warc(warc_hash, registry_prefix)
    logger.info(
        "%s: %d survivors -> %d kept (%.1f%%) in %.1fs (score %.1fs)",
        warc_hash,
        n,
        n_kept,
        100.0 * n_kept / n if n else 0.0,
        payload["wall_s"],
        t_score,
    )
    return payload


def process_warc_v2b(
    spec: PipelineSpec,
    warc_hash: str,
    score_fn,
    model,
    config,
    *,
    bucket: str,
    batch_size: int,
    bucket_tokens: bool,
    registry_prefix: str,
    pooled: tuple | None = None,
    refresh: Callable[[], None] | None = None,
) -> dict:
    """v2 Phase B: score one WARC's pre-survivors and write a {doc_id, modernbert_prob} keeplist.

    Reads ONLY the ``doc_id`` + ``input_ids`` columns (column projection — the big ``html`` column
    is left for Phase C), so the TPU never loads html. Writes a prob for EVERY pre-survivor; Phase C
    applies the threshold (so re-thresholding stays free)."""
    presurvivor_path = f"{spec.presurvivors_prefix(bucket)}/data-{warc_hash}.parquet"
    keeplist_path = f"{spec.keeplist_prefix(bucket)}/data-{warc_hash}.parquet"
    timing_path = f"{spec.namespace(bucket)}/timing_b/data-{warc_hash}.json"

    t0 = time.monotonic()
    table = batch_format.read_table(presurvivor_path, columns=["doc_id", "input_ids"])
    n = table.num_rows
    doc_ids = table.column("doc_id").to_pylist()
    id_lists = table.column("input_ids").to_pylist() if n else []
    t_read = time.monotonic() - t0

    # Optional pooled pre-filter: ~180x cheaper per doc than ModernBERT on the SAME tokens, so
    # scoring everything with it and passing only its survivors on is far cheaper than not. Docs it
    # drops are never seen by ModernBERT and keep a NaN modernbert_prob (Phase C's `>= threshold`
    # excludes NaN for free) — that drop is destructive, which is why pooled_threshold is in the hash.
    pooled_probs = None
    t_pooled = 0.0
    scored_idx = list(range(n))
    if pooled is not None and n:
        t_p = time.monotonic()
        pooled_model, pooled_config = pooled
        pooled_probs = score_survivors(
            score_fn,
            pooled_model,
            pooled_config,
            id_lists,
            batch_size=batch_size,
            pad_token_id=spec.pad_token_id,
            bucket_tokens=bucket_tokens,
            on_batch=refresh,
        )
        t_pooled = time.monotonic() - t_p
        scored_idx = [i for i in scored_idx if pooled_probs[i] >= spec.pooled_threshold]

    t1 = time.monotonic()
    scored = score_survivors(
        score_fn,
        model,
        config,
        [id_lists[i] for i in scored_idx],
        batch_size=batch_size,
        pad_token_id=spec.pad_token_id,
        bucket_tokens=bucket_tokens,
        on_batch=refresh,
    )
    t_score = time.monotonic() - t1

    probs = np.full((n,), np.nan, dtype=np.float32)
    for slot, i in enumerate(scored_idx):
        probs[i] = scored[slot]

    batch_format.write_keeplist(
        keeplist_path,
        doc_ids,
        [float(p) for p in probs],
        pooled_probs=None if pooled_probs is None else [float(p) for p in pooled_probs],
    )
    # NaN >= threshold is False, so pooled-dropped docs never count as kept.
    n_keep = int((probs >= spec.modernbert_threshold).sum()) if n else 0
    try:
        with fsspec.open(timing_path, "w") as f:
            import json

            json.dump(
                {
                    "warc_hash": warc_hash,
                    "n_presurvivors": n,
                    "n_pooled_pass": len(scored_idx) if pooled is not None else None,
                    "n_keep": n_keep,
                    "read_s": round(t_read, 3),
                    "pooled_s": round(t_pooled, 3),
                    "score_s": round(t_score, 3),
                    "wall_s": round(time.monotonic() - t0, 3),
                },
                f,
            )
    except Exception as e:
        logger.warning("failed to write phase-B timing for %s: %s", warc_hash, e)
    _register_completed_warc(warc_hash, registry_prefix)
    wall = time.monotonic() - t0
    pooled_note = "" if pooled is None else f" [pooled {n}->{len(scored_idx)} in {t_pooled:.1f}s]"
    logger.info(
        "B %s: %d presurvivors ->%s %d keep (%.1f%%) in %.1fs (score %.1fs)",
        warc_hash,
        n,
        pooled_note,
        n_keep,
        100.0 * n_keep / n if n else 0.0,
        wall,
        t_score,
    )
    return {
        "n_kept": n_keep,
        # docs_in = presurvivors scored; docs_out = those passing threshold (flow to Phase C).
        "stats": {
            "docs_in": n,
            "docs_out": n_keep,
            "wall_seconds": wall,
            "compute_seconds": {"modernbert": t_score, "read": t_read},
        },
    }


def run_worker(
    spec: PipelineSpec,
    manifest_path: str,
    bucket: str,
    *,
    batch_size: int,
    bucket_tokens: bool,
    shuffle_seed: int,
    poll_seconds: float,
    max_idle_passes: int,
    limit: int | None,
    mode: str = "v1",
    claim_stale_hours: float = 3.0,
) -> None:
    import jax
    from haliax.partitioning import ResourceAxis, set_mesh
    from jax.sharding import Mesh

    # Set the persistent compile cache BEFORE any JAX compilation happens.
    region = region_from_bucket(bucket)
    _assert_ckpt_in_region(spec.modernbert_ckpt_for(bucket), bucket)
    _setup_compile_cache(f"{spec.namespace(bucket)}/_xla_cache")

    n_dev = len(jax.devices())
    if batch_size % n_dev != 0:
        raise ValueError(f"--batch-size {batch_size} must be a multiple of device count {n_dev}")
    mesh = Mesh(np.array(jax.devices()).reshape(n_dev, 1), (ResourceAxis.DATA, ResourceAxis.MODEL))
    logger.info("devices=%s mesh=(%d,1) batch=%d bucket_tokens=%s", jax.devices(), n_dev, batch_size, bucket_tokens)

    with set_mesh(mesh):
        model, config = load_model(spec, mesh, bucket)
        # Both models share the forward signature and the mesh, so ONE jitted score fn serves both.
        pooled = load_pooled_model(spec, mesh, bucket) if spec.pooled_ckpt else None
        if pooled is not None and mode != "v2b":
            raise ValueError(f"{spec.spec_id} has a pooled stage, which only the v2b phase runs; pass --mode v2b")
        score_fn = make_score_fn()

        warc_paths = _load_manifest(manifest_path)
        if limit is not None:
            warc_paths = warc_paths[:limit]
        hashes = [_warc_path_hash(w) for w in warc_paths]
        random.Random(shuffle_seed).shuffle(hashes)

        # v1 scores cpu_survivors -> kept/tombstones; v2b scores a_presurvivors -> b_keeplist.
        # Central claims (us-central1) for cross-region work-stealing; B only claims WARCs whose
        # presurvivor exists in its OWN region bucket, so scoring stays local to A's region.
        central = f"gs://marin-us-central1/{spec.subdir()}"
        if mode == "v2b":
            input_prefix = spec.presurvivors_prefix(bucket)
            registry_prefix = f"{central}/_completed_b"
            claim_root = f"{central}/_claims_b"
            upstream_done_path = f"{central}/_phase_a_end.json"
            process_fn = process_warc_v2b
        else:
            input_prefix = spec.survivors_prefix(bucket)
            registry_prefix = f"{central}/_completed"
            claim_root = f"{central}/_claims"
            upstream_done_path = f"{spec.namespace(bucket)}/_phase1_end.json"
            process_fn = process_warc

        # Best-effort dashboard heartbeat (B is TPU). Monotonic across preemption; never fails a WARC.
        hb = Heartbeat(spec, phase="b", region=region, seed=shuffle_seed, kind="tpu")

        # Phase-end sentinel for the fast self-exit (v2b writes _phase_b_end when B is globally done).
        phase_end_path = f"{central}/_phase_b_end.json" if mode == "v2b" else None
        idle = 0
        total_kept = 0
        total_done = 0
        while True:
            # Fast, scalable self-exit: once the phase-end sentinel exists, B is globally complete, so
            # exit on a single cheap blob check instead of re-listing the O(WARCs) registry each pass.
            if phase_end_path and _phase_is_complete(phase_end_path, manifest_path):
                logger.info("_phase_b_end present — phase B complete; exiting.")
                hb.close("done")
                break
            completed = _load_completed_registry(registry_prefix)
            # One listing each for claims and upstream inputs per pass, instead of a
            # per-WARC probe pair against the central bucket per manifest entry.
            claimed_fresh = _list_fresh_claims(claim_root, claim_stale_hours)
            have_input = _list_input_hashes(input_prefix)
            progressed = 0
            for h in hashes:
                if h in completed:
                    continue
                if have_input is not None:
                    if h not in have_input:
                        continue  # upstream phase has not produced this WARC yet.
                elif not _gcs_exists(f"{input_prefix}/data-{h}.parquet"):
                    continue  # input listing unavailable — per-WARC fallback.
                if h in claimed_fresh:
                    continue  # freshly claimed by another worker; can't be won.
                claim_path = f"{claim_root}/data-{h}"
                if not _claim_warc_atomic(claim_path, stale_hours=claim_stale_hours):
                    continue  # another worker owns it.
                payload = process_fn(
                    spec,
                    h,
                    score_fn,
                    model,
                    config,
                    bucket=bucket,
                    batch_size=batch_size,
                    bucket_tokens=bucket_tokens,
                    registry_prefix=registry_prefix,
                    # Keeps a long WARC's claim alive so peers don't reclaim work in flight; stops
                    # the moment batches stop landing, so a dead/wedged worker still goes stale.
                    refresh=_throttled(lambda p=claim_path: _refresh_claim(p), CLAIM_REFRESH_SECONDS),
                    **({"pooled": pooled} if mode == "v2b" else {}),
                )
                total_kept += payload["n_kept"]
                if payload.get("stats"):
                    hb.record_warc(warc_hash=h, **payload["stats"])
                total_done += 1
                progressed += 1

            phase1_done = _phase_is_complete(upstream_done_path, manifest_path)
            completed_now = _load_completed_registry(registry_prefix)
            remaining = [h for h in hashes if h not in completed_now]
            if not remaining:
                if mode == "v2b":
                    # Signal Phase C that all keeplists are written (its upstream_done gate).
                    try:
                        with fsspec.open(f"{central}/_phase_b_end.json", "w") as f:
                            import json

                            # The manifest is what scopes this flag to THIS run. Without it the
                            # sentinel is namespace-global: a 300-WARC straggler wrote an unlabeled
                            # one and killed every newly-launched B worker of the 10k run.
                            json.dump({"epoch": time.time(), "manifest": manifest_path}, f)
                    except Exception as e:
                        logger.warning("phase B end sentinel write failed: %s", e)
                logger.info("all %d WARCs complete; exiting.", len(hashes))
                hb.close("done")
                break
            if progressed == 0:
                idle += 1
                hb.tick("draining" if phase1_done else "idle")
                logger.info(
                    "idle pass %d/%d (%d remaining, phase1_done=%s)",
                    idle,
                    max_idle_passes,
                    len(remaining),
                    phase1_done,
                )
                if phase1_done and idle >= max_idle_passes:
                    logger.info("phase1 done and no claimable work for %d passes; exiting.", idle)
                    hb.close("done")
                    break
                # Hard idle timeout: an upstream straggler (a WARC that can't be processed) never lets
                # _phase_a_end get written, which would otherwise leave this TPU idling forever. After
                # HARD_IDLE_PASSES of zero claimable work, exit anyway so the TPU is freed.
                if idle >= HARD_IDLE_PASSES:
                    logger.warning(
                        "HARD idle timeout (%d passes) with no claimable work — exiting despite "
                        "phase1_done=%s (likely a stuck straggler upstream).",
                        idle,
                        phase1_done,
                    )
                    hb.close("done")
                    break
                time.sleep(poll_seconds)
            else:
                idle = 0

        logger.info("worker done: processed %d WARCs, %d docs kept.", total_done, total_kept)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="fastpipe_v3")
    ap.add_argument("--manifest", default="experiments/distill/dclm_400m_1x.txt")
    ap.add_argument("--bucket", default="gs://marin-us-east5")
    ap.add_argument("--batch-size", type=int, default=32, help="Must be a multiple of the chip count.")
    ap.add_argument(
        "--no-bucket-tokens",
        dest="bucket_tokens",
        action="store_false",
        help="Disable length bucketing; always pad to max_length (exact score_modernbert_useful parity).",
    )
    ap.add_argument("--shuffle-seed", type=int, default=0, help="Per-worker manifest shuffle for claim diversity.")
    ap.add_argument(
        "--poll-seconds", type=float, default=30.0, help="Sleep between passes when waiting on the CPU phase."
    )
    ap.add_argument(
        "--max-idle-passes", type=int, default=5, help="Exit after this many idle passes once phase 1 is done."
    )
    ap.add_argument("--limit", type=int, default=None, help="Only consider the first N manifest WARCs (smoke).")
    ap.add_argument(
        "--mode",
        default="v1",
        choices=["v1", "v2b"],
        help="v1: score cpu_survivors -> kept/tombstones. v2b: score a_presurvivors -> b_keeplist.",
    )
    ap.add_argument(
        "--claim-stale-hours",
        type=float,
        default=3.0,
        help=(
            "Reclaim a WARC whose claim has not been refreshed for this long. A preempted worker "
            "leaves its claim frozen, so the endgame can park just short of 100%% waiting out this "
            "window; drop it (e.g. 0.25) for a finisher fleet. Live workers refresh continuously, "
            "so a short window steals nothing — it only risks processing a WARC twice, which is "
            "harmless (last writer wins on an identical shard)."
        ),
    )
    ap.set_defaults(bucket_tokens=True)
    args = ap.parse_args()

    spec = get_spec(args.spec)
    run_worker(
        spec,
        args.manifest,
        args.bucket,
        batch_size=args.batch_size,
        bucket_tokens=args.bucket_tokens,
        shuffle_seed=args.shuffle_seed,
        poll_seconds=args.poll_seconds,
        max_idle_passes=args.max_idle_passes,
        limit=args.limit,
        mode=args.mode,
        claim_stale_hours=args.claim_stale_hours,
    )


if __name__ == "__main__":
    main()
