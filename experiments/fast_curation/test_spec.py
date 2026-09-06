# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Guards on the spec hashing contract.

A spec's version hash is baked into its GCS namespace, so a hash that moves silently re-points a
live pipeline at an empty directory and orphans every WARC already extracted under the old one.
These tests pin the hashes of published specs and assert the property that keeps them stable.
"""

from __future__ import annotations

import dataclasses
import gzip
import json

import numpy as np
import pytest

from experiments.fast_curation import batch_format, tpu_phase
from experiments.fast_curation.cpu_phase import _phase_is_complete
from experiments.fast_curation.spec import SPECS, Extractor, get_spec

# Hashes of published specs. A change here means a live corpus moves — bump the spec id instead.
# fastpipe_v3's namespace holds ~3,602 extracted WARCs (us-east5 + us-central1) as of 2026-08-11.
PINNED_VERSIONS = {
    "fastpipe_v3": "da3893385e",
    "lpv11_fastpipe_v1": "2224e3e476",
    "lpv11_fastpipe_v2": "32b74664f1",
    "lpv11_fastpipe_v2_1": "944ca6bc38",
}


@pytest.mark.parametrize(("spec_id", "expected"), sorted(PINNED_VERSIONS.items()))
def test_published_version_hash_is_pinned(spec_id: str, expected: str) -> None:
    assert get_spec(spec_id).version() == expected, (
        f"{spec_id} hash moved — its GCS namespace would change and orphan the existing corpus. "
        "If the cascade really changed, add a NEW spec id rather than mutating this one."
    )


def test_adding_an_unset_optional_field_does_not_change_existing_hashes() -> None:
    """The property that makes the pinning above survivable: unset fields are omitted from the hash."""
    before = {k: s.version() for k, s in SPECS.items()}

    @dataclasses.dataclass(frozen=True)
    class Extended(type(SPECS["fastpipe_v3"])):
        some_future_knob: str | None = None

    extended = Extended(**dataclasses.asdict(SPECS["fastpipe_v3"]))
    assert extended.version() == before["fastpipe_v3"], "an unset new field must not perturb the hash"


def test_pooled_threshold_is_namespace_defining_but_modernbert_is_not() -> None:
    """Pooled drops are destructive (ModernBERT never scores the doc), so its threshold must be
    in the hash; ModernBERT's is re-thresholdable from stored probs and must not be."""
    spec = get_spec("lpv11_fastpipe_v1")
    fields = spec._namespace_fields()
    assert "pooled_threshold" in fields
    assert "modernbert_threshold" not in fields

    retuned = dataclasses.replace(spec, pooled_threshold=spec.pooled_threshold + 0.01)
    assert retuned.version() != spec.version(), "re-tuning pooled must force a new namespace"
    retuned_mb = dataclasses.replace(spec, modernbert_threshold=spec.modernbert_threshold + 0.01)
    assert retuned_mb.version() == spec.version(), "re-tuning ModernBERT must NOT force a new namespace"


def test_lpv11_line_uses_only_lpv11_models() -> None:
    """Mixing an hq-trained stage into the lpv11 cascade would have stages optimizing for targets
    that agree at only 0.325 F1."""
    spec = get_spec("lpv11_fastpipe_v1")
    for path in (spec.fasttext_model, spec.pooled_ckpt, spec.modernbert_ckpt):
        assert "lpv11" in path, f"{path} is not an lpv11-targeted model"
    assert spec.extraction_engine is Extractor.RESILIPARSE_RS
    assert spec.resiliparse_rs_commit, "the Rust extractor build must be pinned by commit"


def test_legacy_specs_default_to_justext() -> None:
    for spec_id in ("fastpipe_v1", "fastpipe_v2", "fastpipe_v3"):
        assert get_spec(spec_id).extraction_engine is Extractor.JUSTEXT


def test_text_line_band_fields_are_namespace_defining() -> None:
    """A hi-accepted doc is kept WITHOUT a terminal prob, so re-tuning ``pooled_hi`` cannot be a
    stored-prob re-filter — it must force a new namespace. Same for the terminal eval context,
    which changes every stored terminal prob."""
    spec = get_spec("lpv11_fastpipe_v2")
    fields = spec._namespace_fields()
    assert "pooled_hi" in fields and "modernbert_max_length" in fields
    assert "modernbert_threshold" not in fields

    for change in ({"pooled_hi": 0.9}, {"modernbert_max_length": 1024}):
        assert dataclasses.replace(spec, **change).version() != spec.version()
    retuned_mb = dataclasses.replace(spec, modernbert_threshold=0.5)
    assert retuned_mb.version() == spec.version(), "the terminal threshold stays late-bound for band docs"


def test_text_line_uses_only_lpv11_text_models() -> None:
    """Every stage of the TEXT line must be lpv11-targeted AND trained on the TEXT representation —
    mixing an HTML-trained stage in would score a representation it never saw."""
    spec = get_spec("lpv11_fastpipe_v2")
    assert spec.is_text_line
    assert spec.modernbert_eval_length == 2048
    for path in (spec.fasttext_model, spec.pooled_ckpt, spec.modernbert_ckpt):
        assert "lpv11" in path.lower(), f"{path} is not an lpv11-targeted model"
        assert "text" in path.lower(), f"{path} is not a TEXT-representation model"
    assert spec.extraction_engine is Extractor.RESILIPARSE_RS
    assert spec.resiliparse_rs_commit, "the Rust extractor build must be pinned by commit"


def test_phase_end_sentinel_is_manifest_scoped(tmp_path):
    """A smaller run's phase-end sentinel must NOT stop a larger run sharing the namespace.

    Regression for a trap that burned three launches: a 300-WARC run finished, wrote
    ``_phase_a_end.json`` for the whole namespace, and every worker of the 10,364-WARC run then
    exited ~5 min after start having claimed nothing — reporting SUCCEEDED, so it looked like slow
    workers rather than workers refusing to work.
    """
    sentinel = tmp_path / "_phase_a_end.json"

    sentinel.write_text(
        json.dumps({"epoch": 1.0, "manifest": "experiments/distill/random_subsets/random_warcs_300.txt"})
    )
    assert not _phase_is_complete(
        str(sentinel), "experiments/distill/dclm_400m_1x.txt"
    ), "a 300-WARC sentinel must not mark the 10k run complete"
    assert _phase_is_complete(
        str(sentinel), "experiments/distill/random_subsets/random_warcs_300.txt"
    ), "its own run must still self-exit"

    # An unlabeled sentinel is unattributable and must be IGNORED, not honoured. tpu_phase's writer
    # shipped without the manifest field, so a straggler of the 300-WARC run wrote one of these into
    # the shared namespace mid-run and every B worker launched afterwards exited in ~42s as
    # SUCCEEDED. Honouring the ambiguous case is what re-opened the trap from the writer side.
    sentinel.write_text(json.dumps({"epoch": 1.0}))
    assert not _phase_is_complete(str(sentinel), "anything.txt")

    assert not _phase_is_complete(str(tmp_path / "nope.json"), "anything.txt")


def test_claim_refresh_is_throttled_and_progress_keyed():
    """Phase B's claim refresh must be driven by completed batches, not wall-clock alone.

    Before this, Phase B never refreshed: a claim's timestamp meant "work started", so a preempted
    holder and a slow WARC were indistinguishable and the stale window had to be hours. That parked
    the 10k run at 99.9% with 10 WARCs orphaned behind a 3h expiry.
    """
    calls = []
    clock = [1000.0]
    refresh = tpu_phase._throttled(lambda: calls.append(clock[0]), 60.0)

    refresh()  # first call always fires, so a claim is freshened as soon as work starts
    assert len(calls) == 1

    refresh()  # immediate repeat is throttled away (a big WARC is thousands of batches)
    assert len(calls) == 1


def test_process_warc_text_b_band_routing(tmp_path, monkeypatch):
    """The early-exit band must route each doc to exactly one fate, and the on-disk artifacts must
    make every fate auditable: hi-accepts land in kept/ with a NaN terminal prob, band-kept docs
    carry both probs, band-rejects and lo-drops land in tombstones (lo-drops with a null terminal
    prob), and the terminal model must be handed ids truncated to its eval context."""
    spec = get_spec("lpv11_fastpipe_v2")
    bucket = str(tmp_path)
    warc_hash = "cafe0001"
    # Four docs, one per fate. hi=0.8883, lo=0.079, terminal threshold=0.4378.
    rows = [
        {
            "doc_id": d,
            "url": "u",
            "warc_hash": warc_hash,
            "snapshot": "s",
            "fasttext_score": 0.5,
            "text": f"text {d}",
            "input_ids": list(range(n_tok)),
            "n_tokens": n_tok,
        }
        for d, n_tok in [("hi", 3), ("band_keep", 4000), ("band_drop", 5), ("lo", 6)]
    ]
    pre_path = f"{spec.presurvivors_prefix(bucket)}/data-{warc_hash}.parquet"
    batch_format.write_presurvivors_text(pre_path, rows)

    canned = iter(
        [
            np.array([0.95, 0.50, 0.40, 0.01], dtype=np.float32),  # pooled pass, all docs
            np.array([0.90, 0.10], dtype=np.float32),  # terminal pass, band docs only
        ]
    )
    seen_id_lists = []

    def fake_score_survivors(score_fn, model, config, id_lists, **kwargs):
        seen_id_lists.append(id_lists)
        return next(canned)

    monkeypatch.setattr(tpu_phase, "score_survivors", fake_score_survivors)
    registered = []
    monkeypatch.setattr(tpu_phase, "_register_completed_warc", lambda h, p: registered.append(h))

    payload = tpu_phase.process_warc_text_b(
        spec,
        warc_hash,
        score_fn=None,
        model=None,
        config=None,
        bucket=bucket,
        batch_size=4,
        bucket_tokens=True,
        registry_prefix="unused",
        pooled=(None, None),
    )

    assert payload["n_kept"] == 2 and registered == [warc_hash]
    # The terminal model saw ONLY the band docs, with ids truncated to its eval context.
    assert [len(ids) for ids in seen_id_lists[1]] == [2048, 5]

    kept = batch_format.read_table(f"{spec.kept_prefix(bucket)}/data-{warc_hash}.parquet")
    assert kept.schema.equals(batch_format.KEPT_SCHEMA_TEXT)
    by_id = {r["doc_id"]: r for r in kept.to_pylist()}
    assert set(by_id) == {"hi", "band_keep"}
    assert np.isnan(by_id["hi"]["modernbert_prob"]), "a hi-accept was never scored by the terminal model"
    assert by_id["band_keep"]["modernbert_prob"] == pytest.approx(0.90)
    assert by_id["hi"]["pooled_prob"] == pytest.approx(0.95)

    with open(f"{spec.tombstones_prefix(bucket)}/data-{warc_hash}.jsonl.gz", "rb") as fh:
        tombs = {r["doc_id"]: r for r in map(json.loads, gzip.decompress(fh.read()).decode().splitlines())}
    assert set(tombs) == {"band_drop", "lo"}
    assert tombs["band_drop"]["modernbert_prob"] == pytest.approx(0.10)
    assert tombs["lo"]["modernbert_prob"] is None, "a lo-drop has no terminal prob to record"


def test_score_survivors_fires_on_batch_per_batch(monkeypatch):
    """The liveness signal must come from real forward progress through the batch loop."""
    seen = []
    n_batches = []

    def fake_score(model, tokens, mask):
        class _A:
            array = np.zeros((4,), dtype=np.float32)

        n_batches.append(1)
        return _A()

    class _Cfg:
        max_seq_len = 128

    ids = [[1, 2, 3]] * 10  # 10 docs at batch_size 4 -> 3 batches
    tpu_phase.score_survivors(
        fake_score,
        object(),
        _Cfg(),
        ids,
        batch_size=4,
        pad_token_id=0,
        bucket_tokens=False,
        on_batch=lambda: seen.append(1),
    )
    assert len(seen) == len(n_batches) == 3, "one liveness tick per completed batch"
