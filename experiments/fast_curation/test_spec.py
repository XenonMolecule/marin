# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Guards on the spec hashing contract.

A spec's version hash is baked into its GCS namespace, so a hash that moves silently re-points a
live pipeline at an empty directory and orphans every WARC already extracted under the old one.
These tests pin the hashes of published specs and assert the property that keeps them stable.
"""

from __future__ import annotations

import dataclasses

import pytest

from experiments.fast_curation.spec import SPECS, Extractor, get_spec

# Hashes of published specs. A change here means a live corpus moves — bump the spec id instead.
# fastpipe_v3's namespace holds ~3,602 extracted WARCs (us-east5 + us-central1) as of 2026-08-11.
PINNED_VERSIONS = {"fastpipe_v3": "da3893385e", "lpv11_fastpipe_v1": "2224e3e476"}


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


def test_phase_end_sentinel_is_manifest_scoped(tmp_path):
    """A smaller run's phase-end sentinel must NOT stop a larger run sharing the namespace.

    Regression for a trap that burned three launches: a 300-WARC run finished, wrote
    ``_phase_a_end.json`` for the whole namespace, and every worker of the 10,364-WARC run then
    exited ~5 min after start having claimed nothing — reporting SUCCEEDED, so it looked like slow
    workers rather than workers refusing to work.
    """
    import json

    from experiments.fast_curation.cpu_phase import _phase_is_complete

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
    from experiments.fast_curation.tpu_phase import _throttled

    calls = []
    clock = [1000.0]
    refresh = _throttled(lambda: calls.append(clock[0]), 60.0)

    refresh()  # first call always fires, so a claim is freshened as soon as work starts
    assert len(calls) == 1

    refresh()  # immediate repeat is throttled away (a big WARC is thousands of batches)
    assert len(calls) == 1


def test_score_survivors_fires_on_batch_per_batch(monkeypatch):
    """The liveness signal must come from real forward progress through the batch loop."""
    import numpy as np

    from experiments.fast_curation import tpu_phase

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
