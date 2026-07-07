# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned pipeline specs for the fast curation cascade.

A :class:`PipelineSpec` is the *complete, hashable definition* of one version of the
fast-extraction cascade (fastText useful-filter -> JustText extraction -> ModernBERT
useful-filter). It is the single source of truth for what ``fastpipe_vN`` means.

Versioning has three layers (see ``VERSIONS.md`` for the human ledger):

1. ``SPECS`` below — the authoritative machine-readable definition.
2. ``VERSIONS.md`` — the human ledger with the rationale for each bump.
3. ``compute_version()[:10]`` baked into the GCS ``namespace()`` — guarantees two
   different configs never collide on disk and lets any output dir be traced back to
   its exact spec.

Bump ``vN -> vN+1`` for ANY namespace-defining change (model, a threshold that *drops*
docs, the JustText fork version, the tokenizer, the step order) and record it in BOTH
``SPECS`` and ``VERSIONS.md`` in the same commit.

The ``modernbert_threshold`` is deliberately NOT namespace-defining: the TPU phase stores
``P(useful)`` for every survivor (kept + tombstone), so re-thresholding ModernBERT is a
cheap downstream re-filter over stored probs — no recompute, no new namespace.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

# Canonical checkpoint + model locations (us-east5, where the cascade was trained).
FASTTEXT_MODEL_V1 = "gs://marin-us-east5/classifiers/useful_fasttext/body_strip_scale_w320_strat_prep_mc500/model.bin"
MODERNBERT_CKPT_V1 = "gs://marin-us-east5/checkpoints/modernbert-useful/mb-clf-base-10M-c8192/hf"

# ModernBERT-base tokenizer / pad id (answerdotai/ModernBERT-base).
MODERNBERT_TOKENIZER = "answerdotai/ModernBERT-base"
MODERNBERT_PAD_TOKEN_ID = 50283

DEFAULT_STEP_ORDER = ("decode", "body_strip", "fasttext", "justext", "tokenize", "modernbert")


def _rebucket(gs_path: str, bucket: str) -> str:
    """Rewrite a ``gs://marin-<region>/<rest>`` path to ``{bucket}/<rest>`` (same relative path)."""
    rest = gs_path.split("/", 3)[3]  # drop the gs://marin-<region>/ prefix
    return f"{bucket.rstrip('/')}/{rest}"


@dataclass(frozen=True)
class PipelineSpec:
    """One version of the fast-extraction cascade.

    Every field except ``modernbert_threshold`` is namespace-defining: it feeds
    :meth:`compute_version`, so changing it produces a fresh GCS namespace.
    """

    spec_id: str  # human contract, e.g. "fastpipe_v1"; carried in the output path prefix.

    # --- stage 1: fastText useful-filter (CPU; DESTRUCTIVE drop -> namespace-defining) ---
    fasttext_model: str
    fasttext_threshold: float
    body_strip_repr: str = "body_strip"

    # --- stage 1b: JustText extraction (CPU; produces the training ``text``) ---
    justext_version: str = "xenon-v4.2.0"  # XenonMolecule/jusText git tag; bumping changes output text.
    justext_lang: str = "English"
    # String joining kept paragraphs in the extracted ``text``. "\n\n" (blank line) matches the gold
    # benchmark; "\n" mashes forum posts/paragraphs together. Namespace-defining (changes output text).
    justext_paragraph_sep: str = "\n\n"
    # A page whose RAW html exceeds this is skipped by JustText (-> "" -> dropped). It is a
    # namespace-defining quality knob: too low and genuinely long articles/books are lost (bad for
    # long-context training); its only purpose is to bound jusText's lxml DOM-parse cost.
    justext_max_html_chars: int = 3_000_000
    # Per-document JustText wall-clock guard (seconds); None = no timeout. lxml's C parser cannot be
    # interrupted by a signal, so a doc exceeding this is hard-killed (its worker terminated) and
    # yields "" (dropped). Namespace-defining (it can change which docs survive), though the actual
    # drops are machine-speed-dependent, so keep it generous so it fires only on pathological pages.
    justext_timeout: float | None = None

    # --- tokenization for ModernBERT (CPU; namespace-defining) ---
    tokenizer_ref: str = MODERNBERT_TOKENIZER
    pad_token_id: int = MODERNBERT_PAD_TOKEN_ID
    max_length: int = 8192
    single_window: bool = True  # True: truncate to ``max_length`` (matches the deployed c8192 ckpt).

    # --- stage 2: ModernBERT useful-filter (TPU) ---
    # The checkpoint is namespace-defining; the threshold is a cheap late-bound view.
    modernbert_ckpt: str = MODERNBERT_CKPT_V1
    modernbert_threshold: float = 0.1974  # METADATA only (NOT in compute_version).

    step_order: tuple[str, ...] = DEFAULT_STEP_ORDER

    def _namespace_fields(self) -> dict:
        """The dict hashed into the version. Excludes ``spec_id`` (carried in the path) and
        ``modernbert_threshold`` (a late-bound re-filter, not a recompute trigger)."""
        d = dataclasses.asdict(self)
        d.pop("spec_id", None)
        d.pop("modernbert_threshold", None)
        return d

    def compute_version(self) -> str:
        """Stable sha256 over the namespace-defining fields."""
        payload = json.dumps(self._namespace_fields(), sort_keys=True, default=list)
        return hashlib.sha256(payload.encode()).hexdigest()

    def version(self) -> str:
        """The 10-char version tag used in paths."""
        return self.compute_version()[:10]

    def subdir(self) -> str:
        """Bucket-relative output prefix: ``documents/fast_curation/{id}-{ver}``.

        Used to build the central completed-WARC registry in ``marin-us-central1``
        (``gs://marin-us-central1/{subdir}/_completed``), matching the dashboard convention.
        """
        return f"documents/fast_curation/{self.spec_id}-{self.version()}"

    def namespace(self, bucket: str = "gs://marin-us-east5") -> str:
        """Root GCS prefix for this spec's outputs: ``{bucket}/documents/fast_curation/{id}-{ver}``."""
        return f"{bucket}/{self.subdir()}"

    # --- per-phase sub-paths under the namespace (the on-disk contract) ---
    def survivors_prefix(self, bucket: str = "gs://marin-us-east5") -> str:
        """Phase-1 output: one fastText-survivor parquet per WARC."""
        return f"{self.namespace(bucket)}/cpu_survivors"

    def kept_prefix(self, bucket: str = "gs://marin-us-east5") -> str:
        """Phase-2 output: survivors with ModernBERT prob >= threshold."""
        return f"{self.namespace(bucket)}/kept"

    def tombstones_prefix(self, bucket: str = "gs://marin-us-east5") -> str:
        """Phase-2 output: dropped survivors WITH their stored prob (free re-thresholding)."""
        return f"{self.namespace(bucket)}/tombstones"

    # --- region-aware model paths ---
    # ``fasttext_model`` / ``modernbert_ckpt`` are stored as the canonical us-east5 absolute paths
    # (they feed compute_version, so they must NOT change). These resolvers rebucket them to the
    # WORKER's region so a job in region R reads R's mirror locally (no cross-region model reads).
    # The model must be mirrored to ``bucket`` first (one-time gcloud cp; same relative path).
    def fasttext_model_for(self, bucket: str) -> str:
        return _rebucket(self.fasttext_model, bucket)

    def modernbert_ckpt_for(self, bucket: str) -> str:
        return _rebucket(self.modernbert_ckpt, bucket)

    # --- v2 (ModernBERT BEFORE JustText) 3-phase sub-paths ---
    def presurvivors_prefix(self, bucket: str = "gs://marin-us-east5") -> str:
        """v2 Phase A output: fastText-survivors carrying RAW html + tokens, NO JustText yet."""
        return f"{self.namespace(bucket)}/a_presurvivors"

    def keeplist_prefix(self, bucket: str = "gs://marin-us-east5") -> str:
        """v2 Phase B output: per-WARC {doc_id, modernbert_prob} for all pre-survivors."""
        return f"{self.namespace(bucket)}/b_keeplist"


V2_STEP_ORDER = ("decode", "body_strip", "fasttext", "tokenize", "modernbert", "justext")


# Registry of all pipeline versions. ADD a new entry (never mutate an old one) per bump.
SPECS: dict[str, PipelineSpec] = {
    "fastpipe_v1": PipelineSpec(
        spec_id="fastpipe_v1",
        fasttext_model=FASTTEXT_MODEL_V1,
        fasttext_threshold=0.0368,
        justext_version="xenon-v4.2.0",
        modernbert_ckpt=MODERNBERT_CKPT_V1,
        modernbert_threshold=0.1974,
    ),
    # v2: identical output to v1 but runs ModernBERT BEFORE JustText, so JustText (the dominant
    # CPU cost) runs only on ModernBERT-survivors (~5x fewer docs). Different step_order -> different
    # version hash -> isolated namespace.
    "fastpipe_v2": PipelineSpec(
        spec_id="fastpipe_v2",
        fasttext_model=FASTTEXT_MODEL_V1,
        fasttext_threshold=0.0368,
        justext_version="xenon-v4.2.0",
        modernbert_ckpt=MODERNBERT_CKPT_V1,
        modernbert_threshold=0.1974,
        step_order=V2_STEP_ORDER,
    ),
    # v3: identical cascade to v2 but raises the JustText html cap 3M -> 50MB (v2's 3M cap dropped
    # genuinely long articles/books -> lost long-context training data) and adds a 60s per-doc
    # timeout so a pathological page is killed rather than allowed to hang a worker. New cap/timeout
    # -> new version hash -> isolated namespace (does NOT reuse the v2 corpus).
    "fastpipe_v3": PipelineSpec(
        spec_id="fastpipe_v3",
        fasttext_model=FASTTEXT_MODEL_V1,
        fasttext_threshold=0.0368,
        justext_version="xenon-v4.2.0",
        justext_max_html_chars=50_000_000,
        justext_timeout=60.0,
        modernbert_ckpt=MODERNBERT_CKPT_V1,
        modernbert_threshold=0.1974,
        step_order=V2_STEP_ORDER,
    ),
}


def get_spec(spec_id: str) -> PipelineSpec:
    """Resolve a spec by id, failing fast (no silent default) on an unknown id."""
    try:
        return SPECS[spec_id]
    except KeyError:
        raise ValueError(f"Unknown pipeline spec {spec_id!r}; known specs: {sorted(SPECS)}") from None
