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
from enum import StrEnum

# Canonical checkpoint + model locations (us-east5, where the cascade was trained).
FASTTEXT_MODEL_V1 = "gs://marin-us-east5/classifiers/useful_fasttext/body_strip_scale_w320_strat_prep_mc500/model.bin"
MODERNBERT_CKPT_V1 = "gs://marin-us-east5/checkpoints/modernbert-useful/mb-clf-base-10M-c8192/hf"

# --- lpv11-targeted line -----------------------------------------------------------------------
# The v1-v3 models above approximate the 8B ``high_quality`` extractor. These approximate
# ``llm_pipeline_v1_1`` instead, which is a materially different target: the two agree at only
# 0.325 F1, keeping 21.5% vs 4.8% of pages. Do NOT mix the two families in one cascade.
FASTTEXT_LPV11_W640 = (
    "gs://marin-us-east5/classifiers/useful_fasttext_lpv11/body_strip_scale_w640_sub0p22_strat_prep_mc500/model.bin"
)
POOLED_LPV11_10M = "gs://marin-us-east5/checkpoints/modernbert-useful/mb-clf-lpv11-pooled-10M/hf"
MODERNBERT_LPV11_10M = "gs://marin-us-east5/checkpoints/modernbert-useful/mb-clf-lpv11-base-10M-c8192/hf"

# --- lpv11 TEXT line (classifiers run on the EXTRACTED text, extraction happens first) ----------
# TEXT-trained models: input = lower(ws-collapse(resiliparse-rs main content of the RAW html)) — the
# one-extraction deployment representation the planner's ``*_textraw_*`` score columns priced
# (comparison_sample.ClassifierInput.TEXT_FROM_RAW). fastText w640 TEXT was trained in us-central2 and
# mirrored here; the neural checkpoints live in us-east5 natively.
FASTTEXT_LPV11_TEXT_W640 = (
    "gs://marin-us-east5/classifiers/useful_fasttext_lpv11/"
    "resiliparse_scale_w640_sub0p22_strat_prep_mc500_TEXT/model.bin"
)
POOLED_LPV11_TEXT_90M = "gs://marin-us-east5/checkpoints/modernbert-useful/mb-clf-lpv11-text-pooled-90M/hf"
ETTIN68_LPV11_TEXT_10M = "gs://marin-us-east5/checkpoints/modernbert-useful/mb-clf-lpv11-text-ettin68-10M/hf"

# Prebuilt Linux artifact for the XenonMolecule fork's Rust extractor (resiliparse._extract_rs).
# Pinned by the COMMIT it was built from: the extracted text is the training text, so a rebuild from
# a different commit changes the corpus and must bump the version.
RESILIPARSE_RS_ARTIFACT = "gs://marin-us-east5/artifacts/resiliparse_rs/latest"
RESILIPARSE_RS_COMMIT = "850891b277e919f671ec19f12aafdce451b327f4"


class Extractor(StrEnum):
    """Which engine produces the training ``text`` in Phase C."""

    JUSTEXT = "justext"  # XenonMolecule/jusText fork; 9.43 docs/s/core
    RESILIPARSE_RS = "resiliparse_rs"  # XenonMolecule/chatnoir-resiliparse Rust engine; 291.8 docs/s/core


# ModernBERT-base tokenizer / pad id (answerdotai/ModernBERT-base).
MODERNBERT_TOKENIZER = "answerdotai/ModernBERT-base"
MODERNBERT_PAD_TOKEN_ID = 50283
# [SEP] — used to derive a shorter-context tokenization from the stored max_length ids without
# re-tokenizing (batch_format.truncate_ids); parity asserted against the real tokenizer in tests.
MODERNBERT_SEP_TOKEN_ID = 50282
# [CLS] — prepended by the gigatoken arrow tokenize path (vectorized special-token assembly);
# parity asserted against the real tokenizer in tests.
MODERNBERT_CLS_TOKEN_ID = 50281

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

    # --- stage 1c: pooled-transformer useful-filter (TPU; OPTIONAL, runs BEFORE ModernBERT) ---
    # ~180x cheaper than ModernBERT per doc (3932.6 vs 21.5 docs/chip/s) on the SAME tokens, so it
    # culls cheaply ahead of the expensive stage. Deploy at HIGH recall (>=0.95): it is a mid-stage,
    # not a ModernBERT replacement.
    #
    # NOTE the asymmetry with ``modernbert_threshold``: ``pooled_threshold`` IS namespace-defining.
    # A doc pooled drops is never scored by ModernBERT, so there is no stored prob to re-threshold
    # against later — the drop is destructive, exactly like fastText's.
    pooled_ckpt: str | None = None  # None => no pooled stage (the v1-v3 line).
    pooled_threshold: float | None = None
    # Early-exit band (the TEXT line): a doc with pooled prob >= ``pooled_hi`` is ACCEPTED outright
    # and the terminal model never scores it; below ``pooled_threshold`` it is dropped; only the
    # uncertain band in between reaches the terminal model. ``pooled_hi`` IS namespace-defining: a
    # hi-accepted doc has no stored terminal prob, so re-tuning hi cannot be a stored-prob re-filter.
    pooled_hi: float | None = None

    # --- stage 2: ModernBERT useful-filter (TPU) ---
    # The checkpoint is namespace-defining; the threshold is a cheap late-bound view.
    modernbert_ckpt: str = MODERNBERT_CKPT_V1
    modernbert_threshold: float = 0.1974  # METADATA only (NOT in compute_version).
    # Eval context of the terminal model when it differs from ``max_length`` (the TEXT line runs
    # ettin68 at 2048 on ids derived from the stored ``max_length`` tokenization — same checkpoint,
    # 3.9x the throughput). Namespace-defining: it changes every stored terminal prob.
    modernbert_max_length: int | None = None

    # --- storage contract version (None = the v1/v2 flat layout) ---
    # 3 = the sharded 8M-scale contract: work-list is shard parquets (not a flat manifest), claims
    # and registries are per-shard, presurvivors/kept use the V3 schemas (NO input_ids — Phase B
    # re-tokenizes from ``text`` via gigatoken), and text columns are written at zstd level 12.
    # Namespace-defining: the on-disk layout is incompatible with v2 readers.
    storage_version: int | None = None

    # --- stage 3: extraction (CPU, Phase C) — produces the training ``text`` ---
    # None means JUSTEXT. It defaults to None rather than to the enum so that specs predating this
    # field hash exactly as they did before it existed — see ``_namespace_fields``. Read it through
    # ``extraction_engine``, never directly.
    extractor: Extractor | None = None
    # Only meaningful for RESILIPARSE_RS; pins the fork commit the prebuilt .so was built from.
    resiliparse_rs_commit: str | None = None

    step_order: tuple[str, ...] = DEFAULT_STEP_ORDER

    @property
    def extraction_engine(self) -> Extractor:
        """The Phase-C engine; ``extractor=None`` is the legacy jusText default."""
        return self.extractor or Extractor.JUSTEXT

    @property
    def is_text_line(self) -> bool:
        """True when the classifiers consume the EXTRACTED text (extraction runs in Phase A).

        The text line is a 2-phase pipeline: Phase A extracts every decoded doc and gates on
        fastText-TEXT; Phase B (pooled band + terminal model) writes the final ``kept/`` directly —
        there is no Phase C.
        """
        return "clf_text" in self.step_order

    @property
    def modernbert_eval_length(self) -> int:
        """The terminal model's eval context (defaults to ``max_length``)."""
        return self.modernbert_max_length or self.max_length

    def _namespace_fields(self) -> dict:
        """The dict hashed into the version. Excludes ``spec_id`` (carried in the path) and
        ``modernbert_threshold`` (a late-bound re-filter, not a recompute trigger).

        **Unset (None) fields are omitted.** Adding an optional field would otherwise change the
        hash of every spec that predates it, silently re-pointing live namespaces at empty
        directories and orphaning their corpora — a real near-miss when the pooled/extractor fields
        were added (``fastpipe_v3`` moved da3893385e -> 47cfdfc9d2, which would have stranded 3,602
        already-extracted WARCs). ``test_spec.py`` pins the known hashes so this cannot drift again.
        """
        d = dataclasses.asdict(self)
        d.pop("spec_id", None)
        d.pop("modernbert_threshold", None)
        return {k: v for k, v in d.items() if v is not None}

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

    def pooled_ckpt_for(self, bucket: str) -> str:
        if self.pooled_ckpt is None:
            raise ValueError(f"{self.spec_id} has no pooled stage")
        return _rebucket(self.pooled_ckpt, bucket)

    # --- v2 (ModernBERT BEFORE JustText) 3-phase sub-paths ---
    def presurvivors_prefix(self, bucket: str = "gs://marin-us-east5") -> str:
        """v2 Phase A output: fastText-survivors carrying RAW html + tokens, NO JustText yet."""
        return f"{self.namespace(bucket)}/a_presurvivors"

    def keeplist_prefix(self, bucket: str = "gs://marin-us-east5") -> str:
        """v2 Phase B output: per-WARC {doc_id, modernbert_prob} for all pre-survivors."""
        return f"{self.namespace(bucket)}/b_keeplist"


V2_STEP_ORDER = ("decode", "body_strip", "fasttext", "tokenize", "modernbert", "justext")
# lpv11 line: pooled culls between fastText and ModernBERT (same tokens, ~180x cheaper), and the
# terminal extraction is the Rust resiliparse fork instead of jusText.
LPV11_STEP_ORDER = ("decode", "body_strip", "fasttext", "tokenize", "pooled", "modernbert", "resiliparse_rs")
# TEXT line: extraction FIRST (resiliparse-rs is cheap enough to run on every decoded doc), then
# every classifier reads lower(collapse(extracted)) — "clf_text". The pooled stage is an early-exit
# BAND (accept >= hi, drop < lo, send the middle to the terminal model at a short eval context).
TEXTPIPE_STEP_ORDER = ("decode", "resiliparse_rs", "clf_text", "fasttext", "tokenize", "pooled_band", "modernbert")


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
    # lpv11_fastpipe_v1: the first cascade targeting llm_pipeline_v1_1 rather than the 8B
    # high_quality run. Every stage is retrained/reselected for that target — mixing families would
    # have stages pulling toward definitions of "useful" that agree at only 0.325 F1.
    #
    # Thresholds are the operating points the cascade planner resolved on the 100k comparison sample
    # (fastText/pooled at recall 0.95, ModernBERT at 0.93 => F1 0.850 vs lpv11). Re-tune ModernBERT's
    # freely (stored probs); re-tuning fastText's or pooled's requires a new namespace.
    "lpv11_fastpipe_v1": PipelineSpec(
        spec_id="lpv11_fastpipe_v1",
        fasttext_model=FASTTEXT_LPV11_W640,
        fasttext_threshold=0.130,
        pooled_ckpt=POOLED_LPV11_10M,
        pooled_threshold=0.178,
        modernbert_ckpt=MODERNBERT_LPV11_10M,
        modernbert_threshold=0.410,
        extractor=Extractor.RESILIPARSE_RS,
        resiliparse_rs_commit=RESILIPARSE_RS_COMMIT,
        justext_max_html_chars=50_000_000,
        justext_timeout=60.0,
        step_order=LPV11_STEP_ORDER,
    ),
    # lpv11_fastpipe_v2_1: IDENTICAL cascade semantics to v2 (same models, thresholds, band, ctx —
    # same corpus content), new namespace for the 8M-scale storage contract: sharded work-list and
    # claims, V3 schemas without input_ids, zstd-12 text. Proven-at-10k v2 stays as the reference.
    "lpv11_fastpipe_v2_1": PipelineSpec(
        spec_id="lpv11_fastpipe_v2_1",
        fasttext_model=FASTTEXT_LPV11_TEXT_W640,
        fasttext_threshold=0.0048,
        pooled_ckpt=POOLED_LPV11_TEXT_90M,
        pooled_threshold=0.079,
        pooled_hi=0.8883,
        modernbert_ckpt=ETTIN68_LPV11_TEXT_10M,
        modernbert_threshold=0.4378,
        modernbert_max_length=2048,
        extractor=Extractor.RESILIPARSE_RS,
        resiliparse_rs_commit=RESILIPARSE_RS_COMMIT,
        justext_max_html_chars=50_000_000,
        justext_timeout=60.0,
        storage_version=3,
        step_order=TEXTPIPE_STEP_ORDER,
    ),
    # lpv11_fastpipe_v2_1_fused: IDENTICAL cascade semantics to v2_1, storage_version=4 — the
    # SINGLE-PHASE contract (fused_phase.py): Phase A runs on the TPU host's own CPUs and feeds
    # the chip through RAM, so NO presurvivors are ever written (no reaper, no _a_done, no
    # cross-phase claims). Output contract (kept/ + catalog + _b_done) is identical to v3.
    # Benchmark protocol: .agents/projects/fused_worker_benchmark.md.
    "lpv11_fastpipe_v2_1_fused": PipelineSpec(
        spec_id="lpv11_fastpipe_v2_1_fused",
        fasttext_model=FASTTEXT_LPV11_TEXT_W640,
        fasttext_threshold=0.0048,
        pooled_ckpt=POOLED_LPV11_TEXT_90M,
        pooled_threshold=0.079,
        pooled_hi=0.8883,
        modernbert_ckpt=ETTIN68_LPV11_TEXT_10M,
        modernbert_threshold=0.4378,
        modernbert_max_length=2048,
        extractor=Extractor.RESILIPARSE_RS,
        resiliparse_rs_commit=RESILIPARSE_RS_COMMIT,
        justext_max_html_chars=50_000_000,
        justext_timeout=60.0,
        storage_version=4,
        step_order=TEXTPIPE_STEP_ORDER,
    ),
    # lpv11_fastpipe_v2: the TEXTONLY-7d config from the planner's optimal-cascade search
    # (.agents/projects/planner_text_classifier_stages.md, 2026-08-23). Extraction runs FIRST; every
    # classifier consumes the extracted text; the pooled 90M gate is an early-exit band; the terminal
    # ettin68 runs at ctx 2048 (F1 within the 0.015 noise floor of 8192 at 3.9x the throughput).
    # Thresholds are the search's held-out operating points on the ``*_textraw_*`` (deployment-input)
    # columns of the 100k sample. ``modernbert_threshold`` stays late-bound over stored probs for
    # BAND docs; hi-accepted docs are kept without a terminal prob, so ``pooled_hi`` is in the hash.
    "lpv11_fastpipe_v2": PipelineSpec(
        spec_id="lpv11_fastpipe_v2",
        fasttext_model=FASTTEXT_LPV11_TEXT_W640,
        fasttext_threshold=0.0048,
        pooled_ckpt=POOLED_LPV11_TEXT_90M,
        pooled_threshold=0.079,
        pooled_hi=0.8883,
        modernbert_ckpt=ETTIN68_LPV11_TEXT_10M,
        modernbert_threshold=0.4378,
        modernbert_max_length=2048,
        extractor=Extractor.RESILIPARSE_RS,
        resiliparse_rs_commit=RESILIPARSE_RS_COMMIT,
        justext_max_html_chars=50_000_000,
        justext_timeout=60.0,
        step_order=TEXTPIPE_STEP_ORDER,
    ),
}


def get_spec(spec_id: str) -> PipelineSpec:
    """Resolve a spec by id, failing fast (no silent default) on an unknown id."""
    try:
        return SPECS[spec_id]
    except KeyError:
        raise ValueError(f"Unknown pipeline spec {spec_id!r}; known specs: {sorted(SPECS)}") from None
