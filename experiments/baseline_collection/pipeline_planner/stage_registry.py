# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Declarative registry of pipeline stages for the cascade-planning dashboard.

This is the single extensibility seam: **adding a model = one entry here** (its score/text column must
exist in the precomputed artifact). Both the precompute job (`precompute_pipeline_matrix.py`) and the
browser engine consume this — the registry is dumped to JSON via `registry_json()` and shipped with the
static frontend.

Each stage is either a CLASSIFIER (a filter keyed on a scalar score column, kept above/below a threshold)
or an EXTRACTOR (terminal; produces text + an implicit keep/abstain decision). The LLM models appear as
both: a logprob-filter classifier AND a terminal extractor (the "an extractor can also filter" point).

Throughput is **docs/chip/s** on TPU (v6e-4, per chip) or **docs/s/core** on CPU. Sources are the
`EXTRACTOR_COMPARISON_RESULTS.md` timing tables (§1–§3b) and `modernbert_inference_benchmark.py` (the
per-context ModernBERT *compute ceiling* on v6e). See per-field notes; several are explicit assumptions.
"""

from __future__ import annotations

import dataclasses
import json
from enum import StrEnum


class Device(StrEnum):
    TPU = "tpu"  # throughput in docs/chip/s
    CPU = "cpu"  # throughput in docs/s/core


class StageKind(StrEnum):
    CLASSIFIER = "classifier"  # scalar-score filter
    EXTRACTOR = "extractor"  # terminal text producer (+ implicit abstain filter)


class Direction(StrEnum):
    HIGH_USEFUL = "high_useful"  # keep iff score >= threshold (probabilities)
    LOW_USEFUL = "low_useful"  # keep iff score <= threshold (LLM marker-logprob: low = useful)


@dataclasses.dataclass(frozen=True)
class Target:
    """An agreement target: the extractor whose decisions + text define "correct".

    F1/precision/recall are measured against ``gold_col`` (1 = the target kept the doc) and
    Levenshtein against the target's text. ``coverage_col`` is 1 where the target actually judged
    the doc; docs outside coverage are dropped from the doc universe entirely (metrics, compute and
    projections alike) so partial coverage can never silently skew a number.
    """

    id: str
    label: str
    gold_col: str
    coverage_col: str
    extractor_id: str  # the stage that IS this target (self-agreement = 1.0 by definition)


TARGETS: list[Target] = [
    Target("lpv11", "llm_pipeline_v1.1", "target_lpv11_useful", "cov_lpv11", "extract_lpv11"),
    Target("high_quality", "8B high_quality", "target_hq_useful", "cov_hq", "extract_8b"),
]
DEFAULT_TARGET_ID = "lpv11"


@dataclasses.dataclass(frozen=True)
class Stage:
    """One selectable pipeline stage. See module docstring for field semantics."""

    id: str
    label: str
    kind: StageKind
    device: Device
    throughput: float  # docs/chip/s (TPU) or docs/s/core (CPU)
    family: str  # group in the UI palette: "fasttext" | "modernbert" | "llm" | "justext"
    ctx: int | None = None  # eval context length where meaningful
    # classifier-only
    score_col: str | None = None
    direction: Direction | None = None
    # extractor-only
    text_col: str | None = None
    label_col: str | None = None  # None => never abstains (jusText)
    # Classifier reads this extractor's TEXT (key = extractor id minus "extract_", e.g. "resiliparse_rs")
    # instead of the HTML. The engine charges that extraction once per doc at the extractor's own
    # throughput/device (see engine.js header), so a text stage's own throughput is the model alone.
    requires_text: str | None = None
    # HYPOTHETICAL stage: reproduces the selected target's decisions exactly, at zero compute cost.
    # Its verdicts come from the target (so it is perfect for whichever target is selected, not for a
    # hard-coded one) and it contributes no time, which makes it a lower bound rather than a model.
    # As a filter it answers "if everything upstream were perfect, what would the rest cost?"
    oracle: bool = False
    # compute caveats surfaced in the UI
    tokenization_bound: bool = False  # TPU throughput is the compute ceiling; apply the global efficiency factor
    throughput_assumed: bool = False  # number is a stand-in, not directly measured
    throughput_source: str = ""


# ModernBERT throughput stored as the MEASURED v6e forward-pass CEILING (docs/chip/s) — real TPU truth from
# modernbert_inference_benchmark.py. The engine multiplies by `tokenization_efficiency` (also measured-anchored):
# 1.0 = pre-tokenized (achieves the ceiling) ; ~0.42 = the naive eval-job rate (EXTRACTOR_COMPARISON_RESULTS.md §2b:
# large 4.99/11.9=0.419, base 6.21/21.5=0.289). Default 1.0 because a 7.9M-WARC production run would pre-tokenize.
_BERT_CEIL = {
    ("base", 1024): 153.4,
    ("base", 2048): 90.3,
    ("base", 4096): 49.1,
    ("base", 8192): 21.5,
    ("large", 1024): 90.7,
    ("large", 2048): 42.2,
    ("large", 4096): 23.9,
    ("large", 8192): 11.9,
    # Ettin-68m geometry (h512 L19, 68M params): same ModernBERT architecture, a third the size of base
    # but only 1.69x its speed — it is nearly as DEEP (19 vs 22 layers), and at 8192 depth dominates.
    ("ettin68", 8192): 36.4,
    # Arch-sweep geometries (all ModernBERT-config), v6e-4 PRECISE run (arch_inference_benchmark warmup 15
    # iters 50, v6e-4-sweep-precise.json; mb-base 21.57 in that run vs registry 21.5). Params do NOT
    # predict speed — DEPTH does: tiny2 (L2) 23.9x, tiny4 (L4) 8.9x, ettin17 (L7) 8.0x, ettin32 (L10)
    # 3.9x, pruned8 (L8, but full h768) 2.5x. Run1 (warmup 3, iters 10) was within 2-7% of these.
    ("ettin32", 8192): 85.1,
    ("ettin17", 8192): 172.3,
    ("pruned8", 8192): 54.9,
    ("tiny4", 8192): 192.6,
    ("tiny2", 8192): 516.1,
    # Ettin-68m at SHORTER eval context. Extracted text averages ~926 tokens (median ~334), so most of an
    # 8192-token forward is padding: 1024 runs 6.7x faster for 0.0031 best-F1 (0.8705 vs 0.8736 over 99,996
    # docs) — inside the 0.015 noise floor — even though 20% of docs truncate. Same checkpoint, shorter
    # window. This is the single biggest throughput lever found for the TEXT track.
    ("ettin68", 4096): 48.36,
    ("ettin68", 2048): 141.52,
    ("ettin68", 1024): 244.37,
}
_BERT_SRC = "measured v6e forward-pass ceiling (modernbert_inference_benchmark.py); naive eval rate = ×0.42 (§2b)"
_ETTIN_SRC = (
    "MEASURED v6e-4 forward-pass ceiling, arch_inference_benchmark warmup=15 iters=50 "
    "(v6e-4-ettin68-precise.json); mb-base in the same run = 21.6 vs registry 21.5. Supersedes run1's "
    "37.5, whose ettin68 cells were noisy (ctx1024 read slower than ctx2048); the precise sweep is monotonic"
)
_SWEEP_SRC = (
    "MEASURED v6e-4 forward-pass ceiling, arch_inference_benchmark warmup=15 iters=50 "
    "(v6e-4-sweep-precise.json; mb-base 21.57 in the same run vs registry 21.5)"
)
# TEXT twins reuse their arch's HTML number: same weights shape, same ctx, one padded forward per doc.
# Text input is ~7% of the HTML's characters, so a length-bucketed deployment would be FASTER than this;
# the resiliparse-rs extraction the text needs is charged separately by the engine (requires_text).
# Score columns are the DEPLOYMENT input — lower(collapse(extract(raw_html))), one extraction — verified
# 2026-08-18 to match the as-trained input within +-0.002 F1 on 99,996 docs (compare_text_input_scores.py).
_TEXT_NOTE = (
    " — TEXT input: same arch/ctx as the HTML twin (conservative; shorter input); scored on the one-extraction "
    "deployment text (lower(collapse(extract(raw_html)))), verified within +-0.002 F1 of the as-trained input; "
    "extraction charged via requires_text"
)

# Measured-anchored efficiency, NOT a guess: 1.0 = pre-tokenized ceiling (both endpoints measured); ~0.42 = naive.
DEFAULT_TOKENIZATION_EFFICIENCY = 1.0


def _bert(
    stage_id: str,
    label: str,
    score_col: str,
    arch: str,
    ctx: int,
    source: str = _BERT_SRC,
    family: str = "modernbert",
    requires_text: str | None = None,
) -> Stage:
    return Stage(
        id=stage_id,
        label=label,
        kind=StageKind.CLASSIFIER,
        device=Device.TPU,
        throughput=_BERT_CEIL[(arch, ctx)],
        family=family,
        ctx=ctx,
        score_col=score_col,
        direction=Direction.HIGH_USEFUL,
        requires_text=requires_text,
        tokenization_bound=True,
        throughput_source=source + (_TEXT_NOTE if requires_text else ""),
    )


# LLM marker-logprob throughput (docs/chip/s) by (model, ctx). Source: EXTRACTOR_COMPARISON_RESULTS.md §2.
# ctx=None marks the full-document context (~26k tokens, MAX_DOC_TOKENS).
_LLM_LOGPROB_TP = {
    ("0p6b", None): 2.77,
    ("0p6b", 4096): 18.7,
    ("0p6b", 8192): 8.35,
    ("0p6b", 16384): 4.30,
    ("1p7b", None): 2.03,
    ("1p7b", 4096): 11.7,
    ("1p7b", 8192): 5.63,
    ("1p7b", 16384): 3.0,
}


def _llm_logprob(model: str, ctx: int | None, score_col: str) -> Stage:
    pretty = {"0p6b": "0.6B", "1p7b": "1.7B"}[model]
    ctx_lbl = "full" if ctx is None else f"{ctx // 1024}k"
    return Stage(
        id=f"llm_logprob_{model}_{ctx_lbl}",
        label=f"LLM logprob {pretty} @{ctx_lbl}",
        kind=StageKind.CLASSIFIER,
        device=Device.TPU,
        throughput=_LLM_LOGPROB_TP[(model, ctx)],
        family="llm",
        ctx=ctx,
        score_col=score_col,
        direction=Direction.LOW_USEFUL,  # low marker-logprob = useful; null = very-useful = always passes
        throughput_source="EXTRACTOR_COMPARISON_RESULTS.md §2",
    )


# Pooled transformers are ~26M params (vs ModernBERT base ~150M) — far cheaper per doc, and their
# cost is near-flat in context (1.7ms@1k -> 2.0ms@8k) so the edge GROWS with ctx. MEASURED on v6e-4
# by arch_inference_benchmark (warmup 15, iters 50, bf16, jax.block_until_ready), full-doc at 8192
# with one forward per doc. Same run put mb-base at 21.61 vs this registry's independently-derived
# 21.5 — a 0.5% cross-check that the two bases are interchangeable.
_POOLED_TP = {"pooled": 3932.6, "pooled_big": 3591.3}  # seqs/s/chip @8192
# funnelbert (4 warm-started ModernBERT-base layers -> 8x mean-pool -> 4 fresh global layers): precise run.
_FUNNELBERT_TP = 92.9
_POOLED_SRC = (
    "MEASURED v6e-4 forward-pass ceiling, arch_inference_benchmark warmup=15 iters=50 "
    "(v6e-4-pooled-precise.json); mb-base in the same run = 21.61 vs registry 21.5"
)


def _pooled(
    stage_id: str,
    label: str,
    score_col: str,
    arch: str = "pooled",
    ctx: int = 8192,
    family: str = "pooled",
    requires_text: str | None = None,
) -> Stage:
    return Stage(
        id=stage_id,
        label=label,
        kind=StageKind.CLASSIFIER,
        device=Device.TPU,
        throughput=_POOLED_TP[arch],
        family=family,
        ctx=ctx,
        score_col=score_col,
        direction=Direction.HIGH_USEFUL,
        requires_text=requires_text,
        tokenization_bound=True,
        throughput_source=_POOLED_SRC + (_TEXT_NOTE if requires_text else ""),
    )


CLASSIFIER_STAGES: list[Stage] = [
    # fastText (CPU, docs/s/core). ALL widths MEASURED over the 100k sample: 356.6-364.0, a 2% spread.
    # The old §3b pair (w160 358 -> w320 288) implied throughput falls with training width; direct
    # measurement disproves that (w320 is 357.6, not 288). Inference cost is set by hashing n-grams of
    # the INPUT document, so the training-time width does not change it. Do not extrapolate by width.
    Stage(
        "fasttext_w80",
        "fastText w80",
        StageKind.CLASSIFIER,
        Device.CPU,
        360.8,
        "fasttext",
        score_col="fasttext_useful_prob",
        direction=Direction.HIGH_USEFUL,
        throughput_source="MEASURED over the full 100k sample, single core, excludes parquet I/O "
        "(model_timing/<col>.json)",
    ),
    Stage(
        "fasttext_w160",
        "fastText w160",
        StageKind.CLASSIFIER,
        Device.CPU,
        356.6,
        "fasttext",
        score_col="fasttext_useful_prob_w160",
        direction=Direction.HIGH_USEFUL,
        throughput_source="MEASURED over the full 100k sample, single core, excludes parquet I/O "
        "(model_timing/<col>.json)",
    ),
    Stage(
        "fasttext_w320",
        "fastText w320",
        StageKind.CLASSIFIER,
        Device.CPU,
        357.6,
        "fasttext",
        score_col="fasttext_useful_prob_w320",
        direction=Direction.HIGH_USEFUL,
        throughput_source="MEASURED over the full 100k sample, single core, excludes parquet I/O "
        "(model_timing/<col>.json)",
    ),
    # ModernBERT 200k survivor, base arch, context sweep.
    _bert("bert_200k_1024", "ModernBERT 200k @1k", "bert_useful_prob_200k_ctx1024", "base", 1024),
    _bert("bert_200k_2048", "ModernBERT 200k @2k", "bert_useful_prob_200k_ctx2048", "base", 2048),
    _bert("bert_200k_4096", "ModernBERT 200k @4k", "bert_useful_prob_200k_ctx4096", "base", 4096),
    _bert("bert_200k_8192", "ModernBERT 200k @8k", "bert_useful_prob_200k_ctx8192", "base", 8192),
    # ModernBERT 1M variants @8192.
    _bert("bert_1M_8192", "ModernBERT 1M @8k", "bert_useful_prob_1M_ctx8192", "base", 8192),
    _bert("bert_base_1M_rand", "ModernBERT base-1M-rand", "bert_useful_prob_base_1M_rand", "base", 8192),
    _bert("bert_large_1M_surv", "ModernBERT large-1M-surv", "bert_useful_prob_large_1M_surv", "large", 8192),
    _bert("bert_base_10M", "ModernBERT base-10M", "bert_useful_prob_base_10M", "base", 8192),
    _bert("bert_large_10M", "ModernBERT large-10M", "bert_useful_prob_large_10M", "large", 8192),
    # LLM marker-logprob filters, full + truncated context.
    _llm_logprob("0p6b", None, "llm_logprob_marker_0p6b"),
    _llm_logprob("0p6b", 4096, "llm_logprob_marker_0p6b_ctx4k"),
    _llm_logprob("0p6b", 8192, "llm_logprob_marker_0p6b_ctx8k"),
    _llm_logprob("0p6b", 16384, "llm_logprob_marker_0p6b_ctx16k"),
    _llm_logprob("1p7b", None, "llm_logprob_marker_1p7b"),
    _llm_logprob("1p7b", 4096, "llm_logprob_marker_1p7b_ctx4k"),
    _llm_logprob("1p7b", 8192, "llm_logprob_marker_1p7b_ctx8k"),
    _llm_logprob("1p7b", 16384, "llm_logprob_marker_1p7b_ctx16k"),
    # --- lpv11-trained classifiers (target = llm_pipeline_v1.1) ------------------------------------
    # fastText throughput falls with training width (w160 358 -> w320 288, both measured §3b); the lpv11
    # widths were not timed, so each inherits its nearest measured width and w640 is extrapolated.
    Stage(
        "fasttext_lpv11_w80",
        "fastText lpv11 w80",
        StageKind.CLASSIFIER,
        Device.CPU,
        360.7,
        "fasttext_lpv11",
        score_col="fasttext_lpv11_prob_w80",
        direction=Direction.HIGH_USEFUL,
        throughput_source="MEASURED over the full 100k sample, single core, excludes parquet I/O "
        "(model_timing/<col>.json)",
    ),
    Stage(
        "fasttext_lpv11_w160",
        "fastText lpv11 w160",
        StageKind.CLASSIFIER,
        Device.CPU,
        361.7,
        "fasttext_lpv11",
        score_col="fasttext_lpv11_prob_w160",
        direction=Direction.HIGH_USEFUL,
        throughput_source="MEASURED over the full 100k sample, single core, excludes parquet I/O "
        "(model_timing/<col>.json)",
    ),
    Stage(
        "fasttext_lpv11_w256",
        "fastText lpv11 w256",
        StageKind.CLASSIFIER,
        Device.CPU,
        356.7,
        "fasttext_lpv11",
        score_col="fasttext_lpv11_prob_w256",
        direction=Direction.HIGH_USEFUL,
        throughput_source="MEASURED over the full 100k sample, single core, excludes parquet I/O "
        "(model_timing/<col>.json)",
    ),
    Stage(
        "fasttext_lpv11_w640",
        "fastText lpv11 w640 (shipped)",
        StageKind.CLASSIFIER,
        Device.CPU,
        364.0,
        "fasttext_lpv11",
        score_col="fasttext_lpv11_prob_w640",
        direction=Direction.HIGH_USEFUL,
        throughput_source="MEASURED over the full 100k sample, single core, excludes parquet I/O "
        "(model_timing/<col>.json)",
    ),
    _bert("bert_lpv11_base_1M", "ModernBERT lpv11 base-1M @8k", "bert_lpv11_prob_base_1M_c8192", "base", 8192),
    _bert("bert_lpv11_base_10M", "ModernBERT lpv11 base-10M @8k", "bert_lpv11_prob_base_10M_c8192", "base", 8192),
    _bert(
        "bert_lpv11_ettin68_10M",
        "Ettin-68m lpv11 10M @8k",
        "bert_lpv11_prob_ettin68_10M",
        "ettin68",
        8192,
        source=_ETTIN_SRC,
    ),
    # --- Arch-sweep 1M survivors (HTML): the speed/accuracy curve vs ModernBERT-base ---------------
    # frozen-7k F1: ettin68 .8367 · pruned8 .8249 · funnelbert .8172 · ettin32 .8161 · tiny4 .8110 ·
    # ettin17 .8068 · tiny2 .7888 (base-1M .8424). Pareto: ettin68 -> pruned8 -> funnelbert -> tiny4 -> tiny2.
    _bert(
        "sweep_ettin68_1M",
        "sweep Ettin-68m 1M (1.7x)",
        "bert_lpv11_prob_ettin68_1M",
        "ettin68",
        8192,
        _ETTIN_SRC,
        "arch_sweep",
    ),
    _bert(
        "sweep_pruned8_1M",
        "sweep pruned8 1M (2.5x)",
        "bert_lpv11_prob_pruned8_1M",
        "pruned8",
        8192,
        _SWEEP_SRC,
        "arch_sweep",
    ),
    _bert(
        "sweep_ettin32_1M",
        "sweep Ettin-32m 1M (3.9x)",
        "bert_lpv11_prob_ettin32_1M",
        "ettin32",
        8192,
        _SWEEP_SRC,
        "arch_sweep",
    ),
    _bert(
        "sweep_ettin17_1M",
        "sweep Ettin-17m 1M (8.0x)",
        "bert_lpv11_prob_ettin17_1M",
        "ettin17",
        8192,
        _SWEEP_SRC,
        "arch_sweep",
    ),
    _bert(
        "sweep_tiny4_1M", "sweep tiny4 1M (8.9x)", "bert_lpv11_prob_tiny4_1M", "tiny4", 8192, _SWEEP_SRC, "arch_sweep"
    ),
    _bert(
        "sweep_tiny2_1M", "sweep tiny2 1M (23.9x)", "bert_lpv11_prob_tiny2_1M", "tiny2", 8192, _SWEEP_SRC, "arch_sweep"
    ),
    Stage(
        "sweep_funnelbert_1M",
        "sweep funnelbert 1M (4.3x)",
        StageKind.CLASSIFIER,
        Device.TPU,
        _FUNNELBERT_TP,
        "arch_sweep",
        ctx=8192,
        score_col="funnelbert_lpv11_prob_1M",
        direction=Direction.HIGH_USEFUL,
        tokenization_bound=True,
        throughput_source=_SWEEP_SRC,
    ),
    # --- TEXT-trained classifiers: input = resiliparse-rs main content (deployment: ONE extraction on raw
    # HTML, then lowercase + whitespace-collapse; scored on exactly that). They can only run AFTER that
    # extraction, so requires_text="resiliparse_rs": the engine charges the extraction (291.8 docs/s/core,
    # CPU) once per doc for the first text stage in the chain; a resiliparse-rs terminal extractor then
    # costs nothing extra. frozen-7k best-F1 (noise floor 0.015 by bootstrap, so read ties, not ranks):
    # text-base-10M .8836 and text-ettin68-10M .8805 are TIED at the top — ettin68 MATCHES base at 1.69x
    # the throughput (36.4 vs 21.57 docs/chip/s), i.e. equal accuracy for 59% of the inference cost; that
    # is the claim, not "more accurate". Then text-base-1M .8671 · text-ettin68-1M .8647 ·
    # text-pooled-90M .8483 · text-pooled-10M .8413 · text-pooled-1M .8207 · fastText TEXT w640 .8196.
    _bert(
        "text_ettin68_10M_c1024",
        "TEXT Ettin-68m 10M @1k (6.7x)",
        "bert_lpv11_textraw_prob_ettin68_10M_c1024",
        "ettin68",
        1024,
        _ETTIN_SRC,
        "text",
        "resiliparse_rs",
    ),
    _bert(
        "text_ettin68_10M_c2048",
        "TEXT Ettin-68m 10M @2k (3.9x)",
        "bert_lpv11_textraw_prob_ettin68_10M_c2048",
        "ettin68",
        2048,
        _ETTIN_SRC,
        "text",
        "resiliparse_rs",
    ),
    _bert(
        "text_ettin68_10M_c4096",
        "TEXT Ettin-68m 10M @4k",
        "bert_lpv11_textraw_prob_ettin68_10M_c4096",
        "ettin68",
        4096,
        _ETTIN_SRC,
        "text",
        "resiliparse_rs",
    ),
    _bert(
        "text_base_10M_c2048",
        "TEXT ModernBERT base 10M @2k",
        "bert_lpv11_textraw_prob_base_10M_c2048",
        "base",
        2048,
        _BERT_SRC,
        "text",
        "resiliparse_rs",
    ),
    _bert(
        "text_base_10M",
        "TEXT ModernBERT base 10M",
        "bert_lpv11_textraw_prob_base_10M",
        "base",
        8192,
        _BERT_SRC,
        "text",
        "resiliparse_rs",
    ),
    _bert(
        "text_ettin68_10M",
        "TEXT Ettin-68m 10M",
        "bert_lpv11_textraw_prob_ettin68_10M",
        "ettin68",
        8192,
        _ETTIN_SRC,
        "text",
        "resiliparse_rs",
    ),
    _bert(
        "text_base_1M",
        "TEXT ModernBERT base 1M",
        "bert_lpv11_textraw_prob_base_1M",
        "base",
        8192,
        _BERT_SRC,
        "text",
        "resiliparse_rs",
    ),
    _bert(
        "text_ettin68_1M",
        "TEXT Ettin-68m 1M",
        "bert_lpv11_textraw_prob_ettin68_1M",
        "ettin68",
        8192,
        _ETTIN_SRC,
        "text",
        "resiliparse_rs",
    ),
    _pooled(
        "text_pooled_90M",
        "TEXT Pooled 90M",
        "pooled_lpv11_textraw_prob_90M",
        family="text",
        requires_text="resiliparse_rs",
    ),
    _pooled(
        "text_pooled_10M",
        "TEXT Pooled 10M",
        "pooled_lpv11_textraw_prob_10M",
        family="text",
        requires_text="resiliparse_rs",
    ),
    _pooled(
        "text_pooled_1M", "TEXT Pooled 1M", "pooled_lpv11_textraw_prob_1M", family="text", requires_text="resiliparse_rs"
    ),
    Stage(
        "fasttext_lpv11_text_w640",
        "TEXT fastText w640",
        StageKind.CLASSIFIER,
        Device.CPU,
        4875.5,
        "fasttext_text",
        score_col="fasttext_lpv11_textraw_prob_w640",
        direction=Direction.HIGH_USEFUL,
        requires_text="resiliparse_rs",
        throughput_source="MEASURED over the full 100k sample, single core, excludes parquet I/O "
        "(model_timing/<col>.json): 13.4x the HTML w640 (364.0) — fastText cost is set by input n-grams and "
        "text is ~7% of the HTML's characters. The resiliparse-rs extraction it needs is charged via requires_text.",
    ),
    _pooled("pooled_lpv11_1M_r5", "Pooled lpv11 1M-r5", "pooled_lpv11_prob_1M_r5"),
    _pooled("pooled_lpv11_10M", "Pooled lpv11 10M", "pooled_lpv11_prob_10M"),
    _pooled("pooled_lpv11_10M_e3", "Pooled lpv11 10M-e3", "pooled_lpv11_prob_10M_e3"),
    _pooled("pooled_lpv11_big_10M", "Pooled lpv11 big-10M", "pooled_lpv11_prob_big_10M", arch="pooled_big"),
    # 90M HTML completes the pooled 3x2 grid (1M/10M/90M x HTML/TEXT): frozen-7k .8331 vs .8483 TEXT.
    # TEXT advantage decays with data (+.0282 at 1M, +.0184 at 10M, +.0152 at 90M) toward a floor, still
    # equal to the noise floor at 90M — data does not buy back what markup truncation costs.
    _pooled("pooled_lpv11_90M", "Pooled lpv11 90M", "pooled_lpv11_prob_90M"),
    # Hypothetical: stand this in front of a real stage to ask "if everything upstream were perfect,
    # what would the rest of the pipeline cost?" It passes exactly the target's keep set, for free.
    Stage(
        "oracle_filter",
        "Oracle filter (perfect, free)",
        StageKind.CLASSIFIER,
        Device.CPU,
        0.0,
        "oracle",
        oracle=True,
        throughput_source="HYPOTHETICAL — perfect agreement with the selected target at zero cost; "
        "a lower bound, not a model",
    ),
]

EXTRACTOR_STAGES: list[Stage] = [
    # Hypothetical terminal: emits the target's own text and abstains exactly as it does, for free.
    # F1 = 1.0 and Levenshtein = 1.0 by construction — use it to price a pipeline's FILTERS alone,
    # with the extraction cost (normally the dominant term) removed.
    Stage(
        "oracle_extract",
        "Oracle extractor (perfect, free)",
        StageKind.EXTRACTOR,
        Device.CPU,
        0.0,
        "oracle",
        oracle=True,
        throughput_source="HYPOTHETICAL — reproduces the selected target exactly at zero cost",
    ),
    # 8B is the gold reference: F1 and Levenshtein are defined against its decisions/text.
    # NOTE: 0.46 was on a *medium* WARC subset; full-length docs generate more tokens → slower. User's
    # prior (~42 yr even at 1206 chips) implies effective ~0.26 docs/chip/s. Flagged provisional pending an
    # empirical full-WARC benchmark — this number dominates the headline projection.
    # The two targets are themselves selectable extractors: with the OTHER target selected each becomes
    # a measurable stage (e.g. "how well does the 8B run agree with lpv11?").
    Stage(
        "extract_lpv11",
        "llm_pipeline_v1.1",
        StageKind.EXTRACTOR,
        Device.TPU,
        0.46,
        "llm",
        text_col="text_lpv11",
        label_col="label_lpv11",
        throughput_assumed=True,
        throughput_source="PENDING — lpv11 two-stage extraction not benchmarked; 8B placeholder",
    ),
    Stage(
        "extract_8b",
        "8B high_quality",
        StageKind.EXTRACTOR,
        Device.TPU,
        0.46,
        "llm",
        text_col="text_8b",
        label_col="label_8b",
        throughput_assumed=True,
        throughput_source="§1 medium-subset; PROVISIONAL — likely optimistic, empirical full-WARC bench pending",
    ),
    Stage(
        "extract_1p7b",
        "1.7B extractor",
        StageKind.EXTRACTOR,
        Device.TPU,
        1.0,
        "llm",
        text_col="text_1p7b",
        label_col="label_1p7b",
        throughput_source="EXTRACTOR_COMPARISON_RESULTS.md §1",
    ),
    Stage(
        "extract_0p6b",
        "0.6B extractor",
        StageKind.EXTRACTOR,
        Device.TPU,
        2.0,
        "llm",
        text_col="text_0p6b",
        label_col="label_0p6b",
        throughput_assumed=True,
        throughput_source="PENDING — 0.6B extraction not benchmarked; ~2x 1.7B placeholder",
    ),
    # The XenonMolecule fork's Rust main-content engine (`resiliparse._extract_rs`) — a DIFFERENT fork
    # from marin's `resiliparse` core dep (krypticmouse's, via extract_simplified_dom -> markdownify).
    # Like jusText it always emits text, so label_col=None (no abstain).
    Stage(
        "extract_resiliparse_rs",
        "resiliparse-rs (fork)",
        StageKind.EXTRACTOR,
        Device.CPU,
        291.8,
        "resiliparse",
        text_col="text_resiliparse_rs",
        label_col=None,
        throughput_source="MEASURED 291.8 docs/s/core (3.43 ms/doc) single-threaded on an Iris CPU worker, "
        "commit 850891b, markdown + main_content over the fork's golden pages (mean 68.9 KB)",
    ),
    # jusText extracts everything (no abstain) → label_col=None.
    Stage(
        "extract_justext",
        "jusText",
        StageKind.EXTRACTOR,
        Device.CPU,
        9.43,
        "justext",
        text_col="text_justext",
        label_col=None,
        throughput_source="EXTRACTOR_COMPARISON_RESULTS.md §3 (standalone bench)",
    ),
]

ALL_STAGES: list[Stage] = CLASSIFIER_STAGES + EXTRACTOR_STAGES
STAGE_BY_ID: dict[str, Stage] = {s.id: s for s in ALL_STAGES}
for _s in CLASSIFIER_STAGES:
    if _s.requires_text and f"extract_{_s.requires_text}" not in STAGE_BY_ID:
        raise ValueError(f"{_s.id}: requires_text={_s.requires_text!r} names no extractor stage")

# Every artifact column the registry references — the precompute job asserts these all exist.
REQUIRED_SCORE_COLS: list[str] = [s.score_col for s in CLASSIFIER_STAGES if s.score_col]
REQUIRED_TEXT_COLS: list[str] = [s.text_col for s in EXTRACTOR_STAGES if s.text_col]
REQUIRED_LABEL_COLS: list[str] = sorted({s.label_col for s in EXTRACTOR_STAGES if s.label_col})


def registry_json() -> str:
    """Serialize the registry + defaults for the static frontend."""
    return json.dumps(
        {
            "default_tokenization_efficiency": DEFAULT_TOKENIZATION_EFFICIENCY,
            "default_target_id": DEFAULT_TARGET_ID,
            "targets": [dataclasses.asdict(t) for t in TARGETS],
            "stages": [dataclasses.asdict(s) for s in ALL_STAGES],
        },
        indent=2,
    )


if __name__ == "__main__":
    print(registry_json())
