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
    is_gold: bool = False
    # compute caveats surfaced in the UI
    tokenization_bound: bool = False  # TPU throughput is the compute ceiling; apply the global efficiency factor
    throughput_assumed: bool = False  # number is a stand-in, not directly measured
    throughput_source: str = ""


# ModernBERT throughput stored as the MEASURED v6e forward-pass CEILING (docs/chip/s) — real TPU truth from
# modernbert_inference_benchmark.py. The engine multiplies by `tokenization_efficiency` (also measured-anchored):
# 1.0 = pre-tokenized (achieves the ceiling) ; ~0.42 = the naive eval-job rate (EXTRACTOR_COMPARISON_RESULTS.md §2b:
# large 4.99/11.9=0.419, base 6.21/21.5=0.289). Default 1.0 because a 7.9M-WARC production run would pre-tokenize.
_BERT_CEIL = {
    ("base", 1024): 153.4, ("base", 2048): 90.3, ("base", 4096): 49.1, ("base", 8192): 21.5,
    ("large", 1024): 90.7, ("large", 2048): 42.2, ("large", 4096): 23.9, ("large", 8192): 11.9,
}
_BERT_SRC = "measured v6e forward-pass ceiling (modernbert_inference_benchmark.py); naive eval rate = ×0.42 (§2b)"

# Measured-anchored efficiency, NOT a guess: 1.0 = pre-tokenized ceiling (both endpoints measured); ~0.42 = naive.
DEFAULT_TOKENIZATION_EFFICIENCY = 1.0


def _bert(stage_id: str, label: str, score_col: str, arch: str, ctx: int) -> Stage:
    return Stage(
        id=stage_id,
        label=label,
        kind=StageKind.CLASSIFIER,
        device=Device.TPU,
        throughput=_BERT_CEIL[(arch, ctx)],
        family="modernbert",
        ctx=ctx,
        score_col=score_col,
        direction=Direction.HIGH_USEFUL,
        tokenization_bound=True,
        throughput_source=_BERT_SRC,
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


CLASSIFIER_STAGES: list[Stage] = [
    # fastText (CPU, docs/s/core). §3b. w80 timing not recorded → assumed equal to w160 (user-approved).
    Stage("fasttext_w80", "fastText w80", StageKind.CLASSIFIER, Device.CPU, 358.0, "fasttext",
          score_col="fasttext_useful_prob", direction=Direction.HIGH_USEFUL,
          throughput_assumed=True, throughput_source="assumed = w160 (§3b); w80 not timed"),
    Stage("fasttext_w160", "fastText w160", StageKind.CLASSIFIER, Device.CPU, 358.0, "fasttext",
          score_col="fasttext_useful_prob_w160", direction=Direction.HIGH_USEFUL,
          throughput_source="EXTRACTOR_COMPARISON_RESULTS.md §3b"),
    Stage("fasttext_w320", "fastText w320", StageKind.CLASSIFIER, Device.CPU, 288.0, "fasttext",
          score_col="fasttext_useful_prob_w320", direction=Direction.HIGH_USEFUL,
          throughput_source="EXTRACTOR_COMPARISON_RESULTS.md §3b"),
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
]

EXTRACTOR_STAGES: list[Stage] = [
    # 8B is the gold reference: F1 and Levenshtein are defined against its decisions/text.
    # NOTE: 0.46 was on a *medium* WARC subset; full-length docs generate more tokens → slower. User's
    # prior (~42 yr even at 1206 chips) implies effective ~0.26 docs/chip/s. Flagged provisional pending an
    # empirical full-WARC benchmark — this number dominates the headline projection.
    Stage("extract_8b", "8B extractor (gold)", StageKind.EXTRACTOR, Device.TPU, 0.46, "llm",
          text_col="text_8b", label_col="label_8b", is_gold=True, throughput_assumed=True,
          throughput_source="§1 medium-subset; PROVISIONAL — likely optimistic, empirical full-WARC bench pending"),
    Stage("extract_1p7b", "1.7B extractor", StageKind.EXTRACTOR, Device.TPU, 1.0, "llm",
          text_col="text_1p7b", label_col="label_1p7b",
          throughput_source="EXTRACTOR_COMPARISON_RESULTS.md §1"),
    Stage("extract_0p6b", "0.6B extractor", StageKind.EXTRACTOR, Device.TPU, 2.0, "llm",
          text_col="text_0p6b", label_col="label_0p6b",
          throughput_assumed=True, throughput_source="PENDING — 0.6B extraction not benchmarked; ~2x 1.7B placeholder"),
    # jusText extracts everything (no abstain) → label_col=None.
    Stage("extract_justext", "jusText", StageKind.EXTRACTOR, Device.CPU, 9.43, "justext",
          text_col="text_justext", label_col=None,
          throughput_source="EXTRACTOR_COMPARISON_RESULTS.md §3 (standalone bench)"),
]

ALL_STAGES: list[Stage] = CLASSIFIER_STAGES + EXTRACTOR_STAGES
STAGE_BY_ID: dict[str, Stage] = {s.id: s for s in ALL_STAGES}

# Every artifact column the registry references — the precompute job asserts these all exist.
REQUIRED_SCORE_COLS: list[str] = [s.score_col for s in CLASSIFIER_STAGES if s.score_col]
REQUIRED_TEXT_COLS: list[str] = [s.text_col for s in EXTRACTOR_STAGES if s.text_col]
REQUIRED_LABEL_COLS: list[str] = sorted({s.label_col for s in EXTRACTOR_STAGES if s.label_col})


def registry_json() -> str:
    """Serialize the registry + defaults for the static frontend."""
    return json.dumps(
        {
            "default_tokenization_efficiency": DEFAULT_TOKENIZATION_EFFICIENCY,
            "stages": [dataclasses.asdict(s) for s in ALL_STAGES],
        },
        indent=2,
    )


if __name__ == "__main__":
    print(registry_json())
