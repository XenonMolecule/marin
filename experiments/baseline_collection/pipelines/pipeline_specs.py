# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extraction-pipeline registry: the multi-call replacement for ExtractionSpec.

Each "pipeline" is a system with its own role-named specs. The frozen source
certs live in ../small-rephraser (HANDOFF_final_pipelines.md, tag project-final):

- ``llm_pipeline_v1`` = two-stage v5.2: a filter call gates chunked extraction.
- ``llm_simple_v1``   = one-call oc3: a merged judge+extract voting head.

The ``pipeline_id`` is the GCS namespace key (parallel to the old ``spec_id``).
The filter/main spec is a swappable field so quality bands can be added later;
for now only the exact frozen specs are registered.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from experiments.baseline_collection.pipelines.prompt import PromptFormatter

PROMPTS_DIR = Path(__file__).parent / "prompts"
_MANIFEST = PROMPTS_DIR / "SHA256SUMS.txt"

# --- Sentinels & drop test (runtime SUBSTRING semantics, per the reference) ---

FILTER_SENTINELS = ("[NO_USEFUL_CONTENT]", "[FILTERED_BY_PIPELINE]", "[DOCUMENT_FILTERED]")
DROP_MARKER = "[NO_USEFUL_CONTENT]"
CONTEXT_SENTINEL = "[CONTEXT_LENGTH_EXCEEDED]"


def is_drop(text: str) -> bool:
    """A payload means DROP iff empty/whitespace or containing a sentinel.

    Substring (not startswith): filter specs may emit an observation line before
    the verdict, so the sentinel need not lead.
    """
    s = (text or "").strip()
    return not s or any(x in s for x in FILTER_SENTINELS)


class PipelineType(StrEnum):
    TWO_STAGE = "two_stage"
    ONE_CALL = "one_call"


@dataclass(frozen=True)
class PipelineBudget:
    """Frozen token/sampling budgets shared by both pipelines (Option B: Qwen
    tokenizer, view 24k / chunks 18k)."""

    filter_view_tokens: int = 24000
    chunk_tokens: int = 18000
    filter_max_tokens: int = 4096
    extract_max_tokens: int = 12288
    chunk_headroom: float = 0.98
    loop_threshold: int = 20
    temperature: float = 0.0
    loop_retry_temperature: float = 0.3
    # Stage-2 view re-render guard: resiliparse main_content view below this many
    # chars is re-rendered with main_content=False (view-collapse guard).
    filter_view_min_chars: int = 200


@dataclass(frozen=True)
class ExtractionGuardConfig:
    """Extraction-side guards added in llm_pipeline_v1.1 (markdown targets).

    Every flag defaults OFF, so a pipeline that omits this config (llm_pipeline_v1)
    runs the original filter→chunk→loop-guard→join path byte-for-byte. v1.1 turns
    them on. Nothing here touches the filter. See research_log HANDOFF_llm_pipeline_v1.1.
    """

    # Before extraction: pick the safe (script-preserving) preprocessing view when
    # the page's content lives in <script> blobs (view_choice.scripts_carry_content).
    view_choice: bool = False
    # Per chunk: coverage/code/tail guards drive a single t=0.3 reroll (replaces the
    # v1 loop-guard rung ladder), then loop-guard truncation.
    reroll_guards: bool = False
    # Per chunk: drop index-wall chunks (is_bloat_chunk) + strip machine-data lines.
    bloat_prefilter: bool = False
    # At merge: clean_output once on the joined text (straddle-boundary filters).
    merge_clean: bool = False
    # After a doc: half-chunk re-extract when the output reads like a summary.
    summarization_retry: bool = False
    reroll_temperature: float = 0.3
    half_chunk_divisor: int = 2


# All guards on — the llm_pipeline_v1.1 extraction upgrade set.
GUARDS_V1_1 = ExtractionGuardConfig(
    view_choice=True,
    reroll_guards=True,
    bloat_prefilter=True,
    merge_clean=True,
    summarization_retry=True,
)


@dataclass(frozen=True)
class TwoStagePipeline:
    pipeline_id: str
    source_cert: str
    filter_spec: str  # swappable quality gate (frozen filter.txt for now)
    head_spec: str
    cont_spec: str
    budget: PipelineBudget = field(default_factory=PipelineBudget)
    type: PipelineType = PipelineType.TWO_STAGE
    # Filter-view rendering: True prunes to main content (llm_pipeline_v1); False
    # renders the FULL page including chrome (high_quality_v2 — its extreme filter
    # is rendering-aware and validated on the unpruned rendering).
    filter_view_main_content: bool = True
    # Extraction-side guards. Default = all off (v1 behavior); v1.1 enables them.
    guards: ExtractionGuardConfig = field(default_factory=ExtractionGuardConfig)


@dataclass(frozen=True)
class OneCallPipeline:
    pipeline_id: str
    source_cert: str
    main_spec: str  # swappable merged judge+extract gate (frozen main.txt for now)
    cont_spec: str
    filter_chunks: int = 1  # oc3 uses 2 (chunk 2 as a second voter)
    budget: PipelineBudget = field(default_factory=PipelineBudget)
    type: PipelineType = PipelineType.ONE_CALL


def _load(rel: str) -> str:
    return (PROMPTS_DIR / rel).read_text(encoding="utf-8")


def verify_prompt_manifest() -> None:
    """Fail loudly if any vendored prompt drifted from its certified bytes."""
    lines = _MANIFEST.read_text(encoding="utf-8").splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        expected, rel = line.split(None, 1)
        actual = hashlib.sha256((PROMPTS_DIR / rel).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(
                f"Prompt fidelity check failed for {rel}: {actual} != {expected}. "
                "Vendored prompt drifted from the frozen small-rephraser cert."
            )


def build_two_stage_v1(budget: PipelineBudget | None = None) -> TwoStagePipeline:
    return TwoStagePipeline(
        pipeline_id="llm_pipeline_v1",
        source_cert="two_stage_v5.2",
        filter_spec=_load("two_stage/filter.txt"),
        head_spec=_load("two_stage/extract_head.txt"),
        cont_spec=_load("two_stage/extract_cont.txt"),
        budget=budget or PipelineBudget(),
    )


def build_high_quality_v2(budget: PipelineBudget | None = None) -> TwoStagePipeline:
    """Extreme-quality two-stage pipeline (extreme_filter_f6_ts3).

    Identical to llm_pipeline_v1 except the filter judges an EXTREME quality bar
    (~10-30% keep vs ~30%) and the filter view is the FULL page (main_content=
    False) — the extreme filter is rendering-aware and expects chrome present.
    Extraction prompts + all budgets are the same frozen files. Reference cert
    frozen at tag ``extreme-f6-final`` (branch ``extreme-v5-fiction``).
    """
    return TwoStagePipeline(
        pipeline_id="high_quality_v2",
        source_cert="extreme_filter_f6_ts3",
        filter_spec=_load("two_stage/extreme_filter_f6_ts3.txt"),
        head_spec=_load("two_stage/extract_head.txt"),
        cont_spec=_load("two_stage/extract_cont.txt"),
        budget=budget or PipelineBudget(),
        filter_view_main_content=False,
    )


def build_llm_pipeline_v1_1(budget: PipelineBudget | None = None) -> TwoStagePipeline:
    """Two-stage v1.1: llm_pipeline_v1's filter (unchanged) + markdown extraction
    (mdx_head/cont_d5) + the extraction-guard suite.

    Filter is deliberately the SAME as llm_pipeline_v1 (``filter.txt``) — the v1.1
    handoff changes only the extraction side and says to keep the filter locked.
    Head/cont are the d5 markdown specs (byte-identical except paragraph 1).
    """
    return TwoStagePipeline(
        pipeline_id="llm_pipeline_v1_1",
        source_cert="two_stage_v5.2+mdx_d5",
        filter_spec=_load("two_stage/filter.txt"),
        head_spec=_load("two_stage/mdx_head_d5.txt"),
        cont_spec=_load("two_stage/mdx_cont_d5.txt"),
        budget=budget or PipelineBudget(),
        filter_view_main_content=True,
        guards=GUARDS_V1_1,
    )


def build_one_call_v1(budget: PipelineBudget | None = None) -> OneCallPipeline:
    return OneCallPipeline(
        pipeline_id="llm_simple_v1",
        source_cert="one_call_oc3",
        main_spec=_load("one_call/main.txt"),
        cont_spec=_load("one_call/extract_cont.txt"),
        filter_chunks=2,
        budget=budget or PipelineBudget(),
    )


verify_prompt_manifest()

# Registry keyed by pipeline_id (the GCS namespace key). llm_simple_v1 ships once
# llm_pipeline_v1 is verified against the cert numbers.
PIPELINES: dict[str, TwoStagePipeline | OneCallPipeline] = {
    "llm_pipeline_v1": build_two_stage_v1(),
    "llm_pipeline_v1_1": build_llm_pipeline_v1_1(),
    "llm_simple_v1": build_one_call_v1(),
    "high_quality_v2": build_high_quality_v2(),
}


def get_pipeline(pipeline_id: str) -> TwoStagePipeline | OneCallPipeline:
    if pipeline_id not in PIPELINES:
        raise ValueError(f"Unknown pipeline_id {pipeline_id!r}; known: {sorted(PIPELINES)}")
    return PIPELINES[pipeline_id]


def make_formatter() -> PromptFormatter:
    """PromptFormatter bound to the vendored SFT templates (system.txt/user.txt)."""
    return PromptFormatter(template_dir=str(PROMPTS_DIR))
