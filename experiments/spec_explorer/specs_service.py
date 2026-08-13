# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble a display-ready description of what a curation method's spec *is*.

Three method classes are reconciled into one shape:

- **single_prompt** — the legacy one-call ``ExtractionSpec`` registry
  (``low/med_low/med/high_quality``): a system message + a rules-laden user
  template.
- **two_stage / one_call** — the multi-call ``pipeline_specs`` registry
  (``llm_pipeline_v1``, ``llm_simple_v1``, ``high_quality_v2``): several
  role-named prompts (filter/main + head/continuation) plus a frozen budget.
- **threshold** — the ``fastpipe_v3`` bands: a ModernBERT keep-top-X% classifier
  gate over a base extraction, not a prompt.

Methods with no LLM spec (``dclm``, ``nemotron_full``, ``resiliparse``) are
returned as **external** with a short description so the UI still has a card.
"""

from __future__ import annotations

import dataclasses

from experiments.baseline_collection.extraction_specs import SPECS
from experiments.baseline_collection.pipelines.pipeline_specs import (
    PIPELINES,
    OneCallPipeline,
    TwoStagePipeline,
)

# Human labels for the role-named prompts inside a pipeline.
_TWO_STAGE_PHASES = (
    ("filter_spec", "Stage 1 — Filter / quality gate"),
    ("head_spec", "Stage 2 — Extract (head chunk)"),
    ("cont_spec", "Stage 2 — Extract (continuation chunk)"),
)
_ONE_CALL_PHASES = (
    ("main_spec", "Merged judge + extract (voting head)"),
    ("cont_spec", "Extract (continuation chunk)"),
)

# fastpipe band -> keep-top fraction. The bare id is the full (100%) extraction.
_FASTPIPE_BANDS = {
    "fastpipe_v3": 100,
    "fastpipe_v3_100": 100,
    "fastpipe_v3_80": 80,
    "fastpipe_v3_60": 60,
    "fastpipe_v3_40": 40,
    "fastpipe_v3_20": 20,
}

# Methods whose extraction is a non-LLM external/heuristic tool.
_EXTERNAL = {
    "dclm": "DCLM's resiliparse-based extraction + fastText quality filter (the DCLM-baseline recipe).",
    "nemotron_full": "Nemotron-CC extraction/quality pipeline (full tier).",
    "resiliparse": "Plain resiliparse main-content extraction, no LLM and no quality filter.",
}


def list_methods() -> list[dict]:
    """All methods this service can describe, with their spec kind."""
    out: list[dict] = []
    for name in SPECS:
        out.append({"method": name, "kind": "single_prompt"})
    for name, pipe in PIPELINES.items():
        out.append({"method": name, "kind": pipe.type.value})
    for name in _FASTPIPE_BANDS:
        out.append({"method": name, "kind": "threshold"})
    for name in _EXTERNAL:
        out.append({"method": name, "kind": "external"})
    return out


def _single_prompt(method: str) -> dict:
    spec = SPECS[method]
    return {
        "method": method,
        "kind": "single_prompt",
        "title": f"{method} — single-call ExtractionSpec",
        "description": spec.description,
        "phases": [
            {"name": "System message", "role": "system", "text": spec.system_message},
            {"name": "Extraction template (rules)", "role": "user", "text": spec.extraction_template},
        ],
        "meta": {"spec_id": spec.spec_id},
    }


def _pipeline(method: str) -> dict:
    pipe = PIPELINES[method]
    phase_fields = _TWO_STAGE_PHASES if isinstance(pipe, TwoStagePipeline) else _ONE_CALL_PHASES
    phases = [{"name": label, "role": field, "text": getattr(pipe, field)} for field, label in phase_fields]
    meta = {"source_cert": pipe.source_cert, "pipeline_id": pipe.pipeline_id}
    if isinstance(pipe, TwoStagePipeline):
        meta["filter_view_main_content"] = pipe.filter_view_main_content
    if isinstance(pipe, OneCallPipeline):
        meta["filter_chunks"] = pipe.filter_chunks
    return {
        "method": method,
        "kind": pipe.type.value,
        "title": f"{method} — {pipe.type.value} pipeline ({pipe.source_cert})",
        "description": (
            "Multi-call, per-document pipeline. Each phase below is one chat() call; "
            "the filter/voting phase gates whether extraction runs."
        ),
        "phases": phases,
        "budget": dataclasses.asdict(pipe.budget),
        "meta": meta,
    }


def _threshold(method: str) -> dict:
    keep = _FASTPIPE_BANDS[method]
    return {
        "method": method,
        "kind": "threshold",
        "title": f"{method} — ModernBERT keep-top-{keep}% band",
        "description": (
            f"Not a prompt. A single fastpipe extraction is scored by a ModernBERT "
            f"usefulness classifier; this band keeps the top {keep}% of documents by "
            f"predicted probability. Lower bands are strict subsets of higher bands."
        ),
        "phases": [],
        "meta": {"classifier": "ModernBERT", "keep_top_percent": keep},
    }


def _external(method: str) -> dict:
    return {
        "method": method,
        "kind": "external",
        "title": f"{method} — external (non-LLM) extractor",
        "description": _EXTERNAL[method],
        "phases": [],
        "meta": {},
    }


def describe_spec(method: str) -> dict:
    """Return a display-ready spec description for ``method``.

    Raises ``KeyError`` (via a ValueError) if the method has no known spec.
    """
    if method in SPECS:
        return _single_prompt(method)
    if method in PIPELINES:
        return _pipeline(method)
    if method in _FASTPIPE_BANDS:
        return _threshold(method)
    if method in _EXTERNAL:
        return _external(method)
    raise ValueError(
        f"No spec known for method {method!r}. Known: "
        f"{sorted(set(SPECS) | set(PIPELINES) | set(_FASTPIPE_BANDS) | set(_EXTERNAL))}"
    )


def main() -> None:
    """Print the catalog + one full spec for a quick manual check."""
    import json

    print(json.dumps(list_methods(), indent=2))
    print("\n=== high_quality ===")
    print(json.dumps(describe_spec("high_quality"), indent=2)[:2000])
    print("\n=== llm_pipeline_v1 ===")
    hq = describe_spec("llm_pipeline_v1")
    print(f"kind={hq['kind']} phases={[p['name'] for p in hq['phases']]} budget_keys={list(hq['budget'])}")


if __name__ == "__main__":
    main()
