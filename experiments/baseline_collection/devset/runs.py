# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load and search devset extraction runs.

A run directory (written by scripts/run_devset_extraction.py) looks like:

    outputs/marin_devset/<run_name>/
        manifest.json           # model, spec file + sha, flags, started_at
        records/<record_id>.json

Each record json holds the extracted text AND the reasoning trace, so spec
iteration can be done over both surfaces:

    {
      "record_id": "...", "hid": "...", "url": "...",
      "text": "...",              # parsed output (or sentinel)
      "reasoning": "...",         # reasoning trace (null if model emitted none)
      "raw_output": "...",        # unparsed completion (markers included)
      "error": null,              # error string if inference failed
      "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
      "html_chars": 0, "input_chars": 0,
      "duration_s": 0.0, "timestamp": "..."
    }
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

RUNS_ROOT = Path("outputs/marin_devset")

# Reserved sentinel prefixes meaning "the extractor dropped this doc".
FILTER_SENTINELS = (
    "[NO_USEFUL_CONTENT]",
    "[FILTERED_BY_PIPELINE]",
    "[DOCUMENT_FILTERED]",
)
CONTEXT_SENTINEL = "[CONTEXT_LENGTH_EXCEEDED]"


class Decision(str, Enum):
    KEEP = "keep"  # non-empty extracted text
    DROP = "drop"  # filter sentinel or empty output
    CONTEXT = "context"  # doc exceeded the context window (not a quality decision)
    ERROR = "error"  # inference failed


@dataclass
class ExtractionRecord:
    record_id: str
    hid: str = ""
    url: str = ""
    text: str = ""
    reasoning: str | None = None
    raw_output: str = ""
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    html_chars: int = 0
    input_chars: int = 0
    duration_s: float = 0.0
    timestamp: str = ""
    # chunked extraction (long docs split at natural breakpoints)
    n_chunks: int = 1
    chunks_run: int | None = None
    chunks: list[dict] | None = None  # per-chunk text/reasoning/tokens/error
    # two-stage runs (filter on a whole-doc text view, then extract)
    stage: str | None = None  # "filter" (dropped there) | "extract"
    filter_reasoning: str | None = None
    filter_text: str | None = None
    filter_tokens: int = 0

    @property
    def decision(self) -> Decision:
        if self.error:
            return Decision.ERROR
        stripped = (self.text or "").strip()
        if stripped.startswith(CONTEXT_SENTINEL):
            return Decision.CONTEXT
        if not stripped or any(stripped.startswith(s) for s in FILTER_SENTINELS):
            return Decision.DROP
        return Decision.KEEP

    @property
    def pred_keep(self) -> bool | None:
        """True/False keep decision; None if no decision was made (error/context)."""
        d = self.decision
        if d == Decision.KEEP:
            return True
        if d == Decision.DROP:
            return False
        return None


@dataclass
class Run:
    run_dir: Path
    manifest: dict = field(default_factory=dict)
    _records: dict[str, ExtractionRecord] | None = None

    @property
    def name(self) -> str:
        return self.run_dir.name

    @property
    def records_dir(self) -> Path:
        return self.run_dir / "records"

    def records(self) -> dict[str, ExtractionRecord]:
        if self._records is None:
            self._records = {}
            for p in sorted(self.records_dir.glob("*.json")):
                rec = _record_from_json(json.loads(p.read_text(encoding="utf-8")))
                self._records[rec.record_id] = rec
        return self._records

    def __getitem__(self, record_id: str) -> ExtractionRecord:
        return self.records()[record_id]

    def __contains__(self, record_id: str) -> bool:
        return record_id in self.records()

    def __len__(self) -> int:
        return len(self.records())


def _record_from_json(d: dict) -> ExtractionRecord:
    known = {f for f in ExtractionRecord.__dataclass_fields__}
    return ExtractionRecord(**{k: v for k, v in d.items() if k in known})


def load_run(run_dir: Path | str) -> Run:
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return Run(run_dir=run_dir, manifest=manifest)


def list_runs(root: Path | str = RUNS_ROOT) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if (p / "records").is_dir())


# ---------------------------------------------------------------------------
# Search over runs (text / reasoning / raw output) — the spec-iteration surface
# ---------------------------------------------------------------------------

SEARCH_FIELDS = ("text", "reasoning", "raw_output")


@dataclass
class SearchHit:
    record: ExtractionRecord
    field: str
    snippets: list[str]


def _snippets(haystack: str, pattern: re.Pattern, context: int = 120, max_snippets: int = 3) -> list[str]:
    out = []
    for m in pattern.finditer(haystack):
        start = max(0, m.start() - context)
        end = min(len(haystack), m.end() + context)
        snippet = haystack[start:end].replace("\n", " ")
        out.append(("…" if start > 0 else "") + snippet + ("…" if end < len(haystack) else ""))
        if len(out) >= max_snippets:
            break
    return out


def search(
    run: Run,
    query: str,
    fields: Sequence[str] = ("reasoning", "text"),
    regex: bool = False,
    ignore_case: bool = True,
    context: int = 120,
) -> list[SearchHit]:
    """Grep the run's outputs/reasoning traces. Returns hits with context snippets."""
    flags = re.IGNORECASE if ignore_case else 0
    pattern = re.compile(query if regex else re.escape(query), flags)
    hits: list[SearchHit] = []
    for rec in run.records().values():
        for f in fields:
            if f not in SEARCH_FIELDS:
                raise ValueError(f"unknown search field {f!r}; choose from {SEARCH_FIELDS}")
            value = getattr(rec, f) or ""
            snips = _snippets(value, pattern, context=context)
            if snips:
                hits.append(SearchHit(record=rec, field=f, snippets=snips))
    return hits
