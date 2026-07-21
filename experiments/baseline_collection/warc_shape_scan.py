# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scan the 10k-pool WARCs' RAW HTML for code/table page-shapes (Zephyr, on Iris).

Extraction (resiliparse) FLATTENS `<table>`/`<pre>`/`<code>` structure, so code/table docs are
invisible in the extracted text. The raw HTML still carries high-precision intent signals —
syntax-highlighter libraries (Prism/highlight.js), `<pre>`/`<code>`, `class="language-…"`, and
table libraries (DataTables/ag-grid) + `<table>`/`<tr>` density. Scans the pre-decoded 10k-pool HTML
parquets (from decode_warcs_clean) and emits qualifying candidates. Detectors ported from the
small-rephraser extract_{code,table}_html.

Scans the already-decoded 10k-pool HTML parquets in-region (no CommonCrawl re-download). Run on Iris:
    uv run iris --cluster=marin job run --region us-east5 --cpu 2 --memory 8GB \\
        --enable-extra-resources --extra cpu -e WANDB_API_KEY <k> -e HF_TOKEN <t> \\
        -- python experiments/baseline_collection/warc_shape_scan.py --limit-files 150
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys

from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("warc_shape_scan")

# Raw HTML for the 10k pool is ALREADY decoded + stored here (from decode_warcs_clean) — scan this
# in-region instead of re-downloading WARCs from CommonCrawl (which is download-bound and slow).
DECODED = "gs://marin-us-east5/documents/bert_pipeline/decoded_10k"
OUT = "gs://marin-us-east5/scratch/provenance_10k_devset/warc_shape"

# --- CODE signals (syntax-highlighter libs = intentional code display, high precision) ---
CODE_LIBRARIES = [
    "prism.js",
    "prism.min.js",
    "prism-",
    'class="language-',
    "class='language-",
    "data-language=",
    "highlight.js",
    "highlight.min.js",
    "hljs",
    'class="hljs',
    "class='hljs",
    "highlightjs",
    "syntaxhighlighter",
    "shcore.js",
    "shbrush",
    "brush:",
    "prettify.js",
    "prettyprint",
    "google-code-prettify",
    "codehilite",
    'class="highlight"',
    'class="codehilite',
    "codemirror",
    "monaco-editor",
    "ace-builds",
    'class="rouge-',
    'class="sourcecode',
    "shiki",
    "gist.github.com",
]
CODE_ELEMENTS = ["<code", "<pre", "<samp", "<kbd", "<listing"]
CODE_CLASSES = [
    'class="code"',
    'class="codeblock',
    'class="code-block',
    'class="code-snippet',
    'class="snippet',
    'class="programlisting',
    "class='programlisting",
    'class="blob-code',
    'class="js-file-line',
    "data-line-number",
    'class="linenumber',
    'class="line-number',
]
_CODE_PATTERNS = [
    r"function\s+\w+\s*\(",
    r"def\s+\w+\s*\(",
    r"class\s+\w+\s*(?:\([^)]*\))?\s*:",
    r"=>\s*\{",
    r"(?:public|private|protected|static)\s+(?:\w+\s+)+\w+\s*\([^)]*\)\s*\{",
    r"func\s+\w+\s*\(",
    r"fn\s+\w+\s*(?:<[^>]+>)?\s*\(",
    r"import\s+(?:\{[^}]+\}|\*\s+as\s+\w+|\w+)\s+from\s+['\"]",
    r"from\s+\w+(?:\.\w+)*\s+import\s+",
    r"#include\s*[<\"][\w./]+[>\"]",
    r"using\s+namespace\s+\w+",
    r"if\s*\([^)]+\)\s*\{",
    r"for\s*\([^)]+\)\s*\{",
    r"#!/(?:usr/)?bin/(?:ba)?sh",
    r"SELECT\s+.+\s+FROM\s+\w+",
    r"CREATE\s+TABLE\s+\w+",
]
_CODE_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _CODE_PATTERNS]
_LANG_RE = re.compile(r"(?:language|lang|brush|syntax|code)-(\w+)", re.IGNORECASE)

# --- TABLE signals (noisier — layout tables masquerade as data; verify downstream) ---
TABLE_LIBS = [
    "datatables",
    "datatable",
    "ag-grid",
    "aggrid",
    "tabulator",
    "handsontable",
    "react-table",
    "v-data-table",
    "muitable",
    "mat-table",
    "slickgrid",
    "jqgrid",
    "tablesorter",
    "footable",
    "table-striped",
    "table-bordered",
    "table-hover",
    "table-responsive",
    "tanstack",
]
TABLE_ATTRS = [
    'role="grid"',
    'role="table"',
    'role="columnheader"',
    'class="table"',
    "class='table'",
    "data-table",
    "aria-rowcount",
]


def _code_score(html_lower: str, html: str) -> dict:
    libs = [k for k in CODE_LIBRARIES if k in html_lower]
    classes = [c for c in CODE_CLASSES if c in html_lower]
    pre, code_el = html_lower.count("<pre"), html_lower.count("<code")
    blocks = min(pre, code_el) or pre
    patterns = sum(len(r.findall(html)) for r in _CODE_RE)
    langs = sorted({m.lower() for m in _LANG_RE.findall(html_lower)})[:6]
    score = 0
    if libs:
        score += 40 + min(len(libs) * 5, 20)
    if blocks:
        score += 15 + min(blocks * 3, 15)
    if classes:
        score += 10
    if patterns:
        score += 10 + min(patterns * 2, 20)
    return {"score": min(score, 100), "highlighted": bool(libs), "blocks": blocks, "patterns": patterns, "langs": langs}


def _table_score(html_lower: str) -> dict:
    return {
        "tables": html_lower.count("<table"),
        "rows": html_lower.count("<tr"),
        "cells": html_lower.count("<td") + html_lower.count("<th"),
        "lib": any(k in html_lower for k in TABLE_LIBS),
        "attrs": any(a in html_lower for a in TABLE_ATTRS),
    }


def _excerpt(html: str, tags: tuple[str, ...], n: int = 1400) -> str:
    lo = html.lower()
    for t in tags:
        i = lo.find(t)
        if i >= 0:
            return html[max(0, i - 100) : i - 100 + n]
    return html[:n]


def scan_record(rec: dict) -> list[dict]:
    """Apply code+table shape detectors to one decoded doc's RAW html; emit 0-2 candidates."""
    html = rec.get("html") or ""
    if not html:
        return []
    hl = html.lower()
    text = (rec.get("text_body") or "")[:1600]
    out: list[dict] = []
    c = _code_score(hl, html)
    # Code: high precision — a syntax-highlighter lib, OR a strong composite score.
    if c["highlighted"] or c["score"] >= 55:
        out.append(
            {
                "url": rec["url"],
                "shape": "code",
                "score": c["score"],
                "metric": json.dumps(c),
                "snippet": text,
                "html_excerpt": _excerpt(html, ("<pre", "<code")),
            }
        )
    t = _table_score(hl)
    # Table: recall-oriented (verify later) — many rows or a real table library.
    if (t["rows"] >= 12 and t["tables"] >= 1 and t["cells"] >= 20) or t["lib"]:
        out.append(
            {
                "url": rec["url"],
                "shape": "table",
                "score": t["rows"],
                "metric": json.dumps(t),
                "snippet": text,
                "html_excerpt": _excerpt(html, ("<table",)),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default="*", help="filename-hash glob suffix, e.g. '0*' scans ~1/16 of the files")
    ap.add_argument("--max-workers", type=int, default=200)
    args = ap.parse_args()

    src = f"{DECODED}/data-{args.files}.parquet"
    logger.info("[shape-scan] scanning decoded-HTML parquets in-region us-east5: %s", src)
    out = f"{OUT}/s-{{shard:05d}}-of-{{total:05d}}.parquet"
    pipeline = (
        Dataset.from_files(src).load_parquet().flat_map(scan_record).reshard(16).write_parquet(out, skip_existing=True)
    )
    ctx = ZephyrContext(
        name="warc-shape-scan",
        max_workers=args.max_workers,
        resources=ResourceConfig(cpu=1, ram="8g", regions=["us-east5"], preemptible=True),
    )
    ctx.execute(pipeline)
    logger.info("[shape-scan] done -> %s", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
