# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Loader for the marin devset.

Each doc is an HTML file plus a sidecar meta json:

    record_00075_caselaw_findlaw_com_0405b7bc5c.html
    record_00075_caselaw_findlaw_com_0405b7bc5c.html.meta.json

Meta fields: url, hid, register, label (keep|weak_keep|drop|weak_drop|unsure|None),
label_source (human|judge_v3|relabel|None), confidence, benchmark, top_quality,
html_verified, gold, gold_path (relative to the devset root, e.g. "gold/<hid>.txt").
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

DEVSET_ROOT = Path("static/warcs/marin_devset_1934_html")

KEEP_LABELS = {"keep", "weak_keep"}
DROP_LABELS = {"drop", "weak_drop"}
WEAK_LABELS = {"weak_keep", "weak_drop"}


@dataclass(frozen=True)
class DevsetDoc:
    record_id: str  # filename stem, e.g. record_00075_caselaw_findlaw_com_0405b7bc5c
    hid: str
    url: str
    register: str
    label: str | None
    label_source: str | None
    confidence: str | None
    benchmark: bool
    top_quality: bool
    gold: bool
    html_path: Path
    gold_path: Path | None

    @property
    def web_domain(self) -> str:
        netloc = urlparse(self.url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc

    @property
    def binary_label(self) -> bool | None:
        """True = keep, False = drop, None = unlabeled/unsure."""
        if self.label in KEEP_LABELS:
            return True
        if self.label in DROP_LABELS:
            return False
        return None

    @property
    def is_weak_label(self) -> bool:
        return self.label in WEAK_LABELS

    def html(self) -> str:
        return self.html_path.read_text(encoding="utf-8", errors="replace")

    def gold_text(self) -> str | None:
        if self.gold_path is None or not self.gold_path.exists():
            return None
        return self.gold_path.read_text(encoding="utf-8", errors="replace")


def load_devset(root: Path | str = DEVSET_ROOT) -> list[DevsetDoc]:
    """Load all docs (sorted by record_id). Skips metas whose HTML is missing."""
    root = Path(root)
    docs: list[DevsetDoc] = []
    for meta_path in sorted(root.glob("*.html.meta.json")):
        html_path = meta_path.with_name(meta_path.name[: -len(".meta.json")])
        if not html_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        gold_path = root / meta["gold_path"] if meta.get("gold_path") else None
        docs.append(
            DevsetDoc(
                record_id=html_path.name[: -len(".html")],
                hid=meta["hid"],
                url=meta.get("url", ""),
                register=meta.get("register") or "unknown",
                label=meta.get("label"),
                label_source=meta.get("label_source"),
                confidence=meta.get("confidence"),
                benchmark=bool(meta.get("benchmark")),
                top_quality=bool(meta.get("top_quality")),
                gold=bool(meta.get("gold")),
                html_path=html_path,
                gold_path=gold_path,
            )
        )
    return docs


def docs_by_id(docs: list[DevsetDoc]) -> dict[str, DevsetDoc]:
    return {d.record_id: d for d in docs}
