# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paper figure: WebOrganizer topic-group composition per corpus, with relative corpus size.

Reads each corpus's grid_v1 datakit store artifact (per-cell training tokens keyed by topic
``cluster_id``), folds the 24 WebOrganizer topics into six groups, and draws either

* ``panel`` (the paper figure, ``topic_share_panel_sns``): one 100%-stacked composition row per corpus
  plus a slim linear total-tokens bar beside it, or
* ``mosaic``: the same rows with thickness proportional to sqrt(tokens).

The per-topic token tally is cached next to the figures (``topic_share_tally.json``); pass
``--refresh`` to re-read the store artifacts from GCS (a few KB each).

    uv run --with seaborn python -m experiments.scaling_law_sweeps.plot_topic_share_panel
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from rigging.filesystem import filesystem

from experiments.scaling_law_sweeps.plot_loss_vs_tokens_by_size import crop_pdf_to_ink

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent.parent.parent / "scratch" / "plots" / "olmo_bpb_10k" / "paper"
TALLY_CACHE = OUT_DIR / "topic_share_tally.json"
STORE_ARTIFACT = "gs://marin-us-east5/datakit/store/{corpus}_10k_gridv1/.artifact.json"
# Only the topic names (in cluster_id order) are read from here.
TOPIC_NAMES_DISTRIBUTION = "gs://marin-us-central1/metadata/grid_v1/dclm_10k/distribution.json"
NUM_TOPICS = 24

# Corpus key -> paper display name. Row order is by total tokens, not by this order.
CORPUS_LABELS = {
    "dclm": "DCLM-baseline",
    "nemotron_full": "Nemotron-CC",
    "fineweb_edu": "FineWeb-Edu",
    "fineweb_cc": "FineWeb",
    "resiliparse": "Resiliparse",
    "lpv11": "GC-LLM",
    "lpv11_fastpipe_v2": "GC-Cascade",
}
# Fixed group order = stack order (left to right) = legend order.
GROUPS = [
    ("Science, Health & Education", ["Science & Tech.", "Health", "Education & Jobs"], "#2a78d6"),
    ("Code & Hardware", ["Software Dev.", "Software", "Hardware"], "#8e63d2"),
    ("Humanities & Civic", ["History", "Politics", "Religion", "Literature", "Crime & Law", "Art & Design"], "#eb8a2f"),
    ("Business & Industry", ["Finance & Business", "Industrial", "Transportation"], "#8c7b6b"),
    (
        "Leisure & Lifestyle",
        [
            "Entertainment",
            "Games",
            "Sports & Fitness",
            "Food & Dining",
            "Home & Hobbies",
            "Fashion & Beauty",
            "Travel",
            "Social Life",
        ],
        "#3fae6c",
    ),
    ("Adult", ["Adult"], "#d64550"),
]
WHITE_INK_GROUPS = {0, 1, 5}
LABEL_PAD_PX = 4
LEGEND_FONTSIZE = 14
SEGMENT_FONTSIZE = 12


def read_tally() -> dict:
    """Training tokens per topic for every corpus, straight from the store artifacts."""
    fs = filesystem("gcs")
    with fs.open(TOPIC_NAMES_DISTRIBUTION) as fh:
        topics = list(json.load(fh)["topic"]["counts"])
    assert len(topics) == NUM_TOPICS, topics
    tally: dict = {"topics": topics}
    for corpus in CORPUS_LABELS:
        with fs.open(STORE_ARTIFACT.format(corpus=corpus)) as fh:
            artifact = json.load(fh)
        artifact = artifact.get("result", artifact)  # lpv11-family artifacts wrap the ledger
        row = [0] * NUM_TOPICS
        for cell in artifact["buckets"]:
            row[cell["cluster_id"]] += cell["total_tokens"]
        tally[corpus] = {"tokens_by_topic": row}
        logger.info("%-20s %7.2fB tokens", corpus, sum(row) / 1e9)
    return tally


def load_tally(refresh: bool) -> dict:
    if TALLY_CACHE.exists() and not refresh:
        return json.loads(TALLY_CACHE.read_text())
    tally = read_tally()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TALLY_CACHE.write_text(json.dumps(tally, indent=1))
    return tally


def group_rows(tally: dict) -> list[tuple[str, list[float], float]]:
    """(corpus, billions of tokens per group, total) sorted by total ascending."""
    topics = tally["topics"]
    assert sorted(t for _, ts, _ in GROUPS for t in ts) == sorted(topics), "every topic must land in one group"
    rows = []
    for corpus in CORPUS_LABELS:
        by_topic = dict(zip(topics, tally[corpus]["tokens_by_topic"], strict=True))
        cols = [sum(by_topic[t] for t in ts) / 1e9 for _, ts, _ in GROUPS]
        rows.append((corpus, cols, sum(cols)))
    return sorted(rows, key=lambda r: r[2])


def _stack(ax, yc: float, height: float, cols: list[float], tot: float, candidates: list) -> None:
    left = 0.0
    for g, v in enumerate(cols):
        share = v / tot * 100
        bar = ax.barh(yc, share, left=left, height=height, color=GROUPS[g][2], edgecolor="white", linewidth=0.8)[0]
        txt = ax.text(
            left + share / 2,
            yc,
            f"{share:.0f}%",
            ha="center",
            va="center",
            fontsize=SEGMENT_FONTSIZE,
            color="white" if g in WHITE_INK_GROUPS else "#222222",
        )
        candidates.append((txt, bar))
        left += share


def _legend_handles() -> list[Patch]:
    return [Patch(facecolor=c, edgecolor="white", label=n) for n, _, c in GROUPS]


def _fit_segment_labels(fig, candidates: list) -> None:
    """Shrink, then drop, any percentage label wider or taller than its segment."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for txt, bar in candidates:
        bb = bar.get_window_extent(renderer)

        def fits(txt=txt, bb=bb) -> bool:
            tb = txt.get_window_extent(renderer)
            return tb.width + 2 * LABEL_PAD_PX <= bb.width and tb.height <= bb.height

        if fits():
            continue
        for size in (11, 10, 9):
            txt.set_fontsize(size)
            if fits():
                break
        else:
            txt.remove()


def draw_panel(rows: list[tuple[str, list[float], float]]):
    candidates: list = []
    fig, (ax, ax_sz) = plt.subplots(1, 2, figsize=(13, 4.0), gridspec_kw={"width_ratios": [3.2, 1], "wspace": 0.14})
    y = list(range(len(rows)))[::-1]
    for yi, (_, cols, tot) in zip(y, rows, strict=True):
        _stack(ax, yi, 0.8, cols, tot, candidates)
        ax_sz.barh(yi, tot, height=0.8, color="#4a4a4a", edgecolor="white", linewidth=0.8)
        ax_sz.text(tot + 6, yi, f"{tot:.1f}B", va="center", fontsize=SEGMENT_FONTSIZE, fontweight="semibold")
    ax.set_yticks(y)
    ax.set_yticklabels([CORPUS_LABELS[r[0]] for r in rows])
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of corpus tokens (%)")
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=True)
    ax_sz.set_ylim(-0.55, len(rows) - 0.45)
    ax_sz.set_yticks([])
    ax_sz.set_xlim(0, max(r[2] for r in rows) * 1.45)
    ax_sz.set_xticks([0, 150, 300])
    ax_sz.set_xlabel("Total tokens (B)")
    ax_sz.grid(axis="y", visible=False)
    sns.despine(ax=ax_sz, left=True)
    fig.legend(
        handles=_legend_handles(),
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.8,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    return fig, candidates


def draw_mosaic(rows: list[tuple[str, list[float], float]]):
    candidates: list = []
    fig, ax = plt.subplots(figsize=(12, 8.0))
    gap = 0.9
    centers, ylabels = [], []
    top = 0.0
    for corpus, cols, tot in rows:
        h = math.sqrt(tot)
        yc = -(top + h / 2)  # stack downward so the smallest corpus is on top
        _stack(ax, yc, h, cols, tot, candidates)
        ax.text(101.5, yc, f"{tot:.1f}B", va="center", fontsize=SEGMENT_FONTSIZE, fontweight="semibold")
        centers.append(yc)
        ylabels.append(CORPUS_LABELS[corpus])
        top += h + gap
    ax.set_yticks(centers)
    ax.set_yticklabels(ylabels, fontsize=14)
    ax.set_ylim(-top + gap - 0.2, 0.2)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Share of corpus tokens (%)   —   row thickness scales with √tokens")
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax, left=True)
    ax.legend(
        handles=_legend_handles(),
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.8,
        bbox_to_anchor=(0.5, -0.13),
    )
    fig.tight_layout()
    return fig, candidates


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("panel", "mosaic"), default="panel")
    ap.add_argument("--refresh", action="store_true", help="Re-read the store artifacts from GCS.")
    args = ap.parse_args()

    rows = group_rows(load_tally(args.refresh))
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    fig, candidates = draw_panel(rows) if args.mode == "panel" else draw_mosaic(rows)
    _fit_segment_labels(fig, candidates)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"topic_share_{args.mode}_sns"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
    crop_pdf_to_ink(stem.with_suffix(".pdf"))
    logger.info("wrote %s.{pdf,png}", stem)


if __name__ == "__main__":
    main()
