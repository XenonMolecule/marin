# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the cross-corpus comparison dataset for the quality x topic grid.

Merges the exact ``metadata/grid_v1/{corpus}/distribution.json`` tables written
by the grid_v1 runs with the projected grid from
:mod:`experiments.baseline_collection.grid_projection_report`, so all corpora sit
in one file with one topic ordering and one set of units.

**Everything is reported in gte tokens, and that is a deliberate constraint.**
grid_v1 stored token mass as the WebOrganizer tokenizer's length, which is capped
at its 8192-token context. That undercounts documents longer than the cap, so it
is not a training-token count — but it is the only unit measured identically for
every corpus, which is what a comparison needs. The llm_pipeline_v1_1 projection
carries a separate, true llama3 total; it is surfaced on its own rather than
mixed into the comparable columns.

The corpora are also at different pipeline stages — high_quality is post-dedup
and post-decontamination, the rest are pre-dedup, and llm_pipeline_v1_1 is a
projection from a partial run. That is recorded per corpus rather than smoothed
over, because a reader comparing raw sizes needs it.

    python -m experiments.baseline_collection.grid_compare \\
        --gridv1 gridv1/ --projection report/projection.json --out compare/
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib

logger = logging.getLogger(__name__)

NUM_TOPICS = 24
BUCKET_LABELS = ("q0 junk", "q1", "q2", "q3", "q4 top")

# Fixed display order, which is also the categorical colour order. Colour follows
# the corpus, not its rank, so this list must not be re-sorted by any metric.
GRIDV1_CORPORA: tuple[tuple[str, str, str], ...] = (
    ("high_quality_10k", "high_quality", "post-dedup + decon"),
    ("dclm_10k", "dclm", "pre-dedup"),
    ("nemotron_full_10k", "nemotron", "pre-dedup"),
    ("fineweb_edu_10k", "fineweb_edu", "pre-dedup"),
    ("fineweb_cc_10k", "fineweb_cc", "pre-dedup"),
    # APPENDED, never inserted: this list is the categorical colour order, so
    # placing resiliparse anywhere but the end would re-colour all five corpora
    # above it and break comparability with every plot already published.
    #
    # Post-decon like high_quality, but by a different route: the original
    # document tree was deleted after tokenization and was rebuilt by recovering
    # the surviving document set from the token cache — see
    # `.agents/projects/resiliparse_grid_url_recovery.md`. Text is byte-identical
    # to the raw extraction; 833,768 documents (0.29%) carry chunk-boundary
    # corruption from the cache and have no recoverable url.
    ("resiliparse_10k", "resiliparse", "post-dedup + decon (reconstructed)"),
)
PROJECTED_KEY = "llm_pipeline_v1_1"


def load_gridv1(path: pathlib.Path) -> dict:
    """One corpus's exact grid, with the topic order recovered from the tally dict.

    ``topic.counts`` is written as ``{label_names[i]: n}`` for i in 0..23, and
    dicts preserve insertion order, so its keys ARE the row order of
    ``grid.docs`` / ``grid.tokens``. Recovering it this way rather than
    hard-coding a list means a checkpoint whose label order ever changed would
    surface as a mismatch instead of silently transposing every row.
    """
    data = json.loads(path.read_text())
    topics = list(data["topic"]["counts"])
    if len(topics) != NUM_TOPICS:
        raise ValueError(f"{path} has {len(topics)} topics, expected {NUM_TOPICS}")
    return {
        "topics": topics,
        "docs": data["grid"]["docs"],
        "tokens": data["grid"]["tokens"],
        "score_mean": data["quality"]["score_mean"],
        "post_decon": data["post_decon"],
    }


def projected_corpus(projection: dict) -> dict:
    """The in-flight run's grid, scaled to the full pool, in the same units.

    ``sampled.gte_tokens`` is the directly comparable quantity; the llama3 total
    rides along separately so the page can show it without letting it into a
    column that is otherwise gte.
    """
    scale = projection["scale"]
    gte = [[v * scale for v in row] for row in projection["sampled"]["gte_tokens"]]
    return {
        "topics": projection["topics"],
        "docs": projection["projected"]["docs"],
        "tokens": gte,
        "score_mean": None,
        "post_decon": False,
        "llama_tokens": projection["projected"]["llama_tokens"],
        "coverage_pct": projection["coverage_pct"],
    }


def build(gridv1_dir: str, projection_path: str, out_dir: str) -> dict:
    root = pathlib.Path(gridv1_dir)
    corpora = []

    projection = json.loads(pathlib.Path(projection_path).read_text())
    proj = projected_corpus(projection)
    topics = proj["topics"]
    corpora.append(
        {
            "key": PROJECTED_KEY,
            "label": "llm_pipeline_v1_1",
            "stage": f"projected from {proj['coverage_pct']:.1f}% of shards, pre-dedup",
            "projected": True,
            **{k: proj[k] for k in ("docs", "tokens", "score_mean", "llama_tokens")},
        }
    )

    for key, label, stage in GRIDV1_CORPORA:
        loaded = load_gridv1(root / f"{key}.json")
        if loaded["topics"] != topics:
            raise ValueError(f"{key} topic order differs from the projection's — rows would not align")
        corpora.append(
            {
                "key": key,
                "label": label,
                "stage": stage,
                "projected": False,
                "docs": loaded["docs"],
                "tokens": loaded["tokens"],
                "score_mean": loaded["score_mean"],
                "llama_tokens": None,
            }
        )

    dataset = {"topics": topics, "buckets": list(BUCKET_LABELS), "corpora": corpora}
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps(dataset))
    for corpus in corpora:
        total = sum(sum(row) for row in corpus["tokens"])
        docs = sum(sum(row) for row in corpus["docs"])
        logger.info("%-22s %10.0f docs  %6.2fB gte tokens  (%s)", corpus["label"], docs, total / 1e9, corpus["stage"])
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gridv1", required=True, help="directory of {corpus}.json distribution files")
    parser.add_argument("--projection", required=True, help="projection.json for the in-flight run")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build(args.gridv1, args.projection, args.out)


if __name__ == "__main__":
    main()
