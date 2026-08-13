# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Merge per-WARC tallies into the 24 x 5 grid and project it to the full pool.

**The estimator is shard-weighted, not WARC-weighted.** Steal mode splits a
WARC's batches across regions, so a WARC is only "finished" once every region
holding a piece of it has reported — which makes a WARC-complete estimator
throw away most of the work in flight, and makes it hostage to the slowest
region. A batch file is a contiguous slice of one WARC's records, and which
batches were stolen to which region is unrelated to their content, so the set of
tallied batches is an unbiased sample of the run's batches. The projection is
therefore

    projected_cell = (tokens in cell across tallied shards / tallied shards)
                     x total shards in the finished run

with the denominator measured directly off the extraction inventory (batches per
eligible WARC x the pool size) rather than assumed. The WARC-complete estimator
is still computed when enough WARCs have finished, and reported alongside as a
cross-check: the two disagreeing would mean the batch-sampling assumption is
wrong.

**Training tokens, not classifier tokens.** The topic model's ``gte_tokens`` are
capped at its 8192-token context, so summing them understates long documents and
is not a training-token estimate. The projection instead converts each cell's
exact character count with a chars-per-token ratio measured on that same cell's
llama3-tokenized subsample. Cells whose subsample is too thin fall back to their
topic row's ratio, then to the global one — the ratio varies by topic (code and
math tokenize denser than prose), so the row is a much better fallback than a
global mean.

**Uncertainty is over batch groups, not documents.** Documents within one WARC
are correlated — a crawl segment revisits the same sites — so a per-document
interval would be far too tight. The interval is a bootstrap that resamples whole
tallies, with a finite-population correction for having measured a known fraction
of the run's shards.

    python -m experiments.baseline_collection.grid_projection_report \\
        --tallies tallies/ --sample sample.json --total-shards 1905829 --out report/
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib

import numpy as np

logger = logging.getLogger(__name__)

NUM_BUCKETS = 5
BUCKET_LABELS = ("q0 junk", "q1", "q2", "q3", "q4 top")
# A cell needs at least this many llama3-tokenized documents before its own
# chars-per-token ratio is trusted; below it the ratio is noisy enough to move
# the projected tokens by more than the sampling error it is meant to refine.
MIN_SAMPLED_DOCS = 50
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 42
# Enough finished WARCs for the cross-check estimator to be worth printing.
MIN_WARCS_FOR_CROSSCHECK = 30


def load_tallies(tally_dir: str) -> list[dict]:
    paths = sorted(pathlib.Path(tally_dir).glob("*.json"))
    if not paths:
        raise ValueError(f"no tallies under {tally_dir}")
    tallies = [json.loads(p.read_text()) for p in paths]
    logger.info("loaded %d tally files", len(tallies))
    return tallies


def stack(tallies: list[dict], field: str) -> np.ndarray:
    """One 24x5 array per tally, in load order."""
    return np.array([np.asarray(t[field], dtype=np.float64) for t in tallies])


def chars_per_token(sampled_chars: np.ndarray, sampled_tokens: np.ndarray, sampled_docs: np.ndarray) -> np.ndarray:
    """Per-cell chars-per-llama3-token, backing off to the topic row then global."""
    global_ratio = sampled_chars.sum() / max(sampled_tokens.sum(), 1.0)
    row_tokens = sampled_tokens.sum(axis=1)
    row_ratio = np.where(row_tokens > 0, sampled_chars.sum(axis=1) / np.maximum(row_tokens, 1.0), global_ratio)

    ratio = np.broadcast_to(row_ratio[:, None], sampled_chars.shape).copy()
    trusted = (sampled_docs >= MIN_SAMPLED_DOCS) & (sampled_tokens > 0)
    ratio[trusted] = sampled_chars[trusted] / sampled_tokens[trusted]
    logger.info(
        "chars/token: global %.3f, %d/%d cells used their own ratio",
        global_ratio,
        int(trusted.sum()),
        trusted.size,
    )
    return ratio


def bootstrap_interval(
    per_tally_tokens: np.ndarray, per_tally_shards: np.ndarray, total_shards: float, fpc: float
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile bootstrap of the ratio estimator, resampling whole tallies.

    Resampling tallies rather than documents is what makes the interval honest:
    the unit of independent variation is a batch group, and documents inside one
    are heavily correlated.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = per_tally_tokens.shape[0]
    point = per_tally_tokens.sum(axis=0) / per_tally_shards.sum() * total_shards
    draws = np.empty((BOOTSTRAP_DRAWS, *point.shape))
    for i in range(BOOTSTRAP_DRAWS):
        pick = rng.integers(0, n, n)
        draws[i] = per_tally_tokens[pick].sum(axis=0) / per_tally_shards[pick].sum() * total_shards
    draws = point + (draws - point) * fpc
    return np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def warc_complete_crosscheck(
    tallies: list[dict], groups_per_warc: dict[str, int], ratio: np.ndarray, pool_warcs: int
) -> dict | None:
    """The WARC-weighted estimator, over WARCs whose every region-group reported."""
    by_warc: dict[str, list[dict]] = {}
    for tally in tallies:
        by_warc.setdefault(tally["warc"], []).append(tally)
    complete = [rows for warc, rows in by_warc.items() if len(rows) == groups_per_warc.get(warc, -1)]
    logger.info("%d WARCs have a tally, %d are complete across every region", len(by_warc), len(complete))
    if len(complete) < MIN_WARCS_FOR_CROSSCHECK:
        return None
    chars = np.array([np.sum([np.asarray(r["chars"], dtype=np.float64) for r in rows], axis=0) for rows in complete])
    tokens = (chars / ratio).sum(axis=0) / len(complete) * pool_warcs
    return {"warcs": len(complete), "llama_tokens": tokens.tolist(), "total": float(tokens.sum())}


def build_report(tally_dir: str, sample_path: str, total_shards: float, pool_warcs: int, out_dir: str) -> dict:
    tallies = load_tallies(tally_dir)
    sample = json.loads(pathlib.Path(sample_path).read_text())

    docs = stack(tallies, "docs")
    chars = stack(tallies, "chars")
    gte = stack(tallies, "gte_tokens")
    shards = np.array([float(t["num_shards"]) for t in tallies])
    s_docs = stack(tallies, "sampled_docs").sum(axis=0)
    s_chars = stack(tallies, "sampled_chars").sum(axis=0)
    s_tokens = stack(tallies, "sampled_llama_tokens").sum(axis=0)

    measured_shards = float(shards.sum())
    scale = total_shards / measured_shards
    fpc = float(np.sqrt(max(1.0 - measured_shards / total_shards, 0.0)))
    ratio = chars_per_token(s_chars, s_tokens, s_docs)

    per_tally_tokens = chars / ratio
    measured_tokens = per_tally_tokens.sum(axis=0)
    lo, hi = bootstrap_interval(per_tally_tokens, shards, total_shards, fpc)
    crosscheck = warc_complete_crosscheck(tallies, sample["groups_per_warc"], ratio, pool_warcs)

    report = {
        "measured_shards": measured_shards,
        "total_shards": total_shards,
        "coverage_pct": measured_shards / total_shards * 100,
        "sampled_warcs": len({t["warc"] for t in tallies}),
        "pool_warcs": pool_warcs,
        "scale": scale,
        "fpc": fpc,
        "topics": tallies[0]["topics"],
        "buckets": list(BUCKET_LABELS),
        "sampled": {
            "docs": docs.sum(axis=0).tolist(),
            "chars": chars.sum(axis=0).tolist(),
            "gte_tokens": gte.sum(axis=0).tolist(),
            "llama_tokens": measured_tokens.tolist(),
        },
        "chars_per_token": ratio.tolist(),
        "projected": {
            "docs": (docs.sum(axis=0) * scale).tolist(),
            "llama_tokens": (measured_tokens * scale).tolist(),
            "llama_tokens_lo": lo.tolist(),
            "llama_tokens_hi": hi.tolist(),
        },
        "crosscheck_warc_weighted": crosscheck,
    }

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "projection.json").write_text(json.dumps(report, indent=2))
    (out / "projection.md").write_text(render_markdown(report))
    logger.info(
        "%.2f%% of shards measured (%.0f/%.0f) -> x%.2f: %.2fB measured, %.1fB projected",
        report["coverage_pct"],
        measured_shards,
        total_shards,
        scale,
        measured_tokens.sum() / 1e9,
        measured_tokens.sum() * scale / 1e9,
    )
    if crosscheck:
        logger.info(
            "cross-check (%d complete WARCs, WARC-weighted): %.1fB",
            crosscheck["warcs"],
            crosscheck["total"] / 1e9,
        )
    return report


def render_markdown(report: dict) -> str:
    topics = report["topics"]
    tokens = np.asarray(report["projected"]["llama_tokens"])
    docs = np.asarray(report["projected"]["docs"])
    lo = np.asarray(report["projected"]["llama_tokens_lo"]).sum()
    hi = np.asarray(report["projected"]["llama_tokens_hi"]).sum()
    lines = [
        f"# llm_pipeline_v1_1 token projection -> {report['pool_warcs']:,} WARCs",
        "",
        f"Measured {report['measured_shards']:,.0f} of a projected {report['total_shards']:,.0f} batch shards "
        f"({report['coverage_pct']:.2f}%), across {report['sampled_warcs']:,} WARCs. Scale x{report['scale']:.2f}.",
        "",
        f"**Projected total: {tokens.sum() / 1e9:.1f}B llama3 tokens** "
        f"(95% CI {lo / 1e9:.1f}-{hi / 1e9:.1f}B), {docs.sum() / 1e6:.1f}M documents, pre-dedup.",
        "",
        "## Projected tokens (billions) by topic x quality",
        "",
        "| topic | " + " | ".join(BUCKET_LABELS) + " | total |",
        "|---|" + "---:|" * (NUM_BUCKETS + 1),
    ]
    for i in np.argsort(-tokens.sum(axis=1)):
        cells = " | ".join(f"{tokens[i, b] / 1e9:.2f}" for b in range(NUM_BUCKETS))
        lines.append(f"| {topics[i]} | {cells} | {tokens[i].sum() / 1e9:.2f} |")
    totals = " | ".join(f"{tokens[:, b].sum() / 1e9:.2f}" for b in range(NUM_BUCKETS))
    lines.append(f"| **total** | {totals} | {tokens.sum() / 1e9:.2f} |")
    if report["crosscheck_warc_weighted"]:
        cc = report["crosscheck_warc_weighted"]
        lines += [
            "",
            f"Cross-check: the WARC-weighted estimator over {cc['warcs']} fully-finished WARCs gives "
            f"{cc['total'] / 1e9:.1f}B, versus {tokens.sum() / 1e9:.1f}B shard-weighted.",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tallies", required=True, help="local directory of tally JSONs")
    parser.add_argument("--sample", required=True, help="sample.json written by the manifest builder")
    parser.add_argument(
        "--total-shards",
        type=float,
        required=True,
        help="batch shards the finished run will hold: shards-per-eligible-WARC x pool size",
    )
    parser.add_argument("--pool-warcs", type=int, default=10364)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_report(args.tallies, args.sample, args.total_shards, args.pool_warcs, args.out)


if __name__ == "__main__":
    main()
