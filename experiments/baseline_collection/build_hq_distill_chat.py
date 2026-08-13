# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the Qwen3-0.6B distillation **chat** dataset from the high_quality 3000
distill parquet (``data/`` useful + ``data_no_useful/`` abstentions).

Each output row is a 3-turn chat that reproduces the teacher's DSPy scaffold,
with two deliberate changes for the student:

  * **Input is body-stripped.** The teacher saw raw HTML; the student is trained
    on ``preprocess_html_for_extraction(raw_html)`` (strip ``<script>``, keep
    ``<body>``) — the cheaper production representation. This is the intended
    input compression, not a teacher mismatch.
  * **No reasoning (non-thinking mode).** The assistant turn begins with the
    empty Qwen3 think block ``<think>\\n\\n</think>\\n\\n`` so the student matches
    ``enable_thinking=False`` inference, then emits the teacher's DSPy output
    fields ``[[ ## text ## ]] … [[ ## completed ## ]]``.

Both classes are included: useful pages (target = the cleaned document) and
``[NO_USEFUL_CONTENT]`` abstentions (target = the marker), balanced **1:1 within
each WARC** so the 12:1 natural prior doesn't collapse the student into always
abstaining.

No length filter here. The teacher *discarded* pages whose raw HTML exceeded its
char budget (``> 28672*6`` chars in ``download_and_extract._filter_by_token_length``),
so those are already absent from both ``data/`` and ``data_no_useful/``. Student-
context fit is enforced later, at tokenize time, by ``filter_by_context_length``;
we never truncate the student input. (A dense long-tail of in-budget pages was
token-truncated by the teacher in ``_format_prompts``; reproducing that exactly
would need per-row tokenization of the raw HTML and is not done.)

Splits reuse the classifier's frozen, snapshot-stratified held-out set
(``held_out_indices``): the 35 val + 35 test WARCs are identical to the
classifier's, so the distilled generator and the classifier share one held-out
set with zero leakage.

Map-only, one durable JSONL.gz shard per WARC (``skip_existing`` semantics via
an existence check), so progress is monotonic under preemption — same pattern as
``fasttext_useful_classifier.run_prep`` and ``build_hq_distill_dataset``.

Usage (run in us-central2, where the parquet lives)::

    uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \\
        --cpu 4 --memory 16GB --disk 20GB --priority interactive --extra cpu \\
        --enable-extra-resources --region us-central2 --job-name hq-distill-chat \\
        -- python experiments/baseline_collection/build_hq_distill_chat.py \\
        --tag smoke --limit-warcs 8        # smoke; drop both flags for the full build
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections.abc import Iterator

import fsspec
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from experiments.baseline_collection.extraction_specs import get_spec
from experiments.baseline_collection.fasttext_useful_classifier import (
    NO_USEFUL_DIR,
    USEFUL_DIR,
    body_strip,
    held_out_indices,
)
from experiments.baseline_collection.fasttext_useful_classifier import (
    _shard_snapshots as shard_snapshots,
)
from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)

# --- Output + teacher-fidelity constants -----------------------------------

OUTPUT_ROOT = "gs://marin-us-central2/datasets/high_quality_3000_distill_chat"

# Qwen3 non-thinking prefix: enable_thinking=False emits exactly this empty think
# block (see experiments/strip_thinking.py _EMPTY_THINK and QWEN_3_CHAT_TEMPLATE).
EMPTY_THINK = "<think>\n\n</think>\n\n"

# Frozen held-out: lowest-index WARC per snapshot -> val, next -> test (k=1),
# identical to the classifier so both models share the held-out set.
K_HOLDOUT = 1

# The teacher's DSPy scaffold for the extreme-quality spec. extraction_template
# still contains the literal ``{example}`` placeholder (and the spec's own braces
# are doubled), so ``.format(example=...)`` here reproduces exactly what the
# worker did in download_and_extract._format_prompts / run_extract_standalone.
_SPEC = get_spec("high_quality")
SYSTEM_MESSAGE = _SPEC.system_message
USER_TEMPLATE = _SPEC.extraction_template


def assistant_content(final_output: str) -> str:
    """Non-thinking assistant turn: empty think block + the teacher's DSPy output
    fields. ``final_output`` is the cleaned document (useful) or the abstention
    marker (negative)."""
    return f"{EMPTY_THINK}[[ ## text ## ]]\n{final_output}\n\n[[ ## completed ## ]]"


def chat_row(raw_html: str, final_output: str) -> dict:
    """One SFT chat example: system (DSPy signature) + user (template with
    body-stripped HTML) + assistant (non-thinking DSPy output)."""
    user = USER_TEMPLATE.format(example=body_strip(raw_html))
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant_content(final_output)},
        ]
    }


def read_pairs(path: str) -> Iterator[tuple[str, str]]:
    """Stream ``(raw_html, final_output)`` from a parquet shard, dropping rows with
    empty HTML or empty target. No length filter: over-budget pages were already
    discarded by the teacher (absent from the parquet), and student-context fit is
    enforced later by ``filter_by_context_length``."""
    for batch in pq.ParquetFile(path).iter_batches(columns=["raw_html", "final_output"], batch_size=1024):
        cols = batch.to_pydict()
        for html, final in zip(cols["raw_html"], cols["final_output"], strict=True):
            if not html or not final:
                continue
            yield html, final


def reservoir_sample(stream: Iterator[tuple[str, str]], k: int, rng: random.Random) -> list[tuple[str, str]]:
    """Uniform sample of ``k`` items from a stream of unknown length (Algorithm R),
    bounded memory. Returns all items if the stream is shorter than ``k``."""
    out: list[tuple[str, str]] = []
    for i, item in enumerate(stream):
        if i < k:
            out.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                out[j] = item
    return out


def stratified_caps(train_indices: list[int], snapshots: list[str], target_total: int) -> dict[int, int]:
    """Per-WARC positive caps that balance training examples ~equally over time.

    The 3000-WARC draw spans ~35 CC snapshots (2013-2017) unevenly, and WARC
    index correlates with snapshot, so an uncapped or prefix-selected train set
    over-represents some eras. We split ``target_total`` equally across the
    snapshots present in ``train_indices``, then equally across each snapshot's
    WARCs, returning ``{warc_index: max_positives}``.

    A WARC with fewer eligible positives than its cap simply contributes fewer
    (the build manifest reports achieved counts); we do not redistribute, so the
    realized total is ``<= target_total`` — temporal *balance* is the goal, not
    hitting the budget to the row.
    """
    by_snapshot: dict[str, list[int]] = {}
    for i in train_indices:
        by_snapshot.setdefault(snapshots[i], []).append(i)
    per_snapshot = max(1, target_total // len(by_snapshot))
    caps: dict[int, int] = {}
    for idxs in by_snapshot.values():
        per_warc = max(1, per_snapshot // len(idxs))
        for i in idxs:
            caps[i] = per_warc
    return caps


def _build_one_warc(spec: dict) -> dict:
    """Write one WARC's chat JSONL shard: useful rows (all, or a uniform sample
    capped at ``spec['max_pos']`` for temporal balancing) + an equal (1:1)
    reservoir sample of negatives. Streaming, atomic (tmp→rename), skip-existing
    for resumability."""
    out = f"{spec['out_root']}/{spec['split']}/data-{spec['index']:05d}.jsonl.gz"
    fs, rpath = fsspec.core.url_to_fs(out)
    if fs.exists(rpath):
        return {"index": spec["index"], "split": spec["split"], "skipped": True}

    max_pos = spec.get("max_pos")  # None -> keep all positives (full / val / test)
    tmp = f"{out}.tmp"
    with fsspec.open(tmp, "wt", compression="gzip", encoding="utf-8") as f:
        # Positives: stream all when uncapped (memory-safe); reservoir-sample when
        # capped so the kept subset is uniform over the WARC, not its prefix.
        if max_pos is None:
            n_pos = 0
            for html, final in read_pairs(spec["useful"]):
                f.write(json.dumps(chat_row(html, final), ensure_ascii=False) + "\n")
                n_pos += 1
        else:
            pos = reservoir_sample(read_pairs(spec["useful"]), max_pos, random.Random(spec["index"]))
            for html, final in pos:
                f.write(json.dumps(chat_row(html, final), ensure_ascii=False) + "\n")
            n_pos = len(pos)
        # Negatives: reservoir-sample exactly n_pos (1:1), independent seed.
        negs = reservoir_sample(read_pairs(spec["no_useful"]), n_pos, random.Random(spec["index"] + 1))
        for html, final in negs:
            f.write(json.dumps(chat_row(html, final), ensure_ascii=False) + "\n")

    tfs, trpath = fsspec.core.url_to_fs(tmp)
    tfs.mv(trpath, rpath)
    return {
        "index": spec["index"],
        "split": spec["split"],
        "snapshot": spec["snapshot"],
        "n_useful": n_pos,
        "n_no_useful": len(negs),
        "skipped": False,
    }


def run_build(tag: str | None, limit_warcs: int | None, only_split: str, target_train_examples: int | None) -> None:
    out_root = OUTPUT_ROOT if not tag else f"{OUTPUT_ROOT}_{tag}"

    useful_shards = sorted(fsspec_glob(f"{USEFUL_DIR}/*.parquet"))
    no_useful_shards = sorted(fsspec_glob(f"{NO_USEFUL_DIR}/*.parquet"))
    if not useful_shards or not no_useful_shards:
        raise RuntimeError(f"missing parquet under {USEFUL_DIR} or {NO_USEFUL_DIR}")
    n_total = min(len(useful_shards), len(no_useful_shards))

    # Compute splits over ALL shards so assignment is stable regardless of
    # --limit-warcs (mirrors fasttext_useful_classifier.run_prep).
    snapshots = shard_snapshots(useful_shards[:n_total])
    val_idx, test_idx = held_out_indices(snapshots, K_HOLDOUT)

    def split_of(i: int) -> str:
        return "val" if i in val_idx else "test" if i in test_idx else "train"

    # Temporal stratification: cap per-WARC positives so training examples are
    # ~balanced across the ~35 snapshots (2013-2017). Computed over the FULL train
    # set so caps are stable under --limit-warcs. No target -> keep everything
    # (every era is still present, just at its natural volume).
    train_indices = [i for i in range(n_total) if split_of(i) == "train"]
    caps = stratified_caps(train_indices, snapshots, target_train_examples) if target_train_examples else {}

    specs = [
        {
            "index": i,
            "useful": useful_shards[i],
            "no_useful": no_useful_shards[i],
            "split": split_of(i),
            "snapshot": snapshots[i],
            "out_root": out_root,
            "max_pos": caps.get(i),  # None unless this is a train WARC and a target was set
        }
        for i in range(n_total)
    ]
    if only_split != "all":
        specs = [s for s in specs if s["split"] == only_split]
    if limit_warcs is not None:
        specs = specs[:limit_warcs]

    logger.info(
        "build %d WARCs (train=%d val=%d test=%d) over %d snapshots, target_train_examples=%s -> %s",
        len(specs),
        sum(1 for s in specs if s["split"] == "train"),
        sum(1 for s in specs if s["split"] == "val"),
        sum(1 for s in specs if s["split"] == "test"),
        len(set(snapshots)),
        target_train_examples,
        out_root,
    )
    manifest = f"{out_root}/_build_manifest-{{shard:05d}}-of-{{total:05d}}.jsonl.gz"
    pipeline = Dataset.from_iterable(specs).map(_build_one_warc).write_jsonl(manifest, skip_existing=False)
    ctx = ZephyrContext(name="hq-distill-chat", max_workers=256, resources=ResourceConfig(cpu=2, ram="16g"))
    ctx.execute(pipeline)
    logger.info("build done -> %s/{train,val,test}/", out_root)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default=None, help="Suffix the output dir (e.g. 'smoke') to isolate test runs.")
    parser.add_argument("--limit-warcs", type=int, default=None, help="Smoke: only process the first N WARCs.")
    parser.add_argument(
        "--only-split", default="all", choices=["all", "train", "val", "test"], help="Restrict to one split."
    )
    parser.add_argument(
        "--target-train-examples",
        type=int,
        default=None,
        help="Stratified subsample: ~this many TRAIN useful examples, balanced equally across "
        "snapshots (negatives matched 1:1). Omit to keep all useful examples.",
    )
    args = parser.parse_args()
    run_build(args.tag, args.limit_warcs, args.only_split, args.target_train_examples)


if __name__ == "__main__":
    main()
