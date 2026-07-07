# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pull a few real sample documents from each subcomponent eval set.

The subcomponent dashboard (`plot_subcomponents_dashboard.py`) shows a
loss-vs-tokens figure per Paloma / Uncheatable-Eval dataset + LIMA. This
script extracts a handful of *actual* documents from each of those eval sets
so the dashboard can show a "view samples" modal — the exact text the loss is
computed on, never a paraphrase or guess.

Sources (all in us-central2, matching the tokenized caches the sweep evals on):
  - Paloma:      gs://marin-us-central2/raw/paloma-fc6827/65cd6fc/{dir}/val/val-*.jsonl.gz
  - Uncheatable: gs://marin-us-central2/raw/uncheatable_eval/{snapshot}/{ds}_{window}.jsonl.gz
                 (frozen 2025-09-01→09-14 window)
  - LIMA:        gs://marin-us-central2/raw/lima_text-68958e9/68958e9/train.jsonl.gz

Only the first shard of each set is read, and only the first N records from it,
so egress is a few hundred KB per dataset. Each record's `text` is truncated to
`--max-chars` for display. Output is a single JSON consumed by the dashboard.

Usage:

    uv run python experiments/scaling_law_sweeps/extract_subcomponent_samples.py \\
        --output scratch/plots/subcomponents_10k/samples.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import subprocess
from pathlib import Path

from experiments.paloma import PALOMA_DATASETS_TO_DIR

logger = logging.getLogger(__name__)

PALOMA_RAW_BASE = "gs://marin-us-central2/raw/paloma-fc6827/65cd6fc"
UNCHEATABLE_SNAPSHOT_DIR = "gs://marin-us-central2/raw/uncheatable_eval/2026.06.28"
LIMA_TEXT_FILE = "gs://marin-us-central2/raw/lima_text-68958e9/68958e9/train.jsonl.gz"

# The 7 English Uncheatable-Eval categories used in the sweep (delphi's canonical
# 7). Files are named `{ds}_{start}to{end}.jsonl.gz`; we glob by the `{ds}_`
# prefix so the frozen date window need not be hard-coded here.
UNCHEATABLE_DATASETS = [
    "ao3_english",
    "arxiv_computer_science",
    "arxiv_physics",
    "bbc_news",
    "github_cpp",
    "github_python",
    "wikipedia_english",
]

DEFAULT_N = 5
# Full doc text is stored so the dashboard can offer an "expand" toggle. A hard
# cap bounds pathological docs (e.g. Paloma's PTB val is one ~400k-char stream)
# so the embedded JSON stays reasonable; the true length is recorded either way.
DEFAULT_MAX_CHARS = 50_000


def _read_records(path: str, n: int, must_contain: str | None = None) -> list[str]:
    """Read up to `n` `text` fields from a (optionally gzipped) jsonl object.

    Streams the object via `gcloud storage cat` (gcsfs has SSL-cert issues in
    some local envs) and stops after `n` matching records. If `must_contain` is
    set, only records whose text contains that substring are kept (used for LIMA,
    whose test-split records — which sort first — are user-prompt-only, so we skip
    to the User+Assistant conversations that make up the bulk of the eval).
    """
    proc = subprocess.Popen(["gcloud", "storage", "cat", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    stream = gzip.GzipFile(fileobj=proc.stdout) if path.endswith(".gz") else proc.stdout
    texts: list[str] = []
    try:
        for line in stream:
            if len(texts) >= n:
                break
            line = line.strip()
            if not line:
                continue
            text = json.loads(line).get("text")
            if text and (must_contain is None or must_contain in text):
                texts.append(text)
    finally:
        proc.stdout.close()  # closing the pipe stops the transfer early
        proc.wait()
    return texts


def _list_shards(glob_pattern: str) -> list[str]:
    res = subprocess.run(["gcloud", "storage", "ls", glob_pattern], capture_output=True, text=True, check=True)
    matches = sorted(line for line in res.stdout.splitlines() if line.strip())
    if not matches:
        raise FileNotFoundError(f"No files match {glob_pattern}")
    return matches


def _gather_samples(shards: list[str], n: int, must_contain: str | None = None) -> list[str]:
    """Gather up to `n` documents across the (sorted) shards of a dataset.

    When a dataset splits into many per-domain/per-subreddit shards, take one
    document from each of the first `n` shards so the sample spans domains
    rather than showing `n` docs from a single domain. When a dataset is a few
    large shards, fall back to taking multiple docs per shard.
    """
    per_shard = 1 if len(shards) >= n else n
    texts: list[str] = []
    for shard in shards:
        if len(texts) >= n:
            break
        for t in _read_records(shard, per_shard, must_contain):
            texts.append(t)
            if len(texts) >= n:
                break
    if not texts:
        raise ValueError(f"No text records found across {len(shards)} shards (first={shards[0]})")
    return texts


def collect_samples(n: int, max_chars: int) -> dict[str, dict]:
    """Return {dataset_key: {samples, char_len, source_glob, n_shards, n_shown}}.

    `samples[i]` is the full document text, hard-capped at `max_chars`.
    `char_len[i]` is the document's true length, so the dashboard can note when
    a stored sample was itself capped below the real doc length.
    """
    out: dict[str, dict] = {}

    def add(key: str, glob_pattern: str, must_contain: str | None = None) -> None:
        shards = _list_shards(glob_pattern)
        raw = _gather_samples(shards, n, must_contain)
        out[key] = {
            "samples": [t[:max_chars] for t in raw],
            "char_len": [len(t) for t in raw],
            "source_glob": glob_pattern,
            "n_shards": len(shards),
            "n_shown": len(raw),
        }
        logger.info("%-32s %d samples across %d shard(s)", key, len(raw), len(shards))

    for key, path_part in PALOMA_DATASETS_TO_DIR.items():
        # Match the tokenizer's own glob (experiments/paloma.py): some sets shard
        # as val-00000000.jsonl.gz, others (c4_100_domains) as val_<domain>.jsonl.gz.
        add(key, f"{PALOMA_RAW_BASE}/{path_part}/val/val*.jsonl.gz")

    for ds in UNCHEATABLE_DATASETS:
        add(ds, f"{UNCHEATABLE_SNAPSHOT_DIR}/{ds}_*.jsonl.gz")

    # LIMA: show User+Assistant conversations (loss is scored over both turns).
    # test-split records sort first and are user-prompt-only, so require a turn.
    add("lima", LIMA_TEXT_FILE, must_contain="\n\nAssistant:")
    return out


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", default="scratch/plots/subcomponents_10k/samples.json")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_N)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    args = parser.parse_args(argv)

    samples = collect_samples(args.num_samples, args.max_chars)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2))
    logger.info("wrote %s (%d datasets)", out_path, len(samples))


if __name__ == "__main__":
    main()
