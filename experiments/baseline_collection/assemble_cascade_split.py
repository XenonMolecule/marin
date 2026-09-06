# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble the cascade chunk parts into final train/dev/test jsonl.gz, in-region.

The cascade jobs write per-(WARC, rank, chunk) parts under {out_root}/{split}/. This
concatenates them into one file per split under {out_root}/final/:

  * dev  <- all val/  survivors
  * test <- all test/ survivors
  * train <- train/ survivors in the frozen shuffled-WARC order, capped at --target rows

Runs as a us-central2 CPU job so all reads/writes stay in-region (no egress). Prints row
counts and the source (useful / no_useful) breakdown per split — the BERT-filter distribution.
"""

import argparse
import gzip
import io
import json
import random
from collections import Counter

import fsspec


def chunk_paths(out_root: str, split: str) -> list[str]:
    """All chunk parts for a split, as fully-qualified gs:// paths."""
    fs, root = fsspec.core.url_to_fs(f"{out_root}/{split}")
    paths = fs.glob(f"{root}/data-*-r*-c*.jsonl.gz")
    return [p if p.startswith("gs://") else f"gs://{p}" for p in paths]


def warc_of(path: str) -> int:
    return int(path.split("/")[-1].split("-")[1])  # data-00761-r00-c0000.jsonl.gz -> 761


def read_lines(path: str):
    with fsspec.open(path, "rb") as f:
        data = f.read()
    for line in gzip.decompress(data).decode("utf-8").splitlines():
        if line:
            yield line


def write_split(out_root: str, name: str, lines: list[str]) -> None:
    buf = io.BytesIO()
    with gzip.open(buf, "wt", encoding="utf-8") as g:
        for ln in lines:
            g.write(ln + "\n")
    with fsspec.open(f"{out_root}/final/{name}.jsonl.gz", "wb") as d:
        d.write(buf.getvalue())


def source_breakdown(lines: list[str]) -> Counter:
    c = Counter()
    for ln in lines:
        c[json.loads(ln).get("source", "?")] += 1
    return c


def assemble_all(out_root: str, split: str, lines: list[str], log) -> None:
    bd = source_breakdown(lines)
    write_split(out_root, {"val": "dev", "test": "test", "train": "train"}[split], lines)
    log(f"{split}: wrote {len(lines)} rows | useful={bd.get('useful',0)} no_useful={bd.get('no_useful',0)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--target", type=int, default=350000, help="Train row cap.")
    ap.add_argument("--shuffle-seed", type=int, default=42)
    args = ap.parse_args()

    def log(m):
        print(m, flush=True)

    with fsspec.open(args.manifest, "rt", encoding="utf-8") as f:
        manifest = json.load(f)

    # dev / test: take ALL survivors (chunk order is fine).
    for split in ("val", "test"):
        paths = sorted(chunk_paths(args.out_root, split))
        lines = [ln for p in paths for ln in read_lines(p)]
        assemble_all(args.out_root, split, lines, log)

    # train: WARCs in the frozen shuffled order; within a WARC concat its chunk parts; cap at target.
    tr = list(manifest["train"])
    random.Random(args.shuffle_seed).shuffle(tr)
    by_warc: dict[int, list[str]] = {}
    for p in chunk_paths(args.out_root, "train"):
        by_warc.setdefault(warc_of(p), []).append(p)
    train_lines: list[str] = []
    for warc in tr:
        if len(train_lines) >= args.target:
            break
        for p in sorted(by_warc.get(warc, [])):
            for ln in read_lines(p):
                train_lines.append(ln)
    train_lines = train_lines[: args.target]
    assemble_all(args.out_root, "train", train_lines, log)


if __name__ == "__main__":
    main()
