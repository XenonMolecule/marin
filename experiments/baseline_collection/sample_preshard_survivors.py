"""Preshard cascade-survivor parts into ``{prefix}_{rank:02d}.txt.gz`` fastText-line shards
for ModernBERT-on-survivors training. MEMORY-BOUNDED (O(1)): single streaming pass, each kept
line is assigned a random shard and written straight to a GCS gzip stream (no in-RAM list, no
tmpfs — survivor docs are large). Output matches read_presharded / the train_classifier glob.

Each line is randomly assigned to one of ``--world`` shards, so any shard (and any --train-rows
prefix of the concatenated glob) is a uniform random sample of the parts — the trainer then slices
dataset size via --train-rows and shuffles. To keep ALL survivors, set ``--n`` above the part total
(p_keep clamps to 1.0 and the kept>=N stop never fires).
"""
import argparse
import contextlib
import random

import fsspec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parts", required=True, help="GCS dir of survivor part files (*.gz).")
    ap.add_argument("--out-prefix", required=True, help="Output prefix; writes {prefix}_{rank:02d}.txt.gz.")
    ap.add_argument("--n", type=int, required=True, help="Target kept docs (set above the part total to keep all).")
    ap.add_argument("--world", type=int, default=40, help="Number of output shards.")
    ap.add_argument("--total", type=int, required=True, help="Approx total docs in parts (sets p_keep).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--part-start", type=int, default=0, help="Process parts[start:end] (parallel slicing).")
    ap.add_argument("--part-end", type=int, default=None, help="Process parts[start:end] (parallel slicing).")
    args = ap.parse_args()

    fs = fsspec.filesystem("gcs")
    parts = sorted("gs://" + p for p in fs.glob(args.parts.replace("gs://", "") + "/*.gz"))
    parts = parts[args.part_start : args.part_end]
    n_parts = len(parts)
    random.seed(args.seed)
    p_keep = min(1.0, (args.n + 20_000) / args.total)  # slight over so we reach N before parts run out
    print(f"streaming {len(parts)} parts, p_keep={p_keep:.3f}, target {args.n} -> {args.world} shards", flush=True)

    # Progress sidecar on GCS (log-plane-independent): cheap to overwrite, readable via `gcloud cat`.
    progress_path = args.out_prefix + "_progress.txt"  # per-prefix → parallel jobs don't clobber

    def write_progress(done_parts: int, kept_so_far: int) -> None:
        with fsspec.open(progress_path, "wt", encoding="utf-8") as pf:
            pf.write(f"parts={done_parts}/{n_parts} kept={kept_so_far} pct={100 * done_parts / max(1, n_parts):.1f}\n")

    counts = [0] * args.world
    useful = 0
    kept = 0
    with contextlib.ExitStack() as stack:
        writers = [
            stack.enter_context(
                fsspec.open(f"{args.out_prefix}_{r:02d}.txt.gz", "wt", compression="gzip", encoding="utf-8")
            )
            for r in range(args.world)
        ]
        for pi, part in enumerate(parts):
            if kept >= args.n:
                break
            with fsspec.open(part, "rt", compression="gzip", encoding="utf-8") as f:
                for line in f:
                    if kept >= args.n:
                        break
                    if random.random() < p_keep:
                        r = random.randint(0, args.world - 1)
                        writers[r].write(line)
                        counts[r] += 1
                        kept += 1
                        if line.startswith("__label__useful"):
                            useful += 1
            if pi % 5 == 0 or pi == n_parts - 1:
                print(f"progress: part {pi + 1}/{n_parts} kept={kept}", flush=True)
                write_progress(pi + 1, kept)
    print(
        f"PRESHARD DONE: kept={kept} useful={useful} ({100 * useful / max(1, kept):.1f}%) "
        f"per-shard={counts} -> {args.out_prefix}_NN.txt.gz",
        flush=True,
    )


if __name__ == "__main__":
    main()
