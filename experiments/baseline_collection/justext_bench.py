"""Benchmark pure jusText-fork extraction throughput (docs/sec per single process).

Reads body-HTML docs from a router part file (rows have `text` = body_strip HTML), warms up
the fastText paragraph model, then times jusText extraction ONLY (no Levenshtein, no I/O in the
timed loop). Reports docs/sec + per-doc latency percentiles + throughput vs doc length.
"""
import argparse
import gzip
import json
import time

import fsspec
import justext

_STOP = None


def extract(html: str) -> int:
    global _STOP
    if _STOP is None:
        _STOP = justext.get_stoplist("English")
    if not html or not html.strip():
        return 0
    try:
        paras = justext.justext(html, _STOP)
        return sum(len(p.text) for p in paras if not p.is_boilerplate)
    except Exception:
        return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-file", required=True, help="router part jsonl.gz with a 'text' field")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=30)
    args = ap.parse_args()

    docs = []
    with fsspec.open(args.in_file, "rb") as f, gzip.open(f, "rt", encoding="utf-8") as g:
        for line in g:
            if line.strip():
                docs.append(json.loads(line).get("text", ""))
            if len(docs) >= args.n + args.warmup:
                break
    print(f"loaded {len(docs)} docs; warming up model on {args.warmup}...", flush=True)
    for d in docs[: args.warmup]:
        extract(d)
    bench = docs[args.warmup :]

    lat = []
    t0 = time.time()
    for d in bench:
        s = time.time()
        extract(d)
        lat.append((time.time() - s) * 1000)  # ms
    dt = time.time() - t0
    lat.sort()
    n = len(bench)
    mean_len = sum(len(d) for d in bench) / n
    p = lambda q: lat[min(n - 1, int(q * n))]
    print(f"\n=== pure jusText extraction (1 process) ===", flush=True)
    print(f"docs: {n} | mean body len: {mean_len:.0f} chars", flush=True)
    print(f"throughput: {n/dt:.1f} docs/sec  ({dt:.1f}s total)", flush=True)
    print(f"per-doc latency ms: mean {1000*dt/n:.1f} | p50 {p(0.5):.1f} | p90 {p(0.9):.1f} | p99 {p(0.99):.1f} | max {lat[-1]:.1f}", flush=True)


if __name__ == "__main__":
    main()
