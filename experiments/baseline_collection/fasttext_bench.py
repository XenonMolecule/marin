"""Benchmark fastText classifier inference throughput (docs/sec per process).

Per doc: to_fasttext_text(body) [ws-collapse + lowercase] then model.predict — the real
per-doc cost given body_strip HTML as input. Benches one or more models on the same sample.
"""
import argparse
import gzip
import json
import re
import time

import fsspec
import fasttext

_WS = re.compile(r"\s+")


def prep(t: str) -> str:
    return _WS.sub(" ", t).strip().lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma list of label=gs://path")
    ap.add_argument("--in-file", required=True, help="router part jsonl.gz with 'text' field")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()

    docs = []
    with fsspec.open(args.in_file, "rb") as f, gzip.open(f, "rt", encoding="utf-8") as g:
        for line in g:
            if line.strip():
                docs.append(prep(json.loads(line).get("text", "")))
            if len(docs) >= args.n + args.warmup:
                break
    mean_len = sum(len(d) for d in docs) / len(docs)
    print(f"loaded {len(docs)} docs (mean prepped len {mean_len:.0f} chars)", flush=True)

    for spec in args.models.split(","):
        label, path = spec.split("=", 1)
        local = f"/app/{label}.bin"
        with fsspec.open(path, "rb") as s, open(local, "wb") as d:
            d.write(s.read())
        m = fasttext.load_model(local)
        for d in docs[: args.warmup]:
            m.predict(d, k=-1)
        bench = docs[args.warmup :]
        lat = []
        t0 = time.time()
        for d in bench:
            s = time.time()
            m.predict(d, k=-1)
            lat.append((time.time() - s) * 1e6)  # microseconds
        dt = time.time() - t0
        lat.sort()
        n = len(bench)
        p = lambda q: lat[min(n - 1, int(q * n))]
        print(f"\n=== {label} ({path.split('/')[-1]}) ===", flush=True)
        print(f"throughput: {n/dt:.0f} docs/sec/process  ({dt:.1f}s for {n})", flush=True)
        print(f"per-doc us: mean {1e6*dt/n:.0f} | p50 {p(0.5):.0f} | p90 {p(0.9):.0f} | p99 {p(0.99):.0f} | max {lat[-1]:.0f}", flush=True)


if __name__ == "__main__":
    main()
