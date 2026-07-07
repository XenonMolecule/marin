"""Stage-2 cascade data builder: score docs with the stage-1 fastText and keep
only the survivors (P(useful) >= threshold). The survivors are the *hard residual*
the cheap stage-1 cannot confidently reject — a specialized stage-2 trains on them.

Sources from WARC shards DISJOINT from stage-1's train/val/test (default >= 300) so
stage-1's scores on these docs are not memorized (no leakage). Writes a gzipped
fastText train file (``__label__x <body_strip text>``) preserving the natural
post-filter ratio. Runs in-region (us-central2); parallel across shards, with the
stage-1 model loaded once and fork-shared (copy-on-write) so RAM stays ~1 model.
"""
import argparse, logging, os, re
import multiprocessing as mp
import fsspec
import fasttext
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("survivor")

# --- inlined from fasttext_useful_classifier.py (avoid the fray import) ---
LABEL_USEFUL = "__label__useful"
LABEL_NO_USEFUL = "__label__no_useful"
_SCRIPT_TAG_RE = re.compile(r"<script\b[^>]*>.*?</script\s*>", re.IGNORECASE | re.DOTALL)
_BODY_TAG_RE = re.compile(r"<body\b[^>]*>(.*?)</body\s*>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"\s+")


def body_strip(html: str) -> str:
    cleaned = _SCRIPT_TAG_RE.sub("", html)
    bodies = [m.group(1) for m in _BODY_TAG_RE.finditer(cleaned)]
    return "".join(bodies) if bodies else cleaned


def to_fasttext_text(html: str, representation: str = "body_strip") -> str:
    return _WS_RE.sub(" ", body_strip(html)).strip().lower()


def predict_useful_prob(model, text: str) -> float:
    # low-level predict (high-level builds a no-copy array that breaks on numpy 2.x)
    for prob, lab in model.f.predict(text, -1, 0.0, "strict"):
        if lab == LABEL_USEFUL:
            return float(prob)
    return 0.0

USEFUL_TMPL = "{base}/data/data-{i:05d}-of-03000.parquet"
NOUSE_TMPL = "{base}/data_no_useful/data-{i:05d}-of-03000.parquet"

# per-worker globals (set by _init_worker under spawn — fresh process, clean gRPC)
_MODEL = None
_BASE = None
_THR = 0.0
_PARTS = None


def _init_worker(model_path, base, thr, parts):
    global _MODEL, _BASE, _THR, _PARTS
    _MODEL = fasttext.load_model(model_path)
    _BASE, _THR, _PARTS = base, thr, parts


def iter_html(path):
    try:
        with fsspec.open(path, "rb") as f:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(columns=["raw_html"], batch_size=2048):
                for h in batch.column("raw_html").to_pylist():
                    if h:
                        yield h
    except FileNotFoundError:
        return


def score_shard(i):
    """Score one WARC shard; write its survivors DIRECTLY to a GCS part file
    ``{_PARTS}/surv_{i}.txt.gz`` (gzip). Skips if the part already exists (resumable).
    NO stdout logging here (worker flooding backpressure-freezes the pool)."""
    part = f"{_PARTS}/surv_{i:05d}.txt.gz"
    fs = fsspec.filesystem("gcs")
    if fs.exists(part.replace("gs://", "")):
        return i, -1, -1, -1, -1  # already done (sentinel)
    kp = kn = sp = sn = 0
    tmp = f"/tmp/surv_{i:05d}.txt.gz"
    import gzip as _gz
    with _gz.open(tmp, "wt", encoding="utf-8") as out:
        for label, tmpl, is_pos in ((LABEL_USEFUL, USEFUL_TMPL, True), (LABEL_NO_USEFUL, NOUSE_TMPL, False)):
            for html in iter_html(tmpl.format(base=_BASE, i=i)):
                text = to_fasttext_text(html[:1_000_000], "body_strip")  # cap pathological multi-MB markup
                if predict_useful_prob(_MODEL, text) >= _THR:
                    out.write(f"{label} {text}\n")
                    if is_pos: kp += 1
                    else: kn += 1
                if is_pos: sp += 1
                else: sn += 1
    with open(tmp, "rb") as src, fsspec.open(part, "wb") as dst:
        dst.write(src.read())
    os.remove(tmp)
    return i, kp, kn, sp, sn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-base", default="gs://marin-us-central2/datasets/high_quality_3000_distill")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.0121)
    ap.add_argument("--shard-start", type=int, default=300)
    ap.add_argument("--shard-end", type=int, default=450)
    ap.add_argument(
        "--exclude-shards",
        default="",
        help="Comma-separated shard indices to SKIP (leakage guard). For the time-sorted pool the "
        "default --shard-start>=300 already excludes the val/test WARCs (which live in shards 0-299). "
        "For a RANDOM-draw pool, shard index has no relation to the original split, so pass the exact "
        "indices whose WARC hash is in the frozen val+test set (computed by mapping the split-manifest "
        "held-out hashes onto the random pool's sorted-hash order). Index-range disjointness does NOT hold.",
    )
    ap.add_argument("--target-survivors", type=int, default=2_460_000)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    log.info("downloading stage-1 model %s ...", args.model)
    local = "/tmp/stage1.bin"
    with fsspec.open(args.model, "rb") as src, open(local, "wb") as dst:
        dst.write(src.read())
    parts = args.out.rstrip("/")  # GCS dir for per-shard part files (ground-truth progress)
    excluded = {int(x) for x in args.exclude_shards.split(",") if x.strip()}
    shards = [i for i in range(args.shard_start, args.shard_end) if i not in excluded]
    if excluded:
        log.info("EXCLUDING %d shards (val/test leakage guard): %s", len(excluded), sorted(excluded))
    log.info("scoring %d shards with %d spawn-workers, threshold %.4f -> parts %s",
             len(shards), args.workers, args.threshold, parts)

    kp = kn = sp = sn = 0
    done = 0
    ctx = mp.get_context("spawn")  # fresh processes -> no inherited gRPC/gcsfs fork breakage
    with ctx.Pool(args.workers, initializer=_init_worker,
                  initargs=(local, args.data_base, args.threshold, parts)) as pool:
        for i, a, b, c, d in pool.imap_unordered(score_shard, shards):
            done += 1
            if a < 0:
                log.info("shard %d already done (skipped)", i); continue
            kp += a; kn += b; sp += c; sn += d
            tot = kp + kn
            log.info("shard %d done (%d/%d) | survivors+=%d cum=%d pos=%d neg=%d surv-rate=%.1f%% useful-frac=%.1f%%",
                     i, done, len(shards), a + b, tot, kp, kn, 100 * tot / max(1, sp + sn), 100 * kp / max(1, tot))
            if tot >= args.target_survivors:
                log.info("hit target %d survivors at %d shards — stopping", args.target_survivors, done)
                pool.terminate(); break
    log.info("DONE this-run survivors=%d pos=%d neg=%d neg:pos=%.2f:1 useful-frac=%.1f%% -> parts in %s",
             kp + kn, kp, kn, kn / max(1, kp), 100 * kp / max(1, kp + kn), parts)


if __name__ == "__main__":
    main()
