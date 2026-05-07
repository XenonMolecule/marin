"""Pre-stage llm_curated .tokens.gz sidecars from us-central1 to us-central2.

Runs on a us-central2 worker (e.g., marin-big-run Ray cluster). Reads an
explicit canonical-paths manifest, performs server-side blob copy for each
listed sidecar to a us-central2 destination bucket, and writes a new
manifest with the rewritten destination paths. Idempotent (skips files
already copied), so it's safe to re-run after a partial failure.

SAFETY: This script only copies the *exact* paths listed in the manifest.
It never globs, never copies directories. The manifest contains only
.tokens.gz sidecars (~4 KB each, ~621 MB total for 145,694 files), never
the much larger batch_*.jsonl.gz files.
"""

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec
from google.cloud import storage

logger = logging.getLogger(__name__)

# Source and destination prefixes.
SRC_PREFIX = "gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/by_region/"
DST_BUCKET_NAME = "marin-us-central2"
DST_PREFIX_REL = "scratch/llm_curated_flop_sidecars/by_region/"
DST_PREFIX = f"gs://{DST_BUCKET_NAME}/{DST_PREFIX_REL}"

EXPECTED_SUFFIX = ".tokens.gz"


def rewrite_path(src: str) -> str:
    assert src.startswith(SRC_PREFIX), f"unexpected source prefix: {src}"
    assert src.endswith(EXPECTED_SUFFIX), f"unexpected suffix (refusing to copy): {src}"
    rel = src[len(SRC_PREFIX):]
    return f"{DST_PREFIX}{rel}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="GCS or local path with one source path per line")
    ap.add_argument("--out-manifest", required=True, help="GCS or local path to write rewritten destination manifest")
    ap.add_argument("--max-workers", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load manifest.
    with fsspec.open(args.manifest, "r") as f:
        sources = [ln.strip() for ln in f if ln.strip()]
    if args.limit:
        sources = sources[: args.limit]
    logger.info("Loaded %d source paths from manifest", len(sources))

    # Validate all paths up front (no surprises mid-run).
    for s in sources:
        rewrite_path(s)
    logger.info("All source paths validated (correct prefix + .tokens.gz suffix)")

    # GCS clients — server-side copy (data flows GCS-to-GCS, not through this VM,
    # but billing is still on the inter-region transfer).
    client = storage.Client()
    src_bucket = client.bucket("marin-us-central1")
    dst_bucket = client.bucket(DST_BUCKET_NAME)

    src_blob_prefix = "documents/baseline_llm_extraction_consolidated/by_region/"

    def copy_one(src_path: str) -> tuple[str, str]:
        rel = src_path[len(SRC_PREFIX):]
        src_blob_name = src_blob_prefix + rel
        dst_blob_name = DST_PREFIX_REL + rel

        dst_blob = dst_bucket.blob(dst_blob_name)
        if dst_blob.exists():
            return src_path, "skip"

        src_blob = src_bucket.blob(src_blob_name)
        # Server-side copy: GCS handles the bytes; we don't proxy.
        src_bucket.copy_blob(src_blob, dst_bucket, dst_blob_name)
        return src_path, "ok"

    t0 = time.monotonic()
    counts = {"ok": 0, "skip": 0, "error": 0}
    errors: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = {pool.submit(copy_one, s): s for s in sources}
        for i, fut in enumerate(as_completed(futs), start=1):
            src = futs[fut]
            try:
                _, status = fut.result()
                counts[status] += 1
            except Exception as e:
                counts["error"] += 1
                errors.append((src, str(e)))
            if i % 5000 == 0:
                elapsed = time.monotonic() - t0
                rate = i / elapsed
                eta = (len(sources) - i) / max(rate, 1e-6)
                logger.info(
                    "Progress: %d/%d (ok=%d skip=%d err=%d, %.0f files/s, ETA %.0fs)",
                    i, len(sources), counts["ok"], counts["skip"], counts["error"], rate, eta,
                )

    logger.info(
        "Done in %.1fs: ok=%d skip=%d err=%d",
        time.monotonic() - t0, counts["ok"], counts["skip"], counts["error"],
    )

    if errors:
        logger.error("First 5 errors:")
        for src, err in errors[:5]:
            logger.error("  %s -> %s", src, err)

    # Write rewritten manifest.
    rewritten = [rewrite_path(s) for s in sources]
    with fsspec.open(args.out_manifest, "w") as f:
        f.write("\n".join(rewritten) + "\n")
    logger.info("Wrote rewritten manifest with %d paths to %s", len(rewritten), args.out_manifest)


if __name__ == "__main__":
    main()
