# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finalize an ExpC 10k method (dclm_10k or nemotron_10k) and launch the sweep.

The expected operator workflow once tokenization completes:

    # 1. Check status:
    uv run python experiments/scaling_law_sweeps/finalize_10k_and_launch.py --status

    # 2. (For dclm_10k or nemotron_10k only) mirror the cache to every Marin
    #    training region. NEVER mirror resiliparse or llm_curated_bos_fixed —
    #    those stay pinned to us-central1.
    uv run python experiments/scaling_law_sweeps/finalize_10k_and_launch.py \\
        --method nemotron_10k --mirror

    # 3. Patch curation_plan.py (cache_hash + d_obs) and launch the sweep:
    uv run python experiments/scaling_law_sweeps/finalize_10k_and_launch.py \\
        --method nemotron_10k --launch

Mirror and launch are decoupled because the mirror takes minutes-to-tens-of-minutes
and is non-fatal to retry, whereas patch+launch is seconds and atomic. After
mirroring, --launch will:
  1. Read `total_tokens` from `train/.stats.json`.
  2. Patch `curation_plan.py`: real cache_hash in `_D_OBS_DEFAULTS` + the
     matching `_method(...)` call in `METHODS`.
  3. Run unit tests as a gate.
  4. Submit an Iris parent at interactive priority running launch_curation_sweep
     with `--child-priority batch`.

Idempotent: re-running after success is a no-op for both --mirror (skips
destinations that already have `train/.stats.json`) and --launch (cache_hash
no longer matches the placeholder so the patch is a no-op).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Where the 10k tokenization pipeline writes outputs.
_KNOWN_10K_TOKENIZED_PATHS: dict[str, str] = {
    "dclm_10k": "gs://marin-us-central2/tokenized/dclm_400m_1x_10k_dclm-3df0ba/",
    "nemotron_10k": "gs://marin-us-central2/tokenized/dclm_400m_1x_10k_nemotron_full-3dcb75/",
}
# The cache directory NAME (last segment) is what curation_plan stores as
# cache_hash. Use the value's path stem (drop the gs://...prefix and trailing /).
_CACHE_HASH_FOR_METHOD: dict[str, str] = {
    method: path.rstrip("/").split("/")[-1] for method, path in _KNOWN_10K_TOKENIZED_PATHS.items()
}
_PLACEHOLDER_HASH_FOR_METHOD: dict[str, str] = {
    "dclm_10k": "TBD_dclm_10k_HASH_PENDING",
    "nemotron_10k": "TBD_nemotron_10k_HASH_PENDING",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
CURATION_PLAN_PATH = REPO_ROOT / "experiments" / "scaling_law_sweeps" / "curation_plan.py"


def _gcs_exists(path: str) -> bool:
    r = subprocess.run(["gcloud", "storage", "ls", path], capture_output=True, text=True)
    return r.returncode == 0 and path in r.stdout


def _gcs_read_json(path: str) -> dict:
    r = subprocess.run(["gcloud", "storage", "cat", path], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def _all_regions() -> dict[str, str]:
    """Region → bucket map (e.g. 'us-east1' → 'gs://marin-us-east1')."""
    from experiments.scaling_law_sweeps.region_tracker import REGION_TO_BUCKET

    return dict(REGION_TO_BUCKET)


def _mirror_cache_to_all_regions(method: str) -> bool:
    """Cross-region copy the 10k tokenized cache from us-central2 to every other
    Marin training region. Skips destinations that already have a `train/.stats.json`
    (idempotent — safe to re-run after partial completion).

    The user explicitly authorized cross-region egress for dclm_10k and nemotron_10k
    (NEVER for resiliparse / llm_curated, those stay pinned to us-central1).

    Returns True on full success (every destination region has .stats.json after
    we're done), False otherwise.
    """
    if method not in ("dclm_10k", "nemotron_10k"):
        logger.error("REFUSING to mirror %s — only dclm_10k/nemotron_10k are authorized.", method)
        return False

    source = _KNOWN_10K_TOKENIZED_PATHS[method].rstrip("/") + "/"
    SOURCE_BUCKET = "gs://marin-us-central2"
    if not source.startswith(SOURCE_BUCKET + "/"):
        logger.error("Source %r is not in us-central2; refusing to mirror.", source)
        return False
    rel_path = source[len(SOURCE_BUCKET) + 1 :]  # e.g. "tokenized/dclm_400m_1x_10k_..-XXX/"

    regions = _all_regions()
    failures: list[str] = []
    for region, bucket in regions.items():
        if region == "us-central2":
            continue  # source
        dest = f"{bucket}/{rel_path}"
        dest_stats = f"{dest}train/.stats.json"
        if _gcs_exists(dest_stats):
            logger.info("[%s] already mirrored (stats.json present), skip", region)
            continue
        logger.info("[%s] mirroring %s → %s ...", region, source, dest)
        # Use `cp -r SRC/* DST/` to avoid the trailing-slash nesting trap
        # (per user-memory: cp -r src/dir/ dst/dir/ can create dst/dir/dir/).
        cmd = [
            "gcloud",
            "storage",
            "cp",
            "--recursive",
            f"{source}*",
            dest,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            logger.error("[%s] cp failed (rc=%d):\n%s", region, r.returncode, r.stderr)
            failures.append(region)
            continue
        # Verify the destination is intact
        if not _gcs_exists(dest_stats):
            logger.error("[%s] copy returned 0 but stats.json missing at dest — investigating", region)
            failures.append(region)
            continue
        logger.info("[%s] mirror complete", region)

    if failures:
        logger.error("Mirror INCOMPLETE: %s did not finish cleanly. Re-run --mirror to retry.", failures)
        return False
    logger.info("Mirror complete in all regions: %s", list(regions))
    return True


def _check_method_status(method: str) -> tuple[bool, str | None, int | None]:
    """Return (ready, cache_hash, d_obs_tokens). cache_hash is the directory name
    (e.g. 'dclm_400m_1x_10k_nemotron_full-3dcb75'); only meaningful when ready=True.
    """
    base = _KNOWN_10K_TOKENIZED_PATHS[method].rstrip("/")
    stats_path = f"{base}/train/.stats.json"
    if not _gcs_exists(stats_path):
        return False, None, None
    try:
        stats = _gcs_read_json(stats_path)
    except Exception as e:
        logger.warning("failed reading %s: %s", stats_path, e)
        return False, None, None
    d_obs = int(stats.get("total_tokens", 0))
    if d_obs <= 0:
        logger.warning("stats.json has total_tokens=%s — not yet finalized", d_obs)
        return False, None, None
    return True, _CACHE_HASH_FOR_METHOD[method], d_obs


def _patch_curation_plan(
    method: str,
    cache_hash: str,
    d_obs_tokens: int,
    pin_region: str | None = None,
) -> bool:
    """Edit curation_plan.py in-place to wire the real cache_hash + d_obs.

    If `pin_region` is provided, also injects `pin_region="..."` into the
    method's `_method(...)` call (e.g. when the cache only exists in one
    region and shouldn't float). Idempotent: re-running after success is a no-op.

    Returns True if a change was made; False if already finalized.
    """
    src = CURATION_PLAN_PATH.read_text()
    placeholder = _PLACEHOLDER_HASH_FOR_METHOD[method]

    if cache_hash in src and f'"{cache_hash}": {d_obs_tokens},' in src:
        logger.info("%s already finalized in curation_plan.py — skipping patch", method)
        return False

    # 1. Replace the placeholder dict entry with the real one.
    placeholder_line = f'    "{placeholder}": 0,'
    real_line = f'    "{cache_hash}": {d_obs_tokens},'
    new_src = src.replace(placeholder_line, real_line, 1)
    if new_src == src:
        logger.warning("Placeholder dict entry not found: %r", placeholder_line)
        return False

    # 2. Replace the cache_hash inside the corresponding `_method(...)` call,
    #    and optionally insert pin_region as an extra kwarg.
    method_call_pattern = re.compile(
        rf'("{method}":\s*_method\(\s*"{method}",\s*)"{re.escape(placeholder)}"',
        re.DOTALL,
    )
    after = method_call_pattern.sub(rf'\1"{cache_hash}"', new_src, count=1)
    if after == new_src:
        logger.warning("Could not patch _method() call for %s", method)
        return False
    new_src = after

    if pin_region:
        # Find the _method(...) closing paren that follows our cache_hash and
        # inject `pin_region=...` before it. Match through the existing
        # sampled_warcs= line (always present for ExpC 10k methods).
        kwarg_insert = re.compile(
            rf'("{method}":\s*_method\(\s*"{method}",\s*"{re.escape(cache_hash)}",\s*sampled_warcs=EXPC_SAMPLED_WARCS),(\s*\))',
            re.DOTALL,
        )
        after2 = kwarg_insert.sub(rf'\1,\n        pin_region="{pin_region}",\2', new_src, count=1)
        if after2 == new_src:
            logger.warning(
                "Could not insert pin_region for %s — leaving without pin (cache might fail to "
                "resolve in regions without it).",
                method,
            )
        else:
            new_src = after2
            logger.info("Injected pin_region=%r into _method('%s', ...)", pin_region, method)

    CURATION_PLAN_PATH.write_text(new_src)
    logger.info("Patched curation_plan.py: %s -> %s (%d tokens)", placeholder, cache_hash, d_obs_tokens)
    return True


def _run_unit_tests() -> bool:
    logger.info("Running curation plan + math tests...")
    r = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "tests/test_curation_plan.py",
            "tests/test_data_curation_math.py",
            "-q",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        logger.error("Tests FAILED. Stdout:\n%s\nStderr:\n%s", r.stdout, r.stderr)
        return False
    logger.info("Tests pass.")
    return True


def _dry_run_count(method: str) -> int | None:
    logger.info("Dry-run: counting ExpC plans for %s", method)
    r = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "experiments/scaling_law_sweeps/launch_curation_sweep.py",
            "--methods",
            method,
            "--experiments",
            "C",
            "--dry-run",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        logger.error("Dry-run FAILED:\n%s", r.stderr)
        return None
    m = re.search(r"TOTAL:\s*(\d+)\s*runs", r.stdout)
    if not m:
        logger.error("Dry-run output didn't contain a TOTAL line:\n%s", r.stdout)
        return None
    return int(m.group(1))


def _launch_iris_parent(method: str) -> str | None:
    job_name = f"curation-expc-coord-{method.replace('_10k', '10k')}"
    logger.info("Launching iris parent: %s", job_name)
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")
    if not wandb_api_key:
        raise RuntimeError("WANDB_API_KEY env var required.")
    if not hf_token:
        raise RuntimeError("HF_TOKEN env var required.")
    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "iris"),
        "--cluster",
        "marin",
        "job",
        "run",
        "--priority",
        "interactive",
        "--no-wait",
        "-e",
        "WANDB_API_KEY",
        wandb_api_key,
        "-e",
        "HF_TOKEN",
        hf_token,
        "--memory",
        "2GB",
        "--cpu",
        "2",
        "--job-name",
        job_name,
        "--",
        "python",
        "experiments/scaling_law_sweeps/launch_curation_sweep.py",
        "--methods",
        method,
        "--experiments",
        "C",
        "--child-priority",
        "batch",
    ]
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        logger.error("Iris submit failed:\n%s\n%s", r.stdout, r.stderr)
        return None
    # Last non-empty line of stdout is the job id (per iris convention).
    last_line = next((ln for ln in reversed(r.stdout.strip().splitlines()) if ln.strip()), "")
    logger.info("Iris parent submitted: %s", last_line)
    return last_line


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--method",
        choices=list(_KNOWN_10K_TOKENIZED_PATHS.keys()),
        help="Which method's 10k tokenization to finalize.",
    )
    parser.add_argument("--status", action="store_true", help="Print status of both 10k tokenizations and exit.")
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Cross-region mirror the cache from us-central2 to every other Marin "
        "training region. Idempotent. Required for dclm_10k and nemotron_10k since "
        "their caches are only produced in us-central2; the launcher floats children "
        "across regions and would fail to read the cache otherwise. Cross-region "
        "egress is real money — only authorized for these two methods.",
    )
    parser.add_argument("--launch", action="store_true", help="After patching, submit the iris parent.")
    parser.add_argument("--no-tests", action="store_true", help="Skip the unit-test gate (not recommended).")
    args = parser.parse_args()

    if args.status:
        for method in _KNOWN_10K_TOKENIZED_PATHS:
            ready, cache_hash, d_obs = _check_method_status(method)
            label = "READY" if ready else "PENDING"
            tok = f"{d_obs:,}" if d_obs is not None else "-"
            logger.info("%-15s %-9s hash=%s d_obs=%s", method, label, cache_hash or "-", tok)
        return 0

    if not args.method:
        parser.error("--method is required unless --status is passed.")

    method = args.method
    ready, cache_hash, d_obs = _check_method_status(method)
    if not ready:
        logger.error("%s is not ready: train/.stats.json missing or empty.", method)
        return 1
    logger.info("%s is READY: hash=%s d_obs=%d", method, cache_hash, d_obs)

    if args.mirror:
        ok = _mirror_cache_to_all_regions(method)
        if not ok:
            logger.error("Mirror failed. NOT proceeding to patch+launch — re-run with --mirror to retry.")
            return 1
        if not args.launch:
            logger.info("Mirror-only mode (no --launch). Done.")
            return 0

    patched = _patch_curation_plan(method, cache_hash, d_obs, pin_region=None)
    if patched and not args.no_tests:
        if not _run_unit_tests():
            logger.error("Tests failed after patching. NOT launching. Inspect curation_plan.py.")
            return 1

    n = _dry_run_count(method)
    if n is None:
        logger.error("Dry-run could not enumerate plans. NOT launching.")
        return 1
    logger.info("Dry-run produced %d ExpC plans for %s", n, method)

    if args.launch:
        job_id = _launch_iris_parent(method)
        if not job_id:
            logger.error("Iris launch failed.")
            return 1
        logger.info("DONE. Parent job: %s", job_id)
    else:
        logger.info("Patched but NOT launching (use --launch to submit).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
