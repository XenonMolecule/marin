# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fan out the resiliparse url-recovery passes on Iris.

Rebuilds the deleted post-decon ``{text, url}`` document tree for the 10,364-WARC
resiliparse corpus so the quality x domain grid can run over it. See
``.agents/projects/resiliparse_grid_url_recovery.md`` for why this is possible at
all and ``recover_tokenized_text.py`` for the decode rules.

Stage order. ``survivor-ids`` and ``hash-raw`` read disjoint inputs and can run
concurrently; everything after ``representatives`` is strictly sequential::

    survivor-ids     decode all 4,451 cache parts -> (row, content_id) per part
    hash-raw         (content_id, shard, line) for all 550M raw docs, bucketed by id
    bucket-survivors re-partition the survivor ids into the same id buckets
    representatives  per bucket, pick one raw doc per surviving id
    keeplists        invert those into one keep-list per raw shard
    materialize      write {text, url} by filtering raw shards to their keep-lists

Everything runs in **us-central2**, where both the raw extraction and the token
cache live, so no stage moves data across regions.

Jobs are pinned ``--preemptible``. Iris otherwise pins small CPU-only jobs to
non-preemptible workers, which in us-central2 means the *reserved* v4 TPU hosts —
the pool that was reclaimed mid-run on 2026-06-25. These passes are long and
restartable (every chunk has a done marker), so preemptible is both the polite
and the correct choice.

    python -m experiments.baseline_collection.launch_url_recovery survivor-ids --tasks 200
    python -m experiments.baseline_collection.launch_url_recovery hash-raw --tasks 200
    python -m experiments.baseline_collection.launch_url_recovery bucket-survivors --tasks 64
    python -m experiments.baseline_collection.launch_url_recovery representatives
    python -m experiments.baseline_collection.launch_url_recovery keeplists
    python -m experiments.baseline_collection.launch_url_recovery materialize --tasks 200
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

CLUSTER = "marin"
REGION = "us-central2"
PRIORITY = "batch"
# Kept low deliberately. Each `iris job run` opens its own SSH tunnel to the
# controller and uploads a ~15 MB workspace bundle; at 16 in flight the
# controller slowed until submissions hit the timeout and waves landed a third
# of their jobs. Fewer, larger jobs plus modest concurrency is the fix.
SUBMIT_PARALLELISM = 6
# Per-submit ceiling. Short on purpose: a hung `iris job run` should free its pool
# slot quickly so the rest of the wave keeps landing, and the stage is re-runnable.
SUBMIT_TIMEOUT = 300
NUM_BUCKETS = 64

BASE = "gs://marin-us-central2/metadata/resiliparse_url_recovery"
SURVIVOR_IDS = f"{BASE}/survivor_ids"
SURVIVOR_BUCKETS = f"{BASE}/survivor_buckets"
RAW_INDEX = f"{BASE}/raw_index"
REPRESENTATIVES = f"{BASE}/representatives"
KEEPLISTS = f"{BASE}/keeplists"

# Deliberately NOT the original `baseline_resiliparse_decon_deduped` path. This
# tree is the same document *set* as the one that was deleted, but it carries url
# and its shard layout follows the raw extraction rather than the original dedup
# output, so reusing the old path would misrepresent it as the original.
DOCUMENTS = "gs://marin-us-central2/documents/baseline_resiliparse_decon_deduped_urls/10364warcs/deduped"
NUM_RAW_SHARDS = 10364
# Parts in the token cache ledger; bucket-survivors refuses to run until all exist.
NUM_CACHE_PARTS = 4451
# train/.stats.json total_elements for resiliparse_decon_10364warcs-beaaf5. Every
# cache row must resolve to exactly one raw document, so this is the number the
# rebuilt corpus has to hit exactly.
EXPECTED_DOCS = 290_262_075
# Known, characterised shortfall: ~0.29% of survivors are long documents whose
# stored tokens carry chunk-boundary corruption, so they decode to text that never
# existed in raw. Recovered separately from the cache. Bound kept just above the
# measured value so any NEW loss still fails the keeplists guard.
ALLOWED_UNMATCHED = 900_000

CACHE = "gs://marin-us-central2/tokenized/resiliparse_decon_10364warcs-beaaf5/train"
RAW_GLOB = "gs://marin-us-central2/extracted/dclm_400m_1x_10k_resiliparse-f0887f/*.jsonl.gz"

MODULE = "experiments.baseline_collection.recover_tokenized_text"


def _sfx(wave: str) -> str:
    """Job names must be unique cluster-wide, so a re-run with a different task
    count needs its own namespace — otherwise it collides with the previous wave
    and every colliding chunk is silently skipped even though its partition of
    the work has changed."""
    return f"-{wave}" if wave else ""


def _submit(name: str, command: list[str], *, cpu: float, memory: str, priority: str) -> tuple[str, bool]:
    """Submit one Iris job. Returns ``(name, ok)``; a rejected submit must not abort the wave."""
    argv = [
        "uv", "run", "iris", "--cluster", CLUSTER, "job", "run",
        "--region", REGION,
        "--priority", priority,
        "--preemptible",
        "--no-wait",
        "--job-name", name,
        "--cpu", str(cpu),
        "--memory", memory,
        "--disk", "20GB",
        "--extra", "cpu",
        "--enable-extra-resources",
        "--max-retries", "2",
    ]  # fmt: skip
    token = os.environ.get("HF_TOKEN")
    if token:
        argv += ["-e", "HF_TOKEN", token]
    argv += ["--", *command]

    # A submit that hangs must not take the wave down with it. `subprocess.run`
    # raises TimeoutExpired rather than returning, and an uncaught raise inside a
    # pool.map worker aborts every remaining submission — that is exactly how a
    # 200-job wave previously landed only 93 jobs. Never let the exception text
    # reach the log either: it embeds argv, which carries HF_TOKEN.
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=SUBMIT_TIMEOUT)
    except subprocess.TimeoutExpired:
        logger.warning("submit TIMEOUT %s after %ds — re-run this stage to retry", name, SUBMIT_TIMEOUT)
        return name, False
    except OSError as exc:
        logger.warning("submit ERROR %s: %s", name, type(exc).__name__)
        return name, False

    ok = result.returncode == 0
    if not ok:
        detail = (result.stderr or result.stdout)[-300:].strip()
        if "already exists" in detail:
            # Re-running a stage to fill gaps is the recovery path, so colliding
            # with a job that already landed is expected, not a failure.
            logger.info("submit SKIP %s (already submitted)", name)
            return name, True
        logger.warning("submit FAILED %s: %s", name, detail)
    return name, ok


def _wave(jobs: list[tuple[str, list[str]]], *, cpu: float, memory: str, priority: str) -> None:
    logger.info("submitting %d jobs at priority=%s ...", len(jobs), priority)
    with ThreadPoolExecutor(max_workers=SUBMIT_PARALLELISM) as pool:
        results = list(pool.map(lambda i: _submit(i[0], i[1], cpu=cpu, memory=memory, priority=priority), jobs))
    landed = sum(ok for _, ok in results)
    logger.info("submitted %d/%d", landed, len(jobs))
    if landed < len(jobs):
        logger.warning("%d submissions failed — re-run this command to retry them", len(jobs) - landed)


def _groups(tasks: int, jobs: int) -> list[list[int]]:
    """Deal chunk indices round-robin across ``jobs`` Iris jobs.

    Submitting is the bottleneck, not compute: every `iris job run` opens its own
    SSH tunnel and pushes a ~15 MB workspace bundle, so a few hundred submissions
    saturate the controller and start timing out. Chunk COUNT still sets the
    output partitioning and must not change between runs; this only changes how
    many chunks each job walks through.
    """
    jobs = max(1, min(jobs, tasks))
    return [[c for c in range(tasks) if c % jobs == g] for g in range(jobs)]


def run_survivor_ids(tasks: int, priority: str, wave: str = "", jobs: int = 40, greedy: bool = False) -> None:
    submissions = [
        (
            f"rp-survivor-ids{_sfx(wave)}-{g:04d}",
            ["python", "-m", MODULE, "survivor-ids",
             "--cache-path", CACHE, "--output-dir", SURVIVOR_IDS,
             "--num-chunks", str(tasks), "--chunk-idx", *[str(c) for c in group],
             *(["--greedy"] if greedy else [])],
        )
        for g, group in enumerate(_groups(tasks, jobs))
    ]  # fmt: skip
    _wave(submissions, cpu=1, memory="8GB", priority=priority)


def run_bucket_survivors(tasks: int, priority: str, wave: str = "") -> None:
    jobs = [
        (
            f"rp-bucket-surv{_sfx(wave)}-{i:04d}",
            [
                "python",
                "-m",
                MODULE,
                "bucket-survivors",
                "--survivor-dir",
                SURVIVOR_IDS,
                "--output-dir",
                SURVIVOR_BUCKETS,
                "--num-chunks",
                str(tasks),
                "--chunk-idx",
                str(i),
                "--num-buckets",
                str(NUM_BUCKETS),
                "--expect-parts",
                str(NUM_CACHE_PARTS),
            ],
        )
        for i in range(tasks)
    ]
    _wave(jobs, cpu=1, memory="12GB", priority=priority)


def run_hash_raw(tasks: int, priority: str, wave: str = "", jobs: int = 40, greedy: bool = False) -> None:
    submissions = [
        (
            f"rp-hash-raw{_sfx(wave)}-{g:04d}",
            ["python", "-m", MODULE, "hash-raw",
             "--raw-glob", RAW_GLOB, "--output-dir", RAW_INDEX,
             "--num-chunks", str(tasks), "--num-buckets", str(NUM_BUCKETS),
             "--chunk-idx", *[str(c) for c in group],
             *(["--greedy"] if greedy else [])],
        )
        for g, group in enumerate(_groups(tasks, jobs))
    ]  # fmt: skip
    _wave(submissions, cpu=1, memory="12GB", priority=priority)


def run_representatives(priority: str, wave: str = "") -> None:
    jobs = [
        (
            f"rp-representatives{_sfx(wave)}-{b:04d}",
            [
                "python",
                "-m",
                MODULE,
                "select-representatives",
                "--survivor-dir",
                SURVIVOR_BUCKETS,
                "--raw-index-dir",
                RAW_INDEX,
                "--output-dir",
                REPRESENTATIVES,
                "--num-buckets",
                str(NUM_BUCKETS),
                "--bucket-idx",
                str(b),
            ],
        )
        for b in range(NUM_BUCKETS)
    ]
    _wave(jobs, cpu=2, memory="24GB", priority=priority)


def run_keeplists(priority: str, wave: str = "") -> None:
    """Single job: inverting ~2.3 GB of (shard, line) in one place beats writing fragments."""
    jobs = [
        (
            f"rp-keeplists{_sfx(wave)}",
            ["python", "-m", MODULE, "shard-keeplists",
             "--representatives-dir", REPRESENTATIVES, "--output-dir", KEEPLISTS,
             "--num-shards", str(NUM_RAW_SHARDS), "--expect-total", str(EXPECTED_DOCS),
             "--allow-missing", str(ALLOWED_UNMATCHED)],
        )
    ]  # fmt: skip
    _wave(jobs, cpu=2, memory="32GB", priority=priority)


def run_materialize(tasks: int, priority: str, wave: str = "", jobs: int = 40, greedy: bool = False) -> None:
    submissions = [
        (
            f"rp-materialize{_sfx(wave)}-{g:04d}",
            ["python", "-m", MODULE, "materialize",
             "--raw-glob", RAW_GLOB, "--keeplist-dir", KEEPLISTS, "--output-dir", DOCUMENTS,
             "--num-chunks", str(tasks), "--chunk-idx", *[str(c) for c in group],
             *(["--greedy"] if greedy else [])],
        )
        for g, group in enumerate(_groups(tasks, jobs))
    ]  # fmt: skip
    _wave(submissions, cpu=1, memory="8GB", priority=priority)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["survivor-ids", "bucket-survivors", "hash-raw", "representatives", "keeplists", "materialize"],
    )
    parser.add_argument("--tasks", type=int, default=200, help="chunk count; sets the output partitioning")
    parser.add_argument("--jobs", type=int, default=40, help="Iris jobs to spread those chunks over")
    parser.add_argument("--wave", default="", help="suffix for job names; required when changing --tasks")
    parser.add_argument("--priority", default=PRIORITY, choices=["interactive", "batch"])
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="let each job drain every outstanding unit rather than only its own chunks; "
        "use when only part of a wave lands, since submitting costs more than running",
    )
    args = parser.parse_args()

    if args.stage == "survivor-ids":
        run_survivor_ids(args.tasks, args.priority, args.wave, args.jobs, args.greedy)
    elif args.stage == "bucket-survivors":
        run_bucket_survivors(args.tasks, args.priority, args.wave)
    elif args.stage == "hash-raw":
        run_hash_raw(args.tasks, args.priority, args.wave, args.jobs, args.greedy)
    elif args.stage == "representatives":
        run_representatives(args.priority, args.wave)
    elif args.stage == "keeplists":
        run_keeplists(args.priority, args.wave)
    else:
        run_materialize(args.tasks, args.priority, args.wave, args.jobs, args.greedy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
