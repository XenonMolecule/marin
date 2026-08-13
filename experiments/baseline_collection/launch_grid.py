# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adaptive fan-out for the quality x domain grid: mirror and label jobs on Iris.

Deliberately adaptive rather than block-scheduled. The interactive band is
contested and its width fluctuates, so this does not try to reserve a fixed
allocation — it submits a wave, reports how much of it actually landed, and is
safe to re-run. Re-running is the recovery mechanism for preemption too: every
shard carries a per-stage done marker, so a relaunched job skips finished work
and picks up wherever its predecessor was killed.

Jobs go in at **interactive** priority. This is load-bearing: the extraction
fleet that occupies most of the cluster runs its TPU children at ``batch``, the
lowest band, so an interactive job preempts them within a scheduler cycle. At
``batch`` we would queue behind that fleet indefinitely instead.

    # mirror the three us-central2 corpora into the compute region (CPU)
    python -m experiments.baseline_collection.launch_grid mirror --dataset dclm_10k

    # fan out topic labelling (TPU)
    python -m experiments.baseline_collection.launch_grid label \\
        --dataset high_quality_10k --stages topic --tasks 24

    # what landed?
    python -m experiments.baseline_collection.launch_grid status
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

from experiments.baseline_collection.grid_corpora import COMPUTE_REGION, GRID_CORPORA

logger = logging.getLogger(__name__)

CLUSTER = "marin"
# ``v5p-8`` is the only v5p shape that schedules as ONE task, and that is why it
# is the default despite being the smallest pool.
#
# Measured in us-central1 on 2026-07-29. Iris derives replicas as chips/8, so
# v5p-8 -> 1 task, v5p-16 -> 2, v5p-32 -> 4, and anything above one task is
# gang-scheduled. That distinction dominates everything else:
#
#   * v5p-8 (1 task):   ASSIGNED in under 30 s, every time.
#   * v5p-32 (4 tasks): 32 tasks sat PENDING for 12 min and never placed, despite
#                       116 preemptible hosts in the group — a gang needs N hosts
#                       free simultaneously in one topology, which preempting
#                       scattered batch workers does not produce.
#
# So the larger groups' headroom is not actually reachable by preemption, and
# chasing it costs wall-clock rather than saving it. A second reason to prefer one
# task per job: replicas only partition their shards correctly if they form a
# real JAX process group, and if they do not, every replica scores the SAME shards.
LABEL_TPU = "v5p-8"
LABEL_MEMORY = "64GB"
MIRROR_CPU = 8
MIRROR_MEMORY = "16GB"
# CPU workers report 8 vCPU / ~15 GB despite the e2-highmem-2 name; request the
# whole box so the HF fast tokenizer can use every core.
LABEL_CPU_CORES = 8
LABEL_CPU_MEMORY = "12GB"
# Default for TPU-bound work (topic), which must preempt the extraction fleet's
# `batch` children to get accelerators at all. Quality is CPU-bound and NOT
# urgent, so it should run at `batch` instead: Iris will happily place a
# CPU-only task on a TPU host that has spare cores, so an interactive quality
# wave silently evicts extraction from accelerators it actually needs.
PRIORITY = "interactive"
SUBMIT_PARALLELISM = 12
# Per-submit ceiling. Each `iris job run` opens its own SSH tunnel and uploads a
# ~15 MB workspace bundle, so a hung submit must free its pool slot quickly
# rather than stalling the wave behind it. Stages are re-runnable via done markers.
SUBMIT_TIMEOUT = 300
# Preemptible capacity here is reclaimed by other users regularly; 0 retries means
# one preemption is fatal to a job. Done markers make retries idempotent.
MAX_RETRIES = 3


def _submit(
    name: str,
    command: list[str],
    *,
    tpu: str | None,
    cpu: float,
    memory: str,
    extra: str,
    priority: str,
    region: str = COMPUTE_REGION,
) -> tuple[str, bool]:
    """Submit one Iris job. Returns ``(name, ok)``; never raises on a failed submit.

    A single rejected submission must not abort the wave — partial capacity is
    the expected case, not an error.
    """
    argv = [
        "uv", "run", "iris", "--cluster", CLUSTER, "job", "run",
        "--region", region,
        "--priority", priority,
        "--no-wait",
        "--job-name", name,
        "--cpu", str(cpu),
        "--memory", memory,
        "--extra", extra,
        # Required for accelerators AND for any request >=4 GB RAM or >=10 GB disk.
        # The mirror jobs are CPU-only but ask for 16 GB, so this is not
        # TPU-conditional — omitting it rejects the whole mirror wave at submit.
        "--enable-extra-resources",
        # Iris defaults to 0 retries, so a SINGLE preemption kills a job for good.
        # Everything here runs on preemptible capacity that other users routinely
        # reclaim, so without this a wave silently thins out over hours and the
        # stage appears to slow down for no visible reason. Retries are free
        # correctness-wise: every shard carries a done marker, so a restarted job
        # resumes rather than redoing work.
        "--max-retries", str(MAX_RETRIES),
    ]  # fmt: skip
    if tpu:
        argv += ["--tpu", tpu]
    token = os.environ.get("HF_TOKEN")
    if token:
        argv += ["-e", "HF_TOKEN", token]
    argv += ["--", *command]

    # `subprocess.run` RAISES on timeout rather than returning non-zero, and an
    # uncaught raise inside a pool.map worker aborts every remaining submission —
    # a whole wave silently lands a fraction of its jobs. Never log the exception
    # text either: it embeds argv, which carries HF_TOKEN.
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=SUBMIT_TIMEOUT)
    except subprocess.TimeoutExpired:
        logger.warning("submit TIMEOUT %s after %ds — re-run to retry", name, SUBMIT_TIMEOUT)
        return name, False
    except OSError as exc:
        logger.warning("submit ERROR %s: %s", name, type(exc).__name__)
        return name, False

    ok = result.returncode == 0
    if not ok:
        logger.warning("submit FAILED %s: %s", name, (result.stderr or result.stdout)[-300:].strip())
    return name, ok


def _wave(
    jobs: list[tuple[str, list[str]]],
    *,
    tpu: str | None,
    cpu: float,
    memory: str,
    extra: str,
    priority: str,
    region: str = COMPUTE_REGION,
) -> None:
    """Submit a wave in parallel and report how much of it landed."""
    logger.info("submitting %d jobs at priority=%s region=%s ...", len(jobs), priority, region)
    with ThreadPoolExecutor(max_workers=SUBMIT_PARALLELISM) as pool:
        results = list(
            pool.map(
                lambda item: _submit(
                    item[0],
                    item[1],
                    tpu=tpu,
                    cpu=cpu,
                    memory=memory,
                    extra=extra,
                    priority=priority,
                    region=region,
                ),
                jobs,
            )
        )
    landed = sum(ok for _, ok in results)
    logger.info("submitted %d/%d", landed, len(jobs))
    if landed < len(jobs):
        logger.warning("%d submissions failed — re-run this command to retry them", len(jobs) - landed)


def run_mirror(dataset: str, tasks: int, priority: str = PRIORITY) -> None:
    """Fan out the mirror stage. CPU only; no accelerator, no HF access needed."""
    corpus = GRID_CORPORA[dataset]
    if corpus.region == COMPUTE_REGION:
        logger.info("%s already lives in %s — no mirror needed", dataset, COMPUTE_REGION)
        return
    jobs = []
    for idx in range(tasks):
        command = [
            "python", "-m", "experiments.baseline_collection.grid_mirror",
            "--dataset", dataset,
            "--num-chunks", str(tasks),
            "--chunk-idx", str(idx),
        ]  # fmt: skip
        jobs.append((f"grid-mirror-{dataset.replace('_', '-')}-{idx:03d}", command))
    _wave(jobs, tpu=None, cpu=MIRROR_CPU, memory=MIRROR_MEMORY, extra="cpu", priority=priority)


def run_label(
    dataset: str,
    stages: str,
    tasks: int,
    source: str,
    quality_model: str | None,
    tpu: str,
    wave: str = "",
    device: str = "tpu",
    priority: str = PRIORITY,
    region: str = COMPUTE_REGION,
    output_base: str | None = None,
) -> None:
    """Fan out the labelling stage across ``tasks`` independent single-host jobs.

    ``wave`` distinguishes a top-up from the original fan-out. Iris job names must
    be unique, so re-running with a larger ``--tasks`` would otherwise collide with
    the first wave's names. Waves are safe to overlap even though they partition
    the shard list differently: done markers, not the partitioning, decide what
    still needs doing, so a second wave simply finds less work and spreads it wider.

    ``region``/``output_base`` exist for a corpus that does not live in
    ``COMPUTE_REGION``. The five original corpora were all labelled into
    us-central1, but ``COMPUTE_REGION`` is a module constant that ``OUTPUT_BASE``
    and ``MIRROR_BASE`` both derive from, so flipping it globally would relocate
    where THEIR results are looked up too. Overriding per run is the safe form.
    """
    jobs = []
    suffix = f"-{wave}" if wave else ""
    # The quality model is 3M params at 512 tokens and its cost is dominated by
    # tokenization on the host, not by matmul — a TPU task measured only ~2.3x a
    # single laptop core. TPU capacity is also fixed at whatever preemption
    # yields, while the CPU pool autoscales (observed growing 5 -> 19 hosts), so
    # for a quality-only wave CPU is both the better fit and the larger pool.
    on_cpu = device == "cpu"
    for idx in range(tasks):
        command = [
            "python", "-m", "experiments.baseline_collection.grid_label", "label",
            "--dataset", dataset,
            "--stages", stages,
            "--source", source,
            "--num-chunks", str(tasks),
            "--chunk-idx", str(idx),
        ]  # fmt: skip
        if quality_model:
            command += ["--quality-model", quality_model]
        if output_base:
            command += ["--output-base", output_base]
        jobs.append((f"grid-{dataset.replace('_', '-')}{suffix}-{idx:03d}", command))
    if on_cpu:
        _wave(
            jobs,
            tpu=None,
            cpu=LABEL_CPU_CORES,
            memory=LABEL_CPU_MEMORY,
            extra="cpu",
            priority=priority,
            region=region,
        )
        return
    _wave(jobs, tpu=tpu, cpu=0.1, memory=LABEL_MEMORY, extra="tpu", priority=priority, region=region)


def run_status() -> None:
    """Print how many grid tasks are pending/running/done, by job prefix."""
    query = """
        SELECT SUBSTR(j.name, 1, 40) AS job, t.state, COUNT(*) AS n
        FROM tasks t JOIN jobs j ON j.job_id = t.job_id
        WHERE j.name LIKE '/michaelryan/grid-%'
        GROUP BY 1, 2 ORDER BY 1, 2
    """
    subprocess.run(["uv", "run", "iris", "--cluster", CLUSTER, "query", query], timeout=600, check=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["mirror", "label", "status"])
    parser.add_argument("--dataset", choices=list(GRID_CORPORA))
    parser.add_argument("--stages", default="topic")
    parser.add_argument("--tasks", type=int, default=24)
    parser.add_argument("--source", choices=["native", "mirror"], default="mirror")
    parser.add_argument("--quality-model", default=None)
    parser.add_argument(
        "--tpu",
        default=LABEL_TPU,
        help="TPU scale group to target. Bigger groups usually hold more preemptible headroom.",
    )
    parser.add_argument(
        "--wave",
        default="",
        help="Tag distinguishing a top-up wave from the first fan-out, e.g. --wave w2.",
    )
    parser.add_argument(
        "--device",
        choices=["tpu", "cpu"],
        default="tpu",
        help="Where to run. Quality-only waves belong on cpu: the stage is tokenizer-bound "
        "and the CPU pool autoscales, whereas TPU capacity does not.",
    )
    parser.add_argument(
        "--priority",
        choices=["batch", "interactive"],
        default=PRIORITY,
        help="Use batch for CPU-bound work so it never evicts accelerator jobs that need TPUs.",
    )
    parser.add_argument(
        "--region",
        default=COMPUTE_REGION,
        help="Iris region to schedule in. Override for a corpus that does not live in COMPUTE_REGION.",
    )
    parser.add_argument(
        "--output-base",
        default=None,
        help="Bucket root for this run's grid outputs, e.g. gs://marin-us-central2. Overrides OUTPUT_BASE "
        "for this dataset only; do NOT flip COMPUTE_REGION, which would relocate the other corpora too.",
    )
    args = parser.parse_args()

    if args.command == "status":
        run_status()
        return
    if not args.dataset:
        raise ValueError(f"--dataset is required for `{args.command}`")
    if args.command == "mirror":
        run_mirror(args.dataset, args.tasks, args.priority)
        return
    # Keyword args from `priority` on: it was previously dropped on the floor here,
    # so `--priority batch` silently ran label waves at `interactive` — the exact
    # thing the quality stage must not do, since a CPU task at interactive evicts
    # extraction from accelerators it actually needs.
    run_label(
        args.dataset,
        args.stages,
        args.tasks,
        args.source,
        args.quality_model,
        args.tpu,
        args.wave,
        args.device,
        priority=args.priority,
        region=args.region,
        output_base=args.output_base,
    )


if __name__ == "__main__":
    main()
