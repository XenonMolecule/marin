# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch RTP eval on the 12 priority + 2 bonus checkpoints.

Each checkpoint runs in its native region to avoid cross-region reads.
TPU type is sized to the model: d512 → v6e-4/v5p-8, d1536 → v5p-8, d2432 → v5p-8 or v6e-8.

Invocation:
    python -m experiments.safety.launch_rtp_priority [--dry-run]
"""

import argparse
import dataclasses
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RtpRun:
    """One model to evaluate."""

    priority: str  # "priority" or "bonus"
    label: str  # human-readable shorthand, e.g. "hq_3000/2e+21-d1536"
    region: str  # GCS region of the checkpoint
    hf_checkpoint: str  # full gs:// URI to the HF step directory
    dim: int  # model dim (drives TPU sizing)


# Resolved from gs://marin-us-central1/metadata/data_curation_fixed_model_results/
# on 2026-05-23 via experiments/safety/find_rtp_checkpoints.py (kept in /tmp).
RUNS: list[RtpRun] = [
    # Priority (12)
    RtpRun(
        "priority",
        "hq_3000/2e+21-d1536",
        "us-east5",
        "gs://marin-us-east5/checkpoints/isoflop-curation/curation-high_quality_3000-expFM_natural-2e+21-d1536-L16-B2048/hf/step-35638",
        1536,
    ),
    RtpRun(
        "priority",
        "hq_3000/9e+20-d2432",
        "us-east5",
        "gs://marin-us-east5/checkpoints/isoflop-curation/curation-high_quality_3000-expFM_natural-9e+20-d2432-L24-B256/hf/step-46685",
        2432,
    ),
    RtpRun(
        "priority",
        "hq_3000/9e+19-d512",
        "us-east5",
        "gs://marin-us-east5/checkpoints/isoflop-curation/curation-high_quality_3000-expFM_natural-9e+19-d512-L6-B512/hf/step-61581",
        512,
    ),
    RtpRun(
        "priority",
        "dclm/3e+20-d2432",
        "us-east5",
        "gs://marin-us-east5/checkpoints/isoflop-curation/curation-dclm-expFM_natural-3e+20-d2432-L24-B64/hf/step-62248",
        2432,
    ),
    RtpRun(
        "priority",
        "dclm/9e+19-d1536",
        "us-east5",  # us-central1 capacity = 0 on 2026-05-23; ~2GB cross-region read.
        "gs://marin-us-central1/checkpoints/isoflop-curation/curation-dclm-expFM_natural-9e+19-d1536-L16-B64/hf/step-57021",
        1536,
    ),
    RtpRun(
        "priority",
        "dclm/9e+18-d512",
        "europe-west4",
        "gs://marin-eu-west4/checkpoints/isoflop-curation/curation-dclm-expFM_natural-9e+18-d512-L6-B64/hf/step-49265",
        512,
    ),
    RtpRun(
        "priority",
        "nemotron/3e+18-d512",
        "us-east5",  # route here: us-central1 capacity = 0 on 2026-05-23. Tiny 50M checkpoint cross-region read.
        "gs://marin-us-central1/checkpoints/isoflop-curation/curation-nemotron_full_bos_fixed-expFM_natural-3e+18-d512-L6-B32/hf/step-32843",
        512,
    ),
    RtpRun(
        "priority",
        "nemotron/9e+19-d1536",
        "us-east5",  # route here: us-central1 capacity = 0. ~2GB cross-region.
        "gs://marin-us-central1/checkpoints/isoflop-curation/curation-nemotron_full_bos_fixed-expFM_natural-9e+19-d1536-L16-B64/hf/step-57021",
        1536,
    ),
    RtpRun(
        "priority",
        "nemotron/2e+20-d2432",
        "us-east5",
        "gs://marin-us-east5/checkpoints/isoflop-curation/curation-nemotron_full_bos_fixed-expFM_natural-2e+20-d2432-L24-B64/hf/step-37348",
        2432,
    ),
    RtpRun(
        "priority",
        "resiliparse_dedup/3e+20-d2432",
        "us-east5",
        "gs://marin-us-east5/checkpoints/isoflop-curation/curation-resiliparse_dedup-expFM_natural-3e+20-d2432-L24-B64/hf/step-62248",
        2432,
    ),
    RtpRun(
        "priority",
        "resiliparse_dedup/2e+20-d1536",
        "us-east5",
        "gs://marin-us-east5/checkpoints/isoflop-curation/curation-resiliparse_dedup-expFM_natural-2e+20-d1536-L16-B128/hf/step-57021",
        1536,
    ),
    RtpRun(
        "priority",
        "resiliparse_dedup/2e+20-d512",
        "us-east5",
        "gs://marin-us-east5/checkpoints/isoflop-curation/curation-resiliparse_dedup-expFM_natural-2e+20-d512-L6-B1024/hf/step-61581",
        512,
    ),
    # Bonus (2)
    RtpRun(
        "bonus",
        "hq_3000/3e+20-d1536",
        "us-east5",
        "gs://marin-us-east5/checkpoints/isoflop-curation/curation-high_quality_3000-expFM_natural-3e+20-d1536-L16-B256/hf/step-47517",
        1536,
    ),
    RtpRun(
        "bonus",
        "nemotron/3e+20-d2432",
        "us-east5",
        "gs://marin-us-east5/checkpoints/isoflop-curation/curation-nemotron_full_bos_fixed-expFM_natural-3e+20-d2432-L24-B64/hf/step-62248",
        2432,
    ),
]

# Region → TPU type. Updated 2026-05-23 after v5p-8 us-east5 queue grew to 92+ workers;
# migrated us-east5 to v6e-8 us-east5-b (shorter queue, validated by smoke).
REGION_TPU = {
    "us-east5": "v6e-8",
    "us-central1": "v6e-8",  # us-central1 has no TPU; jobs scheduled into us-east5 v6e-8 anyway via region override
    "europe-west4": "v6e-8",  # v6e-8 europe-west4-a; validated by dclm/9e+18-d512 succeeding
}


# Safe labels for iris job names (slashes → dashes, + → x).
def safe_label(label: str) -> str:
    return label.replace("/", "-").replace("+", "x")


def submit(run: RtpRun, num_prompts: int, dry_run: bool, max_retries: int) -> str | None:
    job_name = f"rtp-eval-{safe_label(run.label)}"
    model_name = safe_label(run.label)
    tpu = REGION_TPU[run.region]
    # d=2432 (~2B params) OOMs vmem (128MB) on v6e-8 with default max_model_len=4096.
    # Route to v6e-16 us-east1-d (more chips → lower per-chip pressure) and cap context.
    extra_iris_args: list[str] = []
    extra_eval_args: list[str] = []
    region = run.region
    if run.dim == 2432:
        region = "us-east1"
        tpu = "v6e-16"
        extra_eval_args = ["--engine_kwargs", '{"max_model_len": 512}']

    cmd = [
        "uv",
        "run",
        "iris",
        "--cluster",
        "marin",
        "job",
        "run",
        "--region",
        region,
        "--tpu",
        tpu,
        "--memory",
        "128GB",
        "--disk",
        "50GB",
        "--priority",
        "interactive",
        "--extra",
        "marin:vllm",
        "--extra",
        "marin:safety",
        "--enable-extra-resources",
        "--no-wait",
        "--max-retries",
        str(max_retries),
        "--job-name",
        job_name,
        "-e",
        "WANDB_API_KEY",
        os.environ["WANDB_API_KEY"],
        "-e",
        "HF_TOKEN",
        os.environ["HF_TOKEN"],
        "-e",
        "MARIN_VLLM_MODE",
        "native",
        "--",
        "python",
        "-m",
        "experiments.safety.rtp_eval",
        "--model_name",
        model_name,
        "--model_path",
        run.hf_checkpoint,
        "--num_prompts",
        str(num_prompts),
        "--wandb_tags",
        "[rtp,safety,priority-sweep]",
        *extra_eval_args,
    ]
    _ = extra_iris_args  # reserved for future per-job iris flag overrides

    print(f"[{run.priority:8s}] {run.label:42s}  region={run.region:12s}  tpu={tpu}")
    if dry_run:
        print("  DRY-RUN: " + " ".join(cmd))
        return None

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  SUBMIT FAILED: {result.stderr.strip()[-400:]}")
        return None
    # iris prints job ID on the last line of stdout.
    job_id = result.stdout.strip().splitlines()[-1].strip()
    print(f"  → {job_id}")
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-retries", type=int, default=3, help="iris job retries on TPU preemption / transient failures"
    )
    parser.add_argument("--only", type=str, default=None, help="Substring filter on label (e.g. 'hq_3000')")
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Comma-separated EXACT label list to skip (e.g. 'hq_3000/9e+19-d512,nemotron/3e+18-d512')",
    )
    parser.add_argument("--priority-only", action="store_true", help="Skip the 2 bonus runs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    runs = list(RUNS)
    if args.only:
        runs = [r for r in runs if args.only in r.label]
    if args.exclude:
        excluded = {label.strip() for label in args.exclude.split(",")}
        runs = [r for r in runs if r.label not in excluded]
    if args.priority_only:
        runs = [r for r in runs if r.priority == "priority"]

    print(f"Submitting {len(runs)} RTP eval jobs (num_prompts={args.num_prompts}, dry_run={args.dry_run})")
    print()

    job_ids: list[tuple[str, str | None]] = []
    for run in runs:
        job_id = submit(run, args.num_prompts, args.dry_run, args.max_retries)
        job_ids.append((run.label, job_id))

    print()
    print(f"Submitted {sum(1 for _, j in job_ids if j is not None)}/{len(runs)} jobs.")
    failed = [label for label, j in job_ids if j is None]
    if failed and not args.dry_run:
        print(f"FAILED to submit: {failed}")


if __name__ == "__main__":
    main()
