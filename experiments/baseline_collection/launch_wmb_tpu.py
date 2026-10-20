# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage WebMainBench pages to GCS as shards and submit one TPU Iris job per shard running
``wmb_extract_tpu.py`` (production llm_pipeline_v1_1 extraction, filter bypassed).

Mirrors ``launch_gapfill.py``: coordinator-free leaf jobs, TPU extras plus the vLLM/tpu-inference
forks, preemptible, batch priority, region pin so the checkpoint read stays in-region.

    # smoke: 20 pages, one v6e-4 in us-east5
    uv run python experiments/baseline_collection/launch_wmb_tpu.py \\
        --html-dir ~/.../small-rephraser/outputs/webmainbench/html_dir_7809 --limit 20 --num-shards 1 --tag smoke20

    # full set: 16 shards of ~490 pages across the v6e-4 regions
    uv run python experiments/baseline_collection/launch_wmb_tpu.py --html-dir ... --num-shards 16 --tag full7809

Gather afterwards: ``gcloud storage cat gs://.../<tag>/out/shard-*/group_*.jsonl.gz`` concatenated is one
record per page keyed by ``metadata.track_id``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import pathlib

import fsspec
from iris.cli.connect import open_iris_client
from iris.cluster.constraints import Constraint, ConstraintOp, WellKnownAttribute, preemptible_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, tpu_device
from iris.rpc import job_pb2
from marin.external_dependencies import TPU_INFERENCE_FORK_REQUIREMENT, VLLM_FORK_REQUIREMENT

logger = logging.getLogger(__name__)
MODULE = "experiments.baseline_collection.wmb_extract_tpu"
# (tpu_type, region, staging bucket): the SFT checkpoint is mirrored in every one of these regions.
TARGETS = [
    ("v6e-4", "us-east5", "marin-us-east5"),
    ("v6e-4", "europe-west4", "marin-eu-west4"),
    ("v6e-4", "us-east1", "marin-us-east1"),
]


def load_pages(html_dir: pathlib.Path, limit: int | None) -> list[dict]:
    pages = []
    for p in sorted(html_dir.glob("*.html")):
        meta = (
            json.loads(p.with_suffix(".html.meta.json").read_text()) if p.with_suffix(".html.meta.json").exists() else {}
        )
        pages.append(
            {
                "track_id": meta.get("track_id", p.stem),
                "url": meta.get("url", ""),
                "meta": meta.get("meta"),
                "html": p.read_text(encoding="utf-8", errors="replace"),
            }
        )
        if limit and len(pages) >= limit:
            break
    return pages


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--html-dir", required=True)
    parser.add_argument(
        "--tag", required=True, help="run name; staging goes under users/michaelryan/webmainbench/<tag>/"
    )
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="first N pages (smoke test)")
    parser.add_argument("--group-size", type=int, default=50)
    parser.add_argument("--tpu-type", default=None, help="override TARGETS' tpu type")
    parser.add_argument("--pin-region", default=None, help="put every shard in this region")
    parser.add_argument("--priority", choices=["interactive", "batch"], default="batch")
    parser.add_argument("--memory", default="64GB")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--real-filter", action="store_true", help="pass --real-filter to the worker (keep/drop run)")
    parser.add_argument(
        "--model-impl-type",
        default=None,
        choices=["vllm", "flax_nnx"],
        help="tpu-inference MODEL_IMPL_TYPE; 'vllm' (torchax) avoids the JAX port's RoPE-scaling bug in the pinned fork",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", default="lib/iris/config/marin.yaml")
    args = parser.parse_args()
    extra_env = {"MODEL_IMPL_TYPE": args.model_impl_type} if args.model_impl_type else {}

    pages = load_pages(pathlib.Path(args.html_dir).expanduser(), args.limit)
    shards = [pages[i :: args.num_shards] for i in range(args.num_shards)]
    logger.info("%d pages -> %d shards", len(pages), len(shards))
    fs = fsspec.filesystem("gcs")

    with open_iris_client(config_file=pathlib.Path(args.config), workspace=pathlib.Path.cwd()) as client:
        for s, shard in enumerate(shards):
            tpu_type, region, bucket = TARGETS[s % len(TARGETS)]
            tpu_type = args.tpu_type or tpu_type
            if args.pin_region:
                region = args.pin_region
                bucket = next(b for _, r, b in TARGETS if r == region)
            base = f"{bucket}/users/michaelryan/webmainbench/{args.tag}"
            shard_path = f"{base}/shards/shard-{s:02d}.jsonl.gz"
            out_dir = f"gs://{base}/out/shard-{s:02d}"
            if not args.dry_run:
                with fs.open(shard_path, "wb") as f, gzip.open(f, "wt") as z:
                    for p in shard:
                        z.write(json.dumps(p, ensure_ascii=False) + "\n")
            name = f"wmb-{args.tag}-s{s:02d}"
            cmd = [
                "-m",
                MODULE,
                "--input",
                f"gs://{shard_path}",
                "--output",
                out_dir,
                "--group-size",
                str(args.group_size),
                "--max-model-len",
                str(args.max_model_len),
            ]
            if args.real_filter:
                cmd.append("--real-filter")
            if args.dry_run:
                logger.info(
                    "[dry-run] %s on %s/%s (%d pages) -> python %s", name, tpu_type, region, len(shard), " ".join(cmd)
                )
                continue
            job = client.submit(
                entrypoint=Entrypoint.from_command("python", *cmd),
                name=name,
                resources=ResourceSpec(cpu=0.5, memory=args.memory, disk="5GB", device=tpu_device(tpu_type)),
                environment=EnvironmentSpec(
                    extras=["tpu"],
                    pip_packages=[VLLM_FORK_REQUIREMENT, TPU_INFERENCE_FORK_REQUIREMENT],
                    env_vars={"VLLM_TARGET_DEVICE": "tpu", "UV_TORCH_BACKEND": "cpu", **extra_env},
                ),
                constraints=[
                    preemptible_constraint(True),
                    Constraint.create(
                        key=WellKnownAttribute.REGION,
                        op=ConstraintOp.IN,
                        values=[region],
                        mode=job_pb2.CONSTRAINT_MODE_REQUIRED,
                    ),
                ],
                max_retries_preemption=100,
                max_retries_failure=3,
                priority_band=(
                    job_pb2.PRIORITY_BAND_INTERACTIVE if args.priority == "interactive" else job_pb2.PRIORITY_BAND_BATCH
                ),
            )
            logger.info(
                "Submitted %s (%s/%s, %d pages) -> %s  out=%s", name, tpu_type, region, len(shard), job.job_id, out_dir
            )


if __name__ == "__main__":
    main()
