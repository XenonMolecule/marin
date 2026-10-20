# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit one TPU Iris job per WARC running ``rescraper_extract_tpu.py``.

Same job shape as ``launch_wmb_tpu.py``: leaf jobs, TPU extras plus the
vLLM/tpu-inference forks, preemptible. Output lands in the job region's bucket.

    uv run python experiments/baseline_collection/launch_rescraper_tpu.py --tag pilot1 \\
        --warc s3://commoncrawl/crawl-data/.../x.warc.gz --region us-east5
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import pathlib

from iris.cli.connect import open_iris_client
from iris.cluster.constraints import Constraint, ConstraintOp, WellKnownAttribute, preemptible_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, tpu_device
from iris.rpc import job_pb2
from marin.external_dependencies import TPU_INFERENCE_FORK_REQUIREMENT, VLLM_FORK_REQUIREMENT

logger = logging.getLogger(__name__)
MODULE = "experiments.baseline_collection.rescraper_extract_tpu"
REGION_BUCKETS = {
    "us-east5": "marin-us-east5",
    "europe-west4": "marin-eu-west4",
    "us-east1": "marin-us-east1",
}
# The renderer is webpage_converter from MinerU-HTML's mineru-webkit, at the version ReScraper pins.
# trafilatura is Dripper's fallback extractor.
# selectolax 1.0 (2026-10) dropped the Modest parser the renderer imports; 0.4.7 is ReScraper's pin.
RESCRAPER_PACKAGES = ["mineru-webkit==0.1.6", "selectolax==0.4.7", "beautifulsoup4", "trafilatura==2.0.0"]
# One single-chip engine runs per chip.
TPU_CHIPS = {"v6e-4": 4, "v6e-8": 8, "v5p-8": 4}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--warc", action="append", default=[], help="s3://commoncrawl/... (repeatable)")
    parser.add_argument("--input-jsonl", help="gs:// page sample (jsonl.gz) to run instead of WARCs")
    parser.add_argument("--input-rendered-jsonl", help="gs:// pre-rendered page sample; ReScraper phase only")
    parser.add_argument("--tag", required=True, help="run name; output goes under users/michaelryan/rescraper/<tag>/")
    parser.add_argument("--region", required=True, choices=sorted(REGION_BUCKETS))
    parser.add_argument("--tpu-type", default="v6e-4", choices=sorted(TPU_CHIPS))
    parser.add_argument("--priority", choices=["interactive", "batch"], default="batch")
    parser.add_argument("--cpu", type=int, default=16)
    parser.add_argument("--memory", default="200GB", help="four engines compile at once; 64GB was OOM-killed")
    parser.add_argument("--group-size", type=int, default=1000)
    parser.add_argument("--limit-records", type=int, default=None)
    parser.add_argument("--kv-cache-dtype", default="auto", help='"bfloat16" disables vllm-tpu\'s fp8 KV-cache default')
    parser.add_argument("--temperature", type=float, default=1.0, help="ReScraper sampling temperature (0 = greedy)")
    parser.add_argument(
        "--no-prefix-caching", action="store_true", help="disable prefix caching in the ReScraper engine"
    )
    parser.add_argument(
        "--model-impl-type",
        default=None,
        choices=["vllm", "flax_nnx"],
        help="tpu-inference MODEL_IMPL_TYPE: 'vllm' runs the PyTorch model definition via torchax instead of the JAX port",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", default="lib/iris/config/marin.yaml")
    args = parser.parse_args()

    samples = [
        (f, v)
        for f, v in (("--input-jsonl", args.input_jsonl), ("--input-rendered-jsonl", args.input_rendered_jsonl))
        if v
    ]
    if bool(args.warc) == bool(samples) or len(samples) > 1:
        parser.error("pass either --warc (repeatable) or one of --input-jsonl / --input-rendered-jsonl")
    # One job per input: a WARC path, or the page sample.
    inputs = [("--warc", w) for w in args.warc] or samples
    bucket = REGION_BUCKETS[args.region]
    extra_env = {"MODEL_IMPL_TYPE": args.model_impl_type} if args.model_impl_type else {}
    with open_iris_client(config_file=pathlib.Path(args.config), workspace=pathlib.Path.cwd()) as client:
        for flag, source in inputs:
            warc_id = hashlib.sha256(source.encode()).hexdigest()[:12]
            kind = "warc" if flag == "--warc" else "sample"
            out_dir = f"gs://{bucket}/users/michaelryan/rescraper/{args.tag}/{kind}-{warc_id}"
            name = f"rescraper-{args.tag}-{warc_id}"
            cmd = ["-m", MODULE, flag, source, "--output", out_dir, "--group-size", str(args.group_size),
                   "--render-procs", str(max(1, args.cpu - 2)),
                   "--num-chips", str(TPU_CHIPS[args.tpu_type])]  # fmt: skip
            if args.limit_records:
                cmd += ["--limit-records", str(args.limit_records)]
            cmd += ["--kv-cache-dtype", args.kv_cache_dtype, "--temperature", str(args.temperature)]
            if args.no_prefix_caching:
                cmd.append("--no-prefix-caching")
            if args.dry_run:
                logger.info("[dry-run] %s on %s/%s -> python %s", name, args.tpu_type, args.region, " ".join(cmd))
                continue
            job = client.submit(
                entrypoint=Entrypoint.from_command("python", *cmd),
                name=name,
                resources=ResourceSpec(cpu=args.cpu, memory=args.memory, disk="20GB", device=tpu_device(args.tpu_type)),
                environment=EnvironmentSpec(
                    extras=["tpu"],
                    pip_packages=[VLLM_FORK_REQUIREMENT, TPU_INFERENCE_FORK_REQUIREMENT, *RESCRAPER_PACKAGES],
                    env_vars={"VLLM_TARGET_DEVICE": "tpu", "UV_TORCH_BACKEND": "cpu", **extra_env},
                ),
                constraints=[
                    preemptible_constraint(True),
                    Constraint.create(
                        key=WellKnownAttribute.REGION,
                        op=ConstraintOp.IN,
                        values=[args.region],
                        mode=job_pb2.CONSTRAINT_MODE_REQUIRED,
                    ),
                ],
                max_retries_preemption=100,
                max_retries_failure=3,
                priority_band=(
                    job_pb2.PRIORITY_BAND_INTERACTIVE if args.priority == "interactive" else job_pb2.PRIORITY_BAND_BATCH
                ),
            )
            logger.info("Submitted %s (%s/%s) -> %s  out=%s", name, args.tpu_type, args.region, job.job_id, out_dir)


if __name__ == "__main__":
    main()
