# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the OLMo Base-Easy bits-per-byte (bpb) suite on one HF checkpoint, offline, on TPU.

Unlike the OLMES MC suite (``olmes_base``), the bpb tasks are NOT stock lm-eval tasks —
they are pre-materialized oe-eval requests (``requests.jsonl.gz`` carrying a raw
``context`` + gold ``continuation`` per doc). So we bypass the lm-eval task registry and
marin's ``evaluate()`` entirely: load the checkpoint into Levanter exactly as
``levanter.eval_harness.run_eval_harness_main`` does, build the harness LM, and call its
``loglikelihood`` on hand-built ``Instance``s. bpb is then ours to compute:

    bpb = -sum(logprob of continuation tokens) / len(continuation.encode("utf-8")) * log2(e)

Data + model read locally & offline: the launcher syncs the region-local
``eval_datasets/olmo_in_loop_evals/`` (the oe-eval requests) and reuses the
``core_tasks_hub_cache`` (base HF configs, so ``from_hf`` never hits the Hub).

Parity note: OLMo's in-loop harness prepends BOS to the context; Levanter's
``loglikelihood`` tokenizes with ``add_special_tokens=False`` (no BOS). For most tasks
the context already carries few-shot priming, so absolute bpb is close but not
bit-identical to OLMo's published numbers — fine for cross-checkpoint comparison; pin
BOS handling later if exact parity is needed.

Usage (as an iris child; launcher supplies offline env + region pin):

    python -m experiments.scaling_law_sweeps.olmo_bpb.run_olmo_bpb_eval \\
        --hf-checkpoint gs://.../hf/step-N/ \\
        --output-dir gs://.../metadata/olmo_bpb_results/<run_name>/ \\
        --run-name <run_name> \\
        --dataset-cache-gcs gs://<same-bucket>/eval_datasets/olmo_in_loop_evals/ \\
        --hub-cache-gcs gs://<same-bucket>/eval_datasets/core_tasks_hub_cache/ \\
        --tasks all [--limit 8]
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os

from experiments.scaling_law_sweeps.core_tasks.run_core_tasks_eval import _sync_cache
from experiments.scaling_law_sweeps.olmo_bpb.olmo_bpb_tasks_set import resolve_tasks

logger = logging.getLogger(__name__)

_DATASET_CACHE_MARKER = "/eval_datasets/olmo_in_loop_evals/"
_HUB_CACHE_MARKER = "/eval_datasets/core_tasks_hub_cache/"  # reused from the CORE sweep
_LOG2_E = math.log2(math.e)


def _prepare_hub_cache(gcs_cache: str, local_dir: str = "/tmp/olmo_bpb_hf_home") -> int:
    n = _sync_cache(gcs_cache, _HUB_CACHE_MARKER, local_dir)
    os.environ["HF_HOME"] = local_dir
    logger.info("Synced %d hub-cache files -> %s; HF_HOME set", n, local_dir)
    return n


def _prepare_dataset_cache(gcs_cache: str, local_dir: str = "/tmp/olmo_bpb_data") -> str:
    n = _sync_cache(gcs_cache, _DATASET_CACHE_MARKER, local_dir)
    logger.info("Synced %d olmo-eval data files -> %s", n, local_dir)
    return local_dir


def _read_bpb_requests(data_dir: str, task_variant: str, limit: int | None) -> list[dict]:
    """Load the gold-continuation requests for one ``"<task>/<variant>"`` bpb task.

    Keeps exactly the gold continuation per doc (mirrors oe-eval's ce_loss/bpb filter:
    drop non-target continuations when the label is an integer index) and returns dicts
    with ``context`` / ``continuation`` / ``doc_id``.
    """
    path = os.path.join(data_dir, "oe_eval_tasks", task_variant, "requests.jsonl.gz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No requests for task {task_variant!r} at {path}")
    out: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            label, idx = rec.get("label"), rec.get("idx")
            # oe-eval bpb/ce_loss keeps only the target continuation; a non-target
            # numeric idx is skipped. String/None labels => single-continuation task.
            if isinstance(label, int) and label != idx:
                continue
            req = rec["request"]
            out.append({"context": req["context"], "continuation": req["continuation"], "doc_id": rec["doc_id"]})
            if limit is not None and len(out) >= limit:
                break
    return out


def _bpb_for_task(harness, requests: list[dict]) -> dict:
    """Score one task's requests and reduce to per-task bpb. Pure compute (no IO)."""
    from lm_eval.api.instance import Instance

    instances = [
        Instance(request_type="loglikelihood", doc={}, arguments=(r["context"], r["continuation"]), idx=i)
        for i, r in enumerate(requests)
    ]
    results = harness.loglikelihood(instances)  # [(logprob_sum, is_greedy), ...]

    per_doc_bpb: list[float] = []
    per_doc_bpb_no_lead: list[float] = []
    for r, (logprob, _greedy) in zip(requests, results, strict=True):
        cont = r["continuation"]
        byte_len = max(len(cont.encode("utf-8")), 1)
        per_doc_bpb.append(-logprob / byte_len * _LOG2_E)
        # OLMES also reports a leading-space-stripped variant.
        byte_len_no_lead = max(len(cont[1:].encode("utf-8")), 1)
        per_doc_bpb_no_lead.append(-logprob / byte_len_no_lead * _LOG2_E)

    n = len(per_doc_bpb)
    return {
        "bpb": sum(per_doc_bpb) / n,
        "bpb_no_leading_space": sum(per_doc_bpb_no_lead) / n,
        "n_docs": n,
    }


def _write_results(output_dir: str, payload: dict) -> None:
    from rigging.filesystem import filesystem as marin_filesystem

    out = f"{output_dir.rstrip('/')}/results.json"
    with marin_filesystem("gcs").open(out, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %s", out)


def _read_existing_results(output_dir: str) -> dict | None:
    """Load a prior results.json (for --merge), or None if absent/unreadable."""
    from rigging.filesystem import filesystem as marin_filesystem

    path = f"{output_dir.rstrip('/')}/results.json"
    fs = marin_filesystem("gcs")
    if not fs.exists(path):
        return None
    with fs.open(path, "r") as f:
        return json.load(f)


def _write_error(output_dir: str, message: str) -> None:
    """Best-effort _error.txt so the launcher's incomplete-detection surfaces failures."""
    from rigging.filesystem import filesystem as marin_filesystem

    try:
        with marin_filesystem("gcs").open(f"{output_dir.rstrip('/')}/_error.txt", "w") as f:
            f.write(message)
    except Exception as write_err:
        logger.error("Failed to write _error.txt: %s", write_err)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--hf-checkpoint", required=True, help="HF checkpoint dir (gs://...); config.json + safetensors.")
    p.add_argument("--output-dir", required=True, help="GCS dir to write results.json into.")
    p.add_argument("--run-name", default=None)
    p.add_argument("--dataset-cache-gcs", required=True, help="GCS prefix of olmo_in_loop_evals/ in THIS region.")
    p.add_argument("--hub-cache-gcs", required=True, help="GCS prefix of the model-config hub cache in THIS region.")
    p.add_argument("--tasks", default="all", help="'all' or comma-separated '<task>/<variant>' dirs.")
    p.add_argument("--limit", type=int, default=None, help="Cap each task to N docs (smoke test).")
    p.add_argument("--max-length", type=int, default=2048, help="Model context length for scoring.")
    p.add_argument(
        "--merge",
        action="store_true",
        help="Merge newly-scored tasks into an existing results.json instead of overwriting it.",
    )
    p.add_argument(
        "--done-marker",
        default=None,
        help="On success, write this empty marker file under --output-dir. Lets the launcher's "
        "keep-alive detect completion in --merge mode, where results.json already exists.",
    )
    args = p.parse_args()

    try:
        _drive(args)
    except Exception:
        import traceback

        _write_error(args.output_dir, traceback.format_exc())
        raise


def _drive(args) -> None:
    # Every HF surface offline+local BEFORE any HF/levanter import.
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    _prepare_hub_cache(args.hub_cache_gcs)
    data_dir = _prepare_dataset_cache(args.dataset_cache_gcs)

    tasks = resolve_tasks(args.tasks)
    checkpoint = args.hf_checkpoint.rstrip("/")
    logger.info("OLMo bpb eval: checkpoint=%s tasks=%d limit=%s", checkpoint, len(tasks), args.limit)

    import jax
    import jmp
    from levanter.compat.hf_checkpoints import HFCheckpointConverter, load_tokenizer
    from levanter.eval_harness import _LmEvalHarnessWorker
    from levanter.tracker import NoopConfig
    from levanter.trainer import TrainerConfig
    from levanter.utils.tree_utils import inference_mode

    trainer_config = TrainerConfig(
        tracker=NoopConfig(),
        mp=jmp.get_policy("p=bfloat16,c=bfloat16"),
        per_device_eval_parallelism=1,
    )
    trainer_config.initialize()

    model_config = HFCheckpointConverter.from_hf(checkpoint).LevConfigClass()
    tokenizer = load_tokenizer(checkpoint)
    parameter_axis_mapping = trainer_config.parameter_axis_mapping
    compute_axis_mapping = trainer_config.compute_axis_mapping

    # The harness dispatch (broadcast_shard) requires the device mesh to stay ACTIVE, so
    # model load, loglikelihood, and worker.stop() must ALL run inside this context.
    with trainer_config.use_device_mesh():
        converter = model_config.hf_checkpoint_converter().replaced(reference_checkpoint=checkpoint, tokenizer=tokenizer)
        model = converter.load_pretrained(
            model_config.model_type,
            ref=checkpoint,
            dtype=trainer_config.mp.compute_dtype,
            axis_mapping=parameter_axis_mapping,
        )
        model = inference_mode(model, True)

        worker = _LmEvalHarnessWorker(
            trainer_config.EvalBatch,
            model.Pos.resize(args.max_length),
            model,
            compute_axis_mapping,
            tokenizer,
            trainer_config.mp,
            max_packed_segments=64,
        )

        if jax.process_index() != 0:
            worker.worker_message_loop()
            return

        harness = worker.make_harness_lm()
        task_results: dict[str, dict] = {}
        try:
            for task_variant in tasks:
                requests = _read_bpb_requests(data_dir, task_variant, args.limit)
                res = _bpb_for_task(harness, requests)
                task_results[task_variant] = res
                logger.info("  %-45s bpb=%.4f (n=%d)", task_variant, res["bpb"], res["n_docs"])
        finally:
            worker.stop()

    # --merge: fold the newly-scored tasks into a prior results.json (new tasks win on conflict),
    # so a second sweep (e.g. the QA rc tasks) extends the first without recomputing it.
    merged_tasks = task_results
    if args.merge:
        existing = _read_existing_results(args.output_dir)
        if existing and existing.get("tasks"):
            merged_tasks = {**existing["tasks"], **task_results}
            logger.info(
                "Merged %d new tasks into %d existing -> %d total",
                len(task_results),
                len(existing["tasks"]),
                len(merged_tasks),
            )

    macro = sum(r["bpb"] for r in merged_tasks.values()) / len(merged_tasks) if merged_tasks else float("nan")
    payload = {
        "run_name": args.run_name,
        "checkpoint": checkpoint,
        "max_length": args.max_length,
        "limit": args.limit,
        "tasks": merged_tasks,
        "averages": {"macro_bpb": macro},
    }
    _write_results(args.output_dir, payload)
    if args.done_marker:
        from rigging.filesystem import filesystem as marin_filesystem

        with marin_filesystem("gcs").open(f"{args.output_dir.rstrip('/')}/{args.done_marker}", "w") as f:
            f.write("")
        logger.info("Wrote done-marker %s", args.done_marker)
    logger.info("Done. macro_bpb=%.4f over %d tasks", macro, len(merged_tasks))


if __name__ == "__main__":
    main()
