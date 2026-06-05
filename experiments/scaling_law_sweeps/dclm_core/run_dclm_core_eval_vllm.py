# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the GENERATE_UNTIL subset of DCLM CORE tasks via vLLM-TPU.

Why a separate child: vLLM-TPU is much faster than Levanter's InferenceEngine
on autoregressive decode (paged attention + continuous batching) — exactly the
case that dominates wall-time on Llama-2-7B. Levanter, OTOH, is great for
loglikelihood (token packing). So we split:

  - Levanter child (`run_dclm_core_eval.py`)  → 14 multiple_choice tasks
  - vLLM child    (THIS file)                 → 8 generate_until tasks

Both write to the same `gs://.../partial/{run_name}/{task_alias}.json`
prefix. The aggregator (separate cron-style polling) combines them into the
final result when all 22 partials are present.

vLLM-TPU caveats (per memory `feedback_iris_vllm_tpu_standalone.md` and
marin's own `lm_evaluation_harness_evaluator.py`):
  - Needs `marin:tpu` + `marin:eval` + `marin:vllm` extras.
  - `MARIN_VLLM_MODE` unset or `native`.
  - vllm-tpu 0.18+ moved `vllm.utils.get_open_port`; lm-eval still imports
    the old symbol → monkey-patch it back.
  - lm-eval's `local-completions` always sends `seed=1234`; vLLM-TPU rejects
    per-request seeds (no JAX RNG plumbing) → strip from payloads.
  - No logprob support on TPU — gen_until ONLY.

Usage:
    python -m experiments.scaling_law_sweeps.dclm_core.run_dclm_core_eval_vllm \\
        --hf-checkpoint gs://marin-us-east5/.../hf/step-N/ \\
        --output-json gs://marin-us-central1/.../data_curation_core_results/<run>.json \\
        --run-name <run_name> \\
        --max-length 2048
"""

from __future__ import annotations

import argparse
import logging

# Reuse all the partial-checkpoint plumbing + task mapping from the Levanter sibling.
from experiments.scaling_law_sweeps.dclm_core.run_dclm_core_eval import (
    _install_custom_task_path,
    _partial_path,
    _read_partial,
    _resolve_partial_dir,
    _write_partial,
    build_task_configs,
)
from experiments.scaling_law_sweeps.dclm_core.task_mapping import CORE_TASK_MAP

logger = logging.getLogger(__name__)


# Task names whose metric is exact_match / contains / f1 ⇒ generative.
# Source of truth: CORE_TASK_MAP. Anything where the lm-eval task name ends
# with `_generate_until` or the metric is in this set is gen_until.
GEN_METRICS = {"exact_match", "contains", "f1"}


def _gen_until_aliases() -> set[str]:
    """Aliases (in our local naming `<dclm>_<num_fewshot>shot`) that go to vLLM."""
    out: set[str] = set()
    for entry in CORE_TASK_MAP:
        alias = f"{entry.dclm}_{entry.num_fewshot}shot"
        # Conservative: trust the entry.metric (we set it ourselves).
        if entry.metric in GEN_METRICS:
            out.add(alias)
    return out


def _apply_vllm_patches() -> None:
    """vllm-tpu 0.18 + lm-eval compatibility shims (mirrors marin's eval evaluator).

    1) vllm.utils.get_open_port: moved into a submodule in 0.18; lm-eval's
       openai_completions still imports the old path. Patch back.
    2) lm-eval's LocalCompletionsAPI._create_payload always adds `seed=1234`;
       vllm-tpu rejects that (no per-request RNG). Strip it.
    """
    import vllm.utils as _vu  # type: ignore[import]

    if not hasattr(_vu, "get_open_port"):
        for _mod_path in (
            "vllm.utils.network_utils",
            "vllm.utils.network",
            "vllm.utils._utils",
            "vllm.utils.utils",
        ):
            try:
                _mod = __import__(_mod_path, fromlist=["get_open_port"])
                _vu.get_open_port = _mod.get_open_port  # type: ignore[attr-defined]
                break
            except (ImportError, AttributeError):
                continue

    # lm_eval.models is registered lazily via __init__; force the import so the
    # `lm_eval.models.openai_completions` submodule path becomes resolvable.
    import lm_eval.models  # noqa: F401
    from lm_eval.models import openai_completions as _oai_mod

    _orig = _oai_mod.LocalCompletionsAPI._create_payload  # type: ignore[attr-defined]

    def _create_payload_no_seed(self, *args, **kwargs):
        payload = _orig(self, *args, **kwargs)
        payload.pop("seed", None)
        return payload

    _oai_mod.LocalCompletionsAPI._create_payload = _create_payload_no_seed  # type: ignore[attr-defined]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--hf-checkpoint",
        required=True,
        help="HF Hub id or gs:// path. vLLM loads from this; must contain config.json + model files.",
    )
    p.add_argument("--output-json", required=True, help="Final aggregated result path. Used to derive the partials dir.")
    p.add_argument(
        "--run-name", default=None, help="Optional run name override. Default = derived from checkpoint path."
    )
    p.add_argument("--tokenizer", default=None, help="Optional HF tokenizer id/path. Default = checkpoint path.")
    p.add_argument("--limit", type=int, default=None, help="Smoke test: cap each task to N examples. None = full eval.")
    p.add_argument("--max-length", type=int, default=2048, help="vLLM max_model_len. DCLM uses 2048.")
    p.add_argument(
        "--max-gen-toks",
        type=int,
        default=256,
        help="Cap per-prompt generation length. DCLM tasks all have short answers (jeopardy, "
        "bigbench operators, etc.); 256 is generous. Reducing this is the primary speed lever.",
    )
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    _install_custom_task_path()
    _apply_vllm_patches()

    checkpoint_path: str = args.hf_checkpoint.rstrip("/")
    tokenizer_path: str = args.tokenizer or checkpoint_path
    run_name = args.run_name or checkpoint_path.split("/")[-3]
    partial_dir = _resolve_partial_dir(args.output_json, run_name)

    logger.info("DCLM CORE vLLM gen_until eval: checkpoint=%s run_name=%s", checkpoint_path, run_name)
    logger.info("Per-task partials: %s", partial_dir)

    # Build the full 22-task list, then keep only the gen_until subset.
    all_tasks, alias_to_entry = build_task_configs(args.limit)
    gen_alias_set = _gen_until_aliases()
    gen_tasks = [t for t in all_tasks if t.task_alias in gen_alias_set]
    logger.info("Filtered to %d gen_until tasks (of %d total CORE tasks)", len(gen_tasks), len(all_tasks))
    for t in gen_tasks:
        logger.info("  → %s (lm-eval=%s, num_fewshot=%d)", t.task_alias, t.task, t.num_fewshot)

    # Skip-if-done: check existing partials.
    missing_tasks = []
    for t in gen_tasks:
        ppath = _partial_path(partial_dir, t.task_alias)
        if _read_partial(ppath) is not None:
            logger.info("  RESUME ✓ %s (partial exists)", t.task_alias)
        else:
            missing_tasks.append(t)
    if not missing_tasks:
        logger.info("All %d gen_until partials already present. Nothing to do.", len(gen_tasks))
        return
    logger.info("%d gen_until tasks to run", len(missing_tasks))

    # ---------------------------------------------------------------- vLLM up
    from urllib.parse import urlparse

    from marin.evaluation.evaluators.evaluator import ModelConfig
    from marin.inference.vllm_server import VllmEnvironment

    parsed = urlparse(checkpoint_path)
    is_object_store = parsed.scheme in {"gs", "s3"}
    engine_kwargs: dict = {"max_model_len": args.max_length}
    if is_object_store:
        # gs:// path → use runai streamer to load weights directly from GCS.
        engine_kwargs["load_format"] = "runai_streamer"
        model_cfg = ModelConfig(name=run_name, path=checkpoint_path, engine_kwargs=engine_kwargs)
    else:
        # HF Hub id → vLLM downloads through HF.
        model_cfg = ModelConfig(name=checkpoint_path, path=None, engine_kwargs=engine_kwargs)

    logger.info("Starting vLLM-TPU server on port %d...", args.port)
    env = VllmEnvironment(model=model_cfg, host="127.0.0.1", port=args.port, timeout_seconds=3600)

    with env:
        if env.model_id is None:
            raise RuntimeError("vLLM server did not report a model id.")
        served_model_id = env.model_id
        server_url = env.server_url  # ends in /v1
        logger.info("vLLM ready: model_id=%s server_url=%s", served_model_id, server_url)

        # lm-eval setup. local-completions with HF tokenizer for chat-template-free completion.
        # max_gen_toks caps autoregressive length; truncate=True drops over-long prompts to max_length.
        pretrained_args = (
            f"model={served_model_id},"
            f"base_url={server_url}/completions,"
            f"tokenizer={tokenizer_path},"
            "tokenizer_backend=huggingface,"
            "tokenized_requests=False,"
            f"max_gen_toks={args.max_gen_toks},"
            "truncate=True,"
            f"max_length={args.max_length}"
        )

        from lm_eval.evaluator import simple_evaluate
        from lm_eval.tasks import TaskManager

        task_manager = TaskManager()  # custom_tasks already patched in via _install_custom_task_path

        for i, t in enumerate(missing_tasks):
            logger.info(
                "[%d/%d] vLLM eval task %s (lm-eval=%s, num_fewshot=%d)",
                i + 1,
                len(missing_tasks),
                t.task_alias,
                t.task,
                t.num_fewshot,
            )
            try:
                results = simple_evaluate(
                    model="local-completions",
                    tasks=[t.to_dict()],  # pass the task as a dict to preserve alias + num_fewshot
                    model_args=pretrained_args,
                    batch_size="auto",
                    confirm_run_unsafe_code=True,
                    limit=args.limit,
                    log_samples=False,
                    task_manager=task_manager,
                )
            except Exception as e:
                logger.error("Task %s failed: %s — moving on", t.task_alias, e, exc_info=True)
                continue
            if results is None:
                logger.warning("Task %s returned None — skipping", t.task_alias)
                continue
            ppath = _partial_path(partial_dir, t.task_alias)
            _write_partial(ppath, results)
            logger.info("  ✓ wrote partial %s", ppath)

    logger.info("vLLM gen_until eval done. Aggregation is handled by the cron-style aggregator.")


if __name__ == "__main__":
    main()
