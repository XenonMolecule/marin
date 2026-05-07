# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris-native Levanter-backend eval child for logprob MMLU.

Parallels `run_eval_standalone.py` (which uses vLLM-TPU + lm-eval-harness)
but uses Levanter's own forward pass instead. Why two backends?

- vLLM-TPU's loglikelihood path is broken (vllm-project/vllm#8499). We saw
  it here too: the smoke for `mmlu_*` 5-shot crashed with HTTP 500 / err 2701.
- The vLLM-generative variant (`mmlu_*_generative`) works but uses an
  unusual `::` template that confuses Base models into emitting `</s>` as
  the first generated token, deflating Base baselines toward random
  (Qwen3-0.6B-Base scored 24.57% all-7 vs Qwen's reported ~47% MMLU).
- Levanter's `eval_harness` runs lm-eval-harness on TPU directly via
  Levanter's JAX/Haliax forward pass — no vLLM, no template-mismatch
  collapse. It's loglikelihood-only by design, which is exactly what
  `mmlu_*` (the standard non-generative variant) needs.

This standalone exists so we can produce apples-to-apples logprob MMLU
numbers comparable to Qwen's reported MMLU and to the rest of the literature.

USAGE
-----
    python experiments/rephraser/run_eval_levanter_standalone.py \\
        --domain medical-logprob \\
        --model-rel-path Qwen/Qwen3-0.6B-Base \\
        --model-name baseline-qwen3-0.6b-base-medical-logprob
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import types

import fsspec

# Pre-import patch: Levanter's `eval_harness.run_eval_harness_main` calls
# `tokenizer.encode_batch(list[str])` (lines 1690-1691 of eval_harness.py),
# which is part of Levanter's `MarinTokenizer` Protocol but NOT part of
# HuggingFace's PreTrainedTokenizerFast surface. The `LevanterLmEvalEvaluator`
# wrapper passes the HF tokenizer straight through, so for Qwen-family models
# (Qwen2TokenizerFast et al.) the eval crashes immediately. Shim it in by
# adding `encode_batch` to HF tokenizers at AutoTokenizer.from_pretrained
# time. Idempotent — only adds when missing.
from transformers import AutoTokenizer as _AutoTokenizer

_orig_auto_tokenizer_from_pretrained = _AutoTokenizer.from_pretrained


def _patched_auto_tokenizer_from_pretrained(*args, **kwargs):
    tok = _orig_auto_tokenizer_from_pretrained(*args, **kwargs)
    if not hasattr(tok, "encode_batch"):

        def encode_batch(self, texts, *, add_special_tokens: bool = False):
            return [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]

        tok.encode_batch = types.MethodType(encode_batch, tok)
    return tok


_AutoTokenizer.from_pretrained = _patched_auto_tokenizer_from_pretrained
# Levanter's `_hf_hub_retry` wraps from_pretrained but ultimately calls
# AutoTokenizer.from_pretrained, so the shim above propagates through.


# ---------------------------------------------------------------------------
# Tokenizer-vs-model vocab padding shim
# ---------------------------------------------------------------------------
# Qwen3 models ship with vocab=151936 but the Qwen tokenizer.json only
# contains 151669 entries (151665 base + 4 special tokens for instruct
# variants). The 267-entry pad makes the model dimension TPU-divisible.
# Levanter's training path bridges this with `pad_tokenizer_to_match_model=True`
# (see lib/levanter/src/levanter/main/train_lm.py:103), which calls
# `HFCheckpointConverter.with_tokenizer_padded_to_match_model()` before
# `load_pretrained`. The eval-harness path doesn't expose that flag — so the
# raw 151669-entry tokenizer meets the 151936-entry model and the first
# logp(option_token) call segfaults the TPU kernel.
#
# Fix: monkey-patch `run_eval_harness_main` to insert the same pad step
# before model load. We pre-build the converter, apply
# with_tokenizer_padded_to_match_model(), and inject the padded tokenizer
# into the `EvalHarnessMainConfig.the_tokenizer` cached_property slot before
# upstream reads it. Upstream code is unmodified.
import levanter.eval_harness as _lev_eval_harness  # noqa: E402

_orig_run_eval_harness_main = _lev_eval_harness.run_eval_harness_main


def _run_eval_harness_main_with_padded_tokenizer(config):
    """Wrap upstream run_eval_harness_main with a tokenizer-padding step."""
    if config.checkpoint_is_hf:
        # Build the converter the same way upstream does, then apply padding.
        # We reach upstream's tokenizer once via `the_tokenizer` (which
        # populates cached_property's __dict__ slot), then overwrite that
        # slot with the padded tokenizer so upstream's second read returns
        # the padded one.
        unpadded_tokenizer = config.the_tokenizer
        converter = config.model.hf_checkpoint_converter()
        converter = converter.replaced(reference_checkpoint=config.checkpoint_path, tokenizer=unpadded_tokenizer)
        converter = converter.with_tokenizer_padded_to_match_model()
        # cached_property reads `__dict__[name]` first; if present it returns
        # that and never calls the descriptor. Overwriting the slot here is
        # safe because we just populated it via the read above.
        config.__dict__["the_tokenizer"] = converter.tokenizer
    return _orig_run_eval_harness_main(config)


_lev_eval_harness.run_eval_harness_main = _run_eval_harness_main_with_padded_tokenizer


# ---------------------------------------------------------------------------
# Skip the from_hf registry sweep
# ---------------------------------------------------------------------------
# `LevanterLmEvalEvaluator.evaluate()` calls
# `HFCheckpointConverter.from_hf(model_path).LevConfigClass()` to pick the
# right Levanter config class. Internally that iterates over EVERY registered
# `LmConfig` choice (Llama, Gemma, Qwen3, Mistral, ...) and pings HF for each
# class's default reference_checkpoint to compare types. ~10 HF API calls per
# child startup. Fanning out 21 children blows HuggingFace's 1000/5min rate
# limit even with 30s/child staggering, since iris retries multiply the call
# volume.
#
# All our SFT runs are Qwen3-* models. Short-circuit the registry sweep by
# monkey-patching `from_hf` to detect Qwen3 from the config and return a
# pre-built Qwen3 converter directly — zero registry HF calls.
import functools  # noqa: E402

import levanter.compat.hf_checkpoints as _lev_hfc  # noqa: E402

_orig_from_hf = _lev_hfc.HFCheckpointConverter.from_hf


@functools.wraps(_orig_from_hf)
def _patched_from_hf(model_name_or_path, trust_remote_code: bool = False):
    """If the HF config says model_type=qwen3, build the Qwen3 converter directly."""
    try:
        ref = _lev_hfc._coerce_to_rr(model_name_or_path)
        config_class = _lev_hfc.HFCheckpointConverter._infer_config_class(None, ref, trust_remote_code)
        if getattr(config_class, "__name__", "") == "Qwen3Config":
            from levanter.models.qwen import Qwen3Config

            return Qwen3Config().hf_checkpoint_converter()
    except Exception as e:
        logger.warning("Qwen3 short-circuit failed (%s); falling back to registry sweep.", e)
    return _orig_from_hf(model_name_or_path, trust_remote_code=trust_remote_code)


_lev_hfc.HFCheckpointConverter.from_hf = staticmethod(_patched_from_hf)

from experiments.rephraser.run_eval_standalone import (  # noqa: E402  — patch must precede this
    DOMAIN_EVALS,
    DOMAIN_SUMMARY_PREFIX,
    _is_hf_reference,
    _resolve_local_model_path,
    _assert_model_local,
)
from experiments.scaling_law_sweeps import region_tracker  # noqa: E402
from marin.evaluation.evaluators.evaluator import ModelConfig  # noqa: E402
from marin.evaluation.evaluators.levanter_lm_eval_evaluator import LevanterLmEvalEvaluator  # noqa: E402

logger = logging.getLogger(__name__)


# Levanter's eval_harness only supports loglikelihood, so we restrict this
# standalone to logprob-style task lists. The generative-MMLU + HumanEval +
# MBPP + minerva_math runs continue to use the vLLM standalone.
LEVANTER_DOMAINS: tuple[str, ...] = ("medical-logprob",)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--domain", choices=list(LEVANTER_DOMAINS), required=True)
    parser.add_argument(
        "--model-rel-path",
        required=True,
        help="HF checkpoint path RELATIVE to local region bucket, OR an HF reference like 'Qwen/Qwen3-0.6B-Base'.",
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--eval-results-prefix", default=None)
    parser.add_argument("--max-eval-instances", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    region = region_tracker.detect_current_region()
    logger.info(
        "Standalone Levanter EVAL child boot — domain=%s, region=%s, model=%s",
        args.domain,
        region,
        args.model_name,
    )

    # Resolve model path. HF references skip the local-region assert (Levanter
    # downloads from HF Hub). GCS rel-paths get composed and asserted, same as
    # the vLLM standalone — preventing accidental cross-region reads.
    if _is_hf_reference(args.model_rel_path):
        model_path = args.model_rel_path
        logger.info("HF reference: %s", model_path)
    else:
        model_path = _resolve_local_model_path(args.model_rel_path, region)
        _assert_model_local(model_path, region)
        logger.info("Local model path: %s", model_path)

    # Resolve output path. Match the vLLM standalone layout so downstream
    # analysis tools find results in the same conventional places.
    if args.output_path:
        output_path = args.output_path
    elif model_path.startswith("gs://"):
        output_path = f"{model_path.rstrip('/')}/eval/lm_eval_harness_levanter"
    else:
        bucket = region_tracker.REGION_TO_BUCKET[region]
        output_path = f"{bucket}/eval_baselines/{args.model_name}/lm_eval_harness_levanter"
    logger.info("Eval output path: %s", output_path)

    evals = DOMAIN_EVALS[args.domain]
    eval_results_prefix = args.eval_results_prefix or DOMAIN_SUMMARY_PREFIX[args.domain]

    model = ModelConfig(
        name=args.model_name,
        path=model_path,
        engine_kwargs={},
        generation_params=None,
        apply_chat_template=False,  # Base models, no chat template
        base_eval_run_name=args.model_name,
    )

    evaluator = LevanterLmEvalEvaluator()
    # W&B tags max 64 chars per tag. `model={name}` blows past on resiliparse
    # run names (e.g. "medical-resiliparse-lr1e-6_bs64-qwen3-0.6b-base-rerun"
    # + "-medical-logprob" + "model=" prefix = 72 chars). Truncate the tag's
    # model identifier rather than dropping it (still useful for filtering).
    model_tag = f"m={args.model_name}"[:62]
    evaluator.evaluate(
        model=model,
        evals=evals,
        output_path=output_path,
        max_eval_instances=args.max_eval_instances,
        wandb_tags=[
            args.domain,
            "eval",
            "qwen3-base",
            "backend=levanter",
            model_tag,
        ],
    )
    logger.info("Eval finished cleanly.")

    summary = {
        "backend": "levanter",
        "domain": args.domain,
        "model_name": args.model_name,
        "model_path": model_path,
        "model_rel_path": args.model_rel_path,
        "region": region,
        "output_path": output_path,
        "evals": [e.task_alias or e.name for e in evals],
        "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    summary_path = f"{eval_results_prefix.rstrip('/')}/{args.model_name}.json"
    try:
        with fsspec.open(summary_path, "w") as f:
            f.write(json.dumps(summary, indent=2))
        logger.info("Wrote eval summary: %s", summary_path)
    except Exception as e:
        logger.warning("Failed to write eval summary at %s: %s", summary_path, e)

    done_marker = f"{output_path.rstrip('/')}/.{args.domain}_eval_DONE"
    try:
        with fsspec.open(done_marker, "w") as f:
            f.write(
                json.dumps(
                    {
                        "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
                        "backend": "levanter",
                        "domain": args.domain,
                        "model_name": args.model_name,
                        "region": region,
                    }
                )
            )
        logger.info("Wrote DONE marker: %s", done_marker)
    except Exception as e:
        logger.warning("Failed to write DONE marker at %s: %s", done_marker, e)


if __name__ == "__main__":
    main()
