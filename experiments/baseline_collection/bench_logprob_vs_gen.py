# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark: scoring the [NO_USEFUL_CONTENT] marker via prompt_logprobs vs full generation.

Question this answers (for the distilled Qwen3-1.7B no-think extractor, high_quality spec):

  In production we run ``llm.generate(..., max_tokens=6144)`` over every page just to
  discover whether the model emits ``[NO_USEFUL_CONTENT]``. If all we want is the
  useful/not-useful decision, we can instead score the marker's log-probability in a
  SINGLE prefill pass (``prompt_logprobs``, ``max_tokens=1``) — no autoregressive decode.

This script, on real WARC HTML, measures:
  1. Whether ``prompt_logprobs`` is even supported on ``vllm-tpu==0.18`` TPU backend.
  2. Full-generation throughput (docs/s, output tok/s) — the production setting.
  3. Marker-scoring throughput (docs/s) — the proposed shortcut.
  4. The speedup, and a quality teaser: does the marker logprob separate the pages that
     actually generated ``[NO_USEFUL_CONTENT]`` from those that produced real content?

It deliberately reuses ``run_extract_standalone``'s exact engine config and prompt
construction so the numbers transfer to the real fleet. Mirror launch in the module
docstring of ``run_extract_standalone.py``::

    iris --cluster marin job run --region us-east5 --tpu v6e-4 \
        --enable-extra-resources --memory 128GB --disk 100GB \
        --extra vllm --extra tpu --priority interactive --no-wait \
        -e WANDB_API_KEY ... -e HF_TOKEN ... \
        -- python experiments/baseline_collection/bench_logprob_vs_gen.py \
        --model gs://marin-us-east5/checkpoints/qwen3-1.7b-hq-distill-bal350k-proxy100pct-lr7e-6-bs128-mhfix-27b106/hf/step-5063 \
        --manifest experiments/distill/dclm_1p7b_completed_sample200_warcs.txt --num-docs 200
"""

import argparse
import json
import logging
import os
import time
from collections import Counter

from experiments.baseline_collection.download_warcs import _download_one_warc, _load_manifest
from experiments.baseline_collection.extraction_specs import get_spec
from experiments.baseline_collection.run_extract_standalone import (
    MAX_DOC_TOKENS,
    MAX_OUTPUT_TOKENS,
    _clean_text,
    _filter_by_length,
)

logger = logging.getLogger(__name__)

# Default to the 1.7B no-think hq-distill extractor in us-east5 (see project_1p7b_extraction_benchmark).
DEFAULT_MODEL = (
    "gs://marin-us-east5/checkpoints/"
    "qwen3-1.7b-hq-distill-bal350k-proxy100pct-lr7e-6-bs128-mhfix-27b106/hf/step-5063"
)
MARKER = "[NO_USEFUL_CONTENT]"


def _build_prompt_ids(records, tokenizer, template, system_message):
    """Replicate run_extract_standalone._process_batch prompt construction exactly."""
    prompt_ids_list = []
    for record in records:
        html = record.get("html", "")
        tokens = tokenizer.encode(html)
        if len(tokens) > MAX_DOC_TOKENS:
            tokens = tokens[:MAX_DOC_TOKENS]
            html = tokenizer.decode(tokens)
        text = template.format(example=html)
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": text})
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids_list.append(tokenizer.encode(prompt_text))
    return prompt_ids_list


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--manifest", required=True, help="GCS/local path to a WARC manifest")
    parser.add_argument("--spec", default="high_quality")
    parser.add_argument("--num-docs", type=int, default=200, help="Docs to benchmark (from the first WARC)")
    parser.add_argument("--warmup-docs", type=int, default=16, help="Docs used to warm the XLA compile cache")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument(
        "--prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable vLLM prefix caching (production default). MUST be disabled for marker "
            "scoring: prompt_logprobs is incompatible with prefix caching — with it on, vLLM "
            "returns a degenerate length-1 logprobs list and the marker logprob can't be read."
        ),
    )
    parser.add_argument("--tp", type=int, default=None, help="Tensor parallel size (default: TPU device count)")
    parser.add_argument(
        "--result-path",
        default=None,
        help="GCS path to write the JSON summary (default: marin-us-east5 benchmark namespace)",
    )
    parser.add_argument(
        "--speculative-ngram",
        type=int,
        default=0,
        help=(
            "num_speculative_tokens for prompt-lookup (n-gram) speculative decoding. 0 = off. "
            "When >0 the engine drafts tokens by matching the last n-gram against the prompt — "
            "ideal for extraction, which copies long verbatim spans from the HTML. Runs a "
            "generation-only benchmark (vs the no-spec baseline) and skips the logprobs paths."
        ),
    )
    parser.add_argument("--ngram-max", type=int, default=4, help="prompt_lookup_max for n-gram spec decode")
    parser.add_argument("--ngram-min", type=int, default=2, help="prompt_lookup_min for n-gram spec decode")
    args = parser.parse_args()

    # Engine env — verbatim from run_extract_standalone.main().
    marin_pfx = os.environ.get("MARIN_PREFIX")
    cache_dir = os.path.join(marin_pfx, "compilation-cache") if marin_pfx else "/tmp/marin-jax-compilation-cache"
    os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", cache_dir)
    os.environ.setdefault("VLLM_XLA_CACHE_PATH", cache_dir)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import jax

    devices = jax.devices()
    tp = args.tp or len([d for d in devices if d.platform == "tpu"]) or len(devices)
    logger.info("JAX devices (%d): %s | tensor_parallel_size=%d", len(devices), devices, tp)

    from vllm import LLM, SamplingParams

    # Prompt-lookup (n-gram) speculative decoding config — V1 engine takes a dict.
    spec_decode_config = None
    if args.speculative_ngram:
        spec_decode_config = {
            "method": "ngram",
            "num_speculative_tokens": args.speculative_ngram,
            "prompt_lookup_max": args.ngram_max,
            "prompt_lookup_min": args.ngram_min,
        }
        logger.info("speculative_config=%s", spec_decode_config)

    t0 = time.monotonic()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=tp,
        max_model_len=args.max_model_len,
        enable_prefix_caching=args.prefix_caching,
        speculative_config=spec_decode_config,
    )
    logger.info("enable_prefix_caching=%s", args.prefix_caching)
    tokenizer = llm.get_tokenizer()
    logger.info("Engine loaded in %.1fs", time.monotonic() - t0)

    spec = get_spec(args.spec)
    system_message, template = spec.system_message, spec.extraction_template

    # --- Inputs: real HTML from the first WARC(s) until we have enough docs. ---
    warcs = _load_manifest(args.manifest)
    records = []
    for warc in warcs:
        recs = _filter_by_length(_download_one_warc(warc), MAX_DOC_TOKENS)
        records.extend(recs)
        logger.info("Downloaded %s -> %d usable records (cumulative %d)", warc, len(recs), len(records))
        if len(records) >= args.num_docs:
            break
    records = records[: args.num_docs]
    if not records:
        raise RuntimeError("No usable records downloaded — cannot benchmark.")

    prompt_ids_list = _build_prompt_ids(records, tokenizer, template, system_message)
    prompt_lens = [len(p) for p in prompt_ids_list]
    mean_prompt = sum(prompt_lens) / len(prompt_lens)
    logger.info(
        "Built %d prompts | prompt tokens: mean=%.0f min=%d max=%d",
        len(prompt_ids_list),
        mean_prompt,
        min(prompt_lens),
        max(prompt_lens),
    )

    # Warm both compile graphs on a length-diverse subset (shortest + longest + some middle)
    # so the timed runs measure steady-state, not first-shape compilation.
    order = sorted(range(len(prompt_ids_list)), key=lambda i: prompt_lens[i])
    k = min(args.warmup_docs, len(order))
    warm_idx = sorted({order[i] for i in range(0, len(order), max(1, len(order) // k))} | {order[0], order[-1]})
    warm_prompts = [{"prompt_token_ids": prompt_ids_list[i]} for i in warm_idx]

    def tp_prompt(ids):
        return {"prompt_token_ids": ids}

    gen_sp = SamplingParams(temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS)

    logger.info("=== warmup: generation (%d docs) ===", len(warm_prompts))
    llm.generate(warm_prompts, gen_sp)
    # The logprobs paths are orthogonal to spec decode (and can conflict with it), so only
    # warm/run them in the non-speculative benchmark.
    if not spec_decode_config:
        logger.info("=== warmup: 1-token classifier (%d docs) ===", len(warm_prompts))
        llm.generate(warm_prompts, SamplingParams(temperature=0.0, max_tokens=1, logprobs=20))

    # --- Timed full-generation pass (production setting) ---
    gen_prompts = [tp_prompt(p) for p in prompt_ids_list]
    t = time.monotonic()
    gen_out = llm.generate(gen_prompts, gen_sp)
    gen_elapsed = time.monotonic() - t
    gen_out_tokens = sum(len(o.outputs[0].token_ids) for o in gen_out)
    gen_texts = [_clean_text(o.outputs[0].text) for o in gen_out]
    actually_no_useful = [MARKER in t or len(t) < 1 for t in gen_texts]
    n_no_useful = sum(actually_no_useful)
    logger.info(
        "GENERATION: %d docs in %.1fs = %.3f docs/s | %d out-tok = %.0f tok/s | %d/%d -> %s",
        len(gen_prompts),
        gen_elapsed,
        len(gen_prompts) / gen_elapsed,
        gen_out_tokens,
        gen_out_tokens / gen_elapsed,
        n_no_useful,
        len(gen_prompts),
        MARKER,
    )

    def _write_summary(summary):
        print("\n========== BENCHMARK SUMMARY ==========")
        print(json.dumps(summary, indent=2, default=str))
        print("=======================================\n")
        result_path = args.result_path or (
            "gs://marin-us-east5/documents/benchmark_extraction/logprob_vs_gen/qwen3_1p7b_s5063_summary.json"
        )
        try:
            import fsspec

            with fsspec.open(result_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info("Wrote summary to %s", result_path)
        except Exception as e:
            logger.warning("Failed to write summary to %s: %s", result_path, e)

    # Speculative-decoding benchmark is generation-only — emit a focused summary and stop.
    # Compare its tok/s against the no-spec baseline (e.g. v7). The content docs (long verbatim
    # copies from the HTML) are where prompt-lookup should pay off; junk docs barely generate.
    if spec_decode_config:
        content_out_tokens = sum(
            len(o.outputs[0].token_ids) for o, nu in zip(gen_out, actually_no_useful) if not nu
        )
        _write_summary(
            {
                "model": args.model,
                "spec": args.spec,
                "mode": "speculative_ngram",
                "speculative_config": {
                    "method": "ngram",
                    "num_speculative_tokens": args.speculative_ngram,
                    "prompt_lookup_max": args.ngram_max,
                    "prompt_lookup_min": args.ngram_min,
                },
                "enable_prefix_caching": args.prefix_caching,
                "num_docs": len(prompt_ids_list),
                "prompt_tokens": {"mean": mean_prompt, "min": min(prompt_lens), "max": max(prompt_lens)},
                "generation": {
                    "elapsed_s": gen_elapsed,
                    "docs_per_s": len(gen_prompts) / gen_elapsed,
                    "output_tokens": gen_out_tokens,
                    "output_tok_per_s": gen_out_tokens / gen_elapsed,
                    "n_no_useful": n_no_useful,
                    "content_output_tokens": content_out_tokens,
                    "content_docs": len(gen_prompts) - n_no_useful,
                },
            }
        )
        return

    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    # --- Control: does prompt_logprobs work AT ALL? A short prompt removes any
    # length/cache confound. A functional build returns one dict per prompt token. ---
    tiny_ids = tokenizer.encode("The capital of France is Paris.")
    tiny_pls = llm.generate(
        [tp_prompt(tiny_ids)], SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    )[0].prompt_logprobs
    prompt_logprobs_control = {
        "tiny_prompt_tokens": len(tiny_ids),
        "len_prompt_logprobs": (len(tiny_pls) if tiny_pls is not None else None),
        "n_non_none_entries": (sum(1 for e in tiny_pls if e) if tiny_pls is not None else None),
    }
    prompt_logprobs_functional = bool(
        tiny_pls is not None and len(tiny_pls) == len(tiny_ids) and any(tiny_pls)
    )
    logger.info("prompt_logprobs control=%s functional=%s", prompt_logprobs_control, prompt_logprobs_functional)

    # --- First-token classifier via SUPPORTED generation logprobs ---
    # The model emits a DETERMINISTIC scaffold (empty <think></think> + the DSPy field header
    # "[[ ## text ## ]]\n") before the real keep/drop decision — so the token right after
    # </think> is an identical "\n\n" for every doc and carries no signal. Recover the scaffold
    # as the longest common prefix of all generated token streams; the FIRST token after it is
    # the branch point (marker vs content). Prefill doc+scaffold, emit ONE token with logprobs=k.
    marker_ids = tokenizer.encode(MARKER, add_special_tokens=False)
    gen_tok = [list(o.outputs[0].token_ids) for o in gen_out]

    def _lcp_len(seqs):
        m = min((len(s) for s in seqs), default=0)
        k = 0
        while k < m and len({s[k] for s in seqs}) == 1:
            k += 1
        return k

    scaffold_len = _lcp_len(gen_tok)
    scaffold_ids = gen_tok[0][:scaffold_len] if gen_tok else []
    # Junk-signal token = the branch token emitted by docs that actually produced the marker.
    junk_branch = Counter(
        s[scaffold_len] for s, nu in zip(gen_tok, actually_no_useful) if nu and len(s) > scaffold_len
    )
    junk_signal_token = junk_branch.most_common(1)[0][0] if junk_branch else marker_ids[0]
    clf_ids_list = [prompt_ids_list[i] + scaffold_ids for i in range(len(gen_tok))]

    clf_sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    clf_prompts = [tp_prompt(p) for p in clf_ids_list]
    t = time.monotonic()
    clf_out = llm.generate(clf_prompts, clf_sp)
    clf_elapsed = time.monotonic() - t
    logger.info(
        "CLASSIFIER(1-tok): %d docs in %.1fs = %.3f docs/s",
        len(clf_prompts),
        clf_elapsed,
        len(clf_prompts) / clf_elapsed,
    )

    # Read first-token logprobs: branch token + logprob assigned to the junk-signal token.
    pred_junk = []
    marker_first_logprobs = []
    clf_debug = None
    for o in clf_out:
        first = o.outputs[0]
        tok0 = first.token_ids[0]
        lp0 = first.logprobs[0] if first.logprobs else None
        pred_junk.append(tok0 == junk_signal_token)
        if lp0 and junk_signal_token in lp0:
            v = lp0[junk_signal_token]
            marker_first_logprobs.append(v.logprob if hasattr(v, "logprob") else float(v))
        else:
            marker_first_logprobs.append(None)
        if clf_debug is None:
            clf_debug = {
                "scaffold_len": scaffold_len,
                "scaffold_decoded": tokenizer.decode(scaffold_ids)[-120:],
                "junk_signal_token": junk_signal_token,
                "junk_signal_decoded": tokenizer.decode([junk_signal_token]),
                "first_token_id": tok0,
                "marker_first_id": marker_ids[0],
                "lp0_present": lp0 is not None,
                "lp0_keys_sample": (list(lp0)[:8] if lp0 else None),
            }

    # Quality: does the cheap 1-token probe agree with the full-generation NO_USEFUL verdict?
    tp_ = sum(1 for p, a in zip(pred_junk, actually_no_useful) if p and a)
    tn_ = sum(1 for p, a in zip(pred_junk, actually_no_useful) if not p and not a)
    fp_ = sum(1 for p, a in zip(pred_junk, actually_no_useful) if p and not a)
    fn_ = sum(1 for p, a in zip(pred_junk, actually_no_useful) if not p and a)
    n = len(pred_junk)
    lp_first_no_useful = _mean([lp for lp, nu in zip(marker_first_logprobs, actually_no_useful) if nu])
    lp_first_content = _mean([lp for lp, nu in zip(marker_first_logprobs, actually_no_useful) if not nu])

    summary = {
        "model": args.model,
        "spec": args.spec,
        "tpu_devices": len(devices),
        "tensor_parallel_size": tp,
        "enable_prefix_caching": args.prefix_caching,
        "num_docs": len(prompt_ids_list),
        "prompt_tokens": {"mean": mean_prompt, "min": min(prompt_lens), "max": max(prompt_lens)},
        "prompt_logprobs_functional": prompt_logprobs_functional,
        "prompt_logprobs_control": prompt_logprobs_control,
        "generation": {
            "elapsed_s": gen_elapsed,
            "docs_per_s": len(gen_prompts) / gen_elapsed,
            "output_tokens": gen_out_tokens,
            "output_tok_per_s": gen_out_tokens / gen_elapsed,
            "n_no_useful": n_no_useful,
        },
        "first_token_classifier": {
            "elapsed_s": clf_elapsed,
            "docs_per_s": len(clf_prompts) / clf_elapsed,
            "speedup_over_generation": gen_elapsed / clf_elapsed if clf_elapsed else None,
            "accuracy_vs_full_generation": (tp_ + tn_) / n if n else None,
            "confusion": {"tp": tp_, "tn": tn_, "fp": fp_, "fn": fn_},
            "mean_marker_first_logprob_when_no_useful": lp_first_no_useful,
            "mean_marker_first_logprob_when_content": lp_first_content,
            "note": "pred_junk = (first answer token == marker's first token '['); compared to full-gen verdict",
        },
        "first_token_classifier_debug": clf_debug,
    }

    _write_summary(summary)


if __name__ == "__main__":
    main()
