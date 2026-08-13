# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score the distilled extractor LLMs as useful-vs-not classifiers via the marker logprob.

Follows ``bench_logprob_vs_gen.py``: the no-think hq-distill models emit a DETERMINISTIC scaffold
(empty ``<think></think>`` + the DSPy ``[[ ## text ## ]]`` header) before the keep/drop branch. We
prefill ``doc + scaffold``, emit ONE token with ``logprobs``, and read the logprob the model assigns
to the ``[NO_USEFUL_CONTENT]`` marker's first token at that branch point. That logprob is the
classifier scalar (higher = the model wants to abstain = "not useful"); a fully tunable threshold,
analogous to the BERT/fastText prob columns.

Input = the sample's ``stripped_html`` (body_strip; same preprocessing the BERT/fastText classifiers
used) fed through the ``high_quality`` extraction prompt — no WARC re-decode needed.

Writes ``model_scores/<col>/part-i-of-N.parquet`` so score_modernbert_useful's ``join`` adds the
column. Runs as a vLLM-TPU job in us-east5 (where the models + sample live), shardable like the
extraction::

    iris --cluster marin job run --region us-east5 --tpu v6e-4 --enable-extra-resources \\
      --memory 128GB --disk 100GB --extra vllm --extra tpu --priority interactive --no-wait \\
      -e WANDB_API_KEY ... -e HF_TOKEN ... -- \\
      python -m experiments.baseline_collection.score_llm_logprob_useful \\
        --model llm_logprob_marker_1p7b --num-shards 8 --shard-idx 0
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections import Counter

import fsspec

from experiments.baseline_collection.extraction_specs import get_spec
from experiments.baseline_collection.run_extract_standalone import MAX_DOC_TOKENS, MAX_OUTPUT_TOKENS
from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)

OUT_ROOT = "gs://marin-us-east5/documents/extractor_compare/high_quality_200warc"
SCORES_DIR = f"{OUT_ROOT}/model_scores"
SAMPLE_DIR = f"{OUT_ROOT}/sample_100k"
SPEC = "high_quality"
MARKER = "[NO_USEFUL_CONTENT]"
WARMUP_DOCS = 96  # full generations to recover the scaffold + marker branch token (deterministic)

# col -> model checkpoint (the same distilled extractors that produced text_1p7b / text_0p6b).
MODELS: dict[str, str] = {
    "llm_logprob_marker_1p7b": (
        "gs://marin-us-east5/checkpoints/"
        "qwen3-1.7b-hq-distill-bal350k-proxy100pct-lr7e-6-bs128-mhfix-27b106/hf/step-5063"
    ),
    "llm_logprob_marker_0p6b": (
        "gs://marin-us-east5/checkpoints/qwen3-0.6b-hq-distill-bal350k-lr2e-6-bs64-d670f5/hf/step-10127"
    ),
}


def _read_sample(num_shards: int, shard_idx: int) -> tuple[list[str], list[str]]:
    """(warc_record_id, stripped_html) for this shard's contiguous slice of the 100k sample."""
    import math

    import pyarrow.parquet as pq

    ids: list[str] = []
    htmls: list[str] = []
    for path in sorted(fsspec_glob(f"{SAMPLE_DIR}/*.parquet")):
        with fsspec.open(path, "rb") as fh:
            t = pq.ParquetFile(fh).read(columns=["warc_record_id", "stripped_html"])
        ids.extend(t.column("warc_record_id").to_pylist())
        htmls.extend(h or "" for h in t.column("stripped_html").to_pylist())
    if num_shards > 1:
        chunk = math.ceil(len(ids) / num_shards)
        s, e = shard_idx * chunk, min((shard_idx + 1) * chunk, len(ids))
        ids, htmls = ids[s:e], htmls[s:e]
        logger.info("shard %d/%d: docs [%d:%d] = %d", shard_idx, num_shards, s, e, len(ids))
    logger.info("loaded %d docs", len(ids))
    return ids, htmls


def _build_prompt_ids(
    htmls: list[str], tokenizer, template: str, system_message: str | None, html_budget: int = MAX_DOC_TOKENS
) -> list[list[int]]:
    """Replicate run_extract_standalone prompt construction; truncate each HTML to html_budget tokens."""
    out = []
    for html in htmls:
        tokens = tokenizer.encode(html)
        if len(tokens) > html_budget:
            html = tokenizer.decode(tokens[:html_budget])
        text = template.format(example=html)
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": text})
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out.append(tokenizer.encode(prompt_text))
    return out


def _html_budget_for_ctx(tokenizer, template: str, system_message: str | None, max_ctx: int) -> int:
    """Tokens of HTML that fit so the full prompt (+scaffold) lands at ~max_ctx.

    overhead = the prompt with empty HTML (system + template boilerplate + chat wrapper); reserve a
    small margin for the scaffold (~empty <think></think> + DSPy header) + tokenization boundary slop.
    """
    overhead = len(_build_prompt_ids([""], tokenizer, template, system_message, html_budget=0)[0])
    budget = max_ctx - overhead - 48
    logger.info("ctx=%d: prompt overhead=%d tokens -> html budget=%d tokens", max_ctx, overhead, budget)
    if budget < 256:
        raise ValueError(f"max-ctx {max_ctx} too small for overhead {overhead}")
    return budget


def _lcp_len(seqs: list[list[int]]) -> int:
    m = min((len(s) for s in seqs), default=0)
    k = 0
    while k < m and len({s[k] for s in seqs}) == 1:
        k += 1
    return k


def _scaffold_and_branch(llm, prompt_ids: list[list[int]], marker_first_id: int):
    """Generate on a warmup subset to recover the deterministic scaffold + the marker branch token.

    Returns (scaffold_ids, junk_signal_token) — junk_signal_token is the first token docs emit when
    they go down the [NO_USEFUL_CONTENT] branch (== marker's first token unless tokenization differs).
    """
    from vllm import SamplingParams

    warm = prompt_ids[: min(WARMUP_DOCS, len(prompt_ids))]
    gen = llm.generate(
        [{"prompt_token_ids": p} for p in warm], SamplingParams(temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS)
    )
    gen_tok = [list(o.outputs[0].token_ids) for o in gen]
    texts = [o.outputs[0].text for o in gen]
    no_useful = [MARKER in t or len(t.strip()) < 1 for t in texts]
    scaffold_len = _lcp_len(gen_tok)
    scaffold_ids = gen_tok[0][:scaffold_len] if gen_tok else []
    branch = Counter(s[scaffold_len] for s, nu in zip(gen_tok, no_useful, strict=True) if nu and len(s) > scaffold_len)
    junk = branch.most_common(1)[0][0] if branch else marker_first_id
    logger.info(
        "scaffold_len=%d junk_signal_token=%d; warmup no_useful=%d/%d", scaffold_len, junk, sum(no_useful), len(warm)
    )
    return scaffold_ids, junk


def run_score(
    col: str,
    num_shards: int,
    shard_idx: int,
    limit: int | None,
    tp: int | None,
    gen_timing: bool = False,
    max_ctx: int | None = None,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if col not in MODELS:
        raise ValueError(f"unknown --model {col!r}; choices: {list(MODELS)}")
    model_path = MODELS[col]

    marin_pfx = os.environ.get("MARIN_PREFIX")
    cache_dir = os.path.join(marin_pfx, "compilation-cache") if marin_pfx else "/tmp/marin-jax-compilation-cache"
    os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", cache_dir)
    os.environ.setdefault("VLLM_XLA_CACHE_PATH", cache_dir)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import jax

    devices = jax.devices()
    tp = tp or len([d for d in devices if d.platform == "tpu"]) or len(devices)
    logger.info("col=%s model=%s tp=%d devices=%s", col, model_path, tp, devices)

    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, tensor_parallel_size=tp, max_model_len=32768, enable_prefix_caching=True)
    tokenizer = llm.get_tokenizer()
    spec = get_spec(SPEC)

    ids, htmls = _read_sample(num_shards, shard_idx)
    if limit is not None:
        ids, htmls = ids[:limit], htmls[:limit]
        logger.info("LIMIT %d (smoke)", len(ids))
    html_budget = (
        _html_budget_for_ctx(tokenizer, spec.extraction_template, spec.system_message, max_ctx)
        if max_ctx is not None
        else MAX_DOC_TOKENS
    )
    out_col = col if max_ctx is None else f"{col}_ctx{max_ctx // 1024}k"  # e.g. llm_logprob_marker_0p6b_ctx4k
    prompt_ids = _build_prompt_ids(htmls, tokenizer, spec.extraction_template, spec.system_message, html_budget)

    if gen_timing:
        # Full-generation timing ONLY (nothing written): the production extract path, to compare
        # docs/chip/s against the 1-token marker classifier on the same (long) sample docs.
        t = time.monotonic()
        gen = llm.generate(
            [{"prompt_token_ids": p} for p in prompt_ids],
            SamplingParams(temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS),
        )
        el = time.monotonic() - t
        out_toks = sum(len(o.outputs[0].token_ids) for o in gen)
        logger.info(
            "GEN-TIMING %s: %d docs in %.1fs = %.2f docs/s = %.3f docs/chip/s | %d out-tok = %.0f tok/s",
            col,
            len(prompt_ids),
            el,
            len(prompt_ids) / el,
            len(prompt_ids) / el / tp,
            out_toks,
            out_toks / el,
        )
        return

    marker_first_id = tokenizer.encode(MARKER, add_special_tokens=False)[0]
    scaffold_ids, junk = _scaffold_and_branch(llm, prompt_ids, marker_first_id)

    # Classifier: prefill prompt + scaffold, emit ONE token with top-k logprobs; read the logprob
    # assigned to the marker branch token (None if it's not in the top-20 -> very confident "useful").
    clf_prompts = [{"prompt_token_ids": p + scaffold_ids} for p in prompt_ids]
    t = time.monotonic()
    out = llm.generate(clf_prompts, SamplingParams(temperature=0.0, max_tokens=1, logprobs=20))
    elapsed = time.monotonic() - t
    logger.info("classified %d docs in %.1fs = %.2f docs/s", len(clf_prompts), elapsed, len(clf_prompts) / elapsed)

    logprobs: list[float | None] = []
    for o in out:
        lp0 = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else None
        if lp0 and junk in lp0:
            v = lp0[junk]
            logprobs.append(v.logprob if hasattr(v, "logprob") else float(v))
        else:
            logprobs.append(None)

    out_path = f"{SCORES_DIR}/{out_col}/part-{shard_idx:03d}-of-{num_shards:03d}.parquet"
    if limit is not None:
        out_path = f"{SCORES_DIR}/{out_col}_smoke.parquet"
    table = pa.table({"warc_record_id": ids, out_col: pa.array(logprobs, type=pa.float32())})
    with fsspec.open(out_path, "wb") as fh:
        pq.write_table(table, fh)
    present = sum(1 for x in logprobs if x is not None)
    logger.info("wrote %d -> %s (marker-logprob present for %d/%d)", len(ids), out_path, present, len(ids))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Smoke: score only the first N docs.")
    p.add_argument("--tp", type=int, default=None, help="Tensor parallel size (default: TPU device count).")
    p.add_argument(
        "--gen-timing",
        action="store_true",
        help="Full-generation timing only (writes nothing); for the speed comparison.",
    )
    p.add_argument(
        "--max-ctx",
        type=int,
        default=None,
        help="Target total prompt tokens (e.g. 4096/8192/16384); HTML is truncated to fit. "
        "Output column is suffixed _ctx{N//1024}k. Default: full (MAX_DOC_TOKENS).",
    )
    args = p.parse_args()
    run_score(args.model, args.num_shards, args.shard_idx, args.limit, args.tp, args.gen_timing, args.max_ctx)


if __name__ == "__main__":
    main()
