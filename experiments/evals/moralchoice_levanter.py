# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MoralChoice log-likelihood eval on TPU via Levanter (the standard marin JAX stack).

Same evaluation + steerability design as ``moralchoice_loglik`` (which it reuses for
all the pure prompt-building and metric logic), but the model forward runs through
Levanter/haliax instead of torch — so it uses the JAX + libtpu stack already present
in the marin env (no bespoke torch_xla install).

The loglik computation mirrors levanter.eval_harness's worker: logits from
``model(tokens, attn_mask=causal)``, log-softmax over Vocab, then gather the logprob
of each next token (``target = roll(tokens, -1)``) and sum over the continuation span.

Run via Iris in-region with the checkpoint:
    uv run iris --cluster marin job run --enable-extra-resources --tpu v6e-4 \
        --region europe-west4 --memory 128GB --disk 64GB --extra tpu \
        --priority interactive --no-wait --job-name mc-eval-full \
        -e WANDB_API_KEY ... -e HF_TOKEN ... \
        -- python -m experiments.evals.moralchoice_levanter \
            --model gs://marin-eu-west4/.../hf/step-56002 \
            --output gs://marin-eu-west4/metadata/moralchoice_evals/dclm10k-2e19 \
            --steer-rules death pain disable freedom pleasure deceive cheat break_promise break_law duty
"""

from __future__ import annotations

import argparse
import json
import logging

import fsspec

from experiments.evals.moralchoice_loglik import (
    RULE_COLUMNS,
    SCENARIO_BASE,
    _prefers_action1,
    batched,
    build_requests,
    compute_metrics,
    stage_model,
    steer_preamble,
)

logger = logging.getLogger(__name__)


def make_score_fn(model, tokenizer, max_length: int, compute_mapping):
    """Return score(reqs) -> list[(sum_logprob, num_continuation_tokens)] using a Levanter forward."""
    import haliax as hax
    import jax.numpy as jnp
    import numpy as np
    from levanter.layers.attention import AttentionMask

    Vocab = model.Vocab
    pos_name = model.Pos.name
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def _forward(tokens, weight):
        logits = model(tokens, attn_mask=AttentionMask.causal())  # (Batch, Pos, Vocab)
        logprobs = hax.nn.log_softmax(logits, axis=Vocab)
        Pos = tokens.resolve_axis(pos_name)
        target = hax.roll(tokens, -1, Pos)  # target[t] = token at t+1
        # logprob of the actual next token = sum over Vocab of logprobs * one_hot(target).
        onehot = hax.nn.one_hot(target, Vocab, dtype=logprobs.dtype)
        tok_lp = hax.sum(logprobs * onehot, axis=Vocab)  # (Batch, Pos)
        return hax.sum(tok_lp * weight, axis=Pos)  # (Batch,)

    forward = hax.named_jit(_forward, axis_resources=compute_mapping)

    def score(reqs):
        b = len(reqs)
        ids = np.full((b, max_length), pad_id, dtype=np.int32)
        w = np.zeros((b, max_length), dtype=np.float32)
        ntok: list[int] = []
        for i, r in enumerate(reqs):
            p = tokenizer(r.prompt, add_special_tokens=True)["input_ids"]
            c = tokenizer(r.continuation, add_special_tokens=False)["input_ids"]
            if len(p) + len(c) > max_length:
                raise ValueError(f"sequence len {len(p) + len(c)} exceeds max_length {max_length}")
            ids[i, : len(p) + len(c)] = (p + c)[: len(p) + len(c)]
            # logprob of continuation token at abs position j comes from the model output at j-1.
            w[i, len(p) - 1 : len(p) + len(c) - 1] = 1.0
            ntok.append(len(c))
        Batch = hax.Axis("batch", b)
        Pos = hax.Axis(pos_name, max_length)
        tokens = hax.named(jnp.asarray(ids), (Batch, Pos))
        weight = hax.named(jnp.asarray(w), (Batch, Pos))
        sums = np.asarray(forward(tokens, weight).array).tolist()
        return list(zip(sums, ntok, strict=True))

    return score


def get_preferences(score_fn, rows, variant, system_prompt, batch_size) -> dict:
    """Score every scenario; return sid -> per-order booleans 'prefers action1' (sum & norm)."""
    reqs = []
    for row in rows:
        reqs += build_requests(row, variant, system_prompt)
    scores: dict[tuple, tuple[float, int]] = {}
    for chunk in batched(reqs, batch_size):
        padded = chunk + [chunk[-1]] * (batch_size - len(chunk))  # fixed batch shape for XLA
        results = score_fn(padded)
        for r, sc in zip(chunk, results[: len(chunk)], strict=True):
            scores[(r.scenario_id, r.order, r.choice_action)] = sc

    prefs: dict[str, dict] = {}
    for row in rows:
        sid = row["scenario_id"]
        entry = {"sum": {}, "norm": {}}
        for order in ("orig", "swap"):
            lp1, n1 = scores[(sid, order, 1)]
            lp2, n2 = scores[(sid, order, 2)]
            entry["sum"][order] = lp1 > lp2
            entry["norm"][order] = (lp1 / n1) > (lp2 / n2)
        prefs[sid] = entry
    return prefs


def load_model(model_path: str, param_mapping):
    import jax.numpy as jnp
    from levanter.compat.hf_checkpoints import HFCheckpointConverter

    local = stage_model(model_path)
    converter = HFCheckpointConverter.from_hf(local)
    config = converter.config_from_hf_checkpoint(local)
    tokenizer = converter.tokenizer
    model = converter.load_pretrained(config.model_type, ref=local, dtype=jnp.bfloat16, axis_mapping=param_mapping)
    return model, tokenizer


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--variants", nargs="+", default=["ab", "repeat"], choices=["ab", "repeat"])
    p.add_argument("--splits", nargs="+", default=["low", "high"], choices=["low", "high"])
    p.add_argument("--steer-rules", nargs="*", default=[], choices=list(RULE_COLUMNS))
    p.add_argument(
        "--steer-format",
        choices=["plain", "native"],
        default="plain",
        help="How the steering instruction is rendered. 'plain' prepends it as text (default); "
        "'native' wraps it in the llama3 system channel "
        "(<|start_header_id|>system<|end_header_id|>\\n\\n{instr}<|eot_id|>) used to pretrain the "
        "sysprompt-conditioned model.",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--dump-prefs",
        default=None,
        help="Optional path (local or gs://) to write per-(split,variant,condition,scenario) "
        "order-averaged action1 preference, for inspecting individual flips.",
    )
    args = p.parse_args(argv)

    import jmp
    from datasets import load_dataset
    from levanter.tracker import NoopConfig
    from levanter.trainer import TrainerConfig

    trainer = TrainerConfig(
        tracker=NoopConfig(), mp=jmp.get_policy("p=bfloat16,c=bfloat16"), per_device_eval_parallelism=1
    )
    trainer.initialize()
    data = load_dataset(
        "csv",
        data_files={
            "low": f"{SCENARIO_BASE}/moralchoice_low_ambiguity.csv",
            "high": f"{SCENARIO_BASE}/moralchoice_high_ambiguity.csv",
        },
    )

    def render_steer(rule: str) -> str:
        instr = steer_preamble(rule).strip()
        if args.steer_format == "native":
            # Match the [S][D] pretraining channel: system header block, then the prompt follows.
            return f"<|start_header_id|>system<|end_header_id|>\n\n{instr}<|eot_id|>"
        return steer_preamble(rule)

    conditions = [("neutral", "")] + [(rule, render_steer(rule)) for rule in args.steer_rules]
    results: dict = {
        "model": args.model,
        "steer_format": args.steer_format,
        "conditions": [c for c, _ in conditions],
        "splits": {},
    }

    dump: list[dict] = []
    with trainer.use_device_mesh():
        logger.info("Loading model %s via Levanter", args.model)
        model, tokenizer = load_model(args.model, trainer.parameter_axis_mapping)
        score_fn = make_score_fn(model, tokenizer, args.max_length, trainer.compute_axis_mapping)
        _run_all(args, data, conditions, score_fn, results, dump)

    out_path = args.output.rstrip("/") + "/results.json"
    with fsspec.open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", out_path)

    if args.dump_prefs:
        with fsspec.open(args.dump_prefs.rstrip("/") + "/prefs.json", "w") as f:
            json.dump(dump, f)
        logger.info("Wrote per-scenario prefs (%d records)", len(dump))


def _run_all(args, data, conditions, score_fn, results, dump) -> None:
    for split in args.splits:
        rows = list(data[split])
        if args.limit:
            rows = rows[: args.limit]
        results["splits"][split] = {}
        for variant in args.variants:
            per_condition: dict[str, dict] = {}
            for cond_name, system_prompt in conditions:
                logger.info("split=%s variant=%s condition=%s (%d scenarios)", split, variant, cond_name, len(rows))
                prefs = get_preferences(score_fn, rows, variant, system_prompt, args.batch_size)
                m = compute_metrics(rows, prefs)
                per_condition[cond_name] = m
                if dump is not None:
                    for row in rows:
                        sid = row["scenario_id"]
                        dump.append(
                            {
                                "split": split,
                                "variant": variant,
                                "condition": cond_name,
                                "scenario_id": sid,
                                "prefers_action1_norm": _prefers_action1(prefs[sid], "norm"),
                            }
                        )
                logger.info(
                    "  acc=%.3f acc_norm=%.3f consistency=%.3f first_pos_bias=%.3f",
                    m["acc"],
                    m["acc_norm"],
                    m["consistency"],
                    m["first_position_bias"],
                )
            steerability: dict[str, dict] = {}
            base = per_condition["neutral"]["compliance"]
            for rule in args.steer_rules:
                nc = base[rule]["prefer_compliant"]
                sc = per_condition[rule]["compliance"][rule]["prefer_compliant"]
                steerability[rule] = {
                    "n": base[rule]["n"],
                    "neutral": nc,
                    "steered": sc,
                    "delta": (sc - nc) if (nc is not None and sc is not None) else None,
                }
                if steerability[rule]["delta"] is not None:
                    logger.info(
                        "  steerability[%s] neutral=%.3f steered=%.3f delta=%+.3f (n=%d)",
                        rule,
                        nc,
                        sc,
                        steerability[rule]["delta"],
                        base[rule]["n"],
                    )
            results["splits"][split][variant] = {"conditions": per_condition, "steerability": steerability}


if __name__ == "__main__":
    main()
