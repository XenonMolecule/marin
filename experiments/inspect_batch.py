# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect training batches to diagnose NaN loss at specific steps.

This script reconstructs the same data pipeline used by the SFT experiments,
loads the exact batches the training loop would see at given steps, and prints
detailed statistics to help identify problematic data.

PRNG note: The SFT experiments set data_seed=<seed> in the config. In train_lm.py
this means data_key = jrandom.PRNGKey(data_seed), i.e. a direct key from the seed
(NOT split from the trainer seed). This script matches that behavior.

Usage (run on a machine with GCS access):

    # Inspect step 736 and compare with neighbors for the 0.6B model:
    uv run python experiments/inspect_batch.py \
        --steps 734 735 736 737 \
        --cache_path gs://marin-us-central2/tokenized/rephraser_small_check_0213_qwen3-61e2f5 \
        --tokenizer Qwen/Qwen3-0.6B \
        --seq_len 32768 \
        --batch_size 64 \
        --seed 42

    # For the 1.7B model (same tokenizer, same cache as 0.6B):
    uv run python experiments/inspect_batch.py \
        --steps 734 735 736 737 \
        --cache_path gs://marin-us-central2/tokenized/rephraser_small_check_0213_qwen3-61e2f5 \
        --tokenizer Qwen/Qwen3-0.6B \
        --seq_len 32768 \
        --batch_size 64 \
        --seed 42

    # For the 4B-Thinking model (different tokenizer/cache, needs --thinking flag):
    uv run python experiments/inspect_batch.py \
        --steps 734 735 736 737 \
        --cache_path gs://marin-us-central2/tokenized/rephraser_small_check_0213_qwen3_4b_thinking-95af45 \
        --tokenizer Qwen/Qwen3-4B-Thinking-2507 \
        --thinking \
        --seq_len 32768 \
        --batch_size 64 \
        --seed 42

    # For the 8B model:
    uv run python experiments/inspect_batch.py \
        --steps 734 735 736 737 \
        --cache_path gs://marin-us-central2/tokenized/rephraser_small_check_0213_qwen3_8b-45afaf \
        --tokenizer Qwen/Qwen3-0.6B \
        --seq_len 32768 \
        --batch_size 64 \
        --seed 42
"""

import argparse
import asyncio
import csv
import logging
import os

import jax.random as jrandom
import numpy as np
from haliax import Axis
from levanter.data.text import ChatLmDatasetFormat, DatasetComponent, LmDataConfig
from levanter.schedule import BatchSchedule
from levanter.utils.jax_utils import local_cpu_mesh

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _strip_version_hash(name: str) -> str:
    """Remove a trailing version hash suffix like '-61e2f5' from a path basename."""
    if "-" in name:
        prefix, suffix = name.rsplit("-", 1)
        if len(suffix) == 6 and all(c in "0123456789abcdef" for c in suffix):
            return prefix
    return name


def build_data_config(cache_path: str, tokenizer: str, chat_template: str) -> LmDataConfig:
    """Build an LmDataConfig that mirrors what the experiment files create.

    The experiment's lm_data_config() creates two components pointing to the same
    cache: one for training (weight=1.0) and one for validation (weight=0.0).
    Only the training component is used during train_set(); the validation component
    is skipped because its weight is 0.

    The chat_template is needed because load_lm_dataset_cache constructs a
    ChatProcessor to infer the cache exemplar, and ChatProcessor validates that
    the template contains {%generation%}.
    """
    component_name = _strip_version_hash(os.path.basename(cache_path.rstrip("/")))
    fmt = ChatLmDatasetFormat(chat_template=chat_template)

    train_component = DatasetComponent(source=None, cache_dir=cache_path, format=fmt, tags=[])
    val_component = DatasetComponent(source=None, cache_dir=cache_path, format=fmt, tags=[])
    return LmDataConfig(
        components={
            component_name: train_component,
            "rephraser_val": val_component,
        },
        train_weights={
            component_name: 1.0,
            "rephraser_val": 0.0,
        },
        tokenizer=tokenizer,
        cache_dir=None,
        shuffle=True,
        permutation_type="feistel",
        block_cross_document_attention=True,
    )


async def get_examples_for_step(
    dataset,
    batch_schedule: BatchSchedule,
    step: int,
):
    """Get the raw examples (pre-batching) that the DataLoader would serve at a given step."""
    offset = batch_schedule.global_data_offset_by_step(step)
    batch_size = batch_schedule.batch_size_at_step(step)
    indices = list(range(offset, offset + batch_size))

    logger.info(f"Step {step}: fetching indices [{offset}..{offset + batch_size - 1}] ({batch_size} examples)")
    examples = await dataset.get_batch(indices)
    return examples


def analyze_example(example, vocab_size: int | None, idx: int) -> dict:
    """Analyze a single LmExample and return statistics."""
    tokens = np.asarray(example.tokens.array)
    loss_weight = np.asarray(example.loss_weight.array)

    stats = {
        "idx": idx,
        "seq_len": tokens.shape[0],
        "num_tokens_with_loss": int(np.sum(loss_weight > 0)),
        "num_padding_tokens": int(np.sum(tokens == 0)),
        "loss_weight_sum": float(np.sum(loss_weight)),
        "loss_weight_mean": float(np.mean(loss_weight)),
        "token_min": int(np.min(tokens)),
        "token_max": int(np.max(tokens)),
        "has_nan_tokens": bool(np.any(np.isnan(tokens.astype(float)))),
        "has_negative_tokens": bool(np.any(tokens < 0)),
        "unique_token_count": int(np.unique(tokens).shape[0]),
        "fraction_with_loss": float(np.sum(loss_weight > 0) / tokens.shape[0]),
    }

    if vocab_size is not None:
        oov = tokens >= vocab_size
        stats["out_of_vocab_count"] = int(np.sum(oov))
        if np.any(oov):
            oov_ids = np.unique(tokens[oov])
            stats["out_of_vocab_ids"] = oov_ids.tolist()[:20]
            # Find positions of OOV tokens
            oov_positions = np.where(oov)[0]
            stats["out_of_vocab_positions"] = oov_positions.tolist()[:20]

    # Check for extremely long runs of the same token (potential corruption)
    diffs = np.diff(tokens)
    max_run = 0
    current_run = 1
    for d in diffs:
        if d == 0:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    stats["max_repeated_token_run"] = max_run

    # Check loss_weight pattern
    stats["loss_weight_all_zero"] = bool(np.all(loss_weight == 0))
    stats["loss_weight_has_nan"] = bool(np.any(np.isnan(loss_weight)))

    return stats


def print_step_summary(step: int, all_stats: list[dict]):
    """Print a summary of batch statistics for a given step."""
    print(f"\n{'='*80}")
    print(f"STEP {step} — {len(all_stats)} examples in batch")
    print(f"{'='*80}")

    # Aggregate stats
    tokens_with_loss = [s["num_tokens_with_loss"] for s in all_stats]
    padding_counts = [s["num_padding_tokens"] for s in all_stats]
    token_maxes = [s["token_max"] for s in all_stats]
    token_mins = [s["token_min"] for s in all_stats]
    loss_fractions = [s["fraction_with_loss"] for s in all_stats]
    max_runs = [s["max_repeated_token_run"] for s in all_stats]

    print(
        f"\n  Tokens with loss:     min={min(tokens_with_loss):>6}  max={max(tokens_with_loss):>6}  "
        f"mean={np.mean(tokens_with_loss):>8.1f}  median={np.median(tokens_with_loss):>8.1f}"
    )
    print(
        f"  Padding tokens:       min={min(padding_counts):>6}  max={max(padding_counts):>6}  "
        f"mean={np.mean(padding_counts):>8.1f}  median={np.median(padding_counts):>8.1f}"
    )
    print(
        f"  Loss fraction:        min={min(loss_fractions):>6.3f}  max={max(loss_fractions):>6.3f}  "
        f"mean={np.mean(loss_fractions):>8.3f}"
    )
    print(f"  Token ID range:       min={min(token_mins):>6}  max={max(token_maxes):>6}")
    print(f"  Max repeated run:     min={min(max_runs):>6}  max={max(max_runs):>6}  " f"mean={np.mean(max_runs):>8.1f}")

    # Flag problematic examples
    problems = []
    for s in all_stats:
        issues = []
        if s["loss_weight_all_zero"]:
            issues.append("ALL_ZERO_LOSS_WEIGHT")
        if s.get("out_of_vocab_count", 0) > 0:
            issues.append(f"OOV_TOKENS({s['out_of_vocab_count']})")
        if s["has_negative_tokens"]:
            issues.append("NEGATIVE_TOKEN_IDS")
        if s["loss_weight_has_nan"]:
            issues.append("NAN_LOSS_WEIGHT")
        if s["num_tokens_with_loss"] == 0:
            issues.append("NO_LOSS_TOKENS")
        if s["max_repeated_token_run"] > 1000:
            issues.append(f"LONG_REPEAT({s['max_repeated_token_run']})")
        if s["num_padding_tokens"] > s["seq_len"] * 0.99:
            issues.append("NEARLY_ALL_PADDING")
        if issues:
            problems.append((s["idx"], issues))

    if problems:
        print(f"\n  *** FLAGGED EXAMPLES ({len(problems)}/{len(all_stats)}) ***")
        for idx, issues in problems:
            print(f"    Example {idx}: {', '.join(issues)}")
    else:
        print("\n  No flagged examples.")

    # Print per-example details for flagged or unusual examples
    # Sort by loss fraction to find outliers
    sorted_stats = sorted(all_stats, key=lambda s: s["fraction_with_loss"])
    print("\n  Bottom 5 by loss fraction (least supervised signal):")
    for s in sorted_stats[:5]:
        print(
            f"    Example {s['idx']:>3}: loss_frac={s['fraction_with_loss']:.4f}  "
            f"tokens_with_loss={s['num_tokens_with_loss']:>5}  "
            f"padding={s['num_padding_tokens']:>5}  "
            f"token_range=[{s['token_min']}, {s['token_max']}]  "
            f"max_repeat={s['max_repeated_token_run']}"
        )

    print("\n  Top 5 by loss fraction (most supervised signal):")
    for s in sorted_stats[-5:]:
        print(
            f"    Example {s['idx']:>3}: loss_frac={s['fraction_with_loss']:.4f}  "
            f"tokens_with_loss={s['num_tokens_with_loss']:>5}  "
            f"padding={s['num_padding_tokens']:>5}  "
            f"token_range=[{s['token_min']}, {s['token_max']}]  "
            f"max_repeat={s['max_repeated_token_run']}"
        )

    return problems


def decode_example_segments(example, tokenizer) -> dict[str, str]:
    """Decode an example into user text (loss_weight=0) and assistant text (loss_weight>0).

    Returns a dict with keys:
      - assistant_text: decoded tokens where loss_weight > 0
      - user_text: decoded tokens where loss_weight == 0 (excluding padding)
      - full_text: the entire non-padding sequence decoded
    """
    tokens = np.asarray(example.tokens.array)
    loss_weight = np.asarray(example.loss_weight.array)

    # Find non-padding region
    non_pad = np.where(tokens != 0)[0]
    if len(non_pad) == 0:
        return {"assistant_text": "", "user_text": "", "full_text": ""}

    first_real = non_pad[0]
    last_real = non_pad[-1] + 1

    content_tokens = tokens[first_real:last_real]
    content_loss = loss_weight[first_real:last_real]

    # Decode full non-padding sequence
    full_text = tokenizer.decode(content_tokens.tolist(), skip_special_tokens=False)

    # Decode assistant segments (loss_weight > 0)
    assistant_mask = content_loss > 0
    if np.any(assistant_mask):
        assistant_token_ids = content_tokens[assistant_mask].tolist()
        assistant_text = tokenizer.decode(assistant_token_ids, skip_special_tokens=False)
    else:
        assistant_text = ""

    # Decode user segments (loss_weight == 0, excluding padding)
    user_mask = content_loss == 0
    if np.any(user_mask):
        user_token_ids = content_tokens[user_mask].tolist()
        user_text = tokenizer.decode(user_token_ids, skip_special_tokens=False)
    else:
        user_text = ""

    return {
        "assistant_text": assistant_text,
        "user_text": user_text,
        "full_text": full_text,
    }


def flags_for_example(stats: dict) -> list[str]:
    """Return a list of flag strings for an example based on its stats."""
    flags = []
    if stats["loss_weight_all_zero"]:
        flags.append("ALL_ZERO_LOSS_WEIGHT")
    if stats.get("out_of_vocab_count", 0) > 0:
        flags.append(f"OOV_TOKENS({stats['out_of_vocab_count']})")
    if stats["has_negative_tokens"]:
        flags.append("NEGATIVE_TOKEN_IDS")
    if stats["loss_weight_has_nan"]:
        flags.append("NAN_LOSS_WEIGHT")
    if stats["num_tokens_with_loss"] == 0:
        flags.append("NO_LOSS_TOKENS")
    if stats["max_repeated_token_run"] > 1000:
        flags.append(f"LONG_REPEAT({stats['max_repeated_token_run']})")
    if stats["num_padding_tokens"] > stats["seq_len"] * 0.99:
        flags.append("NEARLY_ALL_PADDING")
    return flags


def write_csv(
    output_path: str,
    step_data: dict[int, list[tuple[dict, dict[str, str]]]],
):
    """Write per-example data to a CSV file.

    Args:
        output_path: Path to write the CSV file.
        step_data: Mapping from step number to list of (stats_dict, decoded_segments) tuples.
    """
    fieldnames = [
        "step",
        "example_idx",
        "seq_len",
        "num_tokens_with_loss",
        "fraction_with_loss",
        "num_padding_tokens",
        "max_repeated_token_run",
        "token_min",
        "token_max",
        "unique_token_count",
        "out_of_vocab_count",
        "flags",
        "assistant_text",
        "user_text",
        "full_text",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for step in sorted(step_data.keys()):
            for stats, segments in step_data[step]:
                row = {
                    "step": step,
                    "example_idx": stats["idx"],
                    "seq_len": stats["seq_len"],
                    "num_tokens_with_loss": stats["num_tokens_with_loss"],
                    "fraction_with_loss": f"{stats['fraction_with_loss']:.6f}",
                    "num_padding_tokens": stats["num_padding_tokens"],
                    "max_repeated_token_run": stats["max_repeated_token_run"],
                    "token_min": stats["token_min"],
                    "token_max": stats["token_max"],
                    "unique_token_count": stats["unique_token_count"],
                    "out_of_vocab_count": stats.get("out_of_vocab_count", 0),
                    "flags": "|".join(flags_for_example(stats)),
                    "assistant_text": segments["assistant_text"],
                    "user_text": segments["user_text"],
                    "full_text": segments["full_text"],
                }
                writer.writerow(row)

    logger.info(f"Wrote CSV to {output_path}")


def print_detailed_example(example, tokenizer, example_idx: int, step: int):
    """Print a detailed view of a single example, including decoded tokens."""
    tokens = np.asarray(example.tokens.array)
    loss_weight = np.asarray(example.loss_weight.array)

    print(f"\n  --- Detailed view: Step {step}, Example {example_idx} ---")

    # Find transitions in loss_weight (user -> assistant boundaries)
    transitions = np.where(np.diff(loss_weight))[0]
    print(f"  Loss weight transitions at positions: {transitions.tolist()[:30]}")

    # Decode first and last non-padding tokens
    non_pad = np.where(tokens != 0)[0]
    if len(non_pad) > 0:
        first_real = non_pad[0]
        last_real = non_pad[-1]
        # Show first 200 tokens decoded
        snippet_end = min(first_real + 200, last_real + 1)
        snippet_tokens = tokens[first_real:snippet_end].tolist()
        decoded = tokenizer.decode(snippet_tokens, skip_special_tokens=False)
        print(f"  First ~200 tokens (pos {first_real}-{snippet_end}):")
        print(f"    {decoded[:500]!r}")

        # Show last 100 tokens
        snippet_start = max(last_real - 100, first_real)
        snippet_tokens = tokens[snippet_start : last_real + 1].tolist()
        decoded = tokenizer.decode(snippet_tokens, skip_special_tokens=False)
        print(f"  Last ~100 tokens (pos {snippet_start}-{last_real}):")
        print(f"    {decoded[:500]!r}")
    else:
        print("  ALL PADDING — no real tokens!")


async def main():
    parser = argparse.ArgumentParser(description="Inspect training batches for NaN debugging")
    parser.add_argument("--steps", type=int, nargs="+", required=True, help="Steps to inspect (e.g., 734 735 736 737)")
    parser.add_argument("--cache_path", type=str, required=True, help="GCS path to the tokenized cache directory")
    parser.add_argument(
        "--tokenizer", type=str, required=True, help="HuggingFace tokenizer name (e.g., Qwen/Qwen3-0.6B)"
    )
    parser.add_argument("--seq_len", type=int, default=32768, help="Max sequence length used in training")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--seed", type=int, default=42, help="Data seed (must match experiment config)")
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=None,
        help="Model vocab size for OOV checking. If not set, loaded from tokenizer.",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        default=False,
        help="Use the Qwen3-Thinking chat template instead of the base Qwen3 template",
    )
    parser.add_argument(
        "--detailed",
        type=int,
        nargs="*",
        default=None,
        help="Example indices within the batch to decode in detail (e.g., 0 1 5)",
    )
    parser.add_argument(
        "--csv", type=str, default=None, help="Output path for CSV with decoded text (e.g., batch_736.csv)"
    )
    args = parser.parse_args()

    # Load tokenizer for vocab size and optional decoding
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    vocab_size = args.vocab_size or len(tokenizer)
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}, using vocab_size={vocab_size} for OOV check")

    # Select the correct chat template (must contain {%generation%} markers).
    if args.thinking:
        from experiments.chat_templates.qwen3_thinking_chat_template import QWEN_3_THINKING_CHAT_TEMPLATE

        chat_template = QWEN_3_THINKING_CHAT_TEMPLATE
        logger.info("Using Qwen3-Thinking chat template")
    else:
        from experiments.chat_templates.qwen3_chat_template import QWEN_3_CHAT_TEMPLATE

        chat_template = QWEN_3_CHAT_TEMPLATE
        logger.info("Using Qwen3 base chat template")

    # Build the data config
    data_config = build_data_config(args.cache_path, args.tokenizer, chat_template)

    Pos = Axis("position", args.seq_len)
    batch_schedule = BatchSchedule(args.batch_size)

    # Match the PRNG key derivation in train_lm.py:
    # When data_seed is set (as in our SFT configs), data_key = PRNGKey(data_seed)
    # This overrides the default split from trainer.seed.
    data_key = jrandom.PRNGKey(args.seed)

    logger.info("Building dataset (loading cache metadata from GCS)...")
    with local_cpu_mesh():
        dataset = data_config.train_set(Pos, batch_schedule, key=data_key)

    logger.info(f"Dataset ready. Inspecting steps: {args.steps}")

    all_step_problems = {}
    step_examples = {}
    # For CSV output: step -> list of (stats, decoded_segments)
    csv_data: dict[int, list[tuple[dict, dict[str, str]]]] = {}

    for step in args.steps:
        examples = await get_examples_for_step(dataset, batch_schedule, step)

        all_stats = []
        step_csv_entries = []
        for i, ex in enumerate(examples):
            stats = analyze_example(ex, vocab_size, i)
            all_stats.append(stats)

            if args.csv:
                segments = decode_example_segments(ex, tokenizer)
                step_csv_entries.append((stats, segments))

        if args.csv:
            csv_data[step] = step_csv_entries

        problems = print_step_summary(step, all_stats)
        all_step_problems[step] = problems
        step_examples[step] = examples

        # Print detailed view for specific examples if requested
        if args.detailed is not None:
            detail_indices = args.detailed if args.detailed else list(range(min(3, len(examples))))
            for idx in detail_indices:
                if idx < len(examples):
                    print_detailed_example(examples[idx], tokenizer, idx, step)

        # If any problems found and no --detailed specified, auto-detail the first flagged example
        if problems and args.detailed is None:
            flagged_idx = problems[0][0]
            print_detailed_example(examples[flagged_idx], tokenizer, flagged_idx, step)

    # Write CSV if requested
    if args.csv:
        write_csv(args.csv, csv_data)

    # Cross-step comparison
    print(f"\n{'='*80}")
    print("CROSS-STEP COMPARISON")
    print(f"{'='*80}")
    for step in args.steps:
        n_problems = len(all_step_problems.get(step, []))
        examples = step_examples[step]
        tokens_with_loss = [int(np.sum(np.asarray(ex.loss_weight.array) > 0)) for ex in examples]
        padding = [int(np.sum(np.asarray(ex.tokens.array) == 0)) for ex in examples]
        token_maxes = [int(np.max(np.asarray(ex.tokens.array))) for ex in examples]
        print(
            f"  Step {step}: {n_problems:>2} flagged | "
            f"mean_loss_tokens={np.mean(tokens_with_loss):>8.1f} | "
            f"mean_padding={np.mean(padding):>8.1f} | "
            f"max_token_id={max(token_maxes):>6}"
        )


if __name__ == "__main__":
    asyncio.run(main())
