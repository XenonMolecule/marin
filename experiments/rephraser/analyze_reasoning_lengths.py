# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze token lengths of input, reasoning, and output in rephraser dataset."""

import re
import numpy as np
import tiktoken
from datasets import load_dataset


def parse_message(messages):
    """Parse a chat message list into input, reasoning, and output token counts."""
    enc = tiktoken.get_encoding("cl100k_base")

    # Collect all user turns as "input"
    input_parts = []
    reasoning_parts = []
    output_parts = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role in ("user", "system"):
            input_parts.append(content)
        elif role == "assistant":
            # Split out <think>...</think> as reasoning
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if think_match:
                reasoning_parts.append(think_match.group(1).strip())
                # Everything outside <think> tags is output
                output_text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                if output_text:
                    output_parts.append(output_text)
            else:
                output_parts.append(content)

    input_text = "\n".join(input_parts)
    reasoning_text = "\n".join(reasoning_parts)
    output_text = "\n".join(output_parts)

    return (
        len(enc.encode(input_text)),
        len(enc.encode(reasoning_text)) if reasoning_text else 0,
        len(enc.encode(output_text)) if output_text else 0,
    )


def main():
    print("Loading dataset...")
    ds = load_dataset("MichaelR207/rephraser_late_check_0225", split="train")
    print(f"Loaded {len(ds)} examples")

    # Check column names
    print(f"Columns: {ds.column_names}")

    # Parse all messages
    input_lens = []
    reasoning_lens = []
    output_lens = []

    for row in ds:
        inp, reas, out = parse_message(row["messages"])
        input_lens.append(inp)
        reasoning_lens.append(reas)
        output_lens.append(out)

    input_lens = np.array(input_lens, dtype=float)
    reasoning_lens = np.array(reasoning_lens, dtype=float)
    output_lens = np.array(output_lens, dtype=float)

    # Basic stats
    print("\n=== Token Length Statistics ===")
    for name, arr in [("Input", input_lens), ("Reasoning", reasoning_lens), ("Output", output_lens)]:
        print(f"\n{name}:")
        print(
            f"  mean={arr.mean():.1f}  median={np.median(arr):.1f}  "
            f"std={arr.std():.1f}  min={arr.min():.0f}  max={arr.max():.0f}"
        )
        print(
            f"  p5={np.percentile(arr, 5):.0f}  p25={np.percentile(arr, 25):.0f}  "
            f"p75={np.percentile(arr, 75):.0f}  p95={np.percentile(arr, 95):.0f}"
        )

    # Ratios
    print("\n=== Ratios ===")
    # Avoid division by zero
    mask_input = input_lens > 0
    mask_reasoning = reasoning_lens > 0

    if mask_input.any():
        ratio_reas_inp = reasoning_lens[mask_input] / input_lens[mask_input]
        print(f"\nReasoning / Input (n={mask_input.sum()}):")
        print(
            f"  mean={ratio_reas_inp.mean():.2f}  median={np.median(ratio_reas_inp):.2f}  "
            f"std={ratio_reas_inp.std():.2f}"
        )

        ratio_out_inp = output_lens[mask_input] / input_lens[mask_input]
        print(f"\nOutput / Input (n={mask_input.sum()}):")
        print(
            f"  mean={ratio_out_inp.mean():.2f}  median={np.median(ratio_out_inp):.2f}  "
            f"std={ratio_out_inp.std():.2f}"
        )

    if mask_reasoning.any():
        ratio_out_reas = output_lens[mask_reasoning] / reasoning_lens[mask_reasoning]
        print(f"\nOutput / Reasoning (n={mask_reasoning.sum()}):")
        print(
            f"  mean={ratio_out_reas.mean():.2f}  median={np.median(ratio_out_reas):.2f}  "
            f"std={ratio_out_reas.std():.2f}"
        )

    # Correlations
    print("\n=== Pearson Correlations ===")
    if mask_input.sum() > 1:
        corr_reas_inp = np.corrcoef(input_lens, reasoning_lens)[0, 1]
        corr_out_inp = np.corrcoef(input_lens, output_lens)[0, 1]
        print(f"  Reasoning vs Input:  r={corr_reas_inp:.4f}")
        print(f"  Output vs Input:     r={corr_out_inp:.4f}")
    if mask_reasoning.sum() > 1:
        corr_out_reas = np.corrcoef(reasoning_lens, output_lens)[0, 1]
        print(f"  Output vs Reasoning: r={corr_out_reas:.4f}")

    # Bucketed analysis: reasoning length by input length quintiles
    print("\n=== Reasoning & Output by Input Length Quintile ===")
    if len(input_lens) >= 5:
        quintiles = np.percentile(input_lens, [0, 20, 40, 60, 80, 100])
        print(
            f"{'Quintile':>10} {'Input Range':>20} {'N':>5} {'Avg Reasoning':>15} {'Avg Output':>12} {'Reas/Inp':>10} {'Out/Inp':>10}"
        )
        for i in range(5):
            lo, hi = quintiles[i], quintiles[i + 1]
            if i < 4:
                mask = (input_lens >= lo) & (input_lens < hi)
            else:
                mask = (input_lens >= lo) & (input_lens <= hi)
            if mask.any():
                avg_r = reasoning_lens[mask].mean()
                avg_o = output_lens[mask].mean()
                avg_i = input_lens[mask].mean()
                print(
                    f"{'Q' + str(i+1):>10} {f'[{lo:.0f}, {hi:.0f})':>20} {mask.sum():>5} "
                    f"{avg_r:>15.1f} {avg_o:>12.1f} {avg_r/avg_i:>10.2f} {avg_o/avg_i:>10.2f}"
                )

    # Bucketed analysis: output length by reasoning length quintiles
    print("\n=== Output by Reasoning Length Quintile ===")
    has_reasoning = reasoning_lens > 0
    if has_reasoning.sum() >= 5:
        r_vals = reasoning_lens[has_reasoning]
        o_vals = output_lens[has_reasoning]
        quintiles = np.percentile(r_vals, [0, 20, 40, 60, 80, 100])
        print(f"{'Quintile':>10} {'Reasoning Range':>20} {'N':>5} {'Avg Output':>12} {'Out/Reas':>10}")
        for i in range(5):
            lo, hi = quintiles[i], quintiles[i + 1]
            if i < 4:
                mask = (r_vals >= lo) & (r_vals < hi)
            else:
                mask = (r_vals >= lo) & (r_vals <= hi)
            if mask.any():
                avg_o = o_vals[mask].mean()
                avg_r = r_vals[mask].mean()
                print(
                    f"{'Q' + str(i+1):>10} {f'[{lo:.0f}, {hi:.0f})':>20} {mask.sum():>5} "
                    f"{avg_o:>12.1f} {avg_o/avg_r:>10.2f}"
                )

    # How many have no reasoning?
    no_reasoning = (reasoning_lens == 0).sum()
    print(
        f"\n=== Messages with no reasoning: {no_reasoning}/{len(reasoning_lens)} "
        f"({100*no_reasoning/len(reasoning_lens):.1f}%) ==="
    )


if __name__ == "__main__":
    main()
