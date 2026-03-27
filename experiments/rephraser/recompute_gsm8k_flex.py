# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recompute GSM8K flexible-extract scores from saved lm-eval-harness samples.

The lm-evaluation-harness `flexible-extract` filter has a bug where trailing
punctuation like `..` (a formatting artifact from Q/R/A training data) causes
it to extract `..` instead of the actual number. This script reads the saved
samples JSONL files and recomputes a corrected score using a more robust
answer extraction strategy.

Usage:
    # 0-shot only:
    uv run python experiments/rephraser/recompute_gsm8k_flex.py

    # All conditions (0-shot + 4-shot):
    uv run python experiments/rephraser/recompute_gsm8k_flex.py --include-4shot

    # Custom bucket:
    uv run python experiments/rephraser/recompute_gsm8k_flex.py --bucket gs://marin-us-east5
"""

import argparse
import json
import logging
import re
import subprocess

logger = logging.getLogger(__name__)

# Eval directories keyed by display name -> (glob pattern, task alias).
EVAL_CONDITIONS_0SHOT = {
    "Baseline (no SFT)": ("qwen3-0.6b-baseline-0shot-*", "gsm8k_cot_0shot"),
    "GSM8K plaintext SFT": ("qwen3-0.6b-gsm8k-plaintext-sft-0shot-*", "gsm8k_cot_0shot"),
    "Q/R/A plaintext SFT": ("qwen3-0.6b-qra-plaintext-sft-0shot-*", "gsm8k_cot_0shot"),
    "Resiliparse SFT": ("qwen3-0.6b-resiliparse-sft-0shot-*", "gsm8k_cot_0shot"),
    "Q/R/A plaintext SFT v2": ("qwen3-0.6b-qra-plaintext-sft-v2-0shot-*", "gsm8k_cot_0shot"),
    "Raw Q/R/A markdown SFT": ("qwen3-0.6b-raw-qra-markdown-sft-0shot-*", "gsm8k_cot_0shot"),
}

EVAL_CONDITIONS_4SHOT = {
    "Baseline (no SFT)": ("qwen3-0.6b-baseline-4shot-*", "gsm8k_cot_8shot"),
    "GSM8K chat SFT": ("qwen3-0.6b-gsm8k-chat-sft-vllm-*", "gsm8k_cot_8shot"),
    "GSM8K plaintext SFT": ("qwen3-0.6b-gsm8k-plaintext-sft-vllm-*", "gsm8k_cot_8shot"),
    "Q/R/A chat SFT": ("qwen3-0.6b-qra-chat-sft-vllm-*", "gsm8k_cot_8shot"),
    "Q/R/A plaintext SFT": ("qwen3-0.6b-qra-plaintext-sft-vllm-*", "gsm8k_cot_8shot"),
    "Resiliparse SFT": ("qwen3-0.6b-resiliparse-sft-vllm-*", "gsm8k_cot_8shot"),
    "Q/R/A plaintext SFT v2": ("qwen3-0.6b-qra-plaintext-sft-v2-vllm-*", "gsm8k_cot_8shot"),
    "Raw Q/R/A markdown SFT": ("qwen3-0.6b-raw-qra-markdown-sft-vllm-*", "gsm8k_cot_8shot"),
}


def normalize_number(s: str) -> str | None:
    """Normalize a number string for comparison."""
    s = s.strip().rstrip(".").replace(",", "")
    if not s:
        return None
    try:
        float(s)
        return s
    except ValueError:
        return None


def extract_answer_corrected(raw_output: str) -> str | None:
    """Extract the answer from a raw GSM8K model output.

    Tries multiple strategies in order:
    1. "The answer is X" pattern (Q/R/A plaintext format)
    2. "#### X" pattern (GSM8K training format)
    3. "## Answer" section (raw markdown format)
    4. Last number in the output (fallback)
    """
    text = raw_output.strip()

    # Strategy 1: "The answer is X"
    m = re.search(r"[Tt]he answer is\s+(.+?)(?:\.+\s*$|\s*$)", text)
    if m:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", m.group(1))
        if nums:
            norm = normalize_number(nums[-1])
            if norm is not None:
                return norm

    # Strategy 2: "#### X"
    m = re.search(r"####\s*(.+?)(?:\s*$)", text)
    if m:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", m.group(1))
        if nums:
            norm = normalize_number(nums[-1])
            if norm is not None:
                return norm

    # Strategy 3: "## Answer" section — extract last number from the answer section
    m = re.search(r"## Answer\s*\n(.+)", text, re.DOTALL)
    if m:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", m.group(1))
        if nums:
            norm = normalize_number(nums[-1])
            if norm is not None:
                return norm

    # Strategy 4: Last number in the output
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if nums:
        norm = normalize_number(nums[-1])
        if norm is not None:
            return norm

    return None


def _gcloud_ls(pattern: str) -> list[str]:
    """List GCS paths matching a pattern using gcloud CLI."""
    result = subprocess.run(
        ["gcloud", "storage", "ls", pattern],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]


def _gcloud_cat(path: str) -> str | None:
    """Read a GCS file using gcloud CLI."""
    result = subprocess.run(
        ["gcloud", "storage", "cat", path],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def find_samples_file(bucket: str, dir_pattern: str, task_alias: str) -> str | None:
    """Find the samples JSONL file for a given eval condition."""
    base = f"{bucket}/evaluation/lm_evaluation_harness"
    pattern = f"{base}/{dir_pattern}/{task_alias}/**/samples_*.jsonl"
    files = _gcloud_ls(pattern)
    return files[0] if files else None


def recompute_scores(samples_path: str) -> dict | None:
    """Recompute strict, flexible, and corrected scores from a samples JSONL file."""
    content = _gcloud_cat(samples_path)
    if content is None:
        return None

    lines = [json.loads(line) for line in content.strip().split("\n") if line.strip()]

    strict_samples = [s for s in lines if s.get("filter") == "strict-match"]
    flex_samples = [s for s in lines if s.get("filter") == "flexible-extract"]
    flex_by_id = {s["doc_id"]: s for s in flex_samples}

    total = len(strict_samples)
    if total == 0:
        return None

    strict_correct = 0
    flex_correct = 0
    corrected_correct = 0

    for s in strict_samples:
        doc_id = s["doc_id"]
        target = str(s["target"]).strip().replace(",", "")
        raw = s["resps"][0][0] if s.get("resps") and s["resps"][0] else ""

        is_strict = s.get("exact_match", 0) == 1.0
        flex_s = flex_by_id.get(doc_id)
        is_flex = flex_s.get("exact_match", 0) == 1.0 if flex_s else False

        if is_strict:
            strict_correct += 1
        if is_flex:
            flex_correct += 1

        # Corrected extraction
        extracted = extract_answer_corrected(raw)
        if extracted is not None and extracted == target:
            corrected_correct += 1
        elif is_strict or is_flex:
            # Trust existing correct scores if our extractor misses edge cases
            corrected_correct += 1

    return {
        "total": total,
        "strict": strict_correct,
        "flex": flex_correct,
        "corrected": corrected_correct,
        "strict_pct": strict_correct / total * 100,
        "flex_pct": flex_correct / total * 100,
        "corrected_pct": corrected_correct / total * 100,
    }


def print_table(title: str, conditions: dict, bucket: str):
    """Print a results table for a set of conditions."""
    print(f"\n{'=' * 95}")
    print(f"  {title}")
    print(f"{'=' * 95}")
    print(
        f"{'Condition':<40s} "
        f"{'Strict':>10s} "
        f"{'Flex':>10s} "
        f"{'Corrected':>10s} "
        f"{'Flex→Corr':>10s}"
    )
    print("-" * 85)

    for name, (dir_pattern, task_alias) in conditions.items():
        samples_path = find_samples_file(bucket, dir_pattern, task_alias)
        if samples_path is None:
            print(f"{name:<40s} {'—':>10s} {'—':>10s} {'—':>10s} {'—':>10s}")
            continue

        scores = recompute_scores(samples_path)
        if scores is None:
            print(f"{name:<40s} {'—':>10s} {'—':>10s} {'—':>10s} {'—':>10s}")
            continue

        gap = scores["corrected_pct"] - scores["flex_pct"]
        print(
            f"{name:<40s} "
            f"{scores['strict_pct']:>9.2f}% "
            f"{scores['flex_pct']:>9.2f}% "
            f"{scores['corrected_pct']:>9.2f}% "
            f"{gap:>+9.2f}pp"
        )


def main():
    parser = argparse.ArgumentParser(description="Recompute GSM8K flex scores from saved samples")
    parser.add_argument("--include-4shot", action="store_true", help="Include 4-shot results")
    parser.add_argument("--bucket", default="gs://marin-us-central1", help="GCS bucket")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    print_table("GSM8K 0-shot: Strict vs Flexible vs Corrected", EVAL_CONDITIONS_0SHOT, args.bucket)

    if args.include_4shot:
        print_table("GSM8K 8-shot: Strict vs Flexible vs Corrected", EVAL_CONDITIONS_4SHOT, args.bucket)


if __name__ == "__main__":
    main()
