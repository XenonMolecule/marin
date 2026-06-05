# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Upload reprocessed dataset to HuggingFace with token statistics.

Reads all reassembled output shards (train + val + backfill), computes token
statistics, sorts by global_row_idx, generates a README with token accounting,
and uploads to HuggingFace.

Usage:
    python experiments/distill/upload_to_huggingface.py \
        --repo_id MichaelR207/rephraser_kimi_v1_0331 \
        --train_dirs \
            gs://marin-us-central1/distill/kimi-k2.5_reprocess/reassembled_rows_0_818780_s0.0-0.01-8e57e1 \
            gs://marin-us-central1/distill/kimi-k2.5_reprocess/reassembled_rows_0_818780_s0.01-0.02-2c9eb6 \
            ... \
            gs://marin-us-central1/distill/kimi-k2.5_reprocess/backfill \
        --val_dir gs://marin-us-central1/distill/kimi-k2.5_reprocess/reassembled_rows_818780_818880-a1d9fd \
        --model_name "moonshotai/Kimi-K2.5" \
        --tokenizer cl100k_base
"""

import argparse
import gzip
import json
import logging
import os
import tempfile

import fsspec
import tiktoken
from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)

FINAL_COLUMNS = ["messages", "warc_file", "doc_id", "spec_id", "spec", "model"]


def load_all_rows(dirs: list[str]) -> list[dict]:
    """Load all JSONL.gz rows from multiple directories, dedup by global_row_idx."""
    rows_by_idx: dict[int, dict] = {}
    for d in dirs:
        pattern = os.path.join(d, "*.jsonl.gz")
        files = fsspec_glob(pattern)
        for f in files:
            with fsspec.open(f, "rb") as fh:
                with gzip.open(fh, "rt", encoding="utf-8") as gz:
                    for line in gz:
                        row = json.loads(line)
                        idx = row.get("global_row_idx")
                        if idx is not None:
                            rows_by_idx[idx] = row
    # Sort by global_row_idx
    sorted_rows = [rows_by_idx[k] for k in sorted(rows_by_idx.keys())]
    return sorted_rows


def compute_token_stats(rows: list[dict], encoding_name: str) -> dict:
    """Compute input/reasoning/output token statistics."""
    enc = tiktoken.get_encoding(encoding_name)

    total_input = 0
    total_reasoning = 0
    total_output = 0

    for row in rows:
        msgs = row["messages"]
        # Input = system + user
        inp = msgs[0]["content"] + msgs[1]["content"]
        total_input += len(enc.encode(inp))

        # Output = assistant
        assistant = msgs[2]["content"]
        if "</think>" in assistant:
            think_end = assistant.index("</think>")
            reasoning = assistant[:think_end]
            content = assistant[think_end + len("</think>") :]
            total_reasoning += len(enc.encode(reasoning))
            total_output += len(enc.encode(content))
        else:
            total_output += len(enc.encode(assistant))

    return {
        "input_tokens": total_input,
        "reasoning_tokens": total_reasoning,
        "output_tokens": total_output,
        "total_tokens": total_input + total_reasoning + total_output,
        "num_rows": len(rows),
    }


def format_number(n: int) -> str:
    """Format number with commas."""
    return f"{n:,}"


def _build_stats_table(train: dict, val: dict, total: dict) -> str:
    """Build the markdown token stats table."""

    def row(label: str, key: str) -> str:
        return (
            f"| {label} | {format_number(train[key])} "
            f"| {format_number(val[key])} "
            f"| {format_number(total[key])} |"
        )

    lines = [
        "| Metric | Train | Validation | Total |",
        "|--------|------:|----------:|------:|",
        row("Input tokens", "input_tokens"),
        row("Reasoning tokens", "reasoning_tokens"),
        row("Output tokens", "output_tokens"),
        row("Total", "total_tokens"),
        row("Rows", "num_rows"),
    ]
    return "\n".join(lines)


def generate_readme(
    repo_id: str,
    model_name: str,
    tokenizer_name: str,
    train_stats: dict,
    val_stats: dict,
) -> str:
    """Generate README.md content with token statistics."""
    total_stats = {
        "input_tokens": train_stats["input_tokens"] + val_stats["input_tokens"],
        "reasoning_tokens": train_stats["reasoning_tokens"] + val_stats["reasoning_tokens"],
        "output_tokens": train_stats["output_tokens"] + val_stats["output_tokens"],
        "total_tokens": train_stats["total_tokens"] + val_stats["total_tokens"],
        "num_rows": train_stats["num_rows"] + val_stats["num_rows"],
    }

    readme = f"""---
dataset_info:
  features:
    - name: messages
      list:
        - name: role
          dtype: string
        - name: content
          dtype: string
    - name: warc_file
      dtype: string
    - name: doc_id
      dtype: string
    - name: spec_id
      dtype: string
    - name: spec
      dtype: string
    - name: model
      dtype: string
  splits:
    - name: train
      num_examples: {train_stats["num_rows"]}
    - name: validation
      num_examples: {val_stats["num_rows"]}
license: apache-2.0
---

# {repo_id.split("/")[-1]}

Web extraction distillation dataset reprocessed with **{model_name}**.

This dataset contains HTML-to-text extraction results where each row has a system prompt,
a user prompt containing HTML + extraction spec, and an assistant response with
`<think>` reasoning followed by the extracted text.

## Source

Reprocessed from [MichaelR207/rephraser_late_check_0225](https://huggingface.co/datasets/MichaelR207/rephraser_late_check_0225)
using the same input prompts but replacing GPT-OSS-120B outputs with {model_name} outputs.

## Token Statistics

Token counts computed using the `{tokenizer_name}` tokenizer.

- **Input tokens**: tokens in the prompt sent to the model.
- **Reasoning tokens**: tokens used for chain-of-thought reasoning (`<think>...</think>`).
- **Output tokens**: non-reasoning completion tokens, i.e. the actual extracted text.
- **Total**: input + reasoning + output.

{_build_stats_table(train_stats, val_stats, total_stats)}

## Schema

Each row contains:
- `messages`: list of 3 dicts (`system`, `user`, `assistant`) — compatible with chat fine-tuning
- `warc_file`: source Common Crawl WARC file
- `doc_id`: document identifier within the WARC file
- `spec_id`: extraction specification identifier (0-999)
- `spec`: full text of the extraction specification
- `model`: model used to generate the assistant response

## Coverage

This dataset covers the first 7% of each extraction spec's rows from the source dataset
({format_number(train_stats["num_rows"])} train + {format_number(val_stats["num_rows"])} validation rows).
"""
    return readme


def write_sharded_jsonl(rows: list[dict], output_dir: str, records_per_shard: int = 10000):
    """Write rows as sharded JSONL.gz files, dropping global_row_idx."""
    total_shards = max(1, (len(rows) + records_per_shard - 1) // records_per_shard)
    for shard_idx in range(total_shards):
        start = shard_idx * records_per_shard
        end = min(start + records_per_shard, len(rows))
        shard_rows = rows[start:end]
        shard_path = os.path.join(output_dir, f"data-{shard_idx:05d}-of-{total_shards:05d}.jsonl.gz")
        with open(shard_path, "wb") as f:
            with gzip.open(f, "wt", encoding="utf-8") as gz:
                for row in shard_rows:
                    # Drop global_row_idx from final output
                    clean = {k: row[k] for k in FINAL_COLUMNS if k in row}
                    gz.write(json.dumps(clean, ensure_ascii=False) + "\n")
    return total_shards


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Upload reprocessed dataset to HuggingFace")
    parser.add_argument("--repo_id", required=True, help="HuggingFace repo ID (e.g. MichaelR207/rephraser_kimi_v1_0331)")
    parser.add_argument("--train_dirs", nargs="+", required=True, help="GCS paths to train reassembled output dirs")
    parser.add_argument("--val_dir", required=True, help="GCS path to val reassembled output dir")
    parser.add_argument("--model_name", default="moonshotai/Kimi-K2.5", help="Model name for README")
    parser.add_argument("--tokenizer", default="cl100k_base", help="Tiktoken encoding for token counting")
    parser.add_argument("--records_per_shard", type=int, default=10000, help="Rows per output shard")
    parser.add_argument("--dry_run", action="store_true", help="Compute stats and generate README without uploading")
    args = parser.parse_args()

    # Load train rows
    logger.info("Loading train rows from %d directories...", len(args.train_dirs))
    train_rows = load_all_rows(args.train_dirs)
    logger.info("Train: %d rows (sorted by global_row_idx)", len(train_rows))

    # Load val rows
    logger.info("Loading val rows...")
    val_rows = load_all_rows([args.val_dir])
    logger.info("Val: %d rows", len(val_rows))

    # Compute token stats
    logger.info("Computing token statistics (this may take a few minutes)...")
    train_stats = compute_token_stats(train_rows, args.tokenizer)
    val_stats = compute_token_stats(val_rows, args.tokenizer)

    logger.info(
        "Train: %s input, %s reasoning, %s output, %s total",
        format_number(train_stats["input_tokens"]),
        format_number(train_stats["reasoning_tokens"]),
        format_number(train_stats["output_tokens"]),
        format_number(train_stats["total_tokens"]),
    )
    logger.info(
        "Val: %s input, %s reasoning, %s output, %s total",
        format_number(val_stats["input_tokens"]),
        format_number(val_stats["reasoning_tokens"]),
        format_number(val_stats["output_tokens"]),
        format_number(val_stats["total_tokens"]),
    )

    # Generate README
    readme = generate_readme(args.repo_id, args.model_name, args.tokenizer, train_stats, val_stats)

    if args.dry_run:
        print(readme)
        return

    # Write to temp dir and upload
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write README
        readme_path = os.path.join(tmpdir, "README.md")
        with open(readme_path, "w") as f:
            f.write(readme)

        # Write train shards
        train_dir = os.path.join(tmpdir, "data")
        os.makedirs(train_dir)
        n_train = write_sharded_jsonl(train_rows, train_dir, args.records_per_shard)
        logger.info("Wrote %d train shards", n_train)

        # Write val shards
        val_out_dir = os.path.join(tmpdir, "validation")
        os.makedirs(val_out_dir)
        n_val = write_sharded_jsonl(val_rows, val_out_dir, args.records_per_shard)
        logger.info("Wrote %d val shards", n_val)

        # Upload to HuggingFace
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(folder_path=tmpdir, repo_id=args.repo_id, repo_type="dataset")
        logger.info("Uploaded to https://huggingface.co/datasets/%s", args.repo_id)


if __name__ == "__main__":
    main()
