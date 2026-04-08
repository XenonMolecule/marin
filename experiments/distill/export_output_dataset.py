# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export a clean output-only dataset from a reprocessed chat dataset.

Takes a HuggingFace chat dataset (with messages column) and produces a
dataset with just the extracted output text, stripping reasoning and
DSPy markers. Drops rows where output is [NO_USEFUL_CONTENT].

The output dataset uses CC-BY-4.0 licensing since it contains only the
model-generated extraction output, not the original HTML inputs.

Usage:
    python experiments/distill/export_output_dataset.py \\
        --source_repo MichaelR207/rephraser_kimi_v1_0331 \\
        --dest_repo MichaelR207/rephraser_kimi_v1_0331_output \\
        --license cc-by-4.0

    # Dry run (print stats without uploading):
    python experiments/distill/export_output_dataset.py \\
        --source_repo MichaelR207/rephraser_kimi_v1_0331 \\
        --dest_repo test \\
        --dry_run
"""

import argparse
import logging
import re

logger = logging.getLogger(__name__)

# Pre-compiled patterns for stripping inference artifacts
_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
_FIELD_MARKERS_RE = re.compile(
    r"\[\[\s*##\s*(text|completed)\s*##\s*\]\]",
    flags=re.IGNORECASE,
)
_NO_USEFUL_CONTENT_RE = re.compile(r"\[NO_USEFUL_CONTENT\]", flags=re.IGNORECASE)


def extract_output(assistant_content: str) -> str:
    """Extract clean output text from assistant message.

    Strips <think>...</think> reasoning blocks and DSPy field markers
    ([[ ## text ## ]], [[ ## completed ## ]]).
    """
    text = _THINK_RE.sub("", assistant_content)
    text = _FIELD_MARKERS_RE.sub("", text)
    return text.strip()


def format_number(n: int) -> str:
    return f"{n:,}"


def generate_readme(
    dest_repo: str,
    source_repo: str,
    license_id: str,
    train_count: int,
    val_count: int,
    dropped_count: int,
) -> str:
    return f"""---
dataset_info:
  features:
    - name: output
      dtype: string
    - name: spec
      dtype: string
    - name: spec_id
      dtype: string
    - name: model
      dtype: string
    - name: warc_file
      dtype: string
    - name: doc_id
      dtype: string
  splits:
    - name: train
      num_examples: {train_count}
    - name: validation
      num_examples: {val_count}
license: {license_id}
---

# {dest_repo.split("/")[-1]}

Clean extracted text from web pages, produced by **Kimi-K2.5**.

This dataset contains only the model-generated extraction output (no prompts,
no HTML, no reasoning traces). Suitable for text quality analysis, downstream
NLP tasks, and training data.

## Source

Derived from [{source_repo}](https://huggingface.co/datasets/{source_repo})
by extracting the assistant response, stripping `<think>` reasoning and DSPy
field markers, and dropping rows containing `[NO_USEFUL_CONTENT]`.

## Processing

- Reasoning (`<think>...</think>`) stripped
- DSPy markers (`[[ ## text ## ]]`, `[[ ## completed ## ]]`) stripped
- Rows with `[NO_USEFUL_CONTENT]` dropped ({format_number(dropped_count)} rows removed)

## Schema

| Column | Description |
|--------|-------------|
| `output` | Clean extracted text |
| `spec` | Extraction specification used |
| `spec_id` | Specification identifier (0-999) |
| `model` | Model that generated the extraction |
| `warc_file` | Source Common Crawl WARC file |
| `doc_id` | Document identifier within WARC |

## License

This dataset is licensed under [{license_id.upper()}](https://creativecommons.org/licenses/by/4.0/).
The output text is model-generated extraction from public web content.

## Stats

| Split | Rows |
|-------|-----:|
| Train | {format_number(train_count)} |
| Validation | {format_number(val_count)} |
| Total | {format_number(train_count + val_count)} |
| Dropped (no useful content) | {format_number(dropped_count)} |
"""


def process_split(dataset_split, default_model: str | None = None) -> tuple[list[dict], int]:
    """Process a dataset split, returning (clean_rows, dropped_count)."""
    clean_rows = []
    dropped = 0

    for row in dataset_split:
        messages = row["messages"]
        assistant_content = messages[2]["content"]
        output = extract_output(assistant_content)

        # Drop rows with no useful content
        if _NO_USEFUL_CONTENT_RE.search(output) or not output:
            dropped += 1
            continue

        model = row.get("model", "") or default_model or ""

        clean_rows.append(
            {
                "output": output,
                "spec": row.get("spec", ""),
                "spec_id": row.get("spec_id", ""),
                "model": model,
                "warc_file": row.get("warc_file", ""),
                "doc_id": row.get("doc_id", ""),
            }
        )

    return clean_rows, dropped


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Export clean output dataset from chat dataset")
    parser.add_argument(
        "--source_repo", required=True, help="Source HF dataset (e.g. MichaelR207/rephraser_kimi_v1_0331)"
    )
    parser.add_argument("--dest_repo", required=True, help="Destination HF dataset repo ID")
    parser.add_argument("--license", default="cc-by-4.0", help="License identifier (default: cc-by-4.0)")
    parser.add_argument("--default_model", default=None, help="Model name to fill if source has no model column")
    parser.add_argument("--dry_run", action="store_true", help="Print stats without uploading")
    args = parser.parse_args()

    from datasets import Dataset, DatasetDict, load_dataset

    logger.info("Loading source dataset: %s", args.source_repo)
    ds = load_dataset(args.source_repo)

    # Process train
    logger.info("Processing train split...")
    train_rows, train_dropped = process_split(ds["train"], default_model=args.default_model)
    logger.info("Train: %d kept, %d dropped", len(train_rows), train_dropped)

    # Process validation
    logger.info("Processing validation split...")
    val_rows, val_dropped = process_split(ds["validation"], default_model=args.default_model)
    logger.info("Val: %d kept, %d dropped", len(val_rows), val_dropped)

    total_dropped = train_dropped + val_dropped

    if args.dry_run:
        readme = generate_readme(
            args.dest_repo, args.source_repo, args.license, len(train_rows), len(val_rows), total_dropped
        )
        print(readme)
        print("\nSample output (first row, first 500 chars):")
        print(train_rows[0]["output"][:500])
        return

    # Build and upload — explicit features to avoid type mismatches across splits
    from datasets import Features, Value

    features = Features(
        {
            "output": Value("string"),
            "spec": Value("string"),
            "spec_id": Value("string"),
            "model": Value("string"),
            "warc_file": Value("string"),
            "doc_id": Value("string"),
        }
    )
    train_ds = Dataset.from_list(train_rows, features=features)
    val_ds = Dataset.from_list(val_rows, features=features)
    out_ds = DatasetDict({"train": train_ds, "validation": val_ds})

    logger.info("Pushing to %s...", args.dest_repo)
    out_ds.push_to_hub(args.dest_repo)

    # Upload README with license
    readme = generate_readme(
        args.dest_repo, args.source_repo, args.license, len(train_rows), len(val_rows), total_dropped
    )
    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.dest_repo,
        repo_type="dataset",
    )
    logger.info("Uploaded to https://huggingface.co/datasets/%s", args.dest_repo)


if __name__ == "__main__":
    main()
