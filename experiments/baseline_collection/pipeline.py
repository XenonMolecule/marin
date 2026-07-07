# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Baseline dataset collection pipeline.

Extracts matching records from Nemotron-CC, DCLM, and FineWeb-Edu for a fixed
set of Common Crawl WARC files (a "manifest"). Also extracts raw text via
resiliparse as a fourth "everything" baseline.

All four subsets are tokenized with the standard Marin tokenizer for training
and token-count comparison.

The WARC manifest is a txt file with one ``s3://commoncrawl/...warc.gz`` path
per line. It defaults to the canonical 3000-WARC baseline manifest; pass
``--manifest`` to run the same DAG over a different WARC set. A short label is
derived from the manifest filename (``baseline_warcs_<label>.txt``) and used to
name the download step so distinct manifests produce distinct output trees.
Downstream step names are stable; their output dirs are kept distinct by the
executor's config hash, which transitively depends on the manifest.

Usage (Iris, run in/near us-central2 where the DCLM + Nemotron raw buckets live
to avoid cross-region egress):

    uv run iris --cluster marin job run \\
        --region us-central2 \\
        --priority interactive --no-wait \\
        --cpu 4 --memory 8GB --disk 50GB \\
        --job-name baseline-collection-pipeline \\
        -e WANDB_API_KEY ${WANDB_API_KEY} \\
        -e HF_TOKEN ${HF_TOKEN} \\
        -- python experiments/baseline_collection/pipeline.py \\
        --manifest experiments/distill/baseline_warcs_3000.txt

The launching job is just the executor driver; the per-stage filter steps
request their own RAM via ``remote(...)`` (DCLM ~24GB, Nemotron 16GB,
FineWeb-Edu 32GB), so the driver itself stays small.
"""

import argparse
import sys
from pathlib import Path

from fray import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.execution.remote import remote
from marin.transform.extract_text_from_html import (
    ExtractTextConfig,
    _extract_text,
    _is_non_empty,
)
from zephyr import Dataset, ZephyrContext

from experiments.baseline_collection.download_warcs import (
    IncrementalWarcDownloadConfig,
    download_warcs_incremental,
)
from experiments.baseline_collection.extract_warc_metadata import (
    ExtractWarcMetadataConfig,
    extract_warc_metadata,
)
from experiments.baseline_collection.filter_dclm import FilterDclmConfig, filter_dclm
from experiments.baseline_collection.filter_fineweb_edu import FilterFinewebEduConfig, filter_fineweb_edu
from experiments.baseline_collection.filter_nemotron import (
    FilterNemotronConfig,
    FilterNemotronFullConfig,
    filter_nemotron,
    filter_nemotron_full,
)
from experiments.defaults import default_tokenize

# --- Paths ---

DEFAULT_MANIFEST = str(Path(__file__).resolve().parent.parent / "distill" / "baseline_warcs_3000.txt")

# NOTE: "nemotro-cc" is the real bucket name, not a typo. The executor step that
# produced it hashed to "eeb783" and wrote to this exact (misspelled) directory.
# All sibling scripts (nemotron_explorer, pipeline_10k, validate_url_coverage*)
# reference the same spelling. Do not "correct" it — the data lives here.
NEMOTRON_BASE = "gs://marin-us-central2/raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/data-jsonl"
DCLM_BASE = (
    "gs://marin-us-central2/raw/dclm/a3b142c/huggingface.co/datasets/" "mlfoundations/dclm-baseline-1.0/resolve/a3b142c"
)
FINEWEB_EDU_BASE = "gs://marin-us-central2/raw/fineweb-edu"

TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"

# Selectable end products. Each maps to one tokenize step; the executor pulls in
# that step's transitive deps (download -> metadata -> filter/extract). Scoping
# avoids paying for scans you don't need (e.g. FineWeb/Nemotron-organic/raw-HTML)
# on an expensive fresh download. ``resiliparse`` here is the *raw* extraction +
# tokenize; its fuzzy-deduped variant is produced separately by
# dedup_resiliparse_warc_scaling.py, which reads this step's extraction output.
PRODUCT_NAMES = ["nemotron", "nemotron_full", "dclm", "fineweb", "resiliparse", "raw_html"]

# Manifests follow the convention ``baseline_warcs_<label>.txt``. The label
# identifies the download step's output tree. The default manifest yields the
# label "3000", preserving the historical ``baseline_3000`` step name (and its
# output hash) exactly.
_MANIFEST_PREFIX = "baseline_warcs_"


def manifest_label(manifest_path: str) -> str:
    """Derive a short step-name label from a WARC manifest filename.

    ``baseline_warcs_3000.txt`` -> ``3000``. Manifests that don't follow the
    ``baseline_warcs_<label>`` convention fall back to the full filename stem.
    """
    stem = Path(manifest_path).stem
    if stem.startswith(_MANIFEST_PREFIX):
        return stem[len(_MANIFEST_PREFIX) :]
    return stem


def build_steps(manifest_path: str, products: list[str] = PRODUCT_NAMES) -> list[ExecutorStep]:
    """Build the baseline-collection DAG for a given WARC manifest.

    ``products`` selects which end products to return (default: all). The
    executor resolves each tokenize step's transitive deps, so selecting a
    subset skips the scans the other products would have triggered.

    Only the download step's name carries the manifest label; downstream step
    names are stable. Distinct manifests still produce distinct downstream
    output dirs because the executor's config hash depends transitively on the
    download step's (manifest-derived) output path.
    """
    unknown = set(products) - set(PRODUCT_NAMES)
    if unknown:
        raise ValueError(f"unknown products {sorted(unknown)}; choose from {PRODUCT_NAMES}")
    label = manifest_label(manifest_path)

    # --- Step 1: Download WARCs (incremental, per-file resumable) ---

    download_warcs = ExecutorStep(
        name=f"raw/commoncrawl/baseline_{label}",
        description="Download WARC files from Common Crawl and extract HTML.",
        fn=download_warcs_incremental,
        config=IncrementalWarcDownloadConfig(
            warc_manifest_path=manifest_path,
            output_path=this_output_path(),
        ),
    )

    # --- Step 2a: Extract WARC metadata (record IDs, URLs, file paths) ---

    extract_metadata = ExecutorStep(
        name="metadata/baseline_warc_metadata",
        description="Extract per-record metadata (record_id, url, warc_file, snapshot) from downloaded WARCs.",
        fn=extract_warc_metadata,
        config=ExtractWarcMetadataConfig(
            input_path=download_warcs / "*.jsonl.gz",
            output_path=this_output_path(),
        ),
    )

    # --- Step 2b: Extract raw text via resiliparse (the "everything" baseline) ---

    extract_text = ExecutorStep(
        name="extracted/baseline_resiliparse",
        description="Extract all text from downloaded WARCs via resiliparse (main_content=True).",
        fn=remote(extract_text_fast, resources=ResourceConfig(cpu=4, ram="32g")),
        config=ExtractTextConfig(
            input_path=download_warcs / "*.jsonl.gz",
            output_path=this_output_path(),
        ),
    )

    # --- Step 3a: Filter Nemotron-CC v1 (join on URL per snapshot) ---

    filter_nemotron_step = ExecutorStep(
        name="filtered/baseline_nemotron",
        description="Filter Nemotron-CC v1 organic (kind=actual) records matching our WARCs (join on URL).",
        fn=remote(filter_nemotron, resources=ResourceConfig(cpu=4, ram="16g")),
        config=FilterNemotronConfig(
            metadata_path=extract_metadata / "*.jsonl.gz",
            nemotron_base_path=NEMOTRON_BASE,
            output_path=this_output_path(),
        ),
    )

    # Nemotron "full" baseline: organic + all 5 rephraser-synthetic variants
    # (distill, diverse_qa_pairs, extract_knowledge, knowledge_list, wrap_medium).
    # These are the variants that drive Nemotron-CC's headline token-count advantage
    # over DCLM, so this is the right comparison point for the paper's token totals.
    # The organic-only step above is kept as a separate baseline so we can measure
    # the contribution of the rephraser variants directly.
    filter_nemotron_full_step = ExecutorStep(
        name="filtered/baseline_nemotron_full",
        description="Filter Nemotron-CC v1 organic + synthetic rephraser records matching our WARCs (join on URL).",
        fn=remote(filter_nemotron_full, resources=ResourceConfig(cpu=4, ram="16g")),
        config=FilterNemotronFullConfig(
            metadata_path=extract_metadata / "*.jsonl.gz",
            nemotron_base_path=NEMOTRON_BASE,
            output_path=this_output_path(),
        ),
    )

    # --- Step 3b: Filter DCLM-baseline (join on WARC-Record-ID, full scan) ---

    filter_dclm_step = ExecutorStep(
        name="filtered/baseline_dclm",
        description="Filter DCLM-baseline records matching our WARCs (join on WARC-Record-ID).",
        # filter_dclm builds the full ~156M-record WARC-Record-ID set in memory on
        # this coordinator before dispatching to its inner 128g Zephyr workers, so
        # the wrapper itself needs to hold ~21GB+ set + load spike (24g OOM'd).
        fn=remote(filter_dclm, resources=ResourceConfig(cpu=4, ram="128g")),
        config=FilterDclmConfig(
            metadata_path=extract_metadata / "*.jsonl.gz",
            dclm_base_path=DCLM_BASE,
            output_path=this_output_path(),
        ),
    )

    # --- Step 3c: Filter FineWeb-Edu (join on file_path per snapshot) ---

    filter_fineweb_step = ExecutorStep(
        name="filtered/baseline_fineweb_edu",
        description="Filter FineWeb-Edu records matching our WARCs (join on file_path).",
        fn=remote(filter_fineweb_edu, resources=ResourceConfig(cpu=4, ram="32g")),
        config=FilterFinewebEduConfig(
            metadata_path=extract_metadata / "*.jsonl.gz",
            fineweb_base_path=FINEWEB_EDU_BASE,
            output_path=this_output_path(),
        ),
    )

    # --- Step 4: Tokenize all subsets ---

    tokenize_nemotron = default_tokenize(
        name="baseline_nemotron",
        dataset=filter_nemotron_step / "*.jsonl.gz",
        tokenizer=TOKENIZER,
    )

    tokenize_nemotron_full = default_tokenize(
        name="baseline_nemotron_full",
        dataset=filter_nemotron_full_step / "*.jsonl.gz",
        tokenizer=TOKENIZER,
    )

    # DCLM filter produces 27K+ shards, 90% empty (20 bytes each).
    # The Levanter tokenizer crashes on empty files (IndexError).
    # Consolidate into fewer non-empty shards first.
    consolidate_dclm_step = ExecutorStep(
        name="filtered/baseline_dclm_resharded",
        description="Consolidate sparse DCLM filter output (27K shards, 90% empty) into fewer non-empty shards.",
        fn=consolidate_dclm,
        config=ExtractTextConfig(
            input_path=filter_dclm_step / "*.jsonl.gz",
            output_path=this_output_path(),
        ),
    )

    tokenize_dclm = default_tokenize(
        name="baseline_dclm",
        dataset=consolidate_dclm_step / "*.jsonl.gz",
        tokenizer=TOKENIZER,
    )

    tokenize_fineweb = default_tokenize(
        name="baseline_fineweb_edu",
        dataset=filter_fineweb_step / "*.jsonl.gz",
        tokenizer=TOKENIZER,
    )

    tokenize_resiliparse = default_tokenize(
        name="baseline_resiliparse",
        dataset=extract_text / "*.jsonl.gz",
        tokenizer=TOKENIZER,
    )

    # --- Step 2c: Rename html->text for raw HTML tokenization ---

    rename_html = ExecutorStep(
        name="extracted/baseline_raw_html",
        description="Rename html->text field from downloaded WARCs for raw HTML tokenization.",
        fn=remote(rename_html_to_text, resources=ResourceConfig(cpu=4, ram="16g")),
        config=ExtractTextConfig(
            input_path=download_warcs / "*.jsonl.gz",
            output_path=this_output_path(),
        ),
    )

    tokenize_raw_html = default_tokenize(
        name="baseline_raw_html",
        dataset=rename_html / "*.jsonl.gz",
        tokenizer=TOKENIZER,
    )

    by_product = {
        "nemotron": tokenize_nemotron,
        "nemotron_full": tokenize_nemotron_full,
        "dclm": tokenize_dclm,
        "fineweb": tokenize_fineweb,
        "resiliparse": tokenize_resiliparse,
        "raw_html": tokenize_raw_html,
    }
    return [by_product[p] for p in products]


# --- Remote step bodies (referenced by build_steps) ---


def extract_text_fast(config: ExtractTextConfig) -> None:
    """Wrapper around resiliparse extraction with 500 workers (upstream defaults to 128)."""
    pipeline = (
        Dataset.from_files(config.input_path)
        .load_file()
        .map(_extract_text)
        .filter(_is_non_empty)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz", skip_existing=True)
    )
    # 32 GB/worker: the zephyr default (1 GB) OOM-kills the per-shard subprocess on
    # the largest WARCs (resiliparse holds a whole WARC's HTML in memory). Does not
    # affect the executor step version (body is not hashed), so re-runs reuse output.
    ctx = ZephyrContext(name="extract-text-resiliparse", max_workers=500, resources=ResourceConfig(cpu=4, ram="32g"))
    ctx.put("config", config)
    ctx.execute(pipeline)


def consolidate_dclm(config: ExtractTextConfig) -> None:
    """Re-shard DCLM output into fewer non-empty files.

    The DCLM filter produces 27K+ shards (one per input), 90% empty.
    Reshard into 100 output files so the tokenizer doesn't choke on empties.
    """
    pipeline = (
        Dataset.from_files(config.input_path)
        .load_file()
        .reshard(100)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-00100.jsonl.gz")
    )
    ctx = ZephyrContext(name="consolidate-dclm", max_workers=100)
    ctx.execute(pipeline)


def rename_html_to_text(config: ExtractTextConfig) -> None:
    """Rename 'html' field to 'text' so the tokenizer can read it."""

    def _rename(record: dict) -> dict:
        return {"text": record.get("html", ""), "url": record.get("url", "")}

    pipeline = (
        Dataset.from_files(config.input_path)
        .load_file()
        .map(_rename)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz", skip_existing=True)
    )
    ctx = ZephyrContext(name="rename-html-to-text", max_workers=500)
    ctx.execute(pipeline)


def _parse_args() -> argparse.Namespace:
    """Consume our flags from sys.argv via parse_known_args.

    executor_main is wrapped in ``@draccus.wrap()`` which re-reads sys.argv
    into an ``ExecutorMainConfig``. Leaving our flags in sys.argv crashes that
    downstream parse with "unrecognized arguments". So we use
    ``parse_known_args`` to split, then rewrite sys.argv to retain only the
    unknowns (which are draccus's to interpret).
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help=f"WARC manifest path, one s3://commoncrawl/...warc.gz per line (default: {DEFAULT_MANIFEST}).",
    )
    p.add_argument(
        "--products",
        nargs="+",
        choices=PRODUCT_NAMES,
        default=PRODUCT_NAMES,
        help="Which end products to build (default: all). Scope to e.g. "
        "`dclm nemotron_full resiliparse` to skip unneeded scans.",
    )
    args, unknown = p.parse_known_args()
    # Hand the remainder back to draccus by overwriting sys.argv.
    sys.argv = [sys.argv[0], *unknown]
    return args


# --- Entry point ---

if __name__ == "__main__":
    args = _parse_args()
    if not Path(args.manifest).is_file():
        raise SystemExit(f"manifest not found: {args.manifest}")

    label = manifest_label(args.manifest)
    executor_main(
        steps=build_steps(args.manifest, args.products),
        description=(f"Baseline dataset collection (manifest label={label!r}, " f"products={','.join(args.products)})."),
    )
