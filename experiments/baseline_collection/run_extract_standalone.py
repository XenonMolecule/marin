# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone WARC download + LLM extraction — runs directly on a TPU node.

Per-batch checkpointing: each batch of records writes its own output file to GCS
immediately after generation. On preemption + restart, completed batches are
skipped. Maximum work lost = one batch (~2 min at batch_size=100).

Output layout per WARC::

    output_dir/data-{warc_hash}/
        batch_0000.jsonl.gz
        batch_0001.jsonl.gz
        ...
        _done              # empty marker written after all batches complete

Usage::

    iris job run --tpu v5p-8 --memory 128GB --extra marin:vllm --extra marin:tpu \
        -- python experiments/baseline_collection/run_extract_standalone.py \
        --manifest gs://bucket/manifest.txt --output-subdir documents/baseline_llm_extraction
"""

import argparse
import gzip
import json
import logging
import os
import re
import time
from typing import Any

import fsspec

from experiments.baseline_collection.download_warcs import (
    _download_one_warc,
    _load_manifest,
    _warc_path_hash,
)

logger = logging.getLogger(__name__)

# Regex patterns from postprocess_extraction.py
_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
_FIELD_MARKERS_RE = re.compile(r"\[\[\s*##\s*(text|completed)\s*##\s*\]\]", flags=re.IGNORECASE)
_FILTER_PATTERNS = [re.compile(r"\[NO_USEFUL_CONTENT\]")]
MIN_OUTPUT_CHARS = 50
MAX_DOC_TOKENS = 26624  # 32768 context - 6144 output
# 500 records per batch ≈ 12 min on v5p-8. vLLM's continuous batching makes
# larger batches more efficient (slow prompts overlap with fast ones).
# Per-batch checkpointing means max 12 min lost on preemption.
DEFAULT_BATCH_SIZE = 500


def _normalize_record_id(raw_id: str) -> str:
    """Strip <urn:uuid:...> wrapper → bare UUID for DCLM joins."""
    return raw_id.strip("<>").removeprefix("urn:uuid:")


def _extract_snapshot(warc_path: str) -> str:
    """Extract CC-MAIN-YYYY-WW snapshot from a WARC path for FineWeb-Edu joins."""
    m = re.search(r"CC-MAIN-\d{4}-\d{2}", warc_path)
    return m.group(0) if m else ""


def _clean_text(raw_text: str) -> str:
    text = _THINK_RE.sub("", raw_text)
    # Handle unclosed <think> tags (model didn't output </think>):
    # strip everything from <think> onward
    think_pos = text.find("<think>")
    if think_pos != -1:
        text = text[:think_pos]
    text = _FIELD_MARKERS_RE.sub("", text)
    return text.strip()


def _filter_by_length(records: list[dict], max_tokens: int) -> list[dict]:
    """Character-only length filter (no tokenizer calls — instant)."""
    max_chars = max_tokens * 6
    return [r for r in records if len(r.get("html", "")) <= max_chars]


def _batch_output_path(warc_dir: str, batch_idx: int) -> str:
    return f"{warc_dir}/batch_{batch_idx:04d}.jsonl.gz"


def _done_marker_path(warc_dir: str) -> str:
    return f"{warc_dir}/_done"


def _find_completed_batches_in_dir(warc_dir: str) -> set[int]:
    """Scan a single GCS directory for completed batch files."""
    try:
        files = fsspec.filesystem("gcs").glob(f"{warc_dir.replace('gs://', '')}/batch_*.jsonl.gz")
    except Exception:
        return set()
    completed = set()
    for f in files:
        basename = os.path.basename(f)
        try:
            idx = int(basename.split("_")[1].split(".")[0])
            completed.add(idx)
        except (IndexError, ValueError):
            continue
    return completed


def _find_completed_batches_all_regions(warc_hash: str, output_subdir: str) -> set[int]:
    """Scan ALL regional buckets for completed batch files.

    Enables cross-region resume: Job A writes batches 0-20 in us-central1,
    gets preempted. Job B picks up in eu-west4 and skips batches 0-20.
    """
    from iris.marin_fs import REGION_TO_DATA_BUCKET

    completed = set()
    for bucket in REGION_TO_DATA_BUCKET.values():
        warc_dir = f"gs://{bucket}/{output_subdir}/data-{warc_hash}"
        completed |= _find_completed_batches_in_dir(warc_dir)
    return completed


def _is_warc_done(warc_dir: str) -> bool:
    """Check if the _done marker exists for this WARC."""
    try:
        return fsspec.filesystem("gcs").exists(warc_dir.replace("gs://", "") + "/_done")
    except Exception:
        return False


def _is_warc_done_any_region(warc_hash: str, output_subdir: str) -> bool:
    """Check if a WARC is done in ANY regional bucket."""
    from iris.marin_fs import REGION_TO_DATA_BUCKET

    gcs = fsspec.filesystem("gcs")
    for bucket in REGION_TO_DATA_BUCKET.values():
        path = f"{bucket}/{output_subdir}/data-{warc_hash}/_done"
        try:
            if gcs.exists(path):
                return True
        except Exception:
            continue
    return False


def _is_warc_claimed_any_region(warc_hash: str, output_subdir: str, stale_hours: float = 3.0) -> bool:
    """Check if a WARC is claimed (in-progress) by another job in any region.

    A claim is a ``_claimed`` file in the WARC output dir. Claims older than
    ``stale_hours`` are ignored (the claiming job probably died without
    finishing or releasing the claim).
    """
    from iris.marin_fs import REGION_TO_DATA_BUCKET

    gcs = fsspec.filesystem("gcs")
    now = time.time()
    for bucket in REGION_TO_DATA_BUCKET.values():
        path = f"{bucket}/{output_subdir}/data-{warc_hash}/_claimed"
        try:
            info = gcs.info(path)
            # Check if the claim is stale
            mtime = info.get("updated") or info.get("timeCreated")
            if mtime is not None:
                import datetime

                if isinstance(mtime, str):
                    mtime = datetime.datetime.fromisoformat(mtime.replace("Z", "+00:00"))
                age_hours = (now - mtime.timestamp()) / 3600
                if age_hours > stale_hours:
                    logger.info("Stale claim on %s (%.1fh old), ignoring", warc_hash, age_hours)
                    continue
            return True
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return False


def _claim_warc(warc_dir: str) -> None:
    """Write a claim marker indicating this job is processing the WARC."""
    claim_path = f"{warc_dir}/_claimed"
    with fsspec.open(claim_path, "w") as f:
        f.write(json.dumps({"pid": os.getpid(), "time": time.time(), "host": os.environ.get("HOSTNAME", "unknown")}))


def _write_batch_output(path: str, records: list[dict]) -> None:
    """Write a batch of records to a gzipped JSONL file on GCS."""
    with fsspec.open(path, "wb") as f:
        with gzip.open(f, "wt", encoding="utf-8") as gz:
            for rec in records:
                gz.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_done_marker(warc_dir: str, stats: dict) -> None:
    """Write the _done marker with summary stats."""
    with fsspec.open(_done_marker_path(warc_dir), "w") as f:
        json.dump(stats, f, indent=2)


def _process_batch(
    batch: list[dict],
    llm: Any,
    sampling_params: Any,
    tokenizer: Any,
    template: str,
    system_message: str,
) -> tuple[list[dict], int, int]:
    """Process a single batch through vLLM. Returns (output_records, kept, filtered)."""
    from vllm.inputs.data import TokensPrompt

    # Format prompts
    prompts = []
    for record in batch:
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
        prompt_ids = tokenizer.encode(prompt_text)
        prompts.append(TokensPrompt(prompt_token_ids=prompt_ids))

    # Filter empty prompts
    valid = [(i, p) for i, p in enumerate(prompts) if p["prompt_token_ids"]]
    if not valid:
        return [], 0, len(batch)

    valid_indices, valid_prompts = zip(*valid, strict=True)

    # Generate
    t0 = time.monotonic()
    outputs = llm.generate(list(valid_prompts), sampling_params)
    elapsed = time.monotonic() - t0
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    logger.info(
        "Generated %d prompts in %.1fs (%.1f tok/s)",
        len(valid_prompts),
        elapsed,
        total_tokens / max(elapsed, 0.01),
    )

    # Map outputs back and post-process
    output_map = {}
    for idx, out in zip(valid_indices, outputs, strict=True):
        output_map[idx] = " ".join(o.text for o in out.outputs)

    output_records = []
    kept = 0
    filtered = 0
    for i, record in enumerate(batch):
        raw = output_map.get(i, "")
        cleaned = _clean_text(raw)
        if len(cleaned) < MIN_OUTPUT_CHARS:
            filtered += 1
            continue
        if any(p.search(cleaned) for p in _FILTER_PATTERNS):
            filtered += 1
            continue
        kept += 1
        warc_file = record.get("metadata", {}).get("warc_file", "")
        output_records.append(
            {
                "text": cleaned,
                "generated_text": raw,
                "url": record.get("url", ""),
                # Join keys for DCLM / Nemotron / FineWeb-Edu filtering
                "warc_record_id": _normalize_record_id(record.get("id", "")),
                "warc_file": warc_file,
                "snapshot": _extract_snapshot(warc_file),
            }
        )

    return output_records, kept, filtered


def _process_warc(
    warc_path: str,
    output_dir: str,
    output_subdir: str,
    llm: Any,
    sampling_params: Any,
    tokenizer: Any,
    template: str,
    system_message: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    """Download one WARC, extract via LLM with per-batch checkpointing."""
    h = _warc_path_hash(warc_path)
    warc_dir = f"{output_dir}/data-{h}"

    # Skip if fully done (check local region first, then all regions)
    if _is_warc_done(warc_dir) or _is_warc_done_any_region(h, output_subdir):
        return {"warc": warc_path, "status": "skipped"}

    # Skip if another job claimed it (unless the claim is stale)
    if _is_warc_claimed_any_region(h, output_subdir):
        logger.info("Skipping %s (claimed by another job)", warc_path)
        return {"warc": warc_path, "status": "claimed"}

    # Claim this WARC so other jobs skip it
    _claim_warc(warc_dir)
    logger.info("Claimed %s -> %s", warc_path, warc_dir)

    # Download
    records = _download_one_warc(warc_path)
    if not records:
        logger.warning("No HTML records from %s", warc_path)
        _write_done_marker(warc_dir, {"status": "empty", "records": 0})
        return {"warc": warc_path, "status": "empty", "records": 0}

    logger.info("Downloaded %d records from %s", len(records), warc_path)

    # Filter by length (instant)
    records = _filter_by_length(records, MAX_DOC_TOKENS)
    logger.info("%d records after length filter", len(records))
    if not records:
        _write_done_marker(warc_dir, {"status": "all_filtered", "records": 0})
        return {"warc": warc_path, "status": "all_filtered", "records": 0}

    # Check which batches are already done (scan ALL regions for cross-region resume)
    completed_batches = _find_completed_batches_all_regions(h, output_subdir)
    num_batches = (len(records) + batch_size - 1) // batch_size

    if completed_batches:
        logger.info(
            "Resuming: %d/%d batches already complete (possibly across regions), skipping them",
            len(completed_batches),
            num_batches,
        )

    # Process batches with per-batch checkpointing
    total_kept = 0
    total_filtered = 0

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(records))
        batch = records[batch_start:batch_end]

        # Skip if this batch is already done (in any region)
        if batch_idx in completed_batches:
            logger.info("Batch %d/%d: already done, skipping", batch_idx + 1, num_batches)
            continue

        logger.info(
            "Batch %d/%d (%d records, offset %d-%d)...", batch_idx + 1, num_batches, len(batch), batch_start, batch_end
        )

        # Process batch
        output_records, kept, filtered = _process_batch(batch, llm, sampling_params, tokenizer, template, system_message)
        total_kept += kept
        total_filtered += filtered

        # Write batch output IMMEDIATELY to GCS (checkpoint)
        batch_path = _batch_output_path(warc_dir, batch_idx)
        _write_batch_output(batch_path, output_records)

        # Refresh claim so other jobs know we're still alive (3h stale timeout)
        _claim_warc(warc_dir)

        logger.info(
            "Batch %d/%d: kept=%d, filtered=%d -> %s",
            batch_idx + 1,
            num_batches,
            kept,
            filtered,
            batch_path,
        )

    # All batches done — write completion marker
    stats = {
        "warc": warc_path,
        "total_records": len(records),
        "total_kept": total_kept,
        "total_filtered": total_filtered,
        "num_batches": num_batches,
        "batch_size": batch_size,
    }
    _write_done_marker(warc_dir, stats)
    logger.info("WARC complete: %s (kept=%d, filtered=%d)", warc_path, total_kept, total_filtered)
    return {"warc": warc_path, "status": "done", **stats}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="GCS path to WARC manifest")
    parser.add_argument("--output-subdir", default="documents/baseline_llm_extraction_test")
    parser.add_argument("--model", default=None, help="Model path (auto-resolves from region if not set)")
    parser.add_argument("--tp", type=int, default=None, help="Tensor parallel size (auto-detect from JAX)")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Records per batch (smaller = finer checkpoints)"
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Shuffle WARC order with this seed. Use different seeds per job to spread work.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index into manifest (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End index into manifest (exclusive). Default: all.")
    args = parser.parse_args()

    # Resolve output path from runtime region
    from iris.marin_fs import marin_prefix

    output_dir = f"{marin_prefix()}/{args.output_subdir}"
    logger.info("Output dir: %s", output_dir)

    # Resolve model
    if args.model:
        model_name = args.model
    else:
        model_name = "gs://marin-us-central1/checkpoints/qwen3-8b-rephraser-sft-v4-193d7b/hf/step-1318"
    logger.info("Model: %s", model_name)

    # Set JAX cache env
    marin_pfx = os.environ.get("MARIN_PREFIX")
    cache_dir = os.path.join(marin_pfx, "compilation-cache") if marin_pfx else "/tmp/marin-jax-compilation-cache"
    os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", cache_dir)
    os.environ.setdefault("VLLM_XLA_CACHE_PATH", cache_dir)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # Detect TPU
    import jax

    devices = jax.devices()
    logger.info("JAX devices (%d): %s", len(devices), devices)

    tp = args.tp or len([d for d in devices if d.platform == "tpu"]) or len(devices)
    logger.info("Tensor parallel size: %d", tp)

    # Init vLLM engine
    from vllm import LLM, SamplingParams

    t0 = time.monotonic()
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=6144)
    tokenizer = llm.get_tokenizer()
    logger.info("Engine loaded in %.1fs", time.monotonic() - t0)

    # Extraction prompt
    spec = (
        "Extract the content from this HTML page as clean text. Follow all rules below.\n\n"
        "1. Extract the full page content in reading order. Keep all explanatory text, "
        "discussion, and comments that add substantive information. Begin your output "
        "directly with the page content.\n"
        "2. Remove boilerplate: navigation bars, footers, sidebars, ads, share buttons, "
        "related links, breadcrumbs, cookie banners, and user interface elements. "
        "Do not output framework markers or metadata tags.\n"
        "3. Preserve all technical content exactly as written: code, math notation, "
        "formulas, tables, and data. Preserve the original line breaks and structure "
        "of code blocks.\n"
        "4. Decode all HTML entities to their plain characters (e.g. &amp; to &, "
        "&lt; to <, &gt; to >, &#8217; to ', &#8211; to \u2013). Remove any raw HTML tags.\n"
        "5. Every sentence in your output must come from the source page. Do not add, "
        "invent, or embellish content.\n"
        "6. For pages with multiple authors or speakers (forums, reviews, comments), "
        "preserve who said what. Include usernames or speaker labels so contributions "
        "remain distinguishable.\n"
        "7. If the text contains content spinner templates like {{word1|word2|word3}}, "
        "pick the first option and output clean text. If the majority of the page is "
        "spinner templates, output [NO_USEFUL_CONTENT] instead.\n"
        "8. Output exactly [NO_USEFUL_CONTENT] if ANY of these apply:\n"
        '   - Login, signup, paywall, registration wall, or "you must sign up to view" page\n'
        "   - Error page, empty page, or cookie/captcha wall\n"
        "   - Terms of use, privacy policy, or legal boilerplate page\n"
        "   - User profile page with no substantive content\n"
        "   - Page whose content has been removed, moved, or is no longer available\n"
        "   - Image gallery, photo album listing, or media archive without articles\n"
        "   - The text is incoherent gibberish or garbled encoding throughout\n"
        "   - The page has under ~50 words of substantive content after removing boilerplate\n"
        "   However, for index pages or directory listings, only filter if they contain "
        "nothing but links and titles. If an index page includes real text like discussion "
        "snippets or descriptions, extract it."
    )
    system_message = (
        "Your input fields are:\n1. `html` (str): \n2. `extraction_spec` (str):\n"
        "Your output fields are:\n1. `text` (str):\n"
        "All interactions will be structured in the following way, "
        "with the appropriate values filled in.\n\n"
        "[[ ## html ## ]]\n{html}\n\n[[ ## extraction_spec ## ]]\n{extraction_spec}\n\n"
        "[[ ## text ## ]]\n{text}\n\n[[ ## completed ## ]]\n"
        "In adhering to this structure, your objective is: \n"
        "        Extract the main content text from a given HTML document."
    )
    template = (
        "[[ ## html ## ]]\n{example}\n\n"
        "[[ ## extraction_spec ## ]]\n" + spec + "\n\n"
        "Respond with the corresponding output fields, "
        "starting with the field `[[ ## text ## ]]`, "
        "and then ending with the marker for `[[ ## completed ## ]]`."
    )

    # Load manifest, optionally slice and shuffle
    import random

    warcs = _load_manifest(args.manifest)
    warcs = warcs[args.start : args.end]  # default: all
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(warcs)
        logger.info(
            "Manifest: %d WARCs [%d:%s], shuffled with seed %d", len(warcs), args.start, args.end, args.shuffle_seed
        )
    else:
        logger.info("Manifest: %d WARCs [%d:%s], sequential order", len(warcs), args.start, args.end)

    logger.info("batch_size=%d, output_subdir=%s", args.batch_size, args.output_subdir)

    # Dynamic processing: iterate all WARCs, skip done ones (checked across all regions).
    # Multiple jobs with different shuffle seeds naturally spread across the manifest.
    stats = []
    for i, warc in enumerate(warcs):
        logger.info("=== WARC %d/%d ===", i + 1, len(warcs))
        s = _process_warc(
            warc,
            output_dir,
            args.output_subdir,
            llm,
            sampling_params,
            tokenizer,
            template,
            system_message,
            batch_size=args.batch_size,
        )
        stats.append(s)
        if s["status"] != "skipped":
            logger.info("Result: %s", s)

    # Summary
    done = sum(1 for s in stats if s["status"] == "done")
    skipped = sum(1 for s in stats if s["status"] == "skipped")
    claimed = sum(1 for s in stats if s["status"] == "claimed")
    logger.info("Complete: %d done, %d skipped, %d claimed-by-other, %d total", done, skipped, claimed, len(stats))


if __name__ == "__main__":
    main()
