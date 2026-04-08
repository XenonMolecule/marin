# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test extraction via the standard executor framework (handles TPU env correctly).

This uses ExecutorStep + remote() which is the proven path for vLLM-on-TPU jobs.
"""

from fray.v2 import ResourceConfig, TpuConfig
from marin.execution.executor import ExecutorStep, executor_main
from marin.execution.remote import remote

from experiments.baseline_collection.download_and_extract import (
    DownloadAndExtractConfig,
    run_download_and_extract,
)

# --- Config ---
WARC_MANIFEST = "gs://marin-us-central1/tmp/extraction_test/test_manifest.txt"
REPHRASER_CKPT = "gs://marin-us-central1/checkpoints/qwen3-8b-rephraser-sft-v4-193d7b/hf/step-1318"

EXTRACTION_SPEC = (
    "Extract the content from this HTML page as clean text. Follow all rules below.\n\n"
    "1. Extract the full page content in reading order. Keep all explanatory text, "
    "discussion, and comments that add substantive information.\n"
    "2. Remove boilerplate: navigation bars, footers, sidebars, ads, share buttons, "
    "related links, breadcrumbs, cookie banners, and user interface elements.\n"
    "3. Preserve all technical content exactly as written: code, math notation, "
    "formulas, tables, and data.\n"
    "4. Decode HTML entities to their plain characters (e.g. &amp; to &, &lt; to <, "
    "&gt; to >). Remove any raw HTML tags.\n"
    "5. Every sentence in your output must come from the source page. Do not add, "
    "invent, or embellish content.\n"
    "6. Output exactly [NO_USEFUL_CONTENT] if ANY of these apply:\n"
    "   - Login, signup, paywall, error page, or empty page\n"
    "   - Index page, directory listing, search results, or navigation-only page\n"
    "   - The page has under ~50 words of substantive content after removing boilerplate"
)

SYSTEM_MESSAGE = (
    "Your input fields are:\n"
    "1. `html` (str): \n"
    "2. `extraction_spec` (str):\n"
    "Your output fields are:\n"
    "1. `text` (str):\n"
    "All interactions will be structured in the following way, "
    "with the appropriate values filled in.\n\n"
    "[[ ## html ## ]]\n{html}\n\n"
    "[[ ## extraction_spec ## ]]\n{extraction_spec}\n\n"
    "[[ ## text ## ]]\n{text}\n\n"
    "[[ ## completed ## ]]\n"
    "In adhering to this structure, your objective is: \n"
    "        Extract the main content text from a given HTML document."
)

TEMPLATE = (
    "[[ ## html ## ]]\n{example}\n\n"
    "[[ ## extraction_spec ## ]]\n" + EXTRACTION_SPEC + "\n\n"
    "Respond with the corresponding output fields, "
    "starting with the field `[[ ## text ## ]]`, "
    "and then ending with the marker for `[[ ## completed ## ]]`."
)


extract_step = ExecutorStep(
    name="documents/baseline_llm_extraction_test",
    fn=remote(
        run_download_and_extract,
        pip_dependency_groups=["vllm", "tpu"],
        resources=ResourceConfig(cpu=8, ram="32g", device=TpuConfig(variant="v5p-8")),
    ),
    config=DownloadAndExtractConfig(
        warc_manifest_path=WARC_MANIFEST,
        output_subdir="documents/baseline_llm_extraction_test",
        # Single-region model reference to avoid cross-region GCS error.
        # The worker resolves its region at runtime and picks the matching entry.
        model_name_by_region={
            "us-central1": REPHRASER_CKPT,
        },
        template=TEMPLATE,
        system_message=SYSTEM_MESSAGE,
        max_doc_tokens=28672,
        engine_kwargs={"max_model_len": 32768, "enable_prefix_caching": True},
        generation_kwargs={"temperature": 0.0, "max_tokens": 4096},
        num_workers=1,
        max_records_per_generate=500,
    ),
)

executor_main(
    steps=[extract_step],
    description="Test: 2-WARC LLM extraction",
)
