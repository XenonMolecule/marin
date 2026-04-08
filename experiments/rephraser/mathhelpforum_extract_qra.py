# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract math content from mathhelpforum.com HTML using Q/R/A format.

Uses the same consolidated/filtered HTML as the v1 extraction, but with a
math-specific prompt that produces structured ## Question / ## Reasoning / ## Answer
output instead of raw forum thread markdown.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/mathhelpforum_extract_qra.py

Dry run:
    python experiments/rephraser/mathhelpforum_extract_qra.py --dry_run
"""

import hashlib

from experiments.rephraser.mathhelpforum_extract import (
    REPHRASER_MODEL,
    SYSTEM_MESSAGE,
    USER_TEMPLATE_FMT,
    filter_html,
)
from marin.execution.remote import remote
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.generation.inference_v2 import InferenceV2Config, run_inference_v2
from marin.transform.postprocess_extraction import PostProcessExtractionConfig, postprocess_extraction

# ---------------------------------------------------------------------------
# Q/R/A extraction prompt
# ---------------------------------------------------------------------------
EXTRACTION_SPEC = """\
Extract and restructure this math help forum HTML page into a clean Question/Reasoning/Answer format in Markdown.

First, check if this is an actual math discussion thread with a math question. Output exactly [NO_USEFUL_CONTENT] if ANY of these apply:
- Not a math question/discussion thread (e.g. forum index, category listing, user profile, search results)
- Software announcement, product news, or tool tutorial without a math problem
- Page is empty, login-only, error page, or has no substantive math content
- Not primarily in English

If the page is a valid math thread, extract and restructure it.

Remove all boilerplate: navigation, footers, sidebars, ads, user signatures, join dates, post counts, reaction buttons, and related thread links. Do not include any raw HTML tags in your output.

Output format:
- Start with the thread title as a top-level heading.
- Use up to three sections: ## Question, ## Reasoning, ## Answer.
- **Question**: The original poster's problem, cleaned up but preserving the mathematical content faithfully.
- **Reasoning**: The key steps, explanations, and worked-out logic from the discussion thread. Synthesize the useful replies into a coherent explanation.
- **Answer**: The final answer or solution clearly stated (if it is present in the thread)

Rules:
- Output Markdown only. No meta-commentary.
- Preserve all mathematical content exactly: LaTeX/MathJax using $ or $$ delimiters, equations, formulas, and notation.
- Preserve code blocks verbatim with language tags if applicable.
- If the thread contains multiple distinct questions, use a separate ## Question, ## Reasoning, ## Answer block for each. Solve each question independently — do not mix answers between questions.
- If no clear answer was reached in the thread, omit the Answer section. Do not fabricate or guess an answer.\
"""


def spec_hash(spec_text: str) -> str:
    """Stable 8-char ID from spec content."""
    return hashlib.sha256(spec_text.encode()).hexdigest()[:8]


sid = spec_hash(EXTRACTION_SPEC)
user_template = USER_TEMPLATE_FMT.format(spec=EXTRACTION_SPEC)

# ---------------------------------------------------------------------------
# Step 1: Inference with rephraser model (Q/R/A prompt)
# ---------------------------------------------------------------------------
# Reuses the already-completed filter_html step from the v1 extraction.
inference_step = ExecutorStep(
    name=f"documents/mathhelpforum_qra_{sid}",
    description=f"Run rephraser inference on mathhelpforum HTML with Q/R/A prompt (spec {sid}).",
    fn=remote(run_inference_v2, pip_dependency_groups=["vllm"]),
    config=InferenceV2Config(
        input_path=filter_html / "*.jsonl.gz",
        output_path=this_output_path(),
        model_name=REPHRASER_MODEL,
        input_format="jsonl.gz",
        output_format="jsonl.gz",
        engine_kwargs={
            "max_model_len": 32768,
            "enable_prefix_caching": True,
        },
        generation_kwargs={
            "temperature": 0.0,
            "max_tokens": 4096,
        },
        system_message=SYSTEM_MESSAGE,
        template=user_template,
        prompt_column="html",
        apply_chat_template=True,
        max_doc_tokens=32768 - 4096,
        tensor_parallel_size=4,
        tpu_type="v5p-8",
        num_workers=16,
        records_per_shard=500,
    ),
)

# ---------------------------------------------------------------------------
# Step 2: Post-process extraction output
# ---------------------------------------------------------------------------
postprocess_step_qra = ExecutorStep(
    name=f"processed/mathhelpforum_qra_{sid}",
    description=f"Post-process mathhelpforum Q/R/A extraction output (spec {sid}).",
    fn=postprocess_extraction,
    config=PostProcessExtractionConfig(
        input_path=inference_step / "*.jsonl.gz",
        output_path=this_output_path(),
    ),
)

if __name__ == "__main__":
    executor_main(
        steps=[postprocess_step_qra],
        description="Extract math Q/R/A from mathhelpforum.com HTML pages.",
    )
