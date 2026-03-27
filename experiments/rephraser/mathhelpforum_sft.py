# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SFT fine-tuning on mathhelpforum.com extracted math content.

Takes extracted markdown from the mathhelpforum rephraser pipeline and:
1. Parses structured markdown (## Question / ## Reasoning / ## Answer) into chat messages
2. Tokenizes with Llama3 chat template
3. Fine-tunes the exp2166 final checkpoint (~1.385B Qwen3) for 1 epoch
4. Evaluates on GSM8K test set (8-shot CoT)

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/mathhelpforum_sft.py

Dry run:
    MARIN_PREFIX=gs://marin-us-central1 uv run python \\
        experiments/rephraser/mathhelpforum_sft.py --dry_run true
"""

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import timedelta

import fsspec
import jmp
from zephyr import Dataset, ZephyrContext, load_jsonl

from experiments.chat_templates.llama3pt1_chat_template import LLAMA_3_1_CHAT_TEMPLATE
from experiments.defaults import default_tokenize
from experiments.evals.evals import default_eval
from experiments.llama import compute_num_parameters
from experiments.rephraser.mathhelpforum_extract import postprocess_step
from experiments.rephraser.rephraser_cooldown import _read_token_count, scaling_1e20_qwen3
from fray.cluster import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import ChatLmDatasetFormat, LMMixtureDatasetConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.processing.tokenize import lm_data_config
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (shared across GSM8K / mathhelpforum / resiliparse SFT experiments)
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = (
    "gs://marin-us-central1/" "exp2166-scaling-ladder-nemotron-validation-optimal-1e+20-9563f0" "/checkpoints/step-44660"
)
LLAMA3_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"
BATCH_SIZE = 64
SEQ_LEN = 4096


# ---------------------------------------------------------------------------
# Step 1: Parse extracted markdown into chat messages
# ---------------------------------------------------------------------------
@dataclass
class MathForumToChatConfig:
    input_path: str
    output_path: str


def _is_post_boundary(line: str) -> bool:
    """Return True if the line marks the start of a new forum post.

    Handles many rephraser output formats, including lines inside blockquotes.
    """
    stripped = line.strip()
    if not stripped:
        return False

    # Strip leading blockquote markers so we also detect boundaries inside `> ...`
    content = re.sub(r"^(?:>\s*)+", "", stripped)
    if not content:
        return False

    # ## headings that look like post boundaries
    if content.startswith("## "):
        heading = content[3:].strip().lower()
        if any(heading.startswith(p) for p in (
            "question", "answer", "post by", "reply by",
            "original post", "original question",
            "post 1", "post 2", "post 3", "post 4", "post 5",
            "post #",
        )):
            return True

    # Bold post markers
    if content.startswith("**"):
        # **Label** / **Label:** / **Label (info):**
        if re.match(
            r"^\*\*("
            r"Original post|Original question|Reply|Answer|Question|Post"
            r"|Solution|Follow.?up|Hint|Response"
            r")\b",
            content,
            re.IGNORECASE,
        ):
            return True
        # **username** (Month DDth YYYY, ...) — bold name then date in parens
        if re.match(r"^\*\*[^*]+\*\*\s*\([A-Z][a-z]+ \d", content):
            return True
        # **username (Month DDth YYYY, ...):** — date inside the bold
        if re.match(r"^\*\*[^*]+\([A-Z][a-z]+ \d", content):
            return True
        # **username** – Month DDth YYYY — em-dash date separator
        if re.match(r"^\*\*[^*]+\*\*\s*[–—-]\s*[A-Z][a-z]+ \d", content):
            return True

    # ## Name (Month DDth YYYY, ...) — heading with name and date
    if content.startswith("## ") and re.match(r"^## .+\([A-Z][a-z]+ \d", content):
        return True

    return False


def _parse_markdown_to_messages(text: str) -> list[dict[str, str]] | None:
    """Parse rephraser output into OpenAI chat message dicts.

    The rephraser produces various forum-post formats.  This function detects
    post boundaries (## headings, bold labels, bold usernames) and groups them
    into user/assistant turns:
      - First post  → user message
      - All subsequent posts → single assistant message

    Returns a list of {"role": ..., "content": ...} dicts, or None if unparseable.
    """
    if not text or len(text.strip()) < 20:
        return None

    # Extract title (first # heading, if present)
    title = ""
    title_match = re.match(r"^#\s+(.+?)$", text, re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()

    # Split text into posts at boundary lines
    lines = text.split("\n")
    posts: list[str] = []
    current_lines: list[str] = []

    for line in lines:
        # Skip the title line itself
        if line.strip().startswith("# ") and not line.strip().startswith("## "):
            continue

        if _is_post_boundary(line):
            # Flush accumulated lines as a post
            block = "\n".join(current_lines).strip()
            if block:
                posts.append(block)
            current_lines = []
            # Don't include the boundary line itself (it's metadata, not content)
            continue

        # Horizontal rules (---) separate posts in some rephraser outputs.
        # Only treat as a boundary if there is already accumulated content.
        if re.match(r"^-{3,}\s*$", line.strip()) and current_lines:
            block = "\n".join(current_lines).strip()
            if block:
                posts.append(block)
            current_lines = []
            continue

        current_lines.append(line)

    # Flush the last block
    block = "\n".join(current_lines).strip()
    if block:
        posts.append(block)

    # If we found at least 2 posts via explicit boundaries, use them.
    if len(posts) >= 2:
        user_content = posts[0]
        if title:
            user_content = f"{title}\n\n{user_content}"
        assistant_content = "\n\n".join(posts[1:])
        if user_content.strip() and assistant_content.strip():
            return [
                {"role": "user", "content": user_content.strip()},
                {"role": "assistant", "content": assistant_content.strip()},
            ]

    # Fallback: split on blockquote boundaries.  In forum threads, repliers
    # often quote the original question with ``>``.  Treat the first contiguous
    # "type" of lines (blockquote or non-blockquote) as the user question and
    # the first contiguous block of the OTHER type as the assistant answer.
    bq_lines: list[str] = []
    non_bq_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or (stripped.startswith("# ") and not stripped.startswith("## ")):
            continue
        if stripped.startswith(">"):
            bq_lines.append(re.sub(r"^>\s?", "", stripped))
        else:
            non_bq_lines.append(stripped)

    bq_text = "\n".join(bq_lines).strip()
    non_bq_text = "\n".join(non_bq_lines).strip()

    if bq_text and non_bq_text:
        # Determine which came first in the document to assign roles.
        # The first non-title content line determines the "question" type.
        first_is_bq = False
        for line in lines:
            stripped = line.strip()
            if not stripped or (stripped.startswith("# ") and not stripped.startswith("## ")):
                continue
            first_is_bq = stripped.startswith(">")
            break

        if first_is_bq:
            user_content = bq_text
            assistant_content = non_bq_text
        else:
            user_content = non_bq_text
            assistant_content = bq_text

        if title:
            user_content = f"{title}\n\n{user_content}"

        return [
            {"role": "user", "content": user_content.strip()},
            {"role": "assistant", "content": assistant_content.strip()},
        ]

    return None


def _transform_and_filter(records: list[dict]) -> list[dict]:
    """Transform extracted records into chat format, dropping unparseable ones."""
    results = []
    for record in records:
        text = record.get("text", "")
        messages = _parse_markdown_to_messages(text)
        if messages is not None:
            results.append({"messages": messages})
    return results


def transform_mathforum_to_chat(config: MathForumToChatConfig):
    """Parse extracted mathhelpforum markdown into OpenAI chat messages JSONL."""
    pipeline = (
        Dataset.from_files(config.input_path)
        .flat_map(load_jsonl)
        .map_shard(_transform_and_filter)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )

    with ZephyrContext(name="mathforum-to-chat") as ctx:
        output_files = ctx.execute(pipeline)

    stats = {"output_files": len(output_files), "input_path": config.input_path}
    with fsspec.open(f"{config.output_path}/transform_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Transform complete: {len(output_files)} output shards")


chat_transform_step = ExecutorStep(
    name="documents/mathhelpforum_chat_v2",
    description="Parse mathhelpforum extracted markdown into chat messages JSONL.",
    fn=transform_mathforum_to_chat,
    config=MathForumToChatConfig(
        input_path=postprocess_step / "*.jsonl.gz",
        output_path=this_output_path(),
    ),
    resources=ResourceConfig.with_cpu(cpu=4, ram="16g"),
    pip_dependency_groups=["cpu"],
)

# ---------------------------------------------------------------------------
# Step 2: Tokenize with Llama3 chat template
# ---------------------------------------------------------------------------
CHAT_FORMAT = ChatLmDatasetFormat(
    chat_template=LLAMA_3_1_CHAT_TEMPLATE,
    mask_user_turns=True,
    pack=True,
)

mathforum_tokenized = default_tokenize(
    name="mathhelpforum_chat_sft_v2",
    dataset=chat_transform_step / "**/*.jsonl.gz",
    tokenizer=LLAMA3_TOKENIZER,
    format=CHAT_FORMAT,
)

# ---------------------------------------------------------------------------
# Step 3: Data config
# ---------------------------------------------------------------------------
data_config = lm_data_config(
    training_set=mathforum_tokenized,
    validation_sets={},
)


# ---------------------------------------------------------------------------
# Step 4: Single-epoch SFT training (num_train_steps computed at runtime)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _SFTRunConfig:
    tokenized_path: str
    data_config: LMMixtureDatasetConfig
    output_path: str


def _run_single_epoch_sft(config: _SFTRunConfig):
    """Train for exactly 1 epoch, computing num_train_steps from the tokenized cache."""
    total_tokens = _read_token_count(config.tokenized_path, split="train")
    num_train_steps = math.ceil(total_tokens / (BATCH_SIZE * SEQ_LEN))

    logger.info("=== Mathhelpforum Chat Single-Epoch SFT ===")
    logger.info(f"Total tokens: {total_tokens:,}")
    logger.info(f"Num train steps (1 epoch): {num_train_steps}")
    logger.info(f"Batch size: {BATCH_SIZE}, Seq len: {SEQ_LEN}")

    inner_config = TrainLmConfig(
        data=config.data_config,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="marin",
                tags=["mathhelpforum", "sft", "1e20", "qwen3", "single-epoch"],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=BATCH_SIZE,
            num_train_steps=num_train_steps,
            steps_per_eval=min(50, num_train_steps),
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=min(100, num_train_steps))],
            ),
            mesh=MeshConfig(
                compute_mapping={
                    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                }
            ),
            allow_nondivisible_batch_size=True,
            initialize_from=None,
        ),
        initialize_from_checkpoint_path=CHECKPOINT_PATH,
        initialize_from_hf=False,
        train_seq_len=SEQ_LEN,
        model=scaling_1e20_qwen3,
        optimizer=AdamConfig(
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup=0.03,
            decay=0.97,
            lr_schedule="cosine",
            max_grad_norm=1.0,
        ),
        hf_save_steps=num_train_steps,
    )

    pod_config = TrainLmOnPodConfig(
        train_config=inner_config,
        resources=ResourceConfig.with_tpu("v5p-8"),
        output_path=config.output_path,
    )

    run_levanter_train_lm(pod_config)


mathforum_sft_step = ExecutorStep(
    name="checkpoints/mathhelpforum-sft-1e20-qwen3-v2",
    description=(
        f"Single-epoch chat SFT on mathhelpforum extracted math content "
        f"({compute_num_parameters(scaling_1e20_qwen3, 128256):,} params)."
    ),
    fn=_run_single_epoch_sft,
    config=_SFTRunConfig(
        tokenized_path=mathforum_tokenized,
        data_config=data_config,
        output_path=this_output_path(),
    ),
)

# ---------------------------------------------------------------------------
# Step 5: Evaluate on GSM8K test set (8-shot CoT)
# ---------------------------------------------------------------------------
GSM8K_EVAL = [
    EvalTaskConfig(name="gsm8k_cot", num_fewshot=8, task_alias="gsm8k_cot_8shot"),
]

gsm8k_eval_step = default_eval(
    step=mathforum_sft_step,
    evals=GSM8K_EVAL,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    apply_chat_template=True,
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    executor_main(
        steps=[mathforum_sft_step, gsm8k_eval_step],
        description="Mathhelpforum single-epoch chat SFT + GSM8K eval.",
    )
