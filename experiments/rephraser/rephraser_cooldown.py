# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rephraser cooldown experiment: mix rephraser data into the cooldown phase of a real WSD run.

Instead of training from scratch on tiny rephraser data (rephraser_sweep_v2.py), this experiment:
1. Takes a WSD-trained 1.385B Qwen3 model at step 35,000 (right before cooldown)
2. Extracts the EXACT nemotron cooldown tokens (steps 35k-45k) into a NemotronCooldown dataset
3. Mixes rephraser-extracted tokens into NemotronCooldown for the cooldown phase
4. Resumes from checkpoint with optimizer state preserved, linear LR decay to 0

The ONLY independent variable between conditions is the data mix.
Baseline (nemotron-only cooldown) is defined but NOT auto-executed.

Source run: exp2166-scaling-ladder-nemotron-validation-optimal-1e+20
  - Model: Qwen3Config(hidden=1792, layers=18, heads=14, intermediate=7168) ~1.385B params
  - Optimizer: CautiousConfig (beta1=0.95, beta2=0.9899, eps=1e-15, wd=0.1, adamc_wd=True)
  - WSD schedule: 10% warmup, 70% stable, 20% linear decay to 0
  - Total steps: 44,759 | Cooldown starts ~step 35,808
  - Batch size: 64 | Seq len: 4,096
  - Original final eval (step 44,758): loss=2.828, bpb=0.966, paloma=1.1105, uncheatable=0.8727

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN <your-hf-token> \\
        -- python experiments/rephraser/rephraser_cooldown.py

Dry run:
    python experiments/rephraser/rephraser_cooldown.py --dry_run
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, replace
from datetime import timedelta

import fsspec
import jmp
import numpy as np

from fray.cluster import ResourceConfig
from levanter.data.text import DatasetComponent, LmDataConfig, TextLmDatasetFormat, UrlDatasetSourceConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.main import train_lm
from levanter.main.train_lm import TrainLmConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.cautious import CautiousConfig
from levanter.schedule import BatchSchedule
from levanter.store.cache import SerialCacheWriter
from levanter.checkpoint import CheckpointerConfig
from levanter.trainer import TrainerConfig
from haliax.partitioning import ResourceAxis
from levanter.utils.mesh import MeshConfig

from experiments.defaults import default_validation_sets
from marin.execution.remote import remote
from experiments.evals.task_configs import CORE_TASKS, convert_to_levanter_task_config
from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets import NEMOTRON_WEIGHTS, tokenize_nemotron
from experiments.pretraining_datasets.dclm import dclm_components_llama3
from marin.datakit.download.commoncrawl.download_warc import WarcDownloadConfig, download_and_extract_warcs
from marin.execution.executor import (
    ExecutorStep,
    ensure_versioned,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)
from marin.generation.inference_v2 import InferenceV2Config, run_inference_v2
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import lm_mixture_data_config, step_to_lm_mixture_component
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm
from marin.transform.filter_by_token_length import FilterByTokenLengthConfig, filter_by_token_length
from marin.transform.postprocess_extraction import PostProcessExtractionConfig, postprocess_extraction

from levanter.tracker.wandb import WandbConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source run constants (verified from exp2166 scaling ladder analysis)
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = (
    "gs://marin-us-central1/exp2166-scaling-ladder-nemotron-validation-optimal-1e+20-9563f0" "/checkpoints/step-35000"
)
RESUME_STEP = 35_000
NUM_TRAIN_STEPS = 44_759
COOLDOWN_STEPS = NUM_TRAIN_STEPS - RESUME_STEP  # 9,759
BATCH_SIZE = 64
SEQ_LEN = 4096
LEARNING_RATE = 0.33 * (64**0.5) / 1792  # 0.001473214285714286

# Nemotron mix weights (the 9-component mix from the original 1e20 run)
NEMOTRON_MIX_WEIGHTS = {
    **NEMOTRON_WEIGHTS,
    "starcoderdata": 0.25,
    "proofpile_2": 0.055,
}

# Exact model config for the 1e20 FLOPs scaling law prediction
scaling_1e20_qwen3 = Qwen3Config(
    hidden_dim=1792,
    intermediate_dim=7168,
    num_layers=18,
    num_heads=14,
    num_kv_heads=14,  # MHA — matches predict_optimal_config which sets num_kv_heads=num_heads
    max_seq_len=4096,
    rope=Llama3RotaryEmbeddingsConfig(),
)

# Exact optimizer from the source run
cooldown_optimizer = CautiousConfig(
    learning_rate=LEARNING_RATE,
    weight_decay=0.1,
    min_lr_ratio=0.0,
    warmup=0,  # no warmup — straight into decay
    decay=1.0,  # 100% of steps are linear decay
    beta1=0.95,
    beta2=0.9899494936611666,  # max(0.95, 0.98^(64/128))
    epsilon=1e-15,
    max_grad_norm=1.0,
    adamc_weight_decay=True,
    lr_schedule="linear",
)

# ---------------------------------------------------------------------------
# Rephraser configuration
# ---------------------------------------------------------------------------
REPHRASER_MODEL = "gs://marin-us-central1/checkpoints/qwen3-8b-rephraser-sft-v4-193d7b/hf/step-1318"
REPHRASER_TOKENIZER = "Qwen/Qwen3-8B"
WARC_MANIFEST = os.path.join(os.path.dirname(__file__), "warc_paths.txt")

SPECS = [
    """Extract the main content from the provided HTML into clean Markdown.

First, check if the page should be rejected. Output exactly [NO_USEFUL_CONTENT] if ANY of these apply:
- Not primarily in English
- Login, signup, account, checkout, paywall, or subscribe page
- Error page, captcha, cookie wall, bot check, or "session expired"
- Empty or near-empty page, directory index, or navigation-only page
- User profile, member page, or "who posted" page
- Image gallery or photo album listing without articles
- Search results page with no actual results
- Page where the main content is behind a login wall or paywall
- Product listing, gift card, or e-commerce page with prices/availability
- Social media post that is just an image or a single short caption
- Blog tag page, category page, or archive page that only lists post titles and teasers
- After removing boilerplate, the remaining useful text would be under ~100 words

If the page passes, extract with these rules:
- Output Markdown only. No commentary or analysis.
- Preserve original wording. Do not summarize or rewrite.
- Remove boilerplate: navbars, footers, sidebars, ads, share buttons, related links, breadcrumbs.
- Preserve all technical content exactly: code blocks verbatim with language tags, math/LaTeX using $$ delimiters, chemical formulas, tables.
- Do not truncate or simplify content due to length.
- Include comments/replies only if they add real information (answers, corrections).
- Start with the page title as a top-level heading if available.
""",
]

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

USER_TEMPLATE_FMT = (
    "[[ ## html ## ]]\n{{example}}\n\n"
    "[[ ## extraction_spec ## ]]\n{spec}\n\n"
    "Respond with the corresponding output fields, "
    "starting with the field `[[ ## text ## ]]`, "
    "and then ending with the marker for `[[ ## completed ## ]]`."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def spec_hash(spec_text: str) -> str:
    """Stable 8-char ID from spec content."""
    return hashlib.sha256(spec_text.encode()).hexdigest()[:8]


def load_warc_paths(manifest_path: str) -> list[str]:
    with open(manifest_path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


MAX_MIXIN_FRACTION = 0.30


def _validate_mixin_fraction(
    mixin_name: str,
    mixin_tokens: int,
    cooldown_tokens: int,
    num_train_steps: int,
    max_fraction: float = MAX_MIXIN_FRACTION,
) -> None:
    """Crash if the mixin data fraction is suspiciously high.

    This guards against token-count bugs (e.g. _read_token_count returning
    rows * seq_len instead of actual tokens) silently producing a garbage
    training run.
    """
    total = mixin_tokens + cooldown_tokens
    fraction = mixin_tokens / total
    training_budget = num_train_steps * BATCH_SIZE * SEQ_LEN
    mixin_epochs = (fraction * training_budget) / mixin_tokens if mixin_tokens > 0 else 0
    cooldown_epochs = ((1 - fraction) * training_budget) / cooldown_tokens

    logger.info(
        f"=== Mixin Sanity Check ({mixin_name}) ===\n"
        f"  {mixin_name} tokens:  {mixin_tokens:,}\n"
        f"  cooldown tokens:      {cooldown_tokens:,}\n"
        f"  {mixin_name} fraction: {fraction:.4f}\n"
        f"  {mixin_name} epochs:  {mixin_epochs:.2f}\n"
        f"  cooldown epochs:      {cooldown_epochs:.2f}\n"
        f"  max allowed fraction: {max_fraction}"
    )

    if fraction > max_fraction:
        raise ValueError(
            f"{mixin_name} fraction is {fraction:.4f} ({mixin_tokens:,} / {total:,}), "
            f"which exceeds max_fraction={max_fraction}. "
            f"This likely means _read_token_count returned the wrong value. "
            f"If this fraction is intentional, pass max_mixin_fraction= to the config."
        )


def _read_token_count(cache_path: str, split: str | None = None) -> int:
    """Read total token count from a cache's metadata.

    Prefers .stats.json (which tracks actual pre-padding token count from the tokenize
    pipeline) over shard_ledger.json (which counts rows — only correct for fixed-length
    caches like extract_cooldown_data output).

    Checks two locations in order:
    1. {base}/.stats.json with "total_tokens" key (newer tokenize pipeline)
    2. {base}/part-*/.stats.json with "token_count" key (older tokenize pipeline)

    Raises ValueError if no token count can be determined, rather than silently
    falling back to shard_ledger rows × SEQ_LEN (which is wrong for variable-length
    document tokenizations).

    Args:
        cache_path: Root path of the cache (e.g. GCS output of a tokenize or extraction step).
        split: Optional subdirectory (e.g. "train") for tokenize-pipeline outputs that write
            into split subdirectories. SerialCacheWriter outputs don't use splits.
    """
    base = os.path.join(cache_path, split) if split else cache_path

    stats_path = os.path.join(base, ".stats.json")
    fs, _, _ = fsspec.get_fs_token_paths(stats_path)
    if fs.exists(stats_path):
        with fs.open(stats_path, "r") as f:
            stats = json.load(f)
        if stats.get("total_tokens", 0) > 0:
            return stats["total_tokens"]

    # Check per-shard stats files (older pipeline writes "token_count" to part-*/.stats.json
    # but not to the top-level .stats.json).
    try:
        shard_dirs = [p for p in fs.ls(base, detail=False) if "/part-" in p]
        total_from_shards = 0
        for shard_dir in sorted(shard_dirs):
            shard_stats_path = os.path.join(shard_dir, ".stats.json")
            if fs.exists(shard_stats_path):
                with fs.open(shard_stats_path, "r") as f:
                    shard_stats = json.load(f)
                total_from_shards += shard_stats.get("token_count", 0)
        if total_from_shards > 0:
            logger.info(
                f"_read_token_count: used per-shard stats for {base} "
                f"(total_tokens={total_from_shards:,} from {len(shard_dirs)} shards)"
            )
            return total_from_shards
    except (FileNotFoundError, OSError):
        pass

    raise ValueError(
        f"Could not determine token count for cache at {base}. "
        f"No .stats.json with 'total_tokens' found at {stats_path}, "
        f"and no per-shard part-*/.stats.json with 'token_count' found either. "
        f"If this is a fixed-length cache (e.g. from extract_cooldown_data), "
        f"ensure a .stats.json with 'total_tokens' is written during extraction."
    )


# ---------------------------------------------------------------------------
# Step 6: NemotronCooldown Extraction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExtractCooldownConfig:
    """Config for extracting exact cooldown tokens from the original nemotron mix."""

    nemotron_data_config: LmDataConfig
    start_step: int  # 35,000
    end_step: int  # 44,759
    batch_size: int  # 64
    seq_len: int  # 4,096
    output_path: str


def _extract_tokens_from_example(example) -> np.ndarray:
    """Extract a flat int32 token array from an LmExample or dict."""
    if hasattr(example, "tokens"):
        # LmExample — .tokens is a haliax NamedArray
        return np.asarray(example.tokens.array, dtype=np.int32)
    elif isinstance(example, dict) and "input_ids" in example:
        return np.asarray(example["input_ids"], dtype=np.int32)
    else:
        return np.asarray(example, dtype=np.int32)


def extract_cooldown_data(config: ExtractCooldownConfig):
    """Extract the exact nemotron tokens from cooldown steps of the original 1e20 run.

    Reconstructs the original MixtureDataset, computes the sequence offsets for
    steps [start_step, end_step), and writes those sequences to a new TreeCache.
    This guarantees no second-epoch data since we extract exactly the tokens that
    WOULD have been consumed during cooldown.
    """
    import asyncio
    import gc

    import haliax as hax
    import jax

    schedule = BatchSchedule(config.batch_size)
    start_offset = schedule.global_data_offset_by_step(config.start_step)
    end_offset = schedule.global_data_offset_by_step(config.end_step)
    total_sequences = end_offset - start_offset

    logger.info("=== Extracting NemotronCooldown ===")
    logger.info(f"Original run: step {config.start_step} -> {config.end_step}")
    logger.info(f"Start sequence offset: {start_offset:,}")
    logger.info(f"End sequence offset: {end_offset:,}")
    logger.info(f"Total sequences to extract: {total_sequences:,}")
    logger.info(f"Estimated tokens: {total_sequences * config.seq_len:,}")

    # Build the original dataset from the data config.
    # train_set() returns the training AsyncDataset (MixtureDataset) configured with
    # shuffling, permutation, and mixture block assignments matching the original run.
    pos_axis = hax.Axis("position", config.seq_len)
    key = jax.random.PRNGKey(0)
    dataset = config.nemotron_data_config.train_set(pos_axis, schedule, key=key)

    # Exemplar for the cache: one sequence of int32 token IDs
    exemplar = {"input_ids": np.zeros((config.seq_len,), dtype=np.int32)}

    num_written = 0
    # Use smaller batch size to reduce memory pressure
    fetch_batch_size = 64

    async def _extract():
        nonlocal num_written
        # Write to train/ subdirectory so LmDataConfig.build_caches("train")
        # can find the cache at {cache_dir}/train/ (matches Levanter convention).
        train_path = os.path.join(config.output_path, "train")
        with SerialCacheWriter(train_path, exemplar) as writer:
            for batch_start in range(start_offset, end_offset, fetch_batch_size):
                batch_end = min(batch_start + fetch_batch_size, end_offset)
                indices = list(range(batch_start, batch_end))
                examples = await dataset.get_batch(indices)

                cache_batch = [{"input_ids": _extract_tokens_from_example(ex)} for ex in examples]
                writer.write_batch(cache_batch)
                num_written += len(cache_batch)

                if num_written % 10_000 < fetch_batch_size:
                    logger.info(f"  Extracted {num_written:,}/{total_sequences:,} sequences")

                # Periodically free memory to prevent accumulation
                if num_written % 50_000 < fetch_batch_size:
                    gc.collect()

    asyncio.run(_extract())

    # Write .stats.json so _read_token_count can find the token count without
    # falling back to shard_ledger heuristics.
    total_tokens = num_written * config.seq_len
    train_path = os.path.join(config.output_path, "train")
    stats_path = os.path.join(train_path, ".stats.json")
    with fsspec.open(stats_path, "w") as f:
        json.dump({"total_tokens": total_tokens, "total_elements": 0}, f)

    logger.info(f"Extraction complete. Wrote {num_written:,} sequences to {config.output_path}")
    logger.info(f"Total tokens extracted: {total_tokens:,}")


# ---------------------------------------------------------------------------
# Step 7: Cooldown Training (runtime function)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CooldownTrainingConfig:
    """Config for cooldown training with rephraser data mixed in."""

    rephraser_tokenized_path: str
    cooldown_tokenized_path: str
    output_path: str
    cooldown_component: DatasetComponent
    rephraser_component: DatasetComponent
    rephraser_name: str
    spec_id: str
    validation_configs: dict[str, DatasetComponent] | None = None
    max_mixin_fraction: float = MAX_MIXIN_FRACTION


def run_cooldown_training(config: CooldownTrainingConfig):
    """Read token counts, compute exact weights, build mixture, train.

    The weight computation ensures exactly 1 epoch of rephraser data and
    ~1 epoch of NemotronCooldown data over the cooldown steps.
    """
    rephraser_tokens = _read_token_count(config.rephraser_tokenized_path, split="train")
    cooldown_tokens = _read_token_count(config.cooldown_tokenized_path, split="train")

    # Weights proportional to token counts gives ~1 epoch of each.
    # Levanter normalizes internally, so absolute values set the ratio.
    cooldown_weight = float(cooldown_tokens)
    rephraser_weight = float(rephraser_tokens)
    total_tokens = cooldown_tokens + rephraser_tokens
    rephraser_frac = rephraser_tokens / total_tokens

    _validate_mixin_fraction(
        mixin_name="rephraser",
        mixin_tokens=rephraser_tokens,
        cooldown_tokens=cooldown_tokens,
        num_train_steps=COOLDOWN_STEPS,
        max_fraction=config.max_mixin_fraction,
    )

    logger.info("=== Rephraser Cooldown Training ===")
    logger.info(f"Checkpoint: {CHECKPOINT_PATH}")
    logger.info("Model params: ~1,384,579,840 (Qwen3 1.385B)")
    logger.info(f"Rephraser tokenized path: {config.rephraser_tokenized_path}")
    logger.info(f"NemotronCooldown tokenized path: {config.cooldown_tokenized_path}")
    logger.info(f"Rephraser token count: {rephraser_tokens:,}")
    logger.info(f"Rephraser sequence count: {rephraser_tokens // SEQ_LEN:,}")
    logger.info(f"NemotronCooldown token count: {cooldown_tokens:,}")
    logger.info(f"NemotronCooldown sequence count: {cooldown_tokens // SEQ_LEN:,}")
    logger.info(f"Total cooldown sequences: {COOLDOWN_STEPS * BATCH_SIZE:,}")
    logger.info(f"Rephraser fraction of total: {rephraser_frac:.4f}")
    logger.info(f"Computed rephraser weight: {rephraser_weight:.6f}")
    logger.info(f"Computed cooldown weight: {cooldown_weight:.6f}")
    logger.info(f"Cooldown steps: {COOLDOWN_STEPS}")
    logger.info(f"LR: {LEARNING_RATE:.6f} -> 0 (linear decay over {COOLDOWN_STEPS} steps)")
    logger.info("TPU: v5p-8")
    logger.info("=== Comparison Target (original 1e20 run) ===")
    logger.info("Original eval/loss=2.828, bpb=0.966, paloma=1.1105, uncheatable=0.8727")

    # Build LmDataConfig with two components: NemotronCooldown + Rephraser
    components = {
        "nemotron_cooldown": config.cooldown_component,
        config.rephraser_name: config.rephraser_component,
    }
    weights = {
        "nemotron_cooldown": cooldown_weight,
        config.rephraser_name: rephraser_weight,
    }

    data = LmDataConfig(
        components=components,
        train_weights=weights,
        tokenizer=llama3_tokenizer,
        cache_dir=None,
        shuffle=True,
        permutation_type="feistel",
    )

    # Add validation configs (weight 0) for eval during training
    if config.validation_configs:
        new_components = {
            **data.components,
            **{k: v for k, v in config.validation_configs.items() if k not in data.components},
        }
        new_weights = {
            **data.train_weights,
            **{name: 0.0 for name in config.validation_configs if name not in data.train_weights},
        }
        data = replace(data, components=new_components, train_weights=new_weights)

    inner_config = TrainLmConfig(
        data=data,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="marin",
                tags=[
                    "rephraser-cooldown",
                    f"spec-{config.spec_id}",
                    f"rephraser-tokens={rephraser_tokens}",
                    f"rephraser-frac={rephraser_frac:.4f}",
                    f"cooldown-steps={COOLDOWN_STEPS}",
                ],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=BATCH_SIZE,
            num_train_steps=COOLDOWN_STEPS,
            steps_per_eval=1000,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=COOLDOWN_STEPS)],
            ),
            mesh=MeshConfig(
                compute_mapping={
                    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                }
            ),
            allow_nondivisible_batch_size=True,
        ),
        train_seq_len=SEQ_LEN,
        model=scaling_1e20_qwen3,
        optimizer=cooldown_optimizer,
        initialize_from_checkpoint_path=CHECKPOINT_PATH,
        eval_harness=train_lm.LmEvalHarnessConfig(task_spec=convert_to_levanter_task_config(CORE_TASKS)),
        eval_harness_steps=COOLDOWN_STEPS - 1,
    )

    pod_config = TrainLmOnPodConfig(
        train_config=inner_config,
        resources=ResourceConfig.with_tpu("v5p-8"),
        output_path=config.output_path,
    )

    logger.info(f"Launching cooldown training with resources: {pod_config.resources}")
    run_levanter_train_lm(pod_config)


# ---------------------------------------------------------------------------
# Baseline: NemotronCooldown-only (no rephraser)
# Defined for import, NOT included in all_train_steps.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BaselineCooldownConfig:
    """Config for baseline cooldown training without rephraser data."""

    cooldown_tokenized_path: str
    output_path: str
    cooldown_component: DatasetComponent
    validation_configs: dict[str, DatasetComponent] | None = None


def run_baseline_cooldown_training(config: BaselineCooldownConfig):
    """Nemotron-only cooldown as control. Same checkpoint, LR, optimizer -- no rephraser."""
    cooldown_tokens = _read_token_count(config.cooldown_tokenized_path, split="train")

    logger.info("=== BASELINE Cooldown Training (no rephraser) ===")
    logger.info(f"Checkpoint: {CHECKPOINT_PATH}")
    logger.info(f"NemotronCooldown token count: {cooldown_tokens:,}")
    logger.info(f"NemotronCooldown sequence count: {cooldown_tokens // SEQ_LEN:,}")
    logger.info(f"Cooldown steps: {COOLDOWN_STEPS}")
    logger.info(f"LR: {LEARNING_RATE:.6f} -> 0 (linear decay over {COOLDOWN_STEPS} steps)")
    logger.info("=== Comparison Target (original 1e20 run) ===")
    logger.info("Original eval/loss=2.828, bpb=0.966, paloma=1.1105, uncheatable=0.8727")

    components: dict[str, DatasetComponent] = {
        "nemotron_cooldown": config.cooldown_component,
    }
    weights = {
        "nemotron_cooldown": 1.0,
    }

    data = LmDataConfig(
        components=components,
        train_weights=weights,
        tokenizer=llama3_tokenizer,
        cache_dir=None,
        shuffle=True,
        permutation_type="feistel",
    )

    if config.validation_configs:
        new_components = {
            **data.components,
            **{k: v for k, v in config.validation_configs.items() if k not in data.components},
        }
        new_weights = {
            **data.train_weights,
            **{name: 0.0 for name in config.validation_configs if name not in data.train_weights},
        }
        data = replace(data, components=new_components, train_weights=new_weights)

    inner_config = TrainLmConfig(
        data=data,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="marin",
                tags=[
                    "rephraser-cooldown",
                    "baseline",
                    f"cooldown-steps={COOLDOWN_STEPS}",
                ],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=BATCH_SIZE,
            num_train_steps=COOLDOWN_STEPS,
            steps_per_eval=1000,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=COOLDOWN_STEPS)],
            ),
            mesh=MeshConfig(
                compute_mapping={
                    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                }
            ),
            allow_nondivisible_batch_size=True,
        ),
        train_seq_len=SEQ_LEN,
        model=scaling_1e20_qwen3,
        optimizer=cooldown_optimizer,
        initialize_from_checkpoint_path=CHECKPOINT_PATH,
        eval_harness=train_lm.LmEvalHarnessConfig(task_spec=convert_to_levanter_task_config(CORE_TASKS)),
        eval_harness_steps=COOLDOWN_STEPS - 1,
    )

    pod_config = TrainLmOnPodConfig(
        train_config=inner_config,
        resources=ResourceConfig.with_tpu("v5p-8"),
        output_path=config.output_path,
    )

    logger.info(f"Launching baseline cooldown training with resources: {pod_config.resources}")
    run_levanter_train_lm(pod_config)


# ---------------------------------------------------------------------------
# Build the pipeline
# ---------------------------------------------------------------------------

# Step 1: Download & Extract HTML from WARCs
# SAME name as rephraser_sweep — reuses cached output
warc_paths = load_warc_paths(WARC_MANIFEST)

download_warcs = ExecutorStep(
    name="raw/commoncrawl/rephraser_sweep_batch0",
    description="Download WARC files from Common Crawl and extract HTML.",
    fn=download_and_extract_warcs,
    config=WarcDownloadConfig(
        warc_paths=versioned(tuple(warc_paths)),
        output_path=this_output_path(),
    ),
)

# Step 1b: Pre-filter HTML by token length
# SAME name as rephraser_sweep_v2 — reuses cached output
filter_html = ExecutorStep(
    name="filtered/rephraser_sweep_batch0_v2",
    description=f"Filter HTML documents exceeding {32768 - 4096} tokens.",
    fn=filter_by_token_length,
    config=FilterByTokenLengthConfig(
        input_path=download_warcs / "*.jsonl.gz",
        output_path=this_output_path(),
        tokenizer=REPHRASER_TOKENIZER,
        text_column="html",
        max_tokens=32768 - 4096,
    ),
)

# Step 6: NemotronCooldown Extraction (shared across all specs and baseline)
# Build the EXACT same nemotron mix as the original 1e20 run
nemotron_steps = tokenize_nemotron()
starcoderdata_step = dclm_components_llama3["starcoderdata"]
proofpile_2_step = dclm_components_llama3["proofpile_2"]

nemotron_base_data = lm_mixture_data_config(
    components={**nemotron_steps, "starcoderdata": starcoderdata_step, "proofpile_2": proofpile_2_step},
    weights=NEMOTRON_MIX_WEIGHTS,
    shuffle=True,
    # The original 1e20 run used linear permutation (from .executor_info)
    # lm_mixture_data_config defaults permutation_type to "feistel", but we need "linear"
    # to match the original run's data ordering exactly.
)
# Override permutation_type to match the original run
nemotron_base_data = replace(nemotron_base_data, permutation_type="linear")

extract_cooldown_step = ExecutorStep(
    name="tokenized/nemotron_cooldown_1e20",
    description="Extract exact cooldown tokens (steps 35k-45k) from the original 1e20 nemotron run.",
    fn=extract_cooldown_data,
    config=ExtractCooldownConfig(
        nemotron_data_config=nemotron_base_data,
        start_step=RESUME_STEP,
        end_step=NUM_TRAIN_STEPS,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        output_path=this_output_path(),
    ),
)

# Validation sets (Paloma + Uncheatable Eval)
validation_steps = default_validation_sets(tokenizer=llama3_tokenizer)
validation_component_configs = {
    name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
}

# Pre-build the cooldown component (shared across specs).
# Can't use step_to_lm_mixture_component here because ExtractCooldownConfig is not a TokenizeConfig.
# Build the DatasetComponent manually, pointing at the extract step's output cache.
# We provide a UrlDatasetSourceConfig with empty URLs so that Levanter's build_caches("validation")
# gracefully skips this component (source.get_shard_source("validation") returns None → skip).
# Without a source, build_caches tries to load from {cache_dir}/validation/ and fails hard.
cooldown_component = DatasetComponent(
    source=UrlDatasetSourceConfig(
        train_urls=[],
        validation_urls=[],
        cache_dir=output_path_of(extract_cooldown_step),
        format=TextLmDatasetFormat(),
        tags=["nemotron_cooldown"],
    ),
    cache_dir=output_path_of(extract_cooldown_step),
    format=TextLmDatasetFormat(),
    tags=["nemotron_cooldown"],
)

# ---------------------------------------------------------------------------
# Steps 2-5, 7: Per-spec pipeline
# ---------------------------------------------------------------------------
all_train_steps: list[ExecutorStep] = []

for spec_text in SPECS:
    sid = spec_hash(spec_text)
    user_template = USER_TEMPLATE_FMT.format(spec=spec_text)

    # Step 2: Inference (inference_v2)
    # SAME name as rephraser_sweep_v2 — reuses cached output
    inference_step = ExecutorStep(
        name=f"documents/rephraser_spec_{sid}_v2",
        description=f"Run rephraser inference_v2 for spec {sid}.",
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

    # Step 3: Post-process
    # SAME name as rephraser_sweep_v2 — reuses cached output
    postprocess_step = ExecutorStep(
        name=f"processed/rephraser_spec_{sid}_v2",
        description=f"Post-process extraction output for spec {sid}.",
        fn=postprocess_extraction,
        config=PostProcessExtractionConfig(
            input_path=inference_step / "*.jsonl.gz",
            output_path=this_output_path(),
        ),
    )

    # Step 4: Tokenize with nemotron-compatible tokenizer (Meta-Llama-3.1-8B, NOT 3.2-1B)
    # NEW name (suffixed _cooldown) to avoid conflicting with rephraser_sweep_v2's tokenize step
    tokenize_step = ExecutorStep(
        name=f"tokenized/rephraser_spec_{sid}_cooldown",
        description=f"Tokenize extracted text for spec {sid} (llama3 tokenizer).",
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=[postprocess_step / "*.jsonl.gz"],
            validation_paths=[],
            cache_path=this_output_path(),
            tokenizer=ensure_versioned(llama3_tokenizer),
            format=TextLmDatasetFormat(),
        ),
    )

    # Step 7: Cooldown training with rephraser data mixed in
    rephraser_component = step_to_lm_mixture_component(tokenize_step, include_raw_paths=False)

    train_step = ExecutorStep(
        name=f"cooldown-rephraser-{sid}-v2",
        description=f"Cooldown training for spec {sid}: NemotronCooldown + rephraser mix.",
        fn=run_cooldown_training,
        config=CooldownTrainingConfig(
            rephraser_tokenized_path=tokenize_step,
            cooldown_tokenized_path=extract_cooldown_step,
            output_path=this_output_path(),
            cooldown_component=cooldown_component,
            rephraser_component=rephraser_component,
            rephraser_name=f"rephraser_{sid}",
            spec_id=sid,
            validation_configs=validation_component_configs,
        ),
        # No resources/pip_dependency_groups here — run_cooldown_training calls
        # run_levanter_train_lm which handles TPU allocation internally,
        # matching the pattern used by default_train() in experiments/defaults.py.
    )

    all_train_steps.append(train_step)


# Baseline step (importable, NOT in all_train_steps)
baseline_cooldown_step = ExecutorStep(
    name="cooldown-baseline-nemotron-v2",
    description="Baseline cooldown: NemotronCooldown only, no rephraser.",
    fn=run_baseline_cooldown_training,
    config=BaselineCooldownConfig(
        cooldown_tokenized_path=extract_cooldown_step,
        output_path=this_output_path(),
        cooldown_component=cooldown_component,
        validation_configs=validation_component_configs,
    ),
    # No resources/pip_dependency_groups — run_baseline_cooldown_training calls
    # run_levanter_train_lm internally, which handles TPU allocation.
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    executor_main(
        steps=all_train_steps,  # Only rephraser experiments, NOT baseline
        description="Rephraser cooldown: mix rephraser data into nemotron 1e20 cooldown phase.",
    )
