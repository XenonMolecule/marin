# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SFT Qwen3-0.6B to distill the high_quality web extractor (non-thinking), with a
hyperparameter sweep over two data scales.

Datasets (built by ``build_hq_distill_chat.py``, temporally balanced over 35
snapshots): ``bal350k`` (~350k useful, ~9B train tokens) and ``bal3p5m`` (~3.5M,
~94B). HPs may differ by data scale, so each dataset gets its own full sweep
(no transfer).

Pipeline per dataset: full-fit filter (keep iff the whole chat sequence <= 32k;
keeps the 22-token [NO_USEFUL_CONTENT] targets, never truncates) -> tokenize with
the Qwen3 chat template -> single-epoch SFT from Qwen/Qwen3-0.6B.

Sweep (16 configs/dataset, all independent -> parallel, no staging):
  * Core LR x BS (the dominant axis): LR in {5e-7,1e-6,2e-6,5e-6,1e-5} x BS in {32,64}.
  * Secondary knobs probed at a fixed center (lr=2e-6, bs=32) since they are
    low-sensitivity and ~orthogonal to LR: beta2 in {0.999,0.9999}, weight_decay
    in {0.01,0.1}, plus two delphi-inspired probes (max_grad_norm 0.1; linear
    schedule + warmup 0.1).
Runs are differentiated by val loss (logged to wandb; tagged by dataset+config).

All runs fit on v5p-8 (bs<=64 @ 32k: ~40 GB logits in 95 GB HBM). bal3p5m's 10x
data => ~10x steps (~5 d/run) but same per-step memory; triage by killing weak
runs as val loss separates.

Phases / selectors (env vars):
  * ``HQ_DISTILL_PHASE`` = tokenize (default) | train
  * ``HQ_DISTILL_TAG``   = bal350k | bal3p5m | both (default; train phase only)
Launch bal350k and bal3p5m as separate jobs so bal350k (the ~12 h runs) can be
prioritized ahead of bal3p5m.
"""

import dataclasses
import math
import os

from fray.cluster import ResourceConfig
from levanter.data.text import ChatLmDatasetFormat
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from marin.execution.executor import ExecutorStep, InputName, executor_main, this_output_path
from marin.execution.remote import remote
from marin.processing.tokenize import lm_data_config

from experiments.baseline_collection.filter_fits_context import FilterFitsContextConfig, filter_fits_context
from experiments.chat_templates.qwen3_chat_template import QWEN_3_CHAT_TEMPLATE
from experiments.defaults import default_sft, default_tokenize
from experiments.qwen3 import qwen3_0_6b_hd128
from experiments.simple_sft_config import SimpleSFTConfig

TAGS = ("bal350k", "bal3p5m")
QWEN3_TOKENIZER = "Qwen/Qwen3-0.6B"
MAX_SEQ_LEN = 32_768
SURVIVAL = 0.93  # fraction surviving the <=32k full-fit filter (measured on the smoke sample)
TRAIN_TPU = "v6e-8"  # v5p-8 was slice_failed-scarce for 25h+; v6e-8 (32 GB HBM) needs pdp=1+ce_block (proven by canary2 step 164)

# Path (relative to MARIN_PREFIX) of the chat datasets from build_hq_distill_chat.py.
DATA_REL = "datasets/high_quality_3000_distill_chat"
# Small val subset (10 of 35 held-out WARCs) — eval-loss curve without bloating the cache.
VAL_GLOB = "val/data-0000?.jsonl.gz"

CHAT_FORMAT = ChatLmDatasetFormat(chat_template=QWEN_3_CHAT_TEMPLATE, pack=1)
NO_TORCH_ENV = {
    "TRANSFORMERS_NO_TORCH": "1",
    "TRANSFORMERS_NO_TORCHVISION": "1",
    "USE_TORCH": "0",
    "TORCH_DISABLE_GLOBAL_DEPS": "1",
}

# Manifest train totals (useful=neg, 1:1); * SURVIVAL ~= post-filter examples.
# Refined from filter_stats.json before launch.
MANIFEST_TRAIN_TOTAL = {"bal350k": 696_954, "bal3p5m": 6_954_292}
# Sparse eval — validation is a time sink; just need a clean curve + a guaranteed
# final eval. bal350k is ~10-20k steps, bal3p5m ~100-200k.
EVAL_EVERY = {"bal350k": 2_500, "bal3p5m": 5_000}
MAX_EVAL_BATCHES = 50  # ~50*bs val examples per eval — enough for a loss signal, cheap

# --- Sweep grid ------------------------------------------------------------

LRS = (5e-7, 1e-6, 2e-6, 5e-6, 1e-5)
BSS = (32, 64)
CENTER_LR, CENTER_BS = 2e-6, 32  # fixed center for the secondary-knob probes


@dataclasses.dataclass(frozen=True)
class HP:
    """One sweep config. Defaults match the Levanter AdamW defaults that survived
    prior Qwen3-0.6B SFT sweeps (beta2=None -> 0.95; wd 0; cosine; grad-norm 1)."""

    lr: float
    bs: int
    beta2: float | None = None
    weight_decay: float = 0.0
    warmup: float = 0.03
    lr_schedule: str = "cosine"
    max_grad_norm: float = 1.0

    @property
    def name(self) -> str:
        parts = [f"lr{self.lr:.0e}".replace("e-0", "e-").replace("e+0", "e"), f"bs{self.bs}"]
        if self.beta2 is not None:
            parts.append(f"b2{self.beta2}")
        if self.weight_decay != 0.0:
            parts.append(f"wd{self.weight_decay}")
        if self.max_grad_norm != 1.0:
            parts.append(f"gn{self.max_grad_norm}")
        if self.lr_schedule != "cosine":
            parts.append(self.lr_schedule)
        return "-".join(parts)


def sweep_configs() -> list[HP]:
    core = [HP(lr=lr, bs=bs) for lr in LRS for bs in BSS]  # 10
    secondary = [
        HP(lr=CENTER_LR, bs=CENTER_BS, beta2=0.999),
        HP(lr=CENTER_LR, bs=CENTER_BS, beta2=0.9999),
        HP(lr=CENTER_LR, bs=CENTER_BS, weight_decay=0.01),
        HP(lr=CENTER_LR, bs=CENTER_BS, weight_decay=0.1),
        HP(lr=CENTER_LR, bs=CENTER_BS, max_grad_norm=0.1),  # delphi
        HP(lr=CENTER_LR, bs=CENTER_BS, lr_schedule="linear", warmup=0.1),  # delphi
    ]  # 6
    return core + secondary


# --- Steps -----------------------------------------------------------------


def _filter_step(tag: str, split: str, glob: str) -> ExecutorStep:
    return ExecutorStep(
        name=os.path.join("filtered", f"hq_distill_{tag}_{split}_qwen3_{MAX_SEQ_LEN // 1024}k"),
        description=f"Keep-iff-full-fit filter ({split}) for hq_distill {tag} at {MAX_SEQ_LEN} ctx.",
        fn=remote(
            filter_fits_context,
            resources=ResourceConfig.with_cpu(cpu=8, ram="32g"),
            env_vars=NO_TORCH_ENV,
            pip_dependency_groups=["cpu"],
        ),
        config=FilterFitsContextConfig(
            input_path=InputName.hardcoded(f"{DATA_REL}_{tag}/{glob}"),
            output_path=this_output_path(),
            tokenizer=QWEN3_TOKENIZER,
            seq_len=MAX_SEQ_LEN,
            chat_template=QWEN_3_CHAT_TEMPLATE,
        ),
    )


def _tokenize_steps(tag: str) -> tuple[ExecutorStep, ExecutorStep]:
    """Filter (train + small val) then tokenize each. Returns (train_tok, val_tok)."""
    filtered_train = _filter_step(tag, "train", "train/*.jsonl.gz")
    filtered_val = _filter_step(tag, "val", VAL_GLOB)
    train_tok = default_tokenize(
        name=f"hq_distill_{tag}_qwen3_{MAX_SEQ_LEN // 1024}k",
        dataset=filtered_train / "**/*.jsonl.gz",
        tokenizer=QWEN3_TOKENIZER,
        format=CHAT_FORMAT,
        is_validation=False,
        resources=ResourceConfig.with_cpu(cpu=8, ram="32g", disk="64g"),
    )
    val_tok = default_tokenize(
        name=f"hq_distill_{tag}_val_qwen3_{MAX_SEQ_LEN // 1024}k",
        dataset=filtered_val / "**/*.jsonl.gz",
        tokenizer=QWEN3_TOKENIZER,
        format=CHAT_FORMAT,
        is_validation=True,
        resources=ResourceConfig.with_cpu(cpu=8, ram="32g", disk="16g"),
    )
    return train_tok, val_tok


# Built at import so the tokenize phase has no dependency on the SFT config below.
_TOK = {tag: _tokenize_steps(tag) for tag in TAGS}
TOKENIZE_STEPS = [step for pair in _TOK.values() for step in pair]


def _train_step(tag: str, hp: HP) -> ExecutorStep:
    train_tok, val_tok = _TOK[tag]
    data_config = lm_data_config(training_set=train_tok, validation_sets={f"hq_distill_{tag}_val": val_tok})
    model_config = dataclasses.replace(
        qwen3_0_6b_hd128,
        rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),  # match pretrained Qwen3 RoPE
        max_seq_len=MAX_SEQ_LEN,
    )
    num_train_steps = math.ceil(MANIFEST_TRAIN_TOTAL[tag] * SURVIVAL / hp.bs)
    sft_config = SimpleSFTConfig(
        resources=ResourceConfig.with_tpu(TRAIN_TPU),
        train_batch_size=hp.bs,
        num_train_steps=num_train_steps,
        learning_rate=hp.lr,
        lr_schedule=hp.lr_schedule,
        warmup=hp.warmup,
        decay=0.97,
        min_lr_ratio=0.0,
        weight_decay=hp.weight_decay,
        beta2=hp.beta2,
        max_grad_norm=hp.max_grad_norm,
        tokenizer=QWEN3_TOKENIZER,
        initialize_from_hf="Qwen/Qwen3-0.6B",
        pad_tokenizer_to_match_model=True,
        max_seq_len=MAX_SEQ_LEN,
        steps_per_eval=EVAL_EVERY[tag],
        max_eval_batches=MAX_EVAL_BATCHES,
        steps_per_checkpoint=None,  # keep=[]: no checkpoint museum; rolling ~10-min temp + final only
        steps_per_hf_export=num_train_steps,  # one HF export at the end (for offline eval)
        seed=42,
        z_loss_weight=1e-5,
        skip_bad_steps=True,
        per_device_parallelism=1,  # v6e-8 = 32 GB HBM; pdp=1 + ce_loss_block_size keeps logits in memory
        ce_loss_block_size=1024,  # chunk the 32k x 152k-vocab logits to fit 32 GB (proven by canary2 step 164)
        # Preemption resilience comes from JAX_COMPILATION_CACHE_DIR (set at launch) — persists XLA compile +
        # Pallas autotune across restarts so a restart reaches the 10-min checkpoint instead of re-paying ~76 min.
    )
    return default_sft(
        name=f"qwen3-0.6b-hq-distill-{tag}-{hp.name}",
        tokenized=data_config,
        model_config=model_config,
        sft_config=sft_config,
        tags=["qwen3", "0.6b", "sft", "hq-distill", tag, hp.name],
        wandb_group=f"hq-distill-{tag}",  # all 16 configs of a dataset share one W&B group
    )


def _selected_configs() -> list[HP]:
    """All sweep configs, or a comma-separated subset by name via HQ_DISTILL_CONFIGS
    (e.g. a single canary config to shake out infra before fanning out)."""
    cfgs = sweep_configs()
    only = os.environ.get("HQ_DISTILL_CONFIGS")
    if not only:
        return cfgs
    names = {n.strip() for n in only.split(",") if n.strip()}
    selected = [c for c in cfgs if c.name in names]
    if not selected:
        raise ValueError(f"HQ_DISTILL_CONFIGS={only!r} matched nothing; have {[c.name for c in cfgs]}")
    return selected


def _train_steps(tags: tuple[str, ...]) -> list[ExecutorStep]:
    return [_train_step(tag, hp) for tag in tags for hp in _selected_configs()]


if __name__ == "__main__":
    phase = os.environ.get("HQ_DISTILL_PHASE", "tokenize")
    sel = os.environ.get("HQ_DISTILL_TAG", "both")
    tags = TAGS if sel == "both" else (sel,)
    if any(t not in TAGS for t in tags):
        raise ValueError(f"HQ_DISTILL_TAG must be one of {TAGS} or 'both', got {sel!r}")
    if phase == "tokenize":
        executor_main(steps=[step for tag in tags for step in _TOK[tag]])
    elif phase == "train":
        executor_main(steps=_train_steps(tags))
    else:
        raise ValueError(f"HQ_DISTILL_PHASE must be 'tokenize' or 'train', got {phase!r}")
