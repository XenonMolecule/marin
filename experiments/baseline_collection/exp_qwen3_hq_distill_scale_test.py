# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scale-test: distill the high_quality web extractor into Qwen3-1.7B / 4B on a
~35k-example proxy (1/10 epoch of bal350k, ~0.9B tokens) for fast scale-vs-data
signal before committing the multi-day 350k runs.

Reuses the *existing* tokenized bal350k cache from
``exp_qwen3_0_6b_hq_distill_sft`` (no retokenize): single-epoch SFT, so the
"35k proxy" is just ``num_train_steps = 1/10 epoch`` over the shuffled set.

One run per launch, selected by env so each iris job pins to one region:
  * ``SCALE_MODEL``       = 1.7b | 4b            (required)
  * ``SCALE_LR``          = learning rate float  (required, e.g. 2e-6)
  * ``SCALE_BS``          = global batch size    (default 128)
  * ``SCALE_TPU``         = e.g. v5p-32 | v6e-8  (default v5p-32)
  * ``SCALE_RAM``         = host container RAM    (default by model: 4b="400g" else "128g")
  * ``SCALE_PROXY_FRAC``  = epoch fraction       (default 0.1 = ~35k/0.9B tok)
  * ``SCALE_PREEMPTIBLE`` = 0 to target reserved (default preemptible)
  * ``MARIN_PREFIX``      = gs://marin-<region>  (checkpoint bucket; pin == --region)
"""

import dataclasses
import math
import os

from fray.cluster import ResourceConfig
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_data_config

from experiments.baseline_collection.exp_qwen3_0_6b_hq_distill_sft import (
    _TOK,
    MANIFEST_TRAIN_TOTAL,
    MAX_EVAL_BATCHES,
    MAX_SEQ_LEN,
    QWEN3_TOKENIZER,
    SURVIVAL,
)
from experiments.defaults import default_sft
from experiments.qwen3 import qwen3_0_6b_hd128, qwen3_1_7b, qwen3_4b_hd128
from experiments.simple_sft_config import SimpleSFTConfig

TAG = "bal350k"

# (model_config, HF init checkpoint) per size. head_dim=128 variants match the HF configs.
MODELS = {
    "0.6b": (qwen3_0_6b_hd128, "Qwen/Qwen3-0.6B"),
    "1.7b": (qwen3_1_7b, "Qwen/Qwen3-1.7B"),
    "4b": (qwen3_4b_hd128, "Qwen/Qwen3-4B"),
    # Same 4B architecture as the base, different (instruction-tuned) init weights.
    "4b-instruct": (qwen3_4b_hd128, "Qwen/Qwen3-4B-Instruct-2507"),
}


# Host container RAM per model: the single-host checkpoint serialization gathers the full
# (weights + fp32 master + Adam m/v) state to one VM, so the 4B run needs more than the 128g
# default (it OOM-killed at step 4 serializing its first temp checkpoint).
DEFAULT_RAM = {"1.7b": "128g", "4b": "400g", "4b-instruct": "400g"}


def _build_step(model: str, lr: float, bs: int, tpu: str, proxy_frac: float, preemptible: bool, ram: str):
    if model not in MODELS:
        raise ValueError(f"SCALE_MODEL must be one of {list(MODELS)}, got {model!r}")
    base_cfg, hf_ckpt = MODELS[model]
    train_tok, val_tok = _TOK[TAG]
    data_config = lm_data_config(training_set=train_tok, validation_sets={f"hq_distill_{TAG}_val": val_tok})
    model_config = dataclasses.replace(
        base_cfg,
        rope=DefaultRotaryEmbeddingsConfig(theta=1000000.0, factor=1.0),  # match pretrained Qwen3 RoPE (as 0.6b run)
        max_seq_len=MAX_SEQ_LEN,
    )
    full_epoch_steps = math.ceil(MANIFEST_TRAIN_TOTAL[TAG] * SURVIVAL / bs)
    num_train_steps = max(1, math.ceil(full_epoch_steps * proxy_frac))
    steps_per_eval = max(50, num_train_steps // 5)  # ~5 evals across the short run
    # ~every 250 steps (>=10 across the run): full-epoch runs on preemptible HW MUST checkpoint
    # so they resume after preemption instead of restarting from HF.
    steps_per_checkpoint = max(1, min(num_train_steps // 10, 250))
    sft_config = SimpleSFTConfig(
        resources=ResourceConfig.with_tpu(tpu, preemptible=preemptible, ram=ram),
        train_batch_size=bs,
        num_train_steps=num_train_steps,
        learning_rate=lr,
        lr_schedule="cosine",
        warmup=0.03,
        decay=0.97,
        min_lr_ratio=0.0,
        weight_decay=0.0,
        beta2=None,  # -> 0.95 (matches the 0.6b sweep recipe)
        max_grad_norm=1.0,
        tokenizer=QWEN3_TOKENIZER,  # identical vocab across Qwen3 sizes
        initialize_from_hf=hf_ckpt,
        pad_tokenizer_to_match_model=True,
        max_seq_len=MAX_SEQ_LEN,
        steps_per_eval=steps_per_eval,
        max_eval_batches=MAX_EVAL_BATCHES,
        steps_per_checkpoint=steps_per_checkpoint,  # periodic: survive preemption on full runs
        steps_per_hf_export=num_train_steps,  # one HF export at the end for offline eval
        seed=42,
        z_loss_weight=1e-5,
        skip_bad_steps=True,
        per_device_parallelism=1,  # memory-conservative; chunk logits via ce_loss_block_size
        ce_loss_block_size=1024,
    )
    lr_str = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")
    # SCALE_NAME_SUFFIX gives an independent checkpoint dir + wandb run (e.g. "-bk" for a
    # redundant safety run on different hardware that must NOT collide with the primary).
    suffix = os.environ.get("SCALE_NAME_SUFFIX", "")
    name = f"qwen3-{model}-hq-distill-{TAG}-proxy{int(proxy_frac * 100)}pct-lr{lr_str}-bs{bs}{suffix}"
    print(
        f"[scale-test] {name}: model={model} hf={hf_ckpt} lr={lr} bs={bs} "
        f"steps={num_train_steps} (of {full_epoch_steps} full-epoch) tpu={tpu} preempt={preemptible}"
    )
    return default_sft(
        name=name,
        tokenized=data_config,
        model_config=model_config,
        sft_config=sft_config,
        tags=["qwen3", model, "sft", "hq-distill", TAG, "scale-test", f"lr{lr_str}", f"bs{bs}"],
        wandb_group="hq-distill-scaletest",
    )


if __name__ == "__main__":
    model = os.environ["SCALE_MODEL"]
    lr = float(os.environ["SCALE_LR"])
    bs = int(os.environ.get("SCALE_BS", "128"))
    tpu = os.environ.get("SCALE_TPU", "v5p-32")
    proxy_frac = float(os.environ.get("SCALE_PROXY_FRAC", "0.1"))
    preemptible = os.environ.get("SCALE_PREEMPTIBLE", "1") != "0"
    ram = os.environ.get("SCALE_RAM") or DEFAULT_RAM.get(model, "128g")
    executor_main(steps=[_build_step(model, lr, bs, tpu, proxy_frac, preemptible, ram)])
