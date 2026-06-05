# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Closed-form FLOP calculator for spec-driven extraction.

Calibrated to the 3000-WARC Qwen3-8B baseline (6.72e22 total inference FLOPs,
2.207 T input tokens across all records, from BASELINE_TABLE.md). Lets you
project FLOPs as a function of three knobs:

  * input_frac      (α) — fraction of original input tokens (e.g. 0.9 = -10%)
  * model           — which Qwen3 variant runs the extraction
  * n_warcs         — how many CC WARCs to process

Output tokens are NOT a free knob: they're driven by content per record. The
calculator treats O as a fixed per-record distribution that scales with WARC
count only.

Example:
    python flop_calculator.py --input-frac 0.75 --model qwen3-4b --warcs 10000
"""

import argparse
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Baseline calibration (from experiments/baseline_collection/BASELINE_TABLE.md)
# ---------------------------------------------------------------------------

BASELINE_TOTAL_FLOPS = 6.72e22
BASELINE_INPUT_TOKENS_ALL = 2.207e12
BASELINE_WARCS = 3000
BASELINE_MODEL = "qwen3-8b"

# Component decomposition at baseline (×10^22 FLOPs):
# Prefill non-attn is exact (C_non × sum_input). Attn/decode split derived
# from per-batch CV (11.5%) and kept-records output sum (185.88B). ±2% rounding.
F_PREFILL_NONATTN_BASE = 3.33e22  # ∝ α
F_PREFILL_ATTN_BASE = 2.77e22  # ∝ α²
F_DECODE_NONATTN_BASE = 0.34e22  # α-independent
F_DECODE_ATTN_BASE = 0.28e22  # ≈α (P·O term dominates)


# ---------------------------------------------------------------------------
# Model architecture → per-token cost constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelArch:
    name: str
    n_layers: int
    hidden_dim: int
    intermediate_dim: int
    num_heads: int
    num_kv_heads: int
    vocab_size: int = 151_936  # Qwen3 family

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def c_non_attn(self) -> float:
        """Per-token non-attention FLOPs (MLP + projections + lm_head)."""
        h, inter = self.hidden_dim, self.intermediate_dim
        d, n_h, n_kv = self.head_dim, self.num_heads, self.num_kv_heads
        mlp = 2 * 3 * h * inter
        qkv = 2 * h * (n_h * d + 2 * n_kv * d)
        dense = 2 * h * h
        lm_head = 2 * h * self.vocab_size
        return self.n_layers * (mlp + qkv + dense) + lm_head

    @property
    def c_attn(self) -> float:
        """Per-(token × context-position) attention FLOPs."""
        d, n_h = self.head_dim, self.num_heads
        return self.n_layers * (2 * n_h * d + 3 * n_h + 2 * d * n_h)


# Qwen3 family (configs from HuggingFace). Verify before relying on
# small/large variants for serious estimates.
QWEN3_FAMILY: dict[str, ModelArch] = {
    "qwen3-0.6b": ModelArch(
        "qwen3-0.6b", n_layers=28, hidden_dim=1024, intermediate_dim=3072, num_heads=16, num_kv_heads=8
    ),
    "qwen3-1.7b": ModelArch(
        "qwen3-1.7b", n_layers=28, hidden_dim=2048, intermediate_dim=6144, num_heads=16, num_kv_heads=8
    ),
    "qwen3-4b": ModelArch("qwen3-4b", n_layers=36, hidden_dim=2560, intermediate_dim=9728, num_heads=32, num_kv_heads=8),
    "qwen3-8b": ModelArch(
        "qwen3-8b", n_layers=36, hidden_dim=4096, intermediate_dim=12288, num_heads=32, num_kv_heads=8
    ),
    "qwen3-14b": ModelArch(
        "qwen3-14b", n_layers=40, hidden_dim=5120, intermediate_dim=17408, num_heads=40, num_kv_heads=8
    ),
    "qwen3-32b": ModelArch(
        "qwen3-32b", n_layers=64, hidden_dim=5120, intermediate_dim=25600, num_heads=64, num_kv_heads=8
    ),
}

# Cached baseline constants so we can scale model contribution as a ratio.
_BASELINE_ARCH = QWEN3_FAMILY[BASELINE_MODEL]
_C_NON_BASE = _BASELINE_ARCH.c_non_attn
_C_ATTN_BASE = _BASELINE_ARCH.c_attn


# ---------------------------------------------------------------------------
# The calculator
# ---------------------------------------------------------------------------


def project_flops(input_frac: float, model: str, n_warcs: int) -> dict[str, float]:
    """Project total inference FLOPs.

    Each prefill/decode component scales independently. Per-component model
    scaling factors are separated: non-attention terms ∝ c_non_attn/C_non_base,
    attention terms ∝ c_attn/C_attn_base.
    """
    if model not in QWEN3_FAMILY:
        raise ValueError(f"unknown model {model!r}; known: {sorted(QWEN3_FAMILY)}")
    arch = QWEN3_FAMILY[model]
    warc_scale = n_warcs / BASELINE_WARCS
    beta_non = arch.c_non_attn / _C_NON_BASE
    beta_attn = arch.c_attn / _C_ATTN_BASE
    a = input_frac

    f_pre_non = F_PREFILL_NONATTN_BASE * a * beta_non * warc_scale
    f_pre_attn = F_PREFILL_ATTN_BASE * a * a * beta_attn * warc_scale
    f_dec_non = F_DECODE_NONATTN_BASE * beta_non * warc_scale
    f_dec_attn = F_DECODE_ATTN_BASE * a * beta_attn * warc_scale  # P·O cross-term

    total = f_pre_non + f_pre_attn + f_dec_non + f_dec_attn
    return {
        "prefill_non_attn": f_pre_non,
        "prefill_attn": f_pre_attn,
        "decode_non_attn": f_dec_non,
        "decode_attn": f_dec_attn,
        "total": total,
        "savings_vs_baseline": 1 - total / BASELINE_TOTAL_FLOPS,
        "model": model,
        "input_frac": a,
        "n_warcs": n_warcs,
        "beta_non_attn": beta_non,
        "beta_attn": beta_attn,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-frac", type=float, default=1.0, help="α: fraction of original input tokens (default 1.0)")
    p.add_argument(
        "--model", default="qwen3-8b", choices=sorted(QWEN3_FAMILY), help="Extractor model (default qwen3-8b)"
    )
    p.add_argument("--warcs", type=int, default=BASELINE_WARCS, help=f"Number of WARCs (default {BASELINE_WARCS})")
    args = p.parse_args()

    r = project_flops(args.input_frac, args.model, args.warcs)
    print(f"Model:           {r['model']}")
    print(f"  C_non_attn:    {QWEN3_FAMILY[r['model']].c_non_attn:.3e}  (vs Qwen3-8B β={r['beta_non_attn']:.3f})")
    print(f"  C_attn:        {QWEN3_FAMILY[r['model']].c_attn:.3e}      (vs Qwen3-8B β={r['beta_attn']:.3f})")
    print(f"Input fraction α = {r['input_frac']}")
    print(f"WARCs            = {r['n_warcs']:,} (scale {r['n_warcs']/BASELINE_WARCS:.2f}× of baseline)")
    print()
    print("FLOPs by component:")
    print(f"  Prefill non-attn (∝ α):    {r['prefill_non_attn']:.3e}")
    print(f"  Prefill attn (∝ α²):       {r['prefill_attn']:.3e}")
    print(f"  Decode non-attn (∝ 1):     {r['decode_non_attn']:.3e}")
    print(f"  Decode attn (∝ α):         {r['decode_attn']:.3e}")
    print(f"  TOTAL:                     {r['total']:.3e}")
    print()
    print(f"vs baseline (6.72e22): {100*r['savings_vs_baseline']:+.1f}%  " f"({r['total']/BASELINE_TOTAL_FLOPS:.3f}×)")


if __name__ == "__main__":
    main()
