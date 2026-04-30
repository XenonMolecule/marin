# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Test three implementations of inference FLOP accounting against each other.

Implementations:
  1. Gold: decomposes into position-independent + attention, closed-form arithmetic series
  2. Student loop: calls lm_flops_per_token per decode step (Part A)
  3. Student closed-form: arithmetic series on the full lm_flops_per_token output (Part B)
"""

from levanter.utils.flop_utils import lm_flops_per_token

# Qwen3-8B config (from experiments/qwen3.py + HF config)
QWEN3_8B = dict(
    hidden_dim=4096,
    intermediate_dim=12288,
    num_layers=36,
    num_heads=32,
    num_kv_heads=8,
    vocab_size=151936,
    glu=True,
)


def _lm_fpt(seq_len: int) -> float:
    return lm_flops_per_token(seq_len=seq_len, **QWEN3_8B)


# ---------- Gold implementation (closed-form, exact) ----------


def inference_flops_gold(prompt_len: int, output_len: int) -> float:
    cfg = QWEN3_8B
    h = cfg["hidden_dim"]
    d = h / cfg["num_heads"]  # head_dim
    n_h = cfg["num_heads"]
    n_kv = cfg["num_kv_heads"]
    n_l = cfg["num_layers"]
    V = cfg["vocab_size"]
    inter = cfg["intermediate_dim"]

    # Position-independent FLOPs per token
    mlp = 2 * 3 * h * inter  # SwiGLU: gate, up, down projections
    qkv_proj = 2 * h * (n_h * d + 2 * n_kv * d)
    dense_proj = 2 * h * h
    lm_head = 2 * h * V
    non_attn_per_token = n_l * (mlp + qkv_proj + dense_proj) + lm_head

    # Attention cost per unit of context length (per layer)
    # For 1 query token attending to C context tokens:
    #   key_query_logits: 2 * C * n_h * d
    #   mask:             3 * C * n_h
    #   value_agg:        2 * C * d * n_h
    attn_per_ctx = n_l * (2 * n_h * d + 3 * n_h + 2 * d * n_h)

    # Prefill: P tokens, each attending to P tokens (full attention assumption)
    # Attention total = P * P * attn_per_ctx_per_layer (quadratic)
    # Which equals P * (attn_per_ctx * P) = P * attn_at_seqlen_P
    prefill = non_attn_per_token * prompt_len + attn_per_ctx * prompt_len * prompt_len

    # Decode: O tokens, step i attends to context of length (P + i)
    # Non-attention: non_attn_per_token * O
    # Attention: attn_per_ctx * sum_{i=0}^{O-1} (P + i)
    #          = attn_per_ctx * (O * P + O * (O - 1) / 2)
    total_ctx = output_len * prompt_len + output_len * (output_len - 1) / 2
    decode = non_attn_per_token * output_len + attn_per_ctx * total_ctx

    return prefill + decode


# ---------- Student Part A: loop over decode steps ----------


def inference_flops_student_loop(prompt_len: int, output_len: int) -> float:
    prefill_flops = prompt_len * _lm_fpt(prompt_len)

    decode_flops = 0
    for i in range(output_len):
        current_seq_len = prompt_len + i
        decode_flops += _lm_fpt(current_seq_len)

    return prefill_flops + decode_flops


# ---------- Student Part B: closed-form arithmetic series ----------


def inference_flops_student_closed(prompt_len: int, output_len: int) -> float:
    flops_first_decode = _lm_fpt(prompt_len)
    prefill_flops = prompt_len * flops_first_decode

    if output_len == 0:
        return prefill_flops

    flops_last_decode = _lm_fpt(prompt_len + output_len - 1)
    decode_flops = (output_len / 2) * (flops_first_decode + flops_last_decode)

    return prefill_flops + decode_flops


# ---------- Test cases ----------

TEST_CASES = [
    # (prompt_len, output_len, description)
    (512, 256, "Short prompt, short output (quick factoid question)"),
    (8192, 2048, "Medium doc extraction (typical WARC page)"),
    (28672, 4096, "Max-length extraction (long HTML, full output budget)"),
]


def fmt(flops: float) -> str:
    if flops >= 1e15:
        return f"{flops:.6e} ({flops/1e15:.2f} PFLOPs)"
    elif flops >= 1e12:
        return f"{flops:.6e} ({flops/1e12:.2f} TFLOPs)"
    else:
        return f"{flops:.6e}"


def main():
    print("Qwen3-8B Inference FLOP Accounting — Three Implementations\n")
    print(f"Config: {QWEN3_8B}\n")

    all_pass = True
    for prompt_len, output_len, desc in TEST_CASES:
        print(f"{'='*70}")
        print(f"Test: {desc}")
        print(f"  prompt_len={prompt_len}, output_len={output_len}")
        print()

        gold = inference_flops_gold(prompt_len, output_len)
        loop = inference_flops_student_loop(prompt_len, output_len)
        closed = inference_flops_student_closed(prompt_len, output_len)

        print(f"  Gold (decomposed):     {fmt(gold)}")
        print(f"  Student loop (Part A): {fmt(loop)}")
        print(f"  Student closed (B):    {fmt(closed)}")
        print()

        # Check exact match (these should be identical in floating point
        # since they compute the same arithmetic, just reordered)
        gl_match = abs(gold - loop) / gold < 1e-9
        gc_match = abs(gold - closed) / gold < 1e-9
        lc_match = abs(loop - closed) / loop < 1e-9

        print(f"  Gold vs Loop:   {'PASS' if gl_match else 'FAIL'} (reldiff={abs(gold-loop)/gold:.2e})")
        print(f"  Gold vs Closed: {'PASS' if gc_match else 'FAIL'} (reldiff={abs(gold-closed)/gold:.2e})")
        print(f"  Loop vs Closed: {'PASS' if lc_match else 'FAIL'} (reldiff={abs(loop-closed)/loop:.2e})")
        print()

        if not (gl_match and gc_match and lc_match):
            all_pass = False

    print(f"{'='*70}")
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")


if __name__ == "__main__":
    main()
