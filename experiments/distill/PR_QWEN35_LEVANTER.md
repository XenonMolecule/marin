# PR: Add Qwen3.5 model support to Levanter

**Title:** `Add Qwen3.5 (hybrid GDN + Transformer) model to Levanter`

---

## Summary

Adds Levanter support for the Qwen3.5 dense model family (0.8B, 2B, 4B, 9B, 27B) for text-only SFT training. Qwen3.5 uses a hybrid architecture that alternates GatedDeltaNet (linear attention) layers with standard full-attention layers every 4th layer.

- Load pretrained Qwen3.5 weights from HuggingFace
- Run forward and backward passes with verified HF parity
- Fine-tune on text data using Levanter's existing training infrastructure

The existing GatedDeltaNet layer implementation (already merged) does the heavy lifting. This PR wires it into a full model with the attention layers, config parsing, and checkpoint loading.

## What changed

**New files:**
- `lib/levanter/src/levanter/models/qwen35.py` — `Qwen35Config`, `Qwen35Attention` (packed Q+gate), `Qwen35DecoderLayer` (dual-mode attention/GDN via Optional fields), `Qwen35Transformer` (heterogeneous `BlockSeq` stack), `Qwen35LMHeadModel` with `load_from_hf_checkpoint()`
- `lib/levanter/tests/test_qwen35.py` — 8 unit tests (config, partial RoPE, forward, backward, state dict)
- `lib/levanter/tests/test_qwen35_hf_parity.py` — end-to-end logit comparison against HF (requires transformers >= 5.x)

**Modified files:**
- `lib/levanter/src/levanter/layers/rotary.py` — Add `PartialRotaryEmbeddings` and `PartialRotaryEmbeddingsConfig` for `partial_rotary_factor=0.25` (RoPE on first 25% of head dims, rest pass through)
- `lib/levanter/src/levanter/layers/gated_deltanet.py` — Make `GatedDeltaNet` extend `ModuleWithStateDictSerialization`. Add haliax-compatible `from_state_dict`/`to_state_dict` that auto-detect HF checkpoint format (split projections: `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`) vs internal packed format (`in_proj_qkvz`, `in_proj_ba`). Rename old classmethod to `create_from_state_dict`.
- `lib/levanter/tests/test_gdn_layer.py` — Update 4 call sites for the rename

## Architecture notes

Qwen3.5 differs from standard transformers in several ways that required new components:

| Feature | Implementation |
|---|---|
| Hybrid layer stack (GDN + attention) | `BlockSeq` with `Optional` fields per layer type, following the OLMo3 pattern |
| Packed output gate in attention | `q_proj` outputs `head_dim * 2`; first half is query, second half is sigmoid gate applied before `o_proj` |
| Partial rotary embeddings | New `PartialRotaryEmbeddings` class applies RoPE to first 25% of head dims |
| (1+w) RMSNorm | Reuses existing `GemmaRMSNorm` |
| QK-normalization | `GemmaRMSNorm` on Q and K after projection |
| M-RoPE (multimodal RoPE) | For text-only input, M-RoPE degenerates to standard partial RoPE (verified empirically) |
| Split checkpoint projections | HF saves GDN weights as 4 separate tensors; we repack to 2 packed tensors on load |

## Verification

Tested against HuggingFace transformers 5.x on `Qwen/Qwen3.5-0.8B`:

| Check | Result |
|---|---|
| Forward pass max logit diff | 0.029 |
| Forward pass mean logit diff | 0.003 |
| Backward pass loss diff | 0.0014 |
| Embedding gradient norm relative diff | 0.11% |
| All gradients finite | Yes |
| State dict save/load roundtrip | Exact match (0.0 diff) |
| Config from_hf_config/to_hf_config roundtrip | All fields preserved |

All differences are consistent with float32 accumulation order between JAX and PyTorch.

## Known limitations

These are intentional scope boundaries for this PR, not bugs:

- **No decode/generation.** GDN layers run with `inference=False`. Decode would need `(conv_state, S_state)` carrying across steps. Not needed for SFT.
- **No vision encoder.** Text-only; multimodal support is out of scope.
- **No MoE variants.** Dense models only (0.8B-27B). The 35B-A3B, 122B-A10B, 397B-A17B MoE variants are not supported.
- **GDN attention mask not forwarded.** Padding tokens can leak through the causal conv. Fine for non-padded SFT, needs fixing for variable-length batches.
- **`chunk_size=64` hardcoded** in GDN forward pass. Could be made configurable for memory/speed tuning on different TPU sizes.
- **`BlockSeq` (not `Stacked`).** Heterogeneous layers can't use JAX scan, so compilation is slower for large models. Could be optimized with two interleaved `Stacked` groups in the future.
- **HF parity test needs transformers >= 5.x.** The `qwen3_5` model type is not in transformers 4.x. The parity test auto-skips when unsupported.

## Test plan

- [ ] `pytest lib/levanter/tests/test_qwen35.py -k "not slow"` -- 8 fast unit tests (~25s)
- [ ] `pytest lib/levanter/tests/test_qwen35_hf_parity.py` -- HF logit parity test (~2min, needs transformers >= 5.x)
- [ ] `pytest lib/levanter/tests/test_gdn_layer.py -k "not hf"` -- existing GDN tests still pass
- [ ] Load Qwen3.5-0.8B, run a training step on a small dataset, verify loss decreases
