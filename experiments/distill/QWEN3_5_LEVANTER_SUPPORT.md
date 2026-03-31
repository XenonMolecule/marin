# Qwen3.5 Support in Levanter: Exploration & Implementation Plan

## Architecture Summary

Qwen3.5 is **not** a standard transformer. It is a hybrid **Gated DeltaNet + Transformer** architecture with a vision encoder.

| Property | Qwen3 (what works today) | Qwen3.5 (what we need) |
|---|---|---|
| Architecture class | `Qwen3ForCausalLM` | `Qwen3_5ForConditionalGeneration` |
| Layer type | Full attention in all layers | **Hybrid**: full attention every 4th layer, GatedDeltaNet (linear attention) otherwise |
| head_dim | 128 | **256** |
| RoPE theta | 1M | **10M** |
| partial_rotary_factor | 1.0 (full) | **0.25** (RoPE on 25% of head dim) |
| RoPE type | Standard | **M-RoPE** (multimodal, interleaved, sections [11,11,10]) |
| vocab_size | 151,936 | **248,320** |
| Context window | 32K | **262K** |
| Vision encoder | None | Yes (ViT-based) |
| Output gate | No | **Yes** (`attn_output_gate`) |
| Conv kernel | None | **depthwise causal conv1d** (kernel_size=4) in GDN layers |
| SSM state | None | **Rectangular state S** in GDN layers |

### Per-Model Specs

| Model | hidden_size | heads | kv_heads | layers | head_dim | tie_embeddings |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 1024 | 8 | 2 | 24 | 256 | true |
| Qwen3.5-2B | 2048 | 8 | 2 | 24 | 256 | true |
| Qwen3.5-4B | 2560 | 16 | 4 | 32 | 256 | true |
| Qwen3.5-9B | 4096 | 16 | 4 | 32 | 256 | false |

All models share: `full_attention_interval=4`, `linear_conv_kernel_dim=4`, `linear_key_head_dim=128`, `rope_theta=10M`, `partial_rotary_factor=0.25`.

GDN layers use separate head counts: `linear_num_key_heads` and `linear_num_value_heads` (differ from the full-attention head counts).

---

## What Already Exists in Levanter

### GatedDeltaNet Layer -- FULLY IMPLEMENTED

**File:** `lib/levanter/src/levanter/layers/gated_deltanet.py` (~900 lines)

The complete GatedDeltaNet layer is implemented and HF-parity-tested:
- `GatedDeltaNetConfig`: Configuration dataclass with head dims, counts, conv kernel size
- `GatedDeltaNet`: Full module with projections, depthwise causal conv1d, gated delta rule kernels, gated RMSNorm, output projection
- `chunk_gated_delta_rule()`: Chunkwise-parallel kernel for training/prefill
- `recurrent_gated_delta_rule()`: Sequential kernel for streaming decode
- `to_state_dict()` / `load_state_dict()` / `from_state_dict()`: Weight serialization matching HF naming
- Streaming decode with carried `(conv_state, S_state)`

### GatedDeltaNet Tests -- COMPREHENSIVE

**Files:** `lib/levanter/tests/test_gdn_layer.py` (~400 lines), `lib/levanter/tests/test_gdn_kernels.py` (~673 lines)

Coverage includes:
- Streaming decode matches one-shot prefill
- Chunk size invariance
- Gradient existence (differentiability)
- **HF parity tests** for both prefill and streaming decode (loads HF weights, compares outputs)
- Numerical stability with extreme gates
- Non-divisible sequence length handling
- State continuation across chunks

### Qwen3 Model -- WORKING REFERENCE

**File:** `lib/levanter/src/levanter/models/qwen.py`

- `Qwen3Config(LlamaConfig)`: Standard transformer config with QK-norm
- `Qwen3LMHeadModel(LlamaLMHeadModel)`: Uses `LlamaTransformer` (stacked identical decoder layers)
- HF checkpoint converter with `HfQwen3Config`

### Mamba3 Kernels -- SEPARATE SYSTEM (not needed)

The `lib/levanter/src/levanter/kernels/pallas/mamba3/` code is a separate SSM architecture, not used by GatedDeltaNet. Mentioning for completeness -- it is NOT relevant to Qwen3.5.

---

## What's Missing: The Gaps

### Gap 1: Hybrid Transformer Model Class (CRITICAL)

The existing `LlamaTransformer` creates a homogeneous stack of identical `LlamaDecoderLayer`s (all full attention). Qwen3.5 needs a **heterogeneous** stack where:
- Layers `i % full_attention_interval == (full_attention_interval - 1)` use **full attention** (standard `Attention`)
- All other layers use **GatedDeltaNet** (linear attention)

For a 32-layer model with `full_attention_interval=4`: layers 3, 7, 11, 15, 19, 23, 27, 31 use full attention (8 layers); layers 0-2, 4-6, 8-10, ... use GatedDeltaNet (24 layers).

**What needs to be built:**
- A new decoder layer class (e.g., `Qwen35DecoderLayer`) that can be either attention or GDN
- A new transformer class (e.g., `Qwen35Transformer`) that builds the heterogeneous stack
- The heterogeneous stack must handle the fact that full-attention layers and GDN layers have **completely different state shapes** during inference (KV cache vs conv_state+S_state)

**Complexity:** This is the hardest part. Levanter's `Stacked` / `BlockSeq` assumes homogeneous layers. Options:
1. Use `BlockSeq` (non-scanned) which supports heterogeneous layers but is slower to compile
2. Build two separate `Stacked` groups (attention layers and GDN layers) and interleave their execution
3. Create a unified layer that contains both an Attention and a GDN module but only executes one (wastes memory)

Option 1 is probably the most pragmatic starting point.

### Gap 2: Partial Rotary Embeddings

Qwen3.5 uses `partial_rotary_factor=0.25`, meaning RoPE is only applied to the first 25% of the head dimension (64 out of 256 dims). The remaining 75% is unrotated.

**What needs to be built:**
- Modify the RoPE application in the attention layer to only apply to the first `floor(head_dim * partial_rotary_factor)` dimensions
- The existing `DefaultRotaryEmbeddingsConfig` and `Llama3RotaryEmbeddingsConfig` assume full rotation

**Complexity:** Low-medium. Need to split the Q/K along head_dim, apply RoPE to the first slice, concatenate back.

### Gap 3: M-RoPE (Multimodal Rotary Position Embeddings)

Qwen3.5 uses `mrope_interleaved=true` with `mrope_section=[11,11,10]`. This is a multimodal RoPE variant where position IDs are split into temporal, height, and width components for vision-language alignment.

**What needs to be built:**
- M-RoPE implementation that handles the interleaved section layout
- For text-only SFT, this may simplify to standard RoPE with the same theta (text uses identical position IDs across all 3 sections)

**Complexity:** For text-only training, this might be a non-issue -- need to verify that with identical position IDs across sections, M-RoPE degenerates to standard RoPE. If so, we can skip this for now and just use standard RoPE with `theta=10M` and `partial_rotary_factor=0.25`.

**IMPORTANT: This needs verification.** If M-RoPE with equal position IDs does NOT degenerate to standard RoPE (e.g., due to the interleaved dimension ordering), the model will produce garbage outputs even if everything else is correct.

### Gap 4: Output Gate on Full-Attention Layers

Qwen3.5 has `attn_output_gate=true` on its full-attention layers. This means the attention output is element-wise multiplied by `sigmoid(gate)` before projection, where `gate` comes from a separate linear projection.

**What needs to be built:**
- Modify the attention layer (or create a wrapper) that adds an output gate projection and applies it
- This is similar to the gated RMSNorm in GDN but applied to the attention output

**Complexity:** Low. One extra linear projection + sigmoid + element-wise multiply.

### Gap 5: HF Checkpoint Converter for Qwen3.5

Loading pretrained weights from HuggingFace requires mapping HF state dict keys to Levanter's internal structure.

**What needs to be built:**
- `Qwen35Config.from_hf_config()`: Parse HF's `Qwen3_5Config` (which nests text config under `text_config`)
- `Qwen35Config.to_hf_config()`: Reverse mapping
- State dict key mapping for:
  - Full-attention layers: similar to existing Qwen3 mapping
  - GDN layers: `model.layers.{i}.temporal_block.{in_proj_qkvz,in_proj_ba,conv_weight,A_log,dt_bias,o_norm,out_proj}` (HF naming)
  - MLP layers: same as Qwen3
  - Layer norms: same as Qwen3
- Handle the nested `text_config` in HF's config.json (Qwen3.5 wraps text params under `text_config` because it's multimodal)

**Complexity:** Medium. The GDN layer already has `to_state_dict()` / `load_state_dict()` with correct key naming. The main work is wiring the per-layer mapping into Levanter's `HFCheckpointConverter` framework and handling the full-attention vs GDN distinction per layer index.

### Gap 6: Transformers Version

The current venv has `transformers==4.38.2`. Qwen3.5 requires a much newer version (likely 4.57+, based on the config.json `transformers_version` field). The HF config class `Qwen3_5Config` does not exist in 4.38.2.

**What needs to be done:**
- Either upgrade transformers (risk: breaking changes for existing models)
- Or implement `from_hf_config` by parsing the raw JSON directly (bypass the HF config class)

**Complexity:** Low if parsing raw JSON; unknown risk if upgrading transformers.

---

## Implementation Plan

### Phase 1: Minimal Text-Only SFT (Goal: Train rephraser models)

This phase gets us to "can finetune Qwen3.5 on text data" without full multimodal support.

#### Step 1.1: Verify M-RoPE Degeneracy (1-2 hours)

**Before writing any model code**, verify that M-RoPE with identical position IDs across all 3 sections produces the same output as standard RoPE with `partial_rotary_factor`.

- Write a small test script that:
  1. Loads a Qwen3.5 model in HF transformers (may need a separate venv with transformers>=4.57)
  2. Runs a forward pass with text-only input
  3. Extracts the position embeddings and compares against standard RoPE with theta=10M and partial_rotary_factor=0.25
- If they match: we can use standard partial RoPE for text-only SFT
- If they don't: we need to implement M-RoPE properly

**Why first:** If M-RoPE doesn't simplify, the scope of work increases significantly. Better to know upfront.

#### Step 1.2: Partial Rotary Embeddings (2-4 hours)

Add `partial_rotary_factor` support to Levanter's RoPE.

- Modify `apply_rotary_pos_emb` (or add a wrapper) to:
  1. Split Q/K along head_dim into rotated portion (first `floor(head_dim * factor)` dims) and pass-through portion
  2. Apply RoPE only to the rotated portion
  3. Concatenate back
- Add `partial_rotary_factor` field to `RotaryEmbeddingsConfig`
- Test: compare output against HF's implementation for a single attention layer

#### Step 1.3: Output-Gated Attention (1-2 hours)

Add `attn_output_gate` support.

- Add an optional gate projection to the attention module (or create `GatedAttention` variant)
- Gate = Linear(x) -> sigmoid -> element-wise multiply with attention output
- Test: compare against HF's `Qwen3_5Attention` output

#### Step 1.4: Heterogeneous Decoder Stack (4-8 hours)

Build the hybrid transformer that interleaves full-attention and GDN layers.

- Create `Qwen35DecoderLayer` that wraps either `Attention` or `GatedDeltaNet` + shared MLP + norms
- Create `Qwen35Transformer` that builds the layer stack with correct types per index
- Use `BlockSeq` (non-scanned) for the heterogeneous stack initially
- Handle inference state: full-attention layers produce KV cache, GDN layers produce (conv_state, S_state)
- Test: verify forward pass shape correctness, gradient flow through both layer types

#### Step 1.5: Qwen35Config + HF Weight Loading (4-6 hours)

- Create `Qwen35Config` dataclass with all Qwen3.5 specific fields:
  - `full_attention_interval`, `linear_key_head_dim`, `linear_value_head_dim`, `linear_num_key_heads`, `linear_num_value_heads`, `linear_conv_kernel_dim`, `partial_rotary_factor`, `attn_output_gate`
- Implement `from_hf_config()` by parsing raw JSON (avoid transformers version dependency)
- Implement HF checkpoint converter:
  - Map full-attention layer keys (similar to Qwen3)
  - Map GDN layer keys using `GatedDeltaNet.load_state_dict()` key naming
  - Handle the `text_config` nesting in HF config.json
- Test: load Qwen3.5-0.8B weights, run forward pass, compare logits against HF reference

#### Step 1.6: End-to-End Smoke Test (2-4 hours)

- Load Qwen3.5-0.8B (smallest model) from HF into Levanter
- Run a training step on a tiny dataset (few hundred examples)
- Verify: loss decreases, gradients flow, checkpointing works, HF export works
- Compare perplexity on a held-out set against HF's own forward pass

### Phase 2: Training at Scale

#### Step 2.1: Memory & Performance Profiling (2-4 hours)

- Profile memory usage for each model size on v5p-8/v5p-16
- Determine per_device_parallelism and gradient accumulation settings
- The 256 head_dim and GDN state matrices may change memory characteristics significantly vs Qwen3
- Benchmark training throughput (tokens/sec) and compare to Qwen3 baseline

#### Step 2.2: Experiment Scripts (2-3 hours)

- Create experiment scripts for all 4 model sizes x 2 thinking variants = 8 scripts
- Follow existing pattern: download -> [strip_thinking] -> filter -> tokenize -> train
- Chat template: Qwen3.5 likely has a new chat template (need to extract from tokenizer)
- Set hyperparameters based on Qwen3 experience, scaled for new architecture

#### Step 2.3: Run & Validate (ongoing)

- Launch 9B and 2B first (user priority)
- Monitor training loss curves
- Compare against Qwen3 baselines on same dataset

### Phase 3: Production Hardening (Optional / Future)

- **Scanned layers:** Replace `BlockSeq` with a more efficient execution strategy (e.g., two interleaved `Stacked` groups)
- **Vision support:** Add the ViT encoder for multimodal training
- **M-RoPE proper:** Full multimodal position embedding support if needed
- **Sliding window:** If Qwen3.5 uses sliding window on full-attention layers

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| M-RoPE doesn't simplify for text-only | Medium | High -- need full M-RoPE impl | Verify in Step 1.1 before committing |
| `BlockSeq` too slow for 32 layers | Low | Medium -- slower training | Start with it, optimize later |
| Memory blowup from 256 head_dim + GDN state | Medium | Medium -- need larger TPUs | Profile in Step 2.1, adjust configs |
| HF weight key mapping errors | High | Low -- debuggable | Use GDN's existing `load_state_dict`, test with small model |
| Numerical divergence between Levanter and HF | Medium | High -- silent correctness bug | HF parity tests at every step |
| Transformers version conflict | Low | Medium | Parse raw JSON instead of using HF config class |

## Estimated Total Effort

- **Phase 1 (text-only SFT capability):** 15-28 hours of focused engineering
- **Phase 2 (running experiments):** 6-11 hours + cluster time
- **Phase 3 (hardening):** Scope TBD based on Phase 1 learnings

The GatedDeltaNet layer being fully implemented and tested is a huge head start. The main engineering challenge is the heterogeneous decoder stack (Step 1.4) and getting HF weight loading exactly right (Step 1.5).
