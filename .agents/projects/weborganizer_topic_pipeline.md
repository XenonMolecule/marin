# WebOrganizer/TopicClassifier labelling pipeline (10k pool)

Goal: label our curated 10,364-WARC (`dclm_400m_1x_10k`) corpora with the WebOrganizer 24-way topic
taxonomy, so curation methods can be compared by *topic mix* rather than only by CORE score.
Started 2026-07-17. Greenfield — no WebOrganizer code existed in the repo.

## The model (facts, verified against the HF repo — not guessed)

| Property | Value |
|---|---|
| Repo | `WebOrganizer/TopicClassifier` (+ `-NoURL` variant) |
| Backbone | `Alibaba-NLP/gte-base-en-v1.5`, **137.4M** params (measured) |
| Arch | `NewForSequenceClassification`, `model_type: new`, 12 layers / 768 hidden / 12 heads |
| Labels | **24** topics (Adult … Travel), `id2label` in config.json |
| Input | `"{url}\n\n{text}"` — WebOrganizer `annotate_data/domains.py` `input_template` |
| Train/infer ctx | `max_length=8192` (`define_domains/train_classifier.sh`), bf16 |
| Attention | **full global at every layer** (no local/sliding window) |
| RoPE | `rope_theta=500000`, `rope_scaling={"type":"ntk","factor":2.0}` |
| MLP | `NewGatedMLP` — GLU, `up_gate_proj` 768→2×3072 (no bias), `down_proj` 3072→768 (bias) |
| Norm | `layer_norm`, eps 1e-12 (post-norm, BERT-style) |
| Pooler | BERT-style: CLS → dense 768→768 → tanh → dropout → `classifier` 768→24 |
| Tokenizer | `BertTokenizer`, vocab 30528, `pad_token_id=0`, lowercase |
| `type_vocab_size` | **0** — no token-type embeddings (simplifies the port) |
| `logn_attention_scale` | **false** — skip that code path entirely |

**Loading gotcha (verified 2026-07-17):** config ships `use_memory_efficient_attention: true` and
`unpad_inputs: true`. Both are **xformers CUDA-only** paths. Force both `False` on the config before
`from_pretrained` or the model will not run on CPU/XLA. With that override it loads fine on
transformers **4.57.5** despite the remote code targeting 4.41, and reproduces the model card example
(`Hardware` p=0.969).

## Data: what we have, and where URL lives

URL is **required** for the primary model. Survey of the 10k pool's DOCUMENT layer (verified on GCS
2026-07-17 — the tokenized Levanter caches in `curation_plan.METHODS` carry **no URL**, so everything
below reads one stage upstream):

| Corpus | Path | Region | Shards | URL? |
|---|---|---|---|---|
| `dclm_10k` | `filtered/dclm_400m_1x_10k_dclm_resharded-1fe977` | us-central2 | 100 | ✅ `{text,url,warc_record_id,dclm_fasttext_score,dclm_language_score}` |
| `nemotron_full_10k` | `filtered/dclm_400m_1x_10k_nemotron_full-96bad9` | us-central2 | 24390 | ✅ `{text,url,nemotron_quality,nemotron_kind,...}` |
| `fineweb_edu_10k` | `filtered/dclm_400m_1x_10k_fineweb_edu-0d49e9` | us-central2 | 1469 | ✅ `{text,url,file_path,dump,fineweb_score,...}` |
| `resiliparse_10k` | `extracted/dclm_400m_1x_10k_resiliparse-f0887f` | us-central2 | 10364 | ✅ `{text,url}` |
| `high_quality_10k` | `documents/baseline_high_quality_hf_export/10364warcs/joined` | **us-central1** | 512 parquet | ✅ `{text,url,warc_record_id,warc_file,snapshot}` |

### The dedup URL hole (the one real gap)

`dedup_extracted.py:196` projects records to `{text}` only — **url / warc_record_id are dropped**. So
every `_deduped` / `_decon` tree is text-only. Consequences:

- **high_quality is already solved**: `build_high_quality_hf_export.py` recovers URL by a blake2b-128
  exact-text join against the raw consolidated archive — **19,967,190 / 19,968,996 = 99.99%** matched.
  Use the joined export directly.
- **dclm / nemotron / fineweb_edu / resiliparse**: use the **pre-dedup** document layer above, which
  has URL natively. Labels are per-document, so labelling pre-dedup then joining onto the deduped
  survivor set is equivalent (and lets one labelling pass serve every downstream variant).
- **`-NoURL` is the fallback**, not the default, wherever a join can't be made.

### Gaps to flag

- **MQ / LQ do not exist at 10k.** `med_quality*` / `low_quality*` are 3k-pool WARC-scaling methods
  only; no 10k document tree exists. Labelling them means either running against the 3k pool or
  building the 10k variants first.
- `fineweb_cc_10k` document layer is at `gs://marin-us-central2/documents/baseline_fineweb_cc/` — not
  yet schema-verified.
- `decoded_10k` (`gs://marin-us-east5/documents/bert_pipeline/decoded_10k/`) has
  `{doc_id,warc_hash,url,snapshot,html,text_body}` but only **500** parquet files — it is a partial
  decode of the pool, NOT a complete URL join table. Don't rely on it for full coverage.

## Throughput: the actual problem

gte-base has **full global attention at every layer**, so it is strictly worse than ModernBERT-base at
long context (ModernBERT gets 14/22 layers of radius-64 local attention). Our measured
ModernBERT-base JAX numbers ([[project_modernbert_inference_benchmark]]) bound it from above:

| ctx | v5litepod docs/s/chip | v6e docs/s/chip |
|---|---|---|
| 1024 | 96.7 | 153.4 |
| 2048 | 56.0 | 90.3 |
| 4096 | 28.2 | 49.1 |
| 8192 | 11.8 | 21.5 |

So **context length is the whole ballgame** — 8192→1024 is ~7× on ModernBERT and will be more on
gte (quadratic term is undamped). Two independent levers:

1. **Truncation.** WebOrganizer infers at 8192, but topic is usually decided by the URL + first
   screenful. Phase 0 measures the accuracy cost directly. This is the big constant factor.
2. **Length bucketing.** Most web docs are far shorter than 8192; padding every doc to a static 8192
   shape wastes most of the FLOPs. Sort-and-bucket by token length (the `modernbert_warc_filter.py`
   pattern) — pure win, no accuracy cost.

Rejected: **vLLM**. Its TPU backend's pooling/classify support is not something we can rely on, and
`NewForSequenceClassification` needs `--hf-overrides` gymnastics even on CUDA. We are TPU-only
([[feedback_no_gpu_rental]]), and TPU compute is free on TRC ([[feedback_trc_compute_is_free]]).
Rejected: **torch_xla** — user explicitly prefers the Levanter/JAX path that reached parity for
ModernBERT; torch_xla is the known liability stack ([[project_modernbert_tpu_torch_xla]]).

## Scale: sample for distributions, full-label only if the labels get USED

Corpus sizes differ by ~50×, from `curation_plan.METHODS` (tokens):

| corpus | tokens | ~docs @1k tok |
|---|---|---|
| fineweb_edu_10k | 2.35B | ~2M |
| dclm_10k | 7.33B | 5.9M (exact, from sysprompt cache) |
| nemotron_full_10k | 10.13B | ~10M |
| high_quality_10k | 21.30B | 19.97M (exact) |
| fineweb_cc_10k | 28.00B | ~28M |
| **resiliparse_10k** | **339.97B** | **~340M** |

This splits the work into two very different jobs, and they should not be conflated:

- **Comparing topic distributions across methods** (the stated goal) does **not** need every doc. A
  uniform random sample of ~1M docs/corpus estimates each of the 24 shares to well inside ±0.1%
  absolute — far tighter than any real difference between curation methods. Cost: a few v6e-hours
  for the whole matrix. **Start here.**
- **Full labelling** (all ~400M docs across corpora) is only justified if the labels get *used* —
  topic-conditional filtering or mixture reweighting à la the paper. That's a fleet job dominated by
  resiliparse alone, and should be a separate decision with its own cost estimate.

Sampling must be WARC-disjoint-aware and drawn uniformly across shards, not from shard 0
([[feedback_warc_disjoint_splits]]) — the Phase 0 probe deliberately reads shard 0 only because it is
a smoke, and its label distribution is therefore **indicative, not an estimate**. Note the nemotron
shards are per-crawl (`CC-MAIN-2013-20-*`), so shard 0 is a single 2013 crawl — the most
crawl-skewed of the five.

## STATUS 2026-07-17 — pivoted to a JAX port on TPU; CPU is dead

The original Phase 0 (CPU probe) was **mis-scoped and killed**. Measured on the real code path:
CPU scores an 8k-token doc at **0.03 docs/s** (1.56 docs/s at 500 tok) — 500 docs × a 6-length grid
is a multi-hour job, not the ~15 min assumed. CPU cannot label millions of docs at any fan-out we
have (1M docs @ maxlen 512 ≈ 712 core-hours). **The answer was TPU all along.**

**★ `weborganizer_gte_jax.py` — gte-base in ~200 lines of plain JAX, HF parity 3.3e-6 first try ★**
(seq 128/512/2048, ragged batches, and through the bucketed scorer with order preserved). This
replaces the "port gte into Levanter" plan for now: it runs on the existing `--extra tpu` container
as a normal preemptible Iris job, with no torch_xla install hacks. Details + the architecture
gotchas: [[project_gte_jax_port]]. A proper Levanter port is still the right long-term home.

**Running at the recommended 8192** (user's call — matches the published model), made affordable by
**length bucketing**, not truncation: sort by token length, pad each batch only to its bucket
(128…8192), batch size = `TOKENS_PER_BATCH // bucket`. ≤7 XLA programs. So the truncation-agreement
study is no longer on the critical path — it is a nice-to-have.

**Region pinning is forced**: the 10k document corpora are **NOT mirrored** to us-central1 (verified),
so dclm/nemotron/fineweb_edu/resiliparse must run on **v4 in us-central2** and high_quality on **v5p
in us-central1**. v4 capacity is the current bottleneck — the first probe sat `pending` ~20 min.

## Plan

### Phase 0 — CPU reference + truncation curve  ← SUPERSEDED (see above)
`experiments/baseline_collection/weborganizer_topic_smoke.py`, `probe` subcommand. Runs the reference
HF model in-region on a real sample and emits:
- **golden logits @8192** → the parity oracle for Phase 1 (this is exactly how the ModernBERT
  migration was validated: "HF-classifier roundtrip oracle = the core parity proof")
- **token-length distribution** → sizes the bucketing
- **truncation agreement** vs the 8192 reference, reported both over all docs and restricted to the
  docs the cutoff actually truncates (the only ones that can move — the honest metric)
- reference label distribution per corpus (a sanity check + a first result in its own right)

Launch: CPU Iris job, `--region us-central2` (us-central1 for `high_quality_10k`).

### Phase 1 — Levanter/JAX port
New `lib/levanter/src/levanter/models/gte.py`, mirroring `models/modernbert.py` (~650 lines).
Port surface, in order of risk:
- **RoPE/NTK**: gte's `NTKScalingRotaryEmbedding` computes `base = rope_theta * factor` = 1e6, then
  `inv_freq /= factor**(2/head_dim)`. Levanter's `DefaultRotaryEmbeddings` **already has this exact
  shape** (`inv_freq = 1/(theta**(...)) / config.factor`), so it maps to
  `DefaultRotaryEmbeddingsConfig(theta=1_000_000, factor=2**(2/64))`. **No new rope class needed** —
  but this is the #1 thing the parity oracle must catch.
  (NTK branch only fires because `_set_cos_sin_cache` is called with `8192*2 > max_position_embeddings`.)
- `pack_qkv: true` → fused `qkv_proj` weight `[3*768, 768]`; the state-dict mapping must split it.
- Gated MLP: `up_gate_proj` splits into `(up, gate)`; `gate = gelu(gate)`; `out = down(gate * up)`.
  Note the split order — up FIRST, then gate (`torch.split(up_gate, intermediate_size, dim=-1)`).
- BERT pooler + classifier head; `num_labels=24`.
- Full global attention → simpler than ModernBERT (no `bidirectional_window` plumbing).
Validate with an HF-roundtrip test against the Phase 0 golden logits (rtol ~1e-4 in fp32).

### Phase 2 — TPU throughput benchmark
Mirror `modernbert_inference_benchmark.py`: random-init gte, sweep ctx × backend on v6e-4 and
v5litepod-4, get real docs/s/chip. Feeds the region/fleet sizing.

### Phase 3 — Production scorer
Mirror `score_modernbert_useful.py`: data-parallel over chips, per-corpus shard fan-out, writes
`{warc_record_id | url, topic_choice, topic_logits}` parquet per shard. Length-bucket the input.
Region-pin to each corpus's own bucket ([[feedback_never_cross_region_reads]]).

Reuses `LabelReservoir` from Phase 0 unchanged — see below.

### Phase 4 — Distribution viewer
An HTML tool to compare topic distributions ACROSS datasets and click into a category to read a
random sample of its docs. Prior art to follow: `generate_review_viz.py` → `extraction_review.html`,
`build_relabel_interface.py`, `warc_scaling_dashboard.py`. Inputs are exactly the Phase 0/3 outputs:
`*_summary.json` (`label_distribution` → the comparison chart) and `*_examples.parquet`
(`{dataset,label,prob,n_seen,n_eligible,url,text}` → the drill-down).

## Per-label sampling: collected DURING the run, not post-hoc

`LabelReservoir` in `weborganizer_topic_smoke.py` — per-label reservoir sampling (Algorithm R) fed
batch-by-batch from the scoring loop via `score(..., on_batch=...)`. Properties that make it carry
from the 500-doc smoke to a 20M-doc production pass **unchanged**:

- **Bounded memory** — at most `per_label` docs per label regardless of stream size; no second read
  of the corpus, no holding 20M docs to sample afterwards.
- **Uniform over the eligible stream** — verified empirically (200-position stream, K=10, 4000 trials:
  every position sampled at 0.0500 = K/N, min 0.042 / max 0.061; no position starved or favoured).
- **Random, not confidence-ranked** — a top-N-by-confidence list only shows the easy core of a class.
- **Probability floor** (`--example-min-prob`, default 0.5) keeps the draw out of the model's
  coin-flip zone. `n_seen` (all docs) vs `n_eligible` (cleared the floor) are both recorded, so a
  label whose floor starved its sample is visible rather than silently under-sampled — the run logs
  those labels explicitly.

## Ops notes

- **Container needs `--extra cpu`** (torch 2.10 lives there; `transformers` comes from levanter's
  base deps). The bare container has NO torch — first launch died `ModuleNotFoundError: No module
  named 'torch'`. Also pass `--cpu 16`: the default is `--cpu 0.1`, which would throttle CPU inference.
- **finelog is down** (2026-07-17) → `iris job logs` raises `StatsError: Not Found`; use
  `iris job bug-report` for state. Because stdout is unreliable, the probe writes every result to GCS
  (`*_summary.json`) rather than only logging it ([[feedback_finelog_down_use_bug_report]]).
- us-central2 CPU jobs land on the **reserved v4 pool** — keep them small and short
  ([[feedback_cpu_jobs_off_reserved_tpu]]). The corpora are there, so we can't avoid the region.

## Open decisions

- **Truncation length** — set by Phase 0's agreement curve. Hypothesis: 1024 or 2048 is ~lossless.
- **Store logits or just argmax?** Logits (24 × float32 = 96B/doc) keep the mix analysis flexible
  and cost ~2GB over 20M docs. Recommend keeping them.
- **Do we also want `FormatClassifier`?** Same architecture, same pipeline, ~free once Phase 1 lands
  (only the checkpoint + label set change). The paper's topic×format cross-tab is the real payoff.
- **MQ/LQ**: no 10k document tree exists — decide whether to label the 3k-pool versions or build 10k
  variants first.
