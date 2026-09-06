# michaelryan storage triage — keep / delete manifest (v2, decisions agreed 2026-08-16; nothing executed yet)

## Decisions (2026-08-16)
- Intermediate checkpoints (non-final `hf/step-N`, all optimizer `checkpoints/step-N`): **delete straight away** for
  every run with a DONE marker, and for stale (>7 d) non-DONE runs; the only intermediates worth keeping belong to
  runs training *right now* — the OLMIX **resiliparse swarm** and the **lpv11_fastpipe_v1** cells (name-protected +
  7-day write hold in `build_delete_manifest.py`). Temp (`tmp/ttl=14d/checkpoints-temp`) intermediates of DONE runs
  may go too (`cleanup_done_temp_checkpoints.py`).
- Final `hf/step-<max>` of every model is kept (rephraser-8B etc.). `medical-sft-base` optimizer trees and the kimi
  rephraser training-run intermediates: delete; nothing there will be resumed.
- Tokenized caches: keep **one** regional copy per registered method (pinned region if the planner pins it, else the
  complete copy in us-east5/us-central1); keep **all** copies for open lines (lpv11, resiliparse_decon_10364,
  10k-mix inputs, sysprompt30b, eval caches). Natural-epoch (`expFM_natural`/`expWARC_natural`) experiment inputs are kept.
- B5 / C-a: recommendation = keep the head-biased raw pool `baseline_3000-265ff5` + hq3000 & lpv11 negatives; delete the
  random pool `baseline_3000_random-34884d` and its `high_quality_3000_distill_random/data_no_useful` (pending OK).

## Generated lists (live listing 2026-08-16 15:43Z; dir: scratchpad `storage/manifest_20260816/`)
| list | prefixes | TB | what |
|---|---|---|---|
| `rm_ckpt_intermediates.txt` | 5,476 | 37.1 | non-final hf + optimizer steps (DONE runs; stale non-DONE runs); high_quality/lpv11-named cells split out |
| `rm_ckpt_intermediates_hq_lpv11_DONE.txt` | 2,154 | 9.9 | same rules, but for training cells named high_quality*/llm_pipeline_v1_1*/hq_* (DONE runs' intermediates + 197 orphan/partial copies) — separate script so it is a conscious step |
| `rm_ckpt_whole_runs.txt` | 1,549 | ≈10.4 | orphan copies of runs DONE in another bucket + stale opt-only partials (all orphans verified to have a DONE final elsewhere) |
| `rm_fast_curation.txt` | 54 | ≈9.7 | `a_presurvivors`, `a_chunks`, `kept_chunks`, claims/heartbeats/timing/xla, abandoned v1/v2/v3-6855 namespaces |
| `rm_dedup_intermediates.txt` | 102 | ≈2.3 | `normalize/fuzzy/reshape/minhash` under `documents/baseline_*deduped*/<n>warcs/`, lpv11 excluded |
| `rm_tokenized_mirrors.txt` | 224 | 9.9 | keep-one plan (`tokenized_mirror_plan.csv`); incl. `hq_distill_bal3p5m` (4B abandoned) and per-region fastpipe re-hashes |
| `rm_random_pool_and_negatives.txt` | 2 | 4.5 | both raw 3000-WARC pools (head + random; re-downloadable, manifests in repo) |
| `deferred_manifest_referenced_orphans.txt` | 52 | ≈0.2 | orphan run copies that an `experiments/core_eval_manifests/*` row points at (evals resolve `hf/` largest step there) — kept so no manifest dangles |
| `deferred_hq_lpv11_*` | 94 | ≈2.6 | high_quality/lpv11 data artifacts (dedup intermediates, cache mirrors, random `data_no_useful`) — held per "nothing from the high_quality / lpv11 runs" |
| `deferred_lpv11_*` | 72 | 0.3 | DONE lpv11 cells' intermediates + lpv11 dedup intermediates — only after the lpv11 sweep is closed |
| `keep_final.tsv` | 4,509 | 17.4 | final models kept (verified `config.json`) |
| `hold.tsv` | 196 | 5.1 | recent/protected (lpv11 cells, recent isoflop cells, mb-clf-lpv11 ckpts, live swarm) |
| `review_no_final.tsv` | 93 | 1.5 | opt-only runs outside the sweep tree (modernbert/qwen3-useful sweeps, v5p32 8B kimi runs, cooldown stubs) — untouched |

Total in the main scripts ≈ **74 TB**; +9.9 TB if you also run the hq/lpv11 DONE-cell script (~$630/mo list). Eval cross-check (2026-08-16): all 3,580 gs:// refs in `core_eval_manifests/*` compared against every rm list — no referenced `hf/step-N` is deleted (all suites use `_resolve_final_step` = largest step under `hf/`); 52 referenced orphan copies deferred; 0 DONE runs have an optimizer step beyond their last hf export. Guards in every script refuse: lpv11_fastpipe_v1, olmix-swarm/*resiliparse*, mb-clf-lpv11, pooled, classifiers/, presharded, _clf_token_cache, datasets/{high_quality_3000_distill,llm_pipeline_v1_1_3000_distill}/, tmp/, documents/baseline_llm_extraction/, kept_text, tombstones, */deduped/. Execution: `run_delete_<list>.sh` (dry-run by default, `EXECUTE=1` to delete;
refuses lists containing protected patterns; `xargs -P 2` + throttled gcloud; logs + `.failed`). Run them yourself.


Source: weekly scan listing as of 2026-08-10, rolled up per subtree by
`rollup_objects.py` (in-region Iris job) and analyzed with `triage_report.py`.
Snapshot outputs: `gs://marin-us-central2/scratch/michaelryan/storage_triage/2026-08-10/`.
Owner-matched footprint ≈ 130 TB (~$1.1k/mo). Sizes are TB unless noted.
Sorting principle: **time-to-recreate first, bytes second**. Every DELETE row
names what it can be re-derived from and roughly how long that takes.

Recreation-cost scale used below:
- **T0 irreplaceable / weeks**: LLM (8B rephraser) extraction ≈ 5 v6e-4-h per WARC
  → 10,364 WARCs ≈ 50k TPU-h; trained model weights (each cell hours–days of TPU).
- **T1 days of fleet**: fast_curation cascade (8–17k core-h + ModernBERT TPU per full
  pool), fuzzy dedup of a 10k-WARC corpus (hours–day of Zephyr CPU per corpus).
- **T2 hours**: tokenize from `deduped/` (hours CPU), cross-region cache mirror
  (~$15–20 per TB of egress, minutes–hours), CC re-download of 3000 WARCs (hours),
  HTML decode caches, prepped classifier text.
- **T3 minutes / free**: optimizer state of finished runs, temp checkpoints,
  claims/heartbeats, XLA caches.

## A. Checkpoints (rollup from `ckpt_objects.parquet`, per-file)

### A1. `checkpoints/isoflop-curation/` — 58.2 TB, 5,288 (bucket,run) copies, 4 buckets

| category | copies | total | keep (`hf/step-max`) | intermediate `hf/step-N` | optimizer `checkpoints/step-N` | proposal |
|---|---|---|---|---|---|---|
| DONE marker + hf | 2,655 | 39.9 | 10.0 | 14.3 | 15.6 | **DELETE 29.9** — keep only `hf/step-<max>` + `checkpoints/eval_metrics.jsonl` + `.data_curation_DONE` + `wandb/`. Recreate cost of what's deleted: only useful for resume (done) or intermediate-step evals (T0 to redo, but no consumer reads them today) |
| not done, written < 7 d | 228 | 9.0 | 0.9 | 1.3 | 6.9 | **HOLD** (live sweeps: lpv11, olmix, mix waves) |
| not done, stale > 7 d, has hf | 643 | 6.3 | 2.2 | 2.1 | 2.0 | REVIEW: 977 of the stale copies (5.2 TB) are *orphans of runs that completed in another bucket* → **DELETE 5.2**; the rest are killed/abandoned cells (relaunch would rebuild) |
| not done, stale > 7 d, optimizer only | 1,762 | 3.0 | 0 | 0 | 3.0 | **DELETE 3.0** — preempted/killed partials, no hf, no DONE (T3; only value = resume of an abandoned cell) |

Net: keep ≈ 13–17 TB, reclaim ≈ 38–41 TB (~$340/mo). Largest single wins:
the 2e21-B128 `expFM_natural` d3584 cells (325 GB each: 32 GB final + 195 GB of 6
intermediate hf exports + 97 GB optimizer) — fastpipe_v3_{100,80,60,20}, fineweb_cc_10k,
resiliparse_10k, fineweb_edu_10k, dclm/hq 10k_mix families.

**Question A1-a:** do you want to keep *any* intermediate `hf/step-N` exports (e.g. for
loss-curve/checkpoint-trajectory studies)? Default proposal: no — wandb +
`eval_metrics.jsonl` carry the curves.

### A2. Rephraser / SFT / distill / classifier training runs — 12.05 TB, keep 4.05

| tree | total | keep (final hf) | reclaim (optimizer + non-final hf) | notes |
|---|---|---|---|---|
| `checkpoints/qwen3-8b-rephraser-kimi{,-v2}-*` (5 runs, us-central1) | 3.3 | 0.17 | 3.1 | opt 0.5 TB each, 4–6 optimizer steps + 4–5 hf steps; last write 2026-07-07 |
| `checkpoints/qwen3-4b-rephraser-kimi*` (4), `1.7b` (3), `0.6b` (3) | 1.6 | 0.09 | 1.5 | same shape |
| `checkpoints/qwen3-8b-rephraser-kimi-v2-*-v5p32-*` (3) | 0.29 | 0 | 0.29 | optimizer only, **no hf at all** → failed/abandoned launches |
| `checkpoints/qwen3-1.7b-hq-distill-…-lr1e-5-bs128-mhfix-113d4e` | 0.44 | 0.007 | 0.43 | 21 optimizer steps; last write 2026-07-23 (was "PROTECT — live" in June; now stale) |
| `checkpoints/medical-sft-base/*` (36 runs, us-east5) | 0.90 | 0.06 | 0.84 | 0.6B SFT sweep; opt-only bulk; hf finals tiny |
| `checkpoints/sft-base/*` (48 runs) | 1.73 | 1.73 | 0 | hf finals only — already lean |
| `checkpoints/code-v3-*`, `medical-14b-*` (≈30 runs, 14B) | ~1.7 | ~1.7 | 0 | hf finals only |
| `checkpoints/modernbert-useful/*` (68) | 0.32 | – | ~0.22 | `mb-clf-large-*-chunk-ov` etc. are opt-only sweeps; production `mb-clf-{base,lpv11-*}` must stay (paths hashed into fast_curation spec) |
| `checkpoints/olmix-swarm/*` (556) | 1.38 | 0.36 | 1.02 | opt state of finished swarm cells → DELETE once final-eval wave is done |
| `exp2166-scaling-ladder-…-9563f0` (base for cooldowns) | 0.18 | 0.006 | 0.17 | 9 optimizer steps; keep `checkpoints/step-35000` (the base) + last hf |

Proposal: **DELETE optimizer + non-final hf across A2 ≈ 7.9 TB**; keep every final
`hf/step-<max>` (they are the models; retraining an 8B SFT on v5p-64 is hours of TPU
and needs the HF-hosted datasets, so T0-ish for the production ones).
**Question A2-a:** confirm the three `v5p32` 8B runs and `medical-sft-base` opt-only
trees are abandoned. **A2-b:** any rephraser kimi run you still intend to *resume*?

## B. Data caches (documents / tokenized / datasets / classifiers / datakit)

### B1. `documents/fast_curation/fastpipe_v3-da3893385e` (+ 2 abandoned namespaces) — 10.2 TB, 6 buckets

| stage | TB | proposal |
|---|---|---|
| `a_presurvivors` (raw-HTML carrier) | 5.01 | **DELETE** — T2 (phase A CPU over the WARC read-through cache); the survivor *set* is preserved in `kept_text`+`tombstones` |
| `a_chunks`, `kept_chunks` (chunk staging) | 4.65 | **DELETE** — temp staging that outlived the run |
| `kept` (survivor parquet incl. modernbert_prob) | 0.36 | KEEP (free re-thresholding) |
| `kept_text` (text-only, provenance-bearing; source of dedup) | 0.20 | **KEEP** (T1 to recreate) |
| `tombstones`, `b_keeplist`, `_completed_*`, `_claims_*`, `timing_*`, `_heartbeats`, `_xla_cache` | ≈0.01 | keep `tombstones`,`b_keeplist`,`_completed_*`; delete `_claims_*`,`_heartbeats`,`timing_*`,`_xla_cache` (T3, ~65k tiny objects) |
| `fastpipe_v3-6855733850`, `fastpipe_v2-f78c2b2b7a`, `fastpipe_v1-9b5c93de91` | 0.06 | DELETE (abandoned namespaces per `VERSIONS.md`) |

Reclaim ≈ 9.7 TB (~$70/mo). **Question B1-a:** the lpv11 line
(`lpv11_fastpipe_v1-2224e3e476`) is not in this snapshot's `documents/fast_curation`
(it lives under `documents/baseline_lpv11_*` + `tokenized/lpv11_*`) — confirm no
`a_presurvivors` are needed for the still-pending `grid-lpv11-fastpipe-v1-10k-*` cells.

### B2. `documents/baseline_*_deduped/<n>warcs/<stage>` — 5.94 TB

| stage | TB | proposal |
|---|---|---|
| `deduped/` (+`deduped_df`, `stats`) | 2.60 | **KEEP** — survivor corpora; enables re-tokenizing with a new tokenizer (T1 to redo) |
| `normalize/`, `fuzzy/`, `reshape/`, `minhash/` | 3.34 | **DELETE** — pipeline intermediates (T1 to redo, but only needed to *rerun* dedup with different params, which regenerates them anyway) |

Reclaim ≈ 3.3 TB. Note `_urls` corpora (`baseline_{resiliparse,lpv11_fastpipe_v1}_decon_deduped_urls/…/deduped`, 0.92 TB, in both c2 and e5) are KEEP (URL-recovered survivors for the OLMIX grid) — one region could go once the grid stores are final.

### B3. LLM extraction outputs (T0 — never delete)
- `documents/baseline_llm_extraction_consolidated/` (c1 0.46 + e5 0.28 TB, **10.3M objects**) — the only consolidated copy of the 8B extraction. **KEEP.**
- `documents/baseline_llm_extraction/{spec}/data-*/` raw per-region output (eu-west4 0.23, us-west4 0.15, e5 0.09, … ≈ 0.6 TB, ~10M objects across 6 regions) — **REVIEW**: deletable *only after* verifying the consolidation inventory covers every WARC hash (the `_completed/` registry in c1 + `…_consolidated/inventories/`). Byte win small; object-count win large (Autoclass mgmt fee + listing cost).
- `documents/baseline_high_quality_hf_export/`, `documents/extractor_compare/`, `documents/baseline_llm_pipeline_v1_1_*` — KEEP.

### B4. Tokenized caches — 19.3 TB total; 67 registered (`curation_plan.py`) = 13.7 TB, unregistered 5.6 TB
- Registered caches: keep the canonical copy; **mirror excess (total − largest copy) = 8.1 TB** across regions. Proposal: delete mirrors for caches whose sweeps are DONE (per `run_registry` / results JSONs present); keep mirrors for `lpv11_fastpipe_v1_decon_10364warcs-a16e729` (live), `resiliparse_decon_10364warcs-beaaf5` (OLMIX swarm, e5+c2), the 10k `expFM_natural` inputs still in the mix waves.
- Unregistered ≥ 50 GB: `baseline_resiliparse_{2000,1000,500}warcs` (1.7 TB, 3 regions each — old WARC-scaling ladder, superseded by `resiliparse_dedup_*`), `baseline_llm_curated_bos_fixed_{2000,1000,500}warcs` (0.8 TB), `hq_distill_bal3p5m_qwen3_32k-cb0690` (0.42; the 3.5M distill set — 4B abandoned) → **DELETE ≈ 2.9 TB**. `fastpipe_v3_*-{69b333,98043c,572f3f,…}` are per-region hashes of registered caches, treat as mirrors.
- Recreate cost of any cache: hours from `deduped/` (T2) — as long as B2 `deduped/` stays.

**Question B4-a:** which sweeps are truly closed so their regional mirrors can go? My read: everything `expA/expB/expC`, the 300–3000 random ladders, `fastpipe_v3_*` bands (125-run sweep done), `high/med/low_quality_*warcs`; still open: lpv11, olmix, mix/lambda waves.

### B5. `datasets/` — 7.45 TB (us-central2)
- `high_quality_3000_distill{,_random}/data_no_useful` (4.25 TB) + `llm_pipeline_v1_1_3000_distill/data_no_useful` (1.73 TB): the *NO_USEFUL* rows (raw HTML + empty target). Regenerable from consolidated extraction + WARC cache (T2, hours). HF export exists for hq3000. Proposal: **DELETE `data_no_useful` ≈ 6.0 TB**, keep `data/` (1.05 TB), `_chat_*`, `extractor_eval_set`, `useful_cascade_survivors`, `justext_router_labels`.
  **Question B5-a:** are the classifier training pipelines (fastText/ModernBERT/w640) done sampling negatives from `data_no_useful`? If a future classifier needs fresh negatives, keep one of the two hq3000 sets.

### B6. `classifiers/useful_fasttext{,_lpv11}` — 3.1 TB
- `full_prep_body_strip` (c2 0.84 + 0.76, e5 0.30 + 0.10 = 2.0 TB) prepped train text; `presharded_survivor_*` (0.54) — regenerable (T2). KEEP `model.bin`/`metrics.json`/`LEADERBOARD.md`, the frozen-7k eval set, `presharded_survivor_w640_10M` (used by lpv11 line). Proposal: **DELETE `full_prep_body_strip` in c2 (1.6 TB)** once you confirm the fastText w320/w640 lines are frozen; keep the e5 copies (lpv11 canonical).

### B7. `datakit/{store,quality,cluster_assign,tokenize}/*_gridv1`, `mirror/grid_v1` — 2.6 TB
KEEP stores (OLMIX domains, replicated c1+e5 on purpose); `datakit/tokenize/*_gridv1` parquet (0.73) and `mirror/grid_v1` (0.11) are intermediates → DELETE after the OLMIX final-eval wave.

## C. Raw / intermediates
- `raw/commoncrawl/baseline_3000-265ff5` (2.23 TB) + `baseline_3000_random-34884d` (2.32 TB), us-central2: raw HTML pools. Playbook says keep the 3000 pool. Recreate = CC re-download (T2, hours; manifest-verified). **Question C-a:** keep both? (the random-3000 pool is the base of the `*_random_3000` ladders — all done.)
- `raw/commoncrawl/rephraser_sweep_batch0-231d96` mirrored in **5 buckets** (0.12 TB each) → keep c1 only, DELETE 4 mirrors (0.47 TB).
- `filtered/`, `filtered_subsets/`, `extracted/` (10k + 3000 baselines, us-central2, ~1 TB): re-creatable from raw pools + filters (T2). Keep `extracted/dclm_400m_1x_10k_resiliparse-f0887f` (grid corpus input) and `filtered/dclm_400m_1x_10k_*` (grid corpora); rest REVIEW.
- `documents/bert_pipeline/decoded_10k` (0.81 TB, e5): decoded-HTML cache, T2 → DELETE.
- `cdx/` (12.8 GB), `downloaded/` (166 GB), `mathhelpforum/` — from the March PRESERVE list; keep (small, slow to refetch).

## D. Temp / coordination
- `tmp/ttl=14d/checkpoints-temp` (15.6 TB fleet-wide, mine ≈ the isoflop/olmix children) — auto-expires; nothing to do.
- `tmp/extraction_manifests/` (us-central2) — **no ttl segment, never reaped**; tiny; delete or move under `tmp/ttl=30d/`.
- `users/root/*`, `users/ubuntu/*` (0.45 TB) — not mine (namespacing bug outputs); flag to owners.

## E. Always keep (tiny, irreplaceable)
`metadata/*` results, locks, claims, registries (0.40 TB total incl. others');
`manifests/`, `metadata/*warc_metadata*`; `devset/`, `benchmarks/`;
`infinigram_indices/`, `bm25_indices/`, `url_index/`, `spec_explorer/text_index`;
`eval_datasets/*` per region; `artifacts/resiliparse_rs`; `resources/datakit/quality/pooled_junkgate2`.

## Totals (proposal, pending answers)
| bucket of work | reclaim TB | $/mo (list, ×0.7 discount ≈) |
|---|---|---|
| A1 isoflop-curation non-final | 38–41 | ~$300 |
| A2 SFT/rephraser/olmix optimizer + non-final hf | 7.9 | ~$60 |
| B1 fast_curation staging | 9.7 | ~$70 |
| B2 dedup intermediates | 3.3 | ~$25 |
| B4 cache mirrors + unregistered | 8–11 | ~$70 |
| B5 `data_no_useful` | 6.0 | ~$45 |
| B6 prepped classifier text | 1.6 | ~$12 |
| C raw mirrors + decode caches | 1.3 | ~$10 |
| **total** | **≈ 76–81 TB of ≈130** | **≈ $600/mo** |

Execution guardrails (from the June protocol): re-snapshot RUNNING jobs right before
`rm`; verify each kept `hf/step-<max>/config.json` exists before deleting siblings;
delete optimizer as `<run>/checkpoints/step-*` (never the whole `checkpoints/`, it
holds `eval_metrics.jsonl`); in-region deletes only; user runs the generated `rm`
scripts (`! bash …`).
