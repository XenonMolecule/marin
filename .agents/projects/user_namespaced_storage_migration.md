# User-namespaced storage: PR #6790, how it touches my experiments, and what must be KEPT

Research notes, 2026-08-15. Status: RESEARCH ONLY — nothing moved, nothing deleted.
Owner: michaelryan. Companion memory: `project_user_namespaced_storage_migration`.

## 1. What PR #6790 actually does (merged 2026-06-30, rjpower)

- Adds `marin.experiment.namespacing.user_namespaced_name(name, version)`:
  returns `users/{username}/{name}` **only when `version` is mutable**
  (`dev` or `<label>-dev`); calendar-versioned (`YYYY.MM.DD[.N]`) names are
  unchanged. Idempotent (never stacks `users/x/users/x/`).
- Username = `rigging.provenance.username_segment()`: OS login (or the
  launch identity, see below), email domain dropped, lowercased,
  non-`[a-z0-9_-]` collapsed to `-`; **raises** if unresolvable (no silent
  `users/unknown/`). Mine resolves to **`michaelryan`**.
- Applied only inside `train_lm` (`lib/marin/src/marin/experiment/train.py`)
  and at builders that assemble a training `ArtifactStep` directly (grug
  `moe`/`base`/`scale`, `references/*`, `make_rl_step`, `june_tpu_67b_a2b/*`,
  `evaluation.py` evalchemy steps). Datasets / tokenized caches / fixed
  checkpoints keep shared names so multi-TB caches still cache-hit.
- Address model behind it (#6649): hash-free `{prefix}/{name}/{version}`; the
  old `ExecutorStep` `-<6hex>` suffix world is gone (`marin.execution.executor`
  no longer exists on this branch).
- Follow-ups: #7000 stamps launch identity into the job env
  (`MARIN_PROVENANCE`); #7470 makes `username_segment()` prefer that over the
  worker OS user and warns on machine logins (`root`, `runner`, `ubuntu`,
  `exedev`). Evidence the bug was real: `users/root/{grug,delphi,experiments,
  checkpoints,prot-exp117-*}` and `users/ubuntu/{checkpoints,evals}` exist on
  us-east5 / us-central1 / eu-west4 (~0.45 TB total).
- Open follow-up **#7205 "infra - migrate GCS usage to user buckets"**
  (rjpower, 2026-07-15; empty body). Will Held's clarification: *"move our
  existing GCS storage over to #6790 and move docs, defaults, and references
  to use user storage settings."* Two decisions were asked and never answered:
  (1) code-only vs physical data move; (2) how far to extend namespacing
  (remaining direct `ArtifactStep` sites vs flipping the shared
  `marin-{region}` default). No later comments, no linked PRs. Docs on the
  claude branch (`docs/explanations/marin-prefix.md` "Per-User Namespacing")
  were never merged; only `docs/tutorials/cloud-gpu.md:96` shows the layout.
- Prior art already on the buckets (ad-hoc, no convention): top-level
  `julian/` (23 TB), `pinlin_calvin_xu/` (23 TB), `user/rav/` (6.8 TB),
  `checkpoints/pinlin_calvin_xu/` (27 TB), `scratch/{ahmed,kaiyue,wmoss}`,
  `morrisy/`, `tomat/`, `pc0618/`.

## 2. How it interfaces with MY experiments today: it doesn't

Grep across my experiment dirs (`baseline_collection`, `scaling_law_sweeps`,
`fast_curation`, `data_mixing`, `rephraser`, `distill`, `datakit` (grid parts),
`spec_explorer`, `core_eval_manifests`, `infinigram`, `url_index`):
**zero** callers of `train_lm` / `marin.experiment` / `user_namespaced_name`.
Every write path is one of:

1. **Standalone launchers with explicit `output_path`** —
   `run_curation_train_standalone.py` (`{bucket}/checkpoints/isoflop-curation/{run}`),
   `run_olmix_swarm_standalone.py` (`checkpoints/olmix-swarm/{run}`),
   `launch_modernbert_levanter.py` (`checkpoints/modernbert-useful/{run_id}`),
   `exp_qwen3_0_6b_useful_classifier.py`. They call
   `marin.training.training._prepare_training_run` then
   `levanter.main.train_lm.main()` in-process. Temp checkpoints go to
   `tmp/ttl=14d/checkpoints-temp/...` via `marin_temp_bucket`.
2. **`REGION_TO_BUCKET[region] + relative path`** for anything large and
   region-local (tokenized caches, datakit `*_gridv1` stores, olmix results).
   Six-plus hand-rolled copies of the region→bucket map exist
   (`region_tracker.py:50-69`, `infinigram/targets.py:37-45` keyed
   `eu-west4` not `europe-west4`, `dedup_extracted.py:94`, `fast_curation/{dedup,
   consolidate_kept_text,recover_provenance}.py`, ...).
3. **Hardcoded `gs://marin-us-central1/...` "neutral home"** for tiny
   coordination/results: every `metadata/*_results/`, `metadata/region_locks/`,
   `metadata/olmix_claims/`, fast_curation `_completed_*`/`_claims_*`/
   `_heartbeats`, extraction `_completed/` registry.
4. **"Follow the checkpoint's bucket"** for eval outputs
   (`metadata/olmo_bpb_results/`, `metadata/data_curation_10k_core_results/`,
   `olmes_base_results/`, `mmlu_sl_verb_results/`): scattered across all six
   regions; every consumer scans a region list.
5. **CLI/env defaults**: `--bucket gs://marin-us-east5` (fast_curation),
   `USEFUL_FT_ROOT` (classifier tree), `--results-prefix`, `--output-base`,
   `MARIN_PREFIX` on the `executor_main` tokenize scripts.
6. **Legacy `ExecutorStep` hashed names** (`checkpoints/<name>-<6hex>`,
   `tokenized/<name>-<6hex>`, `raw/`, `filtered/`, `extracted/`) — the
   rephraser/distill/SFT scripts and `experiments/defaults.py:41` still import
   `marin.execution.executor`, which **no longer exists** → those scripts are
   dead at HEAD; their outputs remain valid records.

Consequences:
- Nothing I run today lands under `users/michaelryan/`, and nothing will
  until my launchers are changed. There is no automatic migration hook.
- A migration for my stuff is a *convention* change: pick a root
  (`users/michaelryan/`) and re-point constants/CLI defaults. Sites that
  hash paths into identities need care: `fast_curation/spec.py:36-53`
  (model/artifact paths feed `compute_version()` → changes the namespace
  hash), `experiments/datakit/reference_pipeline.py:192` (`EVAL_ROOT`
  absolute path in the bloom fingerprint), all `experiments/core_eval_manifests/*`
  (full `gs://` literals baked in `output_path`,`hf_dir`), spec_explorer /
  plot scripts scanning fixed `metadata/...` prefixes across `EVAL_REGIONS`.
- `ArtifactStep.adopt(name, version, source=...)` exists to register
  pre-existing data at any path as a typed handle without moving it — the
  cheap way to bridge old locations into the new address model.

## 3. Cost mechanics of physically moving old data (important)

All `marin-*` buckets: **Autoclass ON, terminal ARCHIVE**, no soft-delete,
lifecycle only deletes `tmp/ttl=Nd/`. Fleet mix on 2026-08-10: 46% Coldline,
39.5% Nearline, 9% Standard, 5% Archive (`report.md`).

- In-bucket `gcloud storage mv/cp` = server-side rewrite: **no egress**, no
  Autoclass retrieval or early-deletion fees, Class-A op per object (~$0.05/10k
  → negligible even for 8k-object trees; large trees like
  `documents/baseline_llm_extraction` have millions of objects → check
  object_count first).
- **But** the rewritten object is a *new* object and (expected under
  Autoclass) restarts at Standard: a Coldline TB ($4/mo) becomes a Standard TB
  ($20/mo) for 30 days, Nearline for the next 60, back to Coldline after 90.
  Moving e.g. the 57.7 TB `checkpoints/isoflop-curation/` tree would add
  roughly +$900/mo for month 1 and +$350/mo for months 2–3 (list price;
  ~0.7× with discount). Verify with a 1-object test before any bulk move.
- Cross-region moves are out (egress) — the region layout stays.

Recommendation that falls out: **do not physically move cold bulk**
(checkpoints, tokenized caches, deduped documents). Namespace *new* writes,
and for old data either (a) leave in place and register with a manifest /
`adopt()`, or (b) move only small, hot, hand-maintained things (results
JSONs, classifier weights, devsets, manifests — group A/E/G/H below, all
< ~1 TB except model weights).

## 4. Storage accounting — what is mine, and what must be KEPT

Source: weekly ops scan snapshot `gs://marin-us-central2/storage-report-history/
dir_summary-2026-08-10.parquet` (per bucket × depth-3 prefix, **only prefixes
≥ 1 GiB**) + `tmp/storage-scan/report.md`. Ownership assigned by matching the
path patterns the three code inventories produced. Lower bound: tiny-but-critical
prefixes (results JSONs, manifests, locks) fall below the 1 GiB floor.

Fleet: 3,530 TB, $18.3k/mo. **Mine ≈ 128 TB, ≈ $1,060/mo** (us-east5 50.8 TB,
us-central1 36.6, us-central2 23.6, eu-west4 7.3, us-east1 6.6, us-west4 3.5).

| Tier | Group | TB | $/mo | Keep policy (draft) |
|---|---|---|---|---|
| A | Final model weights: rephraser SFTs (`checkpoints/qwen3*-rephraser-*`, `qwen35-*`), hq-distill 1.7B/0.6B, `modernbert-useful/*`, `qwen3-useful`, medical/code/math SFT sweeps, cooldowns + `exp2166-scaling-ladder...` base, `models/qwen3-8b-extraction` | 10.7 | 34 | **KEEP** final `hf/step-<max>` per run; already curated in `experiments/rephraser/PRESERVE_FOR_WIPE.md` (2026-03-20, ~500 GB list); re-audit for post-March additions (kimi-v2 8B/4B/1.7B, qwen35) |
| B | `checkpoints/isoflop-curation/` (2,494 runs) | 57.7 | 510 | **KEEP `hf/step-<max>` + `checkpoints/eval_metrics.jsonl` + `.data_curation_DONE`** for every completed cell (they are the inputs to every CORE/OLMES/bpb/MMLU result); DELETE optimizer `checkpoints/step-*` and intermediate `hf/step-N` (already done for phase 1/2 in June, 17.4 TiB); the ~17 TB "incomplete & not running" REVIEW class from June is still untriaged |
| B2 | `checkpoints/olmix-swarm/` | 1.4 | 18 | KEEP finals until OLMIX solve is locked & written up (R=30B k=20 already locked → likely deletable after final-eval wave) |
| C | Tokenized caches (`tokenized/{baseline_*,resiliparse_*,fastpipe_v3_*,high/med/low_quality_*,fineweb_cc_10364,dclm_400m_1x_10k_*,lpv11_fastpipe_v1_decon_10364warcs-a16e729,hq_distill_*,rephraser_*,nemotron_cooldown_*,sysprompt30b_*}`) | 19.2 | 122 | KEEP the ones referenced by `curation_plan.py` METHODS + live sweeps + gold reference (`llm_pipeline_v1_1_decon_3000warcs-aa7070`); regional mirrors (`fastpipe_v3_*` in eu-west4/us-east1/us-central1) are deletable once their sweeps are done — decide per method |
| D | Documents: `documents/baseline_*_deduped/`, `*_decon_deduped[_urls]/`, `fast_curation/fastpipe_v3-da3893385e/{kept_text,tombstones}`, `baseline_llm_extraction[_consolidated]`, `bert_pipeline/decoded_10k` (0.8 TB, re-creatable), `extractor_compare`, `high_quality_hf_export` | 18.5 | 149 | KEEP `deduped/` survivor sets, `kept_text`, `tombstones`, consolidated LLM extraction (only copy of expensive vLLM output), `_urls` corpora, hf_export; DELETE `reshape/normalize/minhash/fuzzy` intermediates, `a_presurvivors`, `kept/` (superseded by `kept_text`), `bert_pipeline/decoded_10k` |
| E | Classifiers + datasets + datakit grid: `classifiers/useful_fasttext[_lpv11]/*` (weights + prepped data 3.1 TB), `datasets/{high_quality_3000_distill*,llm_pipeline_v1_1_3000_distill,extractor_eval_set,useful_cascade_survivors,justext_router_labels}` (7.4 TB), `datakit/{store,quality,cluster_assign,tokenize}/*_gridv1`, `mirror/grid_v1`, `artifacts/resiliparse_rs`, `resources/datakit/quality/pooled_junkgate2` | 13.0 | 130 | KEEP model.bin/metrics/leaderboard, frozen-7k eval set, `datasets/*` (HF-exported copies exist for hq3000 — could drop `data_no_useful` 6 TB if HF export is trusted), `datakit/store/*_gridv1` (OLMIX domains); `full_prep_body_strip` (2 TB prepped text) is regenerable |
| F | Re-creatable raw/intermediate: `raw/commoncrawl/{baseline_3000*,rephraser_sweep_batch0*}`, `cdx/`, `downloaded/`, `extracted/`, `filtered/`, `filtered_subsets/`, `manifests/` | 5.2 | 61 | KEEP `manifests/` + `metadata/*warc_metadata*` (provenance roots) + `raw/commoncrawl/baseline_3000-265ff5` (the 3000-WARC pool the playbook says to keep); rest deletable if downstream tokenized/deduped exist (already verified for the 10k pool in `10k_warc_cleanup.md`) |
| G | Results & coordination (tiny, irreplaceable): `metadata/data_curation_{isoflop,10k_natural,fixed_model,warc_scaling,core,10k_core,3k_core,core_bootstrap,10k_core_registry,10k_core_tasks}_results/`, `metadata/{olmo_bpb,olmes_base,mmlu_sl_verb}_results/` (all 6 buckets), `metadata/{olmix,olmix_swarm_*,olmix_claims,grid_v1,region_locks,rtp_eval,*sft_base*}`, `metadata/lpv1_1_projection`, `metadata/resiliparse_url_recovery`, `decontamination/dclm_core_v2`, `devset/`, `benchmarks/`, `experiments/scaling_law_sweeps/run_registry` (repo) | 0.14+ | 1.5 | **KEEP ALL**; these are the natural first candidates to consolidate under `users/michaelryan/metadata/...` (small, hot, mostly us-central1) — but every reader (spec_explorer, plot_*, dashboards, launchers' skip-if-done) hardcodes the current prefixes |
| H | Indices: `infinigram_indices/`, `bm25_indices/`, `url_index/`, `spec_explorer/text_index` | 0.7 | 9 | KEEP (Spec Explorer / BM25 backup depend on them) |
| I | Scratch/temp: `scratch/provenance_10k*`, `scratch/baseline_compare`, `tmp/ttl=2d/warc-cache` (auto), `tmp/extraction_manifests` (**no TTL segment — never reaped**), `sysprompt_pretrain/dclm30b` (~62 GB cleanup already deferred), `distill/`, `eval_datasets/*_hf_cache` (needed per region for eval sweeps) | 1.9 | 27 | mostly deletable except `eval_datasets/*` (KEEP per region) and `scratch/provenance_10k_devset` (devset provenance) |

Not mine (do not touch): `checkpoints/isoflop/` (Adam's scaling ladder, 187 TB),
`adamh-scaling-ladder-*`, `grug/`, `datakit/store_*`/`dedup*`/`minhash` (Will/datakit
team), `raw/nemotron_cc*`, `normalized/*`, `checkpoints/e3956*`, `exp5611*`,
`exp3956*` (SWE/kimi SFT), `pinlin_calvin_xu/`, `julian/`, `user/rav`.

## 5. Gaps in this accounting (to close before any cleanup pass)

1. Snapshot floor is 1 GiB per depth-3 prefix; the KEEP-critical group G is
   mostly below it. Need one cheap delimiter listing per `metadata/` root
   (not a recursive `du`) to enumerate results dirs exhaustively.
2. Object counts matter for move cost and for `_completed_*`/claim registries
   with millions of tiny objects (`documents/baseline_llm_extraction`,
   fast_curation `_claims_*`). Pull `object_count` from the same parquet.
3. Live writers: as of 2026-08-15 my Iris jobs are the pending
   `grid-lpv11-fastpipe-v1-10k-*` cells and `eval-bpb-smoke2`; the OLMIX
   resiliparse swarm (launched today, us-east5) and the 38-run lpv11 sweep write
   into B/B2/G. Any move of `metadata/*` or `checkpoints/isoflop-curation`
   while those run = duplicate writers / lost skip-if-done.
4. Ownership of shared-looking prefixes needs a yes/no from me:
   `raw/commoncrawl` (5.1 TB — which sub-prefixes), `checkpoints/medical-sft-base`
   (0.9 TB), `documents/bert_pipeline`, `tokenized/dclm_400m_1x_10k_*` mirrors in
   6 regions, `eval_datasets/*`.

## 6. Proposed sequencing (not started)

1. Land a single `USER_ROOT = "users/michaelryan"` convention in the fork
   (one constant, e.g. in `experiments/scaling_law_sweeps/region_tracker.py`
   next to `REGION_TO_BUCKET`), and route **new** launchers'
   `--results-prefix`/`--tracker-prefix`/checkpoint roots through it. Keep old
   readers scanning both old and new prefixes.
2. Build the exhaustive KEEP ledger (repo file, same shape as
   `experiments/rephraser/PRESERVE_FOR_WIPE.md` + `preserve_for_wipe.csv`) from
   the parquet + delimiter listings; one row per prefix with tier, size,
   object_count, buckets, reader files, keep/delete/review.
3. Only then decide moves: G (results, ~GBs) and A/E weights (~10 TB, mostly
   Coldline → +$200/mo transient) are the only candidates worth moving; B/C/D
   stay put and get registered.
4. Cleanup pass = the June protocol (`memory: project_storage_cleanup_protocol`):
   running-set snapshot, `.executor_status.lock` heartbeat check, downstream
   SUCCESS check, per-run `hf/step-<max>/config.json` verify, in-region deletes
   only, user runs the `rm` script.
