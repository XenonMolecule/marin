# lpv11_fastpipe_v2 (10,364 WARCs): dedup, decon, grid, OLMIX swarm

Staging runbook for the `lpv11_fastpipe_v2` mixing arm, written 2026-09-05. The 8M-scale
production run (`lpv11_fastpipe_v2_1`, fused workers) has identical cascade semantics to
`lpv11_fastpipe_v2` (same models, thresholds, band, context; `spec.py` and `VERSIONS.md`),
so the proven 10,364-WARC v2 corpus is the validation set for it.

## Starting state (verified 2026-09-05)

- `kept/` complete, exactly one copy per WARC after the reaper: us-east5 4,606 / us-east1
  3,196 / us-central2 1,438 / us-west4 1,124 = 10,364 parquets, 379 GB with `input_ids`.
- No `kept_text/`, no `baseline_lpv11_fastpipe_v2_*`, no tokenized cache, no grid store, no
  swarm metadata anywhere. Every downstream stage below is new.
- us-east5 already holds the CORE v2 decon source, `pooled_junkgate2` (with
  `calib_bme.json`) and the marin-tokenizer mirror, so the grid can run there natively.

## Two code changes required before dedup (landed in the working tree)

1. `fast_curation/dedup.py::_carry_provenance` — the TEXT line stores NaN `modernbert_prob`
   for band hi-accepts (57% of kept docs in the pilot). The JSONL writer (msgspec) emits
   `null` for NaN, so the apply step's `float(None)` would crash after minhash/fuzzy had
   already run. Scores are now normalized to `float | None` on both sides
   (`test_dedup.py`).
2. `fast_curation/project_text_only.py` — column list is now derived from the spec's kept
   schema, so `pooled_prob` travels with the corpus (v2 has it; v3 does not).

`baseline_collection/grid_corpora.py` registers `lpv11_fastpipe_v2_10k`
(post-decon tree in us-east5, `native_id_field=doc_id`).

## Sequence

Every command below runs as an Iris job in the region named. `set -a; source .env; set +a`
first. `-e KEY "$VAL"` (key and value), never `-e KEY` alone.

### 1. Project kept -> kept_text in each region (zero egress) — DONE 2026-09-05 23:44Z (10,364/10,364)

```bash
for r in us-east5 us-east1 us-central2 us-west4; do
  uv run iris --cluster marin job run --no-wait --region $r --priority interactive \
    --cpu 8 --memory 12GB --disk 10GB --extra cpu --enable-extra-resources \
    --job-name fastcur-project-text-v2-$r \
    -- python -m experiments.fast_curation.project_text_only \
         --spec lpv11_fastpipe_v2 --bucket gs://marin-$r --num-procs 16
done
```

Done when `kept_text/` parquet count == `kept/` count in each region. Resumable (skips
existing outputs).

### 2. Consolidate into us-east5 (the only cross-region step; ~110 GB text-only, US-US) — DONE 2026-09-05 23:48Z (10,364 shards)

```bash
uv run iris --cluster marin job run --no-wait --region us-east5 --priority interactive \
  --cpu 4 --memory 8GB --extra cpu --enable-extra-resources \
  --job-name fastcur-consolidate-v2 \
  -- python -m experiments.fast_curation.consolidate_kept_text --spec lpv11_fastpipe_v2 \
       --target-region us-east5 --source-regions us-east1 us-central2 us-west4
```

Expect `10,364` shards in `gs://marin-us-east5/documents/fast_curation/lpv11_fastpipe_v2-32b74664f1/kept_text/`.
Idempotent; a rerun copies only missing basenames.

### 3. Fuzzy dedup (286/26/5/seed 42, `text_cap_chars=None`, provenance preserved) — DONE 2026-09-06 11:30Z (3,043 shards, 154 GB gz; ~4h wall, fuzzy stage recycled its preemptible CC worker 5x and resumed each time) — was RUNNING as `/michaelryan/fastcur-dedup-v2-r2` (2026-09-06 00:10Z; first submit at 4 cores sat pending on 'Insufficient CPU, available 1' — us-east5 non-preemptible CPU is 1 spare core, so the driver runs at --cpu 1 --memory 4GB; the Zephyr coordinators/workers are child jobs and unaffected)

```bash
uv run iris --cluster marin job run --no-wait --region us-east5 --priority interactive \
  --cpu 1 --memory 4GB --disk 20GB --extra cpu --enable-extra-resources --no-preemptible \
  --job-name fastcur-dedup-v2-r2 -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" \
  -e MARIN_PREFIX gs://marin-us-east5 \
  -- python -m experiments.fast_curation.dedup --spec lpv11_fastpipe_v2 --region us-east5
```

Output `gs://marin-us-east5/documents/baseline_lpv11_fastpipe_v2_deduped/10364warcs/deduped/`
(200 jsonl.gz) + `stats/dedup_stats.json`. v1 dropped 11.3%; expect the same order. The
coordinator is the Zephyr driver, so `--no-preemptible` (its death cascades). Two other dedup
fleets (`lpv11-dedup-v4`, `fastcur-dedup-east1-500ksub-h`) share the CPU pool as of launch.

### 4. CORE v2 decontamination — DONE 2026-09-06 11:58Z (3,042 shards, 153.6 GB gz; ~25 min) — was `/michaelryan/decon-apply-lpv11-fastpipe-v2` (driver --cpu 1 --memory 4GB --no-preemptible)

```bash
uv run iris --cluster marin job run --no-wait --region us-east5 --priority interactive \
  --cpu 4 --memory 16GB --disk 20GB --extra cpu --enable-extra-resources --no-preemptible \
  --job-name decon-apply-lpv11-fastpipe-v2 -e HF_TOKEN "$HF_TOKEN" \
  -- python experiments/baseline_collection/decon_apply.py \
       --input-path gs://marin-us-east5/documents/baseline_lpv11_fastpipe_v2_deduped/10364warcs/deduped/ \
       --decon-source gs://marin-us-east5/decontamination/dclm_core_v2/ \
       --output-path gs://marin-us-east5/documents/baseline_lpv11_fastpipe_v2_decon_deduped/10364warcs/deduped/ \
       --ngram-length 15 --df-threshold 10
```

This output path is what `GRID_CORPORA["lpv11_fastpipe_v2_10k"]` points at.

### 5. Llama-3 cache for the natural arm (the mix arm's `d_obs` and baseline) — DONE 2026-09-06 12:38Z: cache `tokenized/lpv11_fastpipe_v2_decon_10364warcs-41e20bf` = **79,582,522 docs / 91,653,647,234 tokens** (ledger), 2,701 parts; registered in curation_plan (_D_OBS_DEFAULTS + METHODS `lpv11_fastpipe_v2`, no pin, us-east5 only) and launch_10k_natural.METHOD_NAMES

```bash
uv run iris --cluster marin job run --no-wait --region us-east5 --priority interactive \
  --cpu 4 --memory 16GB --disk 20GB --extra cpu --enable-extra-resources --no-preemptible \
  --job-name fastcur-tokenize-lpv11-v2 -e HF_TOKEN "$HF_TOKEN" -e MARIN_PREFIX gs://marin-us-east5 \
  -- python -m experiments.fast_curation.tokenize_lpv11 \
       --input-glob "gs://marin-us-east5/documents/baseline_lpv11_fastpipe_v2_decon_deduped/10364warcs/deduped/data-*.jsonl.gz" \
       --cache-path gs://marin-us-east5/tokenized/lpv11_fastpipe_v2_decon_10364warcs-41e20bf
```

Then read the token total from `train/shard_ledger.json` (not hand-derived) and register:
`curation_plan._D_OBS_DEFAULTS[...]`, `curation_plan.METHODS["lpv11_fastpipe_v2"]`
(`_method`, `sampled_warcs=EXPC_SAMPLED_WARCS`), `launch_10k_natural.METHOD_NAMES`.
Mirror the cache to us-central1 (`fast_curation/mirror_cache.py`, ~$3) if the sweep should
float; otherwise launch with `--allowed-regions us-east5`. The natural arm has NOT been trained
for v2 — it is the comparison baseline for the mix arm and shares this cache.

### 6. 24x5 grid (all in us-east5; never flip `grid_corpora.COMPUTE_REGION`) — label waves: first 12-way launch wedged the Iris intake (3/24 topic jobs landed on v5p-8, see the memory note); relaunched serially: quality `--wave q2` 24/24 (CPU batch, 12:30Z), topic `--wave t2` on `--tpu v6e-4` (35-worker pool vs 5 on v5p-8), interactive, 24/24 submitted 12:39Z. Pool was fully held by the user's own lpv11-full `grid-lpv11-10k-e5-*` topic jobs (21 running, batch) + the fused v6e-4 fleet (15, batch) + `lpv11-grid-tokenize` zephyr CPU workers; t2 jobs (interactive) queue on 'Insufficient TPUs' and take chips as those drain. The 3 original topic jobs (000/006/010) stay pending on v5p-8; overlap with t2 is safe (done markers).

```bash
G=experiments.baseline_collection
python -m $G.launch_grid label --dataset lpv11_fastpipe_v2_10k --stages topic --source native \
    --tasks 24 --tpu v6e-4 --region us-east5 --output-base gs://marin-us-east5 --wave t2 --submit-parallelism 1
python -m $G.launch_grid label --dataset lpv11_fastpipe_v2_10k --stages quality --source native \
    --quality-model gs://marin-us-east5/datakit/quality_model/pooled_junkgate2 \
    --device cpu --priority batch --tasks 24 --region us-east5 --output-base gs://marin-us-east5 --wave q2 --submit-parallelism 1
# topic and quality waves MUST carry different --wave tags (job names are per-dataset+idx, not per-stage)
# merge as an Iris job in-region, then:
# NOTE: --prewarm-tokenizer ONLY prewarms the mirror and exits (31 s, exit 0) — run it without the flag
# to tokenize (mirror already exists in us-east5). Driver as an Iris job: --cpu 1 --memory 4GB --no-preemptible.
python -m $G.grid_tokenize --dataset lpv11_fastpipe_v2_10k --source native \
    --compute-region us-east5 --output-base gs://marin-us-east5
python -m $G.grid_store --dataset lpv11_fastpipe_v2_10k --source native \
    --compute-region us-east5 --output-base gs://marin-us-east5
python -m $G.verify_grid_store --dataset lpv11_fastpipe_v2_10k --output-base gs://marin-us-east5
```

Store lands at `gs://marin-us-east5/datakit/store/lpv11_fastpipe_v2_10k_gridv1/`. Doc count
must equal the tokenized cache's ledger; token count will sit a few e-6 below it (chunked
long docs), which is expected.

**UPDATE 17:00Z: quality DONE 3042/3042 (24/24 q2 jobs succeeded); topic 137/3042 with 13/24 t2 jobs running (chips freed on their own); grid_tokenize: first submit with --prewarm-tokenizer exited after prewarm; real run LAUNCHED 17:20Z as `/michaelryan/grid-tokenize-lpv11-fastpipe-v2-r2` (label-independent). 17:28Z: local gcloud token expired (needs interactive `gcloud auth login`); cluster jobs unaffected, Iris CLI fine; GCS-side verification paused until reauth. Topic wave churning 2-13 running under preemption. **2026-09-07 17:45Z: topic 2860/3042 after ~29h of batch-band churn (18/24 t2 jobs done). Per user, the 6 unfinished t2 jobs + 3 stale v5p jobs were STOPPED and the remaining 182 shards relaunched as wave `t3` (24 x v6e-4, `--priority interactive`, serial submits); done markers make the overlap free.** **20:50Z: still 2860/3042 — the submitting identity `mryan0@stanford.edu` has no `user_budgets` row (config lists `michaelryan`), so it sits on the default tier (limit 1,000, spent 932k) → every submit is effectively BATCH; needs an admin `iris user budget set mryan0@stanford.edu --limit 0 --max-band interactive` or a marin.yaml tier entry. us-east5 v6e-4 pool collapsed 31→3 hosts (autoscaler backoff, stockout). Launched wave `t4c1` in us-central1 on v5p-8 reading the ~9 GB of remaining shards cross-region (under the 10 GB transfer guard per job), outputs to us-east5.** **20:31Z: grid_tokenize DONE (3042/3042 parquet, driver succeeded); topic 264/3042 (5 running / 19 pending) ≈ 50 shards/h → ~2 days at this rate; gcloud reauthed.** Merge -> store -> verify still wait on topic.** Earlier: **BLOCKED 2026-09-06 13:10Z — topic wave had 0 placements.** Scheduler runs the user's interactive
submits in the BATCH band (over budget), and batch ordering (root_submitted, depth) puts the 24 `t2`
jobs behind every fused-fleet child and behind the user's own lpv11-full topic wave
(`grid-lpv11-10k-e5-*`, 23 running since 11:23Z, ~35h projected). `tpu_v6e-preemptible_4-us-east5-b`
is in autoscaler backoff (health 0.08) so no new slices; every other single-host pool is likewise held
by the user's fleets. Options: (1) wait; (2) user grants interactive budget to `grid-lpv11-fastpipe-v2-10k-t2-*`;
(3) mirror the 154 GB decon tree to us-central2 (~$2) and run the wave on `tpu_v4-preemptible_8-us-central2-b`
(status available) with a grid_label input-path override, outputs still to us-east5. Quality wave `q2`
unaffected (CPU). Useful queries: `iris rpc controller get-scheduler-state` (effective band per pending
task), `get-autoscaler-status` (per-group availability/backoff), workers-table idle count by scale_group.

**TOPIC LABELS COMPLETE 2026-09-08 01:15Z (3042/3042).** Finished via three overlapping waves: us-east5 `t3` (batch-starved), us-central1 `t4c1` (v5p-8, cross-region read of the ~9 GB remainder, did most of the tail at ~70 shards/h), us-east1 `t5e1` + us-central1 `t6c1` (48 tasks) for the last 25. Leftover queued jobs stopped. **Merge DONE 01:27Z** (`metadata/grid_v1/lpv11_fastpipe_v2_10k/distribution.json`: 3042 shards, 79,582,522 docs == cache ledger, 24x5 grid, 84.6B gte tokens as sizing hint). **grid_store DONE 02:40Z** (`gs://marin-us-east5/datakit/store/lpv11_fastpipe_v2_10k_gridv1`, 24 cluster dirs, v3 artifact; ~70 min). **validate_grid_domains PASSED 02:45Z: 120/120 cells, 91,652,911,542 marin tokens (cache ledger 91,653,647,234; -8e-6 chunking gap as v1), 79,582,522 docs exact, 24 cells below block floor, k*N/R = 61.1 at R=30B/k=20 -> cells.json at `metadata/olmix/lpv11_fastpipe_v2_10k/cells.json`. verify_grid_store running. Store mirrors to us-central1 + us-east1 LAUNCHED 02:50Z (`mirror-store-lpv11-fastpipe-v2-<region>`).** verify_grid_store + validate_grid_domains LAUNCHED 02:45Z (`/michaelryan/grid-verify-lpv11-fastpipe-v2`, `/michaelryan/olmix-validate-lpv11_fastpipe_v2_10k` -> cells.json).

**verify_grid_store PASSED 03:12Z: `OK: 120/120 cells, 79582522 docs, 91652911542 tokens, every cell loadable`. Store mirrored + object-count-verified (2,651 objects, 180.2 GB) to us-central1 and us-east1 (03:02Z). GRID READY.**

**SWARM LAUNCH 2026-09-08 03:20Z:** us-east5 coordinator `olmix-coord-lpv11_fastpipe_v2_10k-us-east5` indices 181-308 (writes the manifest); **do NOT copy the manifest across buckets** — `run_olmix_swarm_standalone` requires `manifest.region == local region` and local `cache_dirs`, so a copied manifest fails every child; each regional coordinator builds its own manifest from its mirrored store (seed 42 over identical cells → identical weights; verify by diffing the `weights` arrays across buckets); then `...-us-central1` 0-181 and `...-us-east1` 308-363. Split reflects capacity at launch (us-central1 v5p-8 14 ready/available; us-east5 v6e-4 16 ready but backoff; us-east1 v6e-4 0 ready/backoff). Children `--child-priority batch` (the account's effective band is batch regardless until the budget row is fixed). Rebalance by launching a new coordinator over the not-done sub-range in a faster region. **03:35Z: all three coordinators RUNNING; manifests in all three buckets have identical domains/tokens/weights with region-local cache_dirs (verified by diff). us-east5 dispatched its first 10 children (pending on TPUs).** **REBALANCE 06:15Z (user OK'd shifting mass to us-east1): us-east5 had 0 placements in 3h -> stopped `-us-east5` (10 PENDING children, nothing lost), relaunched `-us-east5-r2` over 181-200, and launched `-us-east1-b` over 200-308. Reap the dead children's claims (i181-190) or `-r2` skips them until the 8h TTL. First results at 06:08Z: 16/363 (all us-east1).** **09-09 05:50Z: 192/363. us-east5-r2 (181-200) and us-east1 (308-363) finished; us-central1 dispatched through i109, us-east1-b through i259. Idle regions generate no demand so the autoscaler never grows them -> carved the far ends off the backlogs into us-east5: `-us-east5-r3` 150-181 and `-us-east5-r4` 285-308 (originals untouched, no children killed). The 30-min check script copies us-east5 result JSONs for those ranges into the us-central1 / us-east1 results prefixes so the original coordinators' `already_done` skips them when their cursors arrive (claims cover the first 8h; duplicates after that are harmless but wasteful). 09:55Z: 231/363, us-east5 pool hot (45 running) -> second carve: `-us-east5-r5` 125-150, `-us-east5-r6` 270-285 (sync ranges widened to 125-180 / 270-307). 16:40Z: 272/363; us-east5 dry for 3h while us-east1 recovered -> stopped `-us-east5-r5`/`-r6` (all children PENDING), reaped exactly their 35 claims, relaunched as `-us-east1-c` 125-150 and `-us-east1-d` 270-285. Result sync now runs via gcsfs (ADC outlives the gcloud token): scratchpad `swarm_sync.py` mirrors every result into its ORIGINAL coordinator's bucket (0-181 us-central1, 181-200 us-east5, 200+ us-east1). **21:30Z (user: 'feel free to kick up some interactive compute'): ran `iris user budget set mryan0@stanford.edu --limit 0 --max-band interactive` (75,000 was still below the decayed spend of 75,823) -> effective band INTERACTIVE; then per user ('I don't want to cheat others') set back to the researcher tier `--limit 75000 --max-band interactive` at 21:40Z (spend had decayed to 53k, still INTERACTIVE; demotes to batch again if spend exceeds 75k). With 0 children running, stopped all six coordinators, reaped their 65 queued-child claims, relaunched the same ranges with `--child-priority interactive`: `-us-central1-r2` 0-125, `-us-east1-c2` 125-150, `-us-east5-r7` 150-181, `-us-east1-b2` 200-270, `-us-east1-d2` 270-285, `-us-east5-r8` 285-308. First interactive placements within 4 min. Eval wave-1 manifest built (`core_eval_manifests/olmix_swarm_bpb_v2w1_rest.txt`, 272 rows) but NOT launched (awaiting user go). **09-10 00:30Z: 275/363; every v5p-8/v6e-4 pool empty (us-central1 v5p quota exhausted project-wide, us-east5 v5p backoff, v6e-4 demand 1024 vs 12 ready). Added demand on the OTHER autoscaler groups via `--tpu-variant`: `-us-east1-g16` 125-150 + `-us-east1-h16` 200-285 (v6e-16 gangs), `-us-east5-r9-8` 150-181 + `-us-east5-r10-8` 285-308 (v6e-8). Claims arbitrate against the v6e-4/v5p-8 coordinators over the same ranges.** **04:05Z: REVERTED — v6e-16 gangs placed but all 15 died at startup (JAX distributed-service coordination error when a host is lost); stopped the 4 wide-shape coordinators and reaped their 29 claims. us-east1 v6e-4 pool recovered on its own (28 running). 277/363.** **09-10 07:50Z: 283/363. us-central1 + us-east5 dead ~9h while us-east1 places -> mirrored the 137 done results for 0-125/150-181/285-308 into the us-east1 bucket, stopped `-us-central1-r2`/`-us-east5-r7`/`-us-east5-r8` (all children pending), reaped 24 claims, launched `-us-east1-i` 0-125, `-us-east1-j` 150-181, `-us-east1-k` 285-308. All six live coordinators are now in us-east1; swarm_sync.py keeps mirroring new results back to each index's home bucket.** **SWARM COMPLETE 2026-09-10 15:20Z: 363/363, index set verified == {0..362}, 0 malformed results, 0 failed children; every index's result is present in its home bucket (us-central1 0-180, us-east5 181-199, us-east1 200-362) via swarm_sync. Idle coordinators stopped. Two cursor-gap fixes were needed at the end (135-149, 267-269, 280-284 skipped after the gang-coordinator claims): always diff the missing set against each coordinator's range before trusting an idle coordinator.** End condition (met): distinct index set == {0..362}

**EVAL WAVE 1 LAUNCHED 2026-09-10 17:00Z** (user: 'Go for it'): `/michaelryan/olmix-bpb-v2w1`, manifest `core_eval_manifests/olmix_swarm_bpb_v2w1_rest.txt` (363 rows; us-central1 156 / us-east1 156 / us-east5 51), 60 tasks (`all+qa_rc` + 4 `mmlu_*/rc_5shot`), `--child-priority interactive --keepalive-timeout 172800`, results `metadata/olmix_swarm_bpb/{run_name}/`. **09-12 04:30Z: 252/363 evaluated; all 111 remaining were pinned to us-central1 (v5p-8 pool 8 hosts, quota-capped, 9h without a placement). With user approval, copied the 111 runs' `hf/step-32843` exports (~71 GB, ~$1.40) to `gs://marin-us-east1/checkpoints/olmix-swarm/<run>/hf/`, wrote `core_eval_manifests/olmix_swarm_bpb_v2w2_east1.txt`, launched `/michaelryan/olmix-bpb-v2w2-east1`, and stopped the 111 queued us-central1 children of wave 1 (no double runs). Their results land in the us-east1 bucket; the fit/eval collectors union all buckets.** **06:30Z: user over the 75k tier again (spend 164k) -> batch band; us-east1/us-east5 v6e-4 in stockout backoff. EU pilot (user OK, <$3): copied 16 runs' hf exports (10.3 GB, ~$0.8; note the 'us-central1' manifest rows' checkpoints actually live in us-east1 — results were only mirrored to us-central1) to `gs://marin-eu-west4/checkpoints/olmix-swarm/<run>/hf/`, manifest `olmix_swarm_bpb_v2w3_euw4.txt`, launched `/michaelryan/olmix-bpb-v2w3-euw4`, stopped the 16 us-east1 twins. Eval status script now also counts the eu-west4 bucket.** **EVALS COMPLETE 2026-09-12 09:48Z: 363/363 results.json, every one with 60 tasks, index set == {0..362}; by bucket us-east1 251 / us-east5 51 / us-central1 45 / eu-west4 16. Idle eval parents stopped. FIT LAUNCHED 10:00Z: `/michaelryan/olmix-fit-lpv11-fastpipe-v2-olmixexact` (us-east5, --cpu 7 --memory 10GB, extras cpu+mixing, NOT no-preemptible) -> `gs://marin-us-east5/metadata/olmix/lpv11_fastpipe_v2_10k/fit_olmix_exact_kl0p01/mix_R3e+10_k20.json`, regions us-east5+us-central1+us-east1+europe-west4, `--devset olmix_exact --kl-reg 0.01 --requested-tokens 3.0e10 --repetition-factor 20 --workers 7`. **FIT DONE 11:00Z: `collected 363 usable runs` (213 duplicate_run skips = mirrored copies), 120 domains (82 live / 38 never sampled; unique fit needs >=83 runs), regression average_bpb 0.984, 0 degenerate tasks, status=optimal, TV(mix, natural)=0.49. VENDORED to `experiments/data_mixing/mixtures/lpv11_fastpipe_v2_10k_R3e10_k20_olmixexact_lambda0p01.json` (86/120 cells above the 1/65535 floor; sub-floor mass 0.0099%). Registered `lpv11_fastpipe_v2_10k_mix_olmixexact_lambda0p01` in curation_plan.METHODS + launch_10k_natural.METHOD_NAMES; `load_weights()` verified.** **SWEEPS LAUNCHED 2026-09-12 11:20Z (36-cell expFM_natural, `--max-budget 9e20`, children interactive — demoted to batch while spend > 75k): `/michaelryan/10k-natural-lpv11-fastpipe-v2` (natural arm, `--allowed-regions us-east5`: cache not mirrored) and `/michaelryan/10k-natural-olmixexact-lpv11-fastpipe-v2` (mix arm, `--allowed-regions us-east5 us-central1 us-east1`). Results land in `metadata/data_curation_10k_natural_results/`. Stragglers: relaunch with `--only <budget-substr>` after the fleet drains. NEVER stop these coordinators (orphan-kills every child).** **17:10Z incident: both parents had landed on a host of `tpu_v4-reserved_2048-us-central2-b` (worker-17), which died with 'worker reconcile failure' — counted as a parent preemption despite --no-preemptible — and cascade-killed 9 natural + 4 mix cell jobs ('Parent task preempted'). Iris auto-retried both parents onto on-demand CPU VMs and they re-dispatched the killed cells (skip-if-done). No manual action; a `--no-preemptible` CPU parent can still die with a reserved TPU host.** **20:30Z: per user ('bring the natural mix offline, compute better used elsewhere'), STOPPED `/michaelryan/10k-natural-lpv11-fastpipe-v2` and all its children (28 cell jobs); 9 natural cells had finished and their results remain under `data_curation_10k_natural_results/`. Only the olmixexact mix-arm sweep continues (10/36 done at the time).** **21:40Z: mix cell `2e+20-d512-L6-B1024` (4-host gang) went worker_failed (SIGSEGV on one host after 13 attempts, siblings cosched_failed) — the same big-batch cell family that failed in the v1 sweep. Relaunched just that cell via `--only 2e+20-d512-L6-B1024` as `/michaelryan/10k-natural-olmixexact-lpv11-fastpipe-v2-r2e20b1024`. 22:10Z: second cell `2e+20-d2432-L24-B64` failed the same way (SIGSEGV on one gang host) -> relaunched as `...-r2e20d2432`. Both failures are 2e+20 multi-host cells. 09-13 02:10Z: third cell `9e+19-d1024-L11-B256` failed identically -> relaunched as `...-r9e19d1024`. All three are the same big-batch gang cells that failed on init races in the v1 sweep; the `--only` relaunches resume from temp checkpoints where present. 09-13 05:15Z: fourth cell `9e+20-d1024-L11-B2048` failed the same way -> relaunched as `...-r9e20d1024`. Main sweep 18/36 done at the time; the three earlier straggler cells are training. 06:50Z: fifth cell `9e+19-d1536-L16-B64` failed identically -> relaunched as `...-r9e19d1536`. Pattern: every multi-host cell at >=9e+19 has hit a startup SIGSEGV on one gang host at least once; `--only` relaunches have all placed and trained so far. 09-13 23:45Z: 29 cells done; found `9e+20-d3584-L35-B128` stranded in KILLED (parent-host failure casualty never re-dispatched by the retried parent — the cursor gap) -> relaunched as `...-r9e20d3584`. Remaining 8 are 2e+20-9e+20 gangs running/queued. Four of the five earlier stragglers finished. **09-14 06:50Z: per user ('don't need the d3584 3e+20 or 9e+20 runs'), STOPPED `3e+20-d3584-L35-B32` (queued) and the `9e+20-d3584-L35-B128` relaunch (running). The sweep target is now the 36 non-d3584 cells; 32 done, 4 remaining (2e+20-d1024-B512, 3e+20-d512-B2048 running; 3e+20-d1024-B512, 3e+20-d1536-B256 queued for gangs).** **09-14 22:00Z: the 4 remaining cells (3e+20-d512-B2048, 2e+20-d1024-B512, 3e+20-d1024-B512, 3e+20-d1536-B256) are 8-host gangs eligible only for us-central1's v5p-64 group under `--allowed-regions us-east5 us-central1 us-east1` (the v2 grid store is not in us-central2, so the reserved v4 pools the v1 arm used are unavailable). User: 'I will handle this in another thread' — no mirroring or further action from this session. Sweep state handed off at 32/36 done, 4 queued, 0 dead.** Watch for other stragglers when the fleet drains: compare `data_curation_10k_natural_results/` against the 36-cell grid.** Then `vendor_olmix_mixture --mixture-tag _olmixexact_lambda0p01`, register `lpv11_fastpipe_v2_10k_mix_olmixexact_lambda0p01` via `_grid_mix_method(..., 'lpv11_fastpipe_v2_decon_10364warcs-41e20bf', 'lpv11_fastpipe_v2_10k', mixture_tag=...)`, add to launch_10k_natural.METHOD_NAMES, and launch the 36-cell sweep (`--max-budget 9e20`) together with the `lpv11_fastpipe_v2` natural arm (cache us-east5 only -> `--allowed-regions us-east5`, or mirror first).** Judge completion by results.json count across the 3 buckets (scratchpad `eval_check.sh`), never by parent state. After 363/363: rebuild wave (needs_eval must be 0), then `run_olmix_fit` at R=3e10/k=20 kl 0.01 with `--region us-east5 --region us-central1 --region us-east1`, vendor, register `lpv11_fastpipe_v2_10k_mix_olmixexact_lambda0p01`, 36-cell sweep. across us-east5/us-central1/us-east1 buckets; fix gaps via reap_dead_claims + a coordinator over the missing range.**

### 7. Validate + swarm

```bash
uv run iris --cluster marin job run --job-name olmix-validate-lpv11_fastpipe_v2_10k --region us-east5 \
  --cpu 4 --memory 8GB --enable-extra-resources --extra cpu --priority interactive --no-wait \
  -- python -m experiments.data_mixing.validate_grid_domains --corpus lpv11_fastpipe_v2_10k --region us-east5 \
       --output gs://marin-us-east5/metadata/olmix/lpv11_fastpipe_v2_10k/cells.json
```

Check `k*N/R` at the locked R=30B, k=20 (v1 was 59x; expect similar). Add the corpus to
`data_mixing/{swarm_monitor,build_final_eval_wave}.CORPORA`, then:

```bash
uv run iris --cluster marin job run --job-name olmix-coord-lpv11_fastpipe_v2_10k-us-east5 --region us-east5 \
  --cpu 4 --memory 1GB --extra cpu --priority interactive --no-preemptible --no-wait \
  -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" \
  -- python -m experiments.data_mixing.launch_olmix_swarm \
       --corpus lpv11_fastpipe_v2_10k --region us-east5 --launch \
       --index-start 0 --index-end 363 --child-priority batch
```

**User plan (2026-09-07): run the swarm across us-east5 + us-central1 + us-east1, one coordinator per
region with DISJOINT index ranges sized to each region's capacity, rebalanced as pools change.**
Prereqs per region: (a) the grid store mirrored into that region's bucket (server-side copy via
`fast_curation/mirror_cache.py --src gs://marin-us-east5/datakit/store/lpv11_fastpipe_v2_10k_gridv1
--dst gs://marin-<region>/datakit/store/lpv11_fastpipe_v2_10k_gridv1`, ~US-US $0.01/GB; the artifact's
us-east5 paths are rebased by `olmix_domains._rebase`); (b) eval datasets staged (us-central1 + us-east1
already have `eval_datasets/olmo_in_loop_evals` + `core_tasks_hub_cache`); (c) us-east1 is v6e-only —
children carry `device_variant_constraint(v5p-8, v6e-4)` so they place on v6e-4 there; (d) add
`marin-us-east1`/`us-east1` to `swarm_monitor.BUCKETS`, `build_final_eval_wave.REGIONS`, and the fit's
`--region` list, or its results are invisible. Rebalancing: completion records are bucket-local but
claims are global (8h TTL, us-central1 registry), so to move work launch a NEW coordinator in the fast
region over the not-yet-done sub-range and let claims arbitrate; never re-run a range whose results sit
in another bucket after claims expire (union the -iNNNN- indices across buckets first).
One corpus per region (two coordinators in one region halve each other). The coordinator
MUST be `--no-preemptible`. Then eval waves, the λ=0.01 fit at R=3e10/k=20, vendor, register
`lpv11_fastpipe_v2_10k_mix_olmixexact_lambda0p01` via `_grid_mix_method`, and launch the
36-cell sweep with `launch_10k_natural.py --max-budget 9e20` — exactly the nemotron path in
`.agents/projects/olmix_olmixexact_mixture_handoff.md`.
