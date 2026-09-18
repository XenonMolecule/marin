# Port the hero-run fuzzy cluster dedup into this fork

Status: Step 1 (code port) DONE 2026-09-18, commits f47919338b, 3bda144cb7, 6b3edbee30. Local three-stage smoke passed (synthetic chained component of two unrelated
families: old rule keeps 1 of 4, hero rule keeps 2, markers land beside the
right normalized shards). Steps 2 and 4 done (params pinned in `verified_dedup.C075_PARAMS`, drivers wired). Step 5 open; nothing launched on Iris.

## Goal

Replace "drop every non-canonical member of a MinHash connected component" in
`experiments/fast_curation/dedup.py` and
`experiments/baseline_collection/dedup_extracted.py` with the rule the Harrier
hero store was built with: MinHash CC only proposes candidate clusters, and a
document is removed only when a longer surviving document in the same cluster
contains it (word 3-gram containment >= 0.75).

Motivation: fused-worker survival fell 90% / 46% / 30% at 10k / 154k / 500k
WARCs. Transitive closure chains unrelated documents into one component and the
current apply step deletes all but one of them.

## Source of truth

`upstream/rav/hero_store_dup_075` pinned at `c2f981fe3d` (draft PR #8405).

Do not port from the review stack (#8723 / #8724 / #8725). It is the same
algorithm cleaned up, but the solver makes different decisions:

| | hero `c2f981fe3d` | stack |
|---|---|---|
| novel-token check | casefolded whitespace word sets | hashed unigrams, case-sensitive |
| member order tie-break | `(-chars, index)` | `(-chars, id, index)` |
| index candidate cut | `argpartition` on shared count | `lexsort((rank, -shared))` |
| longer-member guard in loop | `if member.chars > other.chars: continue` | folded into `order` |
| `rule_version` | `containment_ngram3_v2` | absent |

Evidence the hero branch built production: `harrier_mix_2026_08_18.py` on main
trains from `store_4d2e363d`, documented as `store_81e7e39a` rebuilt with 16
sources exempt from fuzzy dedup. The hero branch introduces `store_81e7e39a`,
carries the 16-source exemption commit (`fd332ba6d9`), and points
`verified_fuzzy_dups` at `user/rav/dedup/verified/v11-c075-restored`. Main still
points at the older `datakit/verify_fuzzy_dups_c757e4f0`. Inferred from the
repo; confirm with rav or the store's `.artifact.json` if exactness matters.

When the stack lands upstream, reconcile then. Expect marker differences at the
margin, so do not mix hero-rule and stack-rule trees in one comparison.

## Why no upstream sync first

The hero branch is based on 08-18 main; this fork's merge-base is 08-13. It
still imports `zephyr.execution`, and every module it needs already exists here
(`marin.datakit.copartitioned.write_copartitioned_source_manifest`,
`marin.execution.step_status`, `rigging.filesystem.storage_path`,
`rigging.filesystem.factory`, `dupekit.hash_xxh3_64*`). The 549-commit sync
(6 conflicts in dry run, plus fork-wide import moves) is a separate task and
must not gate this.

## Step 1: port code

Snapshot-commit the dirty tree first so the port is one reviewable diff.

Take verbatim from `c2f981fe3d` (`git checkout c2f981fe3d -- <path>`):

- `lib/marin/src/marin/processing/classification/deduplication/cluster_dedup.py`
- `lib/marin/src/marin/processing/classification/deduplication/cluster_verify.py`
- `experiments/datakit/scripts/fuzzy_large_clusters.py`
- `experiments/datakit/scripts/fuzzy_cluster_text.py`
- `experiments/datakit/scripts/fuzzy_cluster_verify.py`
- `tests/processing/classification/deduplication/test_cluster_dedup.py`
- `tests/processing/classification/deduplication/test_cluster_verify.py`

Hand-apply to `verify_fuzzy_dups.py`: only the `VerifiedFuzzyDupsAttrData`
change (`verification` / `local_representatives` become optional, new
`rule: ClusterDedupParams | None`, the exactly-one validator). Skip the
`stats(timeout=...)` / `MemoryStoreUnavailable` hunk; it needs zephyr #8380,
which this fork does not have, and the cluster path never touches the memory
store.

Zephyr commits from the hero branch: only `edfb6de79b` (cache one pyarrow
filesystem per process) was taken. It is backend-agnostic: a fresh connection
pool per written file exhausts ephemeral ports on GCS as well as S3.

Dropped after inspection:

- `9714d891db`, `fecb414201`, `60962a2622` (retries): they classify errors with
  `rigging.filesystem.s3_errors`, which arrives with the #8305 rigging refactor
  this fork lacks, and they target CoreWeave S3 failures. We run on GCS. The
  hero branch's own later commit says they treated the symptom.
- `b841b8d7ae` (sidecar read pool 32 -> 8): sized for 2,048-reducer waves
  offering ~65k connections to one endpoint. It would slow sidecar reads for
  every Zephyr job here. If cluster_text reduce waves show connection-level
  errors, lower `_SIDECAR_READ_CONCURRENCY` in `zephyr/shuffle.py` first.
- `642fcc8aad` (verify retry budget): already inside the tip `cluster_verify.py`.

Also ported: `experiments/datakit/scripts/fuzzy_cluster_sizes.py`, which
`fuzzy_large_clusters.py` imports (`candidate_shard_paths`).

Do not port: `datakit_store.py`, `produce_store.py`, `hero_data*`,
`repack_fuzzy_dups.py` (Focus Crawl layout repair), the source exemption list
(generated / templated corpora; we have one web source), quality-scorer
changes, and the analysis scripts. Revisit `fuzzy_marker_compare.py` and
`fuzzy_verification_cluster_sample.py` in Step 4 if hand sampling gets tedious.

One deviation (done, 6b3edbee30): hero `build_shards` took `--verified`, a
pipeline-verifier artifact, only to pin source order and tags. Tags are now
rebuilt by the rule that artifact used (`source_{rank:03d}` over sorted source
keys, as `build_copartitioned_shards` assigns), so numbering matches and no
removal decision changes.

Gate: `uv run pyrefly check`, the two ported test files, `./infra/pre-commit.py --all-files --fix`.

## Step 2: parameters (pin explicitly, never rely on defaults)

Code default is 0.60; production used 0.75. At 0.60 the hero audit found ~4 of
17 sampled marginal removals were false positives.

```
ClusterDedupParams(
    rule_version="containment_ngram3_v2",
    ngram_size=3,
    minimum_containment=0.75,
    exact_scan_maximum=256,
    probe_ngrams=32,
    maximum_posting_length=512,
    maximum_candidates=32,
)
--max-cluster-size 100000        # fuzzy_cluster_text oversized split
--minimum-size 100000 --stride 256   # fuzzy_large_clusters
```

Candidate generation stays as is (286 perms / 26 bands / 5-char / seed 42 /
~0.75 Jaccard, no paragraph dedup). Hero candidates used the datakit defaults;
check they match `project_baseline_dedup_config_verified` and record any
difference, but do not change ours: every existing curation arm shares it.

## Step 3: resources (the real risk)

Hero defaults assume CoreWeave nodes: workers 32 cpu / 128-192g, map tasks 12g,
reduce tasks 26g, 512g disk. Our Iris CPU pool tops out at 8 core / 16 GB, so a
26g reduce task cannot be scheduled at all.

- Workers 8 cpu / 16g; map `--task-ram` 6g with `--shards-per-task` 2-4.
- Shrink reduce partitions instead of growing RAM: raise `--output-shards`
  until the largest partition plus sort spill fits in ~12g. Start 10k at 512.
- Zephyr external sort spills to tmpfs, which counts as RAM here
  (`feedback_zephyr_external_sort_spills_to_tmpfs`). Budget for it.
- Floor: `MAXIMUM_CLUSTER_CHARS` is 256 MiB per group, and one giant component
  lands in one reduce task. `fuzzy_large_clusters` output tells us the largest
  component before we commit to a size. If 500k has a component that cannot fit
  after the two-level split, lower `--max-cluster-size`; that is a recorded
  deviation (the MinHash split has a containment-recall limit).
- `fuzzy_large_clusters.py` hardcodes map-task `disk="32g"` with no flag, and
  Zephyr rejects a task larger than its worker. Launch that stage with worker
  disk >= 32g, or add a `--task-disk` flag to match the other two scripts.
- Coordinators `--no-preemptible`; `max_task_failures` > 0; region-pin to the
  bucket holding the normalized tree; `--submit-parallelism 1`.

## Step 4: wire the drivers (DONE 2026-09-18)

Both drivers take a required `--dedup-rule {cc_canonical,c075}`. Shared code is
`experiments/baseline_collection/verified_dedup.py`.

```
cc_canonical   ... -> fuzzy -> deduped/                      unchanged; every tree the paper reports
c075           ... -> fuzzy -> dedup_c075/large_clusters
                            -> dedup_c075/cluster_text
                            -> dedup_c075/verified
                            -> deduped_c075/   stats_c075/
```

Paper replicability: tag `paper-dedup-cc-canonical` (38da2431d2) is the driver
code before wiring. After wiring, `--dedup-rule cc_canonical` runs the same
`_apply_fuzzy_dups` into the same `deduped/` path; the only diff against the tag
inside that branch is indentation. The rule is a required flag so no launch
picks one silently. A frozen copy of the drivers was considered and rejected: it
would still import the live `lib/marin` dedup code, so only the tag freezes it.

Both rules share every step through `fuzzy`, so existing reshape / normalize /
minhash / fuzzy outputs are reused and the two rules judge identical candidates.

`apply_verified_dups` integrity: marker files are sparse (a shard with no
removals has none), so a lost file is invisible per shard. The run instead
requires total drops == the `markers` counter on the verifier artifact, and each
shard requires every marked id to be found. A resumed apply (`skip_existing`)
cannot check the total and logs that it did not.

`--cluster-*` flags size the c075 stages; defaults are the hero run's. These
drivers already run 96-192g Zephyr workers, so the 8c/16g ceiling in Step 3
applies to the Iris job that hosts the coordinator, not to the worker pool.

Not wired: `dedup_resiliparse_warc_scaling.py` and
`baseline_collection/dedup/*.py`, which copy the old apply step. Wire them the
same way before comparing those arms against a c075 arm.

## Step 4b: re-apply entry point (DONE 2026-09-18)

`experiments/fast_curation/reapply_verified_dedup.py --tree <root> --output {counts,tree}`
pins a tree's finished `normalize/` and `fuzzy/` and runs only the c075 stages.
It exists because `dedup.py` rebuilds from `kept_text`, and the 154k / 500k fused
trees had `reshape/`, `minhash/`, `fuzzy/metadata/` and `kept_text/` pruned on
2026-09-14. All three validation runs go through it so they share one path.

`--output counts` writes markers plus `stats_c075/survival.json`: documents and
characters removed under cc_canonical and under c075, from one pass over the
normalized shards and the same fuzzy components. `--output tree` also writes
`deduped_c075/`.

Inputs verified on GCS 2026-09-18 (status SUCCESS, artifacts present, attr shards
1:1 with normalize shards):

| tree | region | normalize shards | docs after exact dedup | component members | cc_canonical removes |
|---|---|---|---|---|---|
| `baseline_lpv11_fastpipe_v2_deduped/10364warcs` | us-east5 | 3,042 | 89.6M | 13.76M | 10.04M (11.2%) |
| `baseline_lpv11_fastpipe_v2_1_fused_deduped/154117warcs` | us-east1 | 9,171 | 973.7M | 356.5M | 258.5M (26.5%) |
| `baseline_lpv11_fastpipe_v2_1_fused_deduped/500000warcs` | us-east1 | 29,084 | 2.69B | 1.42B | 1.06B (39.4%) |

Sizing rule: sort spill is tmpfs, so keep a cluster-text partition well under the
reduce task's in-memory budget by raising `--cluster-output-shards`. Workers must
be bin-packable (the CPU VM group is 8 core / 16 GB); hero's 32-core workers never
schedule. The Iris parent only orchestrates: `--cpu 2 --memory 8GB`. A 4-core
non-preemptible parent sat pending 20 min in us-east5 on 2026-09-18.

| tree | workers | task / reduce RAM | output shards | shards per task |
|---|---|---|---|---|
| v2 10k | 64 x (4 cpu, 32g) | 6g / 8g | 512 | 8 |
| 154k (planned) | 256 x (8 cpu, 64g) | 6g / 16g | 4096 | 16 |
| 500k (planned) | 512 x (8 cpu, 64g) | 6g / 16g | 16384 | 32 |

Cost of the two east1 runs (needs user confirmation, over the $10 rule): about
$12-15 of GCS operations (the shuffle reads maps x reduces chunks: ~15M at 500k),
plus ~3 TB of cluster-text scratch and comparable shuffle scratch at $2-4 per day
until deleted. `--output tree` would add ~5-6 TB of `deduped_c075/`, about
$110 per month. Use `--output counts` for the curve and delete
`dedup_c075/cluster_text/` once `survival.json` lands.

## Step 5: validate, smallest first

1. 10k fused tree. Expect survival above the old ~90%. Emit: old CC removals,
   verified removals, rescued count, cluster-size histogram, containment
   histogram of removals.
2. Hand-read ~50 rescued documents (old rule dropped, new keeps) and ~50
   removals in the 0.75-0.85 band. Look for the known failure mode: documents
   that are one unbroken token share almost no word 3-grams except boilerplate
   headers. Web analogue: boilerplate-heavy or template pages.
3. 154k, then 500k. Success = the survival curve flattens materially versus
   90 / 46 / 30. If it does not, chaining was not the main cause and the curve
   is mostly real duplication; stop and report before spending on 2M.
4. Sanity anchor: hero removed 13.36% corpus-wide at 0.75. Different corpus, so
   only an order-of-magnitude check.
5. Downstream: rerun the affected mix/token counts only after 3 passes.

## Open questions

- Confirm with rav that `v11-c075-restored` came from `c2f981fe3d` unmodified
  ("restored" suggests a rebuild or recovery).
- Does any component at 500k exceed what an 8c/16g pool can solve? Answered by
  the `fuzzy_large_clusters` run; cheap, do it first.
- Whether `dedup_extracted.py` arms (resiliparse, trafilatura baselines) should
  be re-deduped for comparability. Probably yes for any arm plotted against a
  re-deduped fastpipe arm.
