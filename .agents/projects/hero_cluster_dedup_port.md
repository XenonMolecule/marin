# Port the hero-run fuzzy cluster dedup into this fork

Status: Step 1 (code port) DONE 2026-09-18, commits f47919338b, 3bda144cb7, 6b3edbee30. Steps 2-5 open; nothing launched.

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
- Coordinators `--no-preemptible`; `max_task_failures` > 0; region-pin to the
  bucket holding the normalized tree; `--submit-parallelism 1`.

## Step 4: wire the drivers

Current DAG: `reshape -> normalize -> minhash -> fuzzy CC -> apply`.
New DAG:

```
... -> fuzzy CC -> large_clusters -> cluster_text -> cluster_verify -> apply_verified
```

- Wrap the three scripts' entry functions as `StepSpec`s (`cluster_verify_step`
  already exists). Put the full `ClusterDedupParams` and `max_cluster_size` in
  `hash_attrs` so a threshold change moves the output path.
- New `_apply_verified_dups`: join normalized shards to the sparse marker tree by
  basename and drop rows where `dup_doc` is true. Markers are sparse: a missing
  marker row or file means keep. A missing file for a shard that the manifest
  says has markers must raise, not read as "no dups"
  (`feedback_never_encode_failure_as_zero`). Keep `_carry_provenance` intact.
- Write to a new tree (`.../deduped_c075/`); never overwrite `deduped/`. The old
  dedup trees stay until results are verified (`project_dedup_verify_fix_todo`).
- Replace the old apply rather than flag-switching it, once Step 5 passes.
  `dedup_extracted.py` gets the same change; factor the shared steps into one
  module both drivers import instead of copying.

Cost note: minhash and CC outputs are reused as-is. Only cluster_text (one
shuffle of candidate text) and cluster_verify (map) are new work, so existing
10k / 154k / 500k candidate trees do not need recomputing.

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
