# Reliable (claimant) fuzzy dedup: runbook

Code: `experiments/fast_curation/rdedup/` (tests beside it). It replaces the Zephyr chain in
`experiments/fast_curation/dedup.py` (kept working, untouched) for runs where that chain cannot
survive the cluster: 8 attempts at 2M WARCs died between 2026-09-24 and 09-27.

## Why it survives

No gang, no stage coordinator, no process whose death loses a stage.

- A run is a frozen `plan.json` under a scratch **run root** (created once with an atomic
  create; a second `init` with different settings is refused). Every task, path and partition
  derives from the plan, never from a listing, so leftovers of another attempt cannot leak in
  (failure mode 7 of the 2M attempts). Output roots carry `.rdedup_run.json` naming their run.
- A stage is `n` numbered tasks plus a seal. Workers claim a task by creating
  `claims/NNNNNN.json` (GCS `if_generation_match=0`), heartbeat by replacing it against the
  generation they last wrote, and finish by creating `done/NNNNNN.json` after the outputs are
  durable. A claim silent for 8 minutes is taken over (generation-matched replace, one winner);
  the old owner's next heartbeat fails and it exits. A task that kills 6 workers in a row is
  left alone ("POISONED" in logs) so it stalls visibly instead of burning the fleet.
- Tasks are deterministic: a zombie writer after a takeover writes identical bytes.
  Verified: killing the worker job mid-pass1 on a 300-WARC run produced byte-identical outputs
  (same CRC32C for all 28 files) and identical stats to an undisturbed run.
- Workers (`worker.py`) are stateless 8-CPU preemptible batch jobs pinned to us-east1. Each runs
  8 slot processes; a slot works on the frontier stage if the stage's memory estimate fits. A
  dead slot is restarted. When the last stage seals, workers exit 0.
- The launcher (`launcher.py run`) is a 1-CPU job in us-central2 (roomy on-demand pool). It
  only adds worker jobs (a chunk at a time, only when nothing is pending) and exits when the
  report stage is sealed. Killing and relaunching it is harmless.
- Shuffles never list data: each map task writes ONE object with one byte range per reduce
  partition; the seal transposes the offsets into `_index.bin`, and a reducer does one small
  ranged read of the index plus one generation-pinned ranged read per map object. Read cost is
  `maps x partitions` GETs (class B), write cost one object per map task.
- Every seal checks conservation (docs in == docs out across the stage) and deletes the
  previous stage's intermediates once its consumer has sealed.
- A failed listing raises; it is never read as "no objects".

## Stages

```
pass1      (unit)      kept -> normalize (reshape+datakit rules), xxh3_128 id, dupekit MinHash
                       (286/26/5/seed 42, no text cap); byte-identical docs collapse inside the unit
exact      (id part.)  global exact dedup; keeps the minimum doc key (earliest file, row) per id;
                       copy count per kept id
lshmap     (unit)      unique docs -> (bucket, doc) by bucket partition; reps carry copy counts
bucket     (bkt part.) non-singleton buckets -> (member, bucket minimum)
ccNNN{a,b,c} (node p.) connected components, FastSV-style (graph.py): a = apply candidates, write
                       labels, request labels; b = answer requests; c = emit min-combined
                       candidates (propagate, hook, shortcut). Done when an `a` stage changes
                       no label. Edges live at their member (<=26 hubs), never at the hub.
canon_a/b  (node p.)   canonical member per component = min of str(xxh3_128("source_000|"+id)),
                       exactly fuzzy_dups' choice; component sizes, oversized list
attrs      (node p.)   per-unit member table (doc, component, is_canonical, component size)
apply_cc   (unit)      re-read kept, write survivors -> {output}/deduped/
c075_text  (unit)      member text keyed by cluster (oversized >100k clusters split as upstream)
c075_solve (c075 p.)   cluster_dedup.find_duplicates with C075_PARAMS (port of solve_text_shard)
c075_marks (node p.)   removals by unit
apply_c075 (unit)      -> {output}/deduped_c075/
report                 {output}/stats/rdedup_stats.json
```

Parity with the Zephyr path (tested in `test_pipeline.py` against the real
normalize/minhash/fuzzy_dups code, and `test_solve.py` against `cluster_verify`): same
empty-text filter, same whitespace compaction, same ids, bit-identical MinHash buckets, same
cluster membership, same canonical, same c075 decisions, same output record (column order,
compacted text, `modernbert_prob` NaN -> null, msgspec JSON). Deliberate differences:

- Exact-duplicate groups keep the copy with the smallest (file, row) key instead of an
  arbitrary one, so survivors' provenance columns can differ for exact duplicates only.
- Components are fully converged. Upstream `connected_components` stops at 10 label-propagation
  iterations and does not converge on large chained components; see the 154k section.
- The 154k run used an earlier Large-Star/Small-Star CC; it was replaced by FastSV before the 2M
  run because Large-Star keys a component's whole member list at its root (a giant component
  would OOM one partition). Both are exact connected components; see the smoke check below.
- Oversized-cluster splits use exact component sizes instead of a stride-256 sample.
- Output files are `data-UUUUUU-CCC.jsonl.gz` (unit, chunk of `files_per_output` WARCs),
  unsorted by id.
- Every output record (both `deduped/` and `deduped_c075/`) ends with two integer fields the
  reference does not write:
  - `exact_copies`: input records whose normalized text has this record's content id (>= 1).
    pass1 counts copies collapsed inside a unit, `exact` sums them per id across units, and
    `lshmap` stores the total on each unique doc's reps entry. Seals check that the counts sum
    to `records_in` at both `exact` and `lshmap`.
  - `fuzzy_cluster_size`: unique (post-exact) docs in the record's connected component; 1 for a
    doc in no component. `canon_b` sends each component's size to every node partition holding
    its members, and `attrs` stores it per member. The `attrs` seal checks that the canonicals'
    sizes sum to the `canon_b` member count.

  Run roots sealed before these fields existed (the 154k and 2M runs) have the old `reps` and
  `attrs` layouts. Re-applying a rule or running `compare_reference.py` on them needs the code
  from before the change.

## Launch on N WARCs

Runs after the 2M use `--rules c075` only (the default). cc_canonical over-removes through
chained components and existed only for comparability with earlier curve points, which the 2M
run completes. Dropping it skips `apply_cc`, one full kept re-read, and the `deduped/` tree.
Pass `--rules cc_canonical c075` only to reproduce an old-rule tree.

Everything below from the repo root; secrets from `.env`.

```bash
set -a; source .env; set +a
# 0. (laptop) iris submit gate: BUILD_DATE in lib/iris/src/iris/_build_info.py must be today.
# 1. Freeze the plan. --listing takes a local parquet (path,size) to avoid a 2M-object LIST;
#    without it init lists kept/ itself (~2,000 list calls at 2M).
uv run python -m experiments.fast_curation.rdedup.launcher init \
  --run-root gs://marin-us-east1/tmp/ttl=14d/rdedup/<run> \
  --output-root gs://marin-us-east1/documents/baseline_lpv11_fastpipe_v2_1_fused_deduped/<N>warcs \
  --spec lpv11_fastpipe_v2_1_fused --bucket gs://marin-us-east1 \
  --expect-files <N> [--warc-hashes <manifest.parquet|txt>]   # --rules defaults to c075
# 2. Start the launcher (control plane, us-central2).
uv run iris --cluster marin job run --no-wait --region us-central2 --cpu 1 --memory 3GB \
  --enable-extra-resources --no-preemptible --priority interactive --job-name rdedup-launch-<tag> -- \
  python -m experiments.fast_curation.rdedup.launcher run --run-root <run root> \
  --region us-east1 --tag <tag> --max-workers <W> --chunk 16
# 3. Progress
uv run python -m experiments.fast_curation.rdedup.launcher status --run-root <run root>
# 4. Token count (in-region job)
python -m experiments.fast_curation.count_deduped_docs \
  --deduped-glob "<output root>/deduped/*.jsonl.gz" --output <output root>/stats/deduped_token_count.json
```

### The full 8M corpus

Once extraction finishes and kept/ holds every WARC (verify the count first):

1. Re-list kept/ into a local `(path,size)` parquet (or omit `--listing`).
2. `init` with a NEW run root (e.g. `rdedup/8m-v1`), `--output-root .../<N>warcs`,
   `--expect-files <N>`, and the default `--rules c075` (output: `deduped_c075/` only). The plan sizes itself from total bytes: ~4x the 2M partition counts.
3. Launch with `--max-workers 300` (or what the east1 pool gives; the launcher backs off).
4. Scratch at 8M is ~4x the 2M peak (see below); it sits in `tmp/ttl=14d/`, so a run must finish
   within 14 days of its intermediates being written, or be moved to a `ttl=30d` root.

### Knobs

- `--unit-gb` (8): compressed kept per pass1/apply task (~14 min per unit on 8 CPUs).
- `--node-partitions` (units/4): CC/canonical task count.
- `--files-per-output` (32): WARCs per output jsonl.gz.
- Worker shape: `--worker-cpus 8 --worker-memory-gb 32` (fits the v6e hosts east1 CPU work lands
  on; 64g+ workers place badly there). Stage memory estimates live in `stage_sequence`.
- `ClaimConfig`: heartbeat 60 s, stale 480 s, 6 attempts.

## Failure behavior

- Preempted/killed worker: its tasks go stale after 8 min and are retried elsewhere; outputs are
  byte-identical. Loss: at most the in-flight tasks' compute.
- Launcher death: nothing lost; relaunch the same command (a new `--tag` avoids name clashes).
- Code bug/OOM in a task: traceback in `stages/<stage>/failures/`; task retried up to 6 times,
  then left claimed (POISONED). Fix the code, delete that task's `claims/<task>.json`, relaunch.
  Changing code mid-run is safe only if task outputs stay identical for the same inputs.
- Poisoned or wrong seal: delete `stages/<stage>/sealed.json` (and the stage's `done/` if its
  outputs must be recomputed); intermediates of the previous stage may already be deleted, in
  which case restart from the earliest stage whose inputs still exist.

Skew lessons from the 2M run (fixed in code; relevant to anyone changing CC):

- Never key a component's member list at one node (Large-Star did; replaced by FastSV).
- Never let labels follow doc-key order: keys follow file order, so component minima sit in
  the first files and one partition receives every hooking candidate. Labels compare in a
  scrambled order (`graph.scramble`), and cc a/b stream their inputs block by block.
- A stage's memory estimate (`mem_gb`) sets how many tasks share a worker. Over-packing shows
  up as repeated attempts with no traceback (OOM kill) and ends POISONED after 6.

## Cleanup

Seals delete consumed intermediates as they go (stash/pass1_exact/exact after lshmap, lshmap after
bucket, each cc round after the next, ...). After the report: `data/reps`, `data/attrs`,
`data/c075_marks` remain (needed only to re-apply a rule); delete the whole run root when done:
`python -c "from experiments.fast_curation.rdedup.store import GcsStore, delete_prefix; print(delete_prefix(GcsStore(), '<run root>/'))"`.
Never delete anything under kept/. (The 2M kept_text/ projection, kept/ minus n_tokens, was verified redundant and deleted on 2026-10-13; rdedup reads kept/.)

## Measured

### 154,117-WARC reproduction (run root `tmp/ttl=14d/rdedup/v154k-a`)

| | reference tree | rdedup |
|---|---|---|
| records in | 1,164,614,123 | 1,164,614,125 |
| exact duplicates | 190,886,090 | 190,886,091 |
| unique docs | 973,728,033 | 973,728,034 |
| cluster members | 356,530,861 | 356,530,860 |
| clusters | 98,017,779 | 98,008,732 |
| cc_canonical survivors | 715,214,951 | 715,205,906 |

- +2 records: one WARC (`data-9e831b627baa.parquet`) was rewritten during the 09-23
  consolidation and now holds 8,129 rows; the subset manifest recorded 8,127 when the reference
  ran. Every other unit's row count matches the manifest exactly. That file accounts for the +1
  exact duplicate, +1 unique doc and -1 member.
- -9,047 clusters: our components are fully converged (12 Large/Small-Star rounds; violations
  34.1M -> 0). The reference stopped at 10 label-propagation iterations. Our largest
  component has 34.5M members; the reference's sampled largest is ~8.4M, and its next ones
  (4.9M, 3.7M, 1.9M) are pieces of the same chain.

Cluster-level check (`compare_reference.py`, in-region job, 30 min):

- Member sets differ only by that one WARC (2 of ours / 3 of theirs not shared).
- 0 reference clusters span two of our components: the reference is a strict refinement of the
  true components. 13,752 of our components (50.6M members) are split into several reference
  clusters, and 96,979 reference cluster labels contain no canonical member, the signature
  of connected components stopped before convergence.
- Applying the REFERENCE canonical flags to our unique docs: 715,214,954 docs /
  2,985,864,476,394 chars vs the reference tree's 715,214,951 / 2,985,864,424,584. The +3 docs
  are the rewritten WARC. The pipeline reproduces the reference exactly apart from that
  input and the convergence fix.
- With converged components, cc_canonical keeps 715,205,906 docs / 2,985,792,419,592 chars
  (-9,045 docs, -0.002% chars vs the reference).

FastSV vs Large-Star on the 300-WARC smoke: byte-identical outputs (28 files, same CRC32C),
89,983 members / 23,256 components; 5 rounds vs 8.

Idempotence: killing the worker job mid-pass1 on the 300-WARC smoke gave byte-identical outputs
and stats to the undisturbed run (two pass1 tasks were taken over at attempt 2).

c075 at 154k: removes 193,409,661 docs and keeps 780,318,373 docs / 4,088,822,794,561 chars.
The reference re-apply removed 195,088,446 and kept 4,083,614,819,769 chars (+0.13% chars
here). c075 compares documents only inside a cluster group, and the groups differ by
construction: converged components merge the reference's fragments (largest 34.5M members)
before the upstream oversized-cluster split (100k cap, MinHash split hash x 16 id buckets)
cuts them, and split counts use exact sizes rather than a stride-256 sample. Same rule, same
parameters, a different partition of candidates. c075 is not expected to match exactly.

154k wall time (48 workers x 8 CPU; 16 after pass1): pass1 ~1h45m, exact+lshmap+bucket 12 min,
CC (Large-Star, 13 rounds) + canonical 18 min, apply_cc 20 min, c075_text 17 min,
c075_solve 4h10m (including ~2.5h lost to the over-packed-memory bug, now fixed),
apply_c075 68 min on 16 workers.
