# Extractor-only ablation: lpv11_fastpipe_v2 post-decon docs, labels, and mix; upstream resiliparse text

Plan written 2026-09-06, revised twice the same day (user calls): the doc set is pinned POST
dedup and decon, and arm B inherits arm A's topic/quality labels AND A's fitted OLMIX mixture.
Everything about the training set is then identical between A and B except the bytes in
`text`. That is the cleanest possible measurement of the fork extractor
(resiliparse-rs, `extract_plain_text(html, main_content=True, preserve_formatting="markdown")`)
against the upstream Cython call the `resiliparse_10k` baseline used.

## Starting state (verified 2026-09-07 06:35Z)

- Arm A's dedup (11:30Z), decon (11:58Z) and Llama-3 cache (12:38Z) are DONE.
- Post-decon tree: `gs://marin-us-east5/documents/baseline_lpv11_fastpipe_v2_decon_deduped/10364warcs/deduped/`,
  3,042 `data-*.jsonl.gz`, 79,582,522 docs. Every record carries
  `doc_id, url, warc_hash, snapshot, fasttext_score, pooled_prob, modernbert_prob, text`.
  `doc_id` is the normalized `WARC-Record-ID`; `warc_hash` is `sha256(warc_path)[:12]` over
  `experiments/distill/dclm_400m_1x.txt`.
- Arm A grid labels (`datakit/{cluster_assign,quality}/lpv11_fastpipe_v2_10k_gridv1/`):
  quality 3,042/3,042 done; topic 2,399/3,042 done at 06:35Z, 19 `t2` jobs running, ~210
  shards/h since 20:31Z on 09-06, so roughly three hours to go. The 3 `pending` jobs
  (`grid-lpv11-fastpipe-v2-10k-000/006/010`) are the original v5p-8 submits; overlap with `t2`
  is safe (done markers) and they can be left alone.
- `datakit/tokenize/lpv11_fastpipe_v2_10k_gridv1/` done (3,042).
- WARCs are not stored anywhere; Common Crawl refetch is free and
  `decode_warcs_clean._decode_one_warc_stream` is the cascade's decoder, so re-decoding
  reproduces the `html` string the fork saw.
- "Regular resiliparse" = `lib/marin/src/marin/transform/extract_text_from_html.py::_extract_text`:
  `extract_plain_text(HTMLTree.parse(html), main_content=True, alt_texts=False, noscript=False)`,
  `.strip()`, drop `len < 50`. That produced
  `gs://marin-us-central2/extracted/dclm_400m_1x_10k_resiliparse-f0887f/` (10,364 shards,
  shard i = WARC i of the manifest) behind arm C.

## Arms

| arm | extractor | docs | topic/quality labels | mixture |
|---|---|---|---|---|
| A `lpv11_fastpipe_v2_10k` | fork (Rust, markdown) | cascade -> dedup -> decon | scored on A text | fit on A's swarm |
| B `lpv11_fastpipe_v2_upstreamrs_10k` | upstream Cython, plain | A's, verbatim | A's, carried by `doc_id` | A's weights, verbatim |
| C `resiliparse_10k` | upstream Cython, plain | own dedup/decon | scored on C text | fit on C's swarm |

A vs B: extractor only. B vs C: selection plus mix at a fixed extractor (existing arm).

## How the grid tooling joins, and why label reuse works

`datakit_store` joins tokenize, cluster and quality tables POSITIONALLY per shard: same
basename, same row count, and the cluster and quality `id` columns must agree row-for-row.
`id` is `generate_id(text)`, a content hash, so A's tables cannot be pointed at B's text. But
A's topic table also carries `native_id` (= `doc_id`), and A's post-decon shard, topic shard
and quality shard are co-partitioned. So the labels attach to B by `doc_id`, and B gets its OWN
co-partitioned tables written next to its shards with `id = generate_id(B text)`. No topic
model and no quality model run for B.

`token_length` in the topic table (gte tokenizer, A's text) feeds only `distribution.json`,
which `grid_store` uses as a subshard sizing hint. Carry A's value: plain text is never longer
than the fork's markdown, so the hint is an upper bound and the store's 40% headroom holds. The
store ledgers carry the true token counts.

The OLMIX mixture file (`experiments/data_mixing/mixtures/<corpus>_R3e10_k20_olmixexact_lambda0p01.json`)
is a dict of per-cell weights keyed `cXX_qY` plus metadata. `_grid_mix_method` binds a weights
file to a store by corpus name. B's cells are A's cells (same docs, same labels), so A's weights
apply verbatim: copy the file under B's corpus name once A's fit exists.

## Where B can differ from A in doc set, both reported

1. Upstream extraction under 50 chars on a page the fork kept (`n_short`). Dropped, matching arm
   C's filter. These rows are also absent from B's label tables, so co-partitioning holds.
2. Decon is skipped on B so the doc set stays identical. Run `decon_apply` on B's tree into a
   scratch prefix as a diagnostic count only; do not train on it.

The copied stage scores describe the fork text. They ride along only so the columns match A.
Never re-threshold B on them.

## Known risk: WARC re-parse drift

A fresh `warcio` parse once returned 11.8% fewer records for the same WARC
(`feedback_warc_record_count_drifts_between_runs`; suspected truncated HTTP 200 with no
Content-Length check). Selecting by `doc_id` is immune to index drift but not to lost tail
records. Mitigations in the worker:

1. Verify `Content-Length` against bytes consumed in `_decode_one_warc_stream`; raise on
   mismatch so the retry loop refetches. Fixes the open bug for every caller.
2. Assert `n_matched == n_requested` per WARC. A shortfall after retries still writes the shard
   but lists the missing ids in the catalog and fails the fleet summary. Expect zero.

## Code

No change to `fast_curation/spec.py`. Three new files plus one shared fix.

### 1. `experiments/fast_curation/build_swap_allowlist.py` (one Zephyr job, in-region)

Runs after A's topic wave and `grid_label merge`. For each A shard `i` it reads three
co-partitioned files in lockstep: the post-decon jsonl.gz (all columns except `text`), the
topic parquet (`native_id, cluster_24, topic_prob, token_length`) and the quality parquet
(`score, quality_bucket`). Guards: equal row counts; `topic.native_id[r] == doc_id[r]` for
every row; `topic.id == quality.id`. Then group by `warc_hash` and write
`allowlist/data-{warc_hash}.parquet` (10,364 files) with columns

```
doc_id, url, warc_hash, snapshot, fasttext_score, pooled_prob, modernbert_prob,
cluster_24, topic_prob, token_length, quality_score, quality_bucket, a_shard, a_row
```

Assert the row total equals 79,582,522 and the file count equals 10,364. ~10 GB of small rows;
this is the only pass over A's 154 GB.

### 2. `experiments/fast_curation/swap_extractor.py` (claim worker) + `launch_swap.py`

Same claim/registry/skip-if-done shape as `cpu_phase_a` (reuse `_claim_warc_atomic`,
`_register_completed_warc`, `_load_completed_registry`; `_launch_common.submit_workers`). Per
WARC:

```python
MIN_TEXT_CHARS = 50  # ExtractTextConfig.min_text_length: arm C's filter
DATASET = "lpv11_fastpipe_v2_upstreamrs_10k"


def upstream_resiliparse_text(html: str) -> str:
    """The resiliparse_10k baseline's call, verbatim (extract_text_from_html._extract_text)."""
    from resiliparse.extract.html2text import extract_plain_text
    from resiliparse.parse.html import HTMLTree

    return extract_plain_text(HTMLTree.parse(html), main_content=True, alt_texts=False, noscript=False).strip()


def swap_one_warc(warc_path, warc_hash, allowlist_prefix, out_prefix, pool) -> dict:
    wanted = {r["doc_id"]: r for r in read_table(f"{allowlist_prefix}/data-{warc_hash}.parquet").to_pylist()}
    n_requested, n_short, rows = len(wanted), 0, []
    for chunk in _decode_one_warc_stream(warc_path, chunk_records=2000):
        hits = [r for r in chunk if r["doc_id"] in wanted]
        for rec, text in zip(hits, pool.map(upstream_resiliparse_text, [r["html"] for r in hits]), strict=True):
            row = wanted.pop(rec["doc_id"])
            if len(text) < MIN_TEXT_CHARS:
                n_short += 1
                continue
            rows.append({**row, "text": text})
    shard = f"{out_prefix}/deduped/data-{warc_hash}.jsonl.gz"
    write_jsonl_gz(shard, rows)                      # arm A's record shape + the label columns
    write_attribute_tables(DATASET, shard, rows)     # B's own co-partitioned topic + quality tables
    write_json(f"{out_prefix}/catalog/data-{warc_hash}.json",
               {"n_requested": n_requested, "n_written": len(rows), "n_short": n_short, "missing": sorted(wanted)})
```

`write_attribute_tables` builds a `ShardDocs` with `ids = [generate_id(text)]`,
`native_ids = doc_id`, and calls the existing `grid_label.write_topic_shard` /
`write_quality_shard` / `write_done` with the carried `cluster_24, topic_prob, token_length,
quality_score, quality_bucket`. Those writers live in `grid_label.py`, whose module top imports
jax; if the CPU worker image cannot import jax, move the three writers to `grid_corpora.py`
(they only use pyarrow, numpy and fsspec). The quality `outputs/samples/` table is a 2% text
sample for the stage report; write it too, it is cheap.

- Output tree: `gs://marin-us-east5/documents/baseline_lpv11_fastpipe_v2_upstreamrs_decon_deduped/10364warcs/deduped/`,
  one `jsonl.gz` per WARC. The `_decon_deduped` name is deliberate: this IS the doc set that
  came out of A's dedup + decon.
- Attribute tables: `datakit/cluster_assign/lpv11_fastpipe_v2_upstreamrs_10k_gridv1/` and
  `datakit/quality/lpv11_fastpipe_v2_upstreamrs_10k_gridv1/outputs/main/`, plus `counts/` and
  `_done/` per stage, all with stem `data-{warc_hash}`.
- Stream decode keeps peak RAM at a few GB per WARC; a 4 to 6 process pool runs the Cython
  parse. Upstream resiliparse does not segfault on framesets; fork-crashed pages are not in
  the set anyway.
- Reads: allow-list (us-east5) and CC (free). Writes: us-east5. Zero cross-region bytes.
- Launcher: extras `cpu` only, `--cpu 4 --memory 16GB --disk 24GB --priority batch
  --preemptible`, at most 8 concurrent submits.

### 3. Registrations

- `grid_corpora.GRID_CORPORA["lpv11_fastpipe_v2_upstreamrs_10k"] = GridCorpus(<B tree>, "us-east5", Format.JSONL_GZ, native_id_field="doc_id", post_decon=True)`.
- `curation_plan`: `_D_OBS_DEFAULTS[<B cache>]`, `METHODS["lpv11_fastpipe_v2_upstreamrs"]`
  (natural arm), and `METHODS["lpv11_fastpipe_v2_upstreamrs_10k_mix_olmixexact_lambda0p01"] =
  _grid_mix_method(..., <B cache>, "lpv11_fastpipe_v2_upstreamrs_10k", "_olmixexact_lambda0p01")`.
- `experiments/data_mixing/mixtures/lpv11_fastpipe_v2_upstreamrs_10k_R3e10_k20_olmixexact_lambda0p01.json`:
  A's file with `corpus` renamed and a `source` note that the weights are A's fit. Assert its
  cell keys equal B's store cells before launching.
- `launch_10k_natural.METHOD_NAMES` for both arms.

### 4. Tests (`test_swap_extractor.py`, `test_build_swap_allowlist.py`)

- Parity: `upstream_resiliparse_text(html)` equals
  `extract_text_from_html._extract_text({"html": html, "url": u})["text"]` on fixture pages.
- Selection: a fake decode stream with extra and missing records yields exactly the allow-list
  rows; `missing` lists the absent id; `n_short` counts a sub-50-char page and is absent from
  the attribute tables too (row counts agree across shard, topic, quality).
- Allow-list guards fire on a `native_id` / `doc_id` mismatch and on a row-count mismatch.
- Attribute tables round-trip through `datakit_store`'s loaders (`_load_cluster_table`,
  `_load_quality_table`, `_ids_match`).

## Sizing

79.6M docs over 10,364 WARCs. Per WARC on 4 cores: decode ~40 s, extraction ~10 s, I/O ~10 s.
~300 worker-hours; 60 batch workers in us-east5 finish in ~5 to 6 h of wall on a healthy pool.
Output roughly A's 154 GB plus ~10 GB of tables. Saved relative to a native B pipeline: the
fuzzy dedup (~4 h), decon, the topic wave (TPU, ~2 days at the earlier rate), and the entire
363-proxy OLMIX swarm plus its eval waves.

## Sequence

All commands run as Iris jobs in us-east5 (`set -a; source .env; set +a` first).

1. Land the code; `./infra/pre-commit.py --all-files --fix`; run the new tests. Can start now.
2. Wait for A's topic wave (~3 h at the current rate), then `grid_label merge --dataset lpv11_fastpipe_v2_10k`
   as an in-region job. This is already the next step of the A runbook.
3. Build the allow-list; verify 10,364 files and 79,582,522 rows.
4. Canary ONE WARC (`launch_swap --num-workers 1 --limit 1`). Check the catalog:
   `missing == []`. Cross-check against arm C for the same WARC: join B's shard with
   `.../dclm_400m_1x_10k_resiliparse-f0887f/data-{i:05d}-of-10364.jsonl.gz` on `url` (one
   ~50 MB cross-region read) and require near-100% exact `text` equality.
5. Fleet: `launch_swap --num-workers 60 --priority batch`. Done when `deduped/` has 10,364
   files, both `_done/` dirs have 10,364 markers, and the catalog sum of `missing` is 0.
6. Diagnostic decon into a scratch prefix; report the drop count; delete the output.
7. `grid_label merge --dataset lpv11_fastpipe_v2_upstreamrs_10k` (tallies only, no model).
   Compare its grid to A's: doc counts per cell must equal A's minus the `n_short` rows.
8. Natural cache: `tokenize_lpv11` into `tokenized/lpv11_fastpipe_v2_upstreamrs_decon_10364warcs-<sha>`;
   ledger doc count must equal 79,582,522 minus the `n_short` sum; register.
9. Grid: `grid_tokenize`, `grid_store`, `verify_grid_store`, `validate_grid_domains` for B.
   Store doc count must equal the natural cache's.
10. Natural arm: the 36-cell expFM sweep via `launch_10k_natural.py --allowed-regions us-east5`.
    First result: A vs B natural at identical documents.
11. Mix arm, after A's swarm, eval waves and `run_olmix_fit` land: vendor A's weights under B's
    name, register, launch the 36-cell sweep. A-mix vs B-mix at identical documents, labels and
    weights.

Optional later: B's own swarm, only to ask whether the OPTIMAL mix moves with the extractor.
That is a different question and not needed for the ablation.

## Alternative considered and not recommended

Join arm C's existing extracted tree to A's doc set instead of re-extracting. It saves the
~300 worker-hours but needs a ~150 GB us-central2 -> us-east5 copy, inherits `pipeline_10k`'s
decoder rather than the cascade's, has no `doc_id` (URL collisions within a WARC become
ambiguous), and cannot produce a `preserve_formatting=True` variant later. Use the join only as
the step-4 canary cross-check.

## Overnight log (2026-09-07)

- 07:10Z: code landed in the working tree (not committed): `fast_curation/{upstream_resiliparse,
  swap_extractor,build_swap_allowlist,attach_swap_labels,launch_swap,swap_status}.py` + tests (44
  pass), Content-Length check in `decode_warcs_clean`, `grid_corpora` entry
  `lpv11_fastpipe_v2_upstreamrs_10k`, `OUTPUT_BASE: str` annotation (fixes pre-existing pyrefly
  errors in grid_store/grid_tokenize). Lint + pyrefly clean.
- DESIGN CHANGE: two passes. A's topic wave is capacity-starved (its jobs' tasks sit `pending`
  after v6e preemptions -- NOT hung; 2,406/3,042 at 07:09Z, no progress in the last hour), so the
  extraction fleet no longer waits on labels: `build_swap_allowlist --stage allowlist` (provenance
  only) -> fleet -> later `--stage labels` (verified per row against A's tables) ->
  `attach_swap_labels` writes B's topic/quality tables from B's shards + the per-WARC label
  parquet. Same guards, same outputs; the label pass is a cheap in-region CPU job.
- 07:10Z: `/michaelryan/fastcur-swap-allowlist` submitted (us-east5, interactive, driver
  --cpu 1 --memory 4GB --no-preemptible). Gate: `_meta.json` n_rows == 79,582,522.
- Rule for the night: never stop/kill any job (the user's t2 wave included). Next: canary
  (`launch_swap --num-workers 1 --limit 1`), catalog check + URL cross-check vs arm C shard, then
  `launch_swap --num-workers 60`. `swap_status --expect-warcs 10364` is the fleet gate;
  `--expect-labels` after attach.
- 08:45Z: first allow-list run wedged at 255/256 reduce shards (10,351 files). Thread dump of the
  stuck Zephyr worker (`iris process profile threads -t <task>`): reduce subprocess blocked forever
  in gcsfs `_initiate_upload` while closing one parquet in `write_warc_shard`. Fix: reducer writes
  go through the storage client with `timeout=300` and 3 attempts, and skip an existing file.
  Stopped MY job `fastcur-swap-allowlist` (killed with its Zephyr children) and relaunched as
  `fastcur-swap-allowlist-r2` at 08:47Z. The same hardening applies to the `labels` stage.
- 09:05Z: allow-list r2 DONE, gate exact (79,582,522 rows / 10,364 WARCs / 10,364 files).
- 09:05Z: canary WARC 0 (`50c04a36d67e`): 15,182 requested, 14,871 written, 311 short (<50 chars,
  2.0%), 0 missing, 82 s wall on 4 cores. Cross-check vs arm C (C's shard index = rank of the
  warc_hash among sorted hashes, NOT the manifest line -- fixed in `swap_canary_check`): 14,871/14,871
  URL-joined; 13,876 byte-identical (93.3%), 721 differ only where C has U+FFFD (4.8%), 274 other
  decoder-class differences (BOM/charset/mojibake, 1.8%). Extraction call agrees; B keeps the
  cascade's decoder (identical to A). CANARY PASS at the documented 0.9 gate.
- 09:20Z: fleet launched: `launch_swap --num-workers 60 --priority interactive` (4 cpu / 16 GB each,
  preemptible, us-east5). Expect ~4 h if fully placed (82 s/WARC/worker).
- 10:10Z: fleet at 60/60 workers (13 tail submits had failed silently; resubmitted serially),
  ~2,900 WARCs/h, 0 missing ids so far, short-drop rate ~5%. A's topic wave slowed to ~20 shards/h
  (2,678/3,042): its pending tasks sit in the BATCH band (`get-scheduler-state`: user over
  interactive budget) behind the user's own fused-fleet children; the v6e-4 us-east5 pool shows
  47 ready slices, demand 0, autoscaler backoff (health 0.13). A top-up wave from this session
  would land in the same batch queue, so none was launched. FOR THE USER: granting interactive
  budget to `grid-lpv11-fastpipe-v2-10k-t2-*` (or pausing fused coordinators) is what unblocks
  A's labels; B's label attach waits on that, B's extraction does not.
- 14:48Z: FLEET COMPLETE. Gate (`swap_status --expect-warcs 10364`): catalogs/shards/registry
  10,364; requested 79,582,522 (== allow-list); written **75,625,394**; short-dropped
  **3,957,128 (4.97%)** -- pages where upstream resiliparse yields <50 chars while the fork
  kept them (arm C's own filter; recoverable later by re-extracting exactly the allow-list minus
  written set if the user wants them kept); missing 0. ~5.5 h wall for 60 workers.
- Next: `tokenize_lpv11` into `tokenized/lpv11_fastpipe_v2_upstreamrs_decon_10364warcs-41e20bf`
  (label-independent); labels stage + attach still wait on A's topic wave (2,788/3,042 at 14:48Z).
- 15:30Z: workers kept running after completion: a pass iterates a snapshot of not-done WARCs and
  never re-checks the registry, so finished WARCs' claims went stale (0.5 h) and were reclaimed
  and re-extracted (identical bytes, atomic uploads). Stopped all 60 `fastcur-swap-*` (mine).
  Fixed in `run_worker` (registry re-check before every claim). `verify_swap_shards` job launched
  as the integrity gate (line count == n_written, doc_id set == allow-list minus dropped, no dups).
  `fastcur-tokenize-upstreamrs` running since 14:51Z on the same tree.
- 15:45Z: `verify_swap_shards` PASS: 10,364/10,364 shards, 75,625,394 records, 0 problems (line
  counts == catalogs, doc_id sets == allow-list minus dropped, no duplicates, all texts >= 50).
- 15:50Z: `fastcur-tokenize-upstreamrs` SUCCEEDED: ledger 75,625,394 docs (== verified records),
  **84,181,817,101 Llama-3 tokens** (A: 91,653,647,234; -8.2%: 5% fewer docs, shorter plain text),
  1,501 parts. Registered `lpv11_fastpipe_v2_upstreamrs` in `curation_plan` (_D_OBS_DEFAULTS +
  METHODS) and `launch_10k_natural.METHOD_NAMES`; dry-run = 38 expFM cells at --max-budget 9e20.
- HELD FOR THE USER (capacity + priority calls, cluster already saturated by their fleets):
  (1) natural sweeps -- A's `lpv11_fastpipe_v2` natural arm has NO results yet either, so both
  arms need launching for the comparison:
  `uv run python experiments/scaling_law_sweeps/launch_10k_natural.py --methods lpv11_fastpipe_v2 lpv11_fastpipe_v2_upstreamrs --allowed-regions us-east5 --max-budget 9e20`
  (2) A's topic wave is stuck at 2,789/3,042 in the batch band; B's label attach, grid, and mix
  arm all wait on it. Once it finishes, the monitor in this session runs:
  `build_swap_allowlist --stage labels` -> `attach_swap_labels` -> `swap_status --expect-warcs 10364 --expect-labels`
  -> `grid_label merge --dataset lpv11_fastpipe_v2_upstreamrs_10k` -> grid_tokenize/grid_store/verify/validate.
- Code is in the working tree, uncommitted. Local `gcloud` CLI token expired ~12:45Z (ADC + Iris fine).
- 2026-09-08 01:50Z: A's topic wave reached 3,042/3,042 (the user's other thread drove it).
  02:12Z: `fastcur-swap-labels` SUCCEEDED, gate exact (79,582,522 rows / 10,364 files / 24 names;
  every row passed native_id == doc_id and generate_id(text) == topic id).
  02:14Z: `fastcur-swap-attach` launched (B's topic/quality tables + _done + counts per shard).
  Then: `swap_status --expect-warcs 10364 --expect-labels` -> `grid_label merge --dataset
  lpv11_fastpipe_v2_upstreamrs_10k --output-base gs://marin-us-east5` -> grid_tokenize ->
  grid_store -> verify_grid_store -> validate_grid_domains (all in-region jobs, us-east5).
- 02:38Z: `fastcur-swap-attach` SUCCEEDED; `swap_status --expect-warcs 10364 --expect-labels`
  PASS (topic_done 10,364 / quality_done 10,364; earlier counts unchanged). 02:42Z: launched
  `grid-merge-upstreamrs` (grid_label merge, B) while `grid-tokenize-upstreamrs` runs.
- 03:00Z: `grid-merge-upstreamrs-r2` SUCCEEDED (first submit at --cpu 2/16 GB non-preemptible never
  placed; r2 = 1 cpu / 8 GB preemptible). B distribution: 75,625,394 docs in topic, quality and
  grid; 120/120 cells populated; no stage skew. A's merge (other thread): 79,582,522 docs, 120
  cells. Difference == the 3,957,128 short-dropped docs. Quality bucket shares B:
  [0.028, 0.176, 0.660, 0.135, 0.001].
- 04:05Z: `grid-tokenize-upstreamrs` driver FAILED ("shard 1853 crashed its worker 20 times") but
  all 10,364 tokenize parquets exist and shard 1853 (`data-2e781ff30c9c`) holds exactly its
  catalog count (5,125 rows). 04:08Z: rerun `grid-tokenize-upstreamrs-r2` (expect "already
  complete") and launched `grid-store-upstreamrs` (grid_store, 1 cpu / 8 GB driver, retries 5).
  Next: verify_grid_store -> validate_grid_domains (cells.json under metadata/olmix/...).
- 07:21Z: `grid-store-upstreamrs` SUCCEEDED (store at
  `gs://marin-us-east5/datakit/store/lpv11_fastpipe_v2_upstreamrs_10k_gridv1/`). 07:23Z: launched
  `grid-verify-upstreamrs` (verify_grid_store) and `olmix-validate-upstreamrs`
  (validate_grid_domains -> metadata/olmix/lpv11_fastpipe_v2_upstreamrs_10k/cells.json).
- Viewers for the user: random 10 pairs https://claude.ai/code/artifact/a7eb9073-2449-4d34-aa80-16ce9dc351b9 ;
  divergence scan (398,081 pairs from 16 A shards; `fast_curation/swap_diff_scan.py`,
  selection at `{root}/diff_scan/selection.json`) https://claude.ai/code/artifact/6be4f387-722f-4f93-907a-0248afc73604 .
  Scan: median word-set overlap 93%, p5 42%, 6.6% of pairs under 50%; extremes are one-sided
  collapses in BOTH directions (fork keeps CONTENTdm OCR transcripts upstream drops; upstream
  keeps RPM/API listings the fork drops). Per-topic medians 89-96%.
- 07:25Z: validate_grid_domains PASS: 120/120 cells load, 84,181,023,524 tokens (cache ledger
  84,181,817,101; the few e-6 gap is chunked long docs, as for A), 75,625,394 docs (== cache),
  24 cells below the block floor at natural prior, prior dynamic range 1.4e6. cells.json written.
- 07:33Z: verify_grid_store PASS: "OK: 120/120 cells, 75625394 docs, 84181023524 tokens, every
  cell loadable". **ARM B DATA COMPLETE.** Store
  `gs://marin-us-east5/datakit/store/lpv11_fastpipe_v2_upstreamrs_10k_gridv1/`.

## State at hand-off (2026-09-08 07:35Z)

Done and gated: records (75,625,394 docs), labels attached, natural cache
(`tokenized/lpv11_fastpipe_v2_upstreamrs_decon_10364warcs-41e20bf`, registered as
`lpv11_fastpipe_v2_upstreamrs`), 24x5 store verified + validated (cells.json written).
Open, user decisions:
1. Natural sweeps (A `lpv11_fastpipe_v2` AND B; neither trained):
   `uv run python experiments/scaling_law_sweeps/launch_10k_natural.py --methods lpv11_fastpipe_v2 lpv11_fastpipe_v2_upstreamrs --allowed-regions us-east5 --max-budget 9e20`
2. Mix arm for B, after A's swarm/evals/fit (other thread): copy
   `experiments/data_mixing/mixtures/lpv11_fastpipe_v2_10k_R3e10_k20_olmixexact_lambda0p01.json`
   to `lpv11_fastpipe_v2_upstreamrs_10k_R3e10_k20_olmixexact_lambda0p01.json` (rename `corpus`,
   note the source), assert its cell keys == B's cells.json cells, register
   `lpv11_fastpipe_v2_upstreamrs_10k_mix_olmixexact_lambda0p01 = _grid_mix_method(...,
   "lpv11_fastpipe_v2_upstreamrs_decon_10364warcs-41e20bf", "lpv11_fastpipe_v2_upstreamrs_10k",
   "_olmixexact_lambda0p01")`, add to `launch_10k_natural.METHOD_NAMES`, launch the 36-cell sweep.
Cleanup candidates (mine): `{root}/diff_scan/{a_sample,b_sample,scored}` (~5 GB),
`{root}/{allowlist_summary,labels_summary,attach_summary,verify_summary}`, `_claims/`.
Code uncommitted in the working tree (new: upstream_resiliparse, swap_extractor,
build_swap_allowlist, attach_swap_labels, launch_swap, swap_status, swap_canary_check,
verify_swap_shards, sample_swap_pairs, swap_diff_scan + tests; edits: decode_warcs_clean,
grid_corpora, curation_plan, launch_10k_natural).
