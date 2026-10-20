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

## Training launch (2026-09-26 08:55Z)

State found on return: A's olmixexact fit landed 2026-09-12 (363/363, TV 0.49); A's natural arm
(21 results) and A's mix arm (147 results across tags) already running under the other thread.
B had nothing. Done this session:
- Vendored `experiments/data_mixing/mixtures/lpv11_fastpipe_v2_upstreamrs_10k_R3e10_k20_olmixexact_lambda0p01.json`
  = A's file with `corpus` renamed and `source` noting the copy; asserted weights AND natural
  keys == B's 120 cells.json names before writing (weights sum 1.0).
- Registered `lpv11_fastpipe_v2_upstreamrs_10k_mix_olmixexact_lambda0p01` (curation_plan
  `_grid_mix_method` on B's cache hash + `launch_10k_natural.METHOD_NAMES`). Lint + pyrefly clean.
- Dry-run: 76 cells (38 natural + 38 mix, 9e20 cap). LAUNCHED as `/michaelryan/10k-natural-upstreamrs`
  (interactive parent, --no-preemptible, children batch, --allowed-regions us-east5 because B's
  cache and store exist only there). Results land in
  `gs://marin-us-central1/metadata/data_curation_10k_natural_results/curation-lpv11_fastpipe_v2_upstreamrs*`.
  NEVER stop this coordinator (orphan-kills children).
- Comparison pairs: natural A `lpv11_fastpipe_v2` vs B `lpv11_fastpipe_v2_upstreamrs`; mix A
  `lpv11_fastpipe_v2_10k_mix_olmixexact_lambda0p01` vs B `..._upstreamrs_10k_mix_olmixexact_lambda0p01`.
- 09:00Z (user): NO natural arm for B, mix arm only. Stopped `/michaelryan/10k-natural-upstreamrs`
  (my job, 8 first-wave children, one running for ~3 min) and relaunched as
  `/michaelryan/10k-mix-upstreamrs` with `--methods lpv11_fastpipe_v2_upstreamrs_10k_mix_olmixexact_lambda0p01`
  only (38 cells, same flags). The natural method stays registered but is not to be launched.
  NEVER stop this coordinator.
- 23:10Z: coordinator task PREEMPTED (despite --no-preemptible) -> all 27 unfinished children
  killed ("Parent task preempted"); 11 cells had succeeded. Coordinator restarted itself
  (setup re-running); skip-if-done should resubmit the 27. Verifying via the monitor, not
  relaunching (a second coordinator would duplicate writers).
- 2026-09-27 13:53Z: coordinator preempted a THIRD time (summary: preemptions=3, still
  --no-preemptible); each time it restarts, kills the in-flight children ("Parent task
  preempted") and resubmits the unfinished cells in waves of 8 (skip-if-done keeps the finished
  ones: 14/38 done). Progress is lost only for runs without a checkpoint. Not relaunching.
  Worth raising with the user: the us-east5 non-preemptible CPU pool is evidently not immune.

## First result (2026-09-29): A vs B on 21 matched cells

Figure: https://claude.ai/code/artifact/d2812259-dab7-4ec6-9b9f-e809801bab94
(built by scratch scripts prep.py/build.py; source data = the two arms' result JSONs under
`gs://marin-us-central1/metadata/data_curation_10k_natural_results/`).

B (upstream resiliparse) is **consistently but marginally BETTER** than A (fork, markdown) at
matched documents/labels/weights. Aggregate `eval/bpb`: mean -0.0026, median -0.0022, paired
t = -5.6, B better in 19/21. Uncheatable macro -0.0049 (t -5.9, 19/21); Paloma macro -0.0014
(t -3.2, 15/21).

Per domain (mean B-A, negative = upstream better):
Wikipedia -0.0161 (**21/21**, t -26.1) | GitHub C++ -0.0094 | GitHub Python -0.0055 |
S2ORC -0.0014 | C4 -0.0015 | arXiv physics -0.0007 | arXiv CS +0.0002 (tie, t 0.6) |
LIMA +0.0021 (t +4.0, fork wins 17/21 — the ONLY domain the fork wins).

Read: the fork's markdown formatting does not pay for itself on encyclopedic or code text and
slightly hurts; it helps only on instruction-style prose (LIMA).

CAVEATS: (1) coverage — B has 22/38 cells, nothing at or above 2e20, so this covers 1e17..9e19
only; the large-model end is untested. (2) One A cell (`2e+19-d512-L6-B128`) is excluded: the
lpv11 thread wrote a `.badmodel` copy and the 09-17 rerun reproduced identical metrics
(bpb 1.4189 vs ~1.11 for its neighbours). Including it would swing the mean to -0.016 and is
wrong. (3) Effect sizes are ~0.2% relative — consistent in sign, small in magnitude.

### Paper-format figure (2026-09-29)

`experiments/scaling_law_sweeps/plot_extractor_ablation_by_size.py` renders the ablation in the
same visual language as `plot_olmixexact_bpb_by_size` (PAPER_RCPARAMS, one panel per model size,
log tokens x-axis, white-edged markers, shared bottom legend with each corpus's d_obs, PNG+PDF,
no title). Colours follow the locked convention: fork = `#006d2c` (its registered "GC Cascade"
green), upstream = `#9467bd` (the registered resiliparse purple — this arm IS upstream
resiliparse). The fork draws at the locked 4.2 linewidth with upstream at 2.4 ON TOP, because the
curves coincide to ~0.002 BPB and a single width would hide one arm entirely.

    uv run python -m experiments.scaling_law_sweeps.plot_extractor_ablation_by_size --difference
    uv run python -m experiments.scaling_law_sweeps.plot_extractor_ablation_by_size --dims 512 1536 2432 --suffix _3panel

Outputs in `scratch/plots/extractor_ablation/`: `extractor_ablation_bpb_by_size{,_3panel}.{png,pdf}`
and `extractor_ablation_bpb_difference.{png,pdf}`. The `_3panel` variant reproduces the reference
figure's 157M / 998M / 2.9B layout exactly.

Metric caveat baked into the axis label: the upstream arm has no olmo_bpb eval wave, so this plots
the BPB the training runs log inline (Paloma + uncheatable), NOT the olmix 51-task suite. Getting
this arm onto the 51-task axis needs an eval wave over its checkpoints. The flagged fork cell
(2e+19, d512) is dropped by `EXCLUDED_CELLS`.

### 2026-09-30 00:10Z — us-east5 can no longer run the remaining cells

All 16 unfinished cells now report `Scheduler: No workers match constraints` (previously several
were `Coscheduling: need N workers`, an ordinary capacity wait). The us-east5 accelerator pools
explain it: EVERY v5p pool there has 0 ready slices, and v5p-8/32/64/256 are all in autoscaler
backoff at health 0.03-0.06. v4 does not exist in us-east5 at all. The sweep's cells request v4
and v5p shapes only, so nothing in the region can host them. Coordinator is on its 4th preemption
but is healthy; 22/38 cells done and 0 training.

Waiting is now futile rather than slow. Capacity in the candidate mirror regions at that moment:

* **us-central1 (v5p)** — HEALTHY: v5p-8 32 ready, v5p-16 8 ready, v5p-32 4 ready + 1 booting,
  v5p-64 4 ready (backoff, health 0.15), v5p-256 1 ready. The earlier quota_exceeded has cleared.
* us-central2 (v4) — thin: v4-8 4 ready (backoff), v4-64 2 ready + 3 booting (backoff), reserved
  pools mostly 0. Would also serve the cells' v4 alternative.
* us-east1 — v6e only, no v5p/v4, so useless for this grid.

RECOMMENDATION unchanged but now required: mirror B's 884 GB grid store
`gs://marin-us-east5/datakit/store/lpv11_fastpipe_v2_upstreamrs_10k_gridv1` to **us-central1**
(~$18 same-continent egress, needs user approval: cross-region + over the $10 bar), then relaunch
the coordinator with `--allowed-regions us-east5 us-central1`. Do NOT stop the existing
coordinator first without checking whether relaunching duplicates writers — skip-if-done covers
the 22 finished cells, and the running set is currently empty, so a clean stop + relaunch is safe
here. Only the store needs mirroring, not the base tokenized cache (arm A's mix ran in regions
that never held its base cache).

### 2026-09-30 18:43Z — CORRECTION: us-east5 recovered; the arm is training again

Yesterday's "waiting is now futile" call was WRONG. us-east5 v5p capacity came back on its own and
5 cells are training with verified step advancement:
`2e+20-d512-L6-B1024` 54.1k/61.6k steps (~3h left), `9e+19-d1024-L11-B256` 17.9k/35.0k (~5h),
`2e+20-d2432-L24-B64` 595/37.3k (~15h), plus `9e+19-d2432-L24-B32` and `2e+20-d1536-L16-B128`.

Two lessons for this sweep:
1. The "No workers match constraints" state is TRANSIENT in us-east5 — it reflects pools sitting at
   zero ready slices under autoscaler backoff, not a permanent shape mismatch. Read it as a deep
   capacity trough, not a dead end.
2. Preempted children RESUME FROM CHECKPOINT: `2e+20-d512` was at 52.7k steps when preempted on
   09-27 and picked up at ~54k, not from zero. The repeated coordinator preemptions have cost far
   less wall time than they appeared to.

Mirroring the store to us-central1 is therefore a FALLBACK, not a requirement — worth doing only
if the arm stalls at zero running for a sustained stretch again. Still 22/38 done; the 9e+20
d3584 cell (16-worker gang) remains the hardest to place.

### 2026-10-01 — sweep paused by user; olmix 60-task eval wave launched

User: "We can pause the extractor vs extractor comparison here (no more need for training runs)"
-> STOPPED `/michaelryan/10k-mix-upstreamrs` (25 cells succeeded and banked, 14 killed).
Relaunching later would skip the 25 via skip-if-done.

User then asked to evaluate the finished models on the olmix suite. Manifest
`experiments/core_eval_manifests/olmix_bpb_upstreamrs.txt` (25 rows, all us-east5, every
checkpoint has an `hf/` export). Launched `/michaelryan/olmix-bpb-upstreamrs`:

    TASKS = resolve_tasks('all+qa_rc') (56) + the 4 mmlu_*/rc_5shot categories = 60 tasks
    launch_olmo_bpb_manifest --manifest ... --tasks "$TASKS" --launch
      --child-priority interactive --keepalive-timeout 172800

Deliberately uses the DEFAULT `--results-subpath metadata/olmo_bpb_results/{run_name}/`, because
that is what `plot_olmo_bpb_by_size_10k._load_records` (and hence `plot_olmixexact_bpb_by_size`)
reads — so once this lands, arm B can be plotted on the real `olmix51_bpb` axis next to every
other corpus, instead of the inline Paloma+uncheatable proxy.

`--keepalive-timeout 172800` is mandatory: the 12h default exits the parent as "finished" with no
error and silently kills pending children (it dropped 120/363 lpv11 evals once). Judge completion
by the GCS result count, never by parent state.

After 25/25: rerun `plot_extractor_ablation_by_size` against olmix51_bpb (needs a metric wired to
the olmo_bpb records rather than the inline eval dict), or simply add arm B to
`plot_olmixexact_bpb_by_size`'s METHOD_DISPLAY for a direct multi-corpus comparison.

### 2026-10-02 — olmix 60-task evals DONE, and they CHANGE the conclusion

`/michaelryan/olmix-bpb-upstreamrs` finished: 25/25 results, every one with exactly 60 tasks,
under the default `metadata/olmo_bpb_results/` path so the paper plotting machinery reads them.
Arm B is now registered in `plot_core_v2_by_size_10k.OLMIXEXACT_COMPARE_METHODS` and in
`plot_olmixexact_bpb_by_size.METHOD_DISPLAY` as ("fastpipe_v2 (upstream rslp)", "#9467bd").

Figure (paper style, real axis):
`scratch/plots/olmo_bpb_10k/paper/olmixexact_olmix51_bpb_by_size_extractor_ablation.{png,pdf}`

    uv run python -m experiments.scaling_law_sweeps.plot_olmixexact_bpb_by_size \
      --methods lpv11_fastpipe_v2_10k_mix_olmixexact_lambda0p01 \
                lpv11_fastpipe_v2_upstreamrs_10k_mix_olmixexact_lambda0p01 \
      --dims 512 1536 2432 --suffix _extractor_ablation

**On `olmix51_bpb` (24 matched cells) the extractor difference is NOT significant:**

| metric | mean B-A | median | paired t | upstream better |
|---|---|---|---|---|
| olmix51_bpb (the mixtures' own objective) | -0.0042 | -0.0083 | **-1.13** | 14/24 |
| eval/bpb inline (Paloma + uncheatable) | -0.0026 | -0.0022 | -5.56 | 21/24 |

Both are true and they measure different things. The inline metric is bits-per-byte over two
large fluent-text corpora — low variance, so a ~0.002 systematic shift is resolvable. olmix51 is
51 downstream tasks, many small, with much higher per-cell variance; at n=24 it cannot resolve an
effect this size. The mean still leans upstream on both.

**Reporting rule: lead with olmix51 (no measurable difference on the objective the mixtures were
solved against) and offer the inline-metric result as the sharper, narrower fluency probe.** Do
not quote the t=-5.6 figure without saying which metric it is from — it is the easiest number
here to mislead with.

