# lpv11 documents, resiliparse-fork text: the selection-vs-extraction ablation

Plan written 2026-10-02. Goal: separate what the LLM pipeline's **selection** buys from what its
**extraction** buys, by taking the lpv11 (`llm_pipeline_v1_1`) corpus's surviving documents and
regenerating only their `text` with the resiliparse-rs fork.

## Why this arm completes the design

|                   | fork extract (resiliparse-rs, markdown) | LLM extract (llm_pipeline_v1_1) |
|-------------------|------------------------------------------|----------------------------------|
| **cascade select** (fastpipe_v2) | **A** `lpv11_fastpipe_v2_10k` — trained | — |
| **LLM select** (lpv11)           | **NEW** (this plan)                      | **L** `lpv11_10k` — trained      |

* **NEW vs L** isolates LLM *extraction* at fixed LLM selection.
* **NEW vs A** isolates LLM *selection* at fixed fork extraction.

Both comparison arms already exist as trained olmixexact mix arms, so the new arm is the only
thing that has to be built. (Arm **B**, `lpv11_fastpipe_v2_upstreamrs_10k`, is the third swap in
the family: cascade select + upstream resiliparse — see `extractor_swap_upstream_resiliparse.md`.)

## Starting state (verified 2026-10-01/02, read-only)

| fact | value |
|---|---|
| Source tree | `gs://marin-us-east5/documents/baseline_llm_pipeline_v1_1_deduped/10364warcs/deduped` |
| Shards | 3,193 `data-NNNNN-of-03193.jsonl.gz` — **index-named, NOT per-WARC** |
| Documents | 80,728,641 |
| Grid tokens | 111,324,334,812 (marin-tokenizer) |
| Record columns | `url, warc_record_id, warc_file, snapshot, pipeline_id, num_chunks, text` |
| Topic labels | `datakit/cluster_assign/lpv11_10k_gridv1/` — 3,193 parquet, **complete in us-east5 only** |
| Quality labels | `datakit/quality/lpv11_10k_gridv1/outputs/main/` — 3,193 parquet, complete |
| Grid store | `datakit/store/lpv11_10k_gridv1/` — 120 cells, in us-east5 + us-central1 + us-east1 |
| Mixture | `experiments/data_mixing/mixtures/lpv11_10k_R3e10_k20_olmixexact_lambda0p01.json` exists |
| WARC pool | the same 10,364-WARC `experiments/distill/dclm_400m_1x.txt` fastpipe_v2 uses (byte-identical to `dclm_400m_1x_warcs.txt`) |

**Provenance is complete — no recovery job is needed.** The two fields the fastpipe swap keys on
are present under different names and derive one-to-one:

* `doc_id` ≡ **`warc_record_id`** (both are the `urn:uuid:`-stripped WARC-Record-ID,
  `decode_warcs_clean._normalize_record_id`)
* `warc_hash` = **`sha256(warc_file)[:12]`** (`decode_warcs_clean._warc_path_hash`), verified
  byte-compatible with arm B's per-WARC filenames

The topic table carries `native_id` = `warc_record_id`, so labels attach exactly as in arm B.
The quality table has no `native_id` and joins positionally, same as before.

## Where the raw HTML comes from — and the decoder trap

The WARCs are not stored anywhere (the 10k raw pool was deleted 2026-08-06), so each worker
**refetches the WARC from Common Crawl** (free ingress; `_s3_to_https` rewrites
`s3://commoncrawl/...` to `https://data.commoncrawl.org/...`), streams it with warcio, and keeps
the records whose id is in that WARC's allow-list. `warc_file` gives the WARC, `warc_record_id`
gives the record inside it. That is the whole chain; nothing is read from a stored HTML tree.

**But lpv11 and the fastpipe line decoded those bytes differently, and that is a real confound.**

| | decoder | behaviour on non-UTF-8 bytes |
|---|---|---|
| lpv11 / `llm_pipeline_v1_1` | `download_warcs._download_one_warc:181` — `content.decode("utf-8", errors="replace")` | inserts `U+FFFD`; no charset detection at all |
| fastpipe_v2 (A), arm B, this plan | `decode_warcs_clean.decode_payload` | WHATWG precedence (BOM → HTTP charset → `<meta charset>` → charset-normalizer → cp1252 → latin-1), strict at every step, and `_iter_decoded_records` **skips any record that still contains `U+FFFD`** |

So re-decoding with the clean decoder does NOT reproduce the bytes the LLM saw on any page that
is not already valid UTF-8. Arm B's canary measured the size of this class: 4.8% of documents
differed from arm C only where the old path had emitted `U+FFFD`.

**Decision: use the clean decoder** (`_decode_one_warc_stream`), for three reasons.

1. It makes the whole swap family share one decoder, so NEW vs A (selection) and NEW vs B differ
   in exactly one thing.
2. Feeding `U+FFFD`-laden HTML to the fork extractor would handicap the new arm for a reason that
   has nothing to do with extraction quality.
3. The clean decoder's `U+FFFD` skip is measurable: those records simply will not appear, so they
   land in the per-WARC `missing` list rather than silently corrupting text.

**The cost, stated plainly:** NEW vs lpv11 then differs in decoder as well as extractor on the
minority of pages that are not clean UTF-8. Quantify it at the canary step — decode one WARC both
ways and report how many allow-listed records differ in HTML. If that class is large, the
alternative is to replicate `content.decode("utf-8", errors="replace")` in the worker, which buys
byte-exact parity with the LLM's input at the cost of parity with arms A and B. Do not pick
silently; report the number.

**Correction on record loss: there is essentially none, and `missing == 0` stays the gate.**
The two decoders select records with the *identical* three filters — `rec_type == "response"`,
non-null `http_headers`, and `"text/html" in Content-Type`
(`decode_warcs_clean.py:246-254` vs `download_warcs.py:169-178`). The clean decoder adds exactly
one further skip, `if REPLACEMENT in html: continue`, and its own comment calls that branch
"unreachable given the latin-1 tail": `decode_payload` falls through cp1252 to latin-1, which maps
every byte 0-255, so it does not manufacture `U+FFFD`. The skip therefore fires only when a page's
own bytes literally encode a replacement character — and such a page carries it under the naive
decoder too.

Crucially, the pages the naive decoder *mangled* (invalid UTF-8, replacement characters inserted)
are exactly the pages the clean decoder handles correctly. It does not drop them; it decodes them
better. So the decoder choice changes HTML **content** on that ~5% of pages, not which records
exist. Keep `missing == 0` as the fleet gate, and treat any non-zero count as a refetch problem,
not an expected decoder artefact.

## Two things that differ from arm B, and they matter

1. **The extractor crashes.** resiliparse-rs SIGSEGVs on `<frameset>` pages (0.108% measured, all
   framesets). That kills the worker process and poisons the pool. `cpu_phase_c` already survives
   it: on `BrokenProcessPool` it re-runs the chunk one doc per process, nulls the offender, and
   rebuilds the pool (`cpu_phase_c.py:144-220`). The new worker MUST reuse that path, plus
   `_install_rust_extractor` with its per-PID unpack dir and the `/root` (never `/tmp`, noexec)
   rule. Arm B's worker needed none of this because upstream resiliparse does not segfault.
2. **There is no flat tokenized cache for lpv11, by design.** `_D_OBS_DEFAULTS` registers the
   sentinel `llm_pipeline_v1_1_decon_10364warcs-gridonly` and `curation_plan.py:378-386` says the
   `tokenized/` path never exists and is never read — the arm is consumed purely as a grid store,
   tokenized by `grid_tokenize` with **marin-tokenizer**, not llama3. The new arm follows the same
   shape: no flat cache, no natural arm, grid store only.

## Steps

Everything runs in **us-east5** (tree, complete labels, store and fits are all co-located; zero
egress). Each step is an Iris job; `set -a; source .env; set +a` first.

### 1. Regroup by WARC — `build_lpv11_swap_allowlist.py`

Clone `experiments/fast_curation/build_swap_allowlist.py` with three substitutions:
`doc_id` → `warc_record_id`, `warc_hash` → `sha256(record["warc_file"])[:12]`, and the label check
`topic.native_id == doc_id` → `topic.native_id == warc_record_id`. Keep both stages:

* `--stage allowlist` — provenance only, grouped into `allowlist/data-{warc_hash}.parquet`.
  Independent of labels, so the extraction fleet can start the moment this lands.
* `--stage labels` — zips each source shard with its topic and quality parquet, asserting equal
  row counts, `native_id == warc_record_id`, `topic.id == quality.id`, and
  `generate_id(text) == topic.id`.

Gates: row total **80,728,641**; `warc_record_id` globally unique; file count ≈ 10,363 (see the
coverage note below). Reuse the bounded storage-client writer from the fastpipe version — the
fsspec path wedged a Zephyr reducer last time.

### 2. Extract — `lpv11_fork_swap_extractor.py` + launcher

Per WARC: refetch from Common Crawl (free), decode with `_decode_one_warc_stream` (the same
decoder, now with the Content-Length check), select records whose `warc_record_id` is in that
WARC's allow-list, and extract with `preprocess.resiliparse_rs_batch` through the
`cpu_phase_c` crash-isolating pool. Write arm A's record shape plus a per-WARC catalog of
`requested / written / short / missing`.

Keep `MIN_TEXT_CHARS = 50` so this swap is comparable to arm B, and report the drop count. Budget
~5% loss on that floor (arm B lost 3,957,128 of 79,582,522, 4.97%).

Fleet: 60 workers, `--cpu 4 --memory 16GB`, batch, us-east5. ~300 worker-hours, 5-6 h wall.
**Re-check the completion registry before every claim** inside a pass — the missing re-check is
what sent arm B's workers into a stale-claim re-extraction loop after the fleet finished.

### 3. Verify, attach labels, build the grid

1. `verify_swap_shards` clone — line count == catalog, `warc_record_id` set == allow-list minus
   dropped, no duplicates.
2. `attach_swap_labels` clone — writes the new arm's co-partitioned topic and quality tables with
   `id = generate_id(new text)`, carrying lpv11's `cluster_24 / topic_prob / token_length /
   score / quality_bucket`. No topic or quality model runs.
3. `grid_label merge` → `grid_tokenize` → `grid_store` → `verify_grid_store` →
   `validate_grid_domains`. `grid_tokenize` already defaults to marin-tokenizer, so nothing to
   change for tokenizer parity with arm L.

### 4. Register and train

* `grid_corpora.GRID_CORPORA["lpv11_forkrs_10k"]` — new tree, us-east5, `JSONL_GZ`,
  `native_id_field="warc_record_id"`, `post_decon=True`.
* `_D_OBS_DEFAULTS["llm_pipeline_v1_1_forkrs_decon_10364warcs-gridonly"]` = the new store's token
  total (a sentinel, exactly like arm L's).
* Vendor `mixtures/lpv11_10k_R3e10_k20_olmixexact_lambda0p01.json` under the new corpus name after
  asserting its 120 cell keys equal the new store's, then
  `_grid_mix_method("lpv11_forkrs_10k_mix_olmixexact_lambda0p01", <sentinel>, "lpv11_forkrs_10k",
  mixture_tag="_olmixexact_lambda0p01")` and add it to `launch_10k_natural.METHOD_NAMES`.
* Launch the 36-cell sweep (`--max-budget 9e20`), then the 60-task olmix eval wave
  (`all+qa_rc` + the 4 MMLU categories, `--keepalive-timeout 172800`, default results subpath).

## Known risks

* **Coverage**: `weborganizer_topic_smoke.py:72` records 10,363 completed WARCs for the lpv11 run,
  one short of the pool. Expect the allow-list to cover 10,363 and do not treat the gap as a bug
  until step 1 confirms which WARC is absent.
* **Frameset segfaults** — handled, but only if the worker reuses the `cpu_phase_c` isolation path.
* **WARC re-parse drift** — `feedback_warc_record_count_drifts_between_runs`; mitigated by
  selecting on `warc_record_id` and asserting `missing == 0` per WARC, plus the Content-Length
  check already landed in `decode_warcs_clean`.
* **us-central1's topic table is only 17 of 3,193 shards.** Build from us-east5 only.
* **Capacity**: the big cells need v5p gangs. Arm B's sweep stalled for days on us-east5's v5p
  backoff and the store was never mirrored. lpv11's store is ALREADY in us-east5 + us-central1 +
  us-east1, so the new arm's store should be mirrored the same way before the sweep rather than
  after it stalls.

## What is NOT needed

No dedup (the tree is already post-dedup), no decon (CORE v2 is mark-only here, so the deduped
tree is the survivor set), no topic or quality model, no OLMIX swarm, no mixture fit, and no flat
llama3 cache.

## Build log

### 2026-10-02 — code landed, allow-list launched

New modules in `experiments/fast_curation/` (lint + pyrefly clean, 10 new tests, 51 across the
whole swap family):

| file | role |
|---|---|
| `lpv11_fork_extractor.py` | claim worker: refetch WARC, select by `warc_record_id`, extract via `cpu_phase_c.ResiliparseRsPool`, write lpv11's record shape + catalog |
| `build_lpv11_swap_allowlist.py` | `--stage allowlist` (provenance) and `--stage labels` (verified lpv11 grid labels), regrouping 3,193 index-named shards into per-WARC parquet |
| `attach_lpv11_fork_labels.py` | joins labels onto the new shards by `warc_record_id`, writes co-partitioned topic/quality tables + `_done` |
| `launch_lpv11_fork.py` | fleet launcher (us-east5, `cpu` extra only) |
| `test_lpv11_fork_swap.py` | hash derivation, selection, crash reporting, label identity guards, attach arithmetic |

Design notes worth keeping:

* The worker reuses `get_spec("lpv11_fastpipe_v2")` purely for its EXTRACTOR settings, so "fork
  extraction" means byte-identically the same thing here as in arm A: pinned commit,
  `main_content=True`, `preserve_formatting="markdown"`, 50 MB html cap.
* Output records keep lpv11's own column names (`warc_record_id`, `warc_file`, ...), so the new
  corpus is schema-comparable with `lpv11_10k` and `native_id_field="warc_record_id"` works
  unchanged. Internally the allow-list carries `warc_hash` derived as `sha256(warc_file)[:12]`.
* The catalog separates `n_short` (below the 50-char floor) from `n_crashed` (frameset SIGSEGV
  victims, which the pool returns as `""` and therefore also trip the floor). `attach` asserts
  unused labels == short + crashed + missing, so a silent mismatch cannot pass.
* `_install_rust_extractor(..., dest=f"/root/resiliparse_rs_{pid}")` — per-PID unpack dir, and
  never under `/tmp` (noexec on Iris CPU workers).
* The worker re-checks the completion registry inside each pass, the fix that stopped arm B's
  stale-claim re-extraction loop.

Registered `lpv11_forkrs_10k` in `grid_corpora.GRID_CORPORA`. **Deliberately NOT yet registered**
in `curation_plan.METHODS` / `launch_10k_natural.METHOD_NAMES`: `_grid_mix_method` needs a
`_D_OBS_DEFAULTS` entry whose value is the new store's token total, which does not exist until
`grid_store` runs. A dangling METHOD_NAMES entry breaks the launcher's default `--methods`, so
both registrations land together after the store is verified.

`/michaelryan/lpv11-fork-allowlist` launched 23:44Z (stage `allowlist`). Gates: row total
**80,728,641**, `warc_record_id` globally unique, ~10,363 files.

### 2026-10-02 07:05Z — allow-list PASSED, decoder question answered with a number

`lpv11-fork-allowlist` succeeded. Gate exact: **80,728,641 rows / 10,364 WARCs / 10,364 files**.
Note it covers the FULL pool — the "10,363 completed WARCs" note in `weborganizer_topic_smoke.py`
does not correspond to a gap in the deduped corpus, so there is no missing-WARC risk after all.

`lpv11-fork-decoder-probe` on WARC 0 (`50c04a36d67e`, 15,298 allow-listed records) measured the
naive-vs-clean decoder difference directly:

| | records | share |
|---|---|---|
| allow-listed | 15,298 | |
| present under BOTH decoders | 15,298 | **100%** |
| identical HTML | 13,759 | 89.9% |
| differ, `U+FFFD` under naive only | 1,420 | 9.3% |
| differ for another reason | 119 | **0.78%** |

Reading:

1. **No record loss, confirmed empirically.** `missing_from_clean` and `missing_from_naive` are
   both empty. The earlier worry that the clean decoder's `U+FFFD` skip would drop lpv11 documents
   was wrong — `missing == 0` is the right fleet gate.
2. **~10% of documents get different input HTML**, and 92% of that difference is the clean decoder
   correctly decoding bytes the old path turned into replacement characters. That is a strict
   improvement in input quality, but it IS a second difference in the NEW-vs-lpv11 comparison and
   must be stated whenever that comparison is reported.
3. **0.78% differ for some other reason** — almost certainly charset-precedence disagreements
   (BOM / HTTP charset / `<meta charset>` winning over a plain UTF-8 read on pages that decode
   under both). Small, but not zero; characterise it if NEW-vs-lpv11 ever turns on a sub-1% effect.

The NEW-vs-fastpipe_v2 comparison (LLM selection at fixed fork extraction) is unaffected: both
arms use the clean decoder, so that one remains a clean single-variable contrast.

### 2026-10-02 08:02Z — canary FAILED on the two-forks import trap, fixed

`lpv11-fork-0` died in `_install_rust_extractor`'s smoke test:
`RuntimeError: resiliparse-rs smoke extraction failed (crashed=['_smoke'], result='')`. The real
error was one frame up, in the pool CHILD:

    File "/app/experiments/fast_curation/upstream_resiliparse.py", line 14, in <module>
      from resiliparse.extract.html2text import extract_plain_text
    ModuleNotFoundError: No module named 'resiliparse.extract.html2text'

Cause: `cpu_phase_c._init_rs_worker` puts the Rust fork FIRST on `sys.path` in every pool child,
and the fork has no `extract.html2text` (see [[project_resiliparse_rs_fork]] — the two forks claim
the same import name and cannot coexist). The `spawn` child re-imports the worker module, whose
module-scope chain reached `upstream_resiliparse`, whose resiliparse import I had hoisted to
module scope on 2026-09-06 to satisfy ruff's PLC0415. Every child died on import; the pool
surfaced it as `BrokenProcessPool`, which its isolation path reports as a hard process death —
so an IMPORT error was displayed as a SIGSEGV. Worth remembering: a smoke-test "SIGSEGV" from
this pool may be an import failure in disguise.

Fix: the resiliparse import in `upstream_resiliparse.py` is function-local again, with a comment
saying why and `# noqa: PLC0415`. The lint rule is simply wrong for this module.

Regression test `test_fork_worker_import_never_pulls_in_upstream_resiliparse` imports the fork
worker in a SUBPROCESS and asserts no `resiliparse.extract*` module is in `sys.modules`. Verified
it fails with the module-scope import reintroduced and passes without it.

Canary relaunched as `lpv11-fork-100`.

### 2026-10-02 11:28Z — EXTRACTION FLEET COMPLETE, plus two findings

10,364/10,364 catalogs. Arithmetic closes exactly: `written + short == requested`, `missing == 0`.

| | |
|---|---|
| documents written | **80,472,692** |
| dropped under the 50-char floor | 255,826 (**0.32%**) |
| frameset crashes, isolated | 10 |
| missing | **0** |
| wall | ~3h20 on 60 workers (~70 s/WARC) |

The 0.32% short-drop is ~15x lower than arm B's 4.97%, which makes sense: the LLM pipeline had
already selected for substantive content, so almost nothing collapses under fork extraction. The
document set survives essentially intact — a cleaner ablation than budgeted.

**Finding 1 — lpv11's own corpus contains 123 duplicated pages.** `sum(n_requested)` is 80,728,518
against 80,728,641 allow-list rows. 48 WARCs are affected. Inspected one
(`69c0fce25586`, record `8dbdc776-...`): the SAME `warc_record_id` and SAME url appear in two
different source shards (`data-00016` row 9228 and `data-02879` row 17694) carrying DIFFERENT LLM
extractions (882 vs 2,139 chars). Fuzzy dedup evidently did not collapse them because the two
extractions are far enough apart in Jaccard terms. The worker keys on `warc_record_id`, so it
writes ONE record per unique page — correct for an extractor comparison, where a second copy
would only duplicate identical fork text. Net: this arm has 123 fewer documents than lpv11
(1.5e-6 of the corpus). Say so when reporting doc counts; do not "fix" it.

**Finding 2 — a real bug in `attach_lpv11_fork_labels`, caught before it ran.** A crashed page's
text is `""`, which trips the `MIN_TEXT_CHARS` floor, so it is counted in `n_short` AND recorded
under `n_crashed`. The assertion summed them, double-counting, and would have failed attach on
exactly the WARCs with a frameset segfault (10 fleet-wide). Fixed to
`n_dropped = n_short + len(missing)`. Regression test
`test_attach_arithmetic_counts_a_crashed_page_once` verified to fail with the double-count
restored. The canary could not have caught this: it had `n_crashed == 0`.

`lpv11-fork-labels` (stage `labels`) launched 11:36Z. Then: `attach_lpv11_fork_labels`,
`grid_label merge`, `grid_tokenize`, `grid_store`, `verify_grid_store`, `validate_grid_domains`,
mirror the store to us-central1 + us-east1 BEFORE the sweep, vendor lpv11's mixture under the new
corpus name, register `_D_OBS_DEFAULTS` + `_grid_mix_method` + `METHOD_NAMES` together, launch.

### 2026-10-03 — GRID COMPLETE; mirrors then sweep

All gates passed. `verify_grid_store`: 120/120 cells loadable. `validate_grid_domains`:
**94,404,061,562 tokens / 80,472,692 docs**, 19 cells below the block floor at natural prior,
`cells.json` written.

**Headline numbers vs lpv11 (its own LLM extraction of the same documents):**

| | documents | tokens | tokens/doc |
|---|---|---|---|
| lpv11 (LLM extract) | 80,728,641 | 111,324,334,812 | 1,379 |
| this arm (fork extract) | 80,472,692 | 94,404,061,562 | 1,173 |
| delta | -255,949 (-0.32%) | -16.92B (-15.2%) | **-15%** |

The 255,949 missing documents are 255,826 that fell under the 50-char floor plus the 123
duplicate record ids. Sampling the dropped ones shows they were thin in lpv11 too (median 328
chars, mean 451) and carry only ~28M tokens, **0.03%** of the corpus. So the doc loss does NOT
explain the token gap: the fork simply writes ~15% less text per retained page. That difference
is what the ablation prices.

Vendored `mixtures/lpv11_forkrs_10k_R3e10_k20_olmixexact_lambda0p01.json` = lpv11's own fit with
`corpus` renamed, after asserting both `weights` and `natural` key sets equal the new store's 120
cells. Same docs in each bucket, same weights -- the mix is identical by construction.

Registered `_D_OBS_DEFAULTS["llm_pipeline_v1_1_forkrs_decon_10364warcs-gridonly"] = 94_404_061_562`
(a grid-only sentinel, as for lpv11), `curation_plan.METHODS["lpv11_forkrs_10k_mix_olmixexact_lambda0p01"]`
via `_grid_mix_method`, and the name in `launch_10k_natural.METHOD_NAMES`. Dry-run: 38 cells.

Store mirrors to us-central1 + us-east1 launched 00:44Z (184.6 GB, ~$7.40, user approved) BEFORE
the sweep -- the lesson from the fastpipe_v2_upstreamrs arm, which stalled for days because its
store was single-region when us-east5 v5p went into backoff. Sweep launches on mirror completion
with `--allowed-regions us-east5 us-central1 us-east1`, then the 60-task olmix eval wave.

