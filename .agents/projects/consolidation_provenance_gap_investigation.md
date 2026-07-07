# Consolidation Provenance Gap — Root-Cause Investigation

**Symptom.** The high_quality 10364-WARC HuggingFace export
(`build_high_quality_hf_export.py`) re-attaches per-doc WARC metadata to the
decontaminated/deduped survivors via an **exact-text join**. Of 19,968,996
survivor docs, **1,806 (0.009%)** had NO byte-identical match in the raw
consolidated archive and received null `url/warc_record_id/warc_file/snapshot`.
The misses are scattered, not clustered.

**Bottom line.** The join key is provably equivalent to the dedup reshape key, so
this is NOT a key-expression bug. The two sides do not read the same set of
files, but the join reads a **superset** of dedup's input, so that asymmetry
makes the join match *more*, not fewer — it cannot by itself explain a miss. The
residual 1,806 are best explained by **a small completeness gap between the
files dedup actually consumed and the files present in the consolidated archive
at join time** (rank-winner copy vs. the copies the transfer shipped), with a
secondary possibility of **rare text-identity drift** (non-UTF8 / surrogate
round-trip, or trailing-content differences between region copies of the "same"
batch). A short in-region diagnostic (spec'd below) is required to decide
between these; code analysis narrows it to these two but cannot fully separate
them.

---

## 1. The join key is equivalent to the dedup reshape key (NOT the bug)

Raw row written at extraction (`run_extract_standalone.py:855-864`):

```python
{
  "text": cleaned,            # _clean_text(raw), always >= MIN_OUTPUT_CHARS=50 (line 836 guards)
  "generated_text": raw,      # uncleaned model output
  "url": ..., "warc_record_id": ..., "warc_file": ..., "snapshot": ...,
}
```

Only **kept** records are written (filtered records `continue` at
`run_extract_standalone.py:837,846,848`), and a record is kept only if
`len(cleaned) >= 50` (line 836). So in every raw row, `text` is present and
non-empty.

Reshape key (`dedup_extracted.py:199`):
```python
text = record.get("text") or record.get("generated_text")
```
Join key (`build_high_quality_hf_export.py:82`), identical expression. Because
`text` is always non-empty in raw rows, **both sides always select `text`**, never
`generated_text`. The `or generated_text` branch is dead for this corpus. The
cleaned `text` is **stored at extraction time** and read verbatim by both dedup
and the join — neither side recomputes `_clean_text`
(`run_extract_standalone.py:364-372`), so cleaning drift is impossible.

Decon (`decon_apply.py:94-99`) only `continue`s (drops) docs; it never mutates
`rec`. The export docstring's premise ("byte-identical to exactly one raw
record") holds. ⇒ The key/identity path is sound.

---

## 2. The two sides read DIFFERENT file sets — but the join reads a SUPERSET

This is the central structural fact and the answer to Question 1.

**Dedup input** (`dedup_extracted.py:_load_canonical_paths`, lines 141-188):
- reads `resolved_{spec}.jsonl.gz` (the resolver manifest),
- for each (warc_hash, batch_idx) takes the single `row["path"]` = the
  **rank-winner copy** chosen by the resolver,
- remaps that one path into the consolidated archive
  (`_remap_to_archive`, lines 109-114) →
  `baseline_llm_extraction_consolidated/by_region/{region}/high_quality/`.

So dedup consumes **exactly one copy per (hash, batch_idx)** — the winner.

**Join raw input** (`build_high_quality_hf_export.py:_raw_files`, lines 199-205):
```python
for region in RAW_REGIONS:
    files.extend(_glob(f"{RAW_BASE}/{region}/high_quality/data-*/batch_*.jsonl.gz"))
```
This globs **every batch file of every region** present in the consolidated
archive — i.e. **all** duplicate copies, not just the resolver winner.

**Therefore the join's raw side ⊇ dedup's input.** Every text dedup saw came from
a winner copy that (per §3) is guaranteed present in the archive, AND the join
also reads the losing copies on top. A correctly-shipped archive should give a
100% match. The miss can only arise if a winner text is **physically absent** from
the archive at join time, or if the archive copy is **not byte-identical** to what
dedup read.

The resolver winner is the same expression on both sides for `text` chars only
in spirit — note the resolver's tiebreaker (`resolve_duplicates.py:122-129`)
ranks on `sum_text_chars`, and **`sum_text_chars` is computed over `r["text"]`
only** (`inventory_region.py:164`: `chars += len(r.get("text", ""))`), matching the
join key. So winner selection is aligned with the join key, not skewed toward
`generated_text`.

---

## 3. Why a winner copy is *normally* guaranteed to be in the archive

`resolve_duplicates.py` filters the resolved manifest to WARCs done in some
region (line 239: `resolved = [r for r in resolved if r["warc_hash"] in
warcs_with_done_anywhere]`) and writes `done_warcs.txt` from the **same**
predicate (lines 282-285). `transfer_region.py` then copies every batch whose
`data-{hash}/` hash is in `done_warcs.txt`
(`transfer_region.py:166-176`). So: winner ⊆ done-hash set ⊆ transferred set.
On paper, every winner batch is copied. The OVERNIGHT_REPORT claims byte-exact
parity across all 5 regions (lines 59-68: equal source/dest byte totals,
689,876 files).

The gap, then, lives in the seams of that "on paper" guarantee.

---

## 4. Root-cause ranking (scattered 1,806 ≈ 0.009%)

### Rank 1 — Archive-vs-dedup-input completeness skew at the batch/copy level (MOST LIKELY)

The resolver ran ONCE (2026-04-19). Its `resolved_high_quality.jsonl.gz` froze a
specific winner path per (hash, batch_idx). Several independent mechanisms can
leave a *specific winner copy's text* unreachable by the join even though gross
byte parity holds:

- **Winner in a region whose copy differs from the transferred copy.** The
  resolver picks the winner from the **inventory** snapshot; transfer copies the
  **live** source. If a batch was re-extracted/rewritten in its source region
  between inventory and transfer (extraction is resumable and steals/re-runs
  WARCs — `run_extract_standalone.py:887-899` claim/steal logic), the bytes the
  resolver inventoried (and that flowed into the tokenized/deduped corpus if the
  reshape ran against the *pre-transfer* regional path) need not equal the bytes
  finally in the archive.
- **Reshape may have read the regional source, not the archive copy, for some
  batches.** `_remap_to_archive` (lines 109-114) remaps to the archive, but the
  live deduped corpus that produced the survivors was built during the
  consolidation window; if any reshape input resolved to a copy that was later
  superseded (or to a region copy with a longer/edited text that won on
  `sum_text_chars`), the survivor text can be a string that exists in *no*
  surviving archive copy.

Why this fits the evidence: misses are **scattered** (one-offs distributed across
many WARCs), exactly what per-batch copy-identity races produce — not a
clustered "whole WARC/region missing" signature.

### Rank 2 — Rare text-identity drift across region copies (PLAUSIBLE)

The join compares Python `str` after `json.loads` + `text.encode("utf-8")`
(`build_high_quality_hf_export.py:96`). Two copies of "the same" batch can differ
by:
- **surrogate / invalid-unicode round-tripping** — model output can contain lone
  surrogates or `U+FFFD`; if region copies were produced by different
  vLLM/transformers versions, the JSON-escaped bytes can differ, so
  `text.encode("utf-8")` differs (or raises and the record is skipped). The
  useful-classifier work already hit U+FFFD decode issues
  (`project_modernbert_warc_bert_pipeline`), so non-UTF8 in this corpus is real.
- **trailing-content / cleaning-boundary differences** — if the same page was
  extracted twice with slightly different `<think>`/field-marker boundaries, the
  stored cleaned `text` differs by a few trailing chars; the resolver's
  `sum_text_chars` tiebreaker (lines 126-129) actively **prefers the
  longer-text copy**, so the deduped corpus can carry a text variant that the
  join's chosen raw copy (smallest `(warc_file, warc_record_id)`,
  `build_high_quality_hf_export.py:142`) doesn't reproduce.

This also yields scattered misses. It's ranked below Rank 1 only because §1
shows cleaning is not recomputed and the stored field is read verbatim — drift
requires the two *stored* copies to already differ, which is a narrower event.

### Rank 3 — Non-deterministic LLM re-extraction (CONTRIBUTING, not primary)

LLM extraction is explicitly stochastic (`dedup_extracted.py:4-9`: "two
extractions of the same page come out as different strings"). The claim/steal
resume path (`run_extract_standalone.py:887-899`) means a WARC can be extracted
in two regions producing different text per record. This is the *upstream
generator* of the Rank-1/Rank-2 divergence rather than an independent cause: it
creates multiple non-identical copies; whether a survivor goes unmatched then
depends on which copy dedup consumed vs. which copies remain in the archive.
Note fuzzy dedup collapses *near*-dups to one canonical, but the canonical it
keeps is one specific stored string; if that exact string's source copy isn't
the one in the archive, the exact-text join misses.

### Rank 4 — Resolver `resolved` vs archive ordering / empty-batch skips (LOW)

`_load_canonical_paths` skips `num_records == 0` batches (lines 162-164) and
the reshape skips empty/blank text (lines 199-202). These reduce dedup input but
cannot create a survivor with no archive match (they only drop). Ruled out as a
direct cause; listed for completeness.

**Ranking summary:** Rank 1 (copy-identity skew between dedup's frozen winner and
the live archive) > Rank 2 (unicode/trailing text drift across copies) > Rank 3
(stochastic re-extraction as the upstream driver) ≫ Rank 4 (empty-skip, ruled
out). 0.009% is consistent with rare per-batch races, not a systematic flaw.

---

## 5. Recommendations to GUARANTEE traceability in future releases

### R1 (PRIMARY) — Carry a stable per-doc id THROUGH dedup and decon

Today the per-doc identifiers (`warc_record_id` UUID + `warc_file`) exist in the
raw rows (`run_extract_standalone.py:861-862`) but are **projected away** at the
very first dedup step (`dedup_extracted.py:202` yields `{"text": text}` only).
Every downstream join is then forced to key on exact text, which is fragile to
exactly the drift in §4.

Fix: make `_reshape_records` (`dedup_extracted.py:196-202`) emit a stable id
alongside text, e.g.:
```python
yield {"text": text, "doc_id": record["warc_record_id"], "warc_file": record["warc_file"]}
```
and propagate `doc_id`/`warc_file` through:
- `normalize_step` (`text_field="text"` stays; add the id columns as passthrough),
- the fuzzy apply (`_apply_fuzzy_dups._process_one`, line 310 yields `{"text": ...}`
  → also yield `doc_id`, `warc_file`),
- `decon_apply.filter_shard` (`decon_apply.py:94-99` already preserves whole
  `rec`, so id survives for free once it's present).

Then the HF export becomes a **trivial id passthrough** — no join, no exact-text
matching, no miss class at all. `warc_record_id` is a CC URN UUID
(`_normalize_record_id`, `run_extract_standalone.py:353-355`), globally unique
per source doc, so it is a sound primary key. This is the single highest-leverage
change and it eliminates the failure mode rather than measuring it.

Caveat to verify: fuzzy dedup collapses near-dup *clusters* to one canonical
row; ensure the id carried is the canonical row's own id (it already is, since
the canonical row is a real normalized record with its passthrough columns).

### R2 — Make consolidation provably lossless with record-count parity, not just byte totals

The OVERNIGHT_REPORT verified **byte/file-count parity** (lines 59-68) but never
verified **record-level coverage of the dedup winners**. Add a post-transfer
assertion in the resolve/transfer flow:
- For every (warc_hash, batch_idx) in `resolved_{spec}.jsonl.gz`, assert the
  winner path exists in the archive AND its decompressed `num_records` equals the
  inventory's `num_records` for that exact copy (the inventory already stores
  `num_records` and `sum_text_chars` per copy — `inventory_region.py:147-170`).
- Fail the consolidation if any winner copy is missing or count-mismatched.

This converts "173 GB == 173 GB" into "every record dedup will read is present
and unchanged," which is the property the export actually depends on.

### R3 — Freeze the dedup input set as an explicit artifact and join against IT

The deduped corpus was built from `_load_canonical_paths` (the resolver winners).
The export joined against `_raw_files()` (all archive copies). Even with R1, make
the export read its raw/metadata side from **the same resolved winner list**
dedup used (`_load_canonical_paths(spec, hashes)`), not a fresh glob. Reading the
identical input set removes the §2 superset/subset asymmetry and guarantees the
metadata source is exactly what produced the text. Concretely, in
`build_high_quality_hf_export._raw_files`, replace the per-region glob with the
resolved-winner paths for the 10364 hashes. (Belt-and-suspenders alongside R1.)

---

## 6. Diagnostic job spec (DO NOT RUN — spec only) to decide Rank 1 vs Rank 2

Run a single **us-central1-pinned** Iris CPU job (zero cross-region reads — the
consolidated archive and survivors are both in `marin-us-central1`). Goal: for
the 1,806 unmatched survivor texts, classify each as "no archive copy at all"
(Rank 1) vs "an archive copy exists that is *almost* equal" (Rank 2).

Inputs:
- Survivors: `gs://marin-us-central1/documents/baseline_high_quality_decon_deduped/10364warcs/deduped/data-*.jsonl.gz`
- Raw archive: `gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/by_region/{region}/high_quality/data-*/batch_*.jsonl.gz`
- Resolver winners: `gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/resolved/resolved_high_quality.jsonl.gz`

Procedure:
1. **Recover the 1,806.** Re-run the join's `attach` reducer (or read the existing
   `JOINED_PREFIX` parquet, filter `warc_file IS NULL`) to get the 1,806 survivor
   texts. Hold their blake2b-128 hashes in a set `MISS`.
2. **Build two raw indexes** over the archive in one streaming pass per batch
   file (flat_map over all `batch_*.jsonl.gz`):
   - `exact_h = blake2b128(text)` for every raw record (the join key);
   - `norm_h = blake2b128(unicodedata.normalize("NFC", text).strip())` AND a
     `prefix_h = blake2b128(text[:512])` for every raw record.
3. **Classify each miss:**
   - in `exact_h` but not joined → join/bucketing bug (should be empty; sanity).
   - not in `exact_h` but in `norm_h` → **Rank 2, unicode/whitespace drift**.
   - not in `norm_h` but `prefix_h` collides with some raw record whose full text
     differs → **Rank 2, trailing/boundary drift**; emit char-level diff length.
   - no `exact/norm/prefix` hit anywhere → **Rank 1, text physically absent**;
     then look up that survivor's expected source via R3's resolved winners (if
     id were carried) or report "absent" with the survivor's first 200 chars.
4. **Cross-check Rank 1 against the archive's completeness:** for each absent
   miss, take its survivor text, and confirm whether the resolver winner copy for
   its (would-be) batch is present in the archive with matching `num_records`
   vs. inventory. A systematic "winner copy missing" cluster confirms a transfer
   completeness gap; a scatter of "winner present but text differs" confirms
   Rank 2.

Output: a small JSON `{rank1_absent, rank2_nfc, rank2_trailing, unexplained}`
histogram + 20 samples per bucket. Cost: read-only, intra-region, ~the same scan
the join already does once → low. This decisively separates "missing batches"
from "text normalization diff" and tells us whether R2 (parity check) or R1/R3
(id passthrough / same-input join) is the load-bearing fix.

---

## Evidence index (file:line)

- Raw row schema: `run_extract_standalone.py:855-864`
- Kept-only + MIN_OUTPUT_CHARS guard: `run_extract_standalone.py:836-853`
- `_clean_text` (stored once, never recomputed downstream): `run_extract_standalone.py:364-372`
- `warc_record_id` normalization (URN UUID): `run_extract_standalone.py:353-355`
- Claim/steal resume (source of multi-copy non-determinism): `run_extract_standalone.py:887-899`
- Reshape key + projection to `{text}` only: `dedup_extracted.py:196-202`
- Dedup reads resolver WINNER copy only: `dedup_extracted.py:141-188`
- Archive remap: `dedup_extracted.py:109-114`
- Fuzzy apply yields `{text}` only: `dedup_extracted.py:305-310`
- Join key (same expression as reshape): `build_high_quality_hf_export.py:80-85`
- Join reads ALL archive copies (superset): `build_high_quality_hf_export.py:199-205`
- Deterministic dup collapse (smallest warc_file,record_id): `build_high_quality_hf_export.py:141-142`
- Null-on-orphan kept for accounting: `build_high_quality_hf_export.py:144-147`
- Resolver tiebreak on `sum_text_chars` (= over `text`): `resolve_duplicates.py:122-129`
- `sum_text_chars` computed over `r["text"]`: `inventory_region.py:164`
- `resolved` and `done_warcs` share the done predicate: `resolve_duplicates.py:239,282-285`
- Transfer copies only done-hash batches: `transfer_region.py:166-176`
- Transfer is byte-copy, no normalization: `transfer_region.py:104-114`
- Decon drops docs, never mutates text: `decon_apply.py:94-99`
- Byte-parity (not record-coverage) verification: `OVERNIGHT_REPORT.md:59-68`
