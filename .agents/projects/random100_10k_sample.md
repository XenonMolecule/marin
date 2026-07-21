# Random-100 WARC sample from the 10k pool (vs the biased head-100)

**Status:** subsetting IN FLIGHT (2026-07-15). Owner: Michael Ryan.

## Why (the bias, quantified)

The existing `*_100` methods are `head -100` of the **date-SORTED**
`experiments/distill/baseline_warcs_3000.txt` — `subset_baselines.py` says it
outright: *"Subsets are deterministic prefixes (head -N)"*. Measured:

| | distinct crawls | span |
|---|---|---|
| biased head-100 (`subsets/baseline_warcs_100.txt`) | **2** (2013-20 x65, 2013-48 x35) | **2013 ONLY** |
| new random-100 (`subsets/baseline_warcs_100_random.txt`) | **60** | 2013→2022 (all 10 yrs) |
| full 10k pool (`distill/dclm_400m_1x.txt`, 10,364) | 89 | 2013→2022 |

So N=100 was trained on 2013-only web while every larger N spans a decade → almost
certainly the "real gap" Michael observed. Overlap between biased-100 and random-100
is **1 WARC** (essentially independent samples).

## The draw (done)

`sample_random_warcs.py --pool experiments/distill/dclm_400m_1x.txt --n 100 --seed 0
--output experiments/distill/subsets/baseline_warcs_100_random.txt`
— uniform over the WHOLE sorted pool, seed 0, byte-identical on re-draw (verified).
Same method/seed as the existing `baseline_warcs_3000_random.txt`.

## Build approach: SUBSET the existing 10k extractions (no re-extraction)

All 3 methods already exist for the full 10,364 WARCs; join fields VERIFIED:

| method | 10k source (region) | join |
|---|---|---|
| dclm | `filtered/dclm_400m_1x_10k_dclm_resharded-1fe977` (c2) | `warc_record_id` via metadata |
| nemotron_full | `filtered/dclm_400m_1x_10k_nemotron_full-96bad9` (c2) | `url` via metadata |
| high_quality | `documents/baseline_high_quality_hf_export/10364warcs/joined` parquet (c1) | `warc_file` DIRECT (no metadata) |

Metadata lookup: `metadata/dclm_400m_1x_10k_warc_metadata-79158f` (warc_record_id,
url, warc_file, snapshot; 10,364 shards → key-set built with a PARALLEL Zephyr pass,
unlike subset_baselines' driver-serial version which would take hours at 10k).

Re-extraction rejected: HQ is ~5h/WARC of LLM work (~500 v6e-hours) and the raw 10k
WARCs may be cleaned up. Subsetting = hours of in-region CPU, $0 egress.

Script: `experiments/baseline_collection/subset_random100_10k.py` (modeled on
subset_baselines.py; adds parquet reader for HQ + parallel keyset).
Launched: `subset-random100-c2` (dclm+nemotron, us-central2),
`subset-random100-c1` (high_quality, us-central1).

## BUG FOUND (caught pre-launch)

`subset_baselines._load_subset_warc_set()` strips blank lines but NOT `#` comments,
while `sample_random_warcs.py` writes a 4-line `#` provenance header → the header
loads as 4 bogus WARC paths (manifest read as 104, not 100). Harmless in practice
(no `warc_file` equals a comment) but corrupts counts/asserts. **This also explains
`baseline_warcs_3000_random.txt` being 3004 lines / "3000 of 3004" matching the pool.**
Fixed locally via `_load_warc_manifest()` (strips `#`), NOT by mutating the shared 3k
path. Verified: loads exactly 100, all `s3://commoncrawl/`.

## DONE — all 3 arms LAUNCHED (2026-07-15)

Caches (tokenized in source region, mirrored to us-east5, methods pin_region=us-east5
where capacity is free since decon was paused). Doc counts validate the subset to
within ~7% of the predicted 1% (100/10364):

| method | cache | tokens | docs | (expected docs) |
|---|---|---|---|---|
| dclm_random_100 | `dclm_random_100warcs-350025` | 67,666,557 | 55,106 | ~57.2k |
| nemotron_full_random_100 | `nemotron_full_random_100warcs-ade768` | 103,440,840 | 183,561 | ~171k |
| high_quality_random_100 | `high_quality_random_100warcs-88f468` | 202,814,071 | 189,852 | ~192.7k |

Registered in curation_plan (_D_OBS + METHODS, pin us-east5) + base names
`dclm_random`/`nemotron_full_random`/`high_quality_random` in warc_scaling_plan.
LAUNCHED **81 runs** (27/method, N=100 grid = hidden 256/512/768/1536 x budgets
3e15..1e20 minus corner drops) via coordinators `warc-random100-launch` (dclm+hq)
and `warc-random100-nemo`. 27 is a SUPERSET of the biased arms' cells (dclm_100 has
11 on the old narrow grid; high_quality_100 has 33) so every biased cell has a
random counterpart.

## KEY FINDING: biased 2013 WARCs are token-DENSER for every method

| method | biased (2013-only) | random (2013-2022) | ratio |
|---|---|---|---|
| dclm | 98M | 68M | 1.44x |
| high_quality | 414M | 203M | 2.0x |
| nemotron_full | 317M | 103M | 3.1x |

So the old N=100 differed from a random sample in BOTH content mix AND data volume
(at fixed FLOPs => different epoch counts). dclm's 1.44x is CLEAN (deterministic
per-WARC filter, no dedup confound) => 2013 WARCs are genuinely denser. nemotron's
3.1x is the largest — consistent with Nemotron-CC covering 2013-era crawls far more
densely than the 2013-2022 average. hq's 2.0x = density + the decon/dedup confound.

## Next

Evals (DCLM Core v2) over the 81 random-100 checkpoints once trained, then compare
biased vs random per method. Michael declined a re-biased-100-from-10k arm, so read
the HQ arm with its dedup confound in mind.

## CONFOUND to flag (HQ)

Existing biased-100 came from the **3000-WARC** extraction; random-100 comes from the
**10k** extraction.
- dclm/nemotron: no real confound — filters are deterministic per-WARC, same WARC → same docs.
- **high_quality: REAL confound** — the 10k HQ export is decon+dedup'd and from a
  different extraction run than the 3k HQ (`high_quality_100warcs-e2ecfb`). A
  biased-vs-random HQ gap could partly be extraction/decon, not sampling.
  Clean fix if wanted: also subset a **biased-100 from the 10k** (same machinery,
  cheap) so both arms share one extraction (+11 HQ runs).
