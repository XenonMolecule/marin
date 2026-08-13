# Random-WARC subset manifests (seed 0)

Reproducible uniform random draws from the 10,364-WARC DCLM 400m-1x pool
(`experiments/distill/dclm_400m_1x.txt`), in standard `--manifest` format
(WARC paths, one per line; `#` provenance headers are skipped by the pipeline).

Derived byte-for-byte from the authoritative `../subsets/baseline_warcs_N_random.txt`
files that the training caches were actually built from (via `subset_random100_10k.py`
→ `curation_plan.METHODS[*_random_N]`). The nested-chain method was verified to
reproduce the existing 1000 and 2000 exactly before extending it.

## Nested extraction chain (use these to scale up incrementally)

    random_warcs_100.txt   ⊂ random_warcs_300.txt   ⊂ random_warcs_500.txt
      ⊂ random_warcs_1000.txt ⊂ random_warcs_2000.txt ⊂ random_warcs_3000_nested.txt

Each file is a strict superset of the previous, so going N→N+ only requires
extracting the NEW WARCs — no re-extraction:

| step            | N     | new WARCs to extract |
|-----------------|-------|----------------------|
| base            | 100   | 100                  |
| 100 → 300       | 300   | +200                 |
| 300 → 500       | 500   | +200                 |
| 500 → 1000      | 1000  | +500                 |
| 1000 → 2000     | 2000  | +1000                |
| 2000 → 3000     | 3000  | +1000                |

Method: 100/300/500/1000 are prefixes of the same seed-0 `random.Random(0).sample`
selection sequence; 2000 = nested-1000 ∪ `random.Random(0).sample(sorted(pool−1000), 1000)`;
3000_nested = nested-2000 ∪ `random.Random(0).sample(sorted(pool−2000), 1000)`.
(Plain `sample(pool, k)` breaks nesting for k≥2000 — CPython switches sampling
algorithm — hence the union construction.)

## Independent 3000 (matches what already ran)

`random_warcs_3000_independent.txt` is a **separate** plain seed-0 draw
(`sample(sorted(pool), 3000)`), NOT nested (it differs from the nested 3000 by
3154 WARCs). This is the draw the already-completed random dclm/nemotron 3000
caches were built from (`baseline_dclm-cf177e`, `baseline_nemotron_full-75f981`).
Use it only to reproduce those specific runs; use `random_warcs_3000_nested.txt`
for the incremental scaling chain.
