# Nemotron-CC Common Crawl Coverage

NVIDIA's Nemotron-CC dataset (Su et al., arXiv:2412.02595) was built from **99 Common Crawl snapshots: CC-MAIN-2013-20 through CC-MAIN-2024-30, inclusive**. Quote from the paper:

> "Combining the techniques above to the 99 snapshots CC-MAIN-2013-20 through CC-MAIN-2024-30 of Common Crawl, we create a 6.3T token dataset."

Cross-referencing the range against per-crawl WARC counts (see `common_crawl_warc_counts.md`):

| Year | Snapshots in Nemotron-CC | WARC files |
|---|---:|---:|
| 2013 | 2 (-20, -48) | 83,500 |
| 2014 | 8 | 403,578 |
| 2015 | 10 | 324,433 |
| 2016 | 9 | 302,520 |
| 2017 | 12 | 847,927 |
| 2018 | 12 | 800,000 |
| 2019 | 12 | 688,000 |
| 2020 | 9 | 571,600 |
| 2021 | 9 | 615,840 |
| 2022 | 6 | 480,000 |
| 2023 | 5 | 428,000 |
| 2024 | 5 (-10, -18, -22, -26, -30) | 450,000 |
| **Total** | **99** | **5,995,398** |

## Headline numbers

- **~6.0M WARC files** are within Nemotron-CC's input scope.
- This is **~75.6%** of all 7.93M WARCs Common Crawl has published to date (2026-04-10).
- The remaining ~1.93M WARCs (CC-MAIN-2024-33 onward) sit outside Nemotron-CC.
- Nemotron-CC's published size is **6.3T tokens** (4.4T globally-deduplicated original + 1.9T synthetically generated) — token count reflects heavy filtering/dedup/synthesis, not raw WARC bytes.

## Caveats

- The paper states the range and count but doesn't enumerate every ID. The "all 99 snapshots between those endpoints in `collinfo.json`" reading yields exactly 99, matching the paper.
- Appendix D lists a separate **13-snapshot ablation subset** for 1T-token experiments (2019-35, 2020-05/29/45, 2021-04/21/43, 2022-05/27/49, 2023-06/14/23). Don't confuse this with the full 99-snapshot dataset.

## Sources

- [Nemotron-CC paper (arXiv)](https://arxiv.org/abs/2412.02595) / [HTML](https://arxiv.org/html/2412.02595v1)
- [NVIDIA ADLR project page](https://research.nvidia.com/labs/adlr/Nemotron-CC/)
