# Checkpoint Size Report

**Generated 2026-05-13** for math/code/medical experiments by Michael Ryan.
Paths are listed verbatim in `checkpoints_paths.txt`; sizes per path in `checkpoint_sizes.tsv` (sorted desc).

## Totals

- **Total: 149.68 TB** across **298 checkpoints**

## By region

| Region | Count | Size |
|---|---:|---:|
| marin-us-central1 | 293 | 149.66 TB |
| marin-us-east5 | 5 | 0.01 TB |

## By domain

| Domain | Count | Size |
|---|---:|---:|
| code | 109 | 82.13 TB |
| medical | 77 | 41.92 TB |
| math | 112 | 25.62 TB |

## By model scale

| Scale | Count | Size |
|---|---:|---:|
| 0.6B | 259 | 75.13 TB |
| 14B | 23 | 74.37 TB |
| 0.6B/8B | 16 | 0.18 TB |

## By experiment family

| Domain | Scale | Family | Count | Total | Avg per ckpt |
|---|---|---|---:|---:|---:|
| code | 14B | code-v3-14b-p2 | 12 | 43.24 TB | 3.60 TB |
| medical | 0.6B | medical-resili | 28 | 27.37 TB | 977.6 GB |
| code | 14B | code-v3-sweep-14B | 10 | 27.17 TB | 2.72 TB |
| medical | 0.6B | medical-extract | 48 | 10.59 TB | 220.6 GB |
| math | 0.6B | math-v2-sweep | 24 | 9.24 TB | 385.1 GB |
| math | 0.6B | math-top3-resili | 27 | 7.10 TB | 262.8 GB |
| medical | 14B | medical-14b | 1 | 3.96 TB | 3.96 TB |
| math | 0.6B | math-top3-extract | 28 | 3.82 TB | 136.3 GB |
| code | 0.6B | code-v3-p2 | 12 | 3.46 TB | 288.6 GB |
| code | 0.6B | code-v3-sweep-0.6B | 24 | 3.05 TB | 127.0 GB |
| math | 0.6B | math-resili-reg | 2 | 2.43 TB | 1.21 TB |
| code | 0.6B | code-v3-p2c | 12 | 1.75 TB | 145.7 GB |
| code | 0.6B | code-v3-p3 | 12 | 1.32 TB | 109.7 GB |
| math | 0.6B | math_multi_v2-sft | 3 | 1.04 TB | 348.1 GB |
| code | 0.6B | code-v3-p2b | 12 | 887.1 GB | 73.9 GB |
| math | 0.6B | math-mix | 6 | 816.0 GB | 136.0 GB |
| code | 0.6B | code-v3-lowlr | 7 | 732.1 GB | 104.6 GB |
| math | 0.6B | math-v2-reg | 2 | 662.8 GB | 331.4 GB |
| math | 0.6B | math-mix-v2 | 4 | 338.6 GB | 84.7 GB |
| code | 0.6B | code-extract-sft | 6 | 295.7 GB | 49.3 GB |
| code | 0.6B | code-resiliparse-sft | 2 | 224.1 GB | 112.1 GB |
| math | 0.6B/8B | mathhelpforum-sft | 16 | 176.5 GB | 11.0 GB |

## Top 20 largest individual checkpoints

| Size | Path |
|---:|---|
| 3.96 TB | `gs://marin-us-central1/checkpoints/medical-14b-extract-default-qwen3-14b-base-64cc1b/` |
| 3.60 TB | `gs://marin-us-central1/checkpoints/code-v3-14b-p2-wd0p001_wu0p03-qwen3-14b-base-cb4fb3/` |
| 3.60 TB | `gs://marin-us-central1/checkpoints/code-v3-14b-p2-wd0p05_wu0p0-qwen3-14b-base-799263/` |
| 3.60 TB | `gs://marin-us-central1/checkpoints/code-v3-14b-p2-wd0p1_wu0p1-qwen3-14b-base-067429/` |
| 3.60 TB | `gs://marin-us-central1/checkpoints/code-v3-14b-p2-wd0p001_wu0p0-qwen3-14b-base-ade815/` |
| 3.60 TB | `gs://marin-us-central1/checkpoints/code-v3-14b-p2-wd0p01_wu0p0-qwen3-14b-base-90dbb3/` |
| 3.60 TB | `gs://marin-us-central1/checkpoints/code-v3-sweep-lr5e-5_bs32-qwen3-14b-base-a5a415/` |
| 3.60 TB | `gs://marin-us-central1/checkpoints/code-v3-sweep-lr1e-5_bs32-qwen3-14b-base-3161d5/` |
| 3.60 TB | `gs://marin-us-central1/checkpoints/code-v3-sweep-lr1e-6_bs32-qwen3-14b-base-69a784/` |
| 3.60 TB | `gs://marin-us-central1/checkpoints/code-v3-14b-p2-wd0p05_wu0p1-qwen3-14b-base-00b606/` |
| 3.60 TB | `gs://marin-us-central1/checkpoints/code-v3-sweep-lr2e-6_bs32-qwen3-14b-base-f6fc2f/` |
| 3.60 TB | `gs://marin-us-central1/checkpoints/code-v3-14b-p2-wd0p01_wu0p03-qwen3-14b-base-67dc11/` |
| 3.60 TB | `gs://marin-us-central1/checkpoints/code-v3-14b-p2-wd0p05_wu0p03-qwen3-14b-base-a090fe/` |
| 3.60 TB | `gs://marin-us-central1/checkpoints/code-v3-14b-p2-wd0p001_wu0p1-qwen3-14b-base-9a6428/` |
| 3.60 TB | `gs://marin-us-central1/checkpoints/code-v3-14b-p2-wd0p1_wu0p03-qwen3-14b-base-a3b0ef/` |
| 3.60 TB | `gs://marin-us-central1/checkpoints/code-v3-14b-p2-wd0p01_wu0p1-qwen3-14b-base-af98c8/` |
| 3.60 TB | `gs://marin-us-central1/checkpoints/code-v3-14b-p2-wd0p1_wu0p0-qwen3-14b-base-325c0b/` |
| 3.60 TB | `gs://marin-us-central1/checkpoints/code-v3-sweep-lr5e-6_bs32-qwen3-14b-base-454db1/` |
| 1.83 TB | `gs://marin-us-central1/checkpoints/code-v3-sweep-lr1e-6_bs64-qwen3-14b-base-5426fd/` |
| 1.83 TB | `gs://marin-us-central1/checkpoints/code-v3-sweep-lr2e-6_bs64-qwen3-14b-base-8dd617/` |

## Stub / failed runs (<1 MB) — 29 checkpoints

These are likely failed launches that wrote only metadata. Safe to investigate / delete.

| Size | Path |
|---:|---|
| 524,270 B | `gs://marin-us-central1/checkpoints/code-resiliparse-qwen3-0.6b-sft-c97c7a/` |
| 224,998 B | `gs://marin-us-central1/checkpoints/mathhelpforum-resiliparse-sft-1e20-qwen3-56a96f/` |
| 194,694 B | `gs://marin-us-central1/checkpoints/code-extract-general-qwen3-0.6b-sft-b2603e/` |
| 169,518 B | `gs://marin-us-central1/checkpoints/mathhelpforum-qra-sft-1e20-qwen3-c06b54/` |
| 154,950 B | `gs://marin-us-central1/checkpoints/mathhelpforum-qra-plaintext-sft-1e20-qwen3-8b0030/` |
| 151,307 B | `gs://marin-us-central1/checkpoints/mathhelpforum-sft-1e20-qwen3-v2-7c56cb/` |
| 106,798 B | `gs://marin-us-central1/checkpoints/mathhelpforum-sft-1e20-qwen3-8bd61a/` |
| 28,492 B | `gs://marin-us-central1/checkpoints/code-v3-sweep-lr3e-4_bs128-qwen3-0.6b-base-67da1f/` |
| 28,492 B | `gs://marin-us-central1/checkpoints/code-v3-sweep-lr1e-4_bs128-qwen3-0.6b-base-ae4129/` |
| 28,491 B | `gs://marin-us-central1/checkpoints/code-v3-sweep-lr5e-6_bs128-qwen3-0.6b-base-111ce5/` |
| 28,491 B | `gs://marin-us-central1/checkpoints/code-v3-sweep-lr5e-5_bs128-qwen3-0.6b-base-ec9a96/` |
| 28,491 B | `gs://marin-us-central1/checkpoints/code-v3-sweep-lr3e-5_bs128-qwen3-0.6b-base-fee229/` |
| 28,491 B | `gs://marin-us-central1/checkpoints/code-v3-sweep-lr1e-5_bs128-qwen3-0.6b-base-f22a55/` |
| 28,461 B | `gs://marin-us-central1/checkpoints/code-extract-commented-qwen3-0.6b-base-sft-1e177b/` |
| 25,446 B | `gs://marin-us-central1/checkpoints/mathhelpforum-sft-1e20-qwen3-b049cb/` |
| 24,377 B | `gs://marin-us-central1/checkpoints/math-v2-sweep-lr5e-7_bs128-qwen3-0.6b-base-138ea0/` |
| 24,377 B | `gs://marin-us-central1/checkpoints/math-v2-sweep-lr5e-6_bs128-qwen3-0.6b-base-1e07c4/` |
| 24,377 B | `gs://marin-us-central1/checkpoints/math-v2-sweep-lr2e-6_bs128-qwen3-0.6b-base-a16374/` |
| 24,377 B | `gs://marin-us-central1/checkpoints/math-v2-sweep-lr2e-5_bs128-qwen3-0.6b-base-2ed112/` |
| 24,377 B | `gs://marin-us-central1/checkpoints/math-v2-sweep-lr1e-6_bs128-qwen3-0.6b-base-bd0074/` |
| 24,377 B | `gs://marin-us-central1/checkpoints/math-v2-sweep-lr1e-5_bs128-qwen3-0.6b-base-c90ed2/` |
| 21,375 B | `gs://marin-us-central1/checkpoints/mathhelpforum-qra-sft-1e20-qwen3-a94f98/` |
| 17,351 B | `gs://marin-us-central1/checkpoints/medical-extract-starter-qwen3-0.6b-sft-afedcf/` |
| 17,332 B | `gs://marin-us-central1/checkpoints/medical-extract-highreg-qwen3-0.6b-e008ec/` |
| 17,328 B | `gs://marin-us-central1/checkpoints/medical-extract-lowreg-qwen3-0.6b-688e2e/` |
| 11,561 B | `gs://marin-us-central1/checkpoints/mathhelpforum-resiliparse-sft-1e20-qwen3-35007c/` |
| 10,676 B | `gs://marin-us-central1/checkpoints/mathhelpforum-extract-general-qwen3-0.6b-base-sft-5e1316/` |
| 10,560 B | `gs://marin-us-central1/checkpoints/mathhelpforum-extract-qra-qwen3-0.6b-base-sft-e5ee40/` |
| 8,007 B | `gs://marin-us-central1/checkpoints/mathhelpforum-resiliparse-qwen3-0.6b-base-sft-56f4c4/` |
