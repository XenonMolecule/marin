# hf/ Exports Pending Final Deletion

Initial test batch (2 ckpts) on 2026-05-13, then full 14B sweep cleanup later
that same day. For each, the inner `checkpoints/` (training state) was deleted
and the `hf/` HF export was kept for possible eval/SFT use.

Total pending: **23 × ~59.09 GB = ~1.36 TB**

A reminder is scheduled to fire on **2026-06-01 09:00 PT** via routine
`trig_01Xb1YBbzsxaum3uSg7CyuAF` (https://claude.ai/code/routines/trig_01Xb1YBbzsxaum3uSg7CyuAF).

## Paths to delete (23 total)

```
gs://marin-us-central1/checkpoints/medical-14b-extract-default-qwen3-14b-base-64cc1b/hf/
gs://marin-us-central1/checkpoints/code-v3-14b-p2-wd0p001_wu0p03-qwen3-14b-base-cb4fb3/hf/
gs://marin-us-central1/checkpoints/code-v3-14b-p2-wd0p001_wu0p0-qwen3-14b-base-ade815/hf/
gs://marin-us-central1/checkpoints/code-v3-14b-p2-wd0p001_wu0p1-qwen3-14b-base-9a6428/hf/
gs://marin-us-central1/checkpoints/code-v3-14b-p2-wd0p01_wu0p0-qwen3-14b-base-90dbb3/hf/
gs://marin-us-central1/checkpoints/code-v3-14b-p2-wd0p01_wu0p03-qwen3-14b-base-67dc11/hf/
gs://marin-us-central1/checkpoints/code-v3-14b-p2-wd0p01_wu0p1-qwen3-14b-base-af98c8/hf/
gs://marin-us-central1/checkpoints/code-v3-14b-p2-wd0p05_wu0p0-qwen3-14b-base-799263/hf/
gs://marin-us-central1/checkpoints/code-v3-14b-p2-wd0p05_wu0p03-qwen3-14b-base-a090fe/hf/
gs://marin-us-central1/checkpoints/code-v3-14b-p2-wd0p05_wu0p1-qwen3-14b-base-00b606/hf/
gs://marin-us-central1/checkpoints/code-v3-14b-p2-wd0p1_wu0p0-qwen3-14b-base-325c0b/hf/
gs://marin-us-central1/checkpoints/code-v3-14b-p2-wd0p1_wu0p03-qwen3-14b-base-a3b0ef/hf/
gs://marin-us-central1/checkpoints/code-v3-14b-p2-wd0p1_wu0p1-qwen3-14b-base-067429/hf/
gs://marin-us-central1/checkpoints/code-v3-sweep-lr1e-5_bs32-qwen3-14b-base-3161d5/hf/
gs://marin-us-central1/checkpoints/code-v3-sweep-lr1e-5_bs64-qwen3-14b-base-a3548e/hf/
gs://marin-us-central1/checkpoints/code-v3-sweep-lr1e-6_bs32-qwen3-14b-base-69a784/hf/
gs://marin-us-central1/checkpoints/code-v3-sweep-lr1e-6_bs64-qwen3-14b-base-5426fd/hf/
gs://marin-us-central1/checkpoints/code-v3-sweep-lr2e-6_bs32-qwen3-14b-base-f6fc2f/hf/
gs://marin-us-central1/checkpoints/code-v3-sweep-lr2e-6_bs64-qwen3-14b-base-8dd617/hf/
gs://marin-us-central1/checkpoints/code-v3-sweep-lr5e-5_bs32-qwen3-14b-base-a5a415/hf/
gs://marin-us-central1/checkpoints/code-v3-sweep-lr5e-5_bs64-qwen3-14b-base-733190/hf/
gs://marin-us-central1/checkpoints/code-v3-sweep-lr5e-6_bs32-qwen3-14b-base-454db1/hf/
gs://marin-us-central1/checkpoints/code-v3-sweep-lr5e-6_bs64-qwen3-14b-base-bbb80c/hf/
```

## Deletion command (when ready)

```bash
# Bulk delete all 23 in parallel
gcloud storage rm -r --quiet $(cat .agents/scratchpad/ckpt_inventory/HF_DELETE_LATER.md \
  | grep -oE 'gs://[^[:space:]`]+/hf/' | sort -u | tr '\n' ' ')
```
