#!/usr/bin/env bash
# Promote the deduped trees from marin-tmp-* (TTL=14d) buckets to permanent
# regional buckets, then relaunch tokenization.
#
# Required because `default_tokenize` -> marin executor refuses to infer a
# region from `marin-tmp-{region}` buckets — it expects `marin-{region}`.
#
# Same-region copies are free egress. Wall-clock ~30-60 min per source in
# parallel.
#
# Usage:
#   bash experiments/baseline_collection/dedup/promote_and_tokenize.sh
#
# Or run the parts individually below.

set -euo pipefail

LLM_CURATED_SRC="gs://marin-tmp-us-central1/ttl=14d/michaelryan/dedup_llm_curated/deduped_d44711c0"
LLM_CURATED_DST="gs://marin-us-central1/documents/baseline_llm_curated_deduped"

RESILIPARSE_SRC="gs://marin-tmp-us-central2/ttl=14d/michaelryan/dedup_resiliparse/deduped_74e77a83"
RESILIPARSE_DST="gs://marin-us-central2/documents/baseline_resiliparse_deduped"

WANDB_API_KEY="<YOUR_WANDB_API_KEY>"
HF_TOKEN="<YOUR_HF_TOKEN>"

echo "==> Step 1: same-region GCS copies (free egress, ~30-60 min in parallel)"
gcloud storage cp -r "${LLM_CURATED_SRC}/data-*.jsonl.gz" "${LLM_CURATED_DST}/" &
PID_LLM=$!
gcloud storage cp -r "${RESILIPARSE_SRC}/data-*.jsonl.gz" "${RESILIPARSE_DST}/" &
PID_RES=$!
wait $PID_LLM $PID_RES
echo "==> Copies complete"

echo "==> Step 2: update tokenize_deduped.py to point at the permanent paths"
# Sed-replace the source paths in the python file.
sed -i.bak \
  -e "s|${LLM_CURATED_SRC}|${LLM_CURATED_DST}|" \
  -e "s|${RESILIPARSE_SRC}|${RESILIPARSE_DST}|" \
  experiments/baseline_collection/dedup/tokenize_deduped.py
echo "==> tokenize_deduped.py updated (backup at .bak)"

echo "==> Step 3: launch tokenization on both regions"
uv run iris --cluster marin job run \
  --cpu 2 --memory 8GB --enable-extra-resources \
  --region us-central1 \
  --no-wait \
  --job-name tokenize-llm-curated-deduped-v2 \
  -e WANDB_API_KEY "${WANDB_API_KEY}" \
  -e HF_TOKEN "${HF_TOKEN}" \
  -- python experiments/baseline_collection/dedup/tokenize_deduped.py --only llm_curated

uv run iris --cluster marin job run \
  --cpu 2 --memory 8GB --enable-extra-resources \
  --region us-central2 \
  --no-wait \
  --job-name tokenize-resiliparse-deduped-v2 \
  -e WANDB_API_KEY "${WANDB_API_KEY}" \
  -e HF_TOKEN "${HF_TOKEN}" \
  -- python experiments/baseline_collection/dedup/tokenize_deduped.py --only resiliparse

echo "==> All jobs launched. Monitor with:"
echo "  uv run iris --cluster marin job summary /michaelryan/tokenize-llm-curated-deduped-v2"
echo "  uv run iris --cluster marin job summary /michaelryan/tokenize-resiliparse-deduped-v2"
