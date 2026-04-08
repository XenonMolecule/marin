#!/bin/bash
# Transfer 8B rephraser checkpoints to Stanford cluster once they finish.
# Polls GCS for step-903 HF exports, then downloads and rsyncs.
#
# Usage: bash scripts/transfer_8b_to_stanford.sh

set -euo pipefail

STANFORD_HOST="mryan0@scdt.stanford.edu"
STANFORD_BASE="/nlp/scr2/nlp/personal-rm/small-rephraser/small-rephraser/models"
LOCAL_BASE="$HOME/models"

NAMES=(
  "qwen3-8b-rephraser-kimi-think-sft"
  "qwen3-8b-rephraser-kimi-stripped-sft"
)

GCS_PATHS=(
  "gs://marin-us-central1/checkpoints/qwen3-8b-rephraser-kimi-think-sft-740315/hf/step-903/"
  "gs://marin-us-central1/checkpoints/qwen3-8b-rephraser-kimi-stripped-sft-aa28a4/hf/step-903/"
)

# Wait for both checkpoints to appear
echo "Waiting for 8B checkpoints to finish..."
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  gcs_path="${GCS_PATHS[$i]}"
  echo "Polling for $name at $gcs_path"
  while ! gcloud storage ls "${gcs_path}config.json" &>/dev/null; do
    echo "  $(date +%H:%M:%S) - $name not ready yet, checking again in 60s..."
    sleep 60
  done
  echo "  $name is ready!"
done

echo "Both checkpoints ready! Starting transfer..."

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  gcs_path="${GCS_PATHS[$i]}"
  local_dir="$LOCAL_BASE/$name"

  echo "========================================"
  echo "Transferring: $name"
  echo "========================================"

  # 1. Download from GCS
  mkdir -p "$local_dir"
  gcloud storage cp -r "${gcs_path}*" "$local_dir/"

  # 2. Rsync to Stanford
  ssh "$STANFORD_HOST" "mkdir -p $STANFORD_BASE/$name"
  rsync -avP "$local_dir/" "$STANFORD_HOST:$STANFORD_BASE/$name/"

  echo "Done: $name"
  echo
done

echo "All 8B checkpoints transferred!"
echo
echo "Serve with:"
echo "  # Think:"
echo "  vllm serve $STANFORD_BASE/qwen3-8b-rephraser-kimi-think-sft/ --reasoning-parser qwen3"
echo "  # Stripped:"
echo "  vllm serve $STANFORD_BASE/qwen3-8b-rephraser-kimi-stripped-sft/ --chat-template $STANFORD_BASE/qwen3_no_think_serving.jinja"
