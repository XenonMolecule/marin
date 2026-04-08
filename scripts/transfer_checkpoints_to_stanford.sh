#!/bin/bash
# Transfer trained rephraser checkpoints from GCS to Stanford cluster.
# Run from a machine with gcloud auth and SSH access to scdt.stanford.edu.
#
# Usage: bash scripts/transfer_checkpoints_to_stanford.sh

set -euo pipefail

STANFORD_HOST="mryan0@scdt.stanford.edu"
STANFORD_BASE="/nlp/scr2/nlp/personal-rm/small-rephraser/small-rephraser/models"
LOCAL_BASE="$HOME/models"

NAMES=(
  "qwen3-0.6b-rephraser-kimi-think-sft"
  "qwen3-0.6b-rephraser-kimi-stripped-sft"
  "qwen3-1.7b-rephraser-kimi-think-sft"
  "qwen3-1.7b-rephraser-kimi-stripped-sft"
)

GCS_PATHS=(
  "gs://marin-us-central1/checkpoints/qwen3-0.6b-rephraser-kimi-think-sft-f7321d/hf/step-903/"
  "gs://marin-us-central1/checkpoints/qwen3-0.6b-rephraser-kimi-stripped-sft-2a6763/hf/step-903/"
  "gs://marin-us-central1/checkpoints/qwen3-1.7b-rephraser-kimi-think-sft-777825/hf/step-903/"
  "gs://marin-us-central1/checkpoints/qwen3-1.7b-rephraser-kimi-stripped-sft-5b3958/hf/step-903/"
)

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  gcs_path="${GCS_PATHS[$i]}"
  local_dir="$LOCAL_BASE/$name"

  echo "========================================"
  echo "Transferring: $name"
  echo "  GCS:      $gcs_path"
  echo "  Local:    $local_dir"
  echo "  Stanford: $STANFORD_HOST:$STANFORD_BASE/$name/"
  echo "========================================"

  # 1. Download from GCS
  mkdir -p "$local_dir"
  gcloud storage cp -r "${gcs_path}*" "$local_dir/"

  # 2. Rsync to Stanford (create remote dir first)
  ssh "$STANFORD_HOST" "mkdir -p $STANFORD_BASE/$name"
  rsync -avP "$local_dir/" "$STANFORD_HOST:$STANFORD_BASE/$name/"

  echo "Done: $name"
  echo
done

echo "All checkpoints transferred!"
