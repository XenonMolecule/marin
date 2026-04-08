#!/bin/bash
# Transfer 8B v2 stripped low-LR rephraser checkpoint to Stanford cluster.
# Run from a machine with gcloud auth and SSH access to scdt.stanford.edu.
#
# Usage: bash scripts/transfer_8b_v2_stripped_to_stanford.sh

set -euo pipefail

STANFORD_HOST="mryan0@scdt.stanford.edu"
STANFORD_BASE="/nlp/scr2/nlp/personal-rm/small-rephraser/small-rephraser/models"
LOCAL_BASE="$HOME/models"

NAME="qwen3-8b-rephraser-kimi-v2-stripped-sft-low-lr"
GCS_PATH="gs://marin-us-central1/checkpoints/qwen3-8b-rephraser-kimi-v2-stripped-sft-low-lr-d3514a/hf/step-1213/"

LOCAL_DIR="$LOCAL_BASE/$NAME"

echo "========================================"
echo "Transferring: $NAME"
echo "  GCS:      $GCS_PATH"
echo "  Local:    $LOCAL_DIR"
echo "  Stanford: $STANFORD_HOST:$STANFORD_BASE/$NAME/"
echo "========================================"

# 1. Download from GCS
mkdir -p "$LOCAL_DIR"
gcloud storage cp -r "${GCS_PATH}*" "$LOCAL_DIR/"

# 2. Rsync to Stanford
ssh "$STANFORD_HOST" "mkdir -p $STANFORD_BASE/$NAME"
rsync -avP "$LOCAL_DIR/" "$STANFORD_HOST:$STANFORD_BASE/$NAME/"

echo "Done: $NAME"
echo
echo "Local copy at: $LOCAL_DIR"
echo "  (safe to delete after verifying the Stanford transfer)"
echo
echo "Serve with:"
echo "  vllm serve $STANFORD_BASE/$NAME/ --reasoning-parser qwen3"
