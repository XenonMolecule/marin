#!/bin/bash
# Download a HuggingFace model to GCS one file at a time.
# Designed for machines with limited disk - downloads each file to /tmp,
# uploads to GCS, then deletes the local copy.
#
# Usage: bash download_to_gcs.sh <HF_REPO> <GCS_DEST> [HF_TOKEN]
# Example: bash download_to_gcs.sh moonshotai/Kimi-K2-Instruct gs://marin-us-east5/models/moonshotai--Kimi-K2-Instruct

set -euo pipefail

HF_REPO="${1:?Usage: download_to_gcs.sh <HF_REPO> <GCS_DEST> [HF_TOKEN]}"
GCS_DEST="${2:?}"
HF_TOKEN="${3:-${HF_TOKEN:-}}"

LOCAL_TMP="/tmp/hf_download_staging"
mkdir -p "$LOCAL_TMP"

echo "============================================"
echo "HF -> GCS streaming download"
echo "Repo:  $HF_REPO"
echo "Dest:  $GCS_DEST"
echo "============================================"

# Install huggingface_hub if needed
pip install -q huggingface_hub 2>/dev/null || pip3 install -q huggingface_hub 2>/dev/null || true

# Get list of files from HF API
AUTH_HEADER=""
if [ -n "$HF_TOKEN" ]; then
  AUTH_HEADER="Authorization: Bearer $HF_TOKEN"
fi

echo "Fetching file list..."
FILE_LIST=$(python3 -c "
from huggingface_hub import list_repo_files
files = list_repo_files('$HF_REPO', token='$HF_TOKEN' if '$HF_TOKEN' else None)
# Include safetensors, json, py, model, tokenizer files
for f in sorted(files):
    ext = f.rsplit('.', 1)[-1] if '.' in f else ''
    if ext in ('safetensors', 'json', 'py', 'model', 'txt') or 'tokenizer' in f or f == 'config.json':
        print(f)
")

TOTAL=$(echo "$FILE_LIST" | wc -l | tr -d ' ')
echo "Found $TOTAL files to download"

COUNT=0
for FILE in $FILE_LIST; do
  COUNT=$((COUNT + 1))

  # Check if already uploaded
  if gcloud storage ls "${GCS_DEST}/${FILE}" >/dev/null 2>&1; then
    echo "[$COUNT/$TOTAL] SKIP (exists): $FILE"
    continue
  fi

  echo "[$COUNT/$TOTAL] Downloading: $FILE"

  # Download from HF
  python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('$HF_REPO', '$FILE', local_dir='$LOCAL_TMP', token='$HF_TOKEN' if '$HF_TOKEN' else None)
" 2>/dev/null

  if [ ! -f "$LOCAL_TMP/$FILE" ]; then
    echo "  ERROR: download failed for $FILE"
    continue
  fi

  SIZE=$(du -h "$LOCAL_TMP/$FILE" | cut -f1)
  echo "  Downloaded ($SIZE), uploading to GCS..."

  # Upload to GCS
  gcloud storage cp "$LOCAL_TMP/$FILE" "${GCS_DEST}/${FILE}" 2>/dev/null

  # Clean up local copy
  rm -f "$LOCAL_TMP/$FILE"
  echo "  Done: $FILE"
done

# Clean up any leftover cache
rm -rf "$LOCAL_TMP"
rm -rf /tmp/.cache/huggingface 2>/dev/null || true

echo "============================================"
echo "Download complete! $COUNT files processed."
echo "Verify: gcloud storage ls $GCS_DEST/"
echo "============================================"
