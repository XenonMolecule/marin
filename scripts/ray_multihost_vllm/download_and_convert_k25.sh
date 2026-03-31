#!/bin/bash
# All-in-one: Download K2.5 INT4 from HuggingFace, convert to FP8, upload to GCS.
# Designed to run unattended on a TPU VM or GCE instance.
#
# Usage: nohup bash download_and_convert_k25.sh <GCS_BUCKET> > /tmp/k25_pipeline.log 2>&1 &
# Example: nohup bash download_and_convert_k25.sh marin-eu-west4 > /tmp/k25_pipeline.log 2>&1 &

set -euo pipefail

BUCKET="${1:?Usage: download_and_convert_k25.sh <GCS_BUCKET>}"
INT4_DEST="gs://${BUCKET}/models/unsloth--Kimi-K2.5"
FP8_DEST="gs://${BUCKET}/models/kimi-k25-fp8"
TEMP_DIR="/tmp/k25_pipeline"

echo "============================================"
echo "K2.5 Pipeline: Download INT4 -> Convert FP8"
echo "Bucket: $BUCKET"
echo "INT4 dest: $INT4_DEST"
echo "FP8 dest: $FP8_DEST"
echo "Started: $(date)"
echo "============================================"

pip install -q huggingface_hub safetensors torch numpy 2>/dev/null

mkdir -p "$TEMP_DIR"

# Step 1: Download INT4 model from HuggingFace
echo ""
echo "=== Step 1: Download unsloth/Kimi-K2.5 INT4 ==="
python3 -c "
import subprocess, os
from huggingface_hub import hf_hub_download

repo = 'unsloth/Kimi-K2.5'
gcs_dest = '${INT4_DEST}'
local_dir = '${TEMP_DIR}/int4'

files = []
for i in range(1, 65):
    files.append(f'model-{i:05d}-of-000064.safetensors')
files.extend([
    'config.json', 'configuration_deepseek.py', 'configuration_kimi_k25.py',
    'generation_config.json', 'kimi_k25_processor.py', 'kimi_k25_vision_processing.py',
    'media_utils.py', 'model.safetensors.index.json', 'modeling_deepseek.py',
    'modeling_kimi_k25.py', 'preprocessor_config.json', 'tokenization_kimi.py',
    'tokenizer_config.json', 'tool_declaration_ts.py',
    'tiktoken.model', 'special_tokens_map.json',
])

print(f'Downloading {len(files)} files', flush=True)

for i, fname in enumerate(files):
    check = subprocess.run(['gcloud', 'storage', 'ls', f'{gcs_dest}/{fname}'], capture_output=True)
    if check.returncode == 0:
        print(f'[{i+1}/{len(files)}] SKIP: {fname}', flush=True)
        continue
    print(f'[{i+1}/{len(files)}] Downloading: {fname}', flush=True)
    try:
        local_path = hf_hub_download(repo, fname, local_dir=local_dir)
        size_mb = os.path.getsize(local_path) / 1e6
        print(f'  Downloaded ({size_mb:.0f} MB), uploading...', flush=True)
        subprocess.run(['gcloud', 'storage', 'cp', local_path, f'{gcs_dest}/{fname}'], check=True)
        os.remove(local_path)
        print(f'  Done: {fname}', flush=True)
    except Exception as e:
        print(f'  ERROR: {e}', flush=True)

print('INT4 download complete!', flush=True)
"

# Step 2: Convert INT4 -> FP8
echo ""
echo "=== Step 2: Convert INT4 -> FP8 ==="
python3 /tmp/convert_k25_int4_to_fp8.py \
  --input "$INT4_DEST" \
  --output "$FP8_DEST" \
  --temp-dir "$TEMP_DIR/convert"

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "INT4 model: $INT4_DEST"
echo "FP8 model: $FP8_DEST"
echo "Finished: $(date)"
echo "============================================"
