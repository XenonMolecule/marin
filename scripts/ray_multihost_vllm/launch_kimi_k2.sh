#!/bin/bash
# Launch Kimi K2 Instruct (1T MoE, DeepseekV3ForCausalLM) on TPU v5p-32.
#
# Prerequisites:
#   1. Weights staged: gs://marin-us-east5/models/moonshotai--Kimi-K2-Instruct/
#   2. TPU allocated: v5p-32 in us-east5-a
#
# Usage:
#   bash launch_kimi_k2.sh <TPU_NAME> [MAX_MODEL_LEN]
#
# Example:
#   bash launch_kimi_k2.sh kimi-k2-v5p32 4096    # smoke test
#   bash launch_kimi_k2.sh kimi-k2-v5p32 131072  # full context

set -euo pipefail

TPU_NAME="${1:?Usage: launch_kimi_k2.sh <TPU_NAME> [MAX_MODEL_LEN]}"
MAX_MODEL_LEN="${2:-4096}"
ZONE="us-east5-a"
MODEL_GCS_PATH="gs://marin-us-east5/models/moonshotai--Kimi-K2-Instruct"

# v5p-32 = 4 hosts × 4 chips = PP=4, TP=4
PP_SIZE=4
TP_SIZE=4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "Kimi K2 Instruct (1T MoE) on v5p-32"
echo "PP=$PP_SIZE TP=$TP_SIZE max_model_len=$MAX_MODEL_LEN"
echo "============================================"

# Step 0: Fix model_type if needed.
# vLLM may not recognize "kimi_k2" model_type. We patch it to "deepseek_v3"
# after gcsfuse mounts but before vLLM loads. The architectures field
# ("DeepseekV3ForCausalLM") is what vLLM actually uses for model lookup.
echo "Step 0: Checking if config.json needs model_type patch..."
NEEDS_PATCH=$(gcloud storage cat "${MODEL_GCS_PATH}/config.json" 2>/dev/null | python3 -c "
import json, sys
c = json.load(sys.stdin)
print('yes' if c.get('model_type') == 'kimi_k2' else 'no')
")

if [ "$NEEDS_PATCH" = "yes" ]; then
  echo "  Patching model_type: kimi_k2 -> deepseek_v3"
  gcloud storage cat "${MODEL_GCS_PATH}/config.json" | python3 -c "
import json, sys
c = json.load(sys.stdin)
c['model_type'] = 'deepseek_v3'
json.dump(c, sys.stdout, indent=2)
" | gcloud storage cp - "${MODEL_GCS_PATH}/config.json"
  echo "  Done."
else
  echo "  model_type already correct, skipping."
fi

# Delegate to main launch script
exec bash "$SCRIPT_DIR/launch.sh" "$TPU_NAME" "$ZONE" "$MODEL_GCS_PATH" "$PP_SIZE" "$TP_SIZE" "$MAX_MODEL_LEN"
