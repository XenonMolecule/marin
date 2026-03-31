#!/bin/bash
# =============================================================================
# Launch Kimi K2-Instruct or K2.5 (1T MoE, DeepseekV3) on TPU v5p-32+ via Ray PP.
#
# Uses latest vLLM-TPU nightly which has native DeepSeek V3 PP support in
# the model code. Only 2 patches needed:
#   1. Remove DeepseekV3 from _PP_DISABLED_MODELS (still gated in nightly)
#   2. Fix hardcoded model params (vs DeepSeek V3 original)
#
# Key: NEW_MODEL_DESIGN=True enables 5D mesh that DeepSeek V3 model requires.
#
# Prerequisites:
#   - TPU allocated and ACTIVE (v5p-32 minimum for FP8 weights)
#   - Weights staged in same-region GCS
#
# Usage:
#   bash launch_kimi_k2_v2.sh <TPU_NAME> [ZONE] [MAX_MODEL_LEN] [MODEL]
#
# Examples:
#   bash launch_kimi_k2_v2.sh kimi-k2-v5p32                           # K2-Instruct, us-central1-a
#   bash launch_kimi_k2_v2.sh kimi-k25-v5p64 us-east5-a 32768 k25    # K2.5
# =============================================================================

set -euo pipefail

TPU_NAME="${1:?Usage: launch_kimi_k2_v2.sh <TPU_NAME> [ZONE] [MAX_MODEL_LEN] [MODEL: k2|k25]}"
ZONE="${2:-us-central1-a}"
MAX_MODEL_LEN="${3:-4096}"
MODEL="${4:-k2}"  # k2 = K2-Instruct, k25 = K2.5
PROJECT="${PROJECT:-hai-gcp-models}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
PORT="${PORT:-8000}"

# Model weights — pick same-region bucket based on model variant
if [ "$MODEL" = "k25" ]; then
  MODEL_NAME="kimi-k25-fp8"
  CONFIG_PATCH="patch_deepseek_v3_config_k25.py"
  case "$ZONE" in
    us-central1*) MODEL_GCS_PATH="gs://marin-us-central1/models/kimi-k25-fp8" ;;
    *)            echo "ERROR: K2.5 FP8 weights only in us-central1. Stage weights first."; exit 1 ;;
  esac
else
  MODEL_NAME="moonshotai--Kimi-K2-Instruct"
  CONFIG_PATCH="patch_deepseek_v3_config.py"
  case "$ZONE" in
    us-central1*) MODEL_GCS_PATH="gs://marin-us-central1/models/moonshotai--Kimi-K2-Instruct" ;;
    us-east5*)    MODEL_GCS_PATH="gs://marin-us-east5/models/moonshotai--Kimi-K2-Instruct" ;;
    *)            echo "ERROR: No staged weights for zone $ZONE. Stage weights first."; exit 1 ;;
  esac
fi

GCS_BUCKET=$(echo "$MODEL_GCS_PATH" | sed 's|gs://||' | cut -d/ -f1)
GCS_SUBPATH=$(echo "$MODEL_GCS_PATH" | sed "s|gs://${GCS_BUCKET}/||")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="$SCRIPT_DIR/patches"

# Get number of workers (= PP size)
NUM_WORKERS=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT" \
  --format="value(networkEndpoints)" 2>/dev/null | tr ';' '\n' | wc -l)
NUM_WORKERS=$(echo "$NUM_WORKERS" | tr -d '[:space:]')

# v5p-X: each host has 4 chips, TP=4
PP_SIZE="$NUM_WORKERS"
TP_SIZE=4

echo "============================================"
echo "Kimi ${MODEL} (1T MoE) — v2 launcher"
echo "============================================"
echo "TPU:           $TPU_NAME ($ZONE)"
echo "Model:         $MODEL_GCS_PATH"
echo "PP=$PP_SIZE TP=$TP_SIZE max_model_len=$MAX_MODEL_LEN"
echo "Workers:       $NUM_WORKERS"
echo "============================================"

# Get head IP
HEAD_IP=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT" --worker=0 \
  --command="hostname -I | awk '{print \$1}'" 2>/dev/null)
echo "Head IP: $HEAD_IP"

# ---------- Step 0: Fix model_type if needed ----------
echo ""
echo "Step 0: Checking config.json model_type..."
NEEDS_PATCH=$(gcloud storage cat "${MODEL_GCS_PATH}/config.json" 2>/dev/null | python3 -c "
import json, sys
c = json.load(sys.stdin)
print('yes' if c.get('model_type') == 'kimi_k2' else 'no')
" 2>/dev/null || echo "no")

if [ "$NEEDS_PATCH" = "yes" ]; then
  echo "  Patching model_type: kimi_k2 -> deepseek_v3"
  gcloud storage cat "${MODEL_GCS_PATH}/config.json" | python3 -c "
import json, sys
c = json.load(sys.stdin)
c['model_type'] = 'deepseek_v3'
json.dump(c, sys.stdout, indent=2)
" | gcloud storage cp - "${MODEL_GCS_PATH}/config.json"
fi

# ---------- Step 1: Setup gcsfuse ----------
echo ""
echo "Step 1: Setup gcsfuse on all hosts..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT" --worker=all \
  --command="
    which gcsfuse >/dev/null 2>&1 || {
      export GCSFUSE_REPO=gcsfuse-\$(lsb_release -c -s)
      echo \"deb https://packages.cloud.google.com/apt \$GCSFUSE_REPO main\" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
      curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
      sudo apt-get update -qq && sudo apt-get install -y -qq gcsfuse
    }
    sudo umount /mnt/gcs-models 2>/dev/null || true
    sudo mkdir -p /mnt/gcs-models
    sudo gcsfuse --implicit-dirs -o allow_other \
      --only-dir '$GCS_SUBPATH' \
      --file-cache-max-size-mb 0 \
      '$GCS_BUCKET' /mnt/gcs-models 2>&1 | grep -E 'mounted|Error'
    ls /mnt/gcs-models/config.json >/dev/null 2>&1 && echo 'MODEL OK' || echo 'MODEL FAIL'
  " 2>/dev/null

# ---------- Step 2: Docker + Ray ----------
echo ""
echo "Step 2: Start Docker containers + Ray cluster..."

# Stop Iris worker (if present) to free TPU, then kill old ray-node containers
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT" --worker=all \
  --command="sudo docker rm -f ray-node 2>/dev/null; echo done" 2>/dev/null

# Docker env vars — critical settings:
#   NEW_MODEL_DESIGN=True         -> 5D mesh for DeepSeek V3 model
#   RAY_memory_monitor_refresh_ms -> prevent Ray OOM killer during weight loading
#   PROCESS_WEIGHTS_ON_TPU        -> run FP8 dequant→requant on TPU instead of CPU (1000x faster)
DOCKER_ENV="-e TPU_MULTIHOST_BACKEND=ray -e TPU_BACKEND_TYPE=jax -e RAY_DEDUP_LOGS=0 -e NEW_MODEL_DESIGN=True -e RAY_memory_monitor_refresh_ms=0 -e PROCESS_WEIGHTS_ON_TPU=1"
DOCKER_VOLS="-v /dev/shm:/dev/shm -v /mnt/gcs-models:/mnt/gcs-models:ro"

# Start head
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT" --worker=0 \
  --command="sudo docker run -d --name ray-node --privileged --net=host --shm-size=16g \
    $DOCKER_ENV $DOCKER_VOLS \
    vllm/vllm-tpu:nightly \
    bash -c \"ray start --head --port=6379 && sleep infinity\"'" 2>/dev/null
sleep 5

# Start workers
for w in $(seq 1 $((NUM_WORKERS - 1))); do
  gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" --project="$PROJECT" --worker="$w" \
    --command="sudo docker run -d --name ray-node --privileged --net=host --shm-size=16g \
      $DOCKER_ENV $DOCKER_VOLS \
      vllm/vllm-tpu:nightly \
      bash -c \"ray start --address=$HEAD_IP:6379 --block\"'" 2>/dev/null &
done
wait
sleep 10

# Verify Ray
echo "Ray cluster status:"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT" --worker=0 \
  --command="sudo docker exec ray-node ray status 2>/dev/null | grep -E \"Active:|TPU\"'" 2>/dev/null

# ---------- Step 3: Apply MINIMAL patches ----------
echo ""
echo "Step 3: Apply patches (minimal set for nightly)..."

# Patches needed with the nightly:
PATCHES=(
  "patch_pp_enable_deepseek.py"       # Remove DeepseekV3 from _PP_DISABLED_MODELS
  "$CONFIG_PATCH"                     # Fix hardcoded params for K2-Instruct or K2.5
  "patch_moe_process_on_tpu.py"       # Move FP8 dequant→requant to TPU (1000x faster)
  "patch_trace_moe_load.py"           # Debug tracing for progress monitoring
)

for w in $(seq 0 $((NUM_WORKERS - 1))); do
  for p in "${PATCHES[@]}"; do
    gcloud compute tpus tpu-vm scp "$PATCH_DIR/$p" "$TPU_NAME:/tmp/$p" \
      --zone="$ZONE" --project="$PROJECT" --worker="$w" 2>/dev/null
  done &
done
wait

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT" --worker=all \
  --command="
    for p in ${PATCHES[*]}; do
      sudo docker cp /tmp/\$p ray-node:/tmp/
      sudo docker exec ray-node python /tmp/\$p 2>&1 | tail -1
    done
  " 2>/dev/null

echo "Patches applied."

# ---------- Step 4: Launch vLLM ----------
echo ""
echo "Step 4: Launch vLLM serve..."
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --zone="$ZONE" --project="$PROJECT" --worker=0 \
  --command="sudo docker exec -d ray-node bash -c 'vllm serve /mnt/gcs-models \
    --tensor-parallel-size $TP_SIZE \
    --pipeline-parallel-size $PP_SIZE \
    --distributed-executor-backend ray \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --port $PORT \
    --trust-remote-code \
    > /tmp/vllm_serve.log 2>&1'" 2>/dev/null

echo ""
echo "============================================"
echo "vLLM serve launched!"
echo "============================================"
echo ""
echo "Tail logs:"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 \\"
echo "    --command=\"sudo docker exec ray-node tail -f /tmp/vllm_serve.log'\""
echo ""
echo "Check health:"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 \\"
echo "    --command=\"sudo docker exec ray-node curl -s http://localhost:$PORT/health'\""
echo ""
echo "Check HBM per chip (verify PP distributes layers):"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=all \\"
echo "    --command=\"sudo docker exec ray-node python3 -c \\\"import jax; [print(f\\\\\\\"{d}: {d.memory_stats()[chr(98)+chr(121)+chr(116)+chr(101)+chr(115)+chr(95)+chr(105)+chr(110)+chr(95)+chr(117)+chr(115)+chr(101)]/1e9:.2f}GB\\\\\\\") for d in jax.local_devices()[:4]]\\\"'\""
echo ""
echo "Send test request:"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --project=$PROJECT --worker=0 \\"
echo "    --command=\"sudo docker exec ray-node curl -s http://localhost:$PORT/v1/chat/completions \\"
echo "      -H Content-Type:application/json \\"
echo "      -d \\'{\\\"model\\\":\\\"/mnt/gcs-models\\\",\\\"messages\\\":[{\\\"role\\\":\\\"user\\\",\\\"content\\\":\\\"What is 2+2?\\\"}],\\\"max_tokens\\\":32}\\''\""
echo ""
echo "Weight loading takes 30-60 min (first load). Subsequent: seconds."
echo "============================================"
