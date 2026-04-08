#!/usr/bin/env bash
# Launch production LLM extraction across all available TPU types.
#
# Usage:
#   bash experiments/baseline_collection/launch_production.sh
#
# Each job reads the full 3000-WARC manifest (or a --start/--end range).
# Claims prevent duplicate work. Shuffle seeds spread jobs across the manifest.
# Per-batch checkpointing survives preemption. Cross-region resume works.
#
# Adjust counts below based on available compute. Overprovision freely —
# the autoscaler serves what it can, and preemptible jobs yield gracefully.

set -euo pipefail

IRIS_CONFIG="lib/iris/examples/marin.yaml"
MANIFEST="experiments/distill/baseline_warcs_3000.txt"
OUTPUT_SUBDIR="documents/baseline_llm_extraction"
SCRIPT="experiments/baseline_collection/run_extract_standalone.py"

# Common args for all jobs
COMMON_ARGS="--manifest $MANIFEST --output-subdir $OUTPUT_SUBDIR"
IRIS_COMMON="--memory 128GB --max-retries 10 --extra marin:vllm --extra marin:tpu --no-wait"

# Multi-host env vars (restrict each VM to local 4 chips)
MULTIHOST_ENV='-e TPU_PROCESS_BOUNDS "1,1,1" -e TPU_CHIPS_PER_PROCESS_BOUNDS "2,2,1"'

SEED=0

submit() {
    local tpu=$1
    local name=$2
    local extra_iris_args=${3:-""}
    local extra_script_args=${4:-""}

    echo "Submitting $name (tpu=$tpu, seed=$SEED)"
    uv run iris --config "$IRIS_CONFIG" job run \
        --tpu "$tpu" $IRIS_COMMON \
        --job-name "$name" \
        $extra_iris_args \
        -- python "$SCRIPT" $COMMON_ARGS --shuffle-seed $SEED $extra_script_args \
        2>&1 | grep "Job submitted" || echo "  FAILED to submit $name"

    SEED=$((SEED + 1))
}

echo "=== Launching production extraction ==="
echo "Manifest: $MANIFEST"
echo "Output: $OUTPUT_SUBDIR"
echo ""

# --- Single-host TPUs ---
echo "--- v5p-8 (single-host, 4 chips) ---"
for i in $(seq 1 10); do
    submit "v5p-8" "extract-prod-v5p8-$i"
done

echo "--- v6e-4 (single-host, 4 chips) ---"
for i in $(seq 1 5); do
    submit "v6e-4" "extract-prod-v6e4-$i"
done

echo "--- v6e-8 (single-host, 8 chips) ---"
for i in $(seq 1 5); do
    submit "v6e-8" "extract-prod-v6e8-$i"
done

echo "--- v5litepod-4 (single-host, 4 chips) ---"
for i in $(seq 1 5); do
    submit "v5litepod-4" "extract-prod-v5e4-$i"
done

echo "--- v5litepod-8 (single-host, 8 chips) ---"
for i in $(seq 1 3); do
    submit "v5litepod-8" "extract-prod-v5e8-$i"
done

# --- Multi-host TPUs (data parallel, each VM runs independently) ---
echo "--- v5p-16 (2-host, 2 jobs/slice) ---"
for i in $(seq 1 10); do
    submit "v5p-16" "extract-prod-v5p16-$i" "$MULTIHOST_ENV" "--tp 4"
done

echo "--- v5p-32 (4-host, 4 jobs/slice) ---"
for i in $(seq 1 5); do
    submit "v5p-32" "extract-prod-v5p32-$i" "$MULTIHOST_ENV" "--tp 4"
done

echo "--- v5litepod-16 (4-host, 4 jobs/slice) ---"
for i in $(seq 1 5); do
    submit "v5litepod-16" "extract-prod-v5e16-$i" "$MULTIHOST_ENV" "--tp 4"
done

echo "--- v6e-16 (4-host, 4 jobs/slice) ---"
for i in $(seq 1 5); do
    submit "v6e-16" "extract-prod-v6e16-$i" "$MULTIHOST_ENV" "--tp 4"
done

echo ""
echo "=== Submitted $SEED jobs ==="
echo "Monitor: uv run iris --config $IRIS_CONFIG job list --prefix /michaelryan/extract-prod"
echo "Output:  gcloud storage ls 'gs://marin-*/documents/baseline_llm_extraction/data-*/_done' | wc -l"
