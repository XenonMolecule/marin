#!/usr/bin/env bash
# Launch separate parent jobs per TPU type for testing.
# Each parent uses --fleet-filter and --max-count to submit a controlled number.
# Existing v5litepod-16 production job (extract-production-v6) is untouched.

set -euo pipefail

IRIS_CONFIG="lib/iris/examples/marin.yaml"
SCRIPT="experiments/baseline_collection/launch_production.py"
COMMON="--output-subdir documents/baseline_llm_extraction --wave-pause 600"

launch() {
    local type=$1
    local count=$2
    local job_name="extract-fleet-${type}"

    echo "Launching $job_name ($type x$count, canary=2)..."
    uv run iris --config "$IRIS_CONFIG" job run \
        --memory 2GB --no-wait --job-name "$job_name" \
        -- python "$SCRIPT" $COMMON --fleet-filter "$type" --max-count "$count" \
        2>&1 | grep "Job submitted" || echo "  FAILED to submit $job_name"
}

echo "=== Launching per-type fleet ==="
echo ""

# 16 each
launch "v6e-16" 16
launch "v5litepod-4" 16
launch "v5litepod-8" 16
launch "v6e-4" 16
launch "v6e-8" 16

# 4 each (multi-host 32)
launch "v5litepod-32" 4
launch "v6e-32" 4

echo ""
echo "=== 7 parent jobs launched ==="
echo "Monitor: gcloud storage ls 'gs://marin-*/documents/baseline_llm_extraction/data-*/batch_0000*' 2>/dev/null | wc -l"
