#!/usr/bin/env bash
# Wait for the us-central1 cluster dashboard to become accessible, then submit the job.
# Usage: bash scripts/wait_and_submit.sh

set -euo pipefail

CLUSTER="us-central1"
DASHBOARD_PORT=8278
POLL_INTERVAL=300  # 5 minutes

echo "Waiting for $CLUSTER dashboard to become accessible..."
echo "Polling every ${POLL_INTERVAL}s. Press Ctrl-C to abort."

while true; do
    # ray_run.py creates its own SSH tunnel, so we just need to check if the
    # cluster's dashboard command succeeds. We use a quick timeout check.
    if uv run scripts/ray/cluster.py --cluster "$CLUSTER" list-jobs >/dev/null 2>&1; then
        echo ""
        echo "$(date): Cluster $CLUSTER is reachable! Submitting job..."
        exec uv run lib/marin/src/marin/run/ray_run.py \
            --cluster "$CLUSTER" \
            --no_wait \
            --env_vars "WANDB_API_KEY=${WANDB_API_KEY}" \
            -- python experiments/exp_qwen3_0_6b_rephraser_sft.py --force_run_failed true
    fi

    echo "$(date): Cluster not reachable yet. Retrying in ${POLL_INTERVAL}s..."
    sleep "$POLL_INTERVAL"
done
