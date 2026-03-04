#!/usr/bin/env bash
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Fetch and filter Ray job logs for monitoring progress.
#
# Usage:
#   ./scripts/ray/check_logs.sh <cluster> <submission_id> [lines] [pattern]
#
# Pattern can be a raw regex or one of these presets:
#   inference  - Zephyr pipeline progress (stage completion, workers, shards)
#   training   - Levanter training progress (loss, eval, steps, checkpoints)
#   errors     - Errors, failures, retries, node deaths
#   all        - Combined: inference + training + errors (default)
#   raw        - No filtering, just tail the last N lines
#
# Lines can be a number (tail last N), or "0"/"all" for unlimited output.
#
# Examples:
#   # Default (all preset): last 30 progress/error lines
#   ./scripts/ray/check_logs.sh us-east5-a ray-run-michaelryan-foo
#
#   # Training-only lines, last 50
#   ./scripts/ray/check_logs.sh us-central1 ray-run-michaelryan-foo 50 training
#
#   # All inference progress lines (unlimited)
#   ./scripts/ray/check_logs.sh us-east5-a ray-run-michaelryan-foo 0 inference
#
#   # Custom regex
#   ./scripts/ray/check_logs.sh us-east5-a ray-run-michaelryan-foo 30 "my_custom|pattern"
#
#   # Raw tail (no filtering)
#   ./scripts/ray/check_logs.sh us-east5-a ray-run-michaelryan-foo 50 raw

set -euo pipefail

CLUSTER="${1:?Usage: check_logs.sh <cluster> <submission_id> [lines] [pattern]}"
JOB_ID="${2:?Usage: check_logs.sh <cluster> <submission_id> [lines] [pattern]}"
LINES="${3:-30}"
PATTERN="${4:-all}"

# Pattern presets
PAT_INFERENCE="zephyr\.execution|in-flight|queued"
PAT_TRAINING="eval/loss|train/loss|train_step|saving checkpoint|step-[0-9]|training_step|wandb"
PAT_ERRORS="FAILED|ERROR|Error|Traceback|Attempt.*failed|ZephyrWorkerError|No workers available|has been marked dead|OOM|retry"
PAT_EXECUTOR="executor.*SUCCESS|executor.*FAILED|already succeeded|skip|cached"

case "${PATTERN}" in
    inference) PATTERN="${PAT_INFERENCE}" ;;
    training)  PATTERN="${PAT_TRAINING}" ;;
    errors)    PATTERN="${PAT_ERRORS}" ;;
    all)       PATTERN="${PAT_INFERENCE}|${PAT_TRAINING}|${PAT_ERRORS}|${PAT_EXECUTOR}" ;;
    raw)       ;; # handled below
    *)         ;; # treat as custom regex
esac

# Use a fixed temp file per job so we don't accumulate files
LOGFILE="/tmp/ray_logs_${JOB_ID}.txt"

echo "Fetching logs for ${JOB_ID} on ${CLUSTER}..."
uv run scripts/ray/cluster.py --cluster "${CLUSTER}" job-logs "${JOB_ID}" > "${LOGFILE}" 2>&1

if [ $? -ne 0 ] || [ ! -s "${LOGFILE}" ]; then
    echo "ERROR: Failed to fetch logs or log file is empty."
    echo "Last lines of output:"
    tail -10 "${LOGFILE}" 2>/dev/null
    exit 1
fi

TOTAL=$(wc -l < "${LOGFILE}" | tr -d ' ')
echo "Fetched ${TOTAL} total log lines -> ${LOGFILE}"
echo "---"

if [ "${PATTERN}" = "raw" ]; then
    if [ "${LINES}" = "0" ] || [ "${LINES}" = "all" ]; then
        cat "${LOGFILE}"
    else
        tail -"${LINES}" "${LOGFILE}"
    fi
else
    if [ "${LINES}" = "0" ] || [ "${LINES}" = "all" ]; then
        grep -E "${PATTERN}" "${LOGFILE}" || echo "(no matching lines found)"
    else
        grep -E "${PATTERN}" "${LOGFILE}" | tail -"${LINES}" || echo "(no matching lines found)"
    fi
fi
