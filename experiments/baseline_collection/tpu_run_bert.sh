#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Run the ModernBERT WARC filter on a dev TPU VM. Sets the on-disk scratch + HF cache to
# writable VM paths (the script's /app default is the Iris container path, absent here),
# then forwards all extra args to modernbert_warc_filter. Use `.venv/bin/python` directly
# (NOT `uv run`, which would re-sync the lockfile and uninstall the out-of-band torch-xla).
#
# Usage (via dev_tpu execute):
#   bash experiments/baseline_collection/tpu_run_bert.sh --decoded-root gs://... --out-root gs://... \
#       --bert-ckpt gs://...mb-1M-e5w4 --manifest experiments/distill/dclm_400m_1x.txt --limit 1 --tpu-type v6e-4
set -euo pipefail

# Leading control flags (consumed here; env-prefixes can't pass through dev_tpu execute):
#   --detach        run detached (survives SSH disconnect), exit 0 so the SSH can close
#   --tag <name>    log file suffix ($HOME/bert_run-<name>.log)
DETACH=0
TAG="fg"
while [ "${1:-}" = "--detach" ] || [ "${1:-}" = "--tag" ]; do
    case "$1" in
        --detach) DETACH=1; shift ;;
        --tag) TAG="$2"; shift 2 ;;
    esac
done

export MBWARC_SCRATCH="${MBWARC_SCRATCH:-$HOME/mbscratch}"
export HF_HOME="${HF_HOME:-$HOME/hf}"
mkdir -p "$MBWARC_SCRATCH" "$HF_HOME"

LOG="$HOME/bert_run-${TAG}.log"
if [ "$DETACH" = "1" ]; then
    # Detached: progress is tracked via GCS done/timing markers, not the SSH stream.
    setsid nohup .venv/bin/python -m experiments.baseline_collection.modernbert_warc_filter "$@" \
        > "$LOG" 2>&1 < /dev/null &
    echo "DETACHED pid=$! log=$LOG"
    sleep 2
else
    .venv/bin/python -m experiments.baseline_collection.modernbert_warc_filter "$@"
fi
