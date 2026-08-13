#!/usr/bin/env bash
# Add us-east1 to the lpv11 10k run — the 6th region (marin-us-west1 has no bucket, so this is the
# last one available). us-east1-d carries v6e, so it contributes to Phase B as well as A/C.
#
# WAITS for the main sequential launcher to finish first: submit concurrency is a GLOBAL budget
# against one controller (each submit opens an SSH tunnel), so overlapping launchers is what
# produced 115 tunnels and zero created jobs earlier tonight.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
set -a; source .env 2>/dev/null; set +a
export SSL_CERT_FILE="$(.venv/bin/python -m certifi 2>/dev/null)"

SPEC=lpv11_fastpipe_v1
M=experiments/distill/dclm_400m_1x.txt
R=us-east1
B=gs://marin-us-east1
log () { echo "$(date -u +%H:%M:%S) $*"; }

log "waiting for the main launcher to finish..."
while pgrep -f "launch_10k_overnight.sh" >/dev/null 2>&1; do sleep 30; done
# let any in-flight tunnels drain before adding more
sleep 60
log "main launcher done; tunnels=$(ps -eo command | grep -c '[c]ompute ssh iris-controller')"

log "B us-east1 v6e-4 n=16"
uv run python -m experiments.fast_curation.launch_tpu --spec $SPEC --manifest $M \
  --bucket $B --region $R --mode v2b --num-workers 16 --seed-start 12000 \
  --priority batch --preemptible --tpu-type v6e-4 --memory 64GB --max-idle-passes 90 2>&1 | tail -1

log "A us-east1 n=45"
uv run python -m experiments.fast_curation.launch_cpu_a --spec $SPEC --manifest $M \
  --bucket $B --region $R --num-workers 45 --seed-start 26000 \
  --priority batch --preemptible --cpu 8 --memory 64GB --disk 30GB --max-idle-passes 60 2>&1 | tail -1

log "C us-east1 n=10"
uv run python -m experiments.fast_curation.launch_cpu_c --spec $SPEC --manifest $M \
  --bucket $B --region $R --num-workers 10 --seed-start 36000 \
  --priority batch --preemptible --cpu 8 --memory 32GB --disk 20GB --max-idle-passes 90 \
  --resiliparse-artifact "$B/artifacts/resiliparse_rs/latest" 2>&1 | tail -1

# eu-west4 v6e-4 only got 1/16 in the main run (transient controller pressure), so retry it here.
log "B eu-west4 v6e-4 retry n=16"
uv run python -m experiments.fast_curation.launch_tpu --spec $SPEC --manifest $M \
  --bucket gs://marin-eu-west4 --region eu-west4 --mode v2b --num-workers 16 --seed-start 12100 \
  --priority batch --preemptible --tpu-type v6e-4 --memory 64GB --max-idle-passes 90 2>&1 | tail -1

log "US-EAST1 + EU-WEST4 RETRY SUBMITTED"
