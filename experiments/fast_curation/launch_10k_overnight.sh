#!/usr/bin/env bash
# Sequential multi-region launch for the lpv11 10k run.
#
# SEQUENTIAL ON PURPOSE. _launch_common caps submits at MAX_CONCURRENT_SUBMITS=8 because every
# submit opens its own SSH tunnel to the controller. Running the region launchers in PARALLEL
# multiplies that cap by the number of regions: 5 x 8 = 40 tunnels, which strangled the controller
# (115 tunnels, job list 50-60s, and 0 of 270 jobs actually created). One launcher at a time.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
set -a; source .env 2>/dev/null; set +a
export SSL_CERT_FILE="$(.venv/bin/python -m certifi 2>/dev/null)"

SPEC=lpv11_fastpipe_v1
M=experiments/distill/dclm_400m_1x.txt
log () { echo "$(date -u +%H:%M:%S) $*"; }

a () {  # region bucket seed n
  log "A  $1 n=$4"
  uv run python -m experiments.fast_curation.launch_cpu_a --spec $SPEC --manifest $M \
    --bucket "$2" --region "$1" --num-workers "$4" --seed-start "$3" \
    --priority batch --preemptible --cpu 8 --memory 64GB --disk 30GB --max-idle-passes 60 2>&1 | tail -1
}
c () {  # region bucket seed n
  log "C  $1 n=$4"
  uv run python -m experiments.fast_curation.launch_cpu_c --spec $SPEC --manifest $M \
    --bucket "$2" --region "$1" --num-workers "$4" --seed-start "$3" \
    --priority batch --preemptible --cpu 8 --memory 32GB --disk 20GB --max-idle-passes 90 \
    --resiliparse-artifact "$2/artifacts/resiliparse_rs/latest" 2>&1 | tail -1
}
b () {  # region bucket seed n tpu_type
  log "B  $1 $5 n=$4"
  uv run python -m experiments.fast_curation.launch_tpu --spec $SPEC --manifest $M \
    --bucket "$2" --region "$1" --mode v2b --num-workers "$4" --seed-start "$3" \
    --priority batch --preemptible --tpu-type "$5" --memory 64GB --max-idle-passes 90 2>&1 | tail -1
}

# --- Phase B FIRST: TPU is the scarce resource, so get claims in the queue early. Every
# single-host type across every region that has the models mirrored; blocked pools just sit
# pending at zero cost and grab a slice the moment one frees.
b us-east5    gs://marin-us-east5    10000 24 v6e-4
b us-east5    gs://marin-us-east5    10100 12 v5p-8
b us-central1 gs://marin-us-central1 10200 16 v5p-8
b eu-west4    gs://marin-eu-west4    10300 16 v6e-4
b eu-west4    gs://marin-eu-west4    10400 12 v5litepod-4
b us-west4    gs://marin-us-west4    10500 16 v5litepod-4
b us-west4    gs://marin-us-west4    10600 8  v5litepod-8
b us-central2 gs://marin-us-central2 10700 8  v4-8

# --- Phase A: the decode-bound feeder. Biggest fleet.
a us-east5    gs://marin-us-east5    20000 55
a us-central1 gs://marin-us-central1 21000 55
a us-central2 gs://marin-us-central2 22000 45
a us-west4    gs://marin-us-west4    23000 45
a eu-west4    gs://marin-eu-west4    24000 45

# --- Phase C: cheapest per WARC (~175 docs/s/core); a small fleet keeps up with B.
c us-east5    gs://marin-us-east5    30000 10
c us-central1 gs://marin-us-central1 31000 10
c us-central2 gs://marin-us-central2 32000 8
c us-west4    gs://marin-us-west4    33000 8
c eu-west4    gs://marin-eu-west4    34000 8

log "ALL LAUNCHES SUBMITTED"
