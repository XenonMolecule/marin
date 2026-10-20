#!/usr/bin/env bash
# Pull the 447M multi-epoch and 998M data-efficient sysprompt checkpoints from GCS onto SC and verify them.
#
# Run ON scdt (the SC data-transfer node), where gcloud is authenticated:
#
#     rsync -av pull_sysprompt30b_epochs_1b_to_sc.sh scdt:~/ && ssh scdt 'bash ~/pull_sysprompt30b_epochs_1b_to_sc.sh'
#
# Layout written under $DST:
#     epochs_447m/{2,4,8}ep/{doc-matched,mix50,sysprompt}/
#     data_efficient_998m/seed{0,1,2}/{doc-matched,mix50,sysprompt}/
# plus README_epochs_and_1b.md and examples/{examples.md,examples.jsonl}. Each model dir gets a SOURCE file naming its run and step.
# Idempotent: a model whose safetensors md5 already matches GCS is skipped.
set -euo pipefail

export PATH=/juice2/scr2/nlp/personal-rm/google-cloud-sdk/bin:$PATH
export CLOUDSDK_CONFIG=/juice2/scr2/nlp/personal-rm/.gcloud
command -v gcloud >/dev/null || { echo "gcloud not found" >&2; exit 1; }
BUCKET=gs://marin-us-central1/checkpoints/isoflop-curation
DST=/juice2/scr2/nlp/personal-rm/dclm_sysprompts/models
declare -A NAMES=([A]=sysprompt [C]=doc-matched [D]=mix50)

gcs_md5() { gcloud storage hash --hex --skip-crc32c "$1" | awk '/md5/ {print $NF}'; }

failed=()
pull() {  # pull <run name> <dest dir>
  local run=$1 dst=$2
  local hf; hf=$(gcloud storage ls "$BUCKET/$run/hf/" | sed 's#/$##' | awk -F'step-' '{print $2" "$0}' | sort -n | tail -1 | cut -d' ' -f2)
  [[ -n "$hf" ]] || { echo "MISSING $run (no hf export)" >&2; failed+=("$dst"); return; }
  local want; want=$(gcs_md5 "$hf/model.safetensors")
  if [[ -f "$dst/model.safetensors" ]] && [[ "$(md5sum "$dst/model.safetensors" | cut -c1-32)" == "$want" ]]; then
    echo "OK      $dst (already verified)"; return
  fi
  echo "PULL    $hf -> $dst"
  mkdir -p "$dst"
  gcloud storage cp -r "$hf/*" "$dst/"
  echo "$run ${hf##*/}" > "$dst/SOURCE"
  local have; have=$(md5sum "$dst/model.safetensors" | cut -c1-32)
  if [[ "$have" == "$want" ]]; then echo "OK      $dst md5=$have"; else echo "BAD     $dst local=$have gcs=$want" >&2; failed+=("$dst"); fi
}

P=curation-sysprompt30b
for arm in A C D; do
  pull "${P}_447m_9e18_${arm}_ep2-expFM_natural-2e+19-d1024-L11-B32"  "$DST/epochs_447m/2ep/${NAMES[$arm]}"
  pull "${P}_447m_9e18_${arm}_ep4-expFM_natural-4e+19-d1024-L11-B64"  "$DST/epochs_447m/4ep/${NAMES[$arm]}"
  pull "${P}_447m_9e18_${arm}_ep8-expFM_natural-7e+19-d1024-L11-B128" "$DST/epochs_447m/8ep/${NAMES[$arm]}"
  for seed in 0 1 2; do
    suf=$([[ $seed == 0 ]] && echo "" || echo "-s$seed")
    pull "${P}_998m_4p8e19_${arm}-expFM_natural-5e+19-d1536-L16-B32${suf}" "$DST/data_efficient_998m/seed$seed/${NAMES[$arm]}"
  done
done

gcloud storage cp "$BUCKET/README_sysprompt30b_epochs_1b.md" "$DST/README_epochs_and_1b.md"
# Example system prompts with document excerpts, for review (see sample_sysprompt30b_examples.py).
mkdir -p "$DST/examples"
gcloud storage cp "$BUCKET/sysprompt30b_examples/examples.md" "$BUCKET/sysprompt30b_examples/examples.jsonl" "$DST/examples/"
echo "failed: ${#failed[@]}"; (( ${#failed[@]} == 0 ))
