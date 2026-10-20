#!/usr/bin/env bash
# Pull the best 447M @ 9e18 model per arm (hp1 sweep + probe1) from GCS onto SC and verify them.
#
# Run ON scdt (the SC data-transfer node), where gcloud is already authenticated:
#
#     rsync -av pull_sysprompt30b_447m_9e18_to_sc.sh scdt:~/ && ssh scdt 'bash ~/pull_sysprompt30b_447m_9e18_to_sc.sh'
#
# Layout written: $DST/{doc-matched,sysprompt,sysprompt-masked,mix50,cooldown}/ plus README.md.
# Idempotent: a model whose safetensors md5 already matches GCS is skipped.
set -euo pipefail

export PATH=/juice2/scr2/nlp/personal-rm/google-cloud-sdk/bin:$PATH
export CLOUDSDK_CONFIG=/juice2/scr2/nlp/personal-rm/.gcloud
command -v gcloud >/dev/null || { echo "gcloud not found" >&2; exit 1; }
BUCKET=gs://marin-us-central1/checkpoints/isoflop-curation
DST=/juice2/scr2/nlp/personal-rm/dclm_sysprompts/models/data_efficient_447m
P=curation-sysprompt30b_447m_9e18

# directory  run name  final step
RUNS=(
  "doc-matched       ${P}_C-expFM_natural-9e+18-d1024-L11-B16-sw-lr1x            54930"
  "sysprompt         ${P}_A-expFM_natural-9e+18-d1024-L11-B32-sw-lr1x            27913"
  "sysprompt-masked  ${P}_AM-expFM_natural-9e+18-d1024-L11-B32-sw-lr1x           27913"
  "mix50             ${P}_D-expFM_natural-9e+18-d1024-L11-B16-sw-lr1x-b2-0p999   55379"
  "cooldown          ${P}_E-expFM_natural-9e+18-d1024-L11-B32-sw-lr1x-b2-0p999   27810"
)

gcs_md5() { gcloud storage hash --hex --skip-crc32c "$1" | awk '/md5/ {print $NF}'; }

failed=()
for line in "${RUNS[@]}"; do
  read -r name run step <<<"$line"
  src="$BUCKET/$run/hf/step-$step"
  dst="$DST/$name"
  want=$(gcs_md5 "$src/model.safetensors")
  if [[ -f "$dst/model.safetensors" ]] && [[ "$(md5sum "$dst/model.safetensors" | cut -c1-32)" == "$want" ]]; then
    echo "OK      $name (already verified)"; continue
  fi
  echo "PULL    $src -> $dst"
  mkdir -p "$dst"
  gcloud storage cp -r "$src/*" "$dst/"
  echo "$run step $step" > "$dst/SOURCE"
  have=$(md5sum "$dst/model.safetensors" | cut -c1-32)
  if [[ "$have" == "$want" ]]; then echo "OK      $name md5=$have"; else echo "BAD     $name local=$have gcs=$want" >&2; failed+=("$name"); fi
done

gcloud storage cp "$BUCKET/README_sysprompt30b_447m_9e18.md" "$DST/README.md"
echo "failed: ${#failed[@]}"
(( ${#failed[@]} == 0 ))
