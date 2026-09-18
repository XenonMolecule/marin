#!/usr/bin/env bash
# Pull the 16 sysprompt30b ladder HF exports from GCS onto SC and verify them.
#
# Run ON scdt (the SC data-transfer node), where gcloud is already authenticated:
#
#     rsync -av pull_sysprompt30b_ladder_to_sc.sh scdt:~/ && ssh scdt 'bash ~/pull_sysprompt30b_ladder_to_sc.sh'
#
# Layout written (mirrors the 998M arm names already under models/):
#     $DST/<rung>/{sysprompt,token-matched,doc-matched,mix50}/   (config, safetensors, tokenizer)
#
# Idempotent: a model whose safetensors md5 already matches GCS is skipped, so
# rerunning after an interruption only fetches what is missing or corrupt.
#
# Options:
#     --fix-tokenizer   copy tokenizer_config.json + special_tokens_map.json from the 998M
#                       doc-matched export into each new dir (these exports were written
#                       by transformers 5.x, which omits the 4.x-style files; harmless on 5.x)
#     --dry-run         print what would be copied, copy nothing
set -euo pipefail

# The SDK on SC is the shared install under the project dir, not on PATH.
export PATH=/juice2/scr2/nlp/personal-rm/google-cloud-sdk/bin:$PATH
export CLOUDSDK_CONFIG=/juice2/scr2/nlp/personal-rm/.gcloud
command -v gcloud >/dev/null || { echo "gcloud not found at /juice2/scr2/nlp/personal-rm/google-cloud-sdk/bin" >&2; exit 1; }
BUCKET=gs://marin-us-central1/checkpoints/isoflop-curation
DST=/juice2/scr2/nlp/personal-rm/dclm_sysprompts/models/ladder
TOKENIZER_DONOR=/juice2/scr2/nlp/personal-rm/dclm_sysprompts/models/doc-matched

FIX_TOKENIZER=0; DRY_RUN=0
for a in "$@"; do
  case "$a" in
    --fix-tokenizer) FIX_TOKENIZER=1 ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

# rung  run-name suffix  final step for arms A B C D
RUNS=(
  "210m_3e18   -expFM_natural-3e+18-d640-L7-B16     46192 46228 45448 45821"
  "358m_9e18   -expFM_natural-9e+18-d896-L10-B32    35912 35922 35330 35621"
  "447m_1p8e19 -expFM_natural-2e+19-d1024-L11-B32   55537 55563 54625 55081"
  "669m_3e19   -expFM_natural-3e+19-d1280-L13-B32   58911 58927 57945 58428"
)
ARMS=(A B C D)
NAMES=(sysprompt token-matched doc-matched mix50)

gcs_md5() {  # hex md5 of one GCS object
  gcloud storage hash --hex --skip-crc32c "$1" | awk '/md5/ {print $NF}'
}

ok=0; copied=0; failed=()
for line in "${RUNS[@]}"; do
  read -r tag suffix sA sB sC sD <<<"$line"
  steps=("$sA" "$sB" "$sC" "$sD")
  rung="${tag%%_*}"
  for i in 0 1 2 3; do
    src="$BUCKET/curation-sysprompt30b_${tag}_${ARMS[$i]}${suffix}/hf/step-${steps[$i]}"
    dst="$DST/$rung/${NAMES[$i]}"
    want=$(gcs_md5 "$src/model.safetensors")
    if [[ -f "$dst/model.safetensors" ]] && [[ "$(md5sum "$dst/model.safetensors" | cut -c1-32)" == "$want" ]]; then
      echo "OK      $rung/${NAMES[$i]} (already verified)"; ok=$((ok+1)); continue
    fi
    if (( DRY_RUN )); then echo "WOULD   $src -> $dst"; continue; fi
    echo "PULL    $src -> $dst"
    mkdir -p "$dst"
    gcloud storage cp -r "$src/*" "$dst/"
    have=$(md5sum "$dst/model.safetensors" | cut -c1-32)
    if [[ "$have" == "$want" ]]; then
      echo "OK      $rung/${NAMES[$i]} md5=$have"; copied=$((copied+1))
    else
      echo "BAD     $rung/${NAMES[$i]} md5 local=$have gcs=$want" >&2; failed+=("$rung/${NAMES[$i]}")
    fi
    if (( FIX_TOKENIZER )); then
      cp -n "$TOKENIZER_DONOR/tokenizer_config.json" "$TOKENIZER_DONOR/special_tokens_map.json" "$dst/" 2>/dev/null || true
    fi
  done
done

(( DRY_RUN )) || gcloud storage cp "$BUCKET/README_sysprompt30b_ladder.md" "$DST/README.md"

echo
echo "verified already: $ok   copied+verified: $copied   failed: ${#failed[@]}"
if (( ${#failed[@]} )); then printf '  FAILED: %s\n' "${failed[@]}"; exit 1; fi
