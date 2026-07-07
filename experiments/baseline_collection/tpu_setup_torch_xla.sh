#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# One-time per-VM setup for running torch_xla ModernBERT inference on a dev TPU VM
# (scripts/iris/dev_tpu.py). torch-xla is NOT in marin's pyproject/lockfile, so it is
# installed out-of-band here (the recipe from project_modernbert_tpu_torch_xla):
#   * torch-xla[tpu]==2.7.0 (pins torch==2.7.0) from the libtpu-releases find-links index
#   * clear_execstack.py strips the executable-stack flag on _XLAC.so (gVisor refuses it;
#     harmless no-op on a plain VM)
# Run from ~/marin (dev_tpu execute cd's there). Idempotent: re-running just re-resolves.
set -euo pipefail

uv pip install -p .venv/bin/python \
    torch==2.7.0 'torch-xla[tpu]==2.7.0' \
    --find-links https://storage.googleapis.com/libtpu-releases/index.html

# Best-effort: clear the exec-stack bit on whatever _XLAC*.so exists. On a real GCE TPU
# VM this is a no-op (no gVisor), and the .so name varies by version, so never fail here.
for so in .venv/lib/python3.11/site-packages/torch_xla/_XLAC*.so; do
    [ -f "$so" ] && .venv/bin/python experiments/baseline_collection/clear_execstack.py "$so" || true
done

# Prove the import works (the real gate — catches any libtpu/ABI issue before a run).
.venv/bin/python -c "import torch_xla.core.xla_model as xm; print('torch_xla import OK')"
echo SETUP_DONE
