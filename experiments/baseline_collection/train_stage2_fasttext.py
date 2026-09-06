# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 2: train the cascade-specialized stage-2 fastText on the survivor residual.
Concatenates the GCS survivor parts, trains with the DCLM recipe (bigrams, minCount 500).

Temp files live under WORKDIR (default /app — a real disk in the iris container, NOT
tmpfs /tmp which is RAM-backed and OOM-killed the prior run when the 12GB train file +
model + upload copy all sat in memory at once). The upload streams (copyfileobj) instead
of reading the whole model into a Python bytes object."""

import os
import shutil

import fasttext
import fsspec

PARTS = "gs://marin-us-central2/datasets/useful_cascade_survivors/parts"
OUT = "gs://marin-us-central2/classifiers/useful_fasttext/cascade_stage2_survivors/model.bin"
WORKDIR = os.environ.get("STAGE2_WORKDIR", "/app")
TRAIN = os.path.join(WORKDIR, "_s2_train.txt")
MODEL = os.path.join(WORKDIR, "_s2_model.bin")

fs = fsspec.filesystem("gcs")
parts = sorted("gs://" + p for p in fs.glob(PARTS.replace("gs://", "") + "/*.gz"))
print(f"concatenating {len(parts)} survivor parts -> {TRAIN}", flush=True)
n = 0
with open(TRAIN, "w", encoding="utf-8") as out:
    for p in parts:
        with fsspec.open(p, "rt", compression="gzip", encoding="utf-8") as f:
            for line in f:
                out.write(line)
                n += 1
print(f"train lines = {n}; training fastText (DCLM recipe, minCount=500) ...", flush=True)
m = fasttext.train_supervised(
    TRAIN,
    epoch=5,
    lr=0.1,
    dim=100,
    wordNgrams=2,
    loss="softmax",
    minCount=500,
)
m.save_model(MODEL)
with open(MODEL, "rb") as src, fsspec.open(OUT, "wb") as dst:
    shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
print(f"PHASE2 DONE: model {os.path.getsize(MODEL)/1e9:.2f}GB -> {OUT}; train_lines={n}", flush=True)
