# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 3: score the 7000-doc frozen test with the stage-2 survivor fastText,
in the SAME order as modernbert_tpu_smoke.read_test (so it aligns to
fasttext_probs_on_bert_test.json and mb-1M-final-e5w4.json). Writes {s2_probs,labels}."""

import json
import random

import fasttext
import fsspec

LABEL_USEFUL = "__label__useful"
S2_MODEL = "gs://marin-us-central2/classifiers/useful_fasttext/cascade_stage2_survivors/model.bin"
OUT = "gs://marin-us-central2/classifiers/useful_fasttext/modernbert_results/stage2_probs_on_bert_test.json"
# prefer us-central2 test copy (local); fall back to us-east5
TEST_GLOBS = [
    "gs://marin-us-central2/classifiers/useful_fasttext/full_prep_body_strip/test/*.txt.gz",
    "gs://marin-us-east5/classifiers/useful_fasttext/full_prep_body_strip/test/*.txt.gz",
]


def read_fasttext(path, max_rows):
    texts, labels = [], []
    with fsspec.open(path, "rt", compression="gzip", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            label, _, text = line.partition(" ")
            if not text:
                continue
            labels.append(1 if label == LABEL_USEFUL else 0)
            texts.append(text)
            if len(texts) >= max_rows:
                break
    return texts, labels


def read_test_aligned(glob_path, max_rows=7000):
    fs = fsspec.filesystem("gcs")
    shards = sorted("gs://" + p for p in fs.glob(glob_path.replace("gs://", "")))
    per = max(1, max_rows // len(shards))
    rng = random.Random(0)
    texts, labels = [], []
    for shard in shards:
        st, sl = read_fasttext(shard, 10**9)
        order = list(range(len(st)))
        rng.shuffle(order)
        for j in order[:per]:
            texts.append(st[j])
            labels.append(sl[j])
    return texts, labels


def main():
    fs = fsspec.filesystem("gcs")
    glob_path = next((g for g in TEST_GLOBS if fs.glob(g.replace("gs://", ""))), None)
    assert glob_path, "no test shards found"
    print(f"reading test from {glob_path}", flush=True)
    texts, labels = read_test_aligned(glob_path)
    print(f"test docs={len(texts)} useful={sum(labels)}", flush=True)

    local = "/tmp/s2.bin"
    with fsspec.open(S2_MODEL, "rb") as src, open(local, "wb") as d:
        d.write(src.read())
    m = fasttext.load_model(local)

    def pu(t):
        for prob, lab in m.f.predict(t, -1, 0.0, "strict"):
            if lab == LABEL_USEFUL:
                return float(prob)
        return 0.0

    probs = [pu(t) for t in texts]
    with fsspec.open(OUT, "wt", encoding="utf-8") as o:
        json.dump({"s2_probs": probs, "labels": labels}, o)
    print(f"PHASE3 SCORES DONE n={len(probs)} useful={sum(labels)} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
