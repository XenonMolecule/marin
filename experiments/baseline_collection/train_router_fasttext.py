"""Train the jusText extractability-router fastText classifier (predict Lev sim >= 0.85).

Reads the router fastText files ({ft_root}/{train,dev,test}.txt.gz, lines
`__label__extractable|needs_llm <text>`), sweeps a few neg:pos training ratios (positives are
the minority "extractable" class), picks the ratio with the best threshold-swept dev F1, then
reports test at that dev-chosen threshold and saves the model + metrics.

Recipe (DCLM, fixed — do not autotune): epoch 5, lr 0.1, dim 100, wordNgrams 2, softmax, minCount.
Output: {out_root}/router.bin, {out_root}/metrics.json. CPU-only, in-region.
"""
import argparse
import gzip
import json
import os
import random

import fsspec
import fasttext

POS = "__label__extractable"
NEG = "__label__needs_llm"


def download_txt(gs_path: str, local: str) -> None:
    with fsspec.open(gs_path, "rb") as f, gzip.open(f, "rt", encoding="utf-8") as g, open(local, "w", encoding="utf-8") as o:
        for line in g:
            o.write(line)


def read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [ln for ln in f if ln.strip()]


def write_lines(path: str, lines) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def count_labels(path: str) -> tuple[int, int]:
    npos = nneg = 0
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.startswith(POS):
                npos += 1
            elif ln.startswith(NEG):
                nneg += 1
    return npos, nneg


def make_ratio_train(train_path: str, ratio, npos: int, nneg: int, seed: int, out_path: str) -> tuple[str, int, int]:
    """Stream the train file (no in-RAM list) -> all positives + negatives subsampled to ratio*pos.

    ratio=None keeps everything (use the file as-is). Negatives are kept probabilistically while
    streaming, preserving the original (already pos/neg-interleaved) order, so memory stays tiny.
    Returns (path_to_use, n_pos, n_neg_kept).
    """
    if ratio is None:
        return train_path, npos, nneg
    keep = min(nneg, int(ratio * npos))
    prob = (keep / nneg) if nneg else 0.0
    rng = random.Random(seed)
    kp = kn = 0
    with open(train_path, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as o:
        for ln in f:
            if ln.startswith(POS):
                o.write(ln)
                kp += 1
            elif ln.startswith(NEG) and rng.random() < prob:
                o.write(ln)
                kn += 1
    return out_path, kp, kn


def pos_prob(model, text: str) -> float:
    labels, probs = model.predict(text, k=-1)
    return dict(zip(labels, probs)).get(POS, 0.0)


def eval_split(model, lines: list[str]) -> list[tuple[float, int]]:
    """-> [(P(extractable), gold_is_pos)] per doc."""
    out = []
    for ln in lines:
        lab, text = ln.split(" ", 1)
        out.append((pos_prob(model, text.rstrip("\n")), int(lab == POS)))
    return out


def best_f1(scored: list[tuple[float, int]]) -> tuple[float, float, float, float]:
    """Sweep threshold -> (best_f1, threshold, precision, recall)."""
    n_pos = sum(g for _, g in scored) or 1
    best = (0.0, 0.5, 0.0, 0.0)
    for t in sorted({s for s, _ in scored} | {0.0}):
        tp = sum(1 for s, g in scored if s >= t and g)
        fp = sum(1 for s, g in scored if s >= t and not g)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / n_pos
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best[0]:
            best = (f1, t, prec, rec)
    return best


def f1_at(scored: list[tuple[float, int]], t: float) -> dict:
    tp = sum(1 for s, g in scored if s >= t and g)
    fp = sum(1 for s, g in scored if s >= t and not g)
    n_pos = sum(g for _, g in scored) or 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / n_pos
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"threshold": t, "f1": f1, "precision": prec, "recall": rec, "n_pos": n_pos}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-root", required=True, help="gs://.../justext_router_labels/fasttext")
    ap.add_argument("--out-root", required=True, help="gs://.../justext_router_labels/model")
    ap.add_argument("--ratios", default="natural,8,4,2", help="neg:pos training ratios; 'natural' keeps all")
    ap.add_argument("--epoch", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--ngrams", type=int, default=2)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for sp in ("train", "dev", "test"):
        download_txt(f"{args.ft_root}/{sp}.txt.gz", f"/app/{sp}.txt")
    # Stream the (large) train file for counts; only dev/test are small enough to hold in RAM.
    tr_pos, tr_neg = count_labels("/app/train.txt")
    dev_lines = read_lines("/app/dev.txt")
    test_lines = read_lines("/app/test.txt")
    print(f"train {tr_pos+tr_neg} ({tr_pos} extractable / {tr_neg} needs_llm), "
          f"dev {len(dev_lines)}, test {len(test_lines)}", flush=True)

    ratios = [None if r == "natural" else float(r) for r in args.ratios.split(",")]
    runs = []
    best_model = None
    best_dev = (-1.0,)
    best_ratio = None
    for r in ratios:
        train_input, npos, nneg = make_ratio_train("/app/train.txt", r, tr_pos, tr_neg, args.seed, "/app/_train_r.txt")
        m = fasttext.train_supervised(
            input=train_input, epoch=args.epoch, lr=args.lr, dim=args.dim,
            wordNgrams=args.ngrams, loss="softmax", minCount=args.min_count, thread=os.cpu_count() or 4,
        )
        dev_scored = eval_split(m, dev_lines)
        bf1, thr, prec, rec = best_f1(dev_scored)
        rlabel = "natural" if r is None else str(r)
        print(f"ratio={rlabel}: train_pos={npos} train_neg={nneg} | dev best_f1={bf1:.4f} "
              f"@thr={thr:.3f} (P={prec:.3f} R={rec:.3f})", flush=True)
        runs.append({"ratio": rlabel, "train_pos": npos, "train_neg": nneg,
                     "dev_best_f1": bf1, "dev_threshold": thr, "dev_precision": prec, "dev_recall": rec})
        if bf1 > best_dev[0]:
            best_dev = (bf1, thr)
            best_model = m
            best_ratio = rlabel

    # Test at the dev-chosen threshold of the best model.
    test_scored = eval_split(best_model, test_lines)
    test_metrics = f1_at(test_scored, best_dev[1])
    print(f"BEST ratio={best_ratio} | dev_f1={best_dev[0]:.4f} @thr={best_dev[1]:.3f} | "
          f"TEST f1={test_metrics['f1']:.4f} P={test_metrics['precision']:.3f} R={test_metrics['recall']:.3f}", flush=True)

    best_model.save_model("/app/router.bin")
    with open("/app/router.bin", "rb") as src, fsspec.open(f"{args.out_root}/router.bin", "wb") as dst:
        dst.write(src.read())
    metrics = {"best_ratio": best_ratio, "dev_best_f1": best_dev[0], "dev_threshold": best_dev[1],
               "test": test_metrics, "sweep": runs,
               "recipe": {"epoch": args.epoch, "lr": args.lr, "dim": args.dim,
                          "wordNgrams": args.ngrams, "minCount": args.min_count}}
    with fsspec.open(f"{args.out_root}/metrics.json", "wt", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved router.bin + metrics.json -> {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
