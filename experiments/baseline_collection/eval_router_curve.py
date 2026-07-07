"""Dump the operating curve (precision / recall / F1 / coverage vs threshold) for the
trained jusText extractability-router fastText model, on dev and test.

coverage = fraction of docs routed to jusText (predicted P(extractable) >= threshold) =
the share of the corpus you keep off the expensive 1.7B path.

Outputs {out_root}/operating_curve.csv and prints, for target precisions, the threshold +
the resulting recall and coverage (on dev, with the test numbers at that same threshold).
"""
import argparse
import gzip
import io

import fsspec
import fasttext

POS = "__label__extractable"


def load_local(gs_path: str, local: str) -> None:
    with fsspec.open(gs_path, "rb") as s, open(local, "wb") as d:
        d.write(s.read())


def scored(model, gs_txt_gz: str):
    """-> list[(P(extractable), gold_is_pos)] for a split."""
    out = []
    with fsspec.open(gs_txt_gz, "rb") as f, gzip.open(f, "rt", encoding="utf-8") as g:
        for ln in g:
            if not ln.strip():
                continue
            lab, text = ln.split(" ", 1)
            labels, probs = model.predict(text.rstrip("\n"), k=-1)
            p = dict(zip(labels, probs)).get(POS, 0.0)
            out.append((p, int(lab == POS)))
    return out


def pr_at(s, t):
    tp = sum(1 for p, g in s if p >= t and g)
    fp = sum(1 for p, g in s if p >= t and not g)
    npos = sum(g for _, g in s) or 1
    n = len(s) or 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / npos
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    cov = (tp + fp) / n  # fraction routed to jusText
    return prec, rec, f1, cov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ft-root", required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    load_local(args.model, "/app/router.bin")
    model = fasttext.load_model("/app/router.bin")
    dev = scored(model, f"{args.ft_root}/dev.txt.gz")
    test = scored(model, f"{args.ft_root}/test.txt.gz")
    print(f"dev {len(dev)} ({sum(g for _,g in dev)} pos), test {len(test)} ({sum(g for _,g in test)} pos)", flush=True)

    grid = [i / 200 for i in range(201)]  # 0.000 .. 1.000 step 0.005
    rows = ["threshold,dev_precision,dev_recall,dev_f1,dev_coverage,test_precision,test_recall,test_f1,test_coverage"]
    for t in grid:
        dp, dr, df, dc = pr_at(dev, t)
        tp, tr, tf, tc = pr_at(test, t)
        rows.append(f"{t:.3f},{dp:.4f},{dr:.4f},{df:.4f},{dc:.4f},{tp:.4f},{tr:.4f},{tf:.4f},{tc:.4f}")
    buf = io.BytesIO(("\n".join(rows) + "\n").encode())
    with fsspec.open(f"{args.out_root}/operating_curve.csv", "wb") as d:
        d.write(buf.getvalue())

    # For target precisions, find the lowest dev threshold achieving it (max recall at that precision).
    print("\n== DEV: lowest threshold reaching each precision target (recall + coverage there) | TEST at same thr ==", flush=True)
    print(f"{'targetP':>7} {'thr':>6} {'devP':>6} {'devR':>6} {'devCov':>7} | {'testP':>6} {'testR':>6} {'testCov':>7}", flush=True)
    for target in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        pick = None
        for t in grid:
            dp, dr, df, dc = pr_at(dev, t)
            if dp >= target and dr > 0:
                pick = (t, dp, dr, dc)
                break
        if pick is None:
            print(f"{target:>7.2f}    —   (precision {target} not reached on dev)", flush=True)
            continue
        t, dp, dr, dc = pick
        tp, tr, tf, tc = pr_at(test, t)
        print(f"{target:>7.2f} {t:>6.3f} {dp:>6.3f} {dr:>6.3f} {dc:>7.3f} | {tp:>6.3f} {tr:>6.3f} {tc:>7.3f}", flush=True)
    print(f"\nsaved -> {args.out_root}/operating_curve.csv", flush=True)


if __name__ == "__main__":
    main()
