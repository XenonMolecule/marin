"""Plot the jusText router operating curve from operating_curve.csv -> PNG."""
import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    f = lambda k: [float(r[k]) for r in rows]
    thr = f("threshold")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: precision-recall curve (dev + test)
    ax1.plot(f("dev_recall"), f("dev_precision"), "-o", ms=3, label="dev", color="C0")
    ax1.plot(f("test_recall"), f("test_precision"), "-o", ms=3, label="test", color="C1")
    ax1.axhline(1442 / 27493, ls=":", color="gray", lw=1, label="base rate (~5%)")
    ax1.set_xlabel("recall (of extractable docs captured)")
    ax1.set_ylabel("precision (jusText right when routed)")
    ax1.set_title("Router PR curve")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3); ax1.legend()

    # Right: precision / recall / coverage vs threshold
    ax2.plot(thr, f("dev_precision"), label="precision (dev)", color="C0")
    ax2.plot(thr, f("test_precision"), label="precision (test)", color="C0", ls="--")
    ax2.plot(thr, f("dev_recall"), label="recall (dev)", color="C2")
    ax2.plot(thr, f("dev_coverage"), label="coverage→jusText (dev)", color="C3")
    ax2.set_xlabel("threshold  P(extractable) ≥ t → route to jusText")
    ax2.set_ylabel("rate")
    ax2.set_title("Precision / Recall / Coverage vs threshold")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3); ax2.legend()

    fig.suptitle("jusText extractability router (predict Lev sim ≥ 0.85) — precision caps ~0.5", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
