"""Build the jusText "extractability router" classification dataset from the cascade 350k corpus.

For each cascade-survivor chat row we recover the body-HTML (user turn) and the gold LLM
extraction (assistant turn), run the jusText fork extractor on the HTML, compute the normalized
Levenshtein similarity of jusText's output to gold, and label:

    label = 1  if sim >= --threshold (default 0.85)  -> jusText extracts this doc well enough
    label = 0  otherwise                             -> route to the 1.7B model

`[NO_USEFUL_CONTENT]` docs (gold = abstention token) naturally get label 0: jusText always
emits some text, so sim ~ 0. The labeled rows train a fastText router predicting which path a
doc should take (cheap jusText vs the LLM).

CPU-only. Shard across workers with --shard/--num-shards (round-robins input lines); each writes
one part, concatenate afterward. Output rows:
    {text, label, lev_sim, justext_len, gold_len, source, warc, bert_prob, ft_prob}
where `text` is the body_strip HTML (case-preserved) — the input for the future router fastText.
"""
import argparse
import gzip
import json
import os
import re
import time

import fsspec
from rapidfuzz.distance import Levenshtein

import justext

# Unwrap markers (from extraction_specs.DEFAULT_USER_TEMPLATE_FMT + cascade_chat_filter chat_row).
HTML_PRE = "[[ ## html ## ]]\n"
HTML_SUF = "\n\n[[ ## extraction_spec ## ]]\n"
TEXT_PRE = "[[ ## text ## ]]\n"
TEXT_SUF = "\n\n[[ ## completed ## ]]"

_STOPLIST = None


def unwrap_html(user_content: str) -> str:
    if HTML_PRE not in user_content:
        return ""
    return user_content.split(HTML_PRE, 1)[1].split(HTML_SUF, 1)[0]


def unwrap_gold(assistant_content: str) -> str:
    if TEXT_PRE not in assistant_content:
        return ""
    return assistant_content.split(TEXT_PRE, 1)[1].rsplit(TEXT_SUF, 1)[0]


def justext_extract(html: str) -> str:
    """jusText fork: keep non-boilerplate paragraphs as the extracted main content.

    Empty/unparseable HTML -> "" (jusText can't handle it; the doc gets sim~0 -> label 0,
    i.e. routed to the LLM, which is the correct outcome). Never raise on a bad doc.
    """
    global _STOPLIST
    if _STOPLIST is None:
        _STOPLIST = justext.get_stoplist("English")
    if not html or not html.strip():
        return ""
    try:
        paragraphs = justext.justext(html, _STOPLIST)
        # Match the gold/production extraction: blank-line ("\n\n") paragraph separation.
        return "\n\n".join(p.text for p in paragraphs if not p.is_boilerplate)
    except Exception:
        return ""


def process_row(d: dict, threshold: float) -> dict:
    msgs = d["messages"]
    user = next(m["content"] for m in msgs if m["role"] == "user")
    asst = next(m["content"] for m in msgs if m["role"] == "assistant")
    html = unwrap_html(user)
    gold = unwrap_gold(asst)
    jt = justext_extract(html)
    sim = Levenshtein.normalized_similarity(jt, gold)
    return {
        "text": html,  # body_strip HTML (case-preserved) -> router fastText input
        "label": int(sim >= threshold),
        "lev_sim": sim,
        "justext_len": len(jt),
        "gold_len": len(gold),
        "source": d.get("source"),
        "warc": d.get("warc"),
        "bert_prob": d.get("bert_prob"),
        "ft_prob": d.get("ft_prob"),
    }


def iter_jsonl_gz(path: str):
    with fsspec.open(path, "rb") as f:
        with gzip.open(f, "rt", encoding="utf-8") as g:
            for line in g:
                if line.strip():
                    yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-file", required=True, help="gs://.../final/{train,dev,test}.jsonl.gz")
    ap.add_argument("--out-file", required=True, help="gs://.../router_labels/{split}/part-{shard}.jsonl.gz")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="Smoke test: only process first N (sharded) rows.")
    ap.add_argument("--log-every", type=int, default=2000)
    args = ap.parse_args()

    justext.get_stoplist("English")  # trigger any first-call model download up front
    fs, rpath = fsspec.core.url_to_fs(args.out_file)
    if fs.exists(rpath):
        print(f"exists, skipping: {args.out_file}", flush=True)
        return

    tmp = f"/app/_router_{args.shard:04d}.jsonl.gz"
    n = pos = 0
    sim_sum = 0.0
    t0 = time.time()
    with gzip.open(tmp, "wt", encoding="utf-8") as out:
        for i, d in enumerate(iter_jsonl_gz(args.in_file)):
            if i % args.num_shards != args.shard:
                continue
            row = process_row(d, args.threshold)
            out.write(json.dumps(row) + "\n")
            n += 1
            pos += row["label"]
            sim_sum += row["lev_sim"]
            if n % args.log_every == 0:
                rate = n / (time.time() - t0)
                print(f"shard {args.shard}/{args.num_shards}: {n} rows, label1={pos} "
                      f"({100*pos/n:.1f}%), mean_sim={sim_sum/n:.3f}, {rate:.0f} rows/s", flush=True)
            if args.limit and n >= args.limit:
                break

    with open(tmp, "rb") as src, fsspec.open(args.out_file, "wb") as dst:
        dst.write(src.read())
    os.remove(tmp)
    dt = time.time() - t0
    print(f"DONE shard {args.shard}/{args.num_shards}: {n} rows, label1={pos} ({100*pos/max(n,1):.1f}%), "
          f"mean_sim={sim_sum/max(n,1):.3f}, {dt:.0f}s ({n/max(dt,1):.0f} rows/s) -> {args.out_file}", flush=True)


if __name__ == "__main__":
    main()
