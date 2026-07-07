"""Export a fastText-ready useful-vs-no_useful training set from the cascade 350k corpus.

Independent of jusText/gold: label is the row's `source` (useful / no_useful), text is the
doc (body_strip HTML, prepped exactly like the existing useful classifier:
``to_fasttext_text`` = collapse whitespace + lowercase). Writes one gzipped fastText file per
split (`__label__<source> <text>` lines), preserving the frozen train/dev/test split.

Output: {out_root}/fasttext_useful/{train,dev,test}.txt.gz  (gunzip before `fasttext supervised`).
"""
import argparse
import gzip
import json
import os
import re
import time

import fsspec

HTML_PRE = "[[ ## html ## ]]\n"
HTML_SUF = "\n\n[[ ## extraction_spec ## ]]\n"
_WS_RE = re.compile(r"\s+")
# dev/test first (small) so the first output + progress logs appear within seconds = liveness.
SPLITS = {"dev": "dev", "test": "test", "train": "train"}  # input final/<k>.jsonl.gz -> output <v>


def unwrap_html(user_content: str) -> str:
    if HTML_PRE not in user_content:
        return ""
    return user_content.split(HTML_PRE, 1)[1].split(HTML_SUF, 1)[0]


def to_fasttext_text(body_html: str) -> str:
    return _WS_RE.sub(" ", body_html).strip().lower()


def iter_jsonl_gz(path: str):
    with fsspec.open(path, "rb") as f:
        with gzip.open(f, "rt", encoding="utf-8") as g:
            for line in g:
                if line.strip():
                    yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-root", required=True, help="gs://.../high_quality_3000_distill_chat_bal350k_cascade/final")
    ap.add_argument("--out-root", required=True, help="gs://.../high_quality_3000_distill_chat_bal350k_cascade/fasttext_useful")
    args = ap.parse_args()

    for in_split, out_name in SPLITS.items():
        in_path = f"{args.final_root}/{in_split}.jsonl.gz"
        out_path = f"{args.out_root}/{out_name}.txt.gz"
        tmp = f"/app/_ft_{out_name}.txt.gz"
        n = useful = nouse = empty = 0
        seen = 0
        t0 = time.time()
        with gzip.open(tmp, "wt", encoding="utf-8") as out:
            for d in iter_jsonl_gz(in_path):
                seen += 1
                user = next((m["content"] for m in d["messages"] if m["role"] == "user"), "")
                ft = to_fasttext_text(unwrap_html(user))
                if not ft:
                    empty += 1
                    continue
                src = d.get("source", "no_useful")
                out.write(f"__label__{src} {ft}\n")
                n += 1
                if src == "useful":
                    useful += 1
                else:
                    nouse += 1
                if seen % 5000 == 0:
                    print(f"{out_name}: {seen} rows read, {n} written, {seen/(time.time()-t0):.0f} rows/s", flush=True)
        with open(tmp, "rb") as src_f, fsspec.open(out_path, "wb") as dst:
            dst.write(src_f.read())
        os.remove(tmp)
        print(f"{out_name}: wrote {n} lines (useful={useful} no_useful={nouse}, skipped_empty={empty}) -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
