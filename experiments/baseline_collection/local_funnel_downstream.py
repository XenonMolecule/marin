"""LOCAL, no-job computation of funnel stages 3 (router) + 4 (1.7B context length)
on the EXISTING ModernBERT survivors (the local 350k dev/test .txt.gz = BERT-kept docs).

For each survivor: route with router.bin; for the 1.7B-routed (P(extractable) < thr)
tokenize the extraction prompt with the Qwen3-1.7B tokenizer and bucket the input length.
"""
import argparse
import gzip
import json
import re

import fasttext
from transformers import AutoTokenizer

from experiments.baseline_collection.extraction_specs import get_spec

ROUTER_POS = "__label__extractable"
CTX = (8192, 16384, 32768, 40960)
_SPEC = get_spec("high_quality")
SYS, UTMPL = _SPEC.system_message, _SPEC.extraction_template
_SCRIPT = re.compile(r"<script\b[^>]*>.*?</script\s*>", re.IGNORECASE | re.DOTALL)
_BODY = re.compile(r"<body\b[^>]*>(.*?)</body\s*>", re.IGNORECASE | re.DOTALL)
_WS = re.compile(r"\s+")

# stage-0 rules (vendored from classifier.py) — to compute D = rules∩ft∩BERT.
_TAG = re.compile(r"<[^>]+>")
_SCRIPT_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_SENT = re.compile(r"[.!?]")
WPS_MAX, LWR_MAX, TTM_MIN, SENT_MIN, VW_MIN = 115.0, 0.923, 0.085, 4, 84


def rules_keep(html):
    h = html[:200_000]
    if not h:
        return False
    visible = _TAG.sub(" ", _COMMENT.sub(" ", _SCRIPT_STYLE.sub(" ", h)))
    words = visible.split()
    nw = len(words)
    if nw < VW_MIN:
        return False
    if len(visible.strip()) / len(h) < TTM_MIN:
        return False
    ns = len(_SENT.findall(visible))
    if ns < SENT_MIN:
        return False
    if nw / ns > WPS_MAX:
        return False
    if sum(1 for w in words if len(w) >= 3) / nw > LWR_MAX:
        return False
    return True


def body_strip(html):
    bodies = [m.group(1) for m in _BODY.finditer(_SCRIPT.sub("", html))]
    return "".join(bodies) if bodies else html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-file", required=True)
    ap.add_argument("--router", required=True)
    ap.add_argument("--threshold", type=float, default=0.355820894241333)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    router = fasttext.load_model(args.router)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

    # D = rules∩ft∩BERT survivors (apply rules to the BERT survivors); routing+context on D.
    n_bert = 0            # ft∩BERT survivors (= all lines in this file)
    n_rules = 0           # D: also pass rules
    n_jt = n_llm = 0      # routing of D
    by_src = {"useful": [0, 0], "no_useful": [0, 0]}  # [justext, 1.7b] among D
    over = {str(t): 0 for t in CTX}
    lens = []
    with gzip.open(args.in_file, "rt", encoding="utf-8") as g:
        for line in g:
            if not line.strip():
                continue
            lab, _, text = line.partition(" ")
            src = "useful" if lab == "__label__useful" else "no_useful"
            n_bert += 1
            if not rules_keep(text):
                continue
            n_rules += 1
            ftt = _WS.sub(" ", text).strip().lower()
            rp = 0.0
            for prob, l in router.f.predict(ftt, -1, 0.0, "strict"):
                if l == ROUTER_POS:
                    rp = float(prob)
                    break
            if rp >= args.threshold:
                n_jt += 1
                by_src[src][0] += 1
            else:
                n_llm += 1
                by_src[src][1] += 1
                bs = body_strip(text)
                ids = tok.apply_chat_template(
                    [{"role": "system", "content": SYS}, {"role": "user", "content": UTMPL.format(example=bs)}],
                    add_generation_prompt=True, tokenize=True,
                )
                lens.append(len(ids))
                for t in CTX:
                    if len(ids) > t:
                        over[str(t)] += 1
            if args.limit and n_bert >= args.limit:
                break

    lens.sort()
    pct = lambda q: lens[min(len(lens) - 1, int(q * len(lens)))] if lens else 0
    print(json.dumps({
        "in_file": args.in_file,
        "n_bert_survivors": n_bert,
        "n_after_rules_D": n_rules,
        "rules_discard_of_bert_survivors_pct": round(100 * (n_bert - n_rules) / n_bert, 2),
        "route_justext": n_jt, "route_1p7b": n_llm,
        "pct_justext_of_D": round(100 * n_jt / n_rules, 2), "pct_1p7b_of_D": round(100 * n_llm / n_rules, 2),
        "by_source_D_[justext,1p7b]": by_src,
        "ctx_over_among_1p7b": {k: f"{v} ({100*v/n_llm:.1f}%)" for k, v in over.items()},
        "tok_len_1p7b": {"mean": round(sum(lens) / len(lens)) if lens else 0, "p50": pct(.5), "p90": pct(.9), "p99": pct(.99), "max": lens[-1] if lens else 0},
    }, indent=2))


if __name__ == "__main__":
    main()
