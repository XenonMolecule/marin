# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare several Gemini models on the gold extraction task, and surface the ACTUAL output diffs.

Runs each model over the noted gold docs with the same prompt, scores token-similarity to the agent
gold, and — the point of this script — computes cross-model agreement and prints unified diffs for the
docs where the models disagree most, so we can judge whether a newer model's differences are cosmetic
(whitespace/boilerplate) or substantive (content kept/dropped) and worth folding into the prompt.

Reads GEMINI_API_KEY from the environment (never stored). Model ids are resolved against the live
ListModels response, so aliases like "gemini-3-pro" map to whatever the API actually publishes.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import urllib.request

from experiments.baseline_collection.gemini_extract_eval import ROOT, gemini, similarity


def list_models() -> list[str]:
    key = os.environ["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}&pageSize=1000"
    d = json.loads(urllib.request.urlopen(url, timeout=60).read())
    return [
        m["name"].split("/")[-1]
        for m in d.get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    ]


def resolve(alias: str, available: list[str]) -> str | None:
    """Map a coarse alias ('gemini-3-pro') to a concrete published id, preferring the shortest/newest."""
    if alias in available:
        return alias
    toks = [t for t in alias.replace("gemini", "").replace("-", " ").split() if t]
    cands = [m for m in available if all(t in m for t in toks) and "gemini" in m]
    # prefer non-preview/stable, then shortest id (least suffixed), then lexicographically latest
    cands.sort(key=lambda m: ("preview" in m, "exp" in m, len(m), m))
    return cands[0] if cands else None


def load_docs() -> list[dict]:
    results = json.load(open(f"{ROOT}/extraction_results_noted.json"))
    staged = json.load(open(f"{ROOT}/noted_staged.json"))
    url2i = {s["url"]: s["_i"] for s in staged}
    docs = []
    for r in results:
        i = url2i.get(r["url"])
        gold = r.get("gold_text") or ""
        if i is None or not gold:
            continue
        # feed the FULL html when a *_full.html exists (doc_0 tabular was capped, unfair to the model)
        full = f"{ROOT}/html_noted/doc_{i}_full.html"
        path = full if os.path.exists(full) else f"{ROOT}/html_noted/doc_{i}.html"
        docs.append({"i": i, "register": r["register"], "gold": gold, "html": open(path).read()})
    return docs


def unified(a: str, b: str, na: str, nb: str, max_lines: int) -> str:
    diff = difflib.unified_diff(a.splitlines(), b.splitlines(), fromfile=na, tofile=nb, lineterm="", n=1)
    out = []
    for ln in diff:
        out.append(ln)
        if len(out) >= max_lines:
            out.append(f"... (diff truncated at {max_lines} lines)")
            break
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gemini-2.5-flash,gemini-3-pro,gemini-3-flash")
    ap.add_argument("--diff-top", type=int, default=3, help="dump diffs for the N most model-divergent docs")
    ap.add_argument("--diff-lines", type=int, default=60)
    ap.add_argument("--out-tag", default="compare")
    args = ap.parse_args()

    prompt = open(f"{ROOT}/prompt_v1.md").read()
    available = list_models()
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    models: dict[str, str] = {}
    for a in requested:
        r = resolve(a, available)
        if r is None:
            print(f"  !! '{a}' not found in ListModels — skipping. (available gemini ids: "
                  f"{[m for m in available if 'gemini' in m]})")
            continue
        models[a] = r
        print(f"  resolved {a!r} -> {r!r}")
    if not models:
        print("no usable models resolved; aborting")
        return 1

    docs = load_docs()
    os.makedirs(f"{ROOT}/gemini_out", exist_ok=True)
    # outputs[model_id][i] = text
    outputs: dict[str, dict[int, str]] = {mid: {} for mid in models.values()}
    for d in docs:
        for alias, mid in models.items():
            out = gemini(prompt, d["html"], mid)
            outputs[mid][d["i"]] = out
            json.dump({"text": out}, open(f"{ROOT}/gemini_out/{args.out_tag}_{mid}_doc_{d['i']}.json", "w"))

    # per-doc table + per-model averages
    mids = list(models.values())
    print("\n=== token-similarity to agent gold ===")
    header = f"{'doc':<6}{'register':<14}" + "".join(f"{a[:16]:>18}" for a in models)
    print(header)
    per_model_sum = {mid: 0.0 for mid in mids}
    rows = []
    for d in docs:
        cells = []
        row = {"i": d["i"], "register": d["register"], "gold_len": len(d["gold"]), "sim": {}, "len": {}}
        for alias, mid in models.items():
            s = similarity(outputs[mid][d["i"]], d["gold"])
            per_model_sum[mid] += s
            cells.append(f"{s:>10.3f}/{len(outputs[mid][d['i']]):>6}")
            row["sim"][mid] = round(s, 3)
            row["len"][mid] = len(outputs[mid][d["i"]])
        rows.append(row)
        print(f"doc_{d['i']:<2} {d['register']:<14}" + "".join(f"{c:>18}" for c in cells))
    print("-" * len(header))
    print(f"{'AVG':<6}{'':<14}" + "".join(f"{per_model_sum[mid]/len(docs):>18.3f}" for mid in mids))

    # cross-model agreement (only meaningful with >=2 models): rank docs by lowest pairwise sim
    if len(mids) >= 2:
        base = mids[0]
        div = []
        for d in docs:
            worst = min(similarity(outputs[base][d["i"]], outputs[m][d["i"]]) for m in mids[1:])
            div.append((worst, d))
        div.sort(key=lambda x: x[0])
        print(f"\n=== actual output diffs for the {args.diff_top} most model-divergent docs "
              f"(baseline {base}) ===")
        for worst, d in div[: args.diff_top]:
            print(f"\n##### doc_{d['i']} [{d['register']}] cross-model sim={worst:.3f} "
                  f"(low = models disagree) #####")
            for m in mids[1:]:
                print(f"\n--- {base}  →  {m} ---")
                print(unified(outputs[base][d["i"]], outputs[m][d["i"]], base, m, args.diff_lines))

    json.dump(
        {"models": models, "rows": rows, "avg": {mid: per_model_sum[mid] / len(docs) for mid in mids}},
        open(f"{ROOT}/gemini_compare_{args.out_tag}.json", "w"),
        indent=1,
    )
    print(f"\nwrote {ROOT}/gemini_compare_{args.out_tag}.json  (+ per-model outputs in gemini_out/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
