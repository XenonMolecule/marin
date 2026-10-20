# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write a reviewable sample of DCLM-30B system prompts and document excerpts.

Reads the `ab-*` training shards of one scale (default `447m_9e18`, a prefix of every larger
set) and writes, under `--out`:

- `examples.md`     readable sample: prompt-wording stats, a uniform random sample, and a few
                    examples for each common prompt opening
- `examples.jsonl`  the same records in full (doc_id, system_prompt, full text, mix50 flag)

Sampling is seeded, so the same command reproduces the same file.

    uv run python experiments/scaling_law_sweeps/sample_sysprompt30b_examples.py \\
        --out gs://marin-us-central1/checkpoints/isoflop-curation/sysprompt30b_examples
"""

import argparse
import collections
import gzip
import json
import random
import re

import fsspec

BUILD = "gs://marin-us-central1/sysprompt_pretrain/dclm30b/build"
OPENING = re.compile(r"^The following document (\w+)")
EXCERPT_CHARS = 600


def opening_verb(prompt: str) -> str:
    m = OPENING.match(prompt)
    return m.group(1) if m else "(other)"


def excerpt(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= EXCERPT_CHARS else flat[:EXCERPT_CHARS].rsplit(" ", 1)[0] + " …"


def load_records(tag: str, n_shards: int, rng: random.Random) -> list[dict]:
    fs = fsspec.filesystem("gcs")
    shards = sorted(p for p in fs.ls(f"{BUILD}/{tag}/train") if "/ab-" in p)
    recs = []
    for shard in rng.sample(shards, n_shards):
        with fs.open(shard, "rb") as f, gzip.open(f, "rt", encoding="utf-8") as g:
            for line in g:
                r = json.loads(line)
                recs.append(
                    {
                        "doc_id": r["doc_id"],
                        "system_prompt": r["system_prompt"],
                        "mix50_conditioned": r["mix_conditioned"],
                        "n_tokens": r["n_tokens"],
                        "text": r["text"],
                        "shard": shard.rsplit("/", 1)[1],
                    }
                )
    return recs


def render(recs: list[dict], random_pick: list[dict], by_verb: dict[str, list[dict]], tag: str, n_shards: int) -> str:
    n = len(recs)
    verbs = collections.Counter(opening_verb(r["system_prompt"]) for r in recs)
    mean_chars = sum(len(r["system_prompt"]) for r in recs) / n
    lines = [
        "# System prompt examples",
        "",
        f"Sampled from the `{tag}` training documents ({n:,} records from {n_shards} randomly chosen "
        "2,000-document shards). This document set is a prefix of every data-efficient and epoch "
        "run's data, so these are prompts the models actually trained on.",
        "",
        "In training, each prompt is wrapped as "
        "`<|start_header_id|>system<|end_header_id|>\\n\\n{prompt}<|eot_id|>` and placed directly "
        "before the document. `sysprompt` models saw every document this way, `mix50` models saw "
        "it on the documents marked `mix50: yes`, and `doc-matched` models never saw prompts. "
        "Excerpts are the first 600 characters of the document with whitespace collapsed; the "
        "JSONL file has the full text. Documents are raw web text and some are explicit.",
        "",
        "## Prompt wording",
        "",
        f"Mean prompt length: {mean_chars:.0f} characters.",
        "",
        "| opening | share |",
        "|---|---|",
    ]
    for v, c in verbs.most_common(10):
        label = "(does not start with “The following document”)" if v == "(other)" else f"The following document {v} …"
        lines.append(f"| {label} | {100 * c / n:.1f}% |")
    lines += ["", "## Random sample", ""]
    lines += [_example(i, r) for i, r in enumerate(random_pick, 1)]
    lines += ["## Examples by opening", ""]
    for v, rs in by_verb.items():
        label = "other openings" if v == "(other)" else f"“The following document {v}”"
        lines += [f"### {label}", ""]
        lines += [_example(i, r) for i, r in enumerate(rs, 1)]
    return "\n".join(lines)


def _example(i: int, r: dict) -> str:
    mix = "yes" if r["mix50_conditioned"] else "no"
    return (
        f"**{i}. Prompt:** {r['system_prompt']}\n\n"
        f"*doc_id `{r['doc_id']}` · {r['n_tokens']:,} tokens · mix50: {mix}*\n\n"
        f"> {excerpt(r['text'])}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="447m_9e18")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, default=20)
    ap.add_argument("--random", type=int, default=40)
    ap.add_argument("--per-opening", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    recs = load_records(a.tag, a.shards, rng)
    random_pick = rng.sample(recs, a.random)
    groups = collections.defaultdict(list)
    for r in recs:
        groups[opening_verb(r["system_prompt"])].append(r)
    common = [v for v, _ in collections.Counter(opening_verb(r["system_prompt"]) for r in recs).most_common(8)]
    if "(other)" in groups and "(other)" not in common:
        common.append("(other)")
    by_verb = {v: rng.sample(groups[v], min(a.per_opening, len(groups[v]))) for v in common}

    picked = {r["doc_id"]: r for r in random_pick + [r for rs in by_verb.values() for r in rs]}
    out = a.out.rstrip("/")
    with fsspec.open(f"{out}/examples.md", "w", encoding="utf-8") as f:
        f.write(render(recs, random_pick, by_verb, a.tag, a.shards))
    with fsspec.open(f"{out}/examples.jsonl", "w", encoding="utf-8") as f:
        for r in picked.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out}/examples.md and {len(picked)} records to {out}/examples.jsonl from {len(recs):,} sampled")


if __name__ == "__main__":
    main()
