# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize JSONL sources for system-prompt ablations on the DCLM-30B corpus.

Three arms per scale, differing only in what reaches the tokenizer:

    A  sysprompt      docs with [S] prepended       -> tokenize conditioned_text
    C  doc-matched    the SAME docs, no [S]         -> tokenize text
    B  token-matched  A's docs + extra docs, no [S] -> tokenize text

SCALE-GENERIC BY DESIGN. Nothing here hardcodes a token budget. Give it a FLOP
budget and a width and it derives the token target from ``completed_adamh`` — the
same heuristic the trainer uses — so the dataset can never drift from the run it
feeds::

    # 998M @ 9e19  -> 14.9B tokens
    ... build_sysprompt30b_dataset.py emit --budget 9e19   --hidden-dim 1536 --tag 998m_9e19
    # 998M @ 1.8e20 -> 29.9B tokens
    ... build_sysprompt30b_dataset.py emit --budget 1.8e20 --hidden-dim 1536 --tag 998m_1p8e20

Two phases, so adding a scale never re-scans the corpus twice:

  ``index``  one parallel pass over the corpus -> per-block doc counts and llama3
             token sums. Scale-independent, written ONCE, reused by every emit.
  ``emit``   pure arithmetic on the index picks the block sets, then writes only
             the shards that scale actually needs (no wasted storage, and no
             reliance on globs to express "the first K shards").

Layout per scale — ``default_tokenize`` takes a single glob, and duplicating tens
of GB of document text to give B its own copy would be wasteful::

    build/{tag}/train/ab-{block:06d}.jsonl.gz     A: conditioned_text ; C: text
    build/{tag}/train/extra-{block:06d}.jsonl.gz  picked up ONLY by B's wider glob

    A, C  ->  build/{tag}/train/ab-*.jsonl.gz
    B     ->  build/{tag}/train/*.jsonl.gz

Each ``ab`` record carries BOTH ``text`` and ``conditioned_text``, so A and C are
the same documents by construction rather than by a matching step that could drift.

Token accounting uses the corpus's per-doc ``n_tokens``, verified to be the llama3
count exactly (300-doc sample, ratio 1.0000). Estimates here are for SELECTION
only — train steps come from the cache's ``.stats.json:total_tokens`` after
tokenization, which accounts for BOS/EOS.

Run in-region (us-central1): corpus, gen blocks and output all live there, and a
cross-region read of a 27 GB corpus is exactly what the cost rules forbid.

Usage::

    uv run --no-sync iris --cluster marin job run --no-wait \\
        --region us-central1 --cpu 32 --memory 64GB --disk 100GB \\
        --priority interactive --extra cpu --enable-extra-resources \\
        --job-name build-sysprompt30b \\
        -- python experiments/scaling_law_sweeps/build_sysprompt30b_dataset.py index
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import multiprocessing as mp
import re

import fsspec
import zstandard

logger = logging.getLogger(__name__)

ROOT = "gs://marin-us-central1/sysprompt_pretrain/dclm30b"
CORPUS_PARTS = f"{ROOT}/corpus/parts"
GEN_ALL = f"{ROOT}/gen_all"
BUILD = f"{ROOT}/build"
INDEX_PATH = f"{BUILD}/index.json"

BLOCK_DOCS = 2000
BLOCKS_PER_PART = 50
DOCS_PER_PART = BLOCK_DOCS * BLOCKS_PER_PART  # 100,000 — verified, see process_part
SEQ_LEN = 4096

# Llama-3 system header slot. Same wrapper as the earlier [S][D] project so
# pretraining conditioning lines up with inference-time system messages; the
# markers map to reserved llama3 ids. Do NOT add BOS — the tokenizer does that.
S_PREFIX = "<|start_header_id|>system<|end_header_id|>\n\n"
S_SUFFIX = "<|eot_id|>"
# Measured on this corpus: 16.0 tokens of prompt + 5 of wrapper.
SYS_TOKENS = 21


def _fs():
    return fsspec.core.url_to_fs(ROOT)[0]


def labeled_blocks(fs) -> set[int]:
    out = set()
    for p in fs.ls(GEN_ALL, detail=False):
        m = re.search(r"block-(\d+)\.jsonl\.zst$", p)
        if m:
            out.add(int(m.group(1)))
    return out


# A malformed line is tolerated, but only up to this share of a block. Below it
# we are skipping the known bad-record shape; above it something is actually
# wrong with the file and we want the failure, not a quietly smaller dataset.
MAX_BAD_LINE_FRACTION = 0.01


def _read_zst_jsonl(fs, path: str) -> list[dict]:
    """Parse a gen block, skipping records that are not valid JSON.

    SC's generator writes an error record when a doc fails to parse, and embeds
    the raw document text — including literal newlines — in the ``error`` field.
    That breaks JSONL: one record spans several lines and none of the fragments
    parse. Observed in exactly 1 of 6,528 blocks (block-000073, 2,000 records
    across 2,003 lines). Joining the fragments cannot repair it, because a JSON
    string may not contain a raw newline.

    Skipping is safe for our purposes: such records carry an ``error`` field and
    no ``system_prompt``, so they were never usable. The threshold keeps this
    from masking genuine corruption — which is what byte-size checks miss, since
    the file here was byte-identical to SC's and had a matching md5.
    """
    with fs.open(path, "rb") as f:
        data = zstandard.ZstdDecompressor().stream_reader(f).read()
    lines = [l for l in data.decode().splitlines() if l.strip()]
    out, bad = [], 0
    for l in lines:
        try:
            out.append(json.loads(l))
        except json.JSONDecodeError:
            bad += 1
    if lines and bad > max(4, int(len(lines) * MAX_BAD_LINE_FRACTION)):
        raise SystemExit(
            f"{path}: {bad}/{len(lines)} lines unparseable — above the "
            f"{MAX_BAD_LINE_FRACTION:.0%} tolerance, refusing to build from it."
        )
    if bad:
        logger.warning("%s: skipped %d unparseable line(s)", path.rsplit("/", 1)[-1], bad)
    return out


def _prompts_for_part(fs, part: int) -> dict[int, tuple[str, str]]:
    """idx -> (system prompt, doc_id) for every labeled block in this part."""
    prompts: dict[int, tuple[str, str]] = {}
    for block in range(part * BLOCKS_PER_PART, (part + 1) * BLOCKS_PER_PART):
        path = f"{GEN_ALL}/block-{block:06d}.jsonl.zst"
        if not fs.exists(path):
            continue
        for r in _read_zst_jsonl(fs, path):
            s = r.get("system_prompt")
            if s:
                prompts[int(r["idx"])] = (s, r.get("doc_id"))
    return prompts


def _iter_part(fs, part: int):
    """Yield (idx, record) for a corpus part.

    Corpus records carry no global index — `shard`/`line` are DCLM source
    provenance. The index is POSITIONAL: the sampler wrote docs in strict
    ascending order, exactly DOCS_PER_PART per part. Verified: part 150 line 0 ==
    gen block 7500 idx 15,000,000 with the same doc_id, and
    239*100,000 + 6,067 == 23,906,067 total docs.
    """
    src = f"{CORPUS_PARTS}/part-{part:05d}.jsonl.zst"
    if not fs.exists(src):
        return
    with fs.open(src, "rb") as f:
        reader = zstandard.ZstdDecompressor().stream_reader(f)
        for i, line in enumerate(io.TextIOWrapper(reader, encoding="utf-8")):
            if line.strip():
                yield part * DOCS_PER_PART + i, json.loads(line)


# ---------------------------------------------------------------- index phase


def index_part(part: int) -> dict:
    """Per-block doc counts and llama3 token sums for one corpus part."""
    fs = _fs()
    labeled = labeled_blocks(fs)
    docs: dict[int, int] = {}
    toks: dict[int, int] = {}
    for idx, d in _iter_part(fs, part):
        b = idx // BLOCK_DOCS
        docs[b] = docs.get(b, 0) + 1
        toks[b] = toks.get(b, 0) + int(d.get("n_tokens") or 0)
    return {
        "part": part,
        "blocks": {str(b): {"docs": docs[b], "tokens": toks[b], "labeled": b in labeled} for b in sorted(docs)},
    }


def cmd_index(a) -> None:
    fs = _fs()
    parts = [p for p in range(240) if a.part_min <= p <= a.part_max]
    with mp.Pool(a.workers) as pool:
        results = pool.map(index_part, parts)

    blocks: dict[str, dict] = {}
    for r in results:
        blocks.update(r["blocks"])
    total_docs = sum(v["docs"] for v in blocks.values())
    total_tokens = sum(v["tokens"] for v in blocks.values())
    lab = {k: v for k, v in blocks.items() if v["labeled"]}
    payload = {
        "blocks": blocks,
        "total_blocks": len(blocks),
        "total_docs": total_docs,
        "total_tokens": total_tokens,
        "labeled_blocks": len(lab),
        "labeled_docs": sum(v["docs"] for v in lab.values()),
        "labeled_tokens": sum(v["tokens"] for v in lab.values()),
        "sys_tokens_per_doc": SYS_TOKENS,
    }
    with fs.open(a.index, "w") as f:
        json.dump(payload, f)
    logger.info(
        "index: %d blocks, %d docs, %.3fB tokens; labeled %d blocks / %d docs / %.3fB tokens",
        len(blocks),
        total_docs,
        total_tokens / 1e9,
        len(lab),
        payload["labeled_docs"],
        payload["labeled_tokens"] / 1e9,
    )


# ----------------------------------------------------------------- emit phase


def target_tokens_for(budget: float, hidden_dim: int) -> tuple[float, int, int]:
    """(tokens, batch_size, train_steps) for a FLOP budget — the SAME source of
    truth the trainer uses, so dataset and run can never disagree."""
    from experiments.scaling_law_sweeps.fixed_model_plan import _candidate_for_fixed_model

    cand = _candidate_for_fixed_model(hidden_dim, budget, seq_len=SEQ_LEN)
    if cand is None:
        raise SystemExit(f"no candidate config for hidden_dim={hidden_dim} at budget={budget:.3g}")
    return cand.tokens, cand.batch_size, cand.train_steps


def select_blocks(index: dict, target_tokens: float) -> dict:
    """Choose block sets for the three arms from the index (pure arithmetic)."""
    blocks = index["blocks"]
    labeled = sorted((int(k) for k, v in blocks.items() if v["labeled"]))
    unlabeled = sorted((int(k) for k, v in blocks.items() if not v["labeled"]))

    ab, ab_doc_tok, ab_cond_tok, ab_docs = [], 0, 0, 0
    for b in labeled:
        v = blocks[str(b)]
        if ab_cond_tok >= target_tokens:
            break
        ab.append(b)
        ab_docs += v["docs"]
        ab_doc_tok += v["tokens"]
        ab_cond_tok += v["tokens"] + SYS_TOKENS * v["docs"]

    # B matches A's TOKEN count using A's docs plus extra ones. Arm B tokenizes
    # `text`, so an extra block does NOT need a system prompt — prefer unlabeled
    # blocks (keeping labeled ones free for A at larger scales) but fall back to
    # leftover labeled blocks. Without that fallback this silently under-fills B
    # once labeling coverage is high: at 1.8e20 arm A already consumes ~98% of
    # the corpus, and at 100% coverage there would be no unlabeled blocks at all.
    ab_set = set(ab)
    candidates = unlabeled + [b for b in labeled if b not in ab_set]
    extra, extra_tok, extra_docs = [], 0, 0
    need = ab_cond_tok - ab_doc_tok  # == SYS_TOKENS * ab_docs
    for b in candidates:
        if extra_tok >= need:
            break
        v = blocks[str(b)]
        extra.append(b)
        extra_docs += v["docs"]
        extra_tok += v["tokens"]
    b_shortfall = max(0.0, need - extra_tok)

    return {
        "ab_blocks": ab,
        "ab_docs": ab_docs,
        "arm_A_tokens_est": ab_cond_tok,
        "arm_C_tokens_est": ab_doc_tok,
        "extra_blocks": extra,
        "extra_docs": extra_docs,
        "arm_B_tokens_est": ab_doc_tok + extra_tok,
        "shortfall": max(0.0, target_tokens - ab_cond_tok),
        "b_shortfall": b_shortfall,
    }


def _check_feasible(index: dict, sel: dict, target_tokens: float, tag: str) -> None:
    """Refuse to build an undersized dataset.

    A dataset that quietly falls short produces a run that looks fine and is
    scientifically wrong — the arms stop being matched and the FLOP budget is not
    what the name says. Fail here, loudly, with the amount still needed.

    The two shortfalls have DIFFERENT fixes, so they are reported separately:
      arm A  -> not enough LABELED docs   -> generate more system prompts
      arm B  -> not enough DOCS at all    -> add more documents to the corpus
    """
    blocks = index["blocks"]
    docs = sum(v["docs"] for v in blocks.values()) or 1
    tok_per_doc = sum(v["tokens"] for v in blocks.values()) / docs

    problems = []
    if sel["shortfall"] > 0:
        need_tok = sel["shortfall"]
        need_docs = need_tok / (tok_per_doc + SYS_TOKENS)
        problems.append(
            f"ARM A is short {need_tok / 1e9:.3f}B tokens.\n"
            f"    have {len(sel['ab_blocks'])} labeled blocks "
            f"({sel['ab_docs'] / 1e6:.2f}M docs, {sel['arm_A_tokens_est'] / 1e9:.3f}B tokens); "
            f"target is {target_tokens / 1e9:.3f}B.\n"
            f"    need ~{need_docs / 1e6:.2f}M MORE labeled docs "
            f"(~{need_docs / BLOCK_DOCS:.0f} more blocks of system prompts)."
        )
    if sel["b_shortfall"] > 0:
        need_tok = sel["b_shortfall"]
        problems.append(
            f"ARM B is short {need_tok / 1e9:.3f}B tokens.\n"
            f"    the corpus has no documents left to token-match arm A "
            f"(labeled+unlabeled are exhausted).\n"
            f"    need ~{need_tok / tok_per_doc / 1e6:.2f}M MORE documents in the corpus."
        )
    if problems:
        raise SystemExit(f"\nCANNOT BUILD {tag} — not enough data:\n\n  " + "\n\n  ".join(problems) + "\n")


def emit_part(args: tuple[int, str, list[int], list[int]]) -> tuple[int, int, int]:
    part, out_dir, ab_blocks, extra_blocks = args
    fs = _fs()
    want_ab, want_extra = set(ab_blocks), set(extra_blocks)
    if not (want_ab | want_extra):
        return (part, 0, 0)
    prompts = _prompts_for_part(fs, part)
    ab_buf: dict[int, list[str]] = {}
    ex_buf: dict[int, list[str]] = {}
    n_ab = n_ex = 0

    for idx, d in _iter_part(fs, part):
        b = idx // BLOCK_DOCS
        if b not in want_ab and b not in want_extra:
            continue
        hit = prompts.get(idx)
        # Turn any drift in the positional mapping into a loud failure rather
        # than silently pairing a system prompt with the wrong document.
        if hit is not None and hit[1] and d.get("doc_id") != hit[1]:
            raise SystemExit(f"doc_id mismatch part={part} idx={idx}: corpus={d.get('doc_id')} gen={hit[1]}")
        rec = {"doc_id": d.get("doc_id"), "idx": idx, "n_tokens": d.get("n_tokens"), "text": d["text"]}
        if b in want_ab and hit:
            rec["system_prompt"] = hit[0]
            rec["conditioned_text"] = f"{S_PREFIX}{hit[0]}{S_SUFFIX}{d['text']}"
            ab_buf.setdefault(b, []).append(json.dumps(rec))
            n_ab += 1
        elif b in want_extra:
            ex_buf.setdefault(b, []).append(json.dumps(rec))
            n_ex += 1

    for prefix, buf in (("ab", ab_buf), ("extra", ex_buf)):
        for b, rows in buf.items():
            with fs.open(f"{out_dir}/{prefix}-{b:06d}.jsonl.gz", "wb") as f:
                with gzip.GzipFile(fileobj=f, mode="wb") as gz:
                    gz.write(("\n".join(rows) + "\n").encode())
    return (part, n_ab, n_ex)


def cmd_emit(a) -> None:
    fs = _fs()
    with fs.open(a.index, "r") as f:
        index = json.load(f)

    if a.target_tokens:
        tokens, batch, steps = a.target_tokens, a.batch_size, 0
    else:
        tokens, batch, steps = target_tokens_for(a.budget, a.hidden_dim)
    logger.info("target: %.4fB tokens (batch=%d, canonical steps=%d)", tokens / 1e9, batch, steps)

    sel = select_blocks(index, tokens)
    _check_feasible(index, sel, tokens, a.tag)

    if a.dry_run:
        logger.info(
            "DRY RUN ok — %s is feasible: A=%.3fB B=%.3fB C=%.3fB tokens "
            "(%d ab blocks + %d extra blocks). Nothing written.",
            a.tag,
            sel["arm_A_tokens_est"] / 1e9,
            sel["arm_B_tokens_est"] / 1e9,
            sel["arm_C_tokens_est"] / 1e9,
            len(sel["ab_blocks"]),
            len(sel["extra_blocks"]),
        )
        return

    out_dir = f"{BUILD}/{a.tag}/train"
    parts = sorted({b // BLOCKS_PER_PART for b in sel["ab_blocks"] + sel["extra_blocks"]})
    payload = [
        (
            p,
            out_dir,
            [b for b in sel["ab_blocks"] if b // BLOCKS_PER_PART == p],
            [b for b in sel["extra_blocks"] if b // BLOCKS_PER_PART == p],
        )
        for p in parts
    ]
    with mp.Pool(a.workers) as pool:
        results = pool.map(emit_part, payload)

    def _steps(tok: float) -> int:
        return int(tok // (batch * SEQ_LEN))

    manifest = {
        "tag": a.tag,
        "budget": a.budget,
        "hidden_dim": a.hidden_dim,
        "target_tokens": tokens,
        "batch_size": batch,
        "canonical_steps": steps,
        "emitted_ab_docs": sum(r[1] for r in results),
        "emitted_extra_docs": sum(r[2] for r in results),
        "globs": {
            "A_sysprompt": f"{out_dir}/ab-*.jsonl.gz  (text_key=conditioned_text)",
            "C_doc_matched": f"{out_dir}/ab-*.jsonl.gz  (text_key=text)",
            "B_token_matched": f"{out_dir}/*.jsonl.gz  (text_key=text)",
        },
        "estimates": {
            "A_tokens": sel["arm_A_tokens_est"],
            "A_steps": _steps(sel["arm_A_tokens_est"]),
            "B_tokens": sel["arm_B_tokens_est"],
            "B_steps": _steps(sel["arm_B_tokens_est"]),
            "C_tokens": sel["arm_C_tokens_est"],
            "C_steps": _steps(sel["arm_C_tokens_est"]),
        },
        "ab_blocks": len(sel["ab_blocks"]),
        "extra_blocks": len(sel["extra_blocks"]),
        "shortfall_tokens": sel["shortfall"],
        "b_shortfall_tokens": sel["b_shortfall"],
        "note": "steps are ESTIMATES; recompute from each cache's .stats.json:total_tokens",
    }
    with fs.open(f"{BUILD}/{a.tag}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(
        "emitted ab_docs=%d extra_docs=%d -> %s", manifest["emitted_ab_docs"], manifest["emitted_extra_docs"], out_dir
    )


# ---------------------------------------------------------------- verify phase


def _verify_block(path: str) -> tuple[str, int, str | None]:
    fs = _fs()
    try:
        with fs.open(path, "rb") as f:
            data = zstandard.ZstdDecompressor().stream_reader(f).read()
        lines = data.decode().splitlines()
        for l in lines:
            if l.strip():
                json.loads(l)
        return (path, len([l for l in lines if l.strip()]), None)
    except Exception as e:
        return (path, -1, f"{type(e).__name__}: {str(e)[:140]}")


def cmd_verify(a) -> None:
    """Parse every gen block. Byte-size checks cannot see a corrupt payload —
    a block can be exactly the right size and still contain a truncated line,
    which surfaces much later as a JSONDecodeError deep inside emit."""
    fs = _fs()
    paths = [p for p in fs.ls(GEN_ALL, detail=False) if re.search(r"block-\d+\.jsonl\.zst$", p)]
    logger.info("verifying %d gen blocks", len(paths))
    with mp.Pool(a.workers) as pool:
        res = pool.map(_verify_block, paths)
    bad = [r for r in res if r[2]]
    odd = [r for r in res if r[2] is None and r[1] != BLOCK_DOCS]
    logger.info("UNPARSEABLE: %d", len(bad))
    for p, _, e in sorted(bad)[:25]:
        logger.info("   %s: %s", p.rsplit("/", 1)[-1], e)
    logger.info("UNEXPECTED ROW COUNT (want %d): %d", BLOCK_DOCS, len(odd))
    for p, n, _ in sorted(odd)[:25]:
        logger.info("   %s: rows=%d", p.rsplit("/", 1)[-1], n)
    with fs.open(f"{BUILD}/gen_verify.json", "w") as f:
        json.dump(
            {
                "checked": len(paths),
                "unparseable": [[p, e] for p, _, e in bad],
                "wrong_rows": [[p, n] for p, n, _ in odd],
            },
            f,
            indent=2,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="One reusable pass: per-block doc/token counts.")
    pi.add_argument("--part-min", type=int, default=0)
    pi.add_argument("--part-max", type=int, default=239)
    pi.add_argument("--workers", type=int, default=28)
    pi.add_argument("--index", default=INDEX_PATH)
    pi.set_defaults(func=cmd_index)

    pe = sub.add_parser("emit", help="Write the shards one scale needs.")
    pe.add_argument("--budget", type=float, default=9e19, help="FLOP budget, e.g. 9e19 or 1.8e20.")
    pe.add_argument("--hidden-dim", type=int, default=1536, help="1536 = 998M.")
    pe.add_argument("--tag", required=True, help="Scale tag, e.g. 998m_9e19.")
    pe.add_argument("--target-tokens", type=float, default=None, help="Override the derived budget.")
    pe.add_argument("--batch-size", type=int, default=64, help="Only used with --target-tokens.")
    pe.add_argument("--workers", type=int, default=28)
    pe.add_argument("--index", default=INDEX_PATH)
    pe.add_argument("--dry-run", action="store_true", help="Check feasibility for this scale and exit without writing.")
    pe.set_defaults(func=cmd_emit)

    pv = sub.add_parser("verify", help="Parse every gen block; find corrupt payloads.")
    pv.add_argument("--workers", type=int, default=24)
    pv.set_defaults(func=cmd_verify)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
