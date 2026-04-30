# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Click-through viewer for the tokenized baseline pretraining caches.

Loads each baseline source's Levanter ``TreeCache`` directly, samples a
handful of tokenized documents, decodes them back through the Llama 3.1 8B
tokenizer, and emits a self-contained HTML dashboard. This shows the exact
text that the pretraining loop consumes — packing just concatenates these
decoded documents.

Usage:
    uv run python experiments/baseline_collection/view_tokenized.py \\
        --output scratch/view_tokenized.html --per-source 40

Then open the file in a browser (no server needed).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from levanter.data.text.cache import load_lm_dataset_cache
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.tokenizers import load_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("view_tokenized")

TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"

# Matches the hashes pinned in experiments/scaling_law_sweeps/curation_plan.py.
SOURCES: dict[str, dict] = {
    "resiliparse": {
        "cache": "gs://marin-us-central2/tokenized/baseline_resiliparse-7278c1/train",
        "description": "Raw resiliparse main-content extraction (no filtering).",
        "color": "#7c3aed",
        "total_tokens": 142_652_598_588,
    },
    "nemotron": {
        "cache": "gs://marin-us-central2/tokenized/baseline_nemotron-c67de9/train",
        "description": "Nemotron-CC organic (kind=actual).",
        "color": "#16a34a",
        "total_tokens": 1_919_401_016,
    },
    "nemotron_full": {
        "cache": "gs://marin-us-central2/tokenized/baseline_nemotron_full-d4e3af/train",
        "description": "Nemotron-CC organic + 5 rephraser-synthetic variants.",
        "color": "#0d9488",
        "total_tokens": 2_695_507_851,
    },
    "dclm": {
        "cache": "gs://marin-us-central2/tokenized/baseline_dclm-23e9be/train",
        "description": "DCLM-baseline, joined on WARC-Record-ID.",
        "color": "#2563eb",
        "total_tokens": 2_663_454_015,
    },
    "fineweb_edu": {
        "cache": "gs://marin-us-central2/tokenized/baseline_fineweb_edu-7a3bc5/train",
        "description": "FineWeb-Edu, joined on file_path per snapshot.",
        "color": "#dc2626",
        "total_tokens": 817_221_529,
    },
}


@dataclass
class DecodedDoc:
    source: str
    doc_index: int
    num_tokens: int
    first_token_id: int
    last_token_id: int
    text: str
    has_bos: bool
    has_eos: bool
    repeat_run: int


def max_repeat_run(arr: np.ndarray) -> int:
    if arr.size == 0:
        return 0
    diffs = np.diff(arr)
    zero_mask = diffs == 0
    if not zero_mask.any():
        return 1
    # longest run of consecutive `True` in zero_mask, plus 1 for the run length.
    best = cur = 0
    for z in zero_mask:
        cur = cur + 1 if z else 0
        if cur > best:
            best = cur
    return best + 1


async def sample_source(source: str, info: dict, n: int, seed: int, max_chars: int, tokenizer) -> list[DecodedDoc]:
    logger.info("[%s] loading cache %s", source, info["cache"])
    cache = load_lm_dataset_cache(info["cache"], TextLmDatasetFormat(), tokenizer, enforce_eos=True)
    total_docs = await cache.async_len()
    logger.info("[%s] cache contains %d documents", source, total_docs)

    rng = random.Random(seed + hash(source) % 10**6)
    indices = rng.sample(range(total_docs), min(n, total_docs))
    indices.sort()

    docs = await cache.get_batch(indices)

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    decoded: list[DecodedDoc] = []
    for idx, doc in zip(indices, docs, strict=True):
        ids = np.asarray(doc["input_ids"])
        text = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
        decoded.append(
            DecodedDoc(
                source=source,
                doc_index=int(idx),
                num_tokens=int(ids.shape[0]),
                first_token_id=int(ids[0]) if ids.size else -1,
                last_token_id=int(ids[-1]) if ids.size else -1,
                text=text[:max_chars],
                has_bos=bool(ids.size and int(ids[0]) == bos_id),
                has_eos=bool(ids.size and int(ids[-1]) == eos_id),
                repeat_run=max_repeat_run(ids),
            )
        )
    return decoded


def doc_to_dict(d: DecodedDoc, max_chars: int) -> dict:
    truncated = len(d.text) >= max_chars
    return {
        "doc_index": d.doc_index,
        "num_tokens": d.num_tokens,
        "first_token_id": d.first_token_id,
        "last_token_id": d.last_token_id,
        "has_bos": d.has_bos,
        "has_eos": d.has_eos,
        "repeat_run": d.repeat_run,
        "text": d.text,
        "truncated": truncated,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Tokenized pretraining — baseline sources</title>
<style>
  :root {
    --bg: #f5f5f7;
    --panel: #ffffff;
    --border: #d0d0d8;
    --muted: #7a7a82;
    --accent: #2563eb;
    --warn: #b54708;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; height: 100%; background: var(--bg); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; color: #1a1a1f; }
  #app { display: grid; grid-template-columns: 280px 1fr; height: 100vh; }
  #sidebar { border-right: 1px solid var(--border); background: #fafafb; display: flex; flex-direction: column; min-height: 0; }
  #sidebar-header { padding: 12px 14px; border-bottom: 1px solid var(--border); }
  #sidebar-header h1 { margin: 0 0 4px 0; font-size: 14px; }
  #sidebar-header .sub { font-size: 11px; color: var(--muted); }
  .mode-tabs { display: flex; gap: 4px; margin-top: 10px; }
  .mode-tab { flex: 1; padding: 6px 10px; border: 1px solid var(--border); background: white; font-size: 12px; cursor: pointer; border-radius: 6px; text-align: center; }
  .mode-tab.active { background: var(--accent); color: white; border-color: var(--accent); }
  #source-list { padding: 6px 0; border-bottom: 1px solid var(--border); }
  .source-row { padding: 8px 14px; cursor: pointer; font-size: 13px; display: flex; justify-content: space-between; align-items: center; }
  .source-row:hover { background: #eef2ff; }
  .source-row.active { background: #dbeafe; border-left: 3px solid var(--accent); }
  .source-row .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; vertical-align: middle; }
  .source-row .count { color: var(--muted); font-size: 11px; }
  #doc-list { flex: 1; overflow-y: auto; }
  .doc-row { padding: 8px 14px; border-bottom: 1px solid #eef; cursor: pointer; font-size: 12px; }
  .doc-row:hover { background: #eef2ff; }
  .doc-row.active { background: #dbeafe; border-left: 3px solid var(--accent); }
  .doc-row .title { font-weight: 600; font-size: 12px; }
  .doc-row .meta { color: var(--muted); font-size: 10px; margin-top: 2px; display: flex; gap: 8px; flex-wrap: wrap; }
  .flag { display: inline-block; padding: 1px 5px; border-radius: 8px; font-size: 10px; color: white; }
  .flag.ok { background: #16a34a; }
  .flag.warn { background: var(--warn); }
  #main { overflow: auto; padding: 18px; min-height: 0; }
  #main-header { display: flex; align-items: baseline; gap: 16px; margin-bottom: 12px; flex-wrap: wrap; }
  #main-header .title { font-size: 14px; font-weight: 600; }
  #main-header .keys { color: var(--muted); font-size: 11px; }
  .stats { background: white; border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; margin-bottom: 14px; font-size: 12px; }
  .stats .row { display: flex; gap: 14px; flex-wrap: wrap; }
  .stats .cell { padding-right: 10px; }
  .stats .cell b { color: #333; }
  .text-panel { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px; font-family: ui-monospace, "SF Mono", Consolas, monospace; font-size: 12.5px; line-height: 1.6; white-space: pre-wrap; max-height: calc(100vh - 240px); overflow-y: auto; }
  .truncated { color: var(--warn); font-size: 11px; margin-top: 8px; border-top: 1px dashed var(--border); padding-top: 6px; }
  .sbs-columns { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); }
  .sbs-column { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; display: flex; flex-direction: column; max-height: calc(100vh - 160px); }
  .sbs-head { padding: 10px 12px; border-bottom: 1px solid var(--border); }
  .sbs-head .src { font-weight: 600; font-size: 13px; }
  .sbs-head .m { color: var(--muted); font-size: 11px; margin-top: 3px; display: flex; gap: 10px; flex-wrap: wrap; }
  .sbs-body { padding: 12px; overflow-y: auto; white-space: pre-wrap; font-family: ui-monospace, monospace; font-size: 12px; line-height: 1.55; flex: 1; }
  .special { background: #fff7e6; color: #9a3412; padding: 0 2px; border-radius: 3px; }
</style>
</head>
<body>
<div id="app">
  <aside id="sidebar">
    <div id="sidebar-header">
      <h1>Tokenized pretraining viewer</h1>
      <div class="sub">Decoded docs sampled from each Levanter cache.</div>
      <div class="mode-tabs">
        <div class="mode-tab active" data-mode="browse">Browse</div>
        <div class="mode-tab" data-mode="sbs">Side-by-side</div>
      </div>
    </div>
    <div id="source-list"></div>
    <div id="doc-list"></div>
  </aside>
  <main id="main"></main>
</div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);

let mode = 'browse';
let activeSource = DATA.source_order[0];
let activeDoc = 0;
let sbsIndex = 0;

function esc(s) { return (s ?? '').toString().replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

function highlightSpecials(text) {
  // Visually call out BOS/EOS tokens so the user can see document boundaries.
  return esc(text)
    .replace(/&lt;\|begin_of_text\|&gt;/g, '<span class="special">&lt;|begin_of_text|&gt;</span>')
    .replace(/&lt;\|end_of_text\|&gt;/g, '<span class="special">&lt;|end_of_text|&gt;</span>')
    .replace(/&lt;\|eot_id\|&gt;/g, '<span class="special">&lt;|eot_id|&gt;</span>');
}

function renderSourceList() {
  const el = document.getElementById('source-list');
  el.innerHTML = DATA.source_order.map(s => {
    const info = DATA.sources[s];
    const n = info.docs.length;
    const active = s === activeSource ? 'active' : '';
    return `<div class="source-row ${active}" data-src="${esc(s)}">
      <span><span class="dot" style="background:${info.color}"></span><b>${esc(s)}</b></span>
      <span class="count">${n} docs</span>
    </div>`;
  }).join('');
  el.querySelectorAll('.source-row').forEach(row => {
    row.onclick = () => { activeSource = row.dataset.src; activeDoc = 0; renderSourceList(); renderDocList(); renderMain(); };
  });
}

function renderDocList() {
  const el = document.getElementById('doc-list');
  if (mode !== 'browse') { el.innerHTML = ''; return; }
  const docs = DATA.sources[activeSource].docs;
  el.innerHTML = docs.map((d, i) => {
    const flags = [];
    if (!d.has_bos) flags.push('<span class="flag warn">no BOS</span>');
    if (!d.has_eos) flags.push('<span class="flag warn">no EOS</span>');
    if (d.repeat_run > 100) flags.push(`<span class="flag warn">run ${d.repeat_run}</span>`);
    if (d.num_tokens < 32) flags.push(`<span class="flag warn">tiny</span>`);
    if (d.truncated) flags.push('<span class="flag">truncated</span>');
    return `<div class="doc-row ${i === activeDoc ? 'active' : ''}" data-i="${i}">
      <div class="title">doc #${d.doc_index.toLocaleString()}</div>
      <div class="meta">
        <span>${d.num_tokens.toLocaleString()} tok</span>
        ${flags.join('')}
      </div>
    </div>`;
  }).join('');
  el.querySelectorAll('.doc-row').forEach(row => {
    row.onclick = () => { activeDoc = parseInt(row.dataset.i, 10); renderDocList(); renderMain(); };
  });
}

function renderBrowseMain() {
  const main = document.getElementById('main');
  const info = DATA.sources[activeSource];
  const d = info.docs[activeDoc];
  if (!d) { main.innerHTML = '<div>No docs.</div>'; return; }
  main.innerHTML = `
    <div id="main-header">
      <div class="title" style="color:${info.color}">${esc(activeSource)} — doc #${d.doc_index.toLocaleString()} (${activeDoc + 1}/${info.docs.length})</div>
      <div class="keys">j/k or ↑/↓ to navigate · ← Browse | Side-by-side →</div>
    </div>
    <div class="stats">
      <div class="row">
        <div class="cell"><b>Source:</b> ${esc(activeSource)}</div>
        <div class="cell"><b>Description:</b> ${esc(info.description)}</div>
      </div>
      <div class="row">
        <div class="cell"><b>Total tokens in cache:</b> ${info.total_tokens.toLocaleString()}</div>
        <div class="cell"><b>Cache documents:</b> ${info.num_docs.toLocaleString()}</div>
        <div class="cell"><b>Median doc tokens:</b> ${info.median_tokens.toLocaleString()}</div>
      </div>
      <div class="row">
        <div class="cell"><b>This doc tokens:</b> ${d.num_tokens.toLocaleString()}</div>
        <div class="cell"><b>First id:</b> ${d.first_token_id} (BOS? ${d.has_bos ? 'yes' : 'no'})</div>
        <div class="cell"><b>Last id:</b> ${d.last_token_id} (EOS? ${d.has_eos ? 'yes' : 'no'})</div>
        <div class="cell"><b>Max repeat run:</b> ${d.repeat_run}</div>
      </div>
    </div>
    <div class="text-panel">${highlightSpecials(d.text)}</div>
    ${d.truncated ? `<div class="truncated">Text truncated for display (${d.text.length.toLocaleString()} chars shown).</div>` : ''}`;
}

function renderSbsMain() {
  const main = document.getElementById('main');
  const maxN = Math.min(...DATA.source_order.map(s => DATA.sources[s].docs.length));
  if (sbsIndex >= maxN) sbsIndex = 0;
  const columns = DATA.source_order.map(s => {
    const info = DATA.sources[s];
    const d = info.docs[sbsIndex];
    if (!d) return '';
    return `<div class="sbs-column">
      <div class="sbs-head">
        <div class="src" style="color:${info.color}">${esc(s)}</div>
        <div class="m"><span>doc #${d.doc_index.toLocaleString()}</span><span>${d.num_tokens.toLocaleString()} tok</span>${d.has_bos ? '' : '<span class="flag warn">no BOS</span>'}${d.has_eos ? '' : '<span class="flag warn">no EOS</span>'}${d.repeat_run > 100 ? `<span class="flag warn">run ${d.repeat_run}</span>` : ''}</div>
      </div>
      <div class="sbs-body">${highlightSpecials(d.text)}</div>
    </div>`;
  }).join('');
  main.innerHTML = `
    <div id="main-header">
      <div class="title">Side-by-side — slot ${sbsIndex + 1} / ${maxN}</div>
      <div class="keys">j/k or ↑/↓ to advance · each column is an independently-sampled doc from that source</div>
    </div>
    <div class="sbs-columns">${columns}</div>`;
}

function renderMain() {
  if (mode === 'browse') renderBrowseMain();
  else renderSbsMain();
}

function setMode(m) {
  mode = m;
  document.querySelectorAll('.mode-tab').forEach(t => t.classList.toggle('active', t.dataset.mode === m));
  renderDocList();
  renderMain();
}

document.querySelectorAll('.mode-tab').forEach(t => t.onclick = () => setMode(t.dataset.mode));
document.addEventListener('keydown', e => {
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) return;
  if (mode === 'browse') {
    const docs = DATA.sources[activeSource].docs;
    if (e.key === 'j' || e.key === 'ArrowDown') { activeDoc = Math.min(docs.length - 1, activeDoc + 1); renderDocList(); renderMain(); }
    if (e.key === 'k' || e.key === 'ArrowUp') { activeDoc = Math.max(0, activeDoc - 1); renderDocList(); renderMain(); }
  } else {
    const maxN = Math.min(...DATA.source_order.map(s => DATA.sources[s].docs.length));
    if (e.key === 'j' || e.key === 'ArrowDown') { sbsIndex = Math.min(maxN - 1, sbsIndex + 1); renderMain(); }
    if (e.key === 'k' || e.key === 'ArrowUp') { sbsIndex = Math.max(0, sbsIndex - 1); renderMain(); }
  }
  if (e.key === 'ArrowRight') setMode('sbs');
  if (e.key === 'ArrowLeft') setMode('browse');
});

renderSourceList();
renderDocList();
renderMain();
</script>
</body>
</html>
"""


async def collect_all(per_source: int, seed: int, max_chars: int) -> dict:
    tokenizer = load_tokenizer(TOKENIZER)

    payload = {"source_order": list(SOURCES.keys()), "sources": {}}
    for source, info in SOURCES.items():
        decoded = await sample_source(source, info, per_source, seed, max_chars, tokenizer)
        token_lens = sorted(d.num_tokens for d in decoded)
        median_tokens = token_lens[len(token_lens) // 2] if token_lens else 0

        # num_docs from the cache itself (not from sample)
        cache = load_lm_dataset_cache(info["cache"], TextLmDatasetFormat(), tokenizer, enforce_eos=True)
        num_docs = await cache.async_len()

        payload["sources"][source] = {
            "description": info["description"],
            "color": info["color"],
            "total_tokens": info["total_tokens"],
            "num_docs": num_docs,
            "median_tokens": median_tokens,
            "docs": [doc_to_dict(d, max_chars) for d in decoded],
        }
    return payload


def render_html(payload: dict, output_path: Path) -> None:
    serialized = json.dumps(payload, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("__PAYLOAD__", serialized)
    output_path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("scratch/view_tokenized.html"))
    parser.add_argument(
        "--per-source", type=int, default=40, help="Number of documents to sample and decode per source."
    )
    parser.add_argument(
        "--max-chars", type=int, default=20000, help="Truncate decoded text to at most this many characters per doc."
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    payload = asyncio.run(collect_all(args.per_source, args.seed, args.max_chars))
    render_html(payload, args.output)
    logger.info("Wrote %s (%.1f MB)", args.output, args.output.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
