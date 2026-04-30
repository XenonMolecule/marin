# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render a self-contained HTML viewer for Nemotron-CC records pulled from the cluster.

Pairs with ``nemotron_explorer.py``, which writes ``samples.jsonl`` and
``lookups.jsonl`` to GCS.

Usage
-----
    # 1. Pull the small outputs from the cluster (few MB total).
    mkdir -p scratch/nemotron_explorer
    gcloud storage cp \\
        gs://marin-us-central2/scratch/nemotron_explorer/samples.jsonl \\
        gs://marin-us-central2/scratch/nemotron_explorer/lookups.jsonl \\
        gs://marin-us-central2/scratch/nemotron_explorer/summary.json \\
        scratch/nemotron_explorer/

    # 2. Render the viewer and open it.
    uv run python experiments/baseline_collection/nemotron_viewer.py \\
        --input-dir scratch/nemotron_explorer \\
        --output scratch/nemotron_viewer.html
    open scratch/nemotron_viewer.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HTML_TEMPLATE = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Nemotron-CC viewer</title>
<style>
  :root { --bg:#f5f5f7; --panel:#fff; --border:#d0d0d8; --muted:#7a7a82; --accent:#2563eb; --warn:#b54708; --ok:#065f46; --hit:#fef3c7; }
  * { box-sizing: border-box; }
  html,body { margin:0; padding:0; height:100%; background:var(--bg); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif; color:#1a1a1f; }
  #app { display:grid; grid-template-columns:380px 1fr; height:100vh; }
  aside { border-right:1px solid var(--border); background:#fafafb; display:flex; flex-direction:column; min-height:0; }
  #hdr { padding:12px 14px; border-bottom:1px solid var(--border); }
  #hdr h1 { margin:0 0 4px; font-size:15px; }
  #hdr .sub { font-size:12px; color:var(--muted); }
  #tabs { display:flex; border-bottom:1px solid var(--border); }
  .tab { flex:1; padding:8px 10px; text-align:center; cursor:pointer; font-size:12px; color:var(--muted); border-bottom:2px solid transparent; }
  .tab.active { color:var(--accent); border-bottom-color:var(--accent); background:white; font-weight:600; }
  #filters { padding:8px 14px; border-bottom:1px solid var(--border); font-size:12px; display:flex; flex-direction:column; gap:6px; }
  #filters input[type="search"] { padding:5px 8px; border:1px solid var(--border); border-radius:6px; font-size:12px; }
  #filters select { padding:5px 6px; border:1px solid var(--border); border-radius:6px; font-size:12px; background:white; width:100%; }
  #list { flex:1; overflow-y:auto; }
  .item { padding:8px 14px; border-bottom:1px solid #eef; cursor:pointer; font-size:12px; }
  .item:hover { background:#eef2ff; }
  .item.active { background:#dbeafe; border-left:3px solid var(--accent); }
  .item .url { color:var(--accent); word-break:break-all; font-size:11px; }
  .item .meta { color:var(--muted); font-size:10px; margin-top:3px; display:flex; gap:4px; flex-wrap:wrap; }
  .chip { display:inline-block; padding:1px 5px; border-radius:9px; font-size:10px; background:#e5e7eb; color:#1f2937; }
  .chip.actual { background:#dcfce7; color:#065f46; }
  .chip.synthetic { background:#fef3c7; color:#92400e; }
  main { overflow:auto; padding:18px; min-height:0; }
  #mh { display:flex; align-items:baseline; gap:14px; margin-bottom:10px; flex-wrap:wrap; }
  #mh .url { font-size:13px; color:var(--accent); word-break:break-all; flex:1; min-width:300px; }
  #mh .keys { color:var(--muted); font-size:11px; }
  #meta-row { font-size:12px; color:var(--muted); display:flex; gap:14px; flex-wrap:wrap; margin-bottom:10px; }
  #meta-row span b { color:#333; }
  #body { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:14px; white-space:pre-wrap; font-family:ui-monospace,"SF Mono",Consolas,monospace; font-size:12.5px; line-height:1.55; }
  mark { background:var(--hit); padding:0 2px; border-radius:2px; }
  .empty { color:var(--muted); font-style:italic; padding:24px; text-align:center; }
  details.summary { margin:8px 14px; font-size:11px; color:var(--muted); }
  details.summary pre { max-height:200px; overflow:auto; background:#f1f5f9; padding:8px; border-radius:6px; font-size:10.5px; }
</style>
</head><body>
<div id="app">
  <aside>
    <div id="hdr">
      <h1>Nemotron-CC raw-data viewer</h1>
      <div class="sub" id="sub"></div>
    </div>
    <div id="tabs">
      <div class="tab" data-tab="lookups">Lookups (<span id="nlook">0</span>)</div>
      <div class="tab" data-tab="samples">Samples (<span id="nsamp">0</span>)</div>
    </div>
    <div id="filters">
      <input type="search" id="q" placeholder="search url, id, or text…">
      <select id="part">
        <option value="">all partitions</option>
      </select>
    </div>
    <div id="list"></div>
    <details class="summary">
      <summary>Partition summary</summary>
      <pre id="summary-pre"></pre>
    </details>
  </aside>
  <main id="main"></main>
</div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
let tab = DATA.lookups.length ? 'lookups' : 'samples';
let selected = 0;
let filtered = [];

document.getElementById('nlook').textContent = DATA.lookups.length.toLocaleString();
document.getElementById('nsamp').textContent = DATA.samples.length.toLocaleString();
document.getElementById('summary-pre').textContent = JSON.stringify(DATA.summary, null, 2);

function esc(s) {
  return (s ?? '').toString().replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function partKey(r) { return r._quality + '/' + r._kind + '/' + r._kind2; }

function populatePartFilter() {
  const parts = new Set();
  [...DATA.lookups, ...DATA.samples].forEach(r => parts.add(partKey(r)));
  const sel = document.getElementById('part');
  sel.innerHTML = '<option value="">all partitions</option>' +
    [...parts].sort().map(p => `<option value="${esc(p)}">${esc(p)}</option>`).join('');
}

function currentList() { return tab === 'lookups' ? DATA.lookups : DATA.samples; }

function applyFilters() {
  const q = document.getElementById('q').value.toLowerCase();
  const pk = document.getElementById('part').value;
  filtered = currentList().filter(r => {
    if (pk && partKey(r) !== pk) return false;
    if (!q) return true;
    return (r.url || '').toLowerCase().includes(q)
        || (r.id || '').toLowerCase().includes(q)
        || (r.text || '').toLowerCase().includes(q);
  });
  selected = 0;
}

function renderTabs() {
  document.querySelectorAll('.tab').forEach(el => {
    el.classList.toggle('active', el.dataset.tab === tab);
  });
}

function renderList() {
  const root = document.getElementById('list');
  if (!filtered.length) { root.innerHTML = '<div class="empty">No records match.</div>'; return; }
  root.innerHTML = filtered.map((r, i) => `
    <div class="item ${i === selected ? 'active' : ''}" data-i="${i}">
      <div class="url">${esc(r.url || '(no url)')}</div>
      <div class="meta">
        <span class="chip ${r._kind}">${esc(r._quality)} · ${esc(r._kind)}/${esc(r._kind2)}</span>
        <span class="chip">${esc(r._snapshot)}</span>
        <span>${(r.text || '').length.toLocaleString()} chars</span>
      </div>
    </div>`).join('');
  root.querySelectorAll('.item').forEach(el => {
    el.onclick = () => { selected = +el.dataset.i; renderList(); renderMain(); };
  });
}

function renderMain() {
  const main = document.getElementById('main');
  if (!filtered.length) { main.innerHTML = '<div class="empty">Nothing selected.</div>'; return; }
  if (selected >= filtered.length) selected = 0;
  const r = filtered[selected];
  const q = document.getElementById('q').value;
  const rawText = r.text || '';
  let body;
  if (q) {
    const re = new RegExp('(' + q.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&') + ')', 'gi');
    body = esc(rawText).replace(re, '<mark>$1</mark>');
  } else {
    body = esc(rawText);
  }
  const metaKeys = Object.keys(r.metadata || {});
  main.innerHTML = `
    <div id="mh">
      <div class="url">${esc(r.url || '(no url)')}</div>
      <div class="keys">${selected + 1}/${filtered.length} · j/k or ↑/↓</div>
    </div>
    <div id="meta-row">
      <span><b>id:</b> ${esc(r.id)}</span>
      <span><b>partition:</b> quality=${esc(r._quality)} · kind=${esc(r._kind)} · kind2=${esc(r._kind2)}</span>
      <span><b>snapshot:</b> ${esc(r._snapshot)}</span>
      <span><b>file:</b> ${esc(r._file)}</span>
      <span><b>text length:</b> ${rawText.length.toLocaleString()}</span>
      ${r.source != null ? `<span><b>source:</b> ${esc(r.source)}</span>` : ''}
      ${r.format != null ? `<span><b>format:</b> ${esc(r.format)}</span>` : ''}
      ${metaKeys.map(k => `<span><b>meta.${esc(k)}:</b> ${esc(typeof r.metadata[k] === 'object' ? JSON.stringify(r.metadata[k]) : r.metadata[k])}</span>`).join('')}
    </div>
    <div id="body">${body}</div>`;
}

function refresh() {
  applyFilters();
  renderTabs();
  renderList();
  renderMain();
}

document.querySelectorAll('.tab').forEach(el => {
  el.onclick = () => { tab = el.dataset.tab; refresh(); };
});
document.getElementById('q').addEventListener('input', refresh);
document.getElementById('part').addEventListener('change', refresh);
document.addEventListener('keydown', e => {
  if (['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)) return;
  if (e.key === 'j' || e.key === 'ArrowDown') { selected = Math.min(filtered.length - 1, selected + 1); renderList(); renderMain(); }
  if (e.key === 'k' || e.key === 'ArrowUp')   { selected = Math.max(0, selected - 1); renderList(); renderMain(); }
});

document.getElementById('sub').textContent =
  `${DATA.summary.total_matched} matched · ${DATA.summary.total_samples} samples`
  + (DATA.summary.lookup_ids.length ? ` · id=${DATA.summary.lookup_ids[0].slice(0, 8)}…` : '')
  + (DATA.summary.lookup_urls.length ? ` · 1 url` : '');

populatePartFilter();
refresh();
</script>
</body></html>
"""


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    indir = Path(args.input_dir)
    payload = {
        "lookups": _load_jsonl(indir / "lookups.jsonl"),
        "samples": _load_jsonl(indir / "samples.jsonl"),
        "summary": _load_json(indir / "summary.json"),
    }
    print(f"Loaded {len(payload['lookups'])} lookups + {len(payload['samples'])} samples from {indir}")

    html = HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote viewer to {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
