# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Side-by-side viewer for baseline dataset collection sources.

Samples records from each of the five baseline sources (resiliparse, nemotron,
nemotron_full, dclm, fineweb_edu), joins them on URL, and emits a single
self-contained HTML dashboard that lets you click through URLs and compare the
extracted text across every source that retained that document.

This is a diagnostic tool — designed to answer "is anything systematically
wrong with one of these pipelines?" by eyeballing aligned records.

Usage:
    uv run python experiments/baseline_collection/compare_sources.py \\
        --output scratch/compare_sources.html

    # Larger sample for deeper browsing:
    uv run python experiments/baseline_collection/compare_sources.py \\
        --output scratch/compare_sources.html \\
        --matched 300 --random-per-source 60

Then open the HTML in a browser (no server needed).
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import random
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import orjson

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("compare_sources")

# Canonical GCS roots for the pipeline.py outputs.
SOURCES: dict[str, dict] = {
    "resiliparse": {
        "path": "gs://marin-us-central2/extracted/baseline_resiliparse-19bdaa",
        "shard_glob": "data-*.jsonl.gz",
        "total_shards": 3000,
        "description": "Raw resiliparse main-content extraction over the 3000-WARC set (no filtering).",
        "color": "#7c3aed",
    },
    "nemotron": {
        "path": "gs://marin-us-central2/filtered/baseline_nemotron-037958",
        "shard_glob": "CC-MAIN-*.jsonl.gz",
        "total_shards": None,
        "description": "Nemotron-CC organic records (kind=actual), joined on URL per snapshot.",
        "color": "#16a34a",
    },
    "nemotron_full": {
        "path": "gs://marin-us-central2/filtered/baseline_nemotron_full-347dfe",
        "shard_glob": "CC-MAIN-*.jsonl.gz",
        "total_shards": None,
        "description": "Nemotron-CC organic + all five rephraser-synthetic variants.",
        "color": "#0d9488",
    },
    "dclm": {
        "path": "gs://marin-us-central2/filtered/baseline_dclm_resharded-1ac313",
        "shard_glob": "data-*.jsonl.gz",
        "total_shards": 100,
        "description": "DCLM-baseline records joined on WARC-Record-ID, resharded to 100 files.",
        "color": "#2563eb",
    },
    "fineweb_edu": {
        "path": "gs://marin-us-central2/filtered/baseline_fineweb_edu-72c2c7",
        "shard_glob": "data-*.jsonl.gz",
        "total_shards": 513,
        "description": "FineWeb-Edu records joined on file_path per snapshot.",
        "color": "#dc2626",
    },
}

META_FIELDS: dict[str, list[str]] = {
    "nemotron": ["nemotron_quality", "nemotron_kind", "nemotron_id"],
    "nemotron_full": ["nemotron_quality", "nemotron_kind", "nemotron_kind2", "nemotron_id"],
    "dclm": ["warc_record_id", "dclm_fasttext_score", "dclm_language_score"],
    "fineweb_edu": ["fineweb_score", "fineweb_int_score", "dump", "file_path"],
    "resiliparse": [],
}


@dataclass
class Record:
    source: str
    url: str
    text: str
    meta: dict


def list_shards(source: str) -> list[str]:
    info = SOURCES[source]
    glob = f"{info['path']}/{info['shard_glob']}"
    out = subprocess.run(["gcloud", "storage", "ls", glob], capture_output=True, text=True, check=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def download_shard(gs_path: str, local_dir: Path) -> Path:
    local_path = local_dir / Path(gs_path.replace("gs://", "").replace("/", "__"))
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path
    subprocess.run(["gcloud", "storage", "cp", gs_path, str(local_path)], check=True, capture_output=True)
    return local_path


def read_records(local_path: Path, source: str, max_chars: int) -> list[Record]:
    keep_fields = META_FIELDS.get(source, [])
    records: list[Record] = []
    with gzip.open(local_path, "rb") as f:
        for line in f:
            try:
                rec = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            text = rec.get("text") or ""
            url = rec.get("url") or ""
            if not url or not text:
                continue
            meta = {k: rec.get(k) for k in keep_fields if k in rec}
            records.append(Record(source=source, url=url, text=text[:max_chars], meta=meta))
    return records


def sample_shards(source: str, num_shards: int, seed: int) -> list[str]:
    all_shards = list_shards(source)
    rng = random.Random(seed + hash(source) % 10**6)
    rng.shuffle(all_shards)
    return all_shards[:num_shards]


def load_source(source: str, shards: list[str], local_dir: Path, max_chars: int) -> list[Record]:
    logger.info("[%s] downloading %d shards", source, len(shards))
    local_paths = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(download_shard, s, local_dir): s for s in shards}
        for fut in as_completed(futures):
            local_paths.append(fut.result())
    records: list[Record] = []
    for p in local_paths:
        records.extend(read_records(p, source, max_chars))
    logger.info("[%s] loaded %d records", source, len(records))
    return records


def build_index(all_records: dict[str, list[Record]]) -> dict[str, dict[str, Record]]:
    """source -> url -> Record (first occurrence kept)."""
    index: dict[str, dict[str, Record]] = {}
    for source, recs in all_records.items():
        by_url: dict[str, Record] = {}
        for r in recs:
            by_url.setdefault(r.url, r)
        index[source] = by_url
    return index


def pick_matches(index: dict[str, dict[str, Record]], target: int, seed: int) -> list[tuple[str, int]]:
    """Pick URLs that appear in the most sources, up to `target`, randomized within a tier."""
    url_to_sources: dict[str, set[str]] = defaultdict(set)
    for source, by_url in index.items():
        for url in by_url:
            url_to_sources[url].add(source)

    # Bucket by count of sources, preferring higher counts.
    by_count: dict[int, list[str]] = defaultdict(list)
    for url, sources in url_to_sources.items():
        if len(sources) >= 2:
            by_count[len(sources)].append(url)

    rng = random.Random(seed)
    picked: list[tuple[str, int]] = []
    for count in sorted(by_count.keys(), reverse=True):
        urls = by_count[count]
        rng.shuffle(urls)
        for u in urls:
            picked.append((u, count))
            if len(picked) >= target:
                return picked
    return picked


def build_payload(index: dict[str, dict[str, Record]], matched_target: int, random_per_source: int, seed: int) -> dict:
    rng = random.Random(seed)

    matched_urls = pick_matches(index, matched_target, seed)
    matched = []
    for url, count in matched_urls:
        entry = {
            "url": url,
            "sources": {},
            "source_count": count,
        }
        for source, by_url in index.items():
            if url in by_url:
                r = by_url[url]
                entry["sources"][source] = {"text": r.text, "meta": r.meta}
        matched.append(entry)

    random_samples: dict[str, list[dict]] = {}
    for source, by_url in index.items():
        items = list(by_url.values())
        rng.shuffle(items)
        picks = items[:random_per_source]
        random_samples[source] = [{"url": r.url, "text": r.text, "meta": r.meta} for r in picks]

    source_stats = {}
    for source, by_url in index.items():
        lens = [len(r.text) for r in by_url.values()]
        source_stats[source] = {
            "n_records": len(lens),
            "mean_chars": int(sum(lens) / len(lens)) if lens else 0,
            "median_chars": int(sorted(lens)[len(lens) // 2]) if lens else 0,
            "min_chars": min(lens) if lens else 0,
            "max_chars": max(lens) if lens else 0,
            "description": SOURCES[source]["description"],
            "color": SOURCES[source]["color"],
            "gcs_path": SOURCES[source]["path"],
        }

    return {
        "matched": matched,
        "random_samples": random_samples,
        "source_stats": source_stats,
        "source_order": list(SOURCES.keys()),
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Baseline sources — side-by-side comparison</title>
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
  #app { display: grid; grid-template-columns: 320px 1fr; height: 100vh; }
  #sidebar { border-right: 1px solid var(--border); background: #fafafb; display: flex; flex-direction: column; min-height: 0; }
  #sidebar-header { padding: 12px 14px; border-bottom: 1px solid var(--border); }
  #sidebar-header h1 { margin: 0 0 6px 0; font-size: 15px; }
  #sidebar-header .sub { font-size: 12px; color: var(--muted); }
  .mode-tabs { display: flex; gap: 4px; margin-top: 10px; }
  .mode-tab { flex: 1; padding: 6px 10px; border: 1px solid var(--border); background: white; font-size: 12px; cursor: pointer; border-radius: 6px; }
  .mode-tab.active { background: var(--accent); color: white; border-color: var(--accent); }
  #filters { padding: 8px 14px; border-bottom: 1px solid var(--border); font-size: 12px; display: flex; gap: 6px; flex-wrap: wrap; }
  #filters input[type="search"] { flex: 1; min-width: 120px; padding: 4px 8px; border: 1px solid var(--border); border-radius: 6px; font-size: 12px; }
  #filters select { padding: 4px 6px; border: 1px solid var(--border); border-radius: 6px; font-size: 12px; background: white; }
  #list { flex: 1; overflow-y: auto; }
  .list-item { padding: 8px 14px; border-bottom: 1px solid #eef; cursor: pointer; font-size: 12px; }
  .list-item:hover { background: #eef2ff; }
  .list-item.active { background: #dbeafe; border-left: 3px solid var(--accent); }
  .list-item .url { color: var(--accent); word-break: break-all; font-size: 11px; }
  .list-item .meta { color: var(--muted); font-size: 10px; margin-top: 3px; display: flex; gap: 6px; flex-wrap: wrap; }
  .src-chip { display: inline-block; padding: 1px 6px; border-radius: 9px; font-size: 10px; color: white; }
  #main { overflow: auto; padding: 18px; min-height: 0; }
  #main-header { display: flex; align-items: baseline; gap: 16px; margin-bottom: 10px; flex-wrap: wrap; }
  #main-header .url { font-size: 13px; color: var(--accent); word-break: break-all; flex: 1; }
  #main-header .keys { color: var(--muted); font-size: 11px; }
  .columns { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); }
  .column { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; display: flex; flex-direction: column; max-height: calc(100vh - 110px); }
  .column-head { padding: 10px 12px; border-bottom: 1px solid var(--border); }
  .column-head .src-name { font-weight: 600; font-size: 13px; }
  .column-head .src-meta { font-size: 11px; color: var(--muted); margin-top: 4px; display: flex; gap: 10px; flex-wrap: wrap; }
  .column-head .src-meta b { color: #333; }
  .column-body { padding: 12px; overflow-y: auto; white-space: pre-wrap; font-family: ui-monospace, "SF Mono", Consolas, monospace; font-size: 12px; line-height: 1.55; flex: 1; }
  .missing { color: var(--muted); font-style: italic; padding: 16px; text-align: center; }
  .truncated { color: var(--warn); font-size: 11px; margin-top: 8px; border-top: 1px dashed var(--border); padding-top: 6px; }
  #stats-banner { background: white; border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; margin-bottom: 12px; font-size: 12px; display: flex; gap: 18px; flex-wrap: wrap; }
  #stats-banner .stat-block { border-left: 3px solid transparent; padding-left: 8px; }
  #stats-banner .stat-name { font-weight: 600; }
  #stats-banner .stat-detail { color: var(--muted); }
  .random-scroller { display: flex; flex-direction: column; gap: 10px; }
  .random-item { border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; background: white; }
  .random-item .url { color: var(--accent); font-size: 12px; word-break: break-all; }
  .random-item .meta { color: var(--muted); font-size: 11px; margin-top: 3px; }
  .random-item .text { font-family: ui-monospace, monospace; font-size: 12px; white-space: pre-wrap; margin-top: 8px; max-height: 320px; overflow: auto; background: #fbfbfd; padding: 8px; border-radius: 6px; }
</style>
</head>
<body>
<div id="app">
  <aside id="sidebar">
    <div id="sidebar-header">
      <h1>Baseline source comparison</h1>
      <div class="sub" id="summary-line"></div>
      <div class="mode-tabs">
        <div class="mode-tab active" data-mode="matched">Matched</div>
        <div class="mode-tab" data-mode="random">Random / source</div>
      </div>
    </div>
    <div id="filters">
      <input type="search" id="q" placeholder="filter urls…">
      <select id="min-count">
        <option value="2">≥2 sources</option>
        <option value="3">≥3 sources</option>
        <option value="4">≥4 sources</option>
        <option value="5">all 5 sources</option>
      </select>
    </div>
    <div id="list"></div>
  </aside>
  <main id="main"></main>
</div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);

let mode = 'matched';
let selected = 0;
let filtered = [];
let selectedSource = DATA.source_order[0];

function esc(s) { return (s ?? '').toString().replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

function renderSummary() {
  const stats = DATA.source_stats;
  const parts = DATA.source_order.map(s => {
    const st = stats[s];
    return `<span style="color:${st.color}">●</span>&nbsp;${esc(s)}: ${st.n_records.toLocaleString()} rec`;
  });
  document.getElementById('summary-line').innerHTML = parts.join(' · ');
}

function currentFilteredMatched() {
  const q = document.getElementById('q').value.toLowerCase();
  const minCount = parseInt(document.getElementById('min-count').value, 10);
  return DATA.matched.filter(m =>
    m.source_count >= minCount &&
    (!q || m.url.toLowerCase().includes(q))
  );
}

function renderList() {
  const list = document.getElementById('list');
  if (mode === 'matched') {
    filtered = currentFilteredMatched();
    list.innerHTML = filtered.map((m, i) => `
      <div class="list-item ${i === selected ? 'active' : ''}" data-i="${i}">
        <div class="url">${esc(m.url)}</div>
        <div class="meta">
          <span>${m.source_count}/${DATA.source_order.length} sources</span>
          ${DATA.source_order.filter(s => s in m.sources).map(s =>
            `<span class="src-chip" style="background:${DATA.source_stats[s].color}">${esc(s)}</span>`
          ).join('')}
        </div>
      </div>`).join('');
    list.querySelectorAll('.list-item').forEach(el => {
      el.onclick = () => { selected = parseInt(el.dataset.i, 10); renderList(); renderMain(); };
    });
  } else {
    const q = document.getElementById('q').value.toLowerCase();
    list.innerHTML = DATA.source_order.map(s => {
      const active = s === selectedSource ? 'active' : '';
      const st = DATA.source_stats[s];
      return `<div class="list-item ${active}" data-src="${esc(s)}">
        <div class="url"><span class="src-chip" style="background:${st.color}">${esc(s)}</span> ${st.n_records.toLocaleString()} loaded</div>
        <div class="meta">${esc(st.description)}</div>
      </div>`;
    }).join('');
    list.querySelectorAll('.list-item').forEach(el => {
      el.onclick = () => { selectedSource = el.dataset.src; renderList(); renderMain(); };
    });
  }
}

function renderMain() {
  const main = document.getElementById('main');
  if (mode === 'matched') {
    if (filtered.length === 0) { main.innerHTML = '<div class="missing">No matches. Loosen filters.</div>'; return; }
    if (selected >= filtered.length) selected = 0;
    const m = filtered[selected];
    const columns = DATA.source_order.map(s => {
      const st = DATA.source_stats[s];
      if (!(s in m.sources)) {
        return `<div class="column">
          <div class="column-head">
            <div class="src-name" style="color:${st.color}">${esc(s)}</div>
            <div class="src-meta">not present for this URL</div>
          </div>
          <div class="column-body missing">— not retained by this pipeline —</div>
        </div>`;
      }
      const entry = m.sources[s];
      const metaRow = Object.entries(entry.meta).map(([k, v]) => `<span><b>${esc(k)}:</b> ${esc(typeof v === 'object' ? JSON.stringify(v) : v)}</span>`).join('');
      return `<div class="column">
        <div class="column-head">
          <div class="src-name" style="color:${st.color}">${esc(s)}</div>
          <div class="src-meta"><span><b>${entry.text.length.toLocaleString()}</b> chars</span>${metaRow}</div>
        </div>
        <div class="column-body">${esc(entry.text)}</div>
      </div>`;
    }).join('');
    main.innerHTML = `
      <div id="stats-banner">
        <div class="stat-block"><div class="stat-name">URL ${selected + 1} / ${filtered.length}</div><div class="stat-detail">${m.source_count} sources retained this doc</div></div>
      </div>
      <div id="main-header">
        <div class="url">${esc(m.url)}</div>
        <div class="keys">↑/↓ or j/k to navigate · ←/→ to switch mode</div>
      </div>
      <div class="columns">${columns}</div>`;
  } else {
    const s = selectedSource;
    const st = DATA.source_stats[s];
    const q = document.getElementById('q').value.toLowerCase();
    const items = (DATA.random_samples[s] || []).filter(r => !q || r.url.toLowerCase().includes(q));
    main.innerHTML = `
      <div id="stats-banner">
        <div class="stat-block" style="border-left-color:${st.color}">
          <div class="stat-name" style="color:${st.color}">${esc(s)} — ${items.length} random samples</div>
          <div class="stat-detail">${esc(st.description)} · chars (min/med/mean/max): ${st.min_chars.toLocaleString()} / ${st.median_chars.toLocaleString()} / ${st.mean_chars.toLocaleString()} / ${st.max_chars.toLocaleString()}</div>
          <div class="stat-detail"><code>${esc(st.gcs_path)}</code></div>
        </div>
      </div>
      <div class="random-scroller">
        ${items.map(r => {
          const metaRow = Object.entries(r.meta).map(([k, v]) => `<span><b>${esc(k)}:</b> ${esc(typeof v === 'object' ? JSON.stringify(v) : v)}</span>`).join(' · ');
          return `<div class="random-item">
            <div class="url">${esc(r.url)}</div>
            <div class="meta">${r.text.length.toLocaleString()} chars ${metaRow ? ' · ' + metaRow : ''}</div>
            <div class="text">${esc(r.text)}</div>
          </div>`;
        }).join('')}
      </div>`;
  }
}

function setMode(m) {
  mode = m;
  document.querySelectorAll('.mode-tab').forEach(t => t.classList.toggle('active', t.dataset.mode === m));
  selected = 0;
  renderList();
  renderMain();
}

document.querySelectorAll('.mode-tab').forEach(t => t.onclick = () => setMode(t.dataset.mode));
document.getElementById('q').addEventListener('input', () => { selected = 0; renderList(); renderMain(); });
document.getElementById('min-count').addEventListener('change', () => { selected = 0; renderList(); renderMain(); });

document.addEventListener('keydown', e => {
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) return;
  if (mode === 'matched') {
    if (e.key === 'j' || e.key === 'ArrowDown') { selected = Math.min(filtered.length - 1, selected + 1); renderList(); renderMain(); }
    if (e.key === 'k' || e.key === 'ArrowUp') { selected = Math.max(0, selected - 1); renderList(); renderMain(); }
  }
  if (e.key === 'ArrowRight') setMode('random');
  if (e.key === 'ArrowLeft') setMode('matched');
});

renderSummary();
renderList();
renderMain();
</script>
</body>
</html>
"""


def render_html(payload: dict, output_path: Path) -> None:
    serialized = json.dumps(payload, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("__PAYLOAD__", serialized)
    output_path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("scratch/compare_sources.html"))
    parser.add_argument("--cache-dir", type=Path, default=Path("scratch/compare_sources_cache"))
    parser.add_argument(
        "--shards-resiliparse", type=int, default=2, help="How many resiliparse shards to sample (each ~60MB)."
    )
    parser.add_argument(
        "--shards-nemotron", type=int, default=12, help="How many nemotron shards to sample (each ~1MB)."
    )
    parser.add_argument("--shards-nemotron-full", type=int, default=12)
    parser.add_argument("--shards-dclm", type=int, default=3, help="How many DCLM shards to sample (each ~50MB).")
    parser.add_argument(
        "--shards-fineweb", type=int, default=8, help="How many FineWeb-Edu shards to sample (each ~5MB)."
    )
    parser.add_argument(
        "--max-chars", type=int, default=12000, help="Truncate each record to at most this many characters."
    )
    parser.add_argument("--matched", type=int, default=200, help="Target number of URLs to include in 'matched' mode.")
    parser.add_argument(
        "--random-per-source", type=int, default=40, help="Number of random records per source for 'random' mode."
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    shard_counts = {
        "resiliparse": args.shards_resiliparse,
        "nemotron": args.shards_nemotron,
        "nemotron_full": args.shards_nemotron_full,
        "dclm": args.shards_dclm,
        "fineweb_edu": args.shards_fineweb,
    }

    all_records: dict[str, list[Record]] = {}
    for source, n in shard_counts.items():
        shards = sample_shards(source, n, args.seed)
        all_records[source] = load_source(source, shards, args.cache_dir, args.max_chars)

    index = build_index(all_records)
    payload = build_payload(index, args.matched, args.random_per_source, args.seed)

    logger.info("URL overlap picked: %d matched URLs (target=%d)", len(payload["matched"]), args.matched)
    render_html(payload, args.output)
    logger.info("Wrote %s (%.1f MB)", args.output, args.output.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
