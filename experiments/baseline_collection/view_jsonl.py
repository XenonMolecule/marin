# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render a jsonl(.gz) file as a self-contained, interactive HTML page.

Use when spot-checking extraction outputs (or any jsonl) in a browser
instead of staring at gzipped JSON. Reads from local paths or gs:// URIs.

    uv run python experiments/baseline_collection/view_jsonl.py \\
        gs://marin-eu-west4/documents/baseline_llm_extraction/med_quality/\\
        data-06e2dfdcfff5/batch_0000.jsonl.gz \\
        --max-records 200 --output scratch/view.html

The resulting HTML has a record list on the left and a detail pane on the
right showing the selected record's text in a scrollable, monospaced view.
URL + warc_record_id + snapshot are surfaced in the header. No server.
"""

from __future__ import annotations

import argparse
import gzip
import html
import io
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _open_jsonl(path: str):
    """Yield (line_idx, decoded_dict) from a local or gs:// jsonl(.gz)."""
    if path.startswith("gs://"):
        from google.cloud import storage as gcs_storage

        bucket_name, _, blob_path = path[len("gs://") :].partition("/")
        client = gcs_storage.Client()
        blob = client.bucket(bucket_name).blob(blob_path)
        raw = blob.download_as_bytes()
        stream = io.BytesIO(raw)
    else:
        stream = open(path, "rb")

    if path.endswith(".gz"):
        f = gzip.open(stream, "rt", encoding="utf-8")
    else:
        f = io.TextIOWrapper(stream, encoding="utf-8")
    try:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except Exception as e:
                logger.warning("Skipping malformed JSON at line %d: %s", i, e)
    finally:
        f.close()


def _record_summary(rec: dict) -> str:
    """Human-readable one-line label for the record list."""
    text = rec.get("text") or rec.get("generated_text") or ""
    text = text.replace("\n", " ").strip()
    head = text[:80]
    url = rec.get("url") or rec.get("warc_file") or ""
    return f"{head}  ⟵  {url[-60:]}" if url else head


def _safe_json_for_script(obj) -> str:
    """JSON-encode for embedding in a ``<script>`` block.

    Records often contain literal ``</`` (e.g. ``</p>`` from leftover HTML
    in the model output). Without escaping, the browser treats the first
    ``</script>`` as the end of the script tag and dumps the remainder as
    plaintext. ``ensure_ascii=False`` keeps the file small but we still
    have to defang the closing-script sequence and HTML comment markers.
    """
    raw = json.dumps(obj, ensure_ascii=False)
    return raw.replace("</", "<\\/").replace("<!--", "<\\!--").replace("-->", "--\\>")


def render_html(records: list[dict], source: str, title: str | None) -> str:
    title = title or source.rsplit("/", 1)[-1]
    # Build the records sidebar + detail pane. All data lives in a JS
    # variable for client-side filtering/highlighting.
    payload = _safe_json_for_script(records)
    summaries = [_record_summary(r) for r in records]
    summary_payload = _safe_json_for_script(summaries)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
  body {{ margin: 0; font-family: -apple-system, system-ui, sans-serif; color: #e0e0e0; background: #1a1a2e; }}
  header {{ padding: 12px 16px; border-bottom: 1px solid #333; background: #16213e; }}
  header h1 {{ font-size: 14px; margin: 0 0 4px; color: #4ecca3; font-weight: normal; }}
  header .source {{ font-size: 11px; color: #888; word-break: break-all; }}
  header .stats {{ font-size: 12px; color: #888; margin-top: 4px; }}
  .layout {{ display: flex; height: calc(100vh - 64px); }}
  .sidebar {{ width: 360px; overflow-y: auto; border-right: 1px solid #333; background: #1a1a2e; }}
  .sidebar input[type=search] {{
    width: calc(100% - 16px); margin: 8px; padding: 6px 8px;
    background: #0f3460; border: 1px solid #333; color: #e0e0e0;
    border-radius: 4px; font-family: inherit; font-size: 12px;
  }}
  .item {{ padding: 8px 12px; border-bottom: 1px solid #2a2a3e; cursor: pointer; font-size: 12px; }}
  .item:hover {{ background: #16213e; }}
  .item.selected {{ background: #0f3460; color: #4ecca3; }}
  .item .idx {{ color: #888; font-size: 10px; margin-right: 6px; }}
  .detail {{ flex: 1; overflow-y: auto; padding: 16px; font-family: 'SF Mono', 'Fira Code', Consolas, monospace; }}
  .detail h2 {{ font-size: 13px; color: #4ecca3; margin: 16px 0 4px; font-weight: normal; }}
  .detail .meta {{ font-size: 12px; color: #888; padding: 8px 0; border-bottom: 1px solid #333; margin-bottom: 12px; }}
  .detail .meta a {{ color: #3498db; }}
  .detail pre {{
    white-space: pre-wrap; word-wrap: break-word; font-size: 13px; line-height: 1.5;
    background: #0f1424; padding: 12px; border-radius: 4px; border: 1px solid #2a2a3e;
    margin: 0;
  }}
  .detail .other {{ font-size: 11px; color: #888; }}
  .empty {{ padding: 32px; color: #888; text-align: center; }}
</style></head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <div class="source">{html.escape(source)}</div>
  <div class="stats">{len(records)} records · use search box to filter</div>
</header>
<div class="layout">
  <div class="sidebar">
    <input type="search" id="filter" placeholder="filter by text, URL, hash...">
    <div id="items"></div>
  </div>
  <div class="detail" id="detail"><div class="empty">Select a record on the left.</div></div>
</div>
<script>
const RECORDS = {payload};
const SUMMARIES = {summary_payload};
let filtered = RECORDS.map((_, i) => i);

const itemsEl = document.getElementById('items');
const detailEl = document.getElementById('detail');
const filterEl = document.getElementById('filter');

function escapeHtml(s) {{
  return (s || '').replace(/[&<>"']/g, c => ({{ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }}[c]));
}}

function renderList() {{
  itemsEl.innerHTML = filtered.map(i => `<div class="item" data-idx="${{i}}"><span class="idx">#${{i}}</span>${{escapeHtml(SUMMARIES[i])}}</div>`).join('');
  for (const el of itemsEl.querySelectorAll('.item')) {{
    el.onclick = () => selectRecord(+el.dataset.idx);
  }}
}}

function selectRecord(idx) {{
  for (const el of itemsEl.querySelectorAll('.item')) {{
    el.classList.toggle('selected', +el.dataset.idx === idx);
  }}
  const r = RECORDS[idx];
  const text = r.text || r.generated_text || '';
  const url = r.url || '';
  const known = ['text', 'generated_text', 'url', 'warc_record_id', 'warc_file', 'snapshot'];
  const others = Object.entries(r).filter(([k]) => !known.includes(k));
  detailEl.innerHTML = `
    <div class="meta">
      Record #${{idx}} · ${{text.length.toLocaleString()}} chars
      ${{url ? `· <a href="${{escapeHtml(url)}}" target="_blank">${{escapeHtml(url)}}</a>` : ''}}
      ${{r.snapshot ? `· snapshot: ${{escapeHtml(r.snapshot)}}` : ''}}
      ${{r.warc_record_id ? `· wrid: ${{escapeHtml(r.warc_record_id)}}` : ''}}
    </div>
    <h2>extracted text</h2>
    <pre>${{escapeHtml(text)}}</pre>
    ${{others.length ? `<h2>other fields</h2><pre class="other">${{escapeHtml(JSON.stringify(Object.fromEntries(others), null, 2))}}</pre>` : ''}}
  `;
}}

filterEl.oninput = () => {{
  const q = filterEl.value.toLowerCase().trim();
  if (!q) {{
    filtered = RECORDS.map((_, i) => i);
  }} else {{
    filtered = [];
    for (let i = 0; i < RECORDS.length; i++) {{
      const blob = (RECORDS[i].text || '') + ' ' + (RECORDS[i].url || '') + ' ' + (RECORDS[i].warc_record_id || '');
      if (blob.toLowerCase().includes(q)) filtered.push(i);
    }}
  }}
  renderList();
}};

renderList();
if (RECORDS.length) selectRecord(0);
</script>
</body></html>"""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="Local path or gs:// URI to a jsonl(.gz) file")
    p.add_argument("--max-records", type=int, default=500, help="Cap records (HTML grows linearly).")
    p.add_argument("--output", default=None, help="HTML output path. Defaults to scratch/view_<basename>.html")
    p.add_argument("--title", default=None, help="HTML page title (defaults to filename).")
    p.add_argument("--open", dest="open_browser", action="store_true", help="Open the result in the default browser.")
    args = p.parse_args()

    records: list[dict] = []
    for _, rec in _open_jsonl(args.source):
        records.append(rec)
        if len(records) >= args.max_records:
            break

    out_path = (
        args.output or f"scratch/view_{Path(args.source.rstrip('/').split('/')[-1]).stem.replace('.jsonl', '')}.html"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(render_html(records, source=args.source, title=args.title))
    logger.info("Rendered %d records → %s", len(records), out_path)

    if args.open_browser:
        import webbrowser

        webbrowser.open(f"file://{Path(out_path).resolve()}")


if __name__ == "__main__":
    main()
