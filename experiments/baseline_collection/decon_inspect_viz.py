#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501  -- embedded HTML/CSS/JS template has intentionally long lines

"""Render decon_inspect's summary JSON into a self-contained inspection page.

Reads a local ``inspect_summary.json`` (produced by ``decon_inspect.py``, pull
it with ``gcloud storage cp``) and writes a single static HTML with the data
embedded — open it directly, no server. Shows the flag rate, the per-eval-task
breakdown, and each flagged corpus doc beside the specific eval item it matched
(matched n-gram highlighted) so you can judge reasonable-vs-too-strict.

Usage::

    gcloud storage cp \\
      gs://marin-us-central1/documents/baseline_high_quality_decon/10364warcs_core_v2/inspect/inspect_summary.json \\
      scratch/
    uv run python experiments/baseline_collection/decon_inspect_viz.py \\
        --summary scratch/inspect_summary.json --out scratch/decon_inspect.html
    open scratch/decon_inspect.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Decontamination inspection — {title}</title>
<style>
  :root {{ --bg:#0f1115; --card:#1a1d24; --fg:#e6e6e6; --muted:#9aa4b2; --mark:#5b8cff; --hit:#ffd54a; }}
  body {{ background:var(--bg); color:var(--fg); font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; margin:0; padding:24px; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  .sub {{ color:var(--muted); margin-bottom:20px; }}
  .big {{ font-size:42px; font-weight:700; color:var(--mark); }}
  .stats {{ display:flex; gap:32px; align-items:baseline; flex-wrap:wrap; margin-bottom:24px; }}
  .stat small {{ display:block; color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.05em; }}
  .section {{ margin:28px 0 12px; font-size:13px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); }}
  .bar {{ display:flex; align-items:center; gap:10px; margin:3px 0; }}
  .bar .name {{ width:300px; color:var(--fg); }}
  .bar .track {{ flex:1; background:#252a33; border-radius:4px; height:16px; position:relative; }}
  .bar .fill {{ background:var(--mark); height:100%; border-radius:4px; }}
  .bar .n {{ width:70px; text-align:right; color:var(--muted); }}
  .controls {{ margin:12px 0; }}
  select, input {{ background:var(--card); color:var(--fg); border:1px solid #2c313b; border-radius:6px; padding:6px 8px; }}
  .card {{ background:var(--card); border:1px solid #262b35; border-radius:10px; padding:14px; margin:12px 0; display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .card h3 {{ margin:0 0 6px; font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }}
  .txt {{ white-space:pre-wrap; word-break:break-word; background:#0c0e12; padding:10px; border-radius:6px; max-height:420px; overflow:auto; }}
  mark {{ background:var(--hit); color:#000; padding:0 2px; border-radius:2px; }}
  .badge {{ display:inline-block; background:#2a3550; color:#bcd0ff; border-radius:999px; padding:1px 9px; font-size:12px; margin-bottom:6px; }}
  .multi {{ color:var(--muted); font-size:12px; margin-top:6px; }}
  .note {{ color:var(--muted); font-size:12px; margin-bottom:4px; }}
  .big.green {{ color:#5bd98c; }}
  .pill {{ display:inline-block; border-radius:999px; padding:1px 8px; font-size:11px; font-weight:600; margin-right:8px; }}
  .pill.gen {{ background:#16351f; color:#7fe0a0; }}
  .pill.ubi {{ background:#3a2f22; color:#e0b97f; }}
  input[type=range] {{ width:280px; vertical-align:middle; }}
  .dfnote {{ color:var(--muted); font-size:12px; }}
</style></head><body>
<h1>Decontamination inspection — {title}</h1>
<div class="sub">Exact {ngram}-gram match against DCLM CORE v2 eval items, with GPT-3-style frequency filtering: an n-gram appearing in &gt; T corpus docs is treated as <b>ubiquitous public text</b> (Bill of Rights, scripture, boilerplate), not leakage. A doc is <b>genuine</b> contamination only if its most-distinctive overlap appears in ≤ T docs.</div>
<div class="stats">
  <div class="stat"><span class="big green" id="genpct"></span><small>genuine contamination (DF≤<span id="tlabel"></span>)</small></div>
  <div class="stat"><span class="big green" id="genn"></span><small>genuine docs</small></div>
  <div class="stat"><span class="big" id="flagged"></span><small>total flagged (any DF)</small></div>
  <div class="stat"><span class="big" id="total"></span><small>total docs</small></div>
</div>
<div class="controls">
  <label>DF threshold T = <b id="tval"></b>&nbsp; <input type="range" id="tslider" min="1" max="1000"></label>
  <span class="dfnote">drag the “ubiquitous” cutoff (GPT-3 used 10) — watch genuine % and the badges update</span>
</div>
<div class="section">Genuine contamination by eval task <span style="text-transform:none">(at the loaded DF≤{df})</span></div>
<div id="bars"></div>
<div class="section">Flagged examples <span style="text-transform:none">(most-distinctive first; corpus doc ↔ matched eval item)</span></div>
<div class="controls">
  <label>task <select id="taskf"><option value="">all</option></select></label>
  <input id="search" placeholder="filter text…" size="22">
  <label><input type="checkbox" id="hideubi"> hide ubiquitous</label>
  <span id="shown" style="color:var(--muted)"></span>
</div>
<div id="samples"></div>
<script>
const DATA = {data};
const esc = s => (s||"").replace(/[&<>]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c]));
function hl(text, ng) {{
  const e = esc(text); if (!ng) return e;
  const i = e.indexOf(esc(ng)); if (i < 0) return e;
  const m = esc(ng);
  return e.slice(0,i) + "<mark>" + m + "</mark>" + e.slice(i+m.length);
}}
function fmt(n) {{ return (n==null?0:n).toLocaleString(); }}
const HIST = Object.entries(DATA.min_df_histogram||{{}}).map(([k,v])=>[+k,v]).sort((a,b)=>a[0]-b[0]);
function genuineAt(T) {{ let s=0; for (const [d,c] of HIST) if (d<=T) s+=c; return s; }}
document.getElementById('flagged').textContent = fmt(DATA.flagged_docs);
document.getElementById('total').textContent = fmt(DATA.total_docs);
// per-task genuine bars at the server-loaded threshold
const ptg = Object.entries(DATA.per_task_genuine||{{}});
const ptmax = ptg.length ? Math.max(...ptg.map(x=>x[1])) : 1;
document.getElementById('bars').innerHTML = ptg.map(([t,n]) =>
  `<div class="bar"><div class="name">${{esc(t)}}</div><div class="track"><div class="fill" style="width:${{100*n/ptmax}}%"></div></div><div class="n">${{fmt(n)}}</div></div>`).join("") || '<div class="dfnote">none genuine at this threshold</div>';
const tf = document.getElementById('taskf');
[...new Set(DATA.samples.map(s=>s.eval_task))].sort().forEach(t => {{
  const o=document.createElement('option'); o.value=t; o.textContent=t; tf.appendChild(o);
}});
const slider = document.getElementById('tslider');
slider.value = DATA.df_threshold || 10;
function render() {{
  const T = +slider.value, t = tf.value, q = document.getElementById('search').value.toLowerCase(), hide = document.getElementById('hideubi').checked;
  document.getElementById('tval').textContent = T;
  document.getElementById('tlabel').textContent = T;
  const g = genuineAt(T);
  document.getElementById('genn').textContent = fmt(g);
  document.getElementById('genpct').textContent = (100*g/DATA.total_docs).toFixed(4) + "%";
  let rows = DATA.samples.slice().sort((a,b)=>(a.doc_min_df||0)-(b.doc_min_df||0));
  rows = rows.filter(s => (!t || s.eval_task===t) &&
    (!q || (s.corpus_text+s.eval_text).toLowerCase().includes(q)) &&
    (!hide || (s.doc_min_df||0)<=T));
  document.getElementById('shown').textContent = `${{rows.length}} of ${{DATA.samples.length}} examples`;
  document.getElementById('samples').innerHTML = rows.map(s => {{
    const gen = (s.doc_min_df||0) <= T;
    return `<div class="card">
      <div><h3>Corpus document</h3>
        <span class="pill ${{gen?'gen':'ubi'}}">${{gen?'GENUINE':'UBIQUITOUS'}}</span><span class="dfnote">most-distinctive overlap in ${{fmt(s.doc_min_df)}} corpus doc(s)</span>
        <div class="note">${{s.corpus_len!=null ? "doc length "+fmt(s.corpus_len)+" chars · match at char "+fmt(s.doc_match_offset) : ""}}</div>
        <div class="txt corpus">${{hl(s.corpus_text, s.matched_ngram)}}</div></div>
      <div><h3>Matched eval item</h3><span class="badge">${{esc(s.eval_task)}}</span>
        <div style="color:var(--muted);font-size:12px;margin-bottom:4px">${{esc(s.eval_item_id)}}</div>
        <div class="txt">${{hl(s.eval_text, s.matched_ngram)}}</div>
        <div class="multi">shown n-gram appears in <b>${{fmt(s.matched_ngram_df)}}</b> corpus docs; matched ${{s.num_eval_items_matched}} eval item(s)<br>n-gram: <mark>${{esc(s.matched_ngram)}}</mark></div>
      </div>
    </div>`;
  }}).join("");
  document.querySelectorAll('#samples .corpus').forEach(box => {{
    const m = box.querySelector('mark');
    if (m) box.scrollTop = Math.max(0, m.offsetTop - box.offsetTop - 60);
  }});
}}
tf.onchange = render; document.getElementById('search').oninput = render;
document.getElementById('hideubi').onchange = render; slider.oninput = render;
render();
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", required=True, help="Local path to inspect_summary.json.")
    parser.add_argument("--out", required=True, help="Output HTML path.")
    parser.add_argument("--title", default="high_quality 10k", help="Page title / dataset label.")
    parser.add_argument("--ngram", type=int, default=13)
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    html = _HTML.format(
        title=args.title,
        ngram=summary.get("ngram_length", args.ngram),
        df=summary.get("df_threshold", 10),
        data=json.dumps(summary),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(
        f"Wrote {out} — flagged {summary['flag_pct']:.4f}% ({summary['flagged_docs']}/{summary['total_docs']}); "
        f"genuine DF<={summary.get('df_threshold')}: {summary.get('genuine_pct', 0):.4f}% "
        f"({summary.get('genuine_docs')}). Open it:"
    )
    print(f"  open {out}")


if __name__ == "__main__":
    main()
