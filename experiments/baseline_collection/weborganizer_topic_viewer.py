# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a self-contained HTML viewer comparing WebOrganizer topic distributions across corpora.

Two questions, one page:
  1. **How do the curation methods differ in topic mix?** — a grouped bar chart, one row per topic,
     one bar per dataset, showing each topic's share of that corpus.
  2. **What does a category actually look like in this corpus?** — click any topic to drill into the
     per-(dataset, topic) random document sample drawn by `LabelReservoir` during the labelling run.

Inputs are exactly what `weborganizer_topic_label.py merge` writes, per dataset:
  * ``distribution.json`` — ``{dataset, n_docs, counts, shares}``
  * ``examples.parquet``  — ``{dataset, label, prob, n_seen, n_eligible, url, text}``

Both are small, so the page embeds them and needs no server::

    python -m experiments.baseline_collection.weborganizer_topic_viewer \\
      --datasets dclm_10k nemotron_full_10k --out topic_review.html
"""

from __future__ import annotations

import argparse
import json
import logging

import fsspec

from experiments.baseline_collection.weborganizer_topic_label import OUT_ROOT
from experiments.baseline_collection.weborganizer_topic_smoke import CORPORA

logger = logging.getLogger(__name__)

TEXT_PREVIEW_CHARS = 4000

# dataviz reference palette, categorical slots 1-5, validated for BOTH modes on the adjacent pairlist
# (light: worst adjacent CVD dE 9.1, normal-vision 19.6; dark: 8.4 / 19.3). Light mode puts three of
# these below 3:1 on the surface, so the relief rule applies -> direct value labels + a table view are
# mandatory here, not optional.
SERIES_LIGHT = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a"]
SERIES_DARK = ["#3987e5", "#008300", "#d55181", "#c98500", "#199e70"]

# The composition view colours by TOPIC. 24 topics cannot each get a hue — cycling a palette past its
# validated slots is how a chart starts lying. But folding the tail into "Other" is just as bad here:
# these topics are fairly even, so a top-8 cut dumps ~48% of every bar into one grey block and hides
# exactly the shift the view exists to show.
#
# So: COMPOSITE ENCODING (hue x texture), the documented escape hatch past the 8-slot ceiling
# ("fold the tail into Other, facet into small multiples, or use composite encoding"). 8 validated
# hues x 3 textures (solid / 45deg / 135deg hatch) = 24 distinct segments, every topic addressable,
# no Other. Segment ORDER is fixed across corpora, so position reinforces identity — a reader tracks
# a topic by slot as much as by fill, and hover names it outright.
TEXTURES = 3
TOPIC_LIGHT = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
TOPIC_DARK = ["#3987e5", "#008300", "#d55181", "#c98500", "#199e70", "#d95926", "#9085e9", "#e66767"]


# Head-to-head diff is a DIVERGING encoding (A over-indexes / B over-indexes / no difference), so it
# takes the documented blue<->red poles with a gray midpoint — never a categorical hue at zero, and
# never two cool hues (the midpoint has to read as "nothing").
DIVERGE_LIGHT = {"pos": "#2a78d6", "neg": "#e34948", "mid": "#f0efec"}
DIVERGE_DARK = {"pos": "#3987e5", "neg": "#e66767", "mid": "#383835"}


def load_dataset(dataset: str, base_dir: str | None = None) -> tuple[dict, list[dict]]:
    """Load one dataset's merged outputs. `base_dir` overrides the GCS location with a local copy.

    The override exists because gcsfs cannot authenticate from some workstations (SSL trust issues)
    even where the `gcloud` CLI works — so `gcloud storage cp` the two small merged files down and
    point this at them.
    """
    import pyarrow.parquet as pq

    corpus = CORPORA[dataset]
    base = base_dir if base_dir else f"gs://marin-{corpus.region}/{OUT_ROOT}/{dataset}"
    with fsspec.open(f"{base}/distribution.json") as fh:
        distribution = json.load(fh)
    with fsspec.open(f"{base}/examples.parquet", "rb") as fh:
        examples = pq.ParquetFile(fh).read().to_pylist()
    for row in examples:
        row["text"] = (row["text"] or "")[:TEXT_PREVIEW_CHARS]
    logger.info("%s: %d docs, %d examples", dataset, distribution["n_docs"], len(examples))
    return distribution, examples


def build_html(distributions: list[dict], examples: list[dict]) -> str:
    payload = json.dumps(
        {
            "distributions": distributions,
            "examples": examples,
            "seriesLight": SERIES_LIGHT,
            "seriesDark": SERIES_DARK,
            "topicLight": TOPIC_LIGHT,
            "topicDark": TOPIC_DARK,
            "textures": TEXTURES,
        }
    )
    return _TEMPLATE.replace("__PAYLOAD__", _script_safe(payload))


def _script_safe(payload: str) -> str:
    """Make a JSON blob safe to inline inside a <script> element.

    The payload carries real web-page text, and some page WILL contain a literal ``</script>``. The
    HTML parser does not care that it sits inside a JS string — it ends the script element there, so
    the page dies and the rest of the JSON renders as visible text. ``<\\/`` is an equivalent escape
    for ``/`` in JSON, so this changes nothing about the parsed data.

    U+2028/U+2029 are line terminators in JS but legal raw inside JSON strings, so they would break
    the literal too.
    """
    return payload.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WebOrganizer topic mix by curation method</title>
<style>
  :root {
    color-scheme: light;
    --surface-0: #f4f4f2; --surface-1: #fcfcfb; --border: #dcdcd6;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #7a7975;
    --grid: #e6e6e1; --pos: #2a78d6; --neg: #e34948; --mid: #f0efec;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --surface-0: #121211; --surface-1: #1a1a19; --border: #34342f;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8e8d84;
      --grid: #2b2b27; --pos: #3987e5; --neg: #e66767; --mid: #383835;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-0: #121211; --surface-1: #1a1a19; --border: #34342f;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8e8d84;
    --grid: #2b2b27; --pos: #3987e5; --neg: #e66767; --mid: #383835;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px; background: var(--surface-0); color: var(--text-primary);
    font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 1240px; margin: 0 auto; }
  h1 { font-size: 20px; margin: 0 0 4px; letter-spacing: -0.01em; }
  .sub { color: var(--text-secondary); margin: 0 0 8px; }
  .panel {
    background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 10px; padding: 18px; margin-bottom: 18px;
  }
  .phead { font-size: 15px; margin: 0 0 4px; }
  .muted { color: var(--text-muted); font-weight: 400; }
  .toolbar {
    display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center;
    padding: 12px 18px; margin-bottom: 18px;
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  }
  .field { display: flex; align-items: center; gap: 7px; color: var(--text-secondary); font-size: 12px; }
  select {
    font: inherit; font-size: 12px; color: var(--text-primary); background: var(--surface-0);
    border: 1px solid var(--border); border-radius: 6px; padding: 3px 6px;
  }
  .legend { display: flex; flex-wrap: wrap; gap: 6px 10px; }
  .legend button {
    display: inline-flex; align-items: center; gap: 6px; cursor: pointer;
    background: none; border: 1px solid transparent; border-radius: 999px;
    padding: 3px 9px; color: var(--text-secondary); font: inherit; font-size: 12px;
  }
  .legend button[aria-pressed="false"] { opacity: .35; }
  .legend button:hover { border-color: var(--border); }
  .swatch { width: 11px; height: 11px; border-radius: 3px; flex: none; }
  .legend-other {
    display: inline-flex; align-items: center; gap: 6px;
    color: var(--text-muted); font-size: 12px; padding: 3px 9px;
  }
  .rows { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; }

  /* composition */
  .stackrow { display: flex; align-items: center; gap: 12px; margin: 8px 0; }
  .stacklab { width: 190px; flex: none; font-size: 12px; text-align: right; line-height: 1.25; }
  .stackwrap { flex: 1; min-width: 320px; }
  .stack { display: flex; gap: 2px; height: 28px; border-radius: 4px; background: var(--surface-0); }
  .seg { position: relative; display: flex; align-items: center; justify-content: center; min-width: 0; }
  .seg:first-child { border-radius: 4px 0 0 4px; }
  .seg:last-child { border-radius: 0 4px 4px 0; }
  .seg-l {
    font-size: 10px; color: #fff; text-shadow: 0 0 3px rgba(0,0,0,.6);
    font-variant-numeric: tabular-nums;
  }
  /* Composite encoding: the hue repeats every 8 topics, the hatch distinguishes the cycle.
     Also the CVD / grayscale-print fallback — identity never rests on hue alone. */
  .tex1 { background-image: repeating-linear-gradient(45deg,
            rgba(255,255,255,.42) 0 3px, rgba(255,255,255,0) 3px 7px); }
  .tex2 { background-image: repeating-linear-gradient(135deg,
            rgba(0,0,0,.40) 0 3px, rgba(0,0,0,0) 3px 7px); }

  /* grouped by-topic */
  .chart td { padding: 3px 0; vertical-align: middle; }
  .topic { white-space: nowrap; padding-right: 12px !important; cursor: pointer; }
  .topic:hover { text-decoration: underline; }
  tr.sel .topic { font-weight: 600; }
  .track { position: relative; min-width: 300px; }
  .bars { display: flex; flex-direction: column; gap: 2px; }
  .bar { height: 9px; border-radius: 0 4px 4px 0; position: relative; }
  .val {
    position: absolute; left: calc(100% + 6px); top: 50%; transform: translateY(-50%);
    font-size: 10px; color: var(--text-muted); font-variant-numeric: tabular-nums; white-space: nowrap;
  }
  .axis {
    color: var(--text-muted); font-size: 11px; border-top: 1px solid var(--grid);
    margin-top: 6px; padding-top: 4px;
  }

  /* diff */
  .diffrow { display: flex; align-items: center; gap: 10px; margin: 3px 0; }
  .difflab { width: 150px; flex: none; font-size: 12px; text-align: right; }
  .diffbar { flex: 1; display: flex; align-items: center; min-width: 260px; height: 15px; position: relative; }
  .diffmid { position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: var(--grid); }
  .dhalf { width: 50%; display: flex; height: 11px; }
  .dhalf.l { justify-content: flex-end; }
  .dfill { height: 100%; }
  .dhalf.l .dfill { border-radius: 4px 0 0 4px; background: var(--neg); }
  .dhalf.r .dfill { border-radius: 0 4px 4px 0; background: var(--pos); }
  .dnum {
    font-size: 10px; color: var(--text-muted); font-variant-numeric: tabular-nums;
    width: 108px; flex: none;
  }

  /* docs */
  .doc { border-top: 1px solid var(--border); padding: 12px 0; }
  .doc-head { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; margin-bottom: 6px; }
  .tag {
    font-size: 11px; padding: 1px 7px; border-radius: 999px;
    border: 1px solid var(--border); color: var(--text-secondary);
  }
  .doc a { color: var(--text-secondary); font-size: 12px; word-break: break-all; }
  .doc pre {
    white-space: pre-wrap; word-break: break-word; margin: 0; max-height: 190px; overflow-y: auto;
    background: var(--surface-0); border: 1px solid var(--border); border-radius: 6px;
    padding: 9px; color: var(--text-secondary); font-size: 12px;
  }
  details { margin-top: 10px; } summary { cursor: pointer; color: var(--text-secondary); font-size: 12px; }
  .tbl th, .tbl td {
    border-bottom: 1px solid var(--border); padding: 5px 9px;
    text-align: right; font-variant-numeric: tabular-nums; font-size: 12px;
  }
  .tbl th:first-child, .tbl td:first-child { text-align: left; }
  .tbl th { color: var(--text-secondary); font-weight: 500; }
  .note { font-size: 12px; color: var(--text-muted); margin: 8px 0 0; }
</style>
</head>
<body>
<div class="wrap">
  <h1>WebOrganizer topic mix by curation method</h1>
  <p class="sub">24-way topic labels over a ~1M-document uniform sample of each curated corpus.</p>

  <div class="toolbar">
    <div class="field">
      <label for="metric">Measure</label>
      <select id="metric">
        <option value="tokens">tokens (what training consumes)</option>
        <option value="docs">documents</option>
      </select>
    </div>
    <div class="field">
      <label for="scale">Scale</label>
      <select id="scale">
        <option value="norm">normalised (each corpus = 100%)</option>
        <option value="abs">to scale (absolute corpus size)</option>
      </select>
    </div>
    <div class="legend" id="legend"></div>
  </div>

  <div class="panel">
    <h2 class="phead">Composition <span class="muted" id="chead"></span></h2>
    <p class="sub" id="csub"></p>
    <div class="legend" id="tlegend"></div>
    <div id="stacks"></div>
    <p class="note" id="cnote"></p>
  </div>

  <div class="panel">
    <h2 class="phead">Head to head <span class="muted">— where two corpora disagree most</span></h2>
    <div class="toolbar" style="margin:6px 0 14px;padding:8px 12px">
      <div class="field"><label for="da">Corpus A</label><select id="da"></select></div>
      <div class="field"><label for="db">vs B</label><select id="db"></select></div>
      <div class="field">
        <label for="dmetric">Compare by</label>
        <select id="dmetric">
          <option value="share">share of corpus (pp)</option>
          <option value="tokens">absolute tokens</option>
        </select>
      </div>
      <div class="field" id="dlegend"></div>
    </div>
    <div id="diff"></div>
  </div>

  <div class="panel">
    <h2 class="phead">By topic <span class="muted">— one bar per corpus</span></h2>
    <div class="toolbar" style="margin:6px 0 14px;padding:8px 12px">
      <div class="field">
        <label for="sort">Sort by</label>
        <select id="sort">
          <option value="mean">share (largest first)</option>
          <option value="spread">disagreement (max minus min)</option>
          <option value="ratio">disagreement (relative, max/min)</option>
        </select>
      </div>
    </div>
    <div class="rows"><table class="chart" id="chart"></table></div>
    <div class="axis" id="axis"></div>
    <details>
      <summary>Table view (exact values)</summary>
      <div class="rows"><table class="tbl" id="tbl"></table></div>
    </details>
  </div>

  <div class="panel">
    <div class="toolbar" style="margin:0 0 14px;padding:8px 12px">
      <div class="field"><label for="docds">Show documents from</label><select id="docds"></select></div>
      <div class="field"><label for="doctopic">Topic</label><select id="doctopic"></select></div>
    </div>
    <div id="docs"></div>
  </div>
</div>
<script>
const DATA = __PAYLOAD__;
const dists = DATA.distributions;
const names = dists.map(d => d.dataset);
const byName = Object.fromEntries(dists.map(d => [d.dataset, d]));
const allTopics = [...new Set(dists.flatMap(d => Object.keys(d.shares)))];

const isDark = () => (document.documentElement.dataset.theme || "")
  ? document.documentElement.dataset.theme === "dark"
  : matchMedia("(prefers-color-scheme: dark)").matches;
const corpusColor = i => (isDark() ? DATA.seriesDark : DATA.seriesLight)[i % DATA.seriesLight.length];
// hue cycles every 8; texture marks which cycle -> 8 x 3 = 24 addressable topics.
const topicColor = i => (isDark() ? DATA.topicDark : DATA.topicLight)[i % 8];
const texClass = i => ["", "tex1", "tex2"][Math.floor(i / 8) % DATA.textures];

const on = new Set(names);
let metric = "tokens", scaleMode = "norm", sortMode = "mean";
let selected = null;

// --- measures -------------------------------------------------------------
// share(): fraction of the corpus. Token share is the default because it is what a training run
// actually consumes — doc share weights a one-line page the same as an 8k-token manual.
const hasTokens = dists.every(d => d.token_shares && Object.keys(d.token_shares).length);
function share(d, t) {
  const src = (metric === "tokens" && hasTokens) ? d.token_shares : d.shares;
  return src[t] || 0;
}
// mass(): absolute tokens of topic t in corpus d, by scaling the SAMPLE's share up to the corpus's
// real token count. This is what makes "is dclm's code as big as all of fineweb_edu?" answerable.
function mass(d, t) { return share(d, t) * (d.corpus_total_tokens || 0); }
const total = d => (scaleMode === "abs" ? (d.corpus_total_tokens || 0) : 1);
const value = (d, t) => (scaleMode === "abs" ? mass(d, t) : share(d, t));

const live = () => dists.filter(d => on.has(d.dataset));
const fmtInt = n => (+n).toLocaleString();
const fmtTok = n => n >= 1e9 ? (n/1e9).toFixed(1) + "B" : n >= 1e6 ? (n/1e6).toFixed(0) + "M" : fmtInt(Math.round(n));
const fmtVal = v => scaleMode === "abs" ? fmtTok(v) : (100*v).toFixed(2) + "%";
const esc = s => (s || "").replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

// --- topic ordering -------------------------------------------------------
const meanOf = t => live().reduce((s, d) => s + share(d, t), 0) / (live().length || 1);
const spreadOf = t => { const v = live().map(d => share(d, t)); return Math.max(...v) - Math.min(...v); };
// Relative gap catches a curation difference that absolute spread buries: Adult at 0.07% vs 2% is a
// 28x filtering effect but only 1.9pp. Floored so an all-zero topic can't divide to Infinity.
const ratioOf = t => { const v = live().map(d => share(d, t)); return Math.max(...v) / Math.max(Math.min(...v), 1e-5); };
const sorters = { mean: meanOf, spread: spreadOf, ratio: ratioOf };
let topics = [...allTopics];
function resort() { topics = [...allTopics].sort((a, b) => sorters[sortMode](b) - sorters[sortMode](a)); }

// Composition segments use a FIXED order (overall mean) so a topic sits in the same slot in every
// row — that is what lets a shift read as a width change instead of a reshuffle. ALL 24 are drawn:
// no "Other", because with a near-even taxonomy that bucket swallows half the bar.
const overallMean = t => dists.reduce((s, d) => s + share(d, t), 0) / dists.length;
let topTopics = [];
function retop() { topTopics = [...allTopics].sort((a, b) => overallMean(b) - overallMean(a)); }

// --- render ---------------------------------------------------------------
function drawLegend() {
  document.getElementById("legend").innerHTML = names.map((n, i) => {
    const d = byName[n];
    const sz = d.corpus_total_tokens ? ` <span class="muted">${fmtTok(d.corpus_total_tokens)} tok</span>` : "";
    return `<button data-n="${n}" aria-pressed="${on.has(n)}">
              <span class="swatch" style="background:${corpusColor(i)}"></span>${n}${sz}</button>`;
  }).join("");
  document.querySelectorAll("#legend button").forEach(b => b.onclick = () => {
    const n = b.dataset.n;
    on.has(n) ? on.delete(n) : on.add(n);
    if (!on.size) on.add(n);
    resort(); render();   // disagreement is defined over VISIBLE corpora, so toggling reorders
  });
}

const small0 = xs => xs.slice().sort((a,b) => (a.corpus_total_tokens||0)-(b.corpus_total_tokens||0))[0];

function drawStacks() {
  const abs = scaleMode === "abs";
  document.getElementById("chead").textContent =
    `— one bar per corpus, coloured by topic${abs ? ", drawn to scale" : ""}`;
  document.getElementById("csub").textContent = abs
    ? "Bar length is the corpus's real token count, so a segment's width is that topic's absolute "
      + "mass. Hide the giants to read the rest."
    : "Each bar is 100% of its corpus, so a distribution shift shows up as segments changing width between rows.";
  // Legend carries the same hue x texture as the segment, and is the only place the 24 names live.
  document.getElementById("tlegend").innerHTML =
    topTopics.map((t, i) => `<button data-t="${t}" aria-pressed="true">
        <span class="swatch ${texClass(i)}" style="background:${topicColor(i)}"></span>${t}</button>`).join("");
  document.querySelectorAll("#tlegend button").forEach(b => b.onclick = () => { selected = b.dataset.t; render(); });

  const shown = live();
  const widest = Math.max(...shown.map(d => total(d)), 1);
  document.getElementById("stacks").innerHTML = shown.map(d => {
    const rowFrac = abs ? (d.corpus_total_tokens || 0) / widest : 1;  // bar length relative to biggest
    const segs = topTopics.map((t, i) => {
      const s = share(d, t);
      const label = s >= 0.06 && rowFrac > 0.15 ? `<span class="seg-l">${(s*100).toFixed(0)}%</span>` : "";
      const detail = abs ? `${fmtTok(mass(d, t))} tok (${(s*100).toFixed(1)}%)` : `${(s*100).toFixed(2)}%`;
      return `<div class="seg ${texClass(i)}" style="width:${(s*100).toFixed(3)}%;background:${topicColor(i)}"
                   title="${d.dataset} — ${t}: ${detail}">${label}</div>`;
    }).join("");
    const sz = d.corpus_total_tokens ? fmtTok(d.corpus_total_tokens) + " tok" : "";
    return `<div class="stackrow">
              <div class="stacklab">${d.dataset}<br><span class="muted">${sz}</span></div>
              <div class="stackwrap">
                <div class="stack" style="width:${(rowFrac*100).toFixed(3)}%">${segs}</div>
              </div>
            </div>`;
  }).join("");
  const big = shown.slice().sort((a,b) => (b.corpus_total_tokens||0)-(a.corpus_total_tokens||0))[0];
  const ratioBigSmall = big && small0(shown)
    ? ((big.corpus_total_tokens||0)/(small0(shown).corpus_total_tokens||1)).toFixed(0) : "";
  const small = small0(shown);
  document.getElementById("cnote").textContent = abs && big && small && big !== small
    ? `${big.dataset} is ${ratioBigSmall}x ${small.dataset} — untick it to see the others at a readable size.`
    : "";
}

// Head-to-head: a DIVERGING encoding (A over-indexes / B over-indexes / no difference), so it uses
// the blue<->red poles with a neutral midpoint — never a categorical hue at zero.
function drawDiff() {
  const a = byName[document.getElementById("da").value];
  const b = byName[document.getElementById("db").value];
  // Two genuinely different questions. "share" asks how the MIX differs (pp) — a corpus can be tiny
  // and still over-index on a topic. "tokens" asks how much MATERIAL each actually contributes, by
  // scaling each side's share up to its real corpus size, so a 145x size gap dominates as it should.
  const dm = document.getElementById("dmetric").value;
  const val = (d, t) => dm === "tokens" ? mass(d, t) : share(d, t);
  const fmtD = v => dm === "tokens"
    ? (v >= 0 ? "+" : "-") + fmtTok(Math.abs(v)) + " tok"
    : (v >= 0 ? "+" : "") + (v * 100).toFixed(2) + "pp";
  document.getElementById("dlegend").innerHTML =
    `<span class="legend-other"><span class="swatch" style="background:var(--pos)"></span>${a.dataset} higher</span>
     <span class="legend-other"><span class="swatch" style="background:var(--neg)"></span>${b.dataset} higher</span>`;
  if (a === b) {
    document.getElementById("diff").innerHTML = '<p class="muted">Pick two different corpora.</p>';
    return;
  }
  if (dm === "tokens" && !(a.corpus_total_tokens && b.corpus_total_tokens)) {
    document.getElementById("diff").innerHTML =
      '<p class="muted">No corpus token total recorded for one of these — absolute comparison unavailable.</p>';
    return;
  }
  const ratio2 = r => r.bv > 0 ? (r.av / r.bv) : Infinity;
  const rows = allTopics.map(t => ({ t, d: val(a, t) - val(b, t), av: val(a, t), bv: val(b, t) }))
                        .sort((x, y) => Math.abs(y.d) - Math.abs(x.d));
  const max = Math.max(...rows.map(r => Math.abs(r.d)), 1e-9);
  document.getElementById("diff").innerHTML = rows.map(r => {
    const w = (Math.abs(r.d) / max) * 100;
    const ratioTxt = isFinite(ratio2(r)) ? (ratio2(r) >= 1 ? ratio2(r).toFixed(1) : (1/ratio2(r)).toFixed(1)) + "x" : "";
    const fmtSide = v => dm === "tokens" ? fmtTok(v) + " tok" : (v*100).toFixed(2) + "%";
    const tip = `${r.t}: ${a.dataset} ${fmtSide(r.av)} vs ${b.dataset} ${fmtSide(r.bv)}`;
    return `<div class="diffrow" title="${tip}">
      <div class="difflab">${r.t}</div>
      <div class="diffbar"><div class="diffmid"></div>
        <div class="dhalf l">${r.d < 0 ? `<div class="dfill" style="width:${w}%"></div>` : ""}</div>
        <div class="dhalf r">${r.d > 0 ? `<div class="dfill" style="width:${w}%"></div>` : ""}</div>
      </div>
      <div class="dnum">${fmtD(r.d)}
        <span class="muted">${ratioTxt}</span></div>
    </div>`;
  }).join("");
}

const axisUnit = () => scaleMode === "abs" ? "tokens" : (metric === "tokens" ? "token share" : "doc share");

function drawChart() {
  const shown = live();
  const maxV = Math.max(...shown.flatMap(d => allTopics.map(t => value(d, t))), 1e-9);
  document.getElementById("chart").innerHTML = topics.map(t => {
    const bars = names.map((n, i) => {
      if (!on.has(n)) return "";
      const v = value(byName[n], t);
      return `<div class="bar" style="width:${((v/maxV)*100).toFixed(2)}%;background:${corpusColor(i)}"
                   title="${n} — ${t}: ${fmtVal(v)}">
                <span class="val">${fmtVal(v)}</span></div>`;
    }).join("");
    return `<tr class="${t === selected ? "sel" : ""}" data-t="${t}">
              <td class="topic">${t}</td><td><div class="track"><div class="bars">${bars}</div></div></td>
            </tr>`;
  }).join("");
  document.querySelectorAll("#chart tr").forEach(tr => tr.onclick = () => { selected = tr.dataset.t; render(); });
  document.getElementById("axis").textContent =
    `0 ————— ${axisUnit()} ————— ${fmtVal(maxV)}`;
  const head = `<tr><th>Topic</th>${names.map(n => `<th>${n}</th>`).join("")}</tr>`;
  document.getElementById("tbl").innerHTML = head + topics.map(t =>
    `<tr><td>${t}</td>${dists.map(d => `<td>${fmtVal(value(d, t))}</td>`).join("")}</tr>`).join("");
}

function drawDocs() {
  // Explicit selectors beat "whatever is toggled above": reading documents is a different task from
  // comparing distributions, and it wants one corpus at a time.
  const dsSel = document.getElementById("docds").value;
  const rows = DATA.examples.filter(e => e.label === selected && (dsSel === "__all" || e.dataset === dsSel));
  const shownNames = dsSel === "__all" ? names.filter(n => on.has(n)) : [dsSel];
  const body = shownNames.map(n => {
    const mine = rows.filter(r => r.dataset === n);
    if (!mine.length) return `<div class="doc"><span class="tag">${n}</span>
      <span class="muted"> no sampled document cleared the probability floor for this topic</span></div>`;
    return mine.map(r => `<div class="doc">
        <div class="doc-head">
          <span class="tag" style="border-color:${corpusColor(names.indexOf(n))}">${n}</span>
          <span class="muted">p=${(+r.prob).toFixed(3)} · ${fmtInt(r.n_seen)} docs of this topic in the sample</span>
        </div>
        <div><a href="${r.url}" target="_blank" rel="noopener">${r.url || "(no url)"}</a></div>
        <pre>${esc(r.text)}</pre></div>`).join("");
  }).join("");
  document.getElementById("docs").innerHTML =
    `<h2 class="phead">${selected} <span class="muted">— random sample of documents</span></h2>${body}`;
}

function syncDocControls() {
  const ds = document.getElementById("docds");
  if (!ds.options.length) {
    ds.add(new Option("all visible corpora", "__all"));
    names.forEach(n => ds.add(new Option(n, n)));
  }
  // Rebuild the topic list against the CHOSEN corpus: the label carries that corpus's own count and
  // share, and flags topics with no sampled document, so picking "nemotron / Software Dev." is one
  // read rather than a guess followed by an empty panel. Alphabetical — you are looking for a known
  // register here, not browsing the ranking (the charts above already rank).
  const tp = document.getElementById("doctopic");
  const dsSel = ds.value || "__all";
  const key = `${dsSel}|${metric}`;
  if (tp.dataset.key !== key) {
    tp.dataset.key = key;
    const cohort = dsSel === "__all" ? live() : [byName[dsSel]];
    const have = new Set(DATA.examples
      .filter(e => dsSel === "__all" || e.dataset === dsSel).map(e => e.label));
    tp.innerHTML = "";
    [...allTopics].sort((a, b) => a.localeCompare(b)).forEach(t => {
      const docs = cohort.reduce((n, d) => n + (d.counts[t] || 0), 0);
      const sh = cohort.reduce((n, d) => n + share(d, t), 0) / (cohort.length || 1);
      const tag = have.has(t) ? "" : "  (no sample)";
      tp.add(new Option(`${t} — ${fmtInt(docs)} docs · ${(sh*100).toFixed(1)}%${tag}`, t));
    });
  }
  tp.value = selected;   // clicking a topic in either chart drives this select too
}

function render() {
  drawLegend(); retop(); drawStacks(); drawDiff(); drawChart(); syncDocControls(); drawDocs();
}

// --- wiring ---------------------------------------------------------------
const da = document.getElementById("da"), db = document.getElementById("db");
names.forEach(n => { da.add(new Option(n, n)); db.add(new Option(n, n)); });
da.value = names[0]; db.value = names[names.length > 1 ? 1 : 0];
da.onchange = db.onchange = drawDiff;
document.getElementById("dmetric").onchange = drawDiff;
document.getElementById("docds").onchange = () => { syncDocControls(); drawDocs(); };
document.getElementById("doctopic").onchange = e => { selected = e.target.value; render(); };

document.getElementById("metric").onchange = e => { metric = e.target.value; resort(); render(); };
document.getElementById("scale").onchange = e => {
  scaleMode = e.target.value;
  // Absolute scale is only defined in tokens: we know each corpus's true TOKEN count, not its true
  // document count, so "to scale" forces the token measure rather than inventing a doc total.
  const m = document.getElementById("metric");
  m.disabled = scaleMode === "abs";
  if (scaleMode === "abs") { metric = "tokens"; m.value = "tokens"; }
  resort(); render();
};
document.getElementById("sort").onchange = e => {
  sortMode = e.target.value; resort();
  if (!topics.includes(selected)) selected = topics[0];
  render();
};
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);

if (!hasTokens) {
  metric = "docs";
  const m = document.getElementById("metric");
  m.value = "docs"; m.disabled = true;
  document.getElementById("scale").disabled = true;
}
resort(); selected = topics[0]; render();
</script>
</body>
</html>
"""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", required=True, choices=list(CORPORA))
    parser.add_argument("--out", default="topic_review.html")
    parser.add_argument(
        "--base-dir",
        help="Read {dataset}/distribution.json + examples.parquet from here instead of GCS "
        "(for workstations where gcsfs cannot authenticate).",
    )
    args = parser.parse_args()

    distributions, examples = [], []
    for dataset in args.datasets:
        base = f"{args.base_dir.rstrip('/')}/{dataset}" if args.base_dir else None
        distribution, rows = load_dataset(dataset, base)
        distributions.append(distribution)
        examples.extend(rows)

    with open(args.out, "w") as fh:
        fh.write(build_html(distributions, examples))
    logger.info("wrote %s (%d datasets, %d examples)", args.out, len(distributions), len(examples))


if __name__ == "__main__":
    main()
