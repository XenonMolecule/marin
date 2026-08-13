# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the cross-corpus comparison as a standalone, self-contained page.

Takes ``compare.json`` from :mod:`experiments.baseline_collection.grid_compare`
and emits a three-tab tool. The tabs answer three different questions and are
deliberately not merged into one view:

* **Quality** — how the five quality buckets divide each corpus, shown as both
  composition (100% stacked, so mixes are comparable across corpora of wildly
  different size) and absolute mass (shared scale, so size is visible at all).
  fineweb_edu and fineweb_cc differ by more than 12x in tokens; either view alone
  is misleading, so both are on screen together.
* **Topics** — the 24-way breakdown, with three encodings because they answer
  different questions: *share* is the corpus's own composition, *absolute* is how
  much material exists, and *index* is over/under-representation against the
  pooled corpora, which is the one that actually tells you what a corpus is
  unusually rich in.
* **Grid** — the full 24 x 5 for one corpus at a time.

The data is embedded rather than fetched: the artifact CSP blocks external
requests, and the whole dataset is a few hundred kilobytes.

    python -m experiments.baseline_collection.grid_compare_page \\
        --compare compare/compare.json --out compare/index.html
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib

logger = logging.getLogger(__name__)

# Categorical slots 1-6, validated as a set on the adjacent pairlist in both
# modes. Three of the light steps sit under 3:1 on the surface, which obliges
# the relief rule — every chart here carries direct labels and an exact table.
# Indexed positionally against DATA.corpora, so this must be at least as long as
# that list or a series renders `undefined`. Entries are APPEND-ONLY: inserting a
# colour re-colours every corpus after it and breaks comparability with plots
# already published. The 7th (brown) is resiliparse — chosen because the existing
# six cover blue/orange/green/amber/pink/dark-green and brown is the clearest
# remaining separation in both themes.
CORPUS_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#8c564b"]
CORPUS_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#a9756a"]
# Ordinal ramp for the five quality buckets: one hue, monotone lightness, and the
# step nearest the surface clears 2:1 in each mode (light starts at step 250,
# dark stops at step 600). Low quality sits nearest the page ground in both.
QUALITY_LIGHT = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
QUALITY_DARK = ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"]
SEQ_LIGHT = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]  # fmt: skip
SEQ_DARK = list(reversed(SEQ_LIGHT))
# Diverging blue<->red with a neutral gray midpoint, for the index view. A hue at
# the midpoint would read as a value rather than as "no difference".
DIV_LIGHT = ["#e34948", "#ec7a72", "#f3a89f", "#f0efec", "#9ec5f4", "#5598e7", "#2a78d6"]
DIV_DARK = ["#e66767", "#d95d5c", "#b35150", "#383835", "#256abf", "#3987e5", "#6da7ec"]


def build_page(data: dict) -> str:
    payload = json.dumps(data, separators=(",", ":"))
    tokens = {
        "corpusLight": CORPUS_LIGHT,
        "corpusDark": CORPUS_DARK,
        "qualityLight": QUALITY_LIGHT,
        "qualityDark": QUALITY_DARK,
        "seqLight": SEQ_LIGHT,
        "seqDark": SEQ_DARK,
        "divLight": DIV_LIGHT,
        "divDark": DIV_DARK,
    }
    return (
        "<title>Quality x topic across the 10k corpora</title>\n"
        + _STYLE
        + "<div class='root'>"
        + _BODY
        + "</div>\n<script>const DATA="
        + payload
        + ";const PAL="
        + json.dumps(tokens, separators=(",", ":"))
        + ";\n"
        + _SCRIPT
        + "</script>\n"
    )


_STYLE = """<style>
  :root {
    color-scheme: light;
    --surface-1:#fcfcfb; --surface-2:#f4f3f0; --surface-3:#eceae5; --border:#dedcd6;
    --ink:#0b0b0b; --ink-2:#52514e; --ink-3:#75746f; --accent:#2a78d6;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --surface-1:#1a1a19; --surface-2:#232322; --surface-3:#2c2c2a; --border:#3a3a37;
      --ink:#ffffff; --ink-2:#c3c2b7; --ink-3:#9b9a92; --accent:#3987e5;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-1:#1a1a19; --surface-2:#232322; --surface-3:#2c2c2a; --border:#3a3a37;
    --ink:#ffffff; --ink-2:#c3c2b7; --ink-3:#9b9a92; --accent:#3987e5;
  }
  * { box-sizing: border-box; }
  body { margin:0; }
  .root {
    background:var(--surface-1); color:var(--ink); min-height:100vh;
    font:16px/1.6 ui-serif, Charter, "Iowan Old Style", Georgia, serif;
    padding:36px 26px 90px; max-width:1280px; margin:0 auto;
  }
  h1 { font-size:29px; line-height:1.2; letter-spacing:-0.015em; margin:0 0 6px; font-weight:600;
       text-wrap:balance; }
  .lede { color:var(--ink-2); max-width:70ch; margin:0 0 26px; font-size:15px; }
  .mono, td, th, .num { font-family:ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
                        font-variant-numeric:tabular-nums; }
  h2 { font-family:ui-monospace, Menlo, monospace; font-size:12px; text-transform:uppercase;
       letter-spacing:0.1em; color:var(--ink-3); margin:34px 0 12px; font-weight:600; }
  h2 .hint { text-transform:none; letter-spacing:0; font-family:inherit; color:var(--ink-3);
             font-weight:400; margin-left:10px; }
  /* controls */
  .bar { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:0 0 8px; }
  .seg { display:inline-flex; background:var(--surface-2); border:1px solid var(--border);
         border-radius:8px; padding:2px; gap:2px; }
  .seg button { font:inherit; font-size:13px; font-family:ui-monospace, Menlo, monospace;
                background:none; border:0; color:var(--ink-2); padding:5px 11px; border-radius:6px;
                cursor:pointer; }
  .seg button[aria-pressed="true"] { background:var(--surface-1); color:var(--ink);
                                     box-shadow:0 1px 2px rgba(0,0,0,.12); font-weight:600; }
  .seg button:focus-visible { outline:2px solid var(--accent); outline-offset:1px; }
  .tabs { display:flex; gap:4px; border-bottom:1px solid var(--border); margin:0 0 20px; }
  .tabs button { font:inherit; font-size:14px; background:none; border:0; border-bottom:2px solid transparent;
                 color:var(--ink-3); padding:9px 15px; cursor:pointer; margin-bottom:-1px; }
  .tabs button[aria-selected="true"] { color:var(--ink); border-bottom-color:var(--accent); font-weight:600; }
  .tabs button:focus-visible { outline:2px solid var(--accent); outline-offset:-2px; }
  .lbl { font-size:12px; color:var(--ink-3); font-family:ui-monospace, Menlo, monospace;
         text-transform:uppercase; letter-spacing:0.07em; }
  /* stacked bars */
  .rows { display:flex; flex-direction:column; gap:9px; }
  .row { display:grid; grid-template-columns:150px 1fr 108px; align-items:center; gap:12px; }
  .row .name { font-size:14px; text-align:right; color:var(--ink); overflow:hidden;
               text-overflow:ellipsis; white-space:nowrap; }
  .row .name em { font-style:normal; color:var(--ink-3); font-size:11px; display:block; line-height:1.3; }
  .track { display:flex; gap:2px; height:30px; }
  .seg-fill { height:100%; display:flex; align-items:center; justify-content:center; overflow:hidden;
              font-family:ui-monospace, Menlo, monospace; font-size:11px; font-weight:600;
              min-width:0; border-radius:2px; }
  .seg-fill:first-child { border-radius:5px 2px 2px 5px; }
  .seg-fill:last-child { border-radius:2px 5px 5px 2px; }
  .row .tot { font-size:13px; font-family:ui-monospace, Menlo, monospace; color:var(--ink);
              font-variant-numeric:tabular-nums; }
  .row .tot em { font-style:normal; color:var(--ink-3); font-size:11px; display:block; }
  /* legend */
  .legend { display:flex; flex-wrap:wrap; align-items:center; gap:14px; margin:14px 0 0;
            color:var(--ink-3); font-size:12px; }
  .key { display:inline-flex; align-items:center; gap:6px; }
  .key i { width:13px; height:13px; border-radius:3px; display:inline-block; }
  /* tables */
  .scroll { overflow-x:auto; margin-top:6px; }
  table { border-collapse:separate; border-spacing:2px; width:100%; }
  th, td { font-size:12.5px; text-align:right; padding:6px 8px; white-space:nowrap; }
  thead th { color:var(--ink-2); font-size:11px; text-transform:uppercase; letter-spacing:0.06em;
             font-weight:600; }
  th.rowh { text-align:left; font-weight:500; color:var(--ink); font-family:ui-serif, Georgia, serif;
            font-size:13.5px; }
  td.cell { border-radius:4px; }
  td.tot, th.tot { font-weight:700; }
  tr.foot td, tr.foot th { border-top:2px solid var(--border); }
  .sortable { cursor:pointer; user-select:none; }
  .sortable:hover { color:var(--ink); text-decoration:underline; }
  .note { background:var(--surface-2); border:1px solid var(--border); border-radius:10px;
          padding:16px 20px; color:var(--ink-2); font-size:14.5px; margin-top:26px; }
  .note li { margin:7px 0; }
  .note strong, .note b { color:var(--ink); }
  code { font-family:ui-monospace, Menlo, monospace; font-size:.88em; background:var(--surface-3);
         padding:1px 5px; border-radius:4px; }
  .hidden { display:none; }
  .cards { display:flex; flex-wrap:wrap; gap:10px; margin-bottom:6px; }
  .card { background:var(--surface-2); border:1px solid var(--border); border-radius:10px;
          padding:12px 16px; flex:1 1 165px; }
  .card .v { font-size:23px; font-weight:650; font-family:ui-monospace, Menlo, monospace;
             font-variant-numeric:tabular-nums; letter-spacing:-0.02em; }
  .card .k { color:var(--ink-2); font-size:12.5px; margin-top:2px; }
  .card .n { color:var(--ink-3); font-size:11px; margin-top:4px; }
  @media (prefers-reduced-motion: no-preference) {
    .seg-fill, td.cell { transition: background-color .15s ease; }
  }
</style>
"""

_BODY = """
  <h1>Quality &times; topic across the 10k-WARC corpora</h1>
  <p class="lede">Exact 24 &times; 5 grids for the five corpora already scored, plus the projected grid for
  the in-flight <code>llm_pipeline_v1_1</code> run. Same WebOrganizer topic classifier, same calibrated
  quality scorer, same bucket edges, so the cells are directly comparable.</p>

  <div class="tabs" role="tablist">
    <button role="tab" id="tab-quality" aria-selected="true" onclick="showTab('quality')">Quality buckets</button>
    <button role="tab" id="tab-topics" aria-selected="false" onclick="showTab('topics')">Topics</button>
    <button role="tab" id="tab-grid" aria-selected="false" onclick="showTab('grid')">Full 24&times;5 grid</button>
  </div>

  <div class="bar">
    <span class="lbl">Measure</span>
    <span class="seg" id="metric-seg">
      <button aria-pressed="true" onclick="setMetric('tokens')">Tokens</button>
      <button aria-pressed="false" onclick="setMetric('docs')">Documents</button>
    </span>
    <span class="lbl" id="metric-note"></span>
  </div>

  <section id="pane-quality">
    <div class="cards" id="q-cards"></div>
    <h2>Composition <span class="hint">every corpus normalised to 100% &mdash; compare the mix</span></h2>
    <div class="rows" id="q-share"></div>
    <div class="legend" id="q-legend"></div>
    <h2>Absolute mass <span class="hint">shared scale &mdash; compare the size</span></h2>
    <div class="rows" id="q-abs"></div>
    <h2>Exact numbers</h2>
    <div class="scroll" id="q-table"></div>
  </section>

  <section id="pane-topics" class="hidden">
    <div class="bar">
      <span class="lbl">Encoding</span>
      <span class="seg" id="topic-mode-seg">
        <button aria-pressed="true" onclick="setTopicMode('share')">Share within corpus</button>
        <button aria-pressed="false" onclick="setTopicMode('abs')">Absolute</button>
        <button aria-pressed="false" onclick="setTopicMode('index')">Index vs. pooled</button>
      </span>
    </div>
    <h2 id="topics-h2"></h2>
    <div class="scroll" id="t-matrix"></div>
    <div class="legend" id="t-legend"></div>
  </section>

  <section id="pane-grid" class="hidden">
    <div class="bar">
      <span class="lbl">Corpus A</span>
      <span class="seg" id="grid-corpus-seg"></span>
    </div>
    <div class="bar">
      <span class="lbl">Compare with</span>
      <span class="seg" id="grid-corpus-b-seg"></span>
    </div>
    <div class="bar" id="grid-mode-bar">
      <span class="lbl">View</span>
      <span class="seg" id="grid-mode-seg">
        <button aria-pressed="true" onclick="setGridMode('side')">Side by side</button>
        <button aria-pressed="false" onclick="setGridMode('ratio')">A &divide; B</button>
      </span>
    </div>
    <h2 id="grid-h2"></h2>
    <div class="scroll" id="g-matrix"></div>
    <div class="legend" id="g-legend"></div>
  </section>

  <div class="note">
    <ul>
      <li><strong>Tokens here are gte tokens, not llama3 tokens.</strong> grid_v1 recorded token mass with the
        topic model's own tokenizer, capped at its 8192-token context. That undercounts long documents, so it
        is not a training-token count &mdash; but it is the only unit measured identically for every corpus.
        For <code>llm_pipeline_v1_1</code> the true llama3 projection is <b id="llama-note"></b>, about 13%
        above its gte figure; expect a similar gap for the others.</li>
      <li><strong>The corpora are at different pipeline stages.</strong> <code>high_quality</code> is
        post-dedup and post-decontamination; the rest are pre-dedup; <code>llm_pipeline_v1_1</code> is a
        projection from a partial run and is pre-dedup. Raw sizes are therefore not like-for-like &mdash;
        the composition views are the fairer comparison.</li>
      <li><strong>q2 dominates everywhere.</strong> The calibrated scorer puts the bulk of general web text in
        the middle bucket for every corpus, so the interesting differences live in the q3/q4 tail and in the
        q0 junk share, not in q2.</li>
    </ul>
  </div>
"""

_SCRIPT = """
let METRIC = 'tokens';
let TOPIC_MODE = 'share';
let TOPIC_SORT = -1;   // -1 = order by pooled share rather than by any one corpus
// The grid tab opens on high_quality vs dclm because that is the comparison the
// mixing decision actually turns on. Resolved by label, not by position, so
// adding a corpus to the registry cannot silently repoint these.
const corpusIndex = (label, fallback) => {
  const i = DATA.corpora.findIndex(c => c.label === label);
  return i < 0 ? fallback : i;
};
let GRID_CORPUS = corpusIndex('high_quality', 0);
let GRID_CORPUS_B = corpusIndex('dclm', -1);
let GRID_MODE = 'side';   // 'side' | 'ratio'

const dark = () => {
  const t = document.documentElement.getAttribute('data-theme');
  if (t === 'dark') return true;
  if (t === 'light') return false;
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
};
const corpusPal  = () => dark() ? PAL.corpusDark  : PAL.corpusLight;
const qualityPal = () => dark() ? PAL.qualityDark : PAL.qualityLight;
const seqPal     = () => dark() ? PAL.seqDark     : PAL.seqLight;
const divPal     = () => dark() ? PAL.divDark     : PAL.divLight;
// Ink that stays legible on a given ramp step. The ordinal and sequential ramps
// both run light->dark in light mode and the reverse in dark mode, so "past the
// middle" is where the fill gets dark enough to need white text.
const inkOn = (i, n) => {
  const deep = dark() ? i < n / 2 : i >= n / 2;
  return deep ? '#ffffff' : '#0b0b0b';
};

const fmt = (v) => {
  if (METRIC === 'docs') {
    if (v >= 1e6) return (v / 1e6).toFixed(1) + 'M';
    if (v >= 1e3) return (v / 1e3).toFixed(0) + 'K';
    return v.toFixed(0);
  }
  if (v >= 1e9) return (v / 1e9).toFixed(1) + 'B';
  if (v >= 1e6) return (v / 1e6).toFixed(0) + 'M';
  if (v >= 1e3) return (v / 1e3).toFixed(0) + 'K';
  return v.toFixed(0);
};
const pct = (v) => (v * 100).toFixed(1) + '%';
const grid = (c) => METRIC === 'tokens' ? c.tokens : c.docs;
const rowSum = (a) => a.reduce((s, v) => s + v, 0);
const bucketTotals = (c) => {
  const g = grid(c), out = new Array(DATA.buckets.length).fill(0);
  for (const row of g) row.forEach((v, b) => out[b] += v);
  return out;
};
const topicTotals = (c) => grid(c).map(rowSum);
const total = (c) => rowSum(bucketTotals(c));

function showTab(name) {
  for (const t of ['quality', 'topics', 'grid']) {
    document.getElementById('pane-' + t).classList.toggle('hidden', t !== name);
    document.getElementById('tab-' + t).setAttribute('aria-selected', String(t === name));
  }
}
function press(segId, idx) {
  const btns = document.getElementById(segId).querySelectorAll('button');
  btns.forEach((b, i) => b.setAttribute('aria-pressed', String(i === idx)));
}
function setMetric(m) { METRIC = m; press('metric-seg', m === 'tokens' ? 0 : 1); renderAll(); }
function setTopicMode(m) {
  TOPIC_MODE = m;
  press('topic-mode-seg', {share: 0, abs: 1, index: 2}[m]);
  renderTopics();
}

/* ---------- quality tab ---------- */
function renderCards() {
  const el = document.getElementById('q-cards');
  const proj = DATA.corpora[0];
  const pooled = DATA.corpora.filter(c => !c.projected);
  const pooledTot = pooled.reduce((s, c) => s + total(c), 0);
  const q4 = DATA.corpora.map(c => ({c, v: bucketTotals(c)[4] / total(c)}))
                         .sort((a, b) => b.v - a.v)[0];
  const q0 = DATA.corpora.map(c => ({c, v: bucketTotals(c)[0] / total(c)}))
                         .sort((a, b) => b.v - a.v)[0];
  el.innerHTML = `
    <div class="card"><div class="v">${fmt(total(proj))}</div>
      <div class="k">llm_pipeline_v1_1, projected</div>
      <div class="n">vs ${fmt(pooledTot)} across the five scored corpora</div></div>
    <div class="card"><div class="v">${pct(q4.v)}</div>
      <div class="k">best q4 share &mdash; ${q4.c.label}</div>
      <div class="n">the top bucket is scarce everywhere</div></div>
    <div class="card"><div class="v">${pct(q0.v)}</div>
      <div class="k">most junk &mdash; ${q0.c.label}</div>
      <div class="n">q0 share of ${METRIC}</div></div>
    <div class="card"><div class="v">${DATA.corpora.length}</div>
      <div class="k">corpora compared</div>
      <div class="n">24 topics &times; 5 buckets each</div></div>`;
}

function stackedRows(elId, normalise) {
  const pal = qualityPal();
  const maxTot = Math.max(...DATA.corpora.map(total));
  document.getElementById(elId).innerHTML = DATA.corpora.map(c => {
    const bt = bucketTotals(c), tot = rowSum(bt);
    const width = normalise ? 100 : (tot / maxTot) * 100;
    const segs = bt.map((v, b) => {
      const frac = v / tot;
      const label = frac >= 0.06 ? (normalise ? pct(frac) : fmt(v)) : '';
      return `<span class="seg-fill" style="flex:${Math.max(v, 1e-9)} 0 0;background:${pal[b]};color:${inkOn(b, 5)}"
        title="${c.label} · ${DATA.buckets[b]}&#10;${fmt(v)} (${pct(frac)})">${label}</span>`;
    }).join('');
    return `<div class="row">
      <div class="name">${c.label}<em>${c.stage}</em></div>
      <div class="track" style="width:${width}%">${segs}</div>
      <div class="tot">${fmt(tot)}<em>${normalise ? '100%' : pct(tot / maxTot) + ' of max'}</em></div>
    </div>`;
  }).join('');
}

function renderQualityLegend() {
  const pal = qualityPal();
  document.getElementById('q-legend').innerHTML =
    DATA.buckets.map((b, i) => `<span class="key"><i style="background:${pal[i]}"></i>${b}</span>`).join('')
    + '<span>&mdash; ordinal ramp, light&rarr;dark with quality.</span>';
}

function renderQualityTable() {
  const head = `<tr><th class="rowh">Corpus</th>${DATA.buckets.map(b => `<th>${b}</th>`).join('')}
    <th class="tot">Total</th></tr>`;
  const body = DATA.corpora.map(c => {
    const bt = bucketTotals(c), tot = rowSum(bt);
    return `<tr><th class="rowh">${c.label}</th>${bt.map(v =>
      `<td>${fmt(v)}<br><span style="color:var(--ink-3);font-size:11px">${pct(v / tot)}</span></td>`).join('')}
      <td class="tot">${fmt(tot)}</td></tr>`;
  }).join('');
  document.getElementById('q-table').innerHTML = `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

/* ---------- topics tab ---------- */
function topicMatrix() {
  // columns = corpora, rows = topics; value depends on the chosen encoding.
  const cols = DATA.corpora.map(c => topicTotals(c));
  const totals = DATA.corpora.map((c, j) => rowSum(cols[j]));
  const pooled = DATA.topics.map((_, i) =>
    DATA.corpora.reduce((s, c, j) => c.projected ? s : s + cols[j][i], 0));
  const pooledTot = rowSum(pooled);
  return DATA.topics.map((t, i) => ({
    topic: t,
    values: DATA.corpora.map((c, j) => {
      const abs = cols[j][i], share = abs / totals[j];
      if (TOPIC_MODE === 'abs') return abs;
      if (TOPIC_MODE === 'share') return share;
      const base = pooled[i] / pooledTot;
      return base > 0 ? share / base : 1;   // index: 1.0 == same as pooled
    }),
    pooledShare: pooled[i] / pooledTot,
  }));
}

function renderTopics() {
  const rows = topicMatrix();
  const order = [...rows.keys()].sort((a, b) => TOPIC_SORT < 0
    ? rows[b].pooledShare - rows[a].pooledShare
    : rows[b].values[TOPIC_SORT] - rows[a].values[TOPIC_SORT]);

  const flat = rows.flatMap(r => r.values);
  const maxV = Math.max(...flat);
  const seq = seqPal(), div = divPal();
  const colour = (v) => {
    if (TOPIC_MODE === 'index') {
      // log2 ratio, clamped at +/-2 doublings, onto the 7-step diverging ramp.
      const l = Math.max(-2, Math.min(2, Math.log2(v || 1e-9)));
      const i = Math.round((l + 2) / 4 * (div.length - 1));
      return {bg: div[i], fg: (i <= 1 || i >= div.length - 2) ? '#ffffff' : 'var(--ink)'};
    }
    // sqrt so the 40x spread between the biggest and smallest topic does not
    // collapse every small cell onto the same near-surface step.
    const i = Math.round(Math.sqrt(Math.max(v, 0) / maxV) * (seq.length - 1));
    return {bg: seq[i], fg: inkOn(i, seq.length)};
  };
  const show = (v) => TOPIC_MODE === 'abs' ? fmt(v)
                    : TOPIC_MODE === 'share' ? pct(v)
                    : (v >= 1 ? '\\u00d7' + v.toFixed(2) : '\\u00d7' + v.toFixed(2));

  document.getElementById('topics-h2').innerHTML = ({
    share: 'Topic share within each corpus <span class="hint">columns each sum to 100%</span>',
    abs:   'Topic mass in absolute ' + METRIC + ' <span class="hint">how much material exists</span>',
    index: 'Over/under-representation vs. the pooled scored corpora <span class="hint">' +
           '&times;1.00 = same as pooled; blue = richer, red = poorer</span>',
  })[TOPIC_MODE];

  const head = `<tr><th class="rowh sortable" onclick="sortTopics(-1)" title="Sort by pooled share">Topic \\u21c5</th>` +
    DATA.corpora.map((c, j) =>
      `<th class="sortable" onclick="sortTopics(${j})" title="Sort by ${c.label}">${c.label} \\u21c5</th>`).join('') +
    `</tr>`;
  const body = order.map(i => {
    const r = rows[i];
    return `<tr><th class="rowh">${r.topic}</th>` + r.values.map((v, j) => {
      const {bg, fg} = colour(v);
      return `<td class="cell" style="background:${bg};color:${fg}"
        title="${DATA.corpora[j].label} · ${r.topic}&#10;${show(v)}">${show(v)}</td>`;
    }).join('') + `</tr>`;
  }).join('');
  document.getElementById('t-matrix').innerHTML =
    `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;

  document.getElementById('t-legend').innerHTML = TOPIC_MODE === 'index'
    ? div.map((c, i) => `<span class="key"><i style="background:${c}"></i>${['\\u00bc\\u00d7','','\\u00bd\\u00d7','1\\u00d7','2\\u00d7','','4\\u00d7'][i]}</span>`).join('')
      + '<span>&mdash; log scale, clamped at 4&times; either way. Click a column header to sort.</span>'
    : '<span>Low</span>' + seq.map(c => `<span class="key"><i style="background:${c}"></i></span>`).join('')
      + '<span>High &mdash; square-root scale; every cell also carries its number. Click a column header to sort.</span>';
}
function sortTopics(j) { TOPIC_SORT = j; renderTopics(); }

/* ---------- grid tab ---------- */
// A shaded cell for one value, normalised against its OWN corpus's largest cell.
// Per-corpus normalisation is deliberate in the side-by-side: the corpora differ
// several-fold in size, so a shared ramp would render the smaller one uniformly
// pale and hide its internal structure. The numbers in the cells carry the
// absolute comparison; the ratio view carries the direct one.
function gridCell(v, maxV, topic, bucket, label, tot) {
  const seq = seqPal();
  const idx = v <= 0 ? 0 : Math.round(Math.sqrt(v / maxV) * (seq.length - 1));
  return `<td class="cell" style="background:${v > 0 ? seq[idx] : 'var(--surface-2)'};
    color:${v > 0 ? inkOn(idx, seq.length) : 'var(--ink-3)'}"
    title="${label} · ${topic} · ${bucket}&#10;${fmt(v)} (${pct(v / tot)})">${fmt(v)}</td>`;
}

function renderGridTab() {
  document.getElementById('grid-corpus-seg').innerHTML = DATA.corpora.map((c, i) =>
    `<button aria-pressed="${i === GRID_CORPUS}" onclick="setGridCorpus(${i})">${c.label}</button>`).join('');
  document.getElementById('grid-corpus-b-seg').innerHTML =
    `<button aria-pressed="${GRID_CORPUS_B < 0}" onclick="setGridCorpusB(-1)">none</button>` +
    DATA.corpora.map((c, i) =>
      `<button aria-pressed="${i === GRID_CORPUS_B}" onclick="setGridCorpusB(${i})">${c.label}</button>`).join('');
  document.getElementById('grid-mode-bar').style.display = GRID_CORPUS_B < 0 ? 'none' : '';

  const a = DATA.corpora[GRID_CORPUS], ga = grid(a), ta = total(a), maxA = Math.max(...ga.flat());
  const b = GRID_CORPUS_B >= 0 ? DATA.corpora[GRID_CORPUS_B] : null;
  const gb = b ? grid(b) : null, tb = b ? total(b) : 0, maxB = b ? Math.max(...gb.flat()) : 1;
  const order = [...ga.keys()].sort((x, y) => rowSum(ga[y]) - rowSum(ga[x]));
  const h2 = document.getElementById('grid-h2');
  const legend = document.getElementById('g-legend');
  const seq = seqPal();

  if (b && GRID_MODE === 'ratio') {
    // Share-of-corpus ratio, not raw ratio: the corpora differ several-fold in
    // total size, so a raw ratio would just restate that everywhere. This asks
    // "is this cell a bigger slice of A than of B?", which is the mixing question.
    const div = divPal();
    h2.innerHTML = `${a.label} &divide; ${b.label} <span class="hint">ratio of each cell's share of its own
      corpus &mdash; blue = richer in ${a.label}, red = richer in ${b.label}</span>`;
    const head = `<tr><th class="rowh">Topic</th>${DATA.buckets.map(x => `<th>${x}</th>`).join('')}
      <th class="tot">Row</th></tr>`;
    const body = order.map(i => {
      const cells = ga[i].map((v, k) => {
        const sa = v / ta, sb = gb[i][k] / tb;
        if (sa === 0 && sb === 0) return `<td class="cell" style="background:var(--surface-2);color:var(--ink-3)">&mdash;</td>`;
        const r = sb > 0 ? sa / sb : Infinity;
        const l = Math.max(-2, Math.min(2, Math.log2(r || 1e-9)));
        const idx = Math.round((l + 2) / 4 * (div.length - 1));
        const txt = !isFinite(r) ? '&infin;' : '\\u00d7' + r.toFixed(2);
        return `<td class="cell" style="background:${div[idx]};
          color:${(idx <= 1 || idx >= div.length - 2) ? '#ffffff' : 'var(--ink)'}"
          title="${DATA.topics[i]} · ${DATA.buckets[k]}&#10;${a.label}: ${fmt(v)} (${pct(sa)})&#10;${b.label}: ${fmt(gb[i][k])} (${pct(sb)})">${txt}</td>`;
      }).join('');
      const ra = rowSum(ga[i]) / ta, rb = rowSum(gb[i]) / tb;
      return `<tr><th class="rowh">${DATA.topics[i]}</th>${cells}
        <td class="tot">${rb > 0 ? '\\u00d7' + (ra / rb).toFixed(2) : '&infin;'}</td></tr>`;
    }).join('');
    document.getElementById('g-matrix').innerHTML =
      `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
    legend.innerHTML = div.map((c, i) =>
      `<span class="key"><i style="background:${c}"></i>${['\\u00bc\\u00d7','','\\u00bd\\u00d7','1\\u00d7','2\\u00d7','','4\\u00d7'][i]}</span>`).join('')
      + `<span>&mdash; log scale, clamped at 4&times;. \\u00d71.00 means the cell is the same share of both corpora.</span>`;
    return;
  }

  if (b) {
    h2.innerHTML = `${a.label} <span class="hint">${fmt(ta)} ${METRIC} &middot; ${a.stage}</span>
      &nbsp;vs&nbsp; ${b.label} <span class="hint">${fmt(tb)} ${METRIC} &middot; ${b.stage}</span>`;
    const head = `<tr><th class="rowh"></th><th colspan="${DATA.buckets.length + 1}"
        style="text-align:center;border-bottom:1px solid var(--border)">${a.label}</th>
      <th colspan="${DATA.buckets.length + 1}"
        style="text-align:center;border-bottom:1px solid var(--border)">${b.label}</th></tr>
      <tr><th class="rowh">Topic</th>${DATA.buckets.map(x => `<th>${x}</th>`).join('')}<th class="tot">Total</th>
      ${DATA.buckets.map(x => `<th>${x}</th>`).join('')}<th class="tot">Total</th></tr>`;
    const body = order.map(i =>
      `<tr><th class="rowh">${DATA.topics[i]}</th>` +
      ga[i].map((v, k) => gridCell(v, maxA, DATA.topics[i], DATA.buckets[k], a.label, ta)).join('') +
      `<td class="tot">${fmt(rowSum(ga[i]))}</td>` +
      gb[i].map((v, k) => gridCell(v, maxB, DATA.topics[i], DATA.buckets[k], b.label, tb)).join('') +
      `<td class="tot">${fmt(rowSum(gb[i]))}</td></tr>`).join('');
    const fa = DATA.buckets.map((_, k) => ga.reduce((s, r) => s + r[k], 0));
    const fb = DATA.buckets.map((_, k) => gb.reduce((s, r) => s + r[k], 0));
    const foot = `<tr class="foot"><th class="rowh">All topics</th>` +
      fa.map(v => `<td class="tot">${fmt(v)}</td>`).join('') + `<td class="tot">${fmt(ta)}</td>` +
      fb.map(v => `<td class="tot">${fmt(v)}</td>`).join('') + `<td class="tot">${fmt(tb)}</td></tr>` +
      `<tr><th class="rowh" style="color:var(--ink-3)">bucket share</th>` +
      fa.map(v => `<td style="color:var(--ink-3)">${pct(v / ta)}</td>`).join('') + `<td></td>` +
      fb.map(v => `<td style="color:var(--ink-3)">${pct(v / tb)}</td>`).join('') + `<td></td></tr>`;
    document.getElementById('g-matrix').innerHTML =
      `<table><thead>${head}</thead><tbody>${body}${foot}</tbody></table>`;
    legend.innerHTML = '<span>Low</span>' + seq.map(x => `<span class="key"><i style="background:${x}"></i></span>`).join('')
      + '<span>High &mdash; square-root scale, shaded <b>within each corpus</b> so the smaller one still '
      + 'shows its structure. Use A &divide; B for the direct comparison.</span>';
    return;
  }

  h2.innerHTML = `${a.label} &mdash; ${fmt(ta)} ${METRIC} <span class="hint">${a.stage}</span>`;
  const head = `<tr><th class="rowh">Topic</th>${DATA.buckets.map(x => `<th>${x}</th>`).join('')}
    <th class="tot">Total</th><th>Share</th></tr>`;
  const body = order.map(i =>
    `<tr><th class="rowh">${DATA.topics[i]}</th>` +
    ga[i].map((v, k) => gridCell(v, maxA, DATA.topics[i], DATA.buckets[k], a.label, ta)).join('') +
    `<td class="tot">${fmt(rowSum(ga[i]))}</td>
     <td style="color:var(--ink-3)">${pct(rowSum(ga[i]) / ta)}</td></tr>`).join('');
  const colTot = DATA.buckets.map((_, k) => ga.reduce((s, r) => s + r[k], 0));
  const foot = `<tr class="foot"><th class="rowh">All topics</th>${colTot.map(v =>
    `<td class="tot">${fmt(v)}</td>`).join('')}<td class="tot">${fmt(ta)}</td><td></td></tr>
    <tr><th class="rowh" style="color:var(--ink-3)">bucket share</th>${colTot.map(v =>
    `<td style="color:var(--ink-3)">${pct(v / ta)}</td>`).join('')}<td></td><td></td></tr>`;
  document.getElementById('g-matrix').innerHTML =
    `<table><thead>${head}</thead><tbody>${body}${foot}</tbody></table>`;
  legend.innerHTML = '<span>Low</span>' + seq.map(x => `<span class="key"><i style="background:${x}"></i></span>`).join('')
    + '<span>High &mdash; square-root scale within this corpus.</span>';
}
function setGridCorpus(i) { GRID_CORPUS = i; renderGridTab(); }
function setGridCorpusB(i) { GRID_CORPUS_B = i; renderGridTab(); }
function setGridMode(m) { GRID_MODE = m; press('grid-mode-seg', m === 'side' ? 0 : 1); renderGridTab(); }

/* ---------- driver ---------- */
function renderAll() {
  document.getElementById('metric-note').textContent =
    METRIC === 'tokens' ? 'gte tokens (capped at 8192) \\u2014 see note below' : 'document counts';
  renderCards();
  stackedRows('q-share', true);
  stackedRows('q-abs', false);
  renderQualityLegend();
  renderQualityTable();
  renderTopics();
  renderGridTab();
}
const proj = DATA.corpora[0];
document.getElementById('llama-note').textContent =
  (proj.llama_tokens.flat().reduce((s, v) => s + v, 0) / 1e9).toFixed(0) + 'B';
renderAll();
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', renderAll);
new MutationObserver(renderAll).observe(document.documentElement, {attributes: true,
  attributeFilter: ['data-theme']});
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compare", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data = json.loads(pathlib.Path(args.compare).read_text())
    pathlib.Path(args.out).write_text(build_page(data))
    logger.info("wrote %s (%d corpora)", args.out, len(data["corpora"]))


if __name__ == "__main__":
    main()
