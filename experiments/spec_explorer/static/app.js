// Copyright The Marin Authors — SPDX-License-Identifier: Apache-2.0
// Spec Explorer frontend: nav routing + performance plots + winners table + spec viewer.

const TABS = [
  { id: 'performance', title: 'Perf', sub: 'method × eval crossover', icon: iconPerf, pending: false },
  { id: 'winners', title: 'Winners', sub: 'best method per domain', icon: iconTrophy, pending: false },
  { id: 'specs', title: 'Specs', sub: 'what each method actually does', icon: iconDoc, pending: false },
  { id: 'coverage', title: 'Coverage', sub: 'Jaccard across strategies', icon: iconGrid, pending: true },
  { id: 'search', title: 'Search', sub: 'edge-case retrieval', icon: iconSearch, pending: true },
  { id: 'compare', title: 'Compare', sub: 'head-to-head A vs B', icon: iconCompare, pending: true },
];

const FAMILY_LABEL = {
  paloma: 'Paloma val loss', uncheatable: 'Uncheatable bpb',
  core_v2: 'DCLM Core v2', olmo_bpb: 'OLMo base_easy bpb', olmes: 'OLMES base',
};

const state = { rows: [], summary: null, methodStyle: {}, modelLabels: {}, updatedAt: null };

// ---------- boot ----------
async function boot() {
  buildRail();
  await loadMeta();
  await loadEval();
  route(location.hash.slice(1) || 'performance');
  document.getElementById('refresh-eval').addEventListener('click', refreshEval);
  window.addEventListener('hashchange', () => route(location.hash.slice(1) || 'performance'));
}

async function loadMeta() {
  const m = await fetch('/api/meta').then(r => r.json());
  state.methodStyle = m.method_style || {};
  state.modelLabels = m.model_labels || {};
}

async function loadEval() {
  const d = await fetch('/api/eval/matrix').then(r => r.json());
  state.rows = d.rows || [];
  state.summary = d.summary;
  state.updatedAt = d.updated_at;
  document.getElementById('eval-updated').textContent =
    state.updatedAt ? `evals: ${fmtTime(state.updatedAt)} · ${state.rows.length} rows` : 'evals: not built';
  buildPerfControls();
  buildWinControls();
}

async function refreshEval() {
  const btn = document.getElementById('refresh-eval');
  btn.classList.add('busy'); btn.innerHTML = '<span class="dot"></span> scanning GCS…';
  try {
    await fetch('/api/eval/refresh', { method: 'POST' });
    await loadEval();
    renderPerf(); renderWinners();
  } finally {
    btn.classList.remove('busy'); btn.innerHTML = '<span class="dot"></span> refresh evals';
  }
}

function colorFor(method) { return (state.methodStyle[method] || {}).color || '#888'; }
function labelFor(method) { return (state.methodStyle[method] || {}).label || method; }
function methodsIn(rows) {
  // stable order following METHOD_STYLE, then any extras
  const known = Object.keys(state.methodStyle);
  const present = new Set(rows.map(r => r.method));
  return known.filter(m => present.has(m)).concat([...present].filter(m => !known.includes(m)));
}

// ---------- nav / routing ----------
function buildRail() {
  const rail = document.getElementById('rail');
  TABS.forEach(t => {
    const b = document.createElement('button');
    b.className = 'nav-btn' + (t.pending ? ' pending' : '');
    b.dataset.tab = t.id;
    b.innerHTML = `${t.icon()}<span>${t.title}</span>`;
    b.addEventListener('click', () => { location.hash = t.id; });
    rail.appendChild(b);
  });
}

function route(tabId) {
  const tab = TABS.find(t => t.id === tabId) || TABS[0];
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab.id));
  document.querySelectorAll('.tab').forEach(s => s.classList.toggle('active', s.dataset.tab === tab.id));
  document.getElementById('tab-title').textContent = tab.title;
  document.getElementById('tab-sub').textContent = tab.sub;
  if (tab.id === 'performance') renderPerf();
  if (tab.id === 'winners') renderWinners();
  if (tab.id === 'specs') ensureSpecs();
  if (tab.id === 'coverage') ensureCoverage();
  if (tab.id === 'search') ensureSearch();
  if (tab.id === 'compare') ensureCompare();
}

// ---------- shared control helpers ----------
function selectEl(id, label, options, value, onChange) {
  const wrap = document.createElement('div'); wrap.className = 'ctl';
  const lab = document.createElement('label'); lab.textContent = label; lab.htmlFor = id;
  const sel = document.createElement('select'); sel.id = id;
  options.forEach(o => {
    const opt = document.createElement('option');
    opt.value = String(o.value); opt.textContent = o.label;
    if (String(o.value) === String(value)) opt.selected = true;
    sel.appendChild(opt);
  });
  sel.addEventListener('change', () => onChange(sel.value));
  wrap.appendChild(lab); wrap.appendChild(sel);
  return wrap;
}
const num = v => Number(v);
const fmtBudget = b => Number(b).toExponential(0).replace('e+', 'e');

// ---------- PERFORMANCE ----------
const FASTPIPE_METHOD_RE = /^fastpipe/;
const perf = { family: 'core_v2', task: null, dim: null, n: null, showFastpipe: false, layout: 'single', hidden: new Set() };

// Layout modes for the performance tab. Every panel is the same scaling ladder
// (metric vs training FLOPs, one line per method); the mode picks how many panels
// and what is held fixed vs faceted.
const PERF_LAYOUTS = [
  { value: 'single', label: 'single plot' },
  { value: 'by_size', label: 'all model sizes' },  // facet over model size, fix N
  { value: 'by_warc', label: 'all WARC counts' },  // facet over N, fix model size
];

const EPOCH_CAPTION = 'Vertical <b>dashed</b> mark per method = 1 epoch on its data. '
  + 'The rise past it is multi-epoch overfitting at fixed WARCs; sub-epoch points '
  + '(left of the dashed line) are undertrained and noisy.';

// dim -> params label ("2.9B"). Prefer the live map from /api/meta (updates
// without an eval rescan when a new dim appears); fall back to the row's baked
// model_label, then the raw dim.
function dimLabel(dim) {
  if (state.modelLabels[dim] != null) return state.modelLabels[dim];
  const r = state.rows.find(x => x.dim === dim && x.model_label);
  return r ? r.model_label : `d${dim}`;
}

// Compact token count, e.g. 60_100_000_000 -> "60B", 850_000_000 -> "850M".
function fmtTokens(t) {
  if (!t) return '';
  if (t >= 1e9) return (t / 1e9 >= 10 ? Math.round(t / 1e9) : (t / 1e9).toFixed(1).replace(/\.0$/, '')) + 'B';
  if (t >= 1e6) return Math.round(t / 1e6) + 'M';
  return String(t);
}

function buildPerfControls() {
  const box = document.getElementById('perf-controls');
  box.innerHTML = '';
  if (!state.summary) { box.innerHTML = notBuiltHint(); return; }
  const s = state.summary;
  if (!s.families.includes(perf.family)) perf.family = s.families[0];
  const tasks = s.tasks[perf.family] || [];
  if (!tasks.includes(perf.task)) perf.task = defaultTask(perf.family, tasks);
  if (perf.dim == null || !s.dims.includes(perf.dim)) perf.dim = s.dims[Math.min(2, s.dims.length - 1)];
  if (perf.n == null || !s.n_warcs.includes(perf.n)) perf.n = pickDefaultN(s.n_warcs);

  box.appendChild(selectEl('perf-family', 'eval family',
    s.families.map(f => ({ value: f, label: FAMILY_LABEL[f] || f })), perf.family, v => { perf.family = v; perf.task = null; buildPerfControls(); renderPerf(); }));
  box.appendChild(selectEl('perf-task', 'task / domain',
    tasks.map(t => ({ value: t, label: prettyTask(t) })), perf.task, v => { perf.task = v; renderPerfMethods(); renderPerf(); }));
  box.appendChild(selectEl('perf-layout', 'view',
    PERF_LAYOUTS, perf.layout, v => { perf.layout = v; buildPerfControls(); renderPerf(); }));
  // Model size is fixed (a picker) except when faceting over it; likewise for N.
  if (perf.layout !== 'by_size')
    box.appendChild(selectEl('perf-dim', 'model size',
      s.dims.map(d => ({ value: d, label: dimLabel(d) })), perf.dim, v => { perf.dim = num(v); renderPerf(); }));
  if (perf.layout !== 'by_warc')
    box.appendChild(selectEl('perf-n', 'WARCs (N)',
      s.n_warcs.map(n => ({ value: n, label: `N=${n}` })), perf.n, v => { perf.n = num(v); renderPerf(); }));
  box.appendChild(selectEl('perf-fastpipe', 'fastpipe bands',
    [{ value: 'off', label: 'hidden' }, { value: 'on', label: 'shown' }],
    perf.showFastpipe ? 'on' : 'off', v => { perf.showFastpipe = v === 'on'; renderPerfMethods(); renderPerf(); }));
  const dir = document.createElement('div'); dir.className = 'ctl';
  dir.innerHTML = `<label>direction</label><span class="badge ${higherBetter(perf.family) ? 'dir-higher' : 'dir-lower'}">${higherBetter(perf.family) ? 'higher is better' : 'lower is better'}</span>`;
  box.appendChild(dir);
  renderPerfMethods();
}

// Every method present for the current (family, task) across all sizes/N, honoring
// the fastpipe toggle — the universe of series the toggle strip can show/hide.
function perfMethodsAvailable() {
  if (!state.rows) return [];
  const rows = state.rows.filter(r => r.eval_family === perf.family && r.task === perf.task);
  return methodsIn(rows).filter(m => perf.showFastpipe || !FASTPIPE_METHOD_RE.test(m));
}

// Clickable per-method show/hide chips. Persist in perf.hidden and apply to every
// panel (single and faceted) via the perfCellTraces filter.
function renderPerfMethods() {
  const box = document.getElementById('perf-methods');
  if (!box) return;
  const methods = perfMethodsAvailable();
  if (methods.length < 2) { box.innerHTML = ''; return; }
  const chips = methods.map(m => {
    const off = perf.hidden.has(m) ? ' off' : '';
    return `<button class="mtoggle${off}" data-m="${escapeHtml(m)}" title="${off ? 'show' : 'hide'} ${escapeHtml(labelFor(m))}">`
      + `<span class="sw" style="background:${colorFor(m)}"></span>${escapeHtml(labelFor(m))}</button>`;
  }).join('');
  const nHidden = methods.filter(m => perf.hidden.has(m)).length;
  box.innerHTML = `<span class="mt-label">methods</span>${chips}`
    + `<span class="mt-sep"></span>`
    + `<button class="mt-act" data-act="all">all</button>`
    + `<button class="mt-act" data-act="none">none</button>`
    + (nHidden ? `<span class="mt-count">${methods.length - nHidden}/${methods.length} shown</span>` : '');
  box.querySelectorAll('[data-m]').forEach(b => b.addEventListener('click', () => {
    const m = b.dataset.m;
    if (perf.hidden.has(m)) perf.hidden.delete(m); else perf.hidden.add(m);
    renderPerfMethods(); renderPerf();
  }));
  box.querySelectorAll('[data-act]').forEach(b => b.addEventListener('click', () => {
    if (b.dataset.act === 'none') perfMethodsAvailable().forEach(m => perf.hidden.add(m));
    else perfMethodsAvailable().forEach(m => perf.hidden.delete(m));
    renderPerfMethods(); renderPerf();
  }));
}

function clearPlots(el) {
  el.querySelectorAll('.js-plotly-plot').forEach(p => Plotly.purge(p));
  el.innerHTML = '';
}

// Build method lines + epoch marks for one (dim, N) panel. Returns the traces
// plus flags so the caller can size the panel and render a shared legend.
async function perfCellTraces(dim, n, tokens = {}) {
  const rows = state.rows.filter(r =>
    r.eval_family === perf.family && r.task === perf.task && r.dim === dim && r.n_warcs === n);
  if (!rows.length) return { traces: [], hasData: false, drewEpochs: false, present: [] };
  const present = methodsIn(rows).filter(m => (perf.showFastpipe || !FASTPIPE_METHOD_RE.test(m)) && !perf.hidden.has(m));
  // At a fixed WARC count, order legend/traces by token count (most data first).
  if (Object.keys(tokens).length) present.sort((a, b) => (tokens[b] || 0) - (tokens[a] || 0));
  const traces = present.map(method => {
    const pts = rows.filter(r => r.method === method).sort((a, b) => a.budget - b.budget);
    const tokLabel = tokens[method] ? ` · ${fmtTokens(tokens[method])} tok` : '';
    return {
      x: pts.map(p => p.budget), y: pts.map(p => p.value),
      mode: 'lines+markers', name: labelFor(method) + tokLabel,
      line: { color: colorFor(method), width: 2 },
      marker: { color: colorFor(method), size: 7, line: { color: '#14130e', width: 1.5 }, opacity: 0.95 },
      opacity: 0.9,
      hovertemplate: `${labelFor(method)}: %{y:.4f}<extra></extra>`,
    };
  });
  // Vertical epoch marks per method (dashed = 1 epoch, dotted = 2), drawn as
  // real line traces (not layout.shapes) so they name the method + epoch on
  // hover — thin colored lines are otherwise impossible to tell apart. x uses
  // the raw FLOP value; y spans the data range so the whole line is hoverable.
  const epochTraces = [];
  try {
    const epochs = await fetch(`/api/epochs?dim=${dim}&n=${n}`).then(r => r.json());
    const yv = rows.map(r => r.value);
    const yLo = Math.min(...yv), yHi = Math.max(...yv);
    const pad = 0.05 * ((yHi - yLo) || 1);
    const mk = (method, x, epochN, dash) => ({
      x: [x, x], y: [yLo - pad, yHi + pad], mode: 'lines', showlegend: false,
      line: { color: colorFor(method), width: 1.5, dash }, opacity: 0.6,
      hovertemplate: `${labelFor(method)} — ${epochN} epoch${epochN > 1 ? 's' : ''}<br>%{x:.2e} FLOPs<extra></extra>`,
    });
    present.forEach(method => {
      const ef = epochs[method];
      if (!ef) return;
      epochTraces.push(mk(method, ef.e1, 1, 'dash'));
    });
  } catch (e) { /* epochs are best-effort */ }
  return { traces: traces.concat(epochTraces), hasData: true, drewEpochs: epochTraces.length > 0, present };
}

async function renderPerf() {
  if (!state.summary) return;
  const el = document.getElementById('perf-plot');
  const cap = document.getElementById('perf-caption');
  clearPlots(el);
  const ytitle = higherBetter(perf.family) ? `${prettyTask(perf.task)} (↑)` : `${prettyTask(perf.task)} (↓)`;
  // Tokens are meaningful only at a fixed WARC count (single + by_size); when we
  // facet over N (by_warc) each panel has a different token count, so skip them.
  const tokens = perf.layout === 'by_warc' ? {} : await fetch(`/api/tokens?n=${perf.n}`).then(r => r.json()).catch(() => ({}));

  if (perf.layout === 'single') {
    const plot = document.createElement('div'); plot.style.width = '100%'; plot.style.height = '520px'; el.appendChild(plot);
    const c = await perfCellTraces(perf.dim, perf.n, tokens);
    if (!c.hasData) { el.innerHTML = emptyPlot('No runs for this (family, task, size, N).'); if (cap) cap.textContent = ''; return; }
    const layout = darkLayout(
      `${FAMILY_LABEL[perf.family]} — ${prettyTask(perf.task)}  ·  ${dimLabel(perf.dim)} · N=${perf.n}`,
      'training FLOPs', ytitle);
    Plotly.newPlot(plot, c.traces, layout, { responsive: true, displaylogo: false });
    if (cap) cap.innerHTML = c.drewEpochs ? EPOCH_CAPTION : '';
    return;
  }

  // Faceted small-multiples: one panel per faceted value (the other axis fixed).
  // Only panels with ≥1 run are shown — most (size, N) combos were never trained.
  const hasData = (dim, n) => state.rows.some(r =>
    r.eval_family === perf.family && r.task === perf.task && r.dim === dim && r.n_warcs === n);
  const all = perf.layout === 'by_size'
    ? state.summary.dims.map(d => ({ dim: d, n: perf.n, title: dimLabel(d) }))
    : state.summary.n_warcs.map(n => ({ dim: perf.dim, n, title: `N=${n}` }));
  const cells = all.filter(c => hasData(c.dim, c.n));
  const omitted = all.length - cells.length;
  const fixed = perf.layout === 'by_size' ? `N=${perf.n}` : dimLabel(perf.dim);
  if (!cells.length) { el.innerHTML = emptyPlot(`No runs at ${fixed} for this family/task.`); if (cap) cap.textContent = ''; return; }

  const legendHost = document.createElement('div'); legendHost.className = 'facet-legend'; el.appendChild(legendHost);
  const grid = document.createElement('div'); grid.className = 'facet-grid';
  // Up to 5 panels per row (fewer if there are fewer panels); they shrink to fit.
  grid.style.gridTemplateColumns = `repeat(${Math.min(cells.length, 5)}, minmax(0, 1fr))`;
  el.appendChild(grid);

  const results = await Promise.all(cells.map(async cell => {
    const wrap = document.createElement('div'); wrap.className = 'facet-cell';
    const title = document.createElement('div'); title.className = 'facet-title'; title.textContent = cell.title;
    const plot = document.createElement('div'); plot.className = 'facet-plot';
    wrap.appendChild(title); wrap.appendChild(plot); grid.appendChild(wrap);
    const c = await perfCellTraces(cell.dim, cell.n, tokens);
    if (!c.hasData) { plot.innerHTML = '<div class="facet-empty">no runs</div>'; return c; }
    const layout = darkLayout('', 'FLOPs', ytitle);
    layout.showlegend = false;
    layout.height = 300;
    layout.margin = { l: 56, r: 14, t: 8, b: 42 };
    Plotly.newPlot(plot, c.traces, layout, { responsive: true, displaylogo: false });
    return c;
  }));

  // One shared legend (union of methods present across panels) instead of a
  // duplicated per-panel legend.
  const methods = [];
  results.forEach(c => (c.present || []).forEach(m => { if (!methods.includes(m)) methods.push(m); }));
  if (Object.keys(tokens).length) methods.sort((a, b) => (tokens[b] || 0) - (tokens[a] || 0));
  legendHost.innerHTML = methods.length
    ? methods.map(m => {
        const tok = tokens[m] ? ` <span class="ftok">${fmtTokens(tokens[m])} tok</span>` : '';
        return `<span class="fleg"><span class="sw" style="background:${colorFor(m)}"></span>${labelFor(m)}${tok}</span>`;
      }).join('')
    : '<span class="hint" style="margin:0">no methods for this selection</span>';

  const anyEpochs = results.some(c => c.drewEpochs);
  if (cap) cap.innerHTML = `Each panel: <b>${FAMILY_LABEL[perf.family]} — ${prettyTask(perf.task)}</b> vs training FLOPs · `
    + (perf.layout === 'by_size' ? `every model size at <b>${fixed}</b>.` : `every WARC count at <b>${fixed}</b>.`)
    + (anyEpochs ? ' Vertical <b>dashed</b> mark = 1 epoch.' : '');
}

// ---------- WINNERS ----------
const win = { family: 'olmo_bpb', dim: null, n: null, budget: 'best' };

function buildWinControls() {
  const box = document.getElementById('win-controls');
  box.innerHTML = '';
  if (!state.summary) { box.innerHTML = notBuiltHint(); return; }
  const s = state.summary;
  if (!s.families.includes(win.family)) win.family = s.families[0];
  if (win.dim == null || !s.dims.includes(win.dim)) win.dim = s.dims[Math.min(2, s.dims.length - 1)];
  if (win.n == null || !s.n_warcs.includes(win.n)) win.n = pickDefaultN(s.n_warcs);
  const budgets = s.budgets.filter(b => hasBudget(win.family, win.dim, win.n, b));

  box.appendChild(selectEl('win-family', 'eval family',
    s.families.map(f => ({ value: f, label: FAMILY_LABEL[f] || f })), win.family, v => { win.family = v; buildWinControls(); renderWinners(); }));
  box.appendChild(selectEl('win-dim', 'model size',
    s.dims.map(d => ({ value: d, label: `d${d}` })), win.dim, v => { win.dim = num(v); buildWinControls(); renderWinners(); }));
  box.appendChild(selectEl('win-n', 'WARCs (N)',
    s.n_warcs.map(n => ({ value: n, label: `N=${n}` })), win.n, v => { win.n = num(v); buildWinControls(); renderWinners(); }));
  box.appendChild(selectEl('win-budget', 'FLOP budget',
    [{ value: 'best', label: 'best over budgets' }].concat(budgets.map(b => ({ value: b, label: fmtBudget(b) }))),
    win.budget, v => { win.budget = v; renderWinners(); }));
}

function renderWinners() {
  if (!state.summary) return;
  const host = document.getElementById('win-table');
  const tasks = (state.summary.tasks[win.family] || []).slice();
  const rows = state.rows.filter(r => r.eval_family === win.family && r.dim === win.dim && r.n_warcs === win.n
    && (win.budget === 'best' || Math.abs(r.budget - num(win.budget)) < 1e-9));
  if (!rows.length) { host.innerHTML = `<div class="empty-state"><div class="big">No data</div><div>No ${FAMILY_LABEL[win.family]} runs at d${win.dim}, N=${win.n}.</div></div>`; return; }

  const higher = higherBetter(win.family);
  const methods = methodsIn(rows);
  // value[method][task] = reduced value (best over budget if requested)
  const value = {};
  methods.forEach(m => value[m] = {});
  rows.forEach(r => {
    const cur = value[r.method][r.task];
    if (cur === undefined) value[r.method][r.task] = r.value;
    else value[r.method][r.task] = win.budget === 'best' ? (higher ? Math.max(cur, r.value) : Math.min(cur, r.value)) : r.value;
  });
  // winner per task
  const winner = {}, winCount = {};
  methods.forEach(m => winCount[m] = 0);
  tasks.forEach(t => {
    let best = null, bestM = null;
    methods.forEach(m => {
      const v = value[m][t];
      if (v === undefined) return;
      if (best === null || (higher ? v > best : v < best)) { best = v; bestM = m; }
    });
    winner[t] = bestM;
    if (bestM) winCount[bestM]++;
  });

  const orderTasks = ['macro', 'Core_v2', 'Core'].filter(t => tasks.includes(t))
    .concat(tasks.filter(t => !['macro', 'Core_v2', 'Core'].includes(t)));

  let h = '<div class="hint">★ = best in column · ' + (higher ? 'higher is better' : 'lower is better') +
    ` · ${win.budget === 'best' ? "each cell = method's best over budgets" : 'FLOP budget ' + fmtBudget(num(win.budget))}</div>`;
  h += '<div class="table-scroll"><table class="grid"><thead><tr><th class="method">method</th><th>wins</th>';
  orderTasks.forEach(t => h += `<th title="${t}">${prettyTask(t)}</th>`);
  h += '</tr></thead><tbody>';
  methods.forEach(m => {
    h += `<tr><td class="method"><span class="mchip"><span class="bar" style="background:${colorFor(m)}"></span>${labelFor(m)}</span></td>`;
    h += `<td class="wincount">${winCount[m] || ''}</td>`;
    orderTasks.forEach(t => {
      const v = value[m][t];
      if (v === undefined) { h += '<td class="empty">—</td>'; return; }
      const isWin = winner[t] === m;
      h += `<td class="${isWin ? 'win' : ''}">${fmtVal(v)}</td>`;
    });
    h += '</tr>';
  });
  h += '</tbody></table></div>';
  host.innerHTML = h;
}

// ---------- SPECS ----------
let specsLoaded = false;
const specState = { methods: [], selected: null, phase: 0, detail: null };

async function ensureSpecs() {
  if (specsLoaded) return;
  specsLoaded = true;
  specState.methods = await fetch('/api/spec/methods').then(r => r.json());
  renderSpecList();
}

function renderSpecList() {
  const host = document.getElementById('spec-list');
  const groups = {};
  specState.methods.forEach(m => { (groups[m.kind] = groups[m.kind] || []).push(m); });
  const KIND_LABEL = { single_prompt: 'Single-call specs', two_stage: 'Two-stage pipelines', one_call: 'One-call pipelines', threshold: 'Classifier bands', external: 'External extractors' };
  let h = '';
  Object.keys(KIND_LABEL).filter(k => groups[k]).forEach(kind => {
    h += `<div class="spec-group-label">${KIND_LABEL[kind]}</div>`;
    groups[kind].forEach(m => {
      const active = specState.selected === m.method ? ' active' : '';
      h += `<button class="spec-item${active}" data-m="${m.method}"><span>${m.method}</span><span class="kind">${m.kind.replace('_', ' ')}</span></button>`;
    });
  });
  host.innerHTML = h;
  host.querySelectorAll('.spec-item').forEach(b => b.addEventListener('click', () => selectSpec(b.dataset.m)));
}

async function selectSpec(method) {
  specState.selected = method; specState.phase = 0;
  renderSpecList();
  const d = document.getElementById('spec-detail');
  d.innerHTML = '<div class="empty-state"><span class="spin"></span></div>';
  specState.detail = await fetch(`/api/spec/${encodeURIComponent(method)}`).then(r => r.json());
  renderSpecDetail();
}

function renderSpecDetail() {
  const d = document.getElementById('spec-detail');
  const s = specState.detail;
  if (!s || s.error) { d.innerHTML = `<div class="empty-state">${s ? s.error : 'error'}</div>`; return; }
  let h = `<h2>${s.title}</h2><div><span class="badge">${s.kind.replace('_', ' ')}</span></div>`;
  h += `<p class="desc">${escapeHtml(s.description || '')}</p>`;
  if (s.budget) {
    h += '<div class="budget-grid">';
    Object.entries(s.budget).forEach(([k, v]) => h += `<span class="kv">${k} <b>${v}</b></span>`);
    h += '</div>';
  }
  if (s.meta && Object.keys(s.meta).length) {
    h += '<div class="budget-grid">';
    Object.entries(s.meta).forEach(([k, v]) => h += `<span class="kv">${k} <b>${escapeHtml(String(v))}</b></span>`);
    h += '</div>';
  }
  if (s.phases && s.phases.length) {
    h += '<div class="phase-tabs">';
    s.phases.forEach((p, i) => h += `<button class="phase-tab${i === specState.phase ? ' active' : ''}" data-i="${i}">${p.name}</button>`);
    h += '</div>';
    h += `<pre class="prompt">${escapeHtml(s.phases[specState.phase].text)}</pre>`;
  } else {
    h += '<p class="hint">No prompt text — this method is a classifier threshold or an external extractor.</p>';
  }
  d.innerHTML = h;
  d.querySelectorAll('.phase-tab').forEach(b => b.addEventListener('click', () => { specState.phase = num(b.dataset.i); renderSpecDetail(); }));
}

// ---------- COVERAGE ----------
let coverageInit = false;
const cov = { key: 'url_h', matrix: null, states: {}, hidden: new Set(), mode: 'containment' };
const FASTPIPE_RE = /^fastpipe/;
function visibleDatasets() { return (cov.matrix ? cov.matrix.datasets : []).filter(d => !cov.hidden.has(d)); }

async function ensureCoverage() {
  if (coverageInit) return;
  coverageInit = true;
  renderCovControls();
  await loadCoverage();  // text resolves locally; no worker strip needed
}

// ---- worker control strip ----
let workerState = [];
async function refreshWorkers() {
  workerState = await fetch('/api/worker/status').then(r => r.json()).catch(() => []);
  renderWorkerStrip();
}
function renderWorkerStrip() {
  const host = document.getElementById('cov-workers');
  if (!host) return;
  let h = '<div class="worker-strip"><span class="wlabel">in-region workers</span>';
  workerState.forEach(w => {
    const cls = w.status === 'RUNNING' ? 'running' : w.status === 'STARTING' ? 'pending' : w.status === 'NONE' ? '' : 'pending';
    const btn = (w.workers && w.workers.length)
      ? `<button data-stop="${w.region}">stop</button>`
      : `<button data-launch="${w.region}">launch</button>`;
    // HA replica summary: warm/desired, and how many are still coming up.
    const warm = w.warm ?? 0, running = w.running ?? 0, desired = w.desired ?? 1;
    const repl = w.status === 'NONE' ? 'none' : `${warm}/${desired} warm` + (running > warm ? ` · ${running - warm} warming` : '');
    const tip = (w.workers || []).map(x => `${x.name.split('-').pop()}:${x.state.toLowerCase()}${x.warm ? '·warm' : ''}`).join(' | ');
    h += `<span class="wchip ${cls}" title="${tip}"><span class="sd"></span>${w.region} <span style="color:var(--ink-faint)">${w.datasets.length}ds · ${repl}</span> ${btn}</span>`;
  });
  h += '<span class="hint" style="margin:0">2 HA replicas/region; one serves while a preempted one re-warms (~7 min, in-region/free).</span></div>';
  host.innerHTML = h;
  host.querySelectorAll('[data-launch]').forEach(b => b.addEventListener('click', () => launchWorker(b.dataset.launch)));
  host.querySelectorAll('[data-stop]').forEach(b => b.addEventListener('click', () => stopWorker(b.dataset.stop)));
}
async function launchWorker(region) {
  const host = document.getElementById('cov-workers');
  host.querySelector(`[data-launch="${region}"]`).textContent = 'launching…';
  await fetch('/api/worker/launch', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ region }) }).then(r => r.json());
  await refreshWorkers();
}
async function stopWorker(region) {
  await fetch('/api/worker/stop', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ region }) });
  await refreshWorkers();
}

function renderCovControls() {
  const box = document.getElementById('cov-controls');
  box.innerHTML = '';
  box.appendChild(selectEl('cov-key', 'identity key',
    [{ value: 'url_h', label: 'url_h — universal (per URL)' },
     { value: 'rid_h', label: 'rid_h — source page' },
     { value: 'text_h', label: 'text_h — identical text' },
     { value: 'dom_h', label: 'dom_h — domain' }],
    cov.key, v => { cov.key = v; cov.matrix = null; loadCoverage(); }));
  box.appendChild(selectEl('cov-mode', 'view',
    [{ value: 'containment', label: 'Containment — row ⊆ col' },
     { value: 'jaccard', label: 'Jaccard — symmetric overlap' }],
    cov.mode, v => { cov.mode = v; renderCovControls(); renderHeatmap(); }));
  const note = document.createElement('div'); note.className = 'ctl';
  note.innerHTML = cov.mode === 'containment'
    ? `<label>reading</label><span class="badge">cell = fraction of ROW also in COL (=|A∩B|/|A|)</span>`
    : `<label>reading</label><span class="badge">Jaccard = |A∩B| / |A∪B|</span>`;
  box.appendChild(note);
}

async function loadCoverage() {
  const hm = document.getElementById('cov-heatmap');
  const sd = document.getElementById('cov-setdiff');
  hm.innerHTML = '<div class="empty-state"><span class="spin"></span> loading keys (first time downloads ~264 MB)…</div>';
  sd.innerHTML = '<div class="empty-state"><span class="spin"></span></div>';
  const m = await fetch(`/api/coverage?key=${cov.key}`).then(r => r.json());
  cov.matrix = m;
  if (!m.datasets || !m.datasets.length) { hm.innerHTML = emptyPlot('No coverage keys available.'); return; }
  if (!cov._defaultsApplied) {  // fastpipe bands hidden by default (declutter)
    m.datasets.filter(d => FASTPIPE_RE.test(d)).forEach(d => cov.hidden.add(d));
    cov._defaultsApplied = true;
  }
  renderCovFilter();
  renderHeatmap();
  renderSetdiff();
}

function renderCovFilter() {
  const host = document.getElementById('cov-filter');
  if (!host || !cov.matrix) return;
  let h = '<div class="cov-filter"><span class="flabel">datasets in grid</span>';
  cov.matrix.datasets.forEach(d => {
    const off = cov.hidden.has(d) ? ' off' : '';
    h += `<span class="fchip${off}" data-d="${d}"><span class="bar" style="background:${colorFor(d)}"></span>${d}</span>`;
  });
  const anyFastpipe = cov.matrix.datasets.some(d => FASTPIPE_RE.test(d));
  const allFastpipeHidden = cov.matrix.datasets.filter(d => FASTPIPE_RE.test(d)).every(d => cov.hidden.has(d));
  if (anyFastpipe) h += `<span class="fchip quick" data-quick="fastpipe">${allFastpipeHidden ? 'show' : 'hide'} fastpipe bands</span>`;
  h += `<span class="fchip quick" data-quick="all">show all</span></div>`;
  host.innerHTML = h;
  host.querySelectorAll('.fchip[data-d]').forEach(c => c.addEventListener('click', () => {
    const d = c.dataset.d;
    cov.hidden.has(d) ? cov.hidden.delete(d) : cov.hidden.add(d);
    renderCovFilter(); renderHeatmap(); renderSetdiff();
  }));
  host.querySelectorAll('.fchip[data-quick]').forEach(c => c.addEventListener('click', () => {
    const q = c.dataset.quick;
    if (q === 'all') cov.hidden.clear();
    else if (q === 'fastpipe') {
      const fp = cov.matrix.datasets.filter(d => FASTPIPE_RE.test(d));
      if (fp.every(d => cov.hidden.has(d))) fp.forEach(d => cov.hidden.delete(d));
      else fp.forEach(d => cov.hidden.add(d));
    }
    renderCovFilter(); renderHeatmap(); renderSetdiff();
  }));
}

function renderHeatmap() {
  const m = cov.matrix;
  const hm = document.getElementById('cov-heatmap');
  const vis = visibleDatasets();
  // NOTE: never set hm.innerHTML on a live Plotly div — it corrupts Plotly's
  // internal state and the next react() silently fails. Purge instead.
  if (vis.length < 2) { Plotly.purge(hm); hm.innerHTML = emptyPlot('Select at least two datasets to compare.'); return; }
  const contain = cov.mode === 'containment';
  const idx = {}; m.datasets.forEach((d, i) => idx[d] = i);
  const M = contain ? m.containment : m.jaccard;
  const z = vis.map(a => vis.map(b => M[idx[a]][idx[b]]));
  const sizeText = vis.map(a => vis.map(b => {
    if (a === b) return `${a}<br>${(m.sizes[a] || 0).toLocaleString()} keys`;
    const pair = m.pairs.find(p => p.a === a && p.b === b);
    if (!pair) return '';
    if (contain) return `<b>${(pair.containment_a_in_b * 100).toFixed(1)}%</b> of ${a}<br>is also in ${b}<br>${pair.intersection.toLocaleString()} of ${pair.a_size.toLocaleString()} · ${a}-only ${pair.a_only.toLocaleString()}`;
    return `${a} ∩ ${b}<br>jaccard ${pair.jaccard.toFixed(3)}<br>${a}-only ${pair.a_only.toLocaleString()}<br>${b}-only ${pair.b_only.toLocaleString()}`;
  }));
  // Reset fully before drawing: purge clears Plotly's internal state, then we
  // clear any leftover spinner HTML, then newPlot rebuilds. This is safe on both
  // the first render (div holds a loading spinner) and re-renders (div is a plot).
  Plotly.purge(hm);
  hm.innerHTML = '';
  Plotly.newPlot(hm, [{
    type: 'heatmap', z: z, x: vis, y: vis,
    text: sizeText, hovertemplate: '%{text}<extra></extra>',
    colorscale: 'Cividis', zmin: 0, zmax: 1, xgap: 2, ygap: 2,
    colorbar: { title: { text: contain ? 'contained' : 'Jaccard', font: { size: 10 } }, tickfont: { size: 9 }, outlinewidth: 0, len: 0.85 },
  }], {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'IBM Plex Mono, monospace', color: '#a49e8d', size: 10 },
    title: { text: contain ? `Containment · ${cov.key}  (row ⊆ col)` : `Pairwise Jaccard · ${cov.key}`, font: { family: 'Bricolage Grotesque, sans-serif', color: '#ece7db', size: 15 }, x: 0.01, xanchor: 'left' },
    margin: { l: 130, r: 20, t: 46, b: 120 },
    xaxis: { title: { text: contain ? 'col = superset →' : '', font: { size: 10 } }, tickangle: -45, tickfont: { size: 10 }, automargin: true },
    yaxis: { autorange: 'reversed', tickfont: { size: 10 }, automargin: true },
  }, { responsive: true, displaylogo: false });

  const rates = m.url_match_rate || {};
  const low = Object.entries(rates).filter(([d, v]) => vis.includes(d) && v != null && v < 0.9).map(([d, v]) => `${d} ${(v * 100).toFixed(0)}%`);
  document.getElementById('cov-caveat').innerHTML = low.length
    ? `⚠ low url_match_rate (${low.join(', ')}) — a small overlap here may be provenance-join loss, not true disjointness.`
    : `All datasets have high url_match_rate — overlaps are trustworthy.`;
}

function renderSetdiff() {
  const host = document.getElementById('cov-setdiff');
  const vis = visibleDatasets();
  vis.forEach(d => { if (!(d in cov.states)) cov.states[d] = 'neutral'; });
  let h = '<div class="setdiff-title">Set-difference explorer</div>';
  h += '<p class="hint">Click a dataset to cycle: neutral → <span style="color:var(--good)">must contain</span> → <span style="color:var(--bad)">must exclude</span>. Answers "which docs does X keep that Y drops?"</p>';
  h += '<div class="chip-row">';
  vis.forEach(d => {
    const st = cov.states[d];
    h += `<span class="dchip" data-d="${d}" data-state="${st}" title="${(cov.matrix.sizes[d] || 0).toLocaleString()} keys">
      <span class="bar" style="background:${colorFor(d)}"></span>${d}<span class="st">${st === 'include' ? '∈ all' : st === 'exclude' ? '∉' : '·'}</span></span>`;
  });
  h += '</div>';
  h += '<div style="display:flex;gap:10px;align-items:center;margin-top:6px">';
  h += '<button class="btn primary" id="cov-compute">compute</button>';
  h += '<button class="btn" id="cov-clear">clear</button></div>';
  h += '<div id="cov-result"></div>';
  host.innerHTML = h;
  host.querySelectorAll('.dchip').forEach(c => c.addEventListener('click', () => {
    const d = c.dataset.d;
    cov.states[d] = { neutral: 'include', include: 'exclude', exclude: 'neutral' }[cov.states[d]];
    renderSetdiff();
  }));
  document.getElementById('cov-compute').addEventListener('click', computeSetdiff);
  document.getElementById('cov-clear').addEventListener('click', () => { vis.forEach(d => cov.states[d] = 'neutral'); renderSetdiff(); });
}

async function computeSetdiff() {
  const vis = new Set(visibleDatasets());
  const include = Object.keys(cov.states).filter(d => cov.states[d] === 'include' && vis.has(d));
  const exclude = Object.keys(cov.states).filter(d => cov.states[d] === 'exclude' && vis.has(d));
  const res = document.getElementById('cov-result');
  if (!include.length) { res.innerHTML = '<p class="hint">Pick at least one “must contain” dataset.</p>'; return; }
  res.innerHTML = '<div class="empty-state" style="padding:24px"><span class="spin"></span></div>';
  const d = await fetch('/api/coverage/setdiff', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ include, exclude, key: cov.key, sample: 0 }),
  }).then(r => r.json());
  const expr = `<b>${include.join(' ∩ ')}</b>${exclude.length ? ' \\ <b>' + exclude.join(' ∪ ') + '</b>' : ''}`;
  const canView = cov.key === 'url_h' && d.count > 0;
  res.innerHTML =
    `<p class="setdiff-expr">docs in ${expr} <span class="hint">(by ${cov.key})</span></p>` +
    `<div class="bignum">${d.count.toLocaleString()}</div>` +
    (canView
      ? `<button class="btn" id="cov-viewdocs">view 8 sample documents →</button>`
      : `<p class="hint">Switch the identity key to <b>url_h</b> to view sample document text.</p>`);
  if (canView) document.getElementById('cov-viewdocs').addEventListener('click', () => viewSampleDocs(include, exclude));
}

const SAMPLE_PAGE = 8;
async function viewSampleDocs(include, exclude, offset = 0) {
  const host = document.getElementById('cov-docs');
  host.innerHTML = '<div class="panel"><div class="empty-state" style="padding:30px"><span class="spin"></span> resolving extracted text in-region…</div></div>';
  const d = await fetch('/api/text/resolve', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ include, exclude, key: 'url_h', sample: SAMPLE_PAGE, offset }),
  }).then(r => r.json());

  if (d.needs_worker) {
    host.innerHTML = `<div class="panel">${workerWarningHtml(d)}</div>`;
    wireWorkerLaunch(host);
    return;
  }
  const avail = d.include || include;
  if (!d.docs || !d.docs.length) {
    // Past the end (or none recoverable): offer to step back if we paged forward.
    const back = offset > 0 ? `<div class="sample-nav"><button class="btn" id="cov-prevdocs">← prev ${SAMPLE_PAGE}</button></div>` : '';
    host.innerHTML = `<div class="panel"><div class="empty-state">No text resolved${offset > 0 ? ' at this offset' : ' (docs may lack recoverable URLs)'}.</div>${back}</div>`;
    if (offset > 0) document.getElementById('cov-prevdocs').addEventListener('click', () => viewSampleDocs(include, exclude, Math.max(0, offset - SAMPLE_PAGE)));
    return;
  }

  const from = offset + 1, to = offset + d.docs.length;
  const hasPrev = offset > 0;
  const hasNext = offset + SAMPLE_PAGE < d.count;
  let h = `<div class="panel"><div class="setdiff-title">Sample documents — how each dataset extracted them</div>
    <p class="hint">docs ${from}–${to} of ${d.count.toLocaleString()} in <b>${include.join(' ∩ ')}</b>${exclude.length ? ' \\ ' + exclude.join(' ∪ ') : ''}. Deterministic sample.</p>${workerWarningHtml(d)}</div>`;
  d.docs.forEach(doc => {
    const ds0 = Object.values(doc.extractions)[0] || {};
    h += `<div class="doc-card"><div class="doc-head"><span class="url">${escapeHtml(ds0.url_key || doc.url_h)}</span>`;
    if (ds0.snapshot) h += `<span>snapshot ${escapeHtml(ds0.snapshot)}</span>`;
    h += `</div>${docColsHtml(doc.extractions, avail)}</div>`;
  });
  h += `<div class="sample-nav">
      ${hasPrev ? `<button class="btn" id="cov-prevdocs">← prev ${SAMPLE_PAGE}</button>` : '<span></span>'}
      ${hasNext ? `<button class="btn" id="cov-nextdocs">next ${SAMPLE_PAGE} →</button>` : '<span></span>'}
    </div>`;
  host.innerHTML = h;
  if (hasPrev) document.getElementById('cov-prevdocs').addEventListener('click', () => viewSampleDocs(include, exclude, Math.max(0, offset - SAMPLE_PAGE)));
  if (hasNext) document.getElementById('cov-nextdocs').addEventListener('click', () => viewSampleDocs(include, exclude, offset + SAMPLE_PAGE));
  wireWorkerLaunch(host);
}

// ---------- SEARCH ----------
let searchInit = false;
const SEARCH_STAGES = ['expand', 'retrieve', 'enrich', 'rerank', 'crossref', 'resolve'];
const STAGE_LABEL = { expand: 'expand query', retrieve: 'BM25 retrieve', enrich: 'load snippets', rerank: 'rerank', crossref: 'cross-ref', resolve: 'load text', done: 'done', queued: 'queued' };
let searchShowFastpipe = false;

let datasetRegions = null;
async function ensureDatasetRegions() {
  if (!datasetRegions) {
    const d = await fetch('/api/coverage/datasets').then(r => r.json()).catch(() => []);
    datasetRegions = {}; d.forEach(x => datasetRegions[x.dataset] = x.region);
  }
  return datasetRegions;
}

function ensureSearch() {
  if (searchInit) return;
  searchInit = true;
  ensureDatasetRegions();
  document.getElementById('search-go').addEventListener('click', () => runSearch(false));
  document.getElementById('search-refresh').addEventListener('click', () => runSearch(true));
  document.getElementById('search-input').addEventListener('keydown', e => { if (e.key === 'Enter') runSearch(false); });
  // toggling example mode re-runs the current query fresh (so the switch takes effect immediately)
  document.getElementById('search-example-mode')?.addEventListener('change', () => {
    if (document.getElementById('search-input').value.trim()) runSearch(true);
  });
  document.getElementById('search-backport')?.addEventListener('click', backportSaved);
  loadSaved();
}

async function backportSaved() {
  const btn = document.getElementById('search-backport');
  const status = document.getElementById('search-backport-status');
  btn.disabled = true; btn.textContent = 'updating…';
  status.textContent = 'recomputing cross-reference across saved searches…';
  try {
    const r = await fetch('/api/search/backport', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ reembed: true }) }).then(r => r.json());
    status.textContent = `updated ${r.updated} saved search${r.updated === 1 ? '' : 'es'}. Re-open one to see the new datasets.`;
  } catch (e) {
    status.textContent = 'update failed: ' + e;
  } finally {
    btn.disabled = false; btn.textContent = '↻ update saved for new datasets';
  }
}

async function runSearch(force) {
  const q = document.getElementById('search-input').value.trim();
  if (!q) return;
  const exampleMode = !!document.getElementById('search-example-mode')?.checked;
  document.getElementById('search-results').innerHTML = '';
  renderSearchProgress('queued', 0);
  const start = await fetch('/api/search', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: q, force, example_mode: exampleMode }) }).then(r => r.json());
  if (start.cached) { document.getElementById('search-progress').innerHTML = ''; renderSearchResult(start.result, true); return; }
  if (start.error) { document.getElementById('search-progress').innerHTML = ''; document.getElementById('search-results').innerHTML = `<div class="empty-state">${escapeHtml(start.error)}</div>`; return; }
  const poll = async () => {
    const p = await fetch('/api/search/progress/' + start.job_id).then(r => r.json());
    renderSearchProgress(p.stage, p.elapsed);
    if (p.done) {
      document.getElementById('search-progress').innerHTML = '';
      if (p.error) document.getElementById('search-results').innerHTML = `<div class="empty-state"><div class="big">Search failed</div><div>${escapeHtml(p.error)}</div></div>`;
      else { renderSearchResult(p.result, false); loadSaved(); }
      return;
    }
    setTimeout(poll, 700);
  };
  poll();
}

function renderSearchProgress(stage, elapsed) {
  const idx = SEARCH_STAGES.indexOf(stage);
  const pct = stage === 'queued' ? 6 : stage === 'done' ? 100 : Math.round(((idx + 1) / (SEARCH_STAGES.length + 1)) * 100);
  let h = `<div class="progress-wrap"><div class="progress-meta"><span>searching — ${STAGE_LABEL[stage] || stage}…</span><span>${(elapsed || 0).toFixed ? elapsed : elapsed || 0}s</span></div>`;
  h += `<div class="progress-track"><div class="progress-fill" style="width:${pct}%"></div></div>`;
  h += '<div class="progress-stages">';
  SEARCH_STAGES.forEach((s, i) => { const cls = s === stage ? 'active' : (i < idx ? 'done' : ''); h += `<span class="st ${cls}">${STAGE_LABEL[s]}</span>`; });
  h += '</div></div>';
  document.getElementById('search-progress').innerHTML = h;
}

let lastSearchResult = null;
const showFp = ds => searchShowFastpipe || !FASTPIPE_METHOD_RE.test(ds);

function renderKeepSummary(summary) {
  const rows = summary.filter(s => showFp(s.dataset));
  if (!rows.length) return '';
  let h = '<div class="keep-summary"><div class="setdiff-title">How many of these docs each strategy keeps vs discards</div>';
  rows.forEach(s => {
    const tot = s.kept + s.dropped, pct = tot ? Math.round(s.kept / tot * 100) : 0;
    h += `<div class="ks-row"><span class="ks-name"><span class="bar" style="background:${colorFor(s.dataset)}"></span>${s.dataset}</span>
      <span class="ks-track"><span class="ks-fill" style="width:${pct}%;background:${colorFor(s.dataset)}"></span></span>
      <span class="ks-num"><b>${s.kept}</b> kept · <span class="ks-drop">${s.dropped} dropped</span></span></div>`;
  });
  return h + '</div>';
}

function renderSearchResult(r, cached) {
  lastSearchResult = r;
  const res = document.getElementById('search-results');
  let h = '';
  if (r.plan) {
    h += '<div class="search-plan">';
    // example-query chips only when the search actually ran in example mode
    if (r.example_mode && (r.plan.example_queries || []).length) {
      h += '<b>example queries:</b> ';
      r.plan.example_queries.forEach(q => h += `<span class="q qex" title="targets an actual example / primary source">${escapeHtml(q)}</span>`);
      h += '<br>';
    }
    h += '<b>expanded queries:</b> ';
    (r.plan.bm25_queries || []).forEach(q => h += `<span class="q">${escapeHtml(q)}</span>`);
    if (r.plan.rationale) h += `<div class="rat">${escapeHtml(r.plan.rationale)}</div>`;
    h += '</div>';
  }
  if (r.keep_summary) h += renderKeepSummary(r.keep_summary);
  const pending = (r.pending_datasets || []).length ? ` · warming: ${r.pending_datasets.join(', ')}` : '';
  h += `<div class="hint" style="display:flex;justify-content:space-between;align-items:center;gap:12px">
    <span>${(r.results || []).length} results · ${r.candidate_count || 0} candidates · ${r.elapsed || '?'}s${cached ? ' · cached' : ''}${pending}</span>
    <label style="cursor:pointer;white-space:nowrap"><input type="checkbox" id="sr-fp" ${searchShowFastpipe ? 'checked' : ''}> show fastpipe bands</label></div>`;
  if (!(r.results || []).length) h += `<div class="empty-state">No relevant results. Try rephrasing, or ↻ to re-run.</div>`;
  (r.results || []).forEach((c, i) => {
    const kept = (c.kept_by || []).filter(showFp), dropped = (c.dropped_by || []).filter(showFp);
    h += '<div class="sr-card"><div class="sr-head">';
    h += `<span class="sr-rank">#${i + 1}</span><span class="sr-url">${escapeHtml(c.url || c.url_h || '—')}</span>`;
    h += `<span class="sr-score">rel ${c.rerank_score ?? '?'} · bm25 ${(c.score || 0).toFixed(1)} · via ${c.dataset}</span></div>`;
    if (c.reason) h += `<div class="sr-reason">${escapeHtml(c.reason)}</div>`;
    const snip = c.snippet || c.preview;
    if (snip) h += `<div class="sr-preview">${escapeHtml(snip)}</div>`;
    h += '<div class="sr-cov"><span class="lbl">kept by</span>';
    kept.forEach(ds => h += `<span class="covchip kept"><span class="bar" style="background:${colorFor(ds)}"></span>${ds}</span>`);
    if (dropped.length) { h += '<span class="lbl">dropped by</span>'; dropped.forEach(ds => h += `<span class="covchip dropped">${ds}</span>`); }
    h += '</div>';
    if (kept.length && c.url_h) h += `<div class="sr-actions"><button class="btn" data-idx="${i}">${VIEW_EXTRACTIONS_LABEL}</button></div>`;
    h += '<div class="doc-slot"></div></div>';
  });
  res.innerHTML = h;
  document.getElementById('sr-fp')?.addEventListener('change', e => { searchShowFastpipe = e.target.checked; renderSearchResult(lastSearchResult, cached); });
  res.querySelectorAll('[data-idx]').forEach(b => b.addEventListener('click', () => viewSearchDoc(b)));
}

// Warning + inline launch button for datasets whose in-region worker isn't up.
// Notice for datasets whose text isn't shown. Two cases from the backend:
//   warming  — a worker IS up but that dataset is still mirroring → retry, no button
//   down     — no worker in the region → offer to launch one
function workerWarningHtml(d) {
  if (!d) return '';
  const warming = d.warming || [];
  const downRegions = d.missing_regions || d.needs_worker || [];
  const downDs = (d.unavailable || []).filter(x => !warming.includes(x));
  let h = '';
  if (warming.length)
    h += `<div class="worker-warn warming">⏳ Extractions for <b>${warming.join(', ')}</b> are still warming in-region (~a few min) — click again shortly.</div>`;
  if (downRegions.length) {
    const btns = downRegions.map(r => `<button class="btn" data-launch-region="${r}">launch ${r} worker</button>`).join(' ');
    const which = downDs.length ? downDs.join(', ') : downRegions.join(', ');
    h += `<div class="worker-warn">⚠ Extractions for <b>${which}</b> aren't shown — no in-region worker (${downRegions.join(', ')}) is running. ${btns} <span class="hint">Text stays in-region; the worker takes ~3 min to warm, then click again.</span></div>`;
  }
  return h;
}
function wireWorkerLaunch(container) {
  container.querySelectorAll('[data-launch-region]').forEach(b => b.addEventListener('click', async () => {
    b.disabled = true; b.textContent = 'launching…';
    await fetch('/api/worker/launch', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ region: b.dataset.launchRegion }) }).catch(() => { });
    b.textContent = 'launching — retry in ~3 min';
  }));
}
function docColsHtml(extractions, datasets) {
  const ds = datasets.filter(d => extractions[d]);
  if (!ds.length) return '';
  let h = `<div class="doc-cols" style="grid-template-columns:repeat(${ds.length},1fr);margin-top:10px">`;
  ds.forEach(dset => {
    const ex = extractions[dset];
    h += `<div class="doc-col"><h4><span class="bar" style="background:${colorFor(dset)}"></span>${dset}</h4>`;
    h += `<div class="meta">${ex.text_len.toLocaleString()} chars</div><pre>${escapeHtml(ex.text)}</pre></div>`;
  });
  return h + '</div>';
}

const VIEW_EXTRACTIONS_LABEL = 'view extractions →';
const COLLAPSE_EXTRACTIONS_LABEL = 'collapse extractions ▲';

function viewSearchDoc(btn) {
  const slot = btn.closest('.sr-card').querySelector('.doc-slot');
  if (slot.dataset.open === '1') {  // toggle closed
    slot.dataset.open = '0';
    slot.innerHTML = '';
    btn.innerHTML = VIEW_EXTRACTIONS_LABEL;
    return;
  }
  slot.dataset.open = '1';
  btn.innerHTML = COLLAPSE_EXTRACTIONS_LABEL;
  const c = lastSearchResult.results[num(btn.dataset.idx)];
  const kept = (c.kept_by || []).filter(showFp);
  const embedded = c.extractions || {};
  const have = kept.filter(ds => embedded[ds]);          // text pre-loaded during search — instant
  const missing = kept.filter(ds => !embedded[ds]);       // worker was down (e.g. fastpipe)
  if (have.length) {
    const regions = [...new Set(missing.map(ds => (datasetRegions || {})[ds]).filter(Boolean))];
    slot.innerHTML = docColsHtml(embedded, have) + workerWarningHtml({ warming: missing });
    wireWorkerLaunch(slot);
    return;
  }
  // Nothing pre-loaded (search ran with the worker down) — resolve live.
  slot.innerHTML = '<div class="empty-state" style="padding:16px"><span class="spin"></span> resolving in-region…</div>';
  fetch('/api/search/doc', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ url_h: c.url_h, datasets: kept }) }).then(r => r.json()).then(d => {
    const cols = docColsHtml(d.extractions || {}, kept);
    const warn = workerWarningHtml(d);
    slot.innerHTML = (cols || (warn ? '' : '<p class="hint">No text resolved.</p>')) + warn;
    wireWorkerLaunch(slot);
  });
}

async function loadSaved() {
  const items = await fetch('/api/search/saved').then(r => r.json()).catch(() => []);
  const host = document.getElementById('search-saved');
  if (!host) return;
  if (!items.length) { host.innerHTML = '<p class="hint">No saved searches yet. Every search is saved here; star to keep favorites at the top.</p>'; return; }
  host.innerHTML = items.map(it => `<div class="saved-item" data-q="${escapeHtml(it.query)}">
    <button class="star ${it.favorite ? 'on' : ''}" data-fav="${escapeHtml(it.query)}" title="favorite">${it.favorite ? '★' : '☆'}</button>
    <span class="q" title="${escapeHtml(it.query)}">${escapeHtml(it.query)}</span>
    <span class="meta">${it.result_count}·${it.elapsed ? it.elapsed + 's' : ''}</span>
    <button class="del" data-del="${escapeHtml(it.query)}" title="delete">✕</button></div>`).join('');
  host.querySelectorAll('.saved-item').forEach(el => el.addEventListener('click', e => { if (e.target.closest('.star') || e.target.closest('.del')) return; openSaved(el.dataset.q); }));
  host.querySelectorAll('[data-fav]').forEach(b => b.addEventListener('click', async e => { e.stopPropagation(); await fetch('/api/search/favorite', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: b.dataset.fav, favorite: !b.classList.contains('on') }) }); loadSaved(); }));
  host.querySelectorAll('[data-del]').forEach(b => b.addEventListener('click', async e => { e.stopPropagation(); await fetch('/api/search/delete', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: b.dataset.del }) }); loadSaved(); }));
}

async function openSaved(q) {
  document.getElementById('search-input').value = q;
  const r = await fetch('/api/search/get?query=' + encodeURIComponent(q)).then(r => r.json());
  if (r && !r.error) renderSearchResult(r, true);
}

// ---------- HEAD-TO-HEAD COMPARE ----------
let compareInit = false;
const cmp = { a: 'high_quality', b: 'dclm', dim: null, n: null };
let cmpCovMatrix = null;

async function ensureCompare() {
  if (!compareInit) {
    compareInit = true;
    cmpCovMatrix = await fetch('/api/coverage?key=url_h').then(r => r.json()).catch(() => null);
  }
  renderCmpControls();
  renderCompare();
}

function renderCmpControls() {
  const box = document.getElementById('cmp-controls');
  box.innerHTML = '';
  if (!state.summary) { box.innerHTML = notBuiltHint(); return; }
  const methods = state.summary.methods;
  if (!methods.includes(cmp.a)) cmp.a = methods[0];
  if (!methods.includes(cmp.b)) cmp.b = methods[1] || methods[0];
  if (cmp.dim == null) cmp.dim = state.summary.dims[Math.min(2, state.summary.dims.length - 1)];
  if (cmp.n == null) cmp.n = pickDefaultN(state.summary.n_warcs);
  box.appendChild(selectEl('cmp-a', 'method A', methods.map(m => ({ value: m, label: labelFor(m) })), cmp.a, v => { cmp.a = v; renderCompare(); }));
  box.appendChild(selectEl('cmp-b', 'method B', methods.map(m => ({ value: m, label: labelFor(m) })), cmp.b, v => { cmp.b = v; renderCompare(); }));
  box.appendChild(selectEl('cmp-dim', 'model size', state.summary.dims.map(d => ({ value: d, label: `d${d}` })), cmp.dim, v => { cmp.dim = num(v); renderCompare(); }));
  box.appendChild(selectEl('cmp-n', 'WARCs (N)', state.summary.n_warcs.map(n => ({ value: n, label: `N=${n}` })), cmp.n, v => { cmp.n = num(v); renderCompare(); }));
}

function cmpBestValue(method, family, task) {
  const rows = state.rows.filter(r => r.method === method && r.eval_family === family && r.task === task && r.dim === cmp.dim && r.n_warcs === cmp.n);
  if (!rows.length) return null;
  const higher = higherBetter(family);
  return rows.reduce((acc, r) => acc === null ? r.value : (higher ? Math.max(acc, r.value) : Math.min(acc, r.value)), null);
}

function renderCompare() {
  const host = document.getElementById('cmp-body');
  if (!state.summary) { host.innerHTML = ''; return; }
  const A = cmp.a, B = cmp.b, cA = colorFor(A), cB = colorFor(B);
  let h = `<div class="cmp-vs"><span class="a" style="color:${cA}">${labelFor(A)}</span><span class="vs">vs</span><span class="a" style="color:${cB}">${labelFor(B)}</span><span class="hint" style="margin-left:12px">at d${cmp.dim} · N=${cmp.n} · best over budgets</span></div>`;

  // Eval deltas + coverage side by side
  h += '<div class="cmp-grid">';
  // --- eval deltas ---
  h += '<div class="cmp-panel"><h3>Evaluation head-to-head</h3><table class="cmp-eval"><thead><tr><th class="task">eval / task</th><th>' + labelFor(A) + '</th><th>' + labelFor(B) + '</th><th>Δ</th></tr></thead><tbody>';
  const famOrder = ['core_v2', 'olmo_bpb', 'paloma', 'uncheatable', 'olmes'];
  let winsA = 0, winsB = 0;
  famOrder.filter(f => state.summary.families.includes(f)).forEach(fam => {
    const tasks = (state.summary.tasks[fam] || []).filter(t => ['macro', 'Core_v2'].includes(t) || (fam === 'paloma'));
    const showTasks = tasks.length ? tasks.slice(0, fam === 'paloma' ? 6 : 4) : (state.summary.tasks[fam] || []).slice(0, 3);
    let any = false;
    showTasks.forEach(t => {
      const va = cmpBestValue(A, fam, t), vb = cmpBestValue(B, fam, t);
      if (va == null && vb == null) return;
      if (!any) { h += `<tr><td class="fam" colspan="4">${FAMILY_LABEL[fam] || fam}</td></tr>`; any = true; }
      const higher = higherBetter(fam);
      let awin = false, bwin = false;
      if (va != null && vb != null) { if (va === vb) {} else if (higher ? va > vb : va < vb) awin = true; else bwin = true; }
      if (awin) winsA++; if (bwin) winsB++;
      const d = (va != null && vb != null) ? (va - vb) : null;
      h += `<tr><td class="task">${prettyTask(t)}</td><td class="${awin ? 'win' : ''}">${va != null ? fmtVal(va) : '—'}</td><td class="${bwin ? 'win' : ''}">${vb != null ? fmtVal(vb) : '—'}</td><td>${d != null ? (d > 0 ? '+' : '') + fmtVal(d) : '—'}</td></tr>`;
    });
  });
  h += `</tbody></table><p class="hint">${labelFor(A)} wins ${winsA} · ${labelFor(B)} wins ${winsB} (★ = better, direction-aware)</p></div>`;

  // --- coverage overlap ---
  h += '<div class="cmp-panel"><h3>Corpus overlap (url_h)</h3>';
  const pair = cmpCovMatrix && cmpCovMatrix.pairs ? cmpCovMatrix.pairs.find(p => p.a === A && p.b === B) : null;
  const pairBA = cmpCovMatrix && cmpCovMatrix.pairs ? cmpCovMatrix.pairs.find(p => p.a === B && p.b === A) : null;
  if (pair) {
    const stat = (label, v, win) => `<div class="cmp-cov-stat"><span>${label}</span><span class="v ${win ? 'win' : ''}">${v}</span></div>`;
    h += stat(`${A} docs`, (cmpCovMatrix.sizes[A] || 0).toLocaleString());
    h += stat(`${B} docs`, (cmpCovMatrix.sizes[B] || 0).toLocaleString());
    h += stat('shared', pair.intersection.toLocaleString());
    h += stat('Jaccard', pair.jaccard.toFixed(4));
    h += stat(`${A} only`, pair.a_only.toLocaleString());
    h += stat(`${B} only`, pair.b_only.toLocaleString());
    h += stat(`% of ${A} inside ${B}`, (pair.containment_a_in_b * 100).toFixed(1) + '%', pair.containment_a_in_b > (pairBA ? pairBA.containment_a_in_b : 0));
    if (pairBA) h += stat(`% of ${B} inside ${A}`, (pairBA.containment_a_in_b * 100).toFixed(1) + '%', pairBA.containment_a_in_b > pair.containment_a_in_b);
    h += `<div class="cmp-doc-btns">
      <button class="btn" data-inc="${A}" data-exc="${B}">docs in ${A} not ${B} →</button>
      <button class="btn" data-inc="${B}" data-exc="${A}">docs in ${B} not ${A} →</button>
      <button class="btn" data-inc="${A},${B}" data-exc="">docs in both →</button></div>`;
  } else {
    h += `<p class="hint">No coverage overlap available for this pair (one may lack a 300-WARC index, or the names differ from the index datasets).</p>`;
  }
  h += '</div></div>';

  // --- doc slot + spec diff ---
  h += '<div id="cmp-docs"></div>';
  h += '<div class="cmp-grid" id="cmp-specs"></div>';
  host.innerHTML = h;

  host.querySelectorAll('[data-inc]').forEach(btn => btn.addEventListener('click', () => cmpViewDocs(btn.dataset.inc.split(','), btn.dataset.exc ? btn.dataset.exc.split(',') : [])));
  renderCmpSpecs(A, B);
}

async function cmpViewDocs(include, exclude) {
  const host = document.getElementById('cmp-docs');
  host.innerHTML = '<div class="panel"><div class="empty-state" style="padding:24px"><span class="spin"></span> resolving in-region…</div></div>';
  const d = await fetch('/api/text/resolve', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ include, exclude, key: 'url_h', sample: 6 }) }).then(r => r.json());
  if (d.needs_worker) { host.innerHTML = `<div class="panel">${workerWarningHtml(d)}</div>`; wireWorkerLaunch(host); return; }
  const avail = d.include || include;
  if (!d.docs || !d.docs.length) { host.innerHTML = ''; return; }
  let h = `<div class="panel"><div class="setdiff-title">Sample docs — in ${include.join(' ∩ ')}${exclude.length ? ' \\ ' + exclude.join(' ∪ ') : ''}</div>${workerWarningHtml(d)}</div>`;
  d.docs.forEach(doc => {
    const ds0 = Object.values(doc.extractions)[0] || {};
    h += `<div class="doc-card"><div class="doc-head"><span class="url">${escapeHtml(ds0.url_key || doc.url_h)}</span></div>${docColsHtml(doc.extractions, avail)}</div>`;
  });
  host.innerHTML = h;
  wireWorkerLaunch(host);
}

async function renderCmpSpecs(A, B) {
  const host = document.getElementById('cmp-specs');
  const [sa, sb] = await Promise.all([
    fetch(`/api/spec/${encodeURIComponent(A)}`).then(r => r.json()).catch(() => null),
    fetch(`/api/spec/${encodeURIComponent(B)}`).then(r => r.json()).catch(() => null),
  ]);
  const panel = (m, s) => {
    if (!s || s.error) return `<div class="cmp-panel cmp-spec"><h3><span class="bar" style="background:${colorFor(m)}"></span>${m}</h3><p class="hint">No spec (external extractor or unknown).</p></div>`;
    let x = `<div class="cmp-panel cmp-spec"><h3><span class="bar" style="background:${colorFor(m)}"></span>${escapeHtml(s.title)}</h3>`;
    x += `<p class="desc" style="font-size:12.5px;color:var(--ink-dim)">${escapeHtml(s.description || '')}</p>`;
    if (s.phases && s.phases.length) x += `<pre class="prompt">${escapeHtml(s.phases[0].text.slice(0, 1400))}${s.phases[0].text.length > 1400 ? '\n… (full text in Specs tab)' : ''}</pre>`;
    return x + '</div>';
  };
  host.innerHTML = panel(A, sa) + panel(B, sb);
}

// ---------- helpers ----------
function higherBetter(fam) { return fam === 'core_v2' || fam === 'olmes'; }
function defaultTask(fam, tasks) {
  const pref = { core_v2: 'Core_v2', olmo_bpb: 'macro', uncheatable: 'macro', olmes: 'macro' }[fam];
  if (pref && tasks.includes(pref)) return pref;
  return tasks[0];
}
function pickDefaultN(ns) { return ns.includes(300) ? 300 : ns[0]; }
function hasBudget(fam, dim, n, b) {
  return state.rows.some(r => r.eval_family === fam && r.dim === dim && r.n_warcs === n && Math.abs(r.budget - b) < 1e-9);
}
function prettyTask(t) {
  if (!t) return t;
  return t.replace(/^task:/, '').replace(/_bpb$/, '').replace(/_/g, ' ');
}
function fmtVal(v) { return Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(4); }
function fmtTime(iso) { try { return new Date(iso).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }); } catch { return iso; } }
function notBuiltHint() {
  return `<div class="hint">No eval data cached. Click <b>refresh evals</b> (top right) to scan GCS — takes ~1–2 min.</div>`;
}
function emptyPlot(msg) { return `<div class="empty-state"><div class="big">Nothing here</div><div>${msg}</div></div>`; }
function escapeHtml(s) { return String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])); }

function darkLayout(title, xtitle, ytitle) {
  return {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'IBM Plex Mono, monospace', color: '#a49e8d', size: 11 },
    title: { text: title, font: { family: 'Bricolage Grotesque, sans-serif', color: '#ece7db', size: 16 }, x: 0.01, xanchor: 'left' },
    margin: { l: 66, r: 18, t: 54, b: 78 },
    xaxis: { title: { text: xtitle, font: { size: 11 } }, type: 'log', gridcolor: '#26251d', zerolinecolor: '#35342a', linecolor: '#35342a', tickfont: { size: 10 } },
    yaxis: { title: { text: ytitle, font: { size: 11 } }, gridcolor: '#26251d', zerolinecolor: '#35342a', linecolor: '#35342a', tickfont: { size: 10 } },
    legend: { orientation: 'h', y: -0.2, x: 0, font: { size: 10 }, bgcolor: 'rgba(0,0,0,0)' },
    hovermode: 'x unified',
    hoverlabel: { bgcolor: '#1c1b14', bordercolor: '#35342a', font: { family: 'IBM Plex Mono, monospace', size: 11, color: '#ece7db' } },
  };
}

// ---------- icons ----------
function iconPerf() { return svg('<path d="M3 3v18h18"/><path d="M7 15l4-5 3 3 5-7"/>'); }
function iconTrophy() { return svg('<path d="M6 4h12v3a6 6 0 0 1-12 0V4z"/><path d="M6 5H3v2a3 3 0 0 0 3 3M18 5h3v2a3 3 0 0 1-3 3M9 20h6M12 13v7"/>'); }
function iconDoc() { return svg('<path d="M6 2h8l4 4v16H6z"/><path d="M14 2v4h4M9 12h6M9 16h6"/>'); }
function iconGrid() { return svg('<rect x="3" y="3" width="18" height="18" rx="1"/><path d="M3 9h18M3 15h18M9 3v18M15 3v18"/>'); }
function iconSearch() { return svg('<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/>'); }
function iconCompare() { return svg('<path d="M12 3v18"/><path d="M5 8l-3 4 3 4M19 8l3 4-3 4"/>'); }
function svg(inner) { return `<svg viewBox="0 0 24 24">${inner}</svg>`; }

boot();
