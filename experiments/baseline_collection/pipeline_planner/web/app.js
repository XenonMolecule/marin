// Copyright The Marin Authors — SPDX-License-Identifier: Apache-2.0
// UI: load registry + per-doc matrix, render the linear cascade, re-evaluate live via Engine.
(() => {
  const DATA = "data/";
  let REG, STAGES_BY_ID, MATRIX, N, PIPE, SELECTED = null, IS_DEMO = false, LAST_R = null;
  const SHARD_SIZE = 100, SHARD_CACHE = {};

  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };

  // ---- formatting ----
  const fmtCount = (x) => {
    if (x >= 1e12) return (x / 1e12).toFixed(2) + "T";
    if (x >= 1e9) return (x / 1e9).toFixed(2) + "B";
    if (x >= 1e6) return (x / 1e6).toFixed(2) + "M";
    if (x >= 1e3) return (x / 1e3).toFixed(1) + "k";
    return Math.round(x).toString();
  };
  const fmtYears = (y) => {
    const d = y * 365.25;
    if (y >= 100) return Math.round(y) + " yr";
    if (y >= 1) return y.toFixed(1) + " yr";        // 1–100 years
    if (d >= 30) return (d / 30.44).toFixed(1) + " mo"; // 1 month … 1 year
    if (d >= 1) return d.toFixed(1) + " d";
    if (d * 24 >= 1) return (d * 24).toFixed(1) + " h";
    return Math.max(1, Math.round(d * 24 * 60)) + " min";
  };
  const pct = (x) => (100 * x).toFixed(1) + "%";

  // ---- demo data (used until the real matrix is downloaded into data/) ----
  function synthMatrix(reg) {
    const n = 2000; let s = 7; const rnd = () => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
    const cols = {}; const gold = new Array(n);
    for (let i = 0; i < n; i++) gold[i] = rnd() < 0.05 ? 1 : 0;
    cols.gold_useful = gold;
    for (const st of reg.stages) {
      if (st.kind !== "classifier") continue;
      const arr = new Array(n);
      for (let i = 0; i < n; i++) {
        if (st.direction === "low_useful") arr[i] = rnd() < 0.1 ? null : (gold[i] ? -4 + 2 * rnd() : -1 + 1.5 * rnd());
        else arr[i] = Math.max(0, Math.min(1, gold[i] ? 0.45 + 0.5 * rnd() : 0.3 * rnd()));
      }
      cols[st.score_col] = arr;
    }
    const lab = (p) => { const a = new Array(n); for (let i = 0; i < n; i++) a[i] = (gold[i] ? rnd() < 0.85 : rnd() < 0.06) ? 1 : 0; return a; };
    cols.label_1p7b_useful = lab(); cols.label_0p6b_useful = lab();
    for (const k of ["1p7b", "0p6b", "justext"]) {
      const lev = new Array(n), both = new Array(n);
      for (let i = 0; i < n; i++) { lev[i] = gold[i] ? 0.55 + 0.4 * rnd() : 0.2 + 0.4 * rnd(); both[i] = gold[i] && rnd() > 0.2 ? 1 : 0; }
      cols["lev_" + k] = lev; cols["both_" + k] = both;
    }
    for (const k of ["8b", "1p7b", "0p6b", "justext"]) { const t = new Array(n); for (let i = 0; i < n; i++) t[i] = gold[i] ? (200 + 800 * rnd()) | 0 : (40 + 200 * rnd()) | 0; cols["tok_" + k] = t; }
    cols.warc_record_id = Array.from({ length: n }, (_, i) => "demo-" + i);
    return { meta: { n, registry: reg }, columns: cols };
  }

  function defaultThreshold(st) { return st.direction === "low_useful" ? -2.0 : 0.5; }

  function defaultPipeline() {
    return {
      filters: [
        { stageId: "fasttext_w80", enabled: true, mode: "threshold", threshold: 0.5, recall: 0.99 },
        { stageId: "bert_200k_8192", enabled: true, mode: "threshold", threshold: 0.5, recall: 0.97 },
      ],
      extractor: { stageId: "extract_1p7b" },
      capacity: { nChips: 300, nCores: 5000, docsPerWarc: 40000, nWarcs: 7925398, tokEfficiency: REG.default_tokenization_efficiency },
    };
  }

  // ---- rendering ----
  function render() {
    const r = Engine.evaluate(MATRIX.columns, N, PIPE, STAGES_BY_ID);
    LAST_R = r;
    renderFlow(r); renderMetrics(r); renderCompute(r); renderCapacity(); renderConfig();
    syncHash();
  }

  // Debounced recompute for live-typed inputs: refresh everything EXCEPT the stage-config panel, so the
  // input the user is mid-edit in keeps focus + caret. The heavy Engine.evaluate only fires DEBOUNCE_MS
  // after the last keystroke/drag, so it never freezes on every digit.
  const DEBOUNCE_MS = 2000;
  let _renderTimer = null;
  function recomputeKeepConfig() {
    const r = Engine.evaluate(MATRIX.columns, N, PIPE, STAGES_BY_ID);
    LAST_R = r;
    renderFlow(r); renderMetrics(r); renderCompute(r); renderCapacity();
    syncHash();
  }
  function debouncedRender() {
    clearTimeout(_renderTimer);
    _renderTimer = setTimeout(recomputeKeepConfig, DEBOUNCE_MS);
  }

  function renderFlow(r) {
    const flow = $("flow"); flow.innerHTML = "";
    const TOTAL = r.totalDocs;
    // input node
    flow.appendChild(connector(TOTAL, TOTAL, "raw input"));
    PIPE.filters.forEach((f, i) => {
      const st = STAGES_BY_ID[f.stageId], sr = r.stages[i];
      flow.appendChild(card(f, st, sr, i));
      const outProj = sr.docsOut * r.scale;
      flow.appendChild(connector(outProj, TOTAL, sr.skipped ? "(disabled)" : null));
    });
    flow.appendChild(extractorCard(r));
  }

  function connector(countProj, total, note) {
    const c = el("div", "connector");
    c.appendChild(el("div", "count", fmtCount(countProj)));
    const bar = el("div", "bar"); bar.style.width = Math.max(3, 64 * (countProj / total)) + "px"; c.appendChild(bar);
    c.appendChild(el("div", "pct", note || pct(countProj / total)));
    return c;
  }

  function card(f, st, sr, idx) {
    const c = el("div", "stage-card" + (f.enabled ? "" : " disabled") + (SELECTED === "f" + idx ? " selected" : ""));
    c.onclick = () => { SELECTED = "f" + idx; render(); };
    const title = el("div", "title");
    title.appendChild(el("span", null, st.label));
    title.appendChild(el("span", "badge " + st.device, st.device));
    c.appendChild(title);
    const thr = f.mode === "recall" ? "recall " + f.recall : "thr " + (sr.threshold == null ? "—" : (+sr.threshold).toFixed(2));
    c.appendChild(el("div", "muted", thr + (st.throughput_assumed ? " ⚠" : "")));
    const ctr = el("div", "ctrls");
    const mk = (label, fn) => { const b = el("button", null, label); b.onclick = (e) => { e.stopPropagation(); fn(); }; return b; };
    ctr.appendChild(mk(f.enabled ? "⊙ on" : "○ off", () => { f.enabled = !f.enabled; render(); }));
    ctr.appendChild(mk("◀", () => { if (idx > 0) { [PIPE.filters[idx - 1], PIPE.filters[idx]] = [PIPE.filters[idx], PIPE.filters[idx - 1]]; render(); } }));
    ctr.appendChild(mk("▶", () => { if (idx < PIPE.filters.length - 1) { [PIPE.filters[idx + 1], PIPE.filters[idx]] = [PIPE.filters[idx], PIPE.filters[idx + 1]]; render(); } }));
    const x = mk("✕", () => { PIPE.filters.splice(idx, 1); SELECTED = null; render(); }); x.style.color = "var(--red)";
    ctr.appendChild(x);
    c.appendChild(ctr);
    return c;
  }

  function extractorCard(r) {
    const ext = STAGES_BY_ID[PIPE.extractor.stageId];
    const c = el("div", "stage-card extractor" + (SELECTED === "ext" ? " selected" : ""));
    c.onclick = () => { SELECTED = "ext"; render(); };
    const title = el("div", "title");
    title.appendChild(el("span", null, "▶ " + ext.label));
    title.appendChild(el("span", "badge " + ext.device, ext.device));
    c.appendChild(title);
    c.appendChild(el("div", "muted", "kept: " + fmtCount(r.extractor.kept * r.scale) + (ext.throughput_assumed ? " ⚠" : "")));
    return c;
  }

  function renderMetrics(r) {
    const m = r.summary;
    // Levenshtein over docs where BOTH the 8B and this extractor produced text (real extraction-quality
    // signal). The union value (which scores classification false-positives as 0 against the 8B's empty
    // text) is shown in the tooltip for transparency, not as the headline.
    const levTip = m.levBoth == null ? "" :
      `vs 8B over the ${fmtCount(m.levBothN * r.scale)} docs (~${fmtCount(m.levBothN)} sampled) both this extractor and the 8B actually extracted. ` +
      `Union (counting classification FPs as 0): ${m.levUnion == null ? "—" : m.levUnion.toFixed(3)} over ${fmtCount(m.levUnionN)} sampled.`;
    const items = [
      ["F1 vs 8B", m.f1.toFixed(3), m.f1 < 0.5 ? "bad" : m.f1 < 0.62 ? "warn" : "", ""],
      ["Precision", m.precision.toFixed(3), "", ""],
      ["Recall", m.recall.toFixed(3), "", ""],
      ["Levenshtein (both extr.)", m.levBoth == null ? "—" : m.levBoth.toFixed(3), "", levTip],
      ["Projected tokens", fmtCount(m.projTokens), "", ""],
      ["TPU time", fmtYears(m.wallTpuYears), m.wallTpuYears > 5 ? "bad" : m.wallTpuYears > 1 ? "warn" : "", ""],
      ["CPU time", fmtYears(m.wallCpuYears), m.wallCpuYears > 5 ? "bad" : m.wallCpuYears > 1 ? "warn" : "", ""],
    ];
    const c = $("metrics"); c.innerHTML = "";
    for (const [k, v, cls, tip] of items) {
      const d = el("div", "metric");
      if (tip) { d.title = tip; d.style.cursor = "help"; }
      d.appendChild(el("div", "v " + cls, v));
      d.appendChild(el("div", "k", k));
      c.appendChild(d);
    }
  }

  function renderCompute(r) {
    const tb = $("compute-table").querySelector("tbody"); tb.innerHTML = "";
    const cap = PIPE.capacity, SPY = 31556952;
    const head = el("tr");
    head.innerHTML = "<th>stage</th><th>dev</th><th class='num'>time on cluster</th><th class='num'>share of its dev</th>";
    tb.appendChild(head);
    for (const row of r.computeBreakdown) {
      const div = row.device === "tpu" ? cap.nChips : cap.nCores;
      const wall = row.unitSec / div / SPY; // wall-clock this stage adds, given the capacity sliders
      const tr = el("tr", "barcell"); tr.style.setProperty("--p", row.pct.toFixed(0) + "%");
      tr.innerHTML = `<td>${row.label}</td><td><span class='badge ${row.device}'>${row.device}</span></td>` +
        `<td class='num'>${fmtYears(wall)}</td><td class='num'>${row.pct.toFixed(0)}%</td>`;
      tb.appendChild(tr);
    }
    const s = r.summary, bound = s.wallTpuYears >= s.wallCpuYears ? "TPU" : "CPU";
    const foot = el("tr");
    foot.innerHTML = `<td colspan="4" class="muted" style="padding-top:10px;border-top:2px solid var(--line);line-height:1.7">` +
      `<b style="color:var(--tpu)">TPU</b> total ${fmtYears(s.wallTpuYears)} on ${cap.nChips} chips &nbsp;·&nbsp; ` +
      `<b style="color:var(--cpu)">CPU</b> total ${fmtYears(s.wallCpuYears)} on ${cap.nCores} cores<br>` +
      `→ pipeline ≈ <b style="color:var(--fg)">${fmtYears(Math.max(s.wallTpuYears, s.wallCpuYears))}</b> ` +
      `(<b>${bound}-bound</b>; TPU &amp; CPU run in parallel, so the slower device sets the wall-clock)</td>`;
    tb.appendChild(foot);
  }

  function renderCapacity() {
    const c = $("capacity"); c.innerHTML = "";
    const cap = PIPE.capacity;
    const mk = (key, label, min, max, step) => {
      const wrap = el("label", null, `<span>${label}</span>`);
      const inp = el("input"); inp.type = "number"; inp.min = min; inp.max = max; inp.step = step; inp.value = cap[key];
      inp.oninput = () => { cap[key] = +inp.value; render(); };
      wrap.appendChild(inp); c.appendChild(wrap);
    };
    mk("nChips", "TPU chips (≤1206)", 100, 1206, 10);
    mk("nCores", "CPU cores", 1000, 10000, 100);
    mk("docsPerWarc", "docs / WARC", 1, 200000, 1000);
    mk("nWarcs", "# WARCs", 1, 7925398, 1000);
    mk("tokEfficiency", "BERT tok-efficiency", 0.1, 1, 0.05);
  }

  function renderConfig() {
    const box = $("stage-config"); box.innerHTML = "<h2>Stage config</h2>";
    if (SELECTED === "ext") {
      const sel = el("select");
      REG.stages.filter((s) => s.kind === "extractor").forEach((s) => { const o = el("option", null, s.label); o.value = s.id; if (s.id === PIPE.extractor.stageId) o.selected = true; sel.appendChild(o); });
      sel.onchange = () => { PIPE.extractor.stageId = sel.value; render(); };
      const g = el("div", "cfg-grid"); g.appendChild(el("label", null, "extractor")); g.appendChild(sel); box.appendChild(g);
      const st = STAGES_BY_ID[PIPE.extractor.stageId];
      box.appendChild(el("div", "muted", `${st.device.toUpperCase()} · ${st.throughput} ${st.device === "tpu" ? "docs/chip/s" : "docs/s/core"}` + (st.throughput_assumed ? " ⚠ provisional" : "")));
      box.appendChild(el("div", "muted", st.throughput_source || ""));
      return;
    }
    if (SELECTED == null || !SELECTED.startsWith("f")) { box.appendChild(el("div", "muted", "Select a stage to configure.")); return; }
    const idx = +SELECTED.slice(1), f = PIPE.filters[idx]; if (!f) return;
    const st = STAGES_BY_ID[f.stageId];
    box.appendChild(el("div", null, `<b>${st.label}</b> <span class='badge ${st.device}'>${st.device}</span>`));
    const g = el("div", "cfg-grid");
    // variant: swap among same-family stages (e.g. ModernBERT context 1k/2k/4k/8k + base/large/1M; fastText widths; LLM ctx)
    const fam = REG.stages.filter((s) => s.family === st.family && s.kind === "classifier");
    if (fam.length > 1) {
      g.appendChild(el("label", null, "model / ctx"));
      const vsel = el("select");
      fam.forEach((s) => { const o = el("option", null, s.label); o.value = s.id; if (s.id === f.stageId) o.selected = true; vsel.appendChild(o); });
      vsel.onchange = () => { f.stageId = vsel.value; f.threshold = defaultThreshold(STAGES_BY_ID[vsel.value]); render(); };
      g.appendChild(vsel);
    }
    // mode
    g.appendChild(el("label", null, "mode"));
    const modeSel = el("select"); ["threshold", "recall"].forEach((mo) => { const o = el("option", null, mo); o.value = mo; if (f.mode === mo) o.selected = true; modeSel.appendChild(o); });
    modeSel.onchange = () => { f.mode = modeSel.value; render(); }; g.appendChild(modeSel);
    if (f.mode === "threshold") {
      g.appendChild(el("label", null, "threshold"));
      const wrap = el("div", "thr-wrap");
      const lo = st.direction === "low_useful" ? -8 : 0, hi = st.direction === "low_useful" ? 2 : 1;
      const rng = el("input"); rng.type = "range";
      rng.min = lo; rng.max = hi; rng.step = (hi - lo) / 200; rng.value = f.threshold;
      const num = el("input"); num.type = "number"; num.className = "thr-num"; num.step = 0.01; num.value = (+f.threshold).toFixed(3);
      // Slider + number stay in sync; recompute is debounced while editing, immediate on release/blur/Enter.
      const apply = (v, immediate) => {
        if (v === "" || isNaN(+v)) return; // let the field be empty mid-type without recomputing
        f.threshold = +v; rng.value = f.threshold;
        immediate ? render() : debouncedRender();
      };
      rng.oninput = () => { num.value = (+rng.value).toFixed(3); apply(rng.value, false); };
      rng.onchange = () => apply(rng.value, true);
      num.oninput = () => apply(num.value, false);
      num.onchange = () => apply(num.value, true);
      wrap.appendChild(rng); wrap.appendChild(num); g.appendChild(wrap);
    } else {
      g.appendChild(el("label", null, "target recall"));
      const inp = el("input"); inp.type = "number"; inp.min = 0.5; inp.max = 1; inp.step = 0.005; inp.value = f.recall;
      const applyR = (immediate) => { if (inp.value === "" || isNaN(+inp.value)) return; f.recall = +inp.value; immediate ? render() : debouncedRender(); };
      inp.oninput = () => applyR(false);
      inp.onchange = () => applyR(true);
      g.appendChild(inp);
    }
    box.appendChild(g);
    box.appendChild(el("div", "muted", `${st.throughput} ${st.device === "tpu" ? "docs/chip/s" : "docs/s/core"}` + (st.throughput_assumed ? " ⚠ assumed" : "") + ` · ${st.direction}`));
    box.appendChild(el("div", "muted", st.throughput_source || ""));
    const bb = el("button", null, "⊙ show borderline docs"); bb.style.marginTop = "8px";
    bb.onclick = () => showBorderline(idx); box.appendChild(bb);
  }

  const FAM_LABEL = { fasttext: "fastText", modernbert: "ModernBERT", llm: "LLM logprob" };
  function renderPalette() {
    const p = $("palette"); p.innerHTML = "<span class='muted'>add filter:</span>";
    const sel = el("select");
    sel.appendChild(el("option", null, "＋ choose a filter to add…"));
    const fams = {};
    REG.stages.filter((s) => s.kind === "classifier").forEach((s) => { (fams[s.family] = fams[s.family] || []).push(s); });
    for (const fam of Object.keys(fams)) {
      const og = document.createElement("optgroup"); og.label = FAM_LABEL[fam] || fam;
      fams[fam].forEach((s) => { const o = el("option", null, s.label); o.value = s.id; og.appendChild(o); });
      sel.appendChild(og);
    }
    sel.onchange = () => {
      const s = STAGES_BY_ID[sel.value]; if (!s) return;
      PIPE.filters.push({ stageId: s.id, enabled: true, mode: "threshold", threshold: defaultThreshold(s), recall: 0.97 });
      SELECTED = "f" + (PIPE.filters.length - 1); render();
    };
    p.appendChild(sel);
  }

  // ---- config sharing: URL hash (base64 JSON) + localStorage library + export ----
  const encodeCfg = () => btoa(unescape(encodeURIComponent(JSON.stringify(PIPE))));
  const decodeCfg = (s) => JSON.parse(decodeURIComponent(escape(atob(s))));
  function syncHash() { try { history.replaceState(null, "", "#cfg=" + encodeCfg()); } catch (e) {} }
  function applyHash() {
    const m = location.hash.match(/cfg=([^&]+)/);
    if (m) { try { const p = decodeCfg(m[1]); if (p && p.filters && p.extractor && p.capacity) PIPE = p; } catch (e) {} }
  }
  const LS_KEY = "cascade_planner_configs";
  const savedConfigs = () => { try { return JSON.parse(localStorage.getItem(LS_KEY) || "{}"); } catch (e) { return {}; } };
  function flash(msg) { const f = $("flash"); if (f) { f.textContent = "  " + msg; setTimeout(() => { if ($("flash")) $("flash").textContent = ""; }, 1500); } }
  function renderToolbar() {
    const t = $("toolbar"); t.innerHTML = "";
    const mk = (label, fn) => { const b = el("button", null, label); b.onclick = fn; return b; };
    t.appendChild(mk("🔗 copy link", () => { syncHash(); navigator.clipboard && navigator.clipboard.writeText(location.href); flash("link copied"); }));
    t.appendChild(mk("💾 save", () => { const name = prompt("config name:"); if (!name) return; const c = savedConfigs(); c[name] = PIPE; localStorage.setItem(LS_KEY, JSON.stringify(c)); renderToolbar(); flash("saved"); }));
    const cfgs = savedConfigs();
    if (Object.keys(cfgs).length) {
      const sel = el("select"); sel.appendChild(el("option", null, "load saved…"));
      Object.keys(cfgs).forEach((k) => { const o = el("option", null, k); o.value = k; sel.appendChild(o); });
      sel.onchange = () => { if (cfgs[sel.value]) { PIPE = JSON.parse(JSON.stringify(cfgs[sel.value])); SELECTED = null; render(); } };
      t.appendChild(sel);
    }
    t.appendChild(mk("🎲 random survivors", showRandomSample));
    t.appendChild(mk("🎲 dropped good docs", showRandomLosses));
    t.appendChild(mk("⬇ export", () => { const b = new Blob([JSON.stringify(PIPE, null, 2)], { type: "application/json" }); const a = document.createElement("a"); a.href = URL.createObjectURL(b); a.download = "pipeline_config.json"; a.click(); }));
    t.appendChild(el("span", "muted", "")).id = "flash";
  }

  // ---- borderline inspector: candidates drawn from the stage's REACHING set (conditioned on upstream filters) ----
  const escapeHtml = (s) => (s || "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  // Word-level LCS diff: <del> = in 8B (reference) only, <ins> = in the chosen extractor only.
  function wordDiff(aStr, bStr) {
    const tok = (s) => (s || "").split(/(\s+)/).filter((x) => x.length);
    let a = tok(aStr), b = tok(bStr);
    const CAP = 4000; // bounds the O(n*m) DP (~32MB at the cap) so the full-doc diff stays responsive
    let note = "";
    if (a.length > CAP || b.length > CAP) { note = `<div class="muted">(diff covers the first ${CAP.toLocaleString()} words — doc is longer)</div>`; a = a.slice(0, CAP); b = b.slice(0, CAP); }
    if (!a.length && !b.length) return "<i>(no text on either side)</i>";
    const n = a.length, m = b.length, dp = [];
    for (let i = 0; i <= n; i++) dp.push(new Uint16Array(m + 1));
    for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--)
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    let i = 0, j = 0; const out = [];
    while (i < n && j < m) {
      if (a[i] === b[j]) { out.push(escapeHtml(a[i])); i++; j++; }
      else if (dp[i + 1][j] >= dp[i][j + 1]) { out.push("<del>" + escapeHtml(a[i++]) + "</del>"); }
      else { out.push("<ins>" + escapeHtml(b[j++]) + "</ins>"); }
    }
    while (i < n) out.push("<del>" + escapeHtml(a[i++]) + "</del>");
    while (j < m) out.push("<ins>" + escapeHtml(b[j++]) + "</ins>");
    return note + out.join("");
  }
  async function fetchDoc(pos, id) {
    const shard = Math.floor(pos / SHARD_SIZE);
    if (!SHARD_CACHE[shard]) SHARD_CACHE[shard] = (async () => {
      try {
        const resp = await fetch(DATA + "docs/shard-" + String(shard).padStart(4, "0") + ".json.gz");
        if (!resp.ok) throw 0;
        const buf = await resp.arrayBuffer();
        // If the server sent Content-Encoding: gzip, the browser already decompressed → buf is JSON text.
        try { return JSON.parse(new TextDecoder().decode(buf)); }
        catch (e) {
          // Served as raw gzip bytes (e.g. some static hosts) → decompress in the browser.
          const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
          return JSON.parse(await new Response(stream).text());
        }
      } catch (e) { return null; }
    })();
    const sh = await SHARD_CACHE[shard]; return sh ? sh[id] : null;
  }
  // Best available EXTRACTED text (prefer 8B gold → 1.7B → 0.6B → jusText; raw HTML only as last resort).
  const EXT_ORDER = [["8B", "text_8b"], ["1.7B", "text_1p7b"], ["0.6B", "text_0p6b"], ["jusText", "text_justext"]];
  function bestExtraction(doc) {
    for (const [lbl, key] of EXT_ORDER) if (doc[key] && doc[key].trim()) return { label: lbl, text: doc[key] };
    return { label: "raw HTML — no model extracted it", text: doc.stripped_html || "" };
  }
  const CAT = { TP: ["TP", "true keep", false], FP: ["FP", "leak — kept junk", true],
                FN: ["FN", "LOST — dropped good", true], TN: ["TN", "true drop", false] };
  function category(kept, gold) { return kept ? (gold ? "TP" : "FP") : (gold ? "FN" : "TN"); }
  function borderDocCard(box, pos, kept, score) {
    const id = MATRIX.columns.warc_record_id[pos], gold = MATRIX.columns.gold_useful[pos] === 1;
    const [tag, desc, mistake] = CAT[category(kept, gold)];
    const col = mistake ? "var(--red)" : "var(--green)";
    const d = el("div", "panel"); d.style.margin = "6px 0"; d.style.borderLeft = "3px solid " + col;
    d.innerHTML = `<span class="badge" style="background:${mistake ? "rgba(240,106,106,.18)" : "rgba(78,204,163,.18)"};color:${col};border:1px solid ${col}">${tag} · ${desc}</span> ` +
      `<span class="muted">score ${score == null ? "null" : (+score).toFixed(3)} · ${id}</span><div class="muted" id="bdoc-${pos}">loading…</div>`;
    box.appendChild(d);
    fetchDoc(pos, id).then((doc) => {
      const t = $("bdoc-" + pos); if (!t) return;
      if (!doc) { t.textContent = IS_DEMO ? "(demo — no doc text)" : "(doc text unavailable)"; return; }
      const ex = bestExtraction(doc);
      t.innerHTML = `<a href="${escapeHtml(doc.url) || "#"}" target="_blank">${escapeHtml(doc.url)}</a> <span class="muted">· extracted (${ex.label})</span>` +
        `<div style="max-height:150px;overflow:auto;white-space:pre-wrap;color:var(--fg);margin-top:4px">${escapeHtml(ex.text.slice(0, 1200)) || "<i>(empty extraction)</i>"}</div>`;
    });
  }
  function showBorderline(idx) {
    $("borderline-panel").classList.remove("collapsed");
    const r = LAST_R, f = PIPE.filters[idx], st = STAGES_BY_ID[f.stageId], sr = r.stages[idx];
    if (sr.skipped) { $("borderline").innerHTML = "<span class='muted'>stage is disabled — enable it to inspect.</span>"; return; }
    const scores = MATRIX.columns[st.score_col], gcol = MATRIX.columns.gold_useful, sc = r.scale;
    const T = sr.threshold, vT = st.direction === "low_useful" ? -T : T;
    let TP = 0, FP = 0, FN = 0, TN = 0;
    const bubbleK = [], bubbleD = [], fnL = [], fpL = [];
    for (let i = 0; i < N; i++) {
      if (!sr.reachMask[i]) continue;
      const dd = Engine.vOf(scores[i], st.direction) - vT, kept = dd >= 0, gold = gcol[i] === 1;
      if (kept) { gold ? TP++ : (FP++, fpL.push(i)); bubbleK.push([dd, i]); }
      else { gold ? (FN++, fnL.push(i)) : TN++; bubbleD.push([-dd, i]); }
    }
    bubbleK.sort((a, b) => a[0] - b[0]); bubbleD.sort((a, b) => a[0] - b[0]);
    const prec = TP + FP ? TP / (TP + FP) : 0, rec = TP + FN ? TP / (TP + FN) : 0;
    const cell = (tag, v, mistake) => `<td style="color:${mistake ? "var(--red)" : "var(--green)"}">${tag} <b>${v.toLocaleString()}</b> <span class="muted">~${fmtCount(v * sc)}</span></td>`;
    const box = $("borderline");
    box.innerHTML = `<h3>${st.label} — threshold ${T == null || !isFinite(T) ? "—" : (+T).toFixed(3)} · ${sr.docsIn.toLocaleString()} docs reaching this stage</h3>` +
      `<table style="max-width:560px;margin:6px 0"><tr><th></th><th>8B useful (+)</th><th>8B not useful (−)</th></tr>` +
      `<tr><td class="muted">kept</td>${cell("TP", TP, false)}${cell("FP", FP, true)}</tr>` +
      `<tr><td class="muted">dropped</td>${cell("FN", FN, true)}${cell("TN", TN, false)}</tr></table>` +
      `<div class="muted" style="margin-bottom:6px">this stage in isolation: precision <b>${prec.toFixed(3)}</b> · recall <b>${rec.toFixed(3)}</b> — FN = good docs it loses, FP = junk it lets through</div>`;
    box.appendChild(el("div", "muted", "Marginal docs (nearest the threshold):"));
    bubbleK.slice(0, 3).forEach(([, pos]) => borderDocCard(box, pos, true, scores[pos]));
    bubbleD.slice(0, 3).forEach(([, pos]) => borderDocCard(box, pos, false, scores[pos]));
    const sampleNear = (list, kept, label) => {
      if (!list.length) return;
      box.appendChild(el("div", "muted", `${label} (showing ${Math.min(4, list.length)} of ${list.length}, closest to threshold):`));
      list.map((i) => [Math.abs(Engine.vOf(scores[i], st.direction) - vT), i]).sort((a, b) => a[0] - b[0])
        .slice(0, 4).forEach(([, pos]) => borderDocCard(box, pos, kept, scores[pos]));
    };
    sampleNear(fnL, false, "❌ FN — good docs LOST here (8B keeps, this filter drops)");
    sampleNear(fpL, true, "⚠️ FP — junk LEAKING through (8B drops, this filter keeps)");
  }

  // For a doc that survived, show — per enabled filter — the threshold move that WOULD have dropped it.
  // (It passed every filter, so each is on the keep side; tightening any one past its score drops it.)
  function catchItHtml(pos) {
    const rows = PIPE.filters.map((f, i) => {
      if (!f.enabled) return null;
      const st = STAGES_BY_ID[f.stageId], score = MATRIX.columns[st.score_col][pos], T = LAST_R.stages[i].threshold;
      const now = T == null || !isFinite(T) ? "—" : (+T).toFixed(3);
      let how;
      if (score == null) how = `<span class="muted">can't drop here (null logprob = model very confident useful)</span>`;
      else if (st.direction === "low_useful") how = `lower threshold below <b>${(+score).toFixed(3)}</b> <span class="muted">(now ${now})</span>`;
      else how = `raise threshold above <b>${(+score).toFixed(3)}</b> <span class="muted">(now ${now})</span>`;
      return `<div class="muted" style="margin-left:6px">· <b>${st.label}</b>: ${how}</div>`;
    }).filter(Boolean);
    if (!rows.length) return "";
    return `<div style="margin:4px 0;border-top:1px dashed var(--line);padding-top:4px"><span style="color:var(--amber)">this leak would be dropped by tightening any one filter:</span>${rows.join("")}</div>`;
  }

  // ---- view random survivors of the FULL pipeline (what actually makes it all the way through) ----
  function showRandomSample() {
    const r = LAST_R; if (!r || !r.finalKept) return;
    $("survivors-panel").classList.remove("collapsed");
    const ids = MATRIX.columns.warc_record_id;
    const key = "text_" + r.extractor.id.replace("extract_", "");
    const kept = [];
    for (let i = 0; i < N; i++) if (r.finalKept[i]) kept.push(i);
    const box = $("survivors");
    if (!kept.length) { box.innerHTML = "<span class='muted'>nothing survives this pipeline — loosen a filter.</span>"; return; }
    const pool = kept.slice(), pick = [];
    for (let k = 0; k < Math.min(6, pool.length); k++) pick.push(pool.splice((Math.random() * pool.length) | 0, 1)[0]);
    const gcol = MATRIX.columns.gold_useful;
    const nGold = pick.reduce((a, p) => a + (gcol[p] === 1 ? 1 : 0), 0);
    box.innerHTML = `<h3>🎲 ${pick.length} random survivors — <span style="color:var(--green)">${nGold} the 8B keeps</span> · <span style="color:var(--red)">${pick.length - nGold} the 8B would drop (leaks)</span> · ${r.summary.keptSample.toLocaleString()}/${N.toLocaleString()} kept (~${fmtCount(r.summary.keptSample * r.scale)} proj) · text: ${r.extractor.label}</h3>` +
      `<div class="muted" style="margin-bottom:6px">diff legend: <del>red</del> = only in 8B (your extractor dropped it) · <ins>green</ins> = only in your extractor (added/changed)</div>`;
    for (const pos of pick) {
      const id = ids[pos], gold = gcol[pos] === 1;
      const badge = gold
        ? `<span class="badge" style="background:rgba(78,204,163,.18);color:var(--green);border:1px solid var(--green)">8B keeps ✓</span>`
        : `<span class="badge" style="background:rgba(240,106,106,.18);color:var(--red);border:1px solid var(--red)">8B would drop ✗</span>`;
      const d = el("div", "panel"); d.style.margin = "6px 0"; d.style.borderLeft = "3px solid " + (gold ? "var(--green)" : "var(--red)");
      d.innerHTML = `${badge} <span class="muted">${id}</span>` + (gold ? "" : catchItHtml(pos)) + `<div class="muted" id="surv-${pos}">loading doc…</div>`;
      box.appendChild(d);
      fetchDoc(pos, id).then((doc) => {
        const t = $("surv-" + pos); if (!t) return;
        if (!doc) { t.textContent = IS_DEMO ? "(demo — no doc text)" : "(doc text unavailable)"; return; }
        const extKey = r.extractor.id.replace("extract_", "");
        const extTxt = doc["text_" + extKey] || "", goldTxt = doc.text_8b || "";
        const fullTxt = extTxt || doc.stripped_html || "";
        const isGold = r.extractor.id === "extract_8b";
        const lev = (MATRIX.columns["lev_" + extKey] || [])[pos];
        t.innerHTML = `<a href="${escapeHtml(doc.url) || "#"}" target="_blank">${escapeHtml(doc.url)}</a>` +
          (isGold ? ` <span class="muted">(this IS the 8B extraction)</span>`
                  : ` · <span class="muted">Lev vs 8B ${lev == null ? "—" : (+lev).toFixed(2)}</span> <button class="difftog">⇄ diff vs 8B</button>`) +
          ` <button class="fulltog">⤢ full doc (${fullTxt.length.toLocaleString()} chars)</button>` +
          `<div style="max-height:180px;overflow:auto;white-space:pre-wrap;color:var(--fg);margin-top:4px" id="survtext-${pos}">${escapeHtml(fullTxt.slice(0, 1500))}${fullTxt.length > 1500 ? " …" : ""}</div>`;
        const tgt = $("survtext-" + pos);
        let view = "trunc"; // trunc | full | diff — full doc & diff coordinate so only one shows at a time
        const setView = (v) => {
          view = v;
          if (v === "full") { tgt.style.maxHeight = "70vh"; tgt.innerHTML = escapeHtml(fullTxt); }
          else if (v === "diff") {
            tgt.style.maxHeight = "70vh";
            tgt.innerHTML = (goldTxt.trim() ? "" : `<div class="muted">8B abstained on this doc (no gold extraction) — everything below is text only your extractor produced:</div>`) + wordDiff(goldTxt, extTxt);
          }
          else { tgt.style.maxHeight = "180px"; tgt.innerHTML = escapeHtml(fullTxt.slice(0, 1500)) + (fullTxt.length > 1500 ? " …" : ""); }
          const fb = t.querySelector(".fulltog"), db = t.querySelector(".difftog");
          if (fb) fb.textContent = v === "full" ? "⤡ collapse" : `⤢ full doc (${fullTxt.length.toLocaleString()} chars)`;
          if (db) db.textContent = v === "diff" ? "✕ hide diff" : "⇄ diff vs 8B";
        };
        const fb = t.querySelector(".fulltog"); fb.onclick = () => setView(view === "full" ? "trunc" : "full");
        const db = t.querySelector(".difftog"); if (db) db.onclick = () => setView(view === "diff" ? "trunc" : "diff");
      });
    }
  }

  // ---- view docs the pipeline DROPS that the 8B keeps (recall cost) ----
  function deathStage(pos, r) {
    // first enabled filter that fails this doc (it passed everything before it).
    for (let i = 0; i < PIPE.filters.length; i++) {
      const f = PIPE.filters[i]; if (!f.enabled) continue;
      const st = STAGES_BY_ID[f.stageId], score = MATRIX.columns[st.score_col][pos], T = r.stages[i].threshold;
      const vT = st.direction === "low_useful" ? -T : T;
      if (Engine.vOf(score, st.direction) < vT) return { st, score, T };
    }
    return null; // passed all filters → the terminal extractor abstained
  }
  function showRandomLosses() {
    const r = LAST_R; if (!r || !r.finalKept) return;
    $("losses-panel").classList.remove("collapsed");
    const gcol = MATRIX.columns.gold_useful, ids = MATRIX.columns.warc_record_id;
    const lost = [];
    for (let i = 0; i < N; i++) if (gcol[i] === 1 && !r.finalKept[i]) lost.push(i);
    const box = $("losses");
    if (!lost.length) { box.innerHTML = "<span class='muted'>nothing the 8B keeps is dropped — perfect recall on the sample. 🎉</span>"; return; }
    const pool = lost.slice(), pick = [];
    for (let k = 0; k < Math.min(6, pool.length); k++) pick.push(pool.splice((Math.random() * pool.length) | 0, 1)[0]);
    box.innerHTML = `<h3>🎲 ${pick.length} random LOSSES — 8B keeps, pipeline drops · ${lost.length}/${N} (~${fmtCount(lost.length * r.scale)} projected lost good docs = recall cost)</h3>`;
    for (const pos of pick) {
      const id = ids[pos], death = deathStage(pos, r);
      const d = el("div", "panel"); d.style.margin = "6px 0"; d.style.borderLeft = "3px solid var(--red)";
      let why;
      if (death) {
        const snow = death.T == null || !isFinite(death.T) ? "—" : (+death.T).toFixed(3);
        const sc = death.score == null ? "null" : (+death.score).toFixed(3);
        const move = death.st.direction === "low_useful" ? `raise threshold above <b>${sc}</b>` : `lower threshold below <b>${sc}</b>`;
        why = `<div style="margin:4px 0;border-top:1px dashed var(--line);padding-top:4px"><span style="color:var(--amber)">dropped at <b>${death.st.label}</b> (score ${sc} vs thr ${snow}) — to recover: ${move} <span class="muted">(may still be dropped by a later filter)</span></span></div>`;
      } else {
        why = `<div class="muted" style="margin:4px 0">passed all filters; the extractor (${r.extractor.label}) abstained on it.</div>`;
      }
      d.innerHTML = `<span class="badge" style="background:rgba(240,106,106,.18);color:var(--red);border:1px solid var(--red)">FN · 8B keeps, dropped</span> <span class="muted">${id}</span>${why}<div class="muted" id="loss-${pos}">loading doc…</div>`;
      box.appendChild(d);
      fetchDoc(pos, id).then((doc) => {
        const t = $("loss-" + pos); if (!t) return;
        if (!doc) { t.textContent = IS_DEMO ? "(demo — no doc text)" : "(doc text unavailable)"; return; }
        const ex = bestExtraction(doc), full = ex.text;
        t.innerHTML = `<a href="${escapeHtml(doc.url) || "#"}" target="_blank">${escapeHtml(doc.url)}</a> <span class="muted">· extracted (${ex.label})</span> <button class="lfull">⤢ full doc (${full.length.toLocaleString()} chars)</button>` +
          `<div style="max-height:170px;overflow:auto;white-space:pre-wrap;color:var(--fg);margin-top:4px" id="losstext-${pos}">${escapeHtml(full.slice(0, 1400))}${full.length > 1400 ? " …" : ""}</div>`;
        let open = false; const fb = t.querySelector(".lfull");
        fb.onclick = () => { open = !open; const tg = $("losstext-" + pos); if (open) { tg.style.maxHeight = "70vh"; tg.innerHTML = escapeHtml(full); fb.textContent = "⤡ collapse"; } else { tg.style.maxHeight = "170px"; tg.innerHTML = escapeHtml(full.slice(0, 1400)) + (full.length > 1400 ? " …" : ""); fb.textContent = `⤢ full doc (${full.length.toLocaleString()} chars)`; } };
      });
    }
  }

  // ---- boot ----
  async function loadJson(path) { const r = await fetch(path); if (!r.ok) throw new Error(path); return r.json(); }

  async function boot() {
    REG = await loadJson(DATA + "registry.json");
    STAGES_BY_ID = Object.fromEntries(REG.stages.map((s) => [s.id, s]));
    try {
      MATRIX = await loadJson(DATA + "pipeline_matrix_10k.json");
      REG = MATRIX.meta.registry || REG; STAGES_BY_ID = Object.fromEntries(REG.stages.map((s) => [s.id, s]));
    } catch (e) { MATRIX = synthMatrix(REG); IS_DEMO = true; }
    N = MATRIX.meta.n;
    PIPE = defaultPipeline(); applyHash();
    $("dataset-info").textContent = (IS_DEMO ? "DEMO synthetic data — " : "") + N.toLocaleString() + " docs";
    $("dataset-info").style.color = IS_DEMO ? "var(--amber)" : "var(--green)";
    $("nwarc-lbl").textContent = "7.9M";
    renderPalette(); renderToolbar(); render();
    document.querySelectorAll(".panel.collapsible > h2").forEach((h) => { h.onclick = () => h.parentElement.classList.toggle("collapsed"); });
  }
  boot();
})();
