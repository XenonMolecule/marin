// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0
//
// Cascade evaluation engine — pure client-side math over the cached per-doc matrix. No model re-runs:
// every pipeline is a sequence of boolean masks + aggregations. Exposed as a global `Engine`.
//
// Usefulness convention: each classifier has a `direction`. We map every score to a "usefulness value"
// v (higher = more useful) so a single rule `keep iff v(score) >= v(threshold)` handles both probability
// columns (high=useful) and LLM marker-logprob columns (low=useful, where null = very-confident-useful).

const Engine = (() => {
  const SEC_PER_YEAR = 31556952;
  const SEC_PER_DAY = 86400;

  // Map a raw score to usefulness-space (higher = more useful), honoring direction + nulls.
  function vOf(score, dir) {
    if (dir === "low_useful") return score == null ? Infinity : -score; // null logprob => most useful
    return score == null ? -Infinity : score; // high_useful (probabilities never null in practice)
  }
  function vOfThreshold(T, dir) {
    return dir === "low_useful" ? -T : T;
  }

  // Threshold (raw score space) that keeps `recall` of the gold-useful docs currently *reaching* this stage.
  function thresholdForRecall(scores, gold, reach, dir, recall) {
    const vs = [];
    for (let i = 0; i < scores.length; i++) if (reach[i] && gold[i]) vs.push(vOf(scores[i], dir));
    if (vs.length === 0) return dir === "low_useful" ? Infinity : -Infinity;
    vs.sort((a, b) => b - a); // most-useful first
    const k = Math.min(vs.length - 1, Math.max(0, Math.floor(recall * vs.length) - 1 < 0 ? Math.floor(recall * vs.length) : Math.ceil(recall * vs.length) - 1));
    const vc = vs[Math.min(vs.length - 1, Math.max(0, Math.ceil(recall * vs.length) - 1))];
    return dir === "low_useful" ? (vc === Infinity ? Infinity : -vc) : vc;
  }

  function bootstrapCI(perDocValues, scale, B = 1000) {
    // perDocValues: length-N array (0 for non-kept). Projected total = mean * scale; bootstrap the mean.
    const n = perDocValues.length;
    let sum = 0;
    for (let i = 0; i < n; i++) sum += perDocValues[i];
    const mean = sum / n;
    const means = new Float64Array(B);
    let seed = 12345;
    const rand = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
    for (let b = 0; b < B; b++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += perDocValues[(rand() * n) | 0];
      means[b] = s / n;
    }
    means.sort();
    const lo = means[(0.025 * B) | 0], hi = means[(0.975 * B) | 0];
    return { total: mean * scale, lo: lo * scale, hi: hi * scale };
  }

  // pipeline = { filters:[{stageId,enabled,mode,threshold,recall,throughput?}], extractor:{stageId,throughput?},
  //             capacity:{nChips,nCores,docsPerWarc,nWarcs,tokEfficiency} }
  // stageById: id -> registry stage. cols: matrix.columns. n = #docs.
  function evaluate(cols, n, pipeline, stageById) {
    const cap = pipeline.capacity;
    const TOTAL_DOCS = cap.docsPerWarc * cap.nWarcs;
    const scale = TOTAL_DOCS / n;
    const gold = cols.gold_useful;

    let reach = new Uint8Array(n).fill(1);
    let reaching = n;
    const stages = [];
    let tpuChipSec = 0, cpuCoreSec = 0;

    for (const f of pipeline.filters) {
      const st = stageById[f.stageId];
      const rec = { stageId: f.stageId, label: st.label, device: st.device, enabled: f.enabled, docsIn: reaching };
      if (!f.enabled) { // disabled => pure pass-through, zero compute
        rec.docsOut = reaching; rec.skipped = true; rec.reachMask = reach; rec.unitSec = 0;
        stages.push(rec); continue;
      }
      const scores = cols[st.score_col];
      let T = f.mode === "recall" ? thresholdForRecall(scores, gold, reach, st.direction, f.recall) : f.threshold;
      const vCut = vOfThreshold(T, st.direction);
      const keep = new Uint8Array(n);
      let out = 0;
      for (let i = 0; i < n; i++) if (reach[i] && vOf(scores[i], st.direction) >= vCut) { keep[i] = 1; out++; }
      const tput = (f.throughput ?? st.throughput) * (st.tokenization_bound ? cap.tokEfficiency : 1);
      const unitSec = (reaching * scale) / tput;
      if (st.device === "tpu") tpuChipSec += unitSec; else cpuCoreSec += unitSec;
      Object.assign(rec, { docsOut: out, threshold: T, unitSec, reachMask: reach });
      stages.push(rec);
      reach = keep; reaching = out;
    }

    // Terminal extractor: processes all survivors (compute), then its own abstain decides final-kept.
    const ext = stageById[pipeline.extractor.stageId];
    const extKey = { extract_8b: "gold_useful", extract_1p7b: "label_1p7b_useful", extract_0p6b: "label_0p6b_useful" }[ext.id];
    const extracted = (i) => (extKey ? cols[extKey][i] === 1 : true); // jusText never abstains
    const finalKept = new Uint8Array(n);
    let kept = 0;
    for (let i = 0; i < n; i++) if (reach[i] && extracted(i)) { finalKept[i] = 1; kept++; }
    const extTput = pipeline.extractor.throughput ?? ext.throughput;
    const extUnitSec = (reaching * scale) / extTput;
    if (ext.device === "tpu") tpuChipSec += extUnitSec; else cpuCoreSec += extUnitSec;

    // Metrics over finalKept vs 8B gold.
    let tp = 0, fp = 0, fn = 0, goldTotal = 0;
    for (let i = 0; i < n; i++) {
      if (gold[i]) goldTotal++;
      if (finalKept[i]) { gold[i] ? tp++ : fp++; } else if (gold[i]) fn++;
    }
    const precision = tp + fp ? tp / (tp + fp) : 0;
    const recallM = tp + fn ? tp / (tp + fn) : 0;
    const f1 = 2 * tp + fp + fn ? (2 * tp) / (2 * tp + fp + fn) : 0;

    // Levenshtein vs 8B over kept (gold extractor => 1.0 by definition).
    const levKey = { extract_1p7b: "lev_1p7b", extract_0p6b: "lev_0p6b", extract_justext: "lev_justext" }[ext.id];
    const bothKey = { extract_1p7b: "both_1p7b", extract_0p6b: "both_0p6b", extract_justext: "both_justext" }[ext.id];
    let levU = 0, levUn = 0, levB = 0, levBn = 0;
    const tokCol = cols["tok_" + ext.id.replace("extract_", "")];
    const perDocTok = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      if (!finalKept[i]) continue;
      perDocTok[i] = tokCol[i] || 0;
      const lv = levKey ? cols[levKey][i] : 1.0;
      levU += lv; levUn++;
      if (!bothKey || cols[bothKey][i] === 1) { levB += lv; levBn++; }
    }
    // perDocTok is per-sample-doc (0 for non-kept); its mean × TOTAL_DOCS = projected total (NOT × scale).
    const projTok = bootstrapCI(perDocTok, TOTAL_DOCS);

    const wallTpuYears = cap.nChips ? tpuChipSec / cap.nChips / SEC_PER_YEAR : 0;
    const wallCpuYears = cap.nCores ? cpuCoreSec / cap.nCores / SEC_PER_YEAR : 0;

    const computeBreakdown = stages.filter((s) => s.enabled).map((s) => ({
      label: s.label, device: s.device, unitSec: s.unitSec,
      pct: s.device === "tpu" ? (tpuChipSec ? 100 * s.unitSec / tpuChipSec : 0) : (cpuCoreSec ? 100 * s.unitSec / cpuCoreSec : 0),
    }));
    computeBreakdown.push({
      label: ext.label + " (extract)", device: ext.device, unitSec: extUnitSec,
      pct: ext.device === "tpu" ? (tpuChipSec ? 100 * extUnitSec / tpuChipSec : 0) : (cpuCoreSec ? 100 * extUnitSec / cpuCoreSec : 0),
    });

    return {
      n, scale, totalDocs: TOTAL_DOCS,
      stages, extractor: { id: ext.id, label: ext.label, device: ext.device, docsIn: reaching, kept, unitSec: extUnitSec },
      finalKept,
      summary: {
        f1, precision, recall: recallM, goldTotal, keptSample: kept,
        levUnion: levUn ? levU / levUn : null, levBoth: levBn ? levB / levBn : null,
        levUnionN: levUn, levBothN: levBn, // sample-doc denominators (multiply by scale to project)
        projTokens: projTok.total, projTokensLo: projTok.lo, projTokensHi: projTok.hi,
        tpuChipSec, cpuCoreSec, wallTpuYears, wallCpuYears,
        wallTpuDays: wallTpuYears * 365, wallCpuDays: wallCpuYears * 365,
      },
      computeBreakdown,
    };
  }

  return { evaluate, thresholdForRecall, vOf, SEC_PER_YEAR, SEC_PER_DAY };
})();

if (typeof module !== "undefined") module.exports = Engine; // node unit tests
