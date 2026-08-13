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
    // Math.imul, NOT `*`: seed * 1103515245 exceeds 2^53 from the second iteration, so a plain
    // multiply silently rounds and `& 0x7fffffff` masks a wrong value. That collapsed this LCG to
    // ~16k distinct outputs, making every bootstrap replicate resample the same biased doc subset —
    // the CI came out not even containing its own point estimate. imul does exact 32-bit math.
    let seed = 12345;
    const rand = () => ((seed = (Math.imul(seed, 1103515245) + 12345) & 0x7fffffff) / 0x7fffffff);
    for (let b = 0; b < B; b++) {
      let s = 0;
      for (let i = 0; i < n; i++) s += perDocValues[(rand() * n) | 0];
      means[b] = s / n;
    }
    means.sort();
    const lo = means[(0.025 * B) | 0], hi = means[(0.975 * B) | 0];
    return { total: mean * scale, lo: lo * scale, hi: hi * scale };
  }

  // Column keys for one (extractor, target) pairing. `null` means "identical by definition":
  // an extractor scored against itself always agrees and always has similarity 1.0.
  // An oracle stage IS the target by construction, so it resolves to the target's own columns.
  function extKeyOf(stage, target) {
    if (stage.oracle && target) return target.extractor_id.replace(/^extract_/, "");
    return stage.id.replace(/^extract_/, "");
  }
  function levKeys(ext, target) {
    if (ext.oracle || ext.id === target.extractor_id) return { lev: null, both: null };
    const k = extKeyOf(ext, target);
    return { lev: `lev_${k}__${target.id}`, both: `both_${k}__${target.id}` };
  }

  // pipeline = { filters:[{stageId,enabled,mode,threshold,recall,throughput?}], extractor:{stageId,throughput?},
  //             capacity:{nChips,nCores,docsPerWarc,nWarcs,tokEfficiency} }
  // stageById: id -> registry stage. cols: matrix.columns. n = #docs. target: registry target.
  function evaluate(cols, n, pipeline, stageById, target) {
    const cap = pipeline.capacity;
    const TOTAL_DOCS = cap.docsPerWarc * cap.nWarcs;
    const gold = cols[target.gold_col];

    // Docs the target never judged leave the universe entirely: they seed the reach mask as 0, so
    // they are excluded from every filter, metric, compute total and projection alike. `scale`
    // therefore projects from the COVERED docs only.
    const cov = cols[target.coverage_col];
    let reach = new Uint8Array(n);
    let reaching = 0;
    for (let i = 0; i < n; i++) if (cov[i] === 1) { reach[i] = 1; reaching++; }
    const covered = reaching;
    const scale = covered ? TOTAL_DOCS / covered : 0;
    const stages = [];
    let tpuChipSec = 0, cpuCoreSec = 0;

    for (const f of pipeline.filters) {
      const st = stageById[f.stageId];
      const rec = { stageId: f.stageId, label: st.label, device: st.device, enabled: f.enabled, docsIn: reaching };
      if (!f.enabled) { // disabled => pure pass-through, zero compute
        rec.docsOut = reaching; rec.skipped = true; rec.reachMask = reach; rec.unitSec = 0;
        stages.push(rec); continue;
      }
      const keep = new Uint8Array(n);
      let out = 0;
      let T = null;
      let unitSec = 0;
      if (st.oracle) {
        // Perfect and free: passes exactly the docs the target keeps, contributing no compute.
        for (let i = 0; i < n; i++) if (reach[i] && gold[i]) { keep[i] = 1; out++; }
      } else {
        const scores = cols[st.score_col];
        T = f.mode === "recall" ? thresholdForRecall(scores, gold, reach, st.direction, f.recall) : f.threshold;
        const vCut = vOfThreshold(T, st.direction);
        for (let i = 0; i < n; i++) if (reach[i] && vOf(scores[i], st.direction) >= vCut) { keep[i] = 1; out++; }
        const tput = (f.throughput ?? st.throughput) * (st.tokenization_bound ? cap.tokEfficiency : 1);
        unitSec = (reaching * scale) / tput;
      }
      if (st.device === "tpu") tpuChipSec += unitSec; else cpuCoreSec += unitSec;
      Object.assign(rec, { docsOut: out, threshold: T, unitSec, reachMask: reach });
      stages.push(rec);
      reach = keep; reaching = out;
    }

    // Terminal extractor: processes all survivors (compute), then its own abstain decides final-kept.
    const ext = stageById[pipeline.extractor.stageId];
    // Oracle terminal abstains exactly as the target does; jusText has no label => never abstains.
    const keepCol = ext.oracle ? target.gold_col : ext.label_col ? ext.label_col + "_useful" : null;
    const extracted = (i) => (keepCol ? cols[keepCol][i] === 1 : true);
    const finalKept = new Uint8Array(n);
    let kept = 0;
    for (let i = 0; i < n; i++) if (reach[i] && extracted(i)) { finalKept[i] = 1; kept++; }
    const extTput = pipeline.extractor.throughput ?? ext.throughput;
    const extUnitSec = ext.oracle ? 0 : (reaching * scale) / extTput;
    if (ext.device === "tpu") tpuChipSec += extUnitSec; else cpuCoreSec += extUnitSec;

    // Metrics over finalKept vs the selected target, across covered docs only.
    let tp = 0, fp = 0, fn = 0, goldTotal = 0;
    for (let i = 0; i < n; i++) {
      if (cov[i] !== 1) continue;
      if (gold[i]) goldTotal++;
      if (finalKept[i]) { gold[i] ? tp++ : fp++; } else if (gold[i]) fn++;
    }
    const precision = tp + fp ? tp / (tp + fp) : 0;
    const recallM = tp + fn ? tp / (tp + fn) : 0;
    const f1 = 2 * tp + fp + fn ? (2 * tp) / (2 * tp + fp + fn) : 0;

    // Levenshtein vs the target over kept docs (the target's own extractor => 1.0 by definition).
    const { lev: levKey, both: bothKey } = levKeys(ext, target);
    let levU = 0, levUn = 0, levB = 0, levBn = 0;
    const tokCol = cols["tok_" + extKeyOf(ext, target)];
    // One slot per COVERED doc (0 for non-kept) so the bootstrap mean matches `scale`'s denominator.
    const perDocTok = new Float64Array(covered);
    let slot = 0;
    for (let i = 0; i < n; i++) {
      if (cov[i] !== 1) continue;
      const s = slot++;
      if (!finalKept[i]) continue;
      perDocTok[s] = tokCol[i] || 0;
      const lv = levKey ? cols[levKey][i] : 1.0;
      levU += lv; levUn++;
      if (!bothKey || cols[bothKey][i] === 1) { levB += lv; levBn++; }
    }
    // perDocTok is per-covered-doc (0 for non-kept); its mean × TOTAL_DOCS = projected total (NOT × scale).
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
      n, covered, scale, totalDocs: TOTAL_DOCS, targetId: target.id, targetLabel: target.label,
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

  return { evaluate, thresholdForRecall, vOf, levKeys, extKeyOf, SEC_PER_YEAR, SEC_PER_DAY };
})();

if (typeof module !== "undefined") module.exports = Engine; // node unit tests
