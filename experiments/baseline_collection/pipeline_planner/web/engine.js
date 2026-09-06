// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0
//
// Cascade evaluation engine — pure client-side math over the cached per-doc matrix. No model re-runs:
// every pipeline is a sequence of boolean masks + aggregations. Exposed as a global `Engine`.
//
// Shape of a pipeline: HTML classifiers -> THE EXTRACTOR -> TEXT classifiers. Exactly one extractor, and
// it is required (nothing downstream exists without extracted text). Where a classifier sits is decided by
// what it READS, not by the user: a stage with `requires_text: "<extractor key>"` consumes that extractor's
// output and so runs after it; everything else reads markup and runs before it. A TEXT classifier is only
// valid when the selected extractor is the one it was trained on, so switching the extractor invalidates
// them loudly instead of scoring them against text they never saw.
//
// EARLY-EXIT stages (`mode: "band"`): a cheap classifier can decide the confident tails itself and hand
// only the uncertain middle to the expensive model behind it. Docs at or above `hi` are ACCEPTED (kept,
// and they skip every later filter); docs below `lo` are dropped; the band in between flows on. This is
// what lets a 3932-docs/chip/s gate spend a 36-docs/chip/s model on ~9% of the corpus instead of 30%.
// Accepted docs still need their text, so the extractor is charged for them too — they leave the
// CLASSIFIER chain, not the pipeline.
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
    let tpuChipSec = 0, cpuCoreSec = 0;

    const ext = stageById[pipeline.extractor.stageId];
    const extKey = extKeyOf(ext, target);

    // Split the filters around the extractor by what each one reads.
    const pre = [], post = [];
    pipeline.filters.forEach((f, filterIdx) => {
      const st = stageById[f.stageId];
      if (!st) throw new Error(`unknown stage ${f.stageId}`);
      if (st.kind === "extractor") throw new Error(`${st.label} is an extractor — select it AS the extractor, not as a filter`);
      if (!st.requires_text) { pre.push({ f, st, filterIdx }); return; }
      if (st.requires_text !== extKey) {
        throw new Error(`${st.label} reads ${st.requires_text} text, but this pipeline extracts with ${ext.label}`);
      }
      post.push({ f, st, filterIdx });
    });

    // Docs an early-exit stage has already accepted: they are kept, and skip all later filters.
    const accepted = new Uint8Array(n);
    let acceptedCount = 0;

    function runFilters(list) {
      const rows = [];
      for (const { f, st, filterIdx } of list) {
        const rec = {
          stageId: f.stageId, filterIdx, label: st.label, device: st.device,
          enabled: f.enabled, docsIn: reaching, afterExtract: !!st.requires_text,
        };
        if (!f.enabled) { // disabled => pure pass-through, zero compute
          rec.docsOut = reaching; rec.skipped = true; rec.reachMask = reach; rec.unitSec = 0;
          rows.push(rec); continue;
        }
        const keep = new Uint8Array(n);
        let out = 0, T = null, unitSec = 0;
        if (f.mode === "band" && !st.oracle) {
          // Early exit: decide the confident tails here, pass only the uncertain band downstream.
          const scores = cols[st.score_col];
          const vHi = vOfThreshold(f.hi, st.direction), vLo = vOfThreshold(f.lo, st.direction);
          let acc = 0;
          for (let i = 0; i < n; i++) {
            if (!reach[i]) continue;
            const v = vOf(scores[i], st.direction);
            if (v >= vHi) { accepted[i] = 1; acc++; }        // kept outright; skips every later filter
            else if (v >= vLo) { keep[i] = 1; out++; }        // uncertain: hand to the next stage
          }
          acceptedCount += acc;
          rec.accepted = acc;
          rec.band = out;
          T = f.lo;
          const tput = (f.throughput ?? st.throughput) * (st.tokenization_bound ? cap.tokEfficiency : 1);
          unitSec = (reaching * scale) / tput;
        } else if (st.oracle) {
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
        rows.push(rec);
        reach = keep; reaching = out;
      }
      return rows;
    }

    const stages = runFilters(pre);

    // THE extractor: runs on every doc the HTML filters passed, then drops the ones it abstains on
    // (jusText / resiliparse-rs have no abstain column, so they pass everything through).
    // Everything still destined for the corpus needs extracting — the band AND anything an early-exit
    // stage already accepted — so the extractor is charged for both.
    const extDocsIn = reaching + acceptedCount;
    const extReachMask = reach;
    const extTput = pipeline.extractor.throughput ?? ext.throughput;
    const extUnitSec = ext.oracle ? 0 : (extDocsIn * scale) / extTput;
    if (ext.device === "tpu") tpuChipSec += extUnitSec; else cpuCoreSec += extUnitSec;
    const extKeepCol = ext.oracle ? target.gold_col : ext.label_col ? ext.label_col + "_useful" : null;
    {
      const keep = new Uint8Array(n);
      let out = 0;
      for (let i = 0; i < n; i++) {
        const ok = !extKeepCol || cols[extKeepCol][i] === 1;
        if (reach[i]) { if (ok) { keep[i] = 1; out++; } }
        else if (accepted[i] && !ok) { accepted[i] = 0; acceptedCount--; }  // the extractor abstained
      }
      reach = keep; reaching = out;
    }
    const extDocsOut = reaching + acceptedCount;

    // TEXT classifiers: they read what the extractor just produced.
    const postStages = runFilters(post);

    // Final keep-set: the band survivors PLUS everything accepted early.
    const finalKept = new Uint8Array(n);
    let kept = 0;
    for (let i = 0; i < n; i++) if (reach[i] || accepted[i]) { finalKept[i] = 1; kept++; }

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

    const rowOf = (s) => ({
      label: s.label, device: s.device, unitSec: s.unitSec,
      pct: s.device === "tpu" ? (tpuChipSec ? 100 * s.unitSec / tpuChipSec : 0) : (cpuCoreSec ? 100 * s.unitSec / cpuCoreSec : 0),
    });
    const computeBreakdown = stages.filter((s) => s.enabled).map(rowOf);
    computeBreakdown.push({
      label: ext.label + " (extract)", device: ext.device, unitSec: extUnitSec,
      pct: ext.device === "tpu" ? (tpuChipSec ? 100 * extUnitSec / tpuChipSec : 0) : (cpuCoreSec ? 100 * extUnitSec / cpuCoreSec : 0),
    });
    computeBreakdown.push(...postStages.filter((s) => s.enabled).map(rowOf));

    return {
      n, covered, scale, totalDocs: TOTAL_DOCS, targetId: target.id, targetLabel: target.label,
      stages, postStages,
      extractor: {
        id: ext.id, label: ext.label, device: ext.device,
        docsIn: extDocsIn, docsOut: extDocsOut, unitSec: extUnitSec, reachMask: extReachMask, kept,
      },
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
