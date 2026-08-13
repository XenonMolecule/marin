// Copyright The Marin Authors — SPDX-License-Identifier: Apache-2.0
// Engine unit tests: `node --test experiments/baseline_collection/pipeline_planner/web/`
//
// Covers the agreement-target machinery: metrics follow the SELECTED target, and docs the target never
// judged leave the universe entirely (metrics, projections and the docs-reaching-stage-1 count alike).

const test = require("node:test");
const assert = require("node:assert");
const Engine = require("./engine.js");

const TARGET_LPV11 = { id: "lpv11", label: "lpv11", gold_col: "target_lpv11_useful", coverage_col: "cov_lpv11", extractor_id: "extract_lpv11" };
const TARGET_HQ = { id: "high_quality", label: "8B", gold_col: "target_hq_useful", coverage_col: "cov_hq", extractor_id: "extract_8b" };

const STAGES = {
  extract_lpv11: { id: "extract_lpv11", label: "lpv11", kind: "extractor", device: "tpu", throughput: 1, text_col: "text_lpv11", label_col: "label_lpv11" },
  extract_8b: { id: "extract_8b", label: "8B", kind: "extractor", device: "tpu", throughput: 1, text_col: "text_8b", label_col: "label_8b" },
  extract_justext: { id: "extract_justext", label: "jusText", kind: "extractor", device: "cpu", throughput: 1, text_col: "text_justext", label_col: null },
  ft: { id: "ft", label: "ft", kind: "classifier", device: "cpu", throughput: 1, score_col: "ft_prob", direction: "high_useful" },
  oracle_filter: { id: "oracle_filter", label: "Oracle filter", kind: "classifier", device: "cpu", throughput: 0, oracle: true },
  oracle_extract: { id: "oracle_extract", label: "Oracle extractor", kind: "extractor", device: "cpu", throughput: 0, oracle: true },
};

// 5 docs. lpv11 covers only the first 4 (doc 4 is from a WARC lpv11 never ran).
// lpv11 keeps docs 0,1 ; the 8B keeps docs 0,2 — so the two targets genuinely disagree.
const COLS = {
  cov_lpv11: [1, 1, 1, 1, 0],
  cov_hq: [1, 1, 1, 1, 1],
  target_lpv11_useful: [1, 1, 0, 0, 0],
  target_hq_useful: [1, 0, 1, 0, 1],
  label_lpv11_useful: [1, 1, 0, 0, 0],
  label_8b_useful: [1, 0, 1, 0, 1],
  ft_prob: [0.9, 0.8, 0.7, 0.1, 0.9],
  lev_8b__lpv11: [0.5, 0.5, 0.5, 0.5, 0.5],
  both_8b__lpv11: [1, 1, 1, 1, 1],
  lev_lpv11__high_quality: [0.4, 0.4, 0.4, 0.4, 0.4],
  both_lpv11__high_quality: [1, 1, 1, 1, 1],
  lev_justext__lpv11: [0.3, 0.3, 0.3, 0.3, 0.3],
  both_justext__lpv11: [1, 1, 1, 1, 1],
  tok_lpv11: [10, 10, 10, 10, 10],
  tok_8b: [20, 20, 20, 20, 20],
  tok_justext: [30, 30, 30, 30, 30],
  warc_record_id: ["a", "b", "c", "d", "e"],
};
const N = 5;

const pipe = (extractorId, filters = []) => ({
  filters,
  extractor: { stageId: extractorId },
  capacity: { nChips: 1, nCores: 1, docsPerWarc: 1, nWarcs: 100, tokEfficiency: 1 },
});

test("an extractor scored against itself agrees perfectly", () => {
  const r = Engine.evaluate(COLS, N, pipe("extract_lpv11"), STAGES, TARGET_LPV11);
  assert.strictEqual(r.summary.f1, 1);
  assert.strictEqual(r.summary.precision, 1);
  assert.strictEqual(r.summary.recall, 1);
  assert.strictEqual(r.summary.levBoth, 1, "self-similarity is 1.0 by definition, not a lookup");
});

test("switching the target re-scores the same pipeline", () => {
  // The 8B as a stage, judged against lpv11: it keeps docs 0 and 2 (doc 4 is outside coverage).
  // lpv11 keeps 0,1 => tp=1 (doc0), fp=1 (doc2), fn=1 (doc1).
  const vsLpv11 = Engine.evaluate(COLS, N, pipe("extract_8b"), STAGES, TARGET_LPV11);
  assert.strictEqual(vsLpv11.summary.f1, 0.5);
  assert.strictEqual(vsLpv11.summary.levBoth, 0.5, "uses lev_8b__lpv11");

  // Same pipeline against its own target is perfect — proving the switch, not the pipeline, moved the number.
  const vsHq = Engine.evaluate(COLS, N, pipe("extract_8b"), STAGES, TARGET_HQ);
  assert.strictEqual(vsHq.summary.f1, 1);
});

test("docs outside the target's coverage leave the universe entirely", () => {
  const r = Engine.evaluate(COLS, N, pipe("extract_8b"), STAGES, TARGET_LPV11);
  assert.strictEqual(r.covered, 4, "doc 4 is uncovered");
  assert.strictEqual(r.scale, 100 / 4, "projection scales from covered docs, not all docs");
  // Doc 4 is gold-useful for the 8B and would be kept by it; under the lpv11 target it must not
  // appear as a true positive, a false positive, or in the projected token total.
  assert.strictEqual(r.summary.goldTotal, 2, "only lpv11's 2 keeps within coverage");
  assert.strictEqual(r.extractor.kept, 2, "docs 0 and 2 — doc 4 never reaches the extractor");

  const hq = Engine.evaluate(COLS, N, pipe("extract_8b"), STAGES, TARGET_HQ);
  assert.strictEqual(hq.covered, 5, "the 8B judged every doc");
  assert.strictEqual(hq.extractor.kept, 3, "doc 4 is back in play under full coverage");
});

test("a filter cascade drops docs before the extractor", () => {
  const filters = [{ stageId: "ft", enabled: true, mode: "threshold", threshold: 0.85, recall: 0.99 }];
  const r = Engine.evaluate(COLS, N, pipe("extract_lpv11", filters), STAGES, TARGET_LPV11);
  // Only doc 0 clears ft_prob >= 0.85 within lpv11's coverage (doc 4 scores 0.9 but is uncovered).
  assert.strictEqual(r.stages[0].docsIn, 4);
  assert.strictEqual(r.stages[0].docsOut, 1);
  assert.strictEqual(r.summary.recall, 0.5, "kept 1 of lpv11's 2 useful docs");
  assert.strictEqual(r.summary.precision, 1);
});

test("a disabled filter is a pure pass-through", () => {
  const filters = [{ stageId: "ft", enabled: false, mode: "threshold", threshold: 0.85, recall: 0.99 }];
  const r = Engine.evaluate(COLS, N, pipe("extract_lpv11", filters), STAGES, TARGET_LPV11);
  assert.strictEqual(r.stages[0].docsOut, 4);
  assert.strictEqual(r.stages[0].unitSec, 0);
  assert.strictEqual(r.summary.f1, 1);
});

test("the oracle extractor is perfect and free", () => {
  const r = Engine.evaluate(COLS, N, pipe("oracle_extract"), STAGES, TARGET_LPV11);
  assert.strictEqual(r.summary.f1, 1, "reproduces the target's decisions exactly");
  assert.strictEqual(r.summary.levBoth, 1, "and its text");
  assert.strictEqual(r.extractor.unitSec, 0, "contributes no compute");
  assert.strictEqual(r.summary.tpuChipSec + r.summary.cpuCoreSec, 0);
  // It follows the SELECTED target, not a hard-coded one.
  const hq = Engine.evaluate(COLS, N, pipe("oracle_extract"), STAGES, TARGET_HQ);
  assert.strictEqual(hq.summary.f1, 1);
  assert.strictEqual(hq.extractor.kept, 3, "the 8B keeps 3 docs; the oracle matches that, not lpv11's 2");
});

test("the oracle filter passes exactly the target's keeps, for free", () => {
  const filters = [{ stageId: "oracle_filter", enabled: true, mode: "threshold", threshold: 0, recall: 0.99 }];
  const r = Engine.evaluate(COLS, N, pipe("extract_lpv11", filters), STAGES, TARGET_LPV11);
  assert.strictEqual(r.stages[0].docsIn, 4, "sees every covered doc");
  assert.strictEqual(r.stages[0].docsOut, 2, "emits exactly lpv11's 2 keeps");
  assert.strictEqual(r.stages[0].unitSec, 0, "costs nothing");
  assert.strictEqual(r.summary.f1, 1);
});

test("an oracle upstream shrinks what a downstream stage has to process", () => {
  // This is the whole point: cost of the real stage is charged on what REACHES it.
  const alone = Engine.evaluate(COLS, N, pipe("extract_lpv11", [
    { stageId: "ft", enabled: true, mode: "threshold", threshold: 0.0, recall: 0.99 },
  ]), STAGES, TARGET_LPV11);
  const behindOracle = Engine.evaluate(COLS, N, pipe("extract_lpv11", [
    { stageId: "oracle_filter", enabled: true, mode: "threshold", threshold: 0, recall: 0.99 },
    { stageId: "ft", enabled: true, mode: "threshold", threshold: 0.0, recall: 0.99 },
  ]), STAGES, TARGET_LPV11);
  assert.strictEqual(alone.stages[0].docsIn, 4);
  assert.strictEqual(behindOracle.stages[1].docsIn, 2, "ft now only sees the oracle's survivors");
  assert.ok(behindOracle.stages[1].unitSec < alone.stages[0].unitSec, "so its compute drops");
  assert.strictEqual(behindOracle.stages[0].unitSec, 0, "and the oracle itself adds nothing");
});

test("the bootstrap CI brackets its own point estimate", () => {
  // Regression: the LCG used a plain `*`, whose intermediate exceeds 2^53 after one iteration, so it
  // collapsed to ~16k distinct draws and every replicate resampled the same biased subset — the CI
  // came out entirely above the mean it was supposed to bracket.
  const r = Engine.evaluate(COLS, N, pipe("extract_lpv11"), STAGES, TARGET_LPV11);
  const { projTokens, projTokensLo, projTokensHi } = r.summary;
  assert.ok(projTokensLo <= projTokens, `lo ${projTokensLo} must be <= point ${projTokens}`);
  assert.ok(projTokens <= projTokensHi, `point ${projTokens} must be <= hi ${projTokensHi}`);
});

test("jusText never abstains, so it keeps everything reaching it", () => {
  const r = Engine.evaluate(COLS, N, pipe("extract_justext"), STAGES, TARGET_LPV11);
  assert.strictEqual(r.extractor.kept, 4, "all covered docs");
  assert.strictEqual(r.summary.recall, 1, "keeps every useful doc...");
  assert.strictEqual(r.summary.precision, 0.5, "...at the cost of precision");
});
