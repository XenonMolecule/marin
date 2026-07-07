# Fast-curation pipeline versions

Human-readable ledger of what each `fastpipe_vN` actually means. The authoritative
machine-readable definition is `SPECS` in `spec.py`; this file explains the *why*. Keep
the two in sync — update both in the same commit when you bump a version.

Each version's `compute_version()[:10]` hash is recorded so any GCS namespace
(`gs://marin-<region>/documents/fast_curation/fastpipe_vN-<hash>/...`) can be traced back
to its exact configuration. The hash covers every namespace-defining field **except**
`modernbert_threshold` (that threshold is a cheap late-bound re-filter over stored probs,
not a recompute trigger).

| field | meaning |
|---|---|
| fastText model + threshold | useful-vs-not filter on `body_strip` text; **below threshold → dropped, no record** |
| JustText version | the XenonMolecule/jusText fork tag; produces the training `text` (output content depends on it) |
| JustText max_html_chars | pages with raw html longer than this **skip JustText → dropped**; a quality knob (too low loses long articles/books) whose purpose is to bound lxml DOM-parse cost |
| JustText timeout | per-doc wall-clock guard (s); a doc exceeding it is hard-killed → dropped (guards the rare page lxml's C parser hangs on) |
| tokenizer / pad / max_length / single_window | how the `body_strip` text is tokenized for ModernBERT |
| ModernBERT checkpoint | the useful-vs-not classifier run on (pre-tokenized) `body_strip` |
| step_order | the cascade order |

Recompute any hash with
`python -c "from experiments.fast_curation.spec import get_spec; print(get_spec('fastpipe_vN').version())"`.

---

## fastpipe_v3  (ACTIVE)

- **Status**: active — the version run over the full 10,364-WARC pool, multi-region.
- **Hash**: `6855733850` → namespace `gs://marin-<region>/documents/fast_curation/fastpipe_v3-6855733850/`.
- **Changed vs v2**: `justext_max_html_chars` **3M → 50M** and `justext_timeout` **None → 60s**. Everything else identical to v2 (same models, thresholds, tokenizer, `V2_STEP_ORDER`).
- **Rationale**: v2's 3M-char html cap silently **dropped genuinely long documents** (long articles/books/docs pages — exactly the long-context training data we want). Raising the cap to 50M keeps them; a 60s per-doc timeout (hard-kill, since lxml's C parser ignores SIGALRM) guards the rare pathological page that would otherwise hang a worker. The 50M cap also fixed the Phase A OOM/oversized-parquet-cell failures that killed ~all v2 Phase A workers at 24GB (Phase A now also runs at 64GB and frees the decoded WARC before writing).
- **Note**: v2's partial output (~69/10364 WARCs) is **discarded** — v3 is a fresh namespace, not a resume of v2.

## fastpipe_v2

- **Status**: superseded by v3 (only ~69 WARCs were produced before the long-doc-drop issue was found; discarded).
- **Hash**: `34e3b152e0`. (The earlier live run used `f78c2b2b7a`, before `max_html_chars`/`timeout` became spec fields; that data is abandoned.)
- **Changed vs v1**: `step_order` → `decode → body_strip → fasttext → tokenize → modernbert → justext` (run ModernBERT **before** JustText so JustText, the dominant CPU cost, only touches ModernBERT-survivors ≈5× fewer docs). Output is intended to be identical to v1; only the compute order (and the 3-phase A/B/C layout) differs.
- **Rationale**: pure efficiency reorder of v1.

## fastpipe_v1

- **Status**: initial prototype (single-phase). Superseded by the v2/v3 3-phase layout.
- **Hash**: `2d1089842d`.
- **fastText**: `gs://marin-us-east5/classifiers/useful_fasttext/body_strip_scale_w320_strat_prep_mc500/model.bin`, threshold **0.0368**.
- **JustText**: `xenon-v4.2.0`, `max_html_chars=3M`, `timeout=None`.
- **Tokenizer**: `answerdotai/ModernBERT-base`, pad `50283`, `max_length=8192`, `single_window=True`.
- **ModernBERT**: `gs://marin-us-east5/checkpoints/modernbert-useful/mb-clf-base-10M-c8192/hf`, threshold **0.1974** on `P(useful)`.
- **step_order**: `decode → body_strip → fasttext → justext → tokenize → modernbert`.
- **Rationale**: first end-to-end version; reproduces the cascade the classifiers were trained/calibrated on. Single-window 8192 (chunked ModernBERT deferred — see `.agents/projects/chunked_modernbert_classifier.md`).
