# WARC → HTML decoding: the `�` (U+FFFD) gotcha and how to avoid it

**TL;DR — when you read HTML out of a WARC, decode the raw payload bytes with the
WHATWG charset algorithm (BOM → HTTP `Content-Type` charset → `<meta>` →
charset-normalizer → **cp1252**, *never* `errors="replace"`). In this repo the correct
decoder already exists: `decode_warcs_clean.decode_payload`. Use it. Do not use
`download_warcs._download_one_warc` for new work — it is the broken one.**

Origin: hand-off notes at `~/Documents/School/Stanford/Research/jusText/WARC-decoding-recommendations.md`
(diagnosed on the general/dev split: **88/1000 docs contained `�`**; ~6.5% of dev3).

## The bug

Every `�` is a real source character that was destroyed at decode time. Examples:

| stored | was | character |
|---|---|---|
| `Ghana�s`, `user�s` | `'s` | curly apostrophe U+2019 |
| `pi� spinte` | `più` | à-grave (cp1252) |
| `�300` | `£300` | pound sign U+00A3 |
| `T�m�` | `Tämä` | Finnish ä |

Root cause: a decoder that reads `<meta charset>` only and then falls back to a **fixed
codec with `errors="replace"`** turns every non-ASCII byte of a legacy-encoded page into
`�`. **82/88 of the corrupted docs declare no `<meta charset>`** — so meta-only resolution
is the core defect.

`U+FFFD` is written *after* the original byte is thrown away → **unrecoverable**. No
downstream `ftfy`/repair helps. Corrupted rows can only be fixed by **re-decoding from the
WARC bytes**. (The separate, smaller `Ã©`/`â€™` *mojibake* case **is** recoverable — those
bytes survived; `ftfy.fix_text()` or `s.encode("latin-1").decode("utf-8")` fixes them.)

## The fix (3 rules)

1. **Decode the raw payload bytes once.** Split the HTTP header block off the WARC
   `response` record, decode the *payload bytes*. Never `.encode().decode()` a `str` you
   already have — that manufactures both `�` and `Ã©`.
2. **Resolve the charset in WHATWG order** (don't trust `<meta>` alone):
   BOM → HTTP `Content-Type` `charset=` (the header the meta tag is missing) → `<meta charset>`
   (first ~1024 B) → statistical (`charset-normalizer`) → **`windows-1252` fallback** (decodes
   any byte, no errors; correct for the legacy Western pages that dominate the failures).
   **Drop `errors="replace"` from the primary path.**
3. **Don't hand-roll it** (or, if you do, validate it). Library option:
   `w3lib.encoding.html_to_unicode(content_type_header, body_bytes)`.

## In THIS repo

- ✅ **Correct decoder (use this):** `experiments/baseline_collection/decode_warcs_clean.py`
  → `decode_payload(raw_bytes, content_type_header)`. Implements rules 1–2 exactly; the
  cp1252/latin-1 tail maps every byte so `�` is never produced. Validated at **0 U+FFFD**
  (its own `scan_fffd` regression mode: `--scan-fffd` → must report zero). `_decode_one_warc`
  wraps it for a whole WARC.
- ❌ **Broken decoder (do NOT use for new work):** `download_warcs._download_one_warc` uses
  `content.decode("utf-8", errors="replace")`. **The extractors (8B/1.7B/0.6B) ran through
  this** (via `run_extract_standalone` / `bench_logprob_vs_gen`), so their `text_*` outputs
  were produced from `�`-corrupted HTML.

## Practical guidance

- Any **new** WARC→HTML pass (jusText, classifiers, re-extraction) must use
  `decode_warcs_clean.decode_payload`, then **validate** with a U+FFFD scan before trusting it.
- The comparison dataset's `raw_html` (where present) already used the clean decoder — but
  the LLM `text_*` columns came from the broken one, so there is a **decode asymmetry**. If a
  strict apples-to-apples extraction comparison matters, re-extract on clean HTML.
- **Don't let the dataset get littered with `�`.** Scan for it. A correct pipeline yields zero.
