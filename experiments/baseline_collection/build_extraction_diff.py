# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build an HTML viewer comparing extractions of the same page across methods.

Given a results JSON (list of dicts per doc), emit a single self-contained HTML file. For each doc it
shows the researcher's note + desideratum, then a responsive row of panels for whichever of these are
present: `raw_html`, `resiliparse_text`, `hq_text`, `gold_text` — plus a global toggle for an hq↔gold
inline word-diff. Panels scroll independently; raw HTML is capped for display.
"""

from __future__ import annotations

import argparse
import difflib
import html
import json
import os
import re
import sys

_TOK = re.compile(r"\S+|\s+")
RAW_HTML_CAP = 120_000  # chars of raw html shown per doc


DIFF_CAP = 120_000  # skip the O(n^2) word-diff above this combined size


def _word_diff(a: str, b: str) -> str:
    """Inline word-level diff of a(hq)->b(gold): <del> for hq-only, <ins> for gold-only."""
    a, b = a or "", b or ""
    if len(a) + len(b) > DIFF_CAP:
        return (
            f'<i>[inline diff skipped — texts too large ({len(a)}+{len(b)} chars); '
            f"use the side-by-side panels]</i>"
        )
    ta, tb = _TOK.findall(a), _TOK.findall(b)
    sm = difflib.SequenceMatcher(a=ta, b=tb, autojunk=False)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            parts.append(html.escape("".join(ta[i1:i2])))
        if tag in ("delete", "replace"):
            parts.append("<del>" + html.escape("".join(ta[i1:i2])) + "</del>")
        if tag in ("insert", "replace"):
            parts.append("<ins>" + html.escape("".join(tb[j1:j2])) + "</ins>")
    return "".join(parts)


def _cached(cache: dict, url: str, which: str) -> str | None:
    """Look up a precomputed diff; supports {url:{hq,resi}} and legacy {url: hq_html}."""
    v = cache.get(url)
    if isinstance(v, dict):
        return v.get(which)
    return v if which == "hq" else None


# (field, label, css-modifier) in display order
_PANELS = [
    ("raw_html", "raw html", "raw"),
    ("resiliparse_text", "resiliparse", "resi"),
    ("hq_text", "hq spec", "hq"),
    ("gold_text", "gold v1", "gold"),
]


def _panel(text: str, label: str, mod: str, cap: int | None = None, extra: str = "") -> str:
    shown = text or ""
    trunc = ""
    if cap and len(shown) > cap:
        shown = shown[:cap]
        trunc = f" &middot; <i>truncated to {cap:,} of {len(text):,}</i>"
    return (
        f'<div class="col panel-{mod}"><div class="lab lab-{mod}">{label} &middot; {len(text or "")} chars{trunc}{extra}</div>'
        f'<pre class="ex">{html.escape(shown)}</pre></div>'
    )


def _write(results: list[dict], out: str, diff_cache: dict | None = None) -> None:
    diff_cache = diff_cache or {}
    raw_dir = os.path.join(os.path.dirname(out) or ".", "raw_pages")
    os.makedirs(raw_dir, exist_ok=True)
    present_mods: list[tuple[str, str]] = []  # (mod, label) present across docs, in panel order
    cards, nav = [], []
    for i, r in enumerate(results):
        nav.append(f'<a href="#d{i}">{i+1}. {html.escape(r.get("register",""))}</a>')
        note = html.escape(r.get("note") or "").strip() or "<i>(no note)</i>"
        fid = f'<div class="fidelity">{html.escape(r["fidelity"])}</div>' if r.get("fidelity") else ""
        panel_parts = []
        for f, lab, mod in _PANELS:
            if r.get(f) is None:
                continue
            if (mod, lab) not in present_mods:
                present_mods.append((mod, lab))
            extra = ""
            if f == "raw_html":
                with open(os.path.join(raw_dir, f"{i}.html"), "w") as rf:
                    rf.write(r.get(f) or "")
                extra = f' &middot; <a href="raw_pages/{i}.html" target="_blank" rel="noopener">view rendered ↗</a>'
            panel_parts.append(_panel(r.get(f) or "", lab, mod, RAW_HTML_CAP if f == "raw_html" else None, extra))
        panels = "".join(panel_parts)
        hq, gold, resi = r.get("hq_text") or "", r.get("gold_text") or "", r.get("resiliparse_text") or ""
        hq_diff = _cached(diff_cache, r["url"], "hq") or _word_diff(hq, gold)
        resi_diff = _cached(diff_cache, r["url"], "resi") or _word_diff(resi, gold)
        cards.append(
            f'<section id="d{i}" class="card">'
            f'<div class="hd"><span class="idx">#{i+1}</span>'
            f'<span class="reg">{html.escape(r.get("register",""))}</span>'
            f'<span class="why">{html.escape(r.get("why",""))}</span>'
            f'<a class="url" href="{html.escape(r["url"])}" target="_blank">source ↗</a></div>'
            f'<div class="note"><b>your note:</b> {note}</div>{fid}'
            f'<div class="cols">{panels}</div>'
            f'<div class="diffwrap diffwrap-hq"><div class="lab lab-diff">hq&rarr;gold &middot; '
            f'<del>hq only</del> &rarr; <ins>gold only</ins></div><pre class="ex diff">{hq_diff}</pre></div>'
            f'<div class="diffwrap diffwrap-resi"><div class="lab lab-diff">resiliparse&rarr;gold &middot; '
            f'<del>resiliparse only</del> &rarr; <ins>gold only</ins></div><pre class="ex diff">{resi_diff}</pre></div>'
            f"</section>"
        )
    ptoggle = "".join(
        f'<label><input type="checkbox" data-mod="{mod}" checked> {html.escape(lab)}</label>' for mod, lab in present_mods
    )
    has_resi = any(r.get("resiliparse_text") for r in results)
    resibtn = '<button id="tresi">resiliparse→gold</button>' if has_resi else ""
    doc = (
        _TEMPLATE.replace("__NAV__", " ".join(nav))
        .replace("__PTOGGLE__", ptoggle)
        .replace("__RESIBTN__", resibtn)
        .replace("__CARDS__", "\n".join(cards))
    )
    with open(out, "w") as f:
        f.write(doc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="scratch/gold_extraction/extraction_results.json")
    ap.add_argument("--out", default="scratch/gold_extraction/extraction_diff.html")
    ap.add_argument("--diff-cache", default=None, help="json {url: precomputed inline-diff html}")
    args = ap.parse_args()
    cache = json.load(open(args.diff_cache)) if args.diff_cache and os.path.exists(args.diff_cache) else {}
    _write(json.load(open(args.results)), args.out, cache)
    print(f"wrote {args.out} ({len(json.load(open(args.results)))} docs)")
    return 0


_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Extraction comparison</title>
<style>
:root{--bg:#fbfbfa;--fg:#1c1c1a;--mut:#6b6b66;--line:#e4e3df;--card:#fff;--raw:#7a7a75;--resi:#8a5cd6;--hq:#b45309;--gold:#15803d;--accent:#2f6fed;--delbg:#fde2e1;--delfg:#8a1c13;--insbg:#d8f2df;--insfg:#0f5c2e}
@media(prefers-color-scheme:dark){:root{--bg:#17181a;--fg:#e6e6e3;--mut:#9a9a95;--line:#2c2e31;--card:#1e2022;--raw:#9a9a95;--resi:#a986ea;--hq:#f0a04b;--gold:#5bbd82;--accent:#6ea0ff;--delbg:#3a1c1a;--delfg:#f0a59d;--insbg:#16351f;--insfg:#7fd39b}}
:root[data-theme=dark]{--bg:#17181a;--fg:#e6e6e3;--mut:#9a9a95;--line:#2c2e31;--card:#1e2022;--raw:#9a9a95;--resi:#a986ea;--hq:#f0a04b;--gold:#5bbd82;--accent:#6ea0ff;--delbg:#3a1c1a;--delfg:#f0a59d;--insbg:#16351f;--insfg:#7fd39b}
:root[data-theme=light]{--bg:#fbfbfa;--fg:#1c1c1a;--mut:#6b6b66;--line:#e4e3df;--card:#fff;--raw:#7a7a75;--resi:#8a5cd6;--hq:#b45309;--gold:#15803d;--accent:#2f6fed;--delbg:#fde2e1;--delfg:#8a1c13;--insbg:#d8f2df;--insfg:#0f5c2e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,BlinkMacSystemFont,sans-serif}
#nav{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:.5rem .8rem;display:flex;gap:.5rem;flex-wrap:wrap;align-items:center;font-size:12.5px}
#nav a{color:var(--accent);text-decoration:none;padding:1px 6px;border:1px solid var(--line);border-radius:5px}
.toggle{margin-left:auto;display:flex;gap:.3rem}
.toggle button{font-size:12.5px;padding:3px 10px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);cursor:pointer}
.toggle button.on{background:var(--accent);color:#fff;border-color:var(--accent);font-weight:600}
main{max-width:1700px;margin:1rem auto;padding:0 1rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;margin:1.1rem 0;padding:.8rem 1rem;scroll-margin-top:3rem}
.hd{display:flex;gap:.7rem;align-items:baseline;flex-wrap:wrap}
.idx{font-weight:700;color:var(--mut)}.reg{font-weight:600;background:color-mix(in srgb,var(--accent) 15%,transparent);color:var(--accent);padding:1px 8px;border-radius:10px;font-size:13px}
.why{color:var(--fg)}.url{margin-left:auto;color:var(--mut);font-size:13px;text-decoration:none}
.note{margin:.5rem 0 .35rem;padding:.45rem .7rem;background:color-mix(in srgb,var(--accent) 9%,transparent);border-left:3px solid var(--accent);border-radius:5px;font-size:14px}
.fidelity{margin:0 0 .7rem;font-size:12.5px;color:var(--mut);font-family:ui-monospace,Menlo,monospace}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:.7rem}
.col{border:1px solid var(--line);border-radius:8px;overflow:hidden;min-width:0}
.lab{font-size:12px;font-weight:700;letter-spacing:.03em;text-transform:uppercase;padding:.35rem .6rem;border-bottom:1px solid var(--line)}
.lab-raw{color:var(--raw)}.lab-resi{color:var(--resi)}.lab-hq{color:var(--hq)}.lab-gold{color:var(--gold)}.lab-diff{color:var(--accent)}
.lab-diff del{color:var(--delfg)}.lab-diff ins{color:var(--insfg)}
.lab a{color:var(--accent);text-decoration:none}.lab a:hover{text-decoration:underline}
.ptoggle{display:flex;gap:.7rem;align-items:center;font-size:12.5px;margin-left:1rem;flex-wrap:wrap}
.ptoggle label{cursor:pointer;user-select:none}.ptoggle b{color:var(--mut)}
body.hide-raw .panel-raw,body.hide-resi .panel-resi,body.hide-hq .panel-hq,body.hide-gold .panel-gold{display:none}
.ex{margin:0;padding:.7rem .8rem;white-space:pre-wrap;word-break:break-word;font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;max-height:65vh;overflow:auto;tab-size:2}
.diffwrap{margin-top:.7rem;border:1px solid var(--line);border-radius:8px;overflow:hidden;display:none}
.diff del{background:var(--delbg);color:var(--delfg);text-decoration:line-through}
.diff ins{background:var(--insbg);color:var(--insfg);text-decoration:none}
body.mode-hq .cols,body.mode-resi .cols{display:none}
body.mode-hq .diffwrap-hq{display:block}
body.mode-resi .diffwrap-resi{display:block}
</style></head><body class="mode-side">
<div id="nav">__NAV__
  <span class="ptoggle"><b>show:</b> __PTOGGLE__</span>
  <span class="toggle"><button id="tside" class="on">panels</button><button id="tdiff">hq→gold</button>__RESIBTN__</span>
</div>
<main>__CARDS__</main>
<script>
const b=document.body;
const modes=[['side','tside'],['hq','tdiff'],['resi','tresi']].map(([m,id])=>[m,document.getElementById(id)]).filter(([m,el])=>el);
function setMode(m){b.classList.remove('mode-hq','mode-resi');if(m==='hq')b.classList.add('mode-hq');if(m==='resi')b.classList.add('mode-resi');modes.forEach(([k,el])=>el.classList.toggle('on',k===m));}
modes.forEach(([k,el])=>el.onclick=()=>setMode(k));
document.querySelectorAll('.ptoggle input').forEach(cb=>cb.onchange=()=>document.body.classList.toggle('hide-'+cb.dataset.mod,!cb.checked));
</script>
</body></html>"""


if __name__ == "__main__":
    sys.exit(main())
