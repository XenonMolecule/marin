# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Build a self-contained relabel page for the dev docs whose recovered HTML snapshot couldn't be
gate-verified (no_signature / mismatch).

These docs HAVE recovered HTML, but the fingerprint didn't match the originally-labeled page (the
snapshot drifted), so the honest fix is to relabel against what the recovered HTML actually is. This
renders each recovered page in a script-disabled sandboxed iframe (the recovered HTML can contain
injected fake system-reminders — never execute it) next to keep/weak_keep/weak_drop/drop controls,
seeded with the current label. "Export" writes `relabel.json` (url -> {label}); feed it to
`export_devset.py build --include-unverified --relabels relabel.json`.
"""

from __future__ import annotations

import json
import logging
import sys
from urllib.parse import urlparse

from resiliparse.extract.html2text import extract_plain_text

from experiments.baseline_collection.export_devset import (
    JUDGE_LABELS,
    REGISTER_MAP,
    _domain_register,
    _unverified_html,
)

logger = logging.getLogger(__name__)

DEFAULT_HUMAN = "/Users/michaelryan/Downloads/devset_hard_labels-11.json"
OUT_HTML = "scratch/relabel_unverified.html"


def _readable_text(html: str) -> str:
    """resiliparse main-content text; fall back to the full page when there's no main block (nav/junk)."""
    for main in (True, False):
        try:
            text = extract_plain_text(html, main_content=main, alt_texts=True)
        except Exception:  # resiliparse can choke on malformed markup
            text = ""
        if len(text.strip()) >= 30:
            return text
    return text


def _current_label(url: str, human: dict, judge: dict) -> tuple[str | None, str]:
    if url in human and human[url].get("label"):
        return human[url]["label"], "human"
    if url in judge and judge[url].get("label"):
        return judge[url]["label"], "judge"
    return None, "none"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    human = json.load(open(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HUMAN))
    judge = json.load(open(JUDGE_LABELS))
    reg_map = json.load(open(REGISTER_MAP))
    html_by_url = _unverified_html(exclude=set())  # all non-match urls -> best recovered html
    logger.info("relabeling %d unverified-snapshot docs", len(html_by_url))

    docs = []
    for url in sorted(html_by_url):
        cur, src = _current_label(url, human, judge)
        html = html_by_url[url]
        text = _readable_text(html)
        docs.append(
            {
                "url": url,
                "register": reg_map.get(url) or _domain_register(urlparse(url).netloc),
                "cur": cur,
                "cur_src": src,
                "text": text,
                "html": html,
            }
        )

    # Escape `</` so a `</script>` inside any embedded page can't close our own <script> tag early.
    payload = json.dumps(docs).replace("</", "<\\/")
    with open(OUT_HTML, "w") as f:
        f.write(_TEMPLATE.replace("__DOCS__", payload))
    logger.info("wrote %s (%d docs)", OUT_HTML, len(docs))
    print(f"open file://{__import__('os').path.abspath(OUT_HTML)}")
    return 0


_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Relabel unverified snapshots</title>
<style>
 :root{--bg:#faf9f7;--fg:#1a1a1a;--mut:#6b6b6b;--line:#e2e0db;--keep:#1a7f4b;--wkeep:#7bb661;--wdrop:#d99a2b;--drop:#c0392b;--accent:#2b6cb0}
 *{box-sizing:border-box} body{margin:0;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:var(--bg);color:var(--fg)}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:14px;align-items:center;z-index:5;flex-wrap:wrap}
 header b{font-size:15px} .mut{color:var(--mut)} kbd{background:#eee;border:1px solid #ccc;border-radius:4px;padding:0 5px;font-size:12px}
 #wrap{display:grid;grid-template-columns:1fr 340px;gap:0;height:calc(100vh - 52px)}
 #docpane{overflow:auto;background:#fff;padding:24px 32px}
 #text{white-space:pre-wrap;font:14px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;max-width:70ch;word-break:break-word}
 #frame{width:100%;height:100%;border:0;background:#fff;display:none}
 #docpane.raw{padding:0} #docpane.raw #text{display:none} #docpane.raw #frame{display:block;height:calc(100vh - 52px)}
 #side{border-left:1px solid var(--line);padding:16px;overflow:auto;background:#fff}
 .url{word-break:break-all;font-size:12px;color:var(--accent)} .reg{display:inline-block;margin:8px 0;padding:2px 8px;background:#eef;border-radius:10px;font-size:12px}
 .cur{margin:10px 0;padding:8px;border:1px dashed var(--line);border-radius:6px;font-size:13px}
 .btns{display:flex;flex-direction:column;gap:8px;margin-top:14px}
 button.lab{padding:10px;border:1px solid var(--line);border-radius:8px;background:#fff;cursor:pointer;font-size:14px;text-align:left}
 button.lab.on{color:#fff;border-color:transparent} .keep.on{background:var(--keep)} .wkeep.on{background:var(--wkeep)} .wdrop.on{background:var(--wdrop)} .drop.on{background:var(--drop)}
 #exp{margin-left:auto;padding:7px 14px;background:var(--accent);color:#fff;border:0;border-radius:8px;cursor:pointer;font-size:14px}
 #rawtog{padding:6px 10px;border:1px solid var(--line);border-radius:8px;background:#fff;cursor:pointer;font-size:13px}
 .note{background:#fff7e6;border:1px solid #f0d9a8;border-radius:6px;padding:8px;font-size:12px;margin-bottom:10px}
</style></head><body>
<header>
 <b>Relabel unverified snapshots</b>
 <span class="mut">doc <b id="idx">1</b>/<b id="tot">0</b> · relabeled <b id="done">0</b></span>
 <span class="mut">keys: <kbd>k</kbd>keep <kbd>w</kbd>weak-keep <kbd>x</kbd>weak-drop <kbd>d</kbd>drop <kbd>⏎</kbd>accept current <kbd>←</kbd><kbd>→</kbd> <kbd>r</kbd>raw</span>
 <button id="rawtog">show raw HTML (r)</button>
 <button id="exp">⬇ export relabel.json</button>
</header>
<div id="wrap">
 <div id="docpane">
   <div id="text"></div>
   <iframe id="frame" sandbox referrerpolicy="no-referrer"></iframe>
 </div>
 <div id="side">
   <div class="note">Recovered snapshot couldn't be fingerprint-verified — <b>label what this page actually is</b> (resiliparse main-content text; toggle raw HTML with <kbd>r</kbd>).</div>
   <div class="url" id="url"></div>
   <span class="reg" id="reg"></span>
   <div class="cur" id="cur"></div>
   <div class="btns">
     <button class="lab keep" data-l="keep" id="bk">Keep <span class="mut">(k)</span></button>
     <button class="lab wkeep" data-l="weak_keep" id="bw">Weak keep <span class="mut">(w)</span></button>
     <button class="lab wdrop" data-l="weak_drop" id="bx">Weak drop <span class="mut">(x)</span></button>
     <button class="lab drop" data-l="drop" id="bd">Drop <span class="mut">(d)</span></button>
   </div>
 </div>
</div>
<script>
const DOCS = __DOCS__;
const KEY = "relabel_unverified_v1";
let labels = {}; try{ labels = JSON.parse(localStorage.getItem(KEY)||"{}"); }catch(e){}
let raw = false;
let i = DOCS.findIndex(d => !labels[d.url]);  // resume at the first not-yet-relabeled doc
if(i < 0) i = 0;
const $ = id => document.getElementById(id);
$("tot").textContent = DOCS.length;
function save(){ localStorage.setItem(KEY, JSON.stringify(labels)); }
function render(){
  const d = DOCS[i];
  $("text").textContent = d.text || "(no extractable text — toggle raw HTML with r)";
  $("frame").srcdoc = d.html;
  $("docpane").classList.toggle("raw", raw);
  $("docpane").scrollTop = 0;
  $("url").textContent = d.url;
  $("reg").textContent = d.register;
  const chosen = labels[d.url] || d.cur;
  $("cur").innerHTML = "current: <b>"+(d.cur||"—")+"</b> <span class=mut>("+d.cur_src+")</span>"+
      (labels[d.url] ? " → relabeled <b>"+labels[d.url]+"</b>" : "");
  document.querySelectorAll("button.lab").forEach(b=>b.classList.toggle("on", b.dataset.l===chosen));
  $("idx").textContent = i+1;
  $("done").textContent = Object.keys(labels).length;
}
function setLabel(l){ labels[DOCS[i].url] = l; save(); if(i<DOCS.length-1) i++; render(); }
function go(n){ i = Math.max(0, Math.min(DOCS.length-1, n)); render(); }
document.querySelectorAll("button.lab").forEach(b=> b.onclick = ()=>setLabel(b.dataset.l));
$("rawtog").onclick = ()=>{ raw=!raw; render(); };
document.onkeydown = e => {
  if(e.key==="k")setLabel("keep"); else if(e.key==="w")setLabel("weak_keep");
  else if(e.key==="x")setLabel("weak_drop"); else if(e.key==="d")setLabel("drop");
  else if(e.key==="Enter"){ const c = labels[DOCS[i].url] || DOCS[i].cur; if(c) setLabel(c); }  // accept shown label
  else if(e.key==="r"){ raw=!raw; render(); }
  else if(e.key==="ArrowRight")go(i+1); else if(e.key==="ArrowLeft")go(i-1);
};
$("exp").onclick = ()=>{
  const out = {}; for(const u in labels) out[u] = {label: labels[u]};
  const blob = new Blob([JSON.stringify(out,null,2)],{type:"application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "relabel.json"; a.click();
};
render();
</script></body></html>"""


if __name__ == "__main__":
    sys.exit(main())
