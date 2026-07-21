# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Build a single self-contained HTML viewer (<=6 MB) for the labeled dev set + the random-250 set.

Read-only: browse docs, sort/filter by register, filter by each pipeline's keep decision (see the
DCLM-vs-hq etc. tradeoffs), and read the gold reference extraction where one exists. To stay under the
size budget it embeds resiliparse main-content TEXT (not raw HTML), truncated, and SAMPLES the dev set
per register while keeping every gold doc.

  python -m experiments.baseline_collection.build_devset_viewer [--per-register 22] [--text-cap 2800] [--gold-cap 4000]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections import Counter

from experiments.baseline_collection.build_relabel_interface import _readable_text

logger = logging.getLogger(__name__)

DEVSET = "scratch/devset_export/devset_1934_html"
_SR = "/Users/michaelryan/Documents/School/Stanford/Research/small-rephraser/static/warcs"
PIPELINE = f"{_SR}/marin_devset_1934_html/pipeline_labels.jsonl"  # written by devset_pipeline_f1 build
RANDOM = f"{_SR}/marin_random_labeled_html"
OUT = f"{DEVSET}/viewer.html"
PIPES = ("hq", "dclm", "nemo", "fwedu", "fwcc")


def _rank(url: str) -> int:
    return int(hashlib.md5(url.encode()).hexdigest(), 16)


def _text(folder: str, html_file: str, cap: int) -> str:
    t = _readable_text(open(os.path.join(folder, html_file)).read())
    return t[:cap]


def run(per_register: int, text_cap: int, gold_cap: int) -> None:
    dev = {json.loads(line)["url"]: json.loads(line) for line in open(f"{DEVSET}/devset.jsonl")}
    flags = {json.loads(line)["url"]: json.loads(line) for line in open(PIPELINE)}

    # sample: keep every gold doc, PLUS up to per_register non-gold per register (lowest-hash) so the
    # pipeline-drop cases show up too, not just the keep-heavy gold docs.
    by_reg: dict[str, list] = {}
    for u, d in dev.items():
        by_reg.setdefault(d["register"], []).append(d)
    chosen = []
    for reg, docs in by_reg.items():
        gold = [d for d in docs if d["gold"]]
        rest = sorted((d for d in docs if not d["gold"]), key=lambda d: _rank(d["url"]))
        chosen += gold + rest[:per_register]
    logger.info("sampled %d dev docs (of %d) across %d registers", len(chosen), len(dev), len(by_reg))

    out = []
    for d in chosen:
        f = flags.get(d["url"], {})
        gold_txt = None
        if d["gold"]:
            gp = os.path.join(DEVSET, "gold", f"{d['hid']}.txt")
            if os.path.exists(gp):
                gold_txt = open(gp).read()[:gold_cap]
        out.append(
            {
                "set": "dev",
                "url": d["url"],
                "register": d["register"],
                "label": d["label"],
                "src": d["label_source"],
                "kept": {p: bool(f.get(f"kept_{p}")) for p in PIPES} if f.get("in_membership") else None,
                "dclm_ft": f.get("dclm_ft"),
                "nemo": f.get("nemo_quality"),
                "fw": f.get("fineweb_score"),
                "text": _text(DEVSET, d["html_file"], text_cap),
                "gold": gold_txt,
            }
        )

    rnd = [json.loads(line) for line in open(f"{RANDOM}/random_labeled.jsonl")]
    for r in rnd:
        out.append(
            {
                "set": "random",
                "url": r["url"],
                "register": r["register_guess"],
                "label": r["label"],
                "src": "hand",
                "year": r["year"],
                "kept": None,
                "text": _text(RANDOM, r["html_file"], text_cap),
                "gold": None,
            }
        )

    payload = json.dumps(out).replace("</", "<\\/")
    html = _TEMPLATE.replace("__DOCS__", payload)
    open(OUT, "w").write(html)
    mb = len(html.encode()) / 1e6
    ng = sum(1 for d in out if d["gold"])
    logger.info("wrote %s — %d docs (%d dev, %d random, %d gold), %.2f MB", OUT, len(out), len(chosen), len(rnd), ng, mb)
    print(
        json.dumps(
            {
                "docs": len(out),
                "gold": ng,
                "MB": round(mb, 2),
                "registers": dict(Counter(d["register"] for d in out).most_common()),
            },
            indent=1,
        )
    )
    if mb > 6:
        logger.warning("OVER 6 MB — lower --per-register / --text-cap / --gold-cap")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-register", type=int, default=22)
    ap.add_argument("--text-cap", type=int, default=2800)
    ap.add_argument("--gold-cap", type=int, default=4000)
    args = ap.parse_args()
    run(args.per_register, args.text_cap, args.gold_cap)
    return 0


_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Marin dev-set viewer</title>
<style>
 :root{--bg:#fbfaf8;--panel:#fff;--fg:#1c1b19;--mut:#6c6a66;--line:#e6e3dd;--accent:#3a6ea5;--keep:#1a7f4b;--drop:#c0392b;--chip:#eef1f5}
 @media (prefers-color-scheme:dark){:root{--bg:#1a1918;--panel:#232120;--fg:#e9e6e1;--mut:#9b9791;--line:#37342f;--accent:#7aa7d6;--chip:#2b2927}}
 :root[data-theme=dark]{--bg:#1a1918;--panel:#232120;--fg:#e9e6e1;--mut:#9b9791;--line:#37342f;--accent:#7aa7d6;--chip:#2b2927}
 :root[data-theme=light]{--bg:#fbfaf8;--panel:#fff;--fg:#1c1b19;--mut:#6c6a66;--line:#e6e3dd;--accent:#3a6ea5;--chip:#eef1f5}
 *{box-sizing:border-box} body{margin:0;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:var(--bg);color:var(--fg)}
 header{position:sticky;top:0;z-index:5;background:var(--panel);border-bottom:1px solid var(--line);padding:8px 14px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
 header b{font-size:15px;letter-spacing:.02em} .mut{color:var(--mut)}
 select,input{font:inherit;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:7px;padding:5px 8px}
 .seg{display:inline-flex;border:1px solid var(--line);border-radius:7px;overflow:hidden}
 .seg button{border:0;background:var(--panel);color:var(--fg);padding:5px 11px;cursor:pointer;font:inherit}
 .seg button.on{background:var(--accent);color:#fff}
 .pipe{display:inline-flex;align-items:center;gap:4px;border:1px solid var(--line);border-radius:20px;padding:2px 4px 2px 9px;font-size:12px}
 .pipe .t{border:0;border-radius:14px;padding:2px 7px;cursor:pointer;font:inherit;font-size:11px;background:var(--chip);color:var(--fg)}
 .pipe .t.keep{background:var(--keep);color:#fff} .pipe .t.drop{background:var(--drop);color:#fff}
 #main{display:grid;grid-template-columns:minmax(300px,380px) 1fr;height:calc(100vh - 49px)}
 #list{overflow:auto;border-right:1px solid var(--line)}
 .row{padding:8px 12px;border-bottom:1px solid var(--line);cursor:pointer} .row:hover{background:var(--chip)} .row.sel{background:var(--chip);box-shadow:inset 3px 0 0 var(--accent)}
 .row .u{font-size:12px;color:var(--accent);word-break:break-all;line-height:1.35}
 .chip{display:inline-block;padding:1px 7px;border-radius:10px;background:var(--chip);font-size:11px;margin:3px 4px 0 0}
 .kb{display:inline-block;width:15px;text-align:center;border-radius:4px;font-size:10px;font-weight:600;margin-right:2px}
 .kb.y{background:var(--keep);color:#fff} .kb.n{background:transparent;color:var(--mut);border:1px solid var(--line)}
 #detail{overflow:auto;padding:18px 24px} #detail h2{font-size:14px;margin:0 0 4px}
 #detail .url{font-size:13px;color:var(--accent);word-break:break-all}
 .meta{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0;align-items:center}
 table.k{border-collapse:collapse;margin:8px 0;font-size:12px} table.k td{border:1px solid var(--line);padding:3px 9px} table.k .v{font-weight:600}
 .cols{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:12px} .cols.single{grid-template-columns:1fr}
 .doc{white-space:pre-wrap;font:13px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:12px;max-height:none;overflow-wrap:anywhere}
 .lbl{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--mut);margin-bottom:4px}
 .trunc{color:var(--mut);font-size:11px;font-style:italic}
 #count{margin-left:auto} #theme{cursor:pointer;border:1px solid var(--line);background:var(--panel);color:var(--fg);border-radius:7px;padding:5px 9px}
</style></head><body>
<header>
 <b>Marin dev-set viewer</b>
 <span class="seg" id="setseg"><button data-s="dev" class="on">Dev set</button><button data-s="random">Random 250</button></span>
 <label>register <select id="reg"></select></label>
 <label>sort <select id="sort"><option value="register">register</option><option value="dclm_ft">DCLM score</option><option value="fw">FineWeb score</option><option value="nemo">Nemotron score</option></select></label>
 <span id="pipes"></span>
 <input id="q" placeholder="search url/text" size="16">
 <span class="mut" id="count"></span>
 <button id="theme" title="toggle theme">◐</button>
</header>
<div id="main">
 <div id="list"></div>
 <div id="detail"><p class="mut">Select a document.</p></div>
</div>
<script>
const DOCS = __DOCS__;
const PIPES = ["hq","dclm","nemo","fwedu","fwcc"];
const $=id=>document.getElementById(id);
let curSet="dev", reg="all", sort="register", q="", sel=null;
const pipeState={}; PIPES.forEach(p=>pipeState[p]="any");  // any|keep|drop
const KEEP=l=>l==="keep"||l==="weak_keep";

// register dropdown (per set)
function fillReg(){
  const regs=[...new Set(DOCS.filter(d=>d.set===curSet).map(d=>d.register))].sort();
  $("reg").innerHTML='<option value="all">all ('+DOCS.filter(d=>d.set===curSet).length+')</option>'+regs.map(r=>'<option>'+r+'</option>').join("");
  reg="all";
}
// pipeline filter pills
$("pipes").innerHTML=PIPES.map(p=>'<span class="pipe">'+p.toUpperCase()+' <button class="t" data-p="'+p+'">any</button></span>').join(" ");
$("pipes").querySelectorAll("button.t").forEach(b=>b.onclick=()=>{
  const p=b.dataset.p, order={any:"keep",keep:"drop",drop:"any"};
  pipeState[p]=order[pipeState[p]]; b.textContent=pipeState[p]; b.className="t "+(pipeState[p]==="any"?"":pipeState[p]); render();
});
function match(d){
  if(d.set!==curSet) return false;
  if(reg!=="all"&&d.register!==reg) return false;
  if(q){const s=(d.url+" "+d.text).toLowerCase(); if(!s.includes(q)) return false;}
  for(const p of PIPES){ if(pipeState[p]==="any") continue; if(!d.kept) return false;
    if(pipeState[p]==="keep"&&!d.kept[p]) return false; if(pipeState[p]==="drop"&&d.kept[p]) return false; }
  return true;
}
function kbadges(d){ if(!d.kept) return '<span class=mut style="font-size:11px">not in 10k pool</span>';
  return PIPES.map(p=>'<span class="kb '+(d.kept[p]?'y':'n')+'" title="'+p+'">'+p[0].toUpperCase()+'</span>').join(""); }
function render(){
  let v=DOCS.filter(match);
  if(sort==="register") v.sort((a,b)=>a.register<b.register?-1:a.register>b.register?1:0);
  else v.sort((a,b)=>(b[sort]??-1)-(a[sort]??-1));
  $("count").textContent=v.length+" docs";
  $("list").innerHTML=v.map((d,i)=>{
    const gi=DOCS.indexOf(d);
    return '<div class="row'+(gi===sel?' sel':'')+'" data-i="'+gi+'">'+
      '<div>'+kbadges(d)+' <span class="chip">'+d.register+'</span> <span class="chip" style="background:'+(KEEP(d.label)?'var(--keep)':'var(--drop)')+';color:#fff">'+(d.label||'—')+'</span>'+(d.gold?' <span class="chip" style="background:#b8860b;color:#fff">gold</span>':'')+'</div>'+
      '<div class="u">'+d.url+'</div></div>';
  }).join("")||'<p class="mut" style="padding:16px">No documents match.</p>';
  $("list").querySelectorAll(".row").forEach(r=>r.onclick=()=>{sel=+r.dataset.i; render(); detail(DOCS[sel]);});
}
function scoreRow(d){
  if(!d.kept) return '<tr><td colspan=2 class=mut>Sampled past the 10k cutoff — no pipeline decisions.</td></tr>';
  return PIPES.map(p=>{const s=p==="dclm"?d.dclm_ft:p==="fwedu"?d.fw:p==="nemo"?d.nemo:null;
    return '<tr><td>'+p.toUpperCase()+'</td><td class="v" style="color:'+(d.kept[p]?'var(--keep)':'var(--drop)')+'">'+(d.kept[p]?'KEEP':'drop')+(s!=null?' <span class=mut>('+(+s).toFixed(3)+')</span>':'')+'</td></tr>';}).join("");
}
function detail(d){
  const gold=d.gold;
  $("detail").innerHTML=
    '<div class="url"><a href="'+d.url+'" target="_blank" rel="noreferrer" style="color:var(--accent)">'+d.url+'</a></div>'+
    '<div class="meta"><span class="chip">'+d.register+'</span>'+
      '<span class="chip" style="background:'+(KEEP(d.label)?'var(--keep)':'var(--drop)')+';color:#fff">'+(d.label||'—')+' · '+d.src+'</span>'+
      (d.year?'<span class="chip">'+d.year+'</span>':'')+(gold?'<span class="chip" style="background:#b8860b;color:#fff">gold extraction</span>':'')+'</div>'+
    '<table class="k"><tr><td class=mut>pipeline</td><td class=mut>decision (raw score)</td></tr>'+scoreRow(d)+'</table>'+
    '<div class="cols'+(gold?'':' single')+'">'+
      '<div><div class="lbl">Extracted text (resiliparse, preview)</div><div class="doc">'+esc(d.text)+'<div class="trunc">… preview truncated …</div></div></div>'+
      (gold?'<div><div class="lbl">Gold reference extraction</div><div class="doc">'+esc(gold)+'<div class="trunc">… preview truncated …</div></div></div>':'')+
    '</div>';
}
function esc(s){return (s||"").replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
$("setseg").querySelectorAll("button").forEach(b=>b.onclick=()=>{$("setseg").querySelectorAll("button").forEach(x=>x.classList.remove("on"));b.classList.add("on");curSet=b.dataset.s;sel=null;fillReg();$("detail").innerHTML='<p class="mut">Select a document.</p>';render();});
$("reg").onchange=e=>{reg=e.target.value;render();};
$("sort").onchange=e=>{sort=e.target.value;render();};
$("q").oninput=e=>{q=e.target.value.toLowerCase();render();};
$("theme").onclick=()=>{const r=document.documentElement,d=r.getAttribute("data-theme")==="dark";r.setAttribute("data-theme",d?"light":"dark");};
fillReg(); render();
</script></body></html>"""


if __name__ == "__main__":
    sys.exit(main())
