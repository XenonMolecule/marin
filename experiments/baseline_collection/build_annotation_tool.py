# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Build a self-contained browser annotation tool for the hard-label keep/drop pass.

Reads the 0.8-threshold hard-sample + per-pipeline extractions (hq/dclm/nemo/resiliparse) +
the eval prompts/keywords, and emits a single static HTML file with the data embedded. The
tool runs entirely in the browser: it shows one doc at a time with the eval item it matched,
the matched keywords highlighted, and the extraction (priority hq -> dclm -> nemo -> resiliparse),
and lets the user label keep/drop/unsure with keyboard shortcuts. Progress is saved to the
browser's localStorage on every action, so closing the tab/window loses nothing. A button
exports the labels as JSON (later: gold valset + soft-verifier seed).

Run locally after hard-sample + fetch-extractions land. Writes scratch/annotate.html.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import pyarrow.parquet as pq

PRIORITY = ["hq", "dclm", "nemo", "resiliparse"]  # extraction display order (best first)
EXTRACT_CAP = 8000  # chars per extraction shown

# Register dropdown (re-bin target). Grouped: topic registers align with LLM-eval / MMLU domains,
# format registers describe page shape (what hq's spec actually keys on). Free-text "+ new register"
# in the tool covers anything the random pool surfaces that isn't here.
REGISTER_GROUPS: dict[str, list[str]] = {
    "topic (eval / MMLU)": [
        "math",
        "science",
        "medical",
        "arxiv",
        "educational",
        "test_prep",
        "history",
        "philosophy_religion",
        "legal",
        "business_econ",
        "social_science",
        "arts_humanities",
    ],
    "format": [
        "fiction",
        "poetry",
        "qa_forum",
        "howto",
        "reference_expository",
        "news",
        "code",
        "tabular",
        "product_commerce",
        "lifestyle",
        "nav_index",
    ],
    "negative / misc": ["junk", "random", "other"],
}


def _load_extractions(ext_dir: str) -> dict[str, dict[str, str]]:
    """method -> {url -> text} from HARD_EXTRACTIONS/{method}/*.parquet pulled locally."""
    out: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for method in PRIORITY:
        for p in glob.glob(f"{ext_dir}/{method}/*.parquet"):
            for r in pq.read_table(p).to_pylist():
                if r.get("url") and r.get("full_text"):
                    out[method][r["url"]] = r["full_text"][:EXTRACT_CAP]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="/tmp/hard_sample.parquet")
    ap.add_argument("--extractions", default="/tmp/hard_extractions")
    ap.add_argument("--examples", default="/tmp/keyword_examples.json")
    ap.add_argument("--hqworse", default="/tmp/hqworse.parquet")
    ap.add_argument("--broad", default="scratch/eval_attribution/keywords_broad.json")
    ap.add_argument("--random-sample", default="/tmp/random_sample.parquet")
    ap.add_argument("--random-ext", default="/tmp/random_resiliparse.parquet")
    ap.add_argument("--register-overrides", default="/tmp/register_overrides.json")
    ap.add_argument("--neg-sample", default="/tmp/neg_sample.parquet")
    ap.add_argument("--neg-text", default="/tmp/neg_text.parquet")
    ap.add_argument("--judge-overlay", default="/tmp/devset_judge_overlay.json")
    ap.add_argument("--out", default="scratch/annotate.html")
    args = ap.parse_args()

    # LLM-judge suggestions (url -> {label, confidence}); shown as read-only hints, NOT the user's
    # labels. High-confidence ones are auto-accepted separately; medium/low form the review queue.
    judge = json.load(open(args.judge_overlay)) if os.path.exists(args.judge_overlay) else {}

    # Content-based register overrides (url -> register) from the LLM reclassification pass; these
    # replace the domain-based guess (which mis-bins platform domains like geeksforgeeks/gradesaver).
    reg_override: dict[str, str] = {}
    if os.path.exists(args.register_overrides):
        reg_override = json.load(open(args.register_overrides))

    subj2kws = {e["subject"]: e["kws"] for e in json.load(open(args.examples))["examples"]}
    hqw = pq.read_table(args.hqworse, columns=["task", "idx", "text"]).to_pylist()
    tit = {(r["task"], str(r["idx"])): r["text"] for r in hqw}
    subj2prompt: dict[str, str] = {}
    for rec in json.load(open(args.broad))["records"]:
        t = tit.get((rec.get("task"), str(rec.get("idx"))))
        if t:
            subj2prompt.setdefault(rec["subject"], t)

    ext = _load_extractions(args.extractions)
    docs = []
    # Matched pool: group AND guess by the corrected register (LLM/WARC-shape override) when present,
    # else the domain-based category — so filtering by register shows all docs of that register.
    for r in pq.read_table(args.sample).to_pylist():
        subjects = [s for s in (r.get("subjects") or "").split("|") if s]
        kws = sorted({k for s in subjects for k in subj2kws.get(s, [])})
        cat = reg_override.get(r["url"], r.get("category", "?"))
        docs.append(
            {
                "url": r["url"],
                "domain": r.get("domain", ""),
                "category": cat,
                "regGuess": cat,
                "snapshot": "",
                "keptHq": int(r.get("kept_hq", 0) or 0),
                "verifier": r.get("verifier_score", 0),
                "frac": round(r.get("best_frac", 0) or 0, 2),
                "subjects": subjects[:5],
                "prompts": [subj2prompt.get(s, "") for s in subjects[:5]],
                "kws": kws,
                "ext": {m: ext.get(m, {}).get(r["url"], "") for m in PRIORITY},
            }
        )
    # Random pool: uniform 2013-2026 timespan draw, resiliparse-only, unbiased anchor for the policy.
    n_random = 0
    if os.path.exists(args.random_sample) and os.path.exists(args.random_ext):
        rtext = {r["url"]: (r.get("full_text") or "")[:EXTRACT_CAP] for r in pq.read_table(args.random_ext).to_pylist()}
        for r in pq.read_table(args.random_sample).to_pylist():
            docs.append(
                {
                    "url": r["url"],
                    "domain": r.get("domain", ""),
                    "category": "random",
                    "regGuess": reg_override.get(r["url"], r.get("reg_guess", "other")),
                    "snapshot": r.get("snapshot", ""),
                    "keptHq": None,
                    "verifier": None,
                    "frac": 0,
                    "subjects": [],
                    "prompts": [],
                    "kws": [],
                    "ext": {m: rtext.get(r["url"], "") if m == "resiliparse" else "" for m in PRIORITY},
                }
            )
            n_random += 1
    # Negatives pool: mined hard-negatives (drop-worthy) grouped by their register, so no register is
    # 'keep-all'. hq-kept ones are flagged (hq false-positives — bad content hq let through).
    n_neg = 0
    if os.path.exists(args.neg_sample) and os.path.exists(args.neg_text):
        # negatives keep their FULL text (esp. code — the whole source matters); the doc pane scrolls.
        gtext = {r["url"]: (r.get("full_text") or "")[:40000] for r in pq.read_table(args.neg_text).to_pylist()}
        for r in pq.read_table(args.neg_sample).to_pylist():
            reg = reg_override.get(r["url"], r.get("register", "other"))
            fp = bool(r.get("kept_hq"))
            docs.append(
                {
                    "url": r["url"],
                    "domain": r.get("domain", ""),
                    "category": reg,
                    "regGuess": reg,
                    "snapshot": "",
                    "keptHq": 1 if fp else None,
                    "verifier": r.get("verifier") if fp else None,
                    "frac": 0,
                    "subjects": [],
                    "prompts": [],
                    "kws": [],
                    "ext": {m: gtext.get(r["url"], "") if m == "resiliparse" else "" for m in PRIORITY},
                }
            )
            n_neg += 1
    # Sort by category then verifier tier; random (verifier None) sorts to its own group at the end.
    docs.sort(key=lambda d: (d["category"], -(d["verifier"] if d["verifier"] is not None else -1)))
    _write_html(docs, args.out, judge)
    print(
        f"wrote {args.out} ({len(docs)} docs: {len(docs) - n_random - n_neg} matched + {n_random} random "
        f"+ {n_neg} negatives; {sum(1 for d in docs if any(d['ext'].values()))} with >=1 extraction)"
    )
    return 0


def _write_html(docs: list[dict], out: str, judge: dict | None = None) -> None:
    data = json.dumps(docs).replace("</", "<\\/")
    html = (
        _TEMPLATE.replace("__DATA__", data)
        .replace("__REGISTERS__", json.dumps(REGISTER_GROUPS))
        .replace("__JUDGE__", json.dumps(judge or {}).replace("</", "<\\/"))
    )
    with open(out, "w") as f:
        f.write(html)


_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Dev-set hard labels</title>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#e2e2e2;--keep:#2a9d4a;--wkeep:#7cc98d;--wdrop:#e0917f;--drop:#c0392b;--uns:#b8860b;--accent:#2f6fed}
*{box-sizing:border-box}body{font:15px/1.55 -apple-system,BlinkMacSystemFont,sans-serif;color:var(--fg);background:var(--bg);margin:0;transition:background .3s ease,box-shadow .3s ease}
body.over{background:#ffb3b3;box-shadow:inset 0 0 0 6px #e53935}body.over #bar{background:#ffd6d6}
#bar{position:sticky;top:0;background:#fafafa;border-bottom:1px solid var(--line);padding:.6rem 1rem;display:flex;gap:1rem;align-items:center;flex-wrap:wrap;z-index:5}
#bar b{font-variant-numeric:tabular-nums}#btimer.on{background:var(--accent);color:#fff;border-color:var(--accent)}#tstat b{color:var(--fg)}.prog{height:8px;background:#eee;border-radius:4px;flex:1;min-width:120px;overflow:hidden}.prog>i{display:block;height:100%;background:var(--keep)}
main{max-width:900px;margin:1.2rem auto;padding:0 1rem}
.meta{color:var(--mut);font-size:13px}.pill{display:inline-block;padding:1px 8px;border-radius:10px;background:#eef;color:#334;font-size:12px;margin-right:.3rem}
.evalq{margin:.6rem 0;padding:.5rem .7rem;background:#eef4ff;border-left:3px solid var(--accent);font-size:14px}
.kw{margin:.4rem 0;font-size:13px;color:#555}.kw mark{background:#fde68a}
.tabs{margin-top:.8rem;border-bottom:1px solid var(--line)}.tabs button{border:0;background:none;padding:.4rem .8rem;cursor:pointer;font-size:14px;color:var(--mut);border-bottom:2px solid transparent}
.tabs button.on{color:var(--fg);border-bottom-color:var(--accent);font-weight:600}.tabs button.empty{opacity:.35}
.doc{white-space:pre-wrap;background:#fbfbfb;border:1px solid var(--line);border-radius:6px;padding:.8rem 1rem;margin-top:.4rem;max-height:52vh;overflow:auto}.doc mark{background:#fde68a}
.acts{position:sticky;bottom:0;background:var(--bg);padding:.8rem 0;display:flex;gap:.6rem;align-items:center;border-top:1px solid var(--line);margin-top:1rem}
.acts button{font-size:15px;padding:.5rem 1.1rem;border-radius:8px;border:1px solid var(--line);cursor:pointer;background:#fff}
.acts .top.on{background:#e8a800;color:#3a2c00;border-color:#e8a800;font-weight:700}
.acts .flag.on{background:#7a4de0;color:#fff;border-color:#7a4de0}
.acts .keep.on{background:var(--keep);color:#fff;border-color:var(--keep)}.acts .wkeep.on{background:var(--wkeep);color:#0a2e15;border-color:var(--wkeep)}.acts .wdrop.on{background:var(--wdrop);color:#3a0f08;border-color:var(--wdrop)}.acts .drop.on{background:var(--drop);color:#fff;border-color:var(--drop)}.acts .uns.on{background:var(--uns);color:#fff;border-color:var(--uns)}
input#notes{flex:1;padding:.45rem .6rem;border:1px solid var(--line);border-radius:6px;font-size:14px}
select,#bar button{padding:.35rem .5rem;border:1px solid var(--line);border-radius:6px;background:#fff;cursor:pointer}
kbd{background:#eee;border-radius:4px;padding:0 4px;font-size:12px}
</style></head><body>
<div id="bar">
  <span>doc <b id="pos">0</b>/<b id="tot">0</b></span>
  <span>labeled <b id="done">0</b> (<b id="pct">0</b>%)</span>
  <span>★ <b id="ntop">0</b></span>
  <span>⚑ <b id="nflag">0</b></span>
  <div class="prog"><i id="pbar"></i></div>
  <label>category <select id="fcat"></select></label>
  <label><input type="checkbox" id="unl"> unlabeled only</label>
  <label title="LLM-judge medium/low-confidence docs that need your review (high-confidence were auto-accepted)"><input type="checkbox" id="rq"> 🤖 review queue (<b id="nrq">0</b>)</label>
  <label title="when to reveal the LLM-judge suggestion badge — 'after I label' hides it until you commit, so it can't bias your call but you still see the agreement">🤖 suggestions <select id="jmode"><option value="always">always</option><option value="after">after I label</option><option value="never">never</option></select></label>
  <button id="exp">⬇ export labels</button>
  <button id="btimer" title="pace timer: per-doc stopwatch + ETA averaged over your last 25 docs">⏱ timer</button>
  <span id="tstat" class="meta" style="display:none"></span>
  <span class="meta">keys: <kbd>k</kbd>keep <kbd>w</kbd>weak-keep <kbd>x</kbd>weak-drop <kbd>d</kbd>drop <kbd>u</kbd>unsure <kbd>t</kbd>★top <kbd>f</kbd>⚑extractor <kbd>→</kbd>next <kbd>←</kbd>prev</span>
</div>
<main>
  <p class="meta">Blind pass: judge the <b>document content</b> only. The <b>source (url/domain), eval item, verifier tier, and which pipelines kept it</b> stay hidden until you label — only our <b>register guess</b> is shown (correct it if wrong; the source is hidden because the pipelines are domain-driven, so seeing the site would leak their decision). Everything reveals after you label, and you can click <kbd>←</kbd> back to inspect. Use <b>weak keep/drop</b> for borderline docs. The <b>random</b> category is an unbiased 2013-2026 web sample (no pipeline ever scored it) — labeling it sets the policy for natural web.</p>
  <div id="head"></div>
  <div id="jbox"></div>
  <div id="evalbox"></div>
  <div class="kw" id="kwbox"></div>
  <div class="tabs" id="tabs"></div>
  <div class="doc" id="body"></div>
  <div class="acts">
    <button class="keep" id="bk">Keep (k)</button>
    <button class="wkeep" id="bwk">Weak keep (w)</button>
    <button class="wdrop" id="bwd">Weak drop (x)</button>
    <button class="drop" id="bd">Drop (d)</button>
    <button class="uns" id="bu">Unsure (u)</button>
    <button class="top" id="btop" title="mark as a TOP-QUALITY exemplar — amazing content, much better than a normal keep (t)">★ top quality (t)</button>
    <button class="flag" id="bflag" title="flag as an extractor-quality benchmark edge case (f)">⚑ extractor case (f)</button>
    <label class="regwrap">register (fix if wrong) <select id="reg"></select></label>
    <input id="notes" placeholder="optional note…">
    <button id="bprev">← prev</button><button id="bnext">next →</button>
  </div>
</main>
<script>
const DOCS = __DATA__;
const PRIORITY = ["hq","dclm","nemo","resiliparse"];
const REGISTER_GROUPS = __REGISTERS__;  // {group: [register,...]} for the dropdown
const JUDGE = __JUDGE__;  // url -> {label, confidence}: LLM-judge suggestions (read-only, not your labels)
function reviewNeeded(url){ const j=JUDGE[url]; return j && j.confidence!=="high"; }  // medium/low = review queue
const KEY = "devset_hard_labels_v1";
let labels = {}; try{ labels = JSON.parse(localStorage.getItem(KEY)||"{}"); }catch(e){}
let idx = 0, tab = null, fcat = "all", unlOnly = false, rqOnly = false;
// LLM-judge badge visibility: "always" | "after" (reveal only once this doc is labeled) | "never".
let judgeMode = "always"; try{ judgeMode = JSON.parse(localStorage.getItem(KEY+"_judgemode")||"null") || (JSON.parse(localStorage.getItem(KEY+"_hidejudge")||"false")?"never":"always"); }catch(e){}
function judgeVisible(d){ return judgeMode==="always" || (judgeMode==="after" && !!(labels[d.url]&&labels[d.url].label)); }
let peeked = new Set();
// Pace timer: per-doc stopwatch + ETA from a rolling window of the last 25 doc-view durations.
const OVER_SEC = 5;  // per-doc pace target; over this, a subtle red background nudge (timer mode only)
let timerOn = false; try{ timerOn = JSON.parse(localStorage.getItem(KEY+"_timer")||"false"); }catch(e){}
let timerUrl = null, timerStart = Date.now(), docTimes = [];
function fmtDur(s){ s=Math.round(s); if(s<60) return s+"s"; const m=Math.floor(s/60); if(m<60) return m+"m "+(s%60)+"s"; return Math.floor(m/60)+"h "+(m%60)+"m"; }
function onDocShown(url){
  if(url===timerUrl) return;  // only a genuine doc change counts
  if(timerOn && timerUrl!==null){
    const dt=(Date.now()-timerStart)/1000;
    if(dt>=0.3 && dt<=600){ docTimes.push(dt); if(docTimes.length>25) docTimes.shift(); }  // skip idle outliers
  }
  timerUrl=url; timerStart=Date.now();
}
function updateTimer(){
  const ts=$("tstat");
  if(!timerOn){ ts.style.display="none"; document.body.classList.remove("over"); return; }
  ts.style.display="";
  const cur=(Date.now()-timerStart)/1000;
  document.body.classList.toggle("over", cur>OVER_SEC);  // subtle red nudge when over pace (timer mode only)
  const avg=docTimes.length? docTimes.reduce((a,b)=>a+b,0)/docTimes.length : 0;
  const done=Object.keys(labels).filter(u=>labels[u].label).length, remain=DOCS.length-done;
  ts.innerHTML = `⏱ <b>${Math.round(cur)}s</b> here · avg <b>${avg?avg.toFixed(1):"-"}s</b>/doc (last ${docTimes.length}) · <b>${remain}</b> left &rarr; ETA <b>${avg?fmtDur(avg*remain):"-"}</b>`;
}
function revealed(d){ return !!(labels[d.url] && labels[d.url].label) || peeked.has(d.url); }
const $ = id => document.getElementById(id);
function save(){ localStorage.setItem(KEY, JSON.stringify(labels)); }
function esc(s){ return (s||"").replace(/[&<>]/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }
function hl(s, kws){ s = esc(s); for(const k of [...kws].sort((a,b)=>b.length-a.length)){ if(!k) continue; const re = new RegExp("(?<![a-z0-9])("+k.replace(/[.*+?^${}()|[\]\\]/g,"\\$&")+")(?![a-z0-9])","ig"); s = s.replace(re,"<mark>$1</mark>"); } return s; }
function view(){ let v = DOCS.map((d,i)=>({d,i})); if(fcat!=="all") v = v.filter(x=>x.d.category===fcat); if(unlOnly) v = v.filter(x=>!(labels[x.d.url]&&labels[x.d.url].label)); if(rqOnly) v = v.filter(x=>reviewNeeded(x.d.url) && !(labels[x.d.url]&&labels[x.d.url].label)); return v; }
function cur(){ const v = view(); if(!v.length) return null; idx = Math.max(0, Math.min(idx, v.length-1)); return v[idx]; }
function firstExt(d){ for(const m of PRIORITY){ if(d.ext[m]) return m; } return null; }
function defaultExt(d){ return d.ext.resiliparse ? "resiliparse" : firstExt(d); }  // fullest extraction for judging
function render(){
  const cats = [...new Set(DOCS.map(d=>d.category))].sort();
  if($("fcat").children.length===0){ $("fcat").innerHTML = "<option value=all>all ("+DOCS.length+")</option>" + cats.map(c=>`<option value="${c}">${c} (${DOCS.filter(d=>d.category===c).length})</option>`).join(""); }
  const v = view(), c = cur();
  const done = Object.keys(labels).filter(u=>labels[u].label).length, pct = Math.round(100*done/DOCS.length);
  const nflag = Object.keys(labels).filter(u=>labels[u].benchmark).length;
  const nrq = DOCS.filter(x=>reviewNeeded(x.url) && !(labels[x.url]&&labels[x.url].label)).length;
  $("nrq").textContent = nrq;
  { const j = c && judgeVisible(c.d) && JUDGE[c.d.url]; const jb=$("jbox");
    if(j){ const rv = j.confidence!=="high";
      jb.innerHTML = `<div class="meta" style="padding:.3rem .6rem;background:#f2f0fb;border-left:3px solid #7a4de0;border-radius:4px;margin:.3rem 0">🤖 judge ${rv?"suggests":"auto-labeled"}: <b>${esc(j.label)}</b> <span style="opacity:.7">(${esc(j.confidence)} conf)</span>${rv?" — review &amp; confirm/override below":""}</div>`;
    } else jb.innerHTML=""; }
  const ntop = Object.keys(labels).filter(u=>labels[u].top_quality).length;
  $("pos").textContent = v.length? idx+1 : 0; $("tot").textContent = v.length; $("done").textContent = done; $("pct").textContent = pct; $("pbar").style.width = pct+"%"; $("nflag").textContent = nflag; $("ntop").textContent = ntop;
  if(!c){ $("head").innerHTML = "<p>No docs in this filter.</p>"; $("evalbox").innerHTML=$("kwbox").innerHTML=$("tabs").innerHTML=$("body").innerHTML=""; return; }
  const d = c.d, L = labels[d.url]||{}, lab = L.label, rev = revealed(d);
  onDocShown(d.url); updateTimer();
  const reg = L.register || d.regGuess;  // only the register is exposed a-priori (pre-filled with our guess)
  const isRandom = d.category === "random";
  if(rev){
    const mis = reg && reg !== d.regGuess ? ` <span class="bad">⚠ re-binned ${esc(d.regGuess)}→${esc(reg)}</span>` : "";
    const hqbadge = d.keptHq==null ? "" : (d.keptHq ? ` · <b style="color:var(--keep)">hq KEPT ✓</b>` : ` · <b style="color:var(--drop)">hq dropped ✗</b>`);
    const prov = isRandom
      ? `<span class="pill">random 2013-2026 · ${esc(d.snapshot||"")}</span> (no pipeline scored this)`
      : `<span class="pill">register: ${esc(d.regGuess)}</span> · verifier ${d.verifier}/4 (excl. hq)${hqbadge} · frac ${d.frac}`;
    $("head").innerHTML = `<div class="meta">${prov} · <b>${esc(d.domain)}</b> · <a href="${esc(d.url)}" target="_blank">url</a>${mis}</div>`;
    $("evalbox").innerHTML = isRandom
      ? "<div class='evalq'>Random natural-web doc — no eval-item match by construction.</div>"
      : d.subjects.map((s,i)=>`<div class="evalq"><b>${esc(s)}</b>${d.prompts[i]?" → <i>"+esc(d.prompts[i])+"</i>":""}</div>`).join("");
    $("kwbox").innerHTML = isRandom ? "" : "matched keywords: " + d.kws.map(k=>`<mark>${esc(k)}</mark>`).join(", ");
    if(tab===null || !d.ext[tab]) tab = defaultExt(d);
    $("tabs").innerHTML = PRIORITY.map(m=>`<button data-m="${m}" class="${m===tab?"on":""} ${d.ext[m]?"":"empty"}">${m}${d.ext[m]?"":" (none)"}</button>`).join("");
    $("tabs").querySelectorAll("button").forEach(b=>b.onclick=()=>{ if(d.ext[b.dataset.m]){ tab=b.dataset.m; render(); }});
    $("body").innerHTML = tab? hl(d.ext[tab], d.kws) : "<i>no extraction available</i>";
  } else {
    const mis = reg !== d.regGuess ? ` <span class="bad">⚠ →${esc(reg)}</span>` : "";
    const tag = isRandom ? `<span class="pill">random web</span>` : `register guess: <span class="pill">${esc(d.regGuess)}</span>`;
    $("head").innerHTML = `<div class="meta">Document <b>${idx+1}</b> — source hidden for unbiased judging · ${tag}${mis} (fix ▾ if wrong) · <a href="#" id="peek">reveal source + details</a></div>`;
    $("evalbox").innerHTML = ""; $("kwbox").innerHTML = ""; $("tabs").innerHTML = "";
    const t = defaultExt(d);
    $("body").innerHTML = t? esc(d.ext[t]) : "<i>no extraction available (empty/redirect page — a valid DROP)</i>";
    const pk = $("peek"); if(pk) pk.onclick = e => { e.preventDefault(); peeked.add(d.url); render(); };
  }
  setRegOptions(reg);
  $("bk").className = "keep"+(lab==="keep"?" on":""); $("bwk").className = "wkeep"+(lab==="weak_keep"?" on":""); $("bwd").className = "wdrop"+(lab==="weak_drop"?" on":""); $("bd").className = "drop"+(lab==="drop"?" on":""); $("bu").className = "uns"+(lab==="unsure"?" on":"");
  $("bflag").className = "flag"+(L.benchmark?" on":""); $("btop").className = "top"+(L.top_quality?" on":"");
  $("notes").value = L.note || "";
}
function setLabel(l){ const c = cur(); if(!c) return; const u = c.d.url; labels[u] = labels[u]||{}; labels[u].label = l; labels[u].ts = Date.now(); save(); render(); }
function toggleFlag(){ const c = cur(); if(!c) return; const u = c.d.url; labels[u] = labels[u]||{}; labels[u].benchmark = !labels[u].benchmark; save(); render(); }
function toggleTop(){ const c = cur(); if(!c) return; const u = c.d.url; labels[u] = labels[u]||{}; labels[u].top_quality = !labels[u].top_quality; save(); render(); }
function scrollTop(){ window.scrollTo(0,0); const b=$("body"); if(b) b.scrollTop=0; }
function next(){ const v=view(); if(idx<v.length-1){ idx++; tab=null; render(); scrollTop(); } }
function prev(){ if(idx>0){ idx--; tab=null; render(); scrollTop(); } }
$("bk").onclick=()=>setLabel("keep"); $("bwk").onclick=()=>setLabel("weak_keep"); $("bwd").onclick=()=>setLabel("weak_drop"); $("bd").onclick=()=>setLabel("drop"); $("bu").onclick=()=>setLabel("unsure");
$("bflag").onclick=toggleFlag; $("btop").onclick=toggleTop;
$("btimer").onclick=()=>{ timerOn=!timerOn; localStorage.setItem(KEY+"_timer",JSON.stringify(timerOn)); timerStart=Date.now(); $("btimer").className=timerOn?"on":""; updateTimer(); };
setInterval(updateTimer, 1000);
$("btimer").className = timerOn ? "on" : "";
$("bnext").onclick=next; $("bprev").onclick=prev;
$("fcat").onchange=e=>{ fcat=e.target.value; idx=0; render(); };
$("unl").onchange=e=>{ unlOnly=e.target.checked; idx=0; render(); };
$("rq").onchange=e=>{ rqOnly=e.target.checked; idx=0; render(); };
$("jmode").value=judgeMode;
$("jmode").onchange=e=>{ judgeMode=e.target.value; localStorage.setItem(KEY+"_judgemode",JSON.stringify(judgeMode)); render(); };
$("notes").oninput=e=>{ const c=cur(); if(c){ labels[c.d.url]=labels[c.d.url]||{}; labels[c.d.url].note=e.target.value; save(); } };
let customRegs = [];
try{ customRegs = JSON.parse(localStorage.getItem(KEY+"_customregs")||"[]"); }catch(e){}
function setRegOptions(sel){
  const groups = Object.entries(REGISTER_GROUPS).map(([g,rs])=>
    `<optgroup label="${g}">`+rs.map(r=>`<option value="${r}">${r}</option>`).join("")+"</optgroup>").join("");
  const custom = customRegs.length ? `<optgroup label="custom">`+customRegs.map(r=>`<option value="${r}">${r}</option>`).join("")+"</optgroup>" : "";
  // if the current value isn't a known option (a custom register from a prior session), surface it
  const known = new Set([...Object.values(REGISTER_GROUPS).flat(), ...customRegs]);
  const extra = (sel && !known.has(sel)) ? `<option value="${sel}">${sel}</option>` : "";
  $("reg").innerHTML = groups + custom + extra + `<option value="__custom__">+ new register…</option>`;
  $("reg").value = sel;
}
$("reg").onchange=e=>{ const c=cur(); if(!c) return;
  let val = e.target.value;
  if(val==="__custom__"){ val = (prompt("New register name:")||"").trim().toLowerCase().replace(/\s+/g,"_");
    if(!val){ setRegOptions(labels[c.d.url]&&labels[c.d.url].register || c.d.regGuess); return; }
    if(!customRegs.includes(val)){ customRegs.push(val); localStorage.setItem(KEY+"_customregs", JSON.stringify(customRegs)); } }
  labels[c.d.url]=labels[c.d.url]||{}; labels[c.d.url].register=val; save(); render();
};
$("exp").onclick=()=>{ const blob=new Blob([JSON.stringify(labels,null,2)],{type:"application/json"}); const a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download="devset_hard_labels.json"; a.click(); };
document.onkeydown=e=>{ if(e.target.tagName==="INPUT") return; if(e.key==="k")setLabel("keep"); else if(e.key==="w")setLabel("weak_keep"); else if(e.key==="x")setLabel("weak_drop"); else if(e.key==="d")setLabel("drop"); else if(e.key==="u")setLabel("unsure"); else if(e.key==="f")toggleFlag(); else if(e.key==="t")toggleTop(); else if(e.key==="ArrowRight"||e.key===" ")next(); else if(e.key==="ArrowLeft")prev(); };
render();
</script></body></html>"""


if __name__ == "__main__":
    sys.exit(main())
