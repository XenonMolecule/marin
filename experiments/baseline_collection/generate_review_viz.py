# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a local HTML review viewer for gold extractions: Sonnet vs gemini side-by-side + gate flags.

Lets a human validate the flagged docs alongside the manager's automated gates. Reads the manifest, the
Sonnet outputs (extract_out/<hid>.txt), the gemini cross-check outputs (gemini_<hid>.txt), the capped
resiliparse baseline, and the gate results, and emits a single self-contained file:// page.
"""

from __future__ import annotations

import difflib
import html as H
import json
import sys

ROOT = "scratch/gold_extraction"


def _read(path: str) -> str:
    try:
        return open(path).read()
    except FileNotFoundError:
        return ""


def _wdiff_inline(a: str, b: str) -> str:
    """Word-level diff of a single reworded line (keeps fine detail without confetti-ing the whole doc)."""
    aw, bw = a.split(), b.split()
    sm = difflib.SequenceMatcher(a=aw, b=bw, autojunk=False)
    out: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.append(H.escape(" ".join(aw[i1:i2])))
        if tag in ("delete", "replace"):
            out.append("<del class=w>" + H.escape(" ".join(aw[i1:i2])) + "</del>")
        if tag in ("insert", "replace"):
            out.append("<ins class=w>" + H.escape(" ".join(bw[j1:j2])) + "</ins>")
    return " ".join(out)


def _is_fmt(line: str) -> bool:
    """A diff line that is noise, not dropped content: a code fence, a re-quote, or a short fragment."""
    t = line.strip()
    return ("```" in t) or t.startswith(">") or len(t) < 40


def _diff(a: str, b: str, context: int = 3) -> str:
    """Git-diff-style Sonnet(a)->gemini(b): line-level, unchanged runs collapsed, each changed line a
    full-width block. Changed lines are classed `sub` (substantive — a REAL difference) or `fmt`
    (fence/re-quote/short — noise). A header counts each and offers a substantive-only toggle, so a
    scary-looking all-green diff that is purely fences/re-quotes reads as 0 substantive differences."""
    la, lb = a.splitlines(), b.splitlines()
    sm = difflib.SequenceMatcher(a=la, b=lb, autojunk=False)
    out: list[str] = []
    n_sub = n_fmt = 0

    def cell(tag: str, line: str) -> str:
        nonlocal n_sub, n_fmt
        fmt = _is_fmt(line)
        n_fmt += fmt
        n_sub += not fmt
        return f'<{tag} class={"fmt" if fmt else "sub"}>{H.escape(line) or "&nbsp;"}</{tag}>'

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            blk = la[i1:i2]
            if len(blk) > context * 2 + 1:
                out += [f'<span class=eq>{H.escape(x) or "&nbsp;"}</span>' for x in blk[:context]]
                out.append(f"<span class=gap>⋯ {len(blk) - context * 2} unchanged lines ⋯</span>")
                out += [f'<span class=eq>{H.escape(x) or "&nbsp;"}</span>' for x in blk[-context:]]
            else:
                out += [f'<span class=eq>{H.escape(x) or "&nbsp;"}</span>' for x in blk]
        elif op == "replace" and (i2 - i1) == 1 and (j2 - j1) == 1:
            out.append(f"<span class=chg>{_wdiff_inline(la[i1], lb[j1])}</span>")
        else:
            out += [cell("del", x) for x in la[i1:i2]]
            out += [cell("ins", x) for x in lb[j1:j2]]

    verdict = "⚠ REAL content differs — review" if n_sub else "✓ differences are formatting/quotes only — nothing substantive dropped"
    hdr = (f'<div class=diffhdr><b class=real>{n_sub}</b> substantive · '
           f'<b class=fmt>{n_fmt}</b> formatting/quote lines &nbsp; <span class=vd>{verdict}</span>'
           f'<button onclick="document.body.classList.toggle(\'subonly\')">substantive-only</button></div>')
    return hdr + "".join(out)


def main() -> int:
    manifest = json.load(open(sys.argv[1]))
    resi = json.load(open(f"{ROOT}/val_resiliparse.json")) if len(sys.argv) < 3 else json.load(open(sys.argv[2]))
    checks = {r["url"]: r for r in json.load(open(f"{ROOT}/extraction_checks.json"))}

    docs = []
    for d in manifest:
        hid = d["html"].split("/")[-1].replace(".html", "")
        c = checks.get(d["url"], {})
        sonnet = _read(f"{ROOT}/extract_out/{hid}.txt")
        # prefer the merged-gold candidate as the comparison column; fall back to gemini
        merged = _read(f"{ROOT}/extract_out/{hid}_merged.txt")
        gemini = merged if merged else _read(f"{ROOT}/extract_out/gemini_{hid}.txt")
        docs.append({
            "url": d["url"],
            "register": d["register"],
            "hid": hid,
            "sonnet": sonnet,
            "gemini": gemini,
            "compare_label": "MERGED GOLD" if merged else "gemini-2.5-flash",
            "resi": resi.get(d["url"], ""),
            "diff": _diff(sonnet, gemini) if gemini and not gemini.startswith("[") else "",
            "verdict": c.get("verdict", "?"),
            "issues": c.get("issues", []),
            "stats": {k: c.get(k) for k in ("chars", "inversions", "matched_paras", "cleanliness", "resi_ratio", "gem_ratio")},
        })

    data = json.dumps(docs).replace("</", "<\\/")
    page = _TEMPLATE.replace("__DATA__", data)
    out = f"{ROOT}/extraction_review.html"
    open(out, "w").write(page)
    print(f"wrote {out}  ({len(docs)} docs)")
    print(f"open:  file://{__import__('os').path.abspath(out)}")
    return 0


_TEMPLATE = r"""<!doctype html><html><head><meta charset=utf-8><title>Gold extraction review</title>
<style>
:root{--bg:#12141a;--panel:#1b1e27;--line:#2b2f3a;--tx:#e6e8ee;--mut:#9aa3b2;--acc:#6ea8fe;--ok:#4ec98a;--flag:#e6a94e}
*{box-sizing:border-box}body{margin:0;font:13px/1.5 -apple-system,system-ui,sans-serif;background:var(--bg);color:var(--tx);display:flex;height:100vh}
#list{width:280px;border-right:1px solid var(--line);overflow:auto;flex:none}
#list .doc{padding:10px 12px;border-bottom:1px solid var(--line);cursor:pointer}
#list .doc:hover{background:#20242f}#list .doc.sel{background:#242938;border-left:3px solid var(--acc)}
.badge{font-size:11px;padding:1px 7px;border-radius:10px;font-weight:600}
.PASS{background:rgba(78,201,138,.18);color:var(--ok)}.FLAG{background:rgba(230,169,78,.18);color:var(--flag)}
.reg{color:var(--mut);font-size:11px}.u{font-size:11px;color:var(--mut);word-break:break-all;margin-top:3px}
#main{flex:1;overflow:auto;padding:16px 20px}
.stats{display:flex;gap:14px;flex-wrap:wrap;margin:8px 0 4px;color:var(--mut);font-size:12px}
.stats b{color:var(--tx);font-variant-numeric:tabular-nums}
.issues{margin:8px 0;padding:8px 12px;background:rgba(230,169,78,.08);border:1px solid rgba(230,169,78,.3);border-radius:6px}
.issues div{color:var(--flag);margin:2px 0}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
.pane{background:var(--panel);border:1px solid var(--line);border-radius:8px;min-width:0}
.pane h3{margin:0;padding:8px 12px;border-bottom:1px solid var(--line);font-size:12px;color:var(--mut);position:sticky;top:0;background:var(--panel)}
.pane pre{margin:0;padding:12px;white-space:pre-wrap;word-break:break-word;max-height:70vh;overflow:auto;font:12px/1.5 ui-monospace,Menlo,monospace}
pre.diff{line-height:1.5}
.eq{display:block;color:#6b7280}
.gap{display:block;text-align:center;color:var(--acc);opacity:.6;font-size:11px;padding:4px 0}
.chg{display:block;background:rgba(110,168,254,.10);border-left:3px solid var(--acc);padding-left:6px}
pre.diff del{display:block;background:rgba(230,110,110,.14);border-left:3px solid #e06e6e;padding-left:6px;color:#f4c4c4;text-decoration:none}
pre.diff ins{display:block;background:rgba(78,201,138,.14);border-left:3px solid #4ec98a;padding-left:6px;color:#c7f5da;text-decoration:none}
del.w{display:inline;background:rgba(230,110,110,.3);border:none;padding:0 1px}
ins.w{display:inline;background:rgba(78,201,138,.3);border:none;padding:0 1px}
.legend{color:var(--mut);font-size:11px;margin-bottom:6px}.legend ins,.legend del{padding:0 5px;text-decoration:none}
pre.diff del.fmt,pre.diff ins.fmt{opacity:.35;border-left-width:2px;font-size:11px}
pre.diff del.sub,pre.diff ins.sub{outline:1px solid rgba(255,255,255,.18)}
.diffhdr{padding:8px 11px;margin-bottom:8px;border-radius:6px;background:#242938;font-size:12px}
.diffhdr b.real{color:var(--flag)}.diffhdr b.fmt{color:var(--mut)}.diffhdr .vd{color:var(--mut);margin:0 10px}
.diffhdr button{background:#333a4d;color:var(--tx);border:1px solid var(--line);border-radius:5px;padding:2px 8px;cursor:pointer}
body.subonly pre.diff .eq,body.subonly pre.diff .gap,body.subonly pre.diff .fmt{display:none}
.tog{margin:10px 0}.tog button{background:#242938;color:var(--tx);border:1px solid var(--line);border-radius:6px;padding:4px 10px;cursor:pointer;margin-right:6px}
.tog button.on{background:var(--acc);color:#0b0d12;border-color:var(--acc)}
h2{margin:0 0 2px;font-size:15px}
</style></head><body>
<div id=list></div><div id=main></div>
<script>
const DOCS=__DATA__;let cur=0,mode='side';
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function renderList(){document.getElementById('list').innerHTML=DOCS.map((d,i)=>
 `<div class="doc${i==cur?' sel':''}" onclick="sel(${i})"><span class="badge ${d.verdict}">${d.verdict}</span> <span class="reg">${d.register}</span><div class="u">${esc(d.url)}</div></div>`).join('')}
function sel(i){cur=i;renderList();renderMain()}
function renderMain(){const d=DOCS[cur];const s=d.stats;
 const stats=`<div class=stats><span>chars <b>${s.chars}</b></span><span>inv <b>${s.inversions}</b></span>`+
  `<span>matched <b>${s.matched_paras}</b></span><span>clean <b>${s.cleanliness}</b></span>`+
  `<span>resi× <b>${s.resi_ratio??'–'}</b></span><span>gem× <b>${s.gem_ratio??'–'}</b></span></div>`;
 const iss=d.issues.length?`<div class=issues>${d.issues.map(x=>`<div>⚑ ${esc(x)}</div>`).join('')}</div>`:'';
 const panes={sonnet:['Sonnet (current gold)',d.sonnet],gemini:[(d.compare_label||'gemini')+' (candidate)',d.gemini],resi:['resiliparse (baseline)',d.resi]};
 let body;
 if(mode==='side') body=`<div class=cols><div class=pane><h3>${panes.sonnet[0]} · ${panes.sonnet[1].length}c</h3><pre>${esc(panes.sonnet[1])}</pre></div><div class=pane><h3>${panes.gemini[0]} · ${panes.gemini[1].length}c</h3><pre>${esc(panes.gemini[1])}</pre></div></div>`;
 else if(mode==='diff') body=`<div class=legend><del>red</del> = removed from Sonnet · <ins>green</ins> = added by ${d.compare_label||'candidate'} (recovered content)</div><div class=pane><h3>Sonnet → ${d.compare_label||'candidate'} inline diff (unchanged runs collapsed)</h3><pre class=diff>${d.diff||'(unavailable)'}</pre></div>`;
 else{const p=panes[mode];body=`<div class=pane><h3>${p[0]} · ${p[1].length}c</h3><pre>${esc(p[1])}</pre></div>`}
 document.getElementById('main').innerHTML=`<h2><span class="badge ${d.verdict}">${d.verdict}</span> ${d.register}</h2><div class=u>${esc(d.url)}</div>${stats}${iss}`+
  `<div class=tog>${['side','diff','sonnet','gemini','resi'].map(m=>`<button class="${mode==m?'on':''}" onclick="setmode('${m}')">${m}</button>`).join('')}</div>${body}`}
function setmode(m){mode=m;renderMain()}
renderList();renderMain();
</script></body></html>"""


if __name__ == "__main__":
    sys.exit(main())
