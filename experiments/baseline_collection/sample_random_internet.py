# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Draw a fresh random 1k-doc sample of the open web spanning 2013-2026, held separate from the dev set.

Source: `internet_timespan_sample/v1` (us-east5) — a 1/50 random HTML sample across all CommonCrawl
snapshots 2013-2026 (~102,853 docs, one crawl per shard). No language/quality filter: the point is to
see what a raw random slice of the internet actually looks like. Excludes urls already in the dev set
so the two are disjoint.

  sample  (Iris CPU, in-region us-east5) — hash-sample ~TARGET*OVERSAMPLE docs (non-empty html, not in
          the dev set) into one small parquet; reads the 2.85 GiB in-region, only the sample leaves.
  build   (local) — pull the sample, take exactly --n by hash order, write the small-rephraser
          `sample_1k`-style tree + a resiliparse-text labeling interface (keep/weak_keep/weak_drop/drop).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections import Counter
from urllib.parse import urlparse

import fsspec
import pyarrow.parquet as pq
from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext

from experiments.baseline_collection.build_devset import _domain_register
from experiments.baseline_collection.build_relabel_interface import _readable_text

logger = logging.getLogger(__name__)

TIMESPAN = "gs://marin-us-east5/documents/internet_timespan_sample/v1"
WORKSPACE = "gs://marin-us-east5/scratch/random_internet_1k"
EXCLUDE = f"{WORKSPACE}/exclude_urls.json"  # dev-set urls to hold out (written before the job)
SAMPLE_OUT = f"{WORKSPACE}/sample"  # parquet {doc_id, url, snapshot, html, hfrac}

SEED = "random-internet-2026"
TARGET = 1000
POOL = 102853  # docs in the timespan sample; only used to size the hash threshold


def _hfrac(doc_id: str) -> float:
    return int(hashlib.md5(f"{SEED}|{doc_id}".encode()).hexdigest(), 16) / 2**128


def run_sample(max_workers: int, oversample: float) -> None:
    """Hash-sample ~TARGET*oversample docs into SAMPLE_OUT (in-region us-east5)."""
    fs = fsspec.filesystem("gcs")
    with fs.open(EXCLUDE.replace("gs://", "")) as f:
        exclude = set(json.load(f))
    frac = min(1.0, TARGET * oversample / POOL)
    logger.info("hash threshold %.4g (~%d docs), excluding %d dev urls", frac, int(frac * POOL), len(exclude))

    def keep(r: dict) -> dict | None:
        if not r.get("html") or r.get("url") in exclude:
            return None
        h = _hfrac(r["doc_id"])
        if h >= frac:
            return None
        return {"doc_id": r["doc_id"], "url": r["url"], "snapshot": r["snapshot"], "html": r["html"], "hfrac": h}

    pipeline = (
        Dataset.from_files(f"{TIMESPAN}/*.parquet")
        .load_parquet(columns=["doc_id", "url", "snapshot", "html"])
        .map(keep)
        .filter(lambda x: x is not None)
        .reshard(4)
        .write_parquet(f"{SAMPLE_OUT}/s-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=False)
    )
    ZephyrContext(
        name="sample-random-internet",
        max_workers=max_workers,
        resources=ResourceConfig(cpu=1, ram="4g", regions=["us-east5"], preemptible=True),
    ).execute(pipeline)
    logger.info("wrote sample -> %s", SAMPLE_OUT)


def _year(snapshot: str) -> str:
    parts = (snapshot or "").split("-")
    return parts[2] if len(parts) > 2 else "?"


def run_build(out_dir: str, n: int) -> None:
    fs = fsspec.filesystem("gcs")
    files = [f"gs://{f}" for f in fs.glob(f"{SAMPLE_OUT}/*.parquet".replace("gs://", ""))]
    if not files:
        raise SystemExit(f"no sample at {SAMPLE_OUT} — run `sample` first")
    rows: list[dict] = []
    for f in files:
        rows.extend(pq.read_table(f, filesystem=fs, columns=["doc_id", "url", "snapshot", "html", "hfrac"]).to_pylist())
    n_sampled = len(rows)
    rows.sort(key=lambda r: r["hfrac"])  # deterministic uniform order
    rows = rows[:n]
    logger.info("selected %d of %d sampled docs", len(rows), n_sampled)

    html_dir = os.path.join(out_dir, "marin_random_1k_html")
    os.makedirs(html_dir, exist_ok=True)
    docs, jsonl = [], []
    yr_dist: Counter = Counter()
    for i, r in enumerate(rows):
        url, html = r["url"], r["html"]
        hid = hashlib.md5(url.encode()).hexdigest()[:10]
        dom = (urlparse(url).netloc or "unknown").lower()
        slug = "".join(c if c.isalnum() else "_" for c in dom).strip("_") or "unknown"
        base = f"record_{i:05d}_{slug}_{hid}"
        with open(os.path.join(html_dir, base + ".html"), "w") as fh:
            fh.write(html)
        meta = {
            "url": url,
            "hid": hid,
            "snapshot": r["snapshot"],
            "year": _year(r["snapshot"]),
            "register_guess": _domain_register(dom),
            "label": None,
        }
        with open(os.path.join(html_dir, base + ".html.meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        docs.append({**meta, "text": _readable_text(html), "html": html})
        jsonl.append({**meta, "html_file": base + ".html", "html_len": len(html)})
        yr_dist[meta["year"]] += 1

    with open(os.path.join(html_dir, "random_1k.jsonl"), "w") as f:
        for row in jsonl:
            f.write(json.dumps(row) + "\n")
    payload = json.dumps(docs).replace("</", "<\\/")
    with open(os.path.join(html_dir, "label_random_1k.html"), "w") as f:
        f.write(_LABEL_TEMPLATE.replace("__DOCS__", payload))
    with open(os.path.join(html_dir, "random_1k_meta.json"), "w") as f:
        json.dump({"num_docs": len(jsonl), "seed": SEED, "year_dist": dict(sorted(yr_dist.items()))}, f, indent=2)
    logger.info("wrote %d docs -> %s", len(jsonl), html_dir)
    print(json.dumps({"num_docs": len(jsonl), "year_dist": dict(sorted(yr_dist.items()))}, indent=2))


def run_labeled(src_dir: str, labels_path: str, out_dir: str) -> None:
    """Write just the labeled subset (label folded into meta) to its own small-rephraser-style folder."""
    labels = json.load(open(labels_path))
    src = os.path.join(src_dir, "marin_random_1k_html")
    rows = [json.loads(line) for line in open(os.path.join(src, "random_1k.jsonl"))]
    out = os.path.join(out_dir, "marin_random_labeled_html")
    os.makedirs(out, exist_ok=True)

    jsonl, dist = [], Counter()
    for r in rows:
        lab = (labels.get(r["url"]) or {}).get("label")
        if not lab:
            continue
        i = len(jsonl)
        slug = r["html_file"].split("_", 2)[2].rsplit("_", 1)[0]  # reuse the domain slug from the 1k filename
        base = f"record_{i:05d}_{slug}_{r['hid']}"
        html = open(os.path.join(src, r["html_file"])).read()
        with open(os.path.join(out, base + ".html"), "w") as f:
            f.write(html)
        meta = {k: r[k] for k in ("url", "hid", "snapshot", "year", "register_guess")}
        meta["label"] = lab
        with open(os.path.join(out, base + ".html.meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        jsonl.append({**meta, "html_file": base + ".html", "html_len": len(html)})
        dist[lab] += 1

    with open(os.path.join(out, "random_labeled.jsonl"), "w") as f:
        for row in jsonl:
            f.write(json.dumps(row) + "\n")
    with open(os.path.join(out, "random_labeled_meta.json"), "w") as f:
        json.dump({"num_docs": len(jsonl), "seed": SEED, "label_dist": dict(dist.most_common())}, f, indent=2)
    logger.info("wrote %d labeled docs -> %s", len(jsonl), out)
    print(json.dumps({"num_docs": len(jsonl), "label_dist": dict(dist.most_common())}, indent=2))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    sp = sub.add_parser("sample", help="Iris CPU: hash-sample the timespan pool in-region")
    sp.add_argument("--max-workers", type=int, default=32)
    sp.add_argument("--oversample", type=float, default=1.5, help="draw TARGET*this, trim to exactly --n in build")
    bl = sub.add_parser("build", help="local: explode the sample + build the labeling interface")
    bl.add_argument("--out-dir", default="scratch/random_internet_export")
    bl.add_argument("--n", type=int, default=TARGET)
    lb = sub.add_parser("labeled", help="local: write the labeled subset to its own folder")
    lb.add_argument("--labels", required=True, help="random_1k_labels.json exported from the interface")
    lb.add_argument("--src-dir", default="scratch/random_internet_export")
    lb.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    if args.mode == "sample":
        run_sample(args.max_workers, args.oversample)
    elif args.mode == "labeled":
        run_labeled(args.src_dir, args.labels, args.out_dir)
    else:
        run_build(args.out_dir, args.n)
    return 0


_LABEL_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Label random-internet 1k</title>
<style>
 :root{--bg:#faf9f7;--fg:#1a1a1a;--mut:#6b6b6b;--line:#e2e0db;--keep:#1a7f4b;--wkeep:#7bb661;--wdrop:#d99a2b;--drop:#c0392b;--accent:#2b6cb0}
 *{box-sizing:border-box} body{margin:0;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:var(--bg);color:var(--fg)}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:14px;align-items:center;z-index:5;flex-wrap:wrap}
 header b{font-size:15px} .mut{color:var(--mut)} kbd{background:#eee;border:1px solid #ccc;border-radius:4px;padding:0 5px;font-size:12px}
 #wrap{display:grid;grid-template-columns:1fr 320px;height:calc(100vh - 52px)}
 #docpane{overflow:auto;background:#fff;padding:24px 32px}
 #text{white-space:pre-wrap;font:14px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;max-width:74ch;word-break:break-word}
 #frame{width:100%;height:calc(100vh - 52px);border:0;background:#fff;display:none}
 #docpane.raw{padding:0} #docpane.raw #text{display:none} #docpane.raw #frame{display:block}
 #side{border-left:1px solid var(--line);padding:16px;overflow:auto;background:#fff}
 .url{word-break:break-all;font-size:12px;color:var(--accent)} .meta{margin:8px 0;font-size:12px;color:var(--mut)}
 .chip{display:inline-block;padding:2px 8px;background:#eef;border-radius:10px;font-size:12px;margin-right:6px}
 .btns{display:flex;flex-direction:column;gap:8px;margin-top:14px}
 button.lab{padding:10px;border:1px solid var(--line);border-radius:8px;background:#fff;cursor:pointer;font-size:14px;text-align:left}
 button.lab.on{color:#fff;border-color:transparent} .keep.on{background:var(--keep)} .wkeep.on{background:var(--wkeep)} .wdrop.on{background:var(--wdrop)} .drop.on{background:var(--drop)}
 #exp{margin-left:auto;padding:7px 14px;background:var(--accent);color:#fff;border:0;border-radius:8px;cursor:pointer;font-size:14px}
 #rawtog,#jump{padding:6px 10px;border:1px solid var(--line);border-radius:8px;background:#fff;cursor:pointer;font-size:13px}
</style></head><body>
<header>
 <b>Random-internet 1k</b>
 <span class="mut">doc <b id="idx">1</b>/<b id="tot">0</b> · labeled <b id="done">0</b></span>
 <span class="mut">keys: <kbd>k</kbd>keep <kbd>w</kbd>weak-keep <kbd>x</kbd>weak-drop <kbd>d</kbd>drop <kbd>u</kbd>unlabel <kbd>←</kbd><kbd>→</kbd> <kbd>r</kbd>raw</span>
 <button id="rawtog">raw HTML (r)</button>
 <input id="jump" type="number" min="1" placeholder="# go" style="width:70px">
 <button id="exp">⬇ export random_1k_labels.json</button>
</header>
<div id="wrap">
 <div id="docpane"><div id="text"></div><iframe id="frame" sandbox referrerpolicy="no-referrer"></iframe></div>
 <div id="side">
   <div class="url" id="url"></div>
   <div class="meta"><span class="chip" id="year"></span><span class="chip" id="reg"></span></div>
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
const KEY = "random_internet_1k_v1";
let labels = {}; try{ labels = JSON.parse(localStorage.getItem(KEY)||"{}"); }catch(e){}
let raw = false;
let i = DOCS.findIndex(d => !labels[d.url]); if(i < 0) i = 0;
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
  $("year").textContent = d.year;
  $("reg").textContent = "guess: " + d.register_guess;
  const chosen = labels[d.url];
  document.querySelectorAll("button.lab").forEach(b=>b.classList.toggle("on", b.dataset.l===chosen));
  $("idx").textContent = i+1;
  $("done").textContent = Object.keys(labels).length;
}
function setLabel(l){ labels[DOCS[i].url] = l; save(); if(i<DOCS.length-1) i++; render(); }
function go(n){ i = Math.max(0, Math.min(DOCS.length-1, n)); render(); }
document.querySelectorAll("button.lab").forEach(b=> b.onclick = ()=>setLabel(b.dataset.l));
$("rawtog").onclick = ()=>{ raw=!raw; render(); };
$("jump").onchange = e => { const v=parseInt(e.target.value,10); if(v>=1&&v<=DOCS.length) go(v-1); };
document.onkeydown = e => {
  if(e.target.tagName==="INPUT") return;
  if(e.key==="k")setLabel("keep"); else if(e.key==="w")setLabel("weak_keep");
  else if(e.key==="x")setLabel("weak_drop"); else if(e.key==="d")setLabel("drop");
  else if(e.key==="u"){ delete labels[DOCS[i].url]; save(); render(); }
  else if(e.key==="r"){ raw=!raw; render(); }
  else if(e.key==="ArrowRight")go(i+1); else if(e.key==="ArrowLeft")go(i-1);
};
$("exp").onclick = ()=>{
  const out = {}; for(const u in labels) out[u] = {label: labels[u]};
  const blob = new Blob([JSON.stringify(out,null,2)],{type:"application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "random_1k_labels.json"; a.click();
};
render();
</script></body></html>"""


if __name__ == "__main__":
    sys.exit(main())
