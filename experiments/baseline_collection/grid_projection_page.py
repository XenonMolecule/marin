# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the token projection as a standalone HTML page.

Takes the ``projection.json`` written by
:mod:`experiments.baseline_collection.grid_projection_report` and emits a
self-contained page: the 24 x 5 grid as a heatmap with the numbers written into
the cells, the two marginal distributions, and the method notes a reader needs to
know what the projection does and does not claim.

Cell colour is a **log** ramp. Token mass across the grid spans four orders of
magnitude — the largest cell holds more than a thousand times the smallest — so a
linear ramp would render 110 of the 120 cells as the same near-white step and
show nothing. The legend says so explicitly, and every cell carries its number,
so the colour is an aid to scanning rather than the only encoding.

    python -m experiments.baseline_collection.grid_projection_page \\
        --projection report/projection.json --out report/index.html
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pathlib

logger = logging.getLogger(__name__)

# Sequential blue, steps 100 -> 700 (validated ramp; see the dataviz palette).
RAMP_LIGHT = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]  # fmt: skip
RAMP_DARK = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"]  # fmt: skip


def fmt_tokens(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.1f}B"
    if value >= 1e6:
        return f"{value / 1e6:.0f}M"
    if value >= 1e3:
        return f"{value / 1e3:.0f}K"
    return f"{value:.0f}"


def ramp_index(value: float, vmax: float) -> int:
    """Log-scaled step for ``value``, 0 for an empty cell."""
    if value <= 0:
        return -1
    span = math.log10(max(vmax, 10.0)) - 3.0  # floor the ramp at 1K tokens
    frac = (math.log10(max(value, 1e3)) - 3.0) / max(span, 1e-9)
    return max(0, min(len(RAMP_LIGHT) - 1, int(frac * (len(RAMP_LIGHT) - 1) + 0.5)))


def build_page(report: dict) -> str:
    topics = report["topics"]
    buckets = report["buckets"]
    tokens = report["projected"]["llama_tokens"]
    docs = report["projected"]["docs"]
    lo = report["projected"]["llama_tokens_lo"]
    hi = report["projected"]["llama_tokens_hi"]

    row_total = [sum(row) for row in tokens]
    col_total = [sum(tokens[i][b] for i in range(len(topics))) for b in range(len(buckets))]
    grand = sum(row_total)
    vmax = max(max(row) for row in tokens)
    order = sorted(range(len(topics)), key=lambda i: -row_total[i])

    head = (
        "<tr><th class='topic'>Topic</th>"
        + "".join(f"<th>{b}</th>" for b in buckets)
        + "<th class='tot'>Total</th><th class='share'>Share</th></tr>"
    )
    rows = []
    for i in order:
        cells = []
        for b in range(len(buckets)):
            step = ramp_index(tokens[i][b], vmax)
            style = "" if step < 0 else f" style='--step:{step}'"
            tip = (
                f"{topics[i]} · {buckets[b]}&#10;"
                f"{fmt_tokens(tokens[i][b])} tokens&#10;"
                f"95% CI {fmt_tokens(lo[i][b])}–{fmt_tokens(hi[i][b])}&#10;"
                f"{docs[i][b] / 1e6:.2f}M docs"
            )
            cells.append(f"<td class='cell'{style} title=\"{tip}\">{fmt_tokens(tokens[i][b])}</td>")
        share = row_total[i] / grand * 100
        rows.append(
            f"<tr><th class='topic'>{topics[i]}</th>{''.join(cells)}"
            f"<td class='tot'>{fmt_tokens(row_total[i])}</td>"
            f"<td class='share'><span class='bar' style='--w:{share / max(1e-9, max(row_total) / grand * 100) * 100}%'></span>"
            f"<span class='pct'>{share:.1f}%</span></td></tr>"
        )
    foot = (
        "<tr class='foot'><th class='topic'>All topics</th>"
        + "".join(f"<td class='tot'>{fmt_tokens(v)}</td>" for v in col_total)
        + f"<td class='tot'>{fmt_tokens(grand)}</td><td></td></tr>"
    )
    qshare = "".join(
        f"<div class='qcol'><div class='qbar' style='--h:{v / max(col_total) * 100}%;--step:{4 + b}'></div>"
        f"<div class='qval'>{fmt_tokens(v)}</div><div class='qlab'>{buckets[b]}</div>"
        f"<div class='qpct'>{v / grand * 100:.1f}%</div></div>"
        for b, v in enumerate(col_total)
    )

    n = report["sampled_warcs"]
    pool = report["pool_warcs"]
    sampled_tokens = sum(sum(r) for r in report["sampled"]["llama_tokens"])
    sampled_docs = sum(sum(r) for r in report["sampled"]["docs"])
    lo_total = sum(sum(r) for r in lo)
    hi_total = sum(sum(r) for r in hi)

    return f"""<title>llm_pipeline_v1_1 token projection</title>
<style>
  .viz-root {{
    color-scheme: light;
    --surface-1: #fcfcfb; --surface-2: #f4f3f0; --border: #dedcd6;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #75746f;
    --accent: #2a78d6;
    {"".join(f"--s{i}: {c};" for i, c in enumerate(RAMP_LIGHT))}
    --cell-ink: var(--text-primary);
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) .viz-root {{
      color-scheme: dark;
      --surface-1: #1a1a19; --surface-2: #232322; --border: #3a3a37;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #9b9a92;
      --accent: #3987e5;
      {"".join(f"--s{i}: {c};" for i, c in enumerate(RAMP_DARK))}
    }}
  }}
  :root[data-theme="dark"] .viz-root {{
    color-scheme: dark;
    --surface-1: #1a1a19; --surface-2: #232322; --border: #3a3a37;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #9b9a92;
    --accent: #3987e5;
    {"".join(f"--s{i}: {c};" for i, c in enumerate(RAMP_DARK))}
  }}
  * {{ box-sizing: border-box; }}
  /* Three roles, all from system stacks — the artifact CSP blocks font CDNs and a
     linked webfont would fail silently to a default. Serif for prose (this is a
     measurement memo, and the serif says "read me" rather than "product"), mono
     everywhere digits line up, so the grid reads as instrument output. */
  .viz-root {{
    background: var(--surface-1); color: var(--text-primary);
    font: 16px/1.6 ui-serif, Charter, "Iowan Old Style", Georgia, serif;
    padding: 40px 28px 80px; max-width: 1180px; margin: 0 auto;
  }}
  .mono, td, th, .stat .v, .qval, .qpct {{
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  }}
  h1 {{ font-size: 30px; line-height: 1.2; letter-spacing: -0.015em; margin: 0 0 8px;
       font-weight: 600; text-wrap: balance; }}
  .sub {{ color: var(--text-secondary); margin: 0 0 34px; max-width: 66ch; font-size: 15px; }}
  h2 {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
       font-size: 12px; text-transform: uppercase; letter-spacing: 0.1em;
       color: var(--text-muted); margin: 46px 0 14px; font-weight: 600; }}
  .hero {{ display: flex; flex-wrap: wrap; gap: 14px; }}
  .stat {{ background: var(--surface-2); border: 1px solid var(--border);
           border-radius: 10px; padding: 16px 20px; flex: 1 1 210px; }}
  .stat .v {{ font-size: 30px; font-weight: 650; letter-spacing: -0.02em;
              font-variant-numeric: tabular-nums; }}
  .stat .k {{ color: var(--text-secondary); font-size: 13px; margin-top: 3px; }}
  .stat .n {{ color: var(--text-muted); font-size: 12px; margin-top: 6px; }}
  .scroll {{ overflow-x: auto; }}
  table {{ border-collapse: separate; border-spacing: 2px; width: 100%; min-width: 760px; }}
  th, td {{ font-variant-numeric: tabular-nums; font-size: 13px; text-align: right;
            padding: 7px 9px; white-space: nowrap; }}
  thead th {{ color: var(--text-secondary); font-weight: 600; font-size: 12px;
              text-transform: uppercase; letter-spacing: 0.05em; padding-bottom: 4px; }}
  th.topic {{ text-align: left; font-weight: 500; color: var(--text-primary); }}
  td.cell {{ background: var(--s0); border-radius: 4px; color: var(--cell-ink); }}
  /* Substring matching is safe only because the higher indices are emitted
     LAST: a cell at step 10 matches both `--step:1` and `--step:10`, and the
     later rule is the one that wins. */
  {"".join(f"td.cell[style*='--step:{i}']{{background:var(--s{i});}}" for i in range(len(RAMP_LIGHT)))}
  {"".join(f"td.cell[style*='--step:{i}']{{color:{'#ffffff' if i >= 7 else 'var(--text-primary)'};}}" for i in range(len(RAMP_LIGHT)))}
  td.cell:not([style]) {{ background: var(--surface-2); color: var(--text-muted); }}
  td.tot {{ font-weight: 650; }}
  tr.foot td {{ border-top: 2px solid var(--border); }}
  td.share {{ position: relative; width: 140px; text-align: left; }}
  .bar {{ display: inline-block; height: 8px; width: var(--w); background: var(--accent);
          border-radius: 4px; vertical-align: middle; max-width: 90px; }}
  .pct {{ color: var(--text-secondary); font-size: 12px; margin-left: 6px; }}
  .qrow {{ display: flex; gap: 12px; align-items: flex-end; height: 190px; }}
  .qcol {{ flex: 1; display: flex; flex-direction: column; justify-content: flex-end;
           align-items: center; height: 100%; }}
  .qbar {{ width: 100%; height: var(--h); background: var(--s6); border-radius: 4px 4px 0 0; }}
  {"".join(f".qbar[style*='--step:{i}']{{background:var(--s{i});}}" for i in range(4, 12))}
  .qval {{ font-weight: 650; font-size: 14px; margin-top: 8px; font-variant-numeric: tabular-nums; }}
  .qlab {{ color: var(--text-secondary); font-size: 12px; }}
  .qpct {{ color: var(--text-muted); font-size: 12px; }}
  .legend {{ display: flex; align-items: center; gap: 8px; margin: 12px 0 0;
             color: var(--text-muted); font-size: 12px; }}
  .swatches {{ display: flex; gap: 2px; }}
  .sw {{ width: 20px; height: 10px; border-radius: 2px; }}
  {"".join(f".sw.i{i}{{background:var(--s{i});}}" for i in range(len(RAMP_LIGHT)))}
  .notes {{ background: var(--surface-2); border: 1px solid var(--border);
            border-radius: 10px; padding: 18px 22px; color: var(--text-secondary); }}
  .notes li {{ margin: 8px 0; }}
  .notes strong {{ color: var(--text-primary); }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.9em;
          background: var(--surface-1); padding: 1px 5px; border-radius: 4px;
          border: 1px solid var(--border); }}
</style>
<div class="viz-root">
  <h1>llm_pipeline_v1_1 &mdash; projected yield at 10,364 WARCs</h1>
  <p class="sub">A random sample of the in-flight extraction &mdash; {report['measured_shards']:,.0f}
  batch shards across {n:,} WARCs, {report['coverage_pct']:.1f}% of the {report['total_shards']:,.0f}
  the finished run will hold &mdash; scored with the same WebOrganizer topic classifier and calibrated
  quality scorer used for the grid_v1 corpora, then scaled &times;{report['scale']:.1f} by shard count
  to the full {pool:,}-WARC pool.</p>

  <div class="hero">
    <div class="stat"><div class="v">{fmt_tokens(grand)}</div>
      <div class="k">projected llama3 tokens</div>
      <div class="n">95% CI {fmt_tokens(lo_total)} &ndash; {fmt_tokens(hi_total)}</div></div>
    <div class="stat"><div class="v">{sum(sum(r) for r in docs) / 1e6:.1f}M</div>
      <div class="k">projected documents</div>
      <div class="n">{grand / max(sum(sum(r) for r in docs), 1):.0f} tokens/doc</div></div>
    <div class="stat"><div class="v">{fmt_tokens(sampled_tokens)}</div>
      <div class="k">measured so far</div>
      <div class="n">{sampled_docs / 1e6:.2f}M docs over {n:,} WARCs</div></div>
    <div class="stat"><div class="v">{sum(1 for r in tokens for v in r if v > 0)}/120</div>
      <div class="k">cells populated</div>
      <div class="n">24 topics &times; 5 quality buckets</div></div>
  </div>

  <h2>Projected tokens by topic &times; quality</h2>
  <div class="scroll"><table>
    <thead>{head}</thead>
    <tbody>{''.join(rows)}{foot}</tbody>
  </table></div>
  <div class="legend"><span>fewer tokens</span>
    <span class="swatches">{''.join(f"<span class='sw i{i}'></span>" for i in range(len(RAMP_LIGHT)))}</span>
    <span>more</span><span>&mdash; log scale; the grid spans four orders of magnitude,
    so every cell also carries its number.</span></div>

  <h2>Token mass by quality bucket</h2>
  <div class="qrow">{qshare}</div>

  <h2>How to read this</h2>
  <div class="notes"><ul>
    <li><strong>This is a linear extrapolation of a pre-dedup corpus.</strong> The sample is
      raw extraction output. The full run will still go through dedup and decontamination,
      which remove documents, and near-duplicate rates <em>rise</em> with corpus size &mdash;
      so treat {fmt_tokens(grand)} as an upper bound on trainable tokens, not a forecast of them.</li>
    <li><strong>Tokens are llama3 tokens</strong>, the tokenizer every curation baseline uses.
      Each cell's exact character count is converted with a chars-per-token ratio measured on a
      systematic subsample of that same cell, so topics that tokenize densely are not flattened
      by a global average.</li>
    <li><strong>The interval is a bootstrap over WARCs, not documents.</strong> Documents inside
      one WARC are correlated, so a per-document interval would be far too tight. A
      finite-population correction is applied for sampling {n:,} of {pool:,}.</li>
    <li><strong>Sampling is uniform over completed WARCs.</strong> Every region processes WARCs
      in one shared random order, so a partially-finished run is still a uniform random
      subsample rather than a size-biased one.</li>
  </ul></div>
</div>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--projection", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = json.loads(pathlib.Path(args.projection).read_text())
    pathlib.Path(args.out).write_text(build_page(report))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
