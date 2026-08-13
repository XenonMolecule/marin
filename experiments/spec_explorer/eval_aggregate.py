# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate every eval family into one normalized long table keyed by run.

Scans all regions for the four result shapes and folds them into rows of
``(method, n_warcs, budget, dim, eval_family, task, value, higher_is_better)``,
joined on ``run_name_core`` (parsed by :mod:`catalog`). The frontend pivots
this table for the performance plots and the bolded winners table.

Families and where they live (per region bucket ``gs://marin-<region>/``):

- ``paloma`` / ``uncheatable`` — inside the training summary
  ``metadata/data_curation_{warc_scaling,fixed_model,10k_natural}_results/
  <run>.json`` under the ``eval`` dict. LOWER is better.
- ``core_v2`` — ``metadata/data_curation_10k_core_results/<run>_summary.json``
  -> ``dclm.Core_v2``. HIGHER is better.
- ``olmo_bpb`` — ``metadata/olmo_bpb_results/<run>/results.json`` -> macro +
  reader-side category groups + individual tasks. LOWER is better.
- ``olmes`` — the consolidated ``metadata/olmes_base_summary.csv`` (one small
  file) -> ``mean_olmes`` macro per run. HIGHER is better. We deliberately do
  NOT read the per-run ``olmes_base_results/<run>/results.json`` files: lm-eval
  dumps per-sample outputs, so each is ~98 MB (~64 GB across all runs, mostly
  cross-region egress) — far too expensive for a per-task accuracy we don't
  surface.

Reads are small JSON/CSV metadata files, listed once per region and downloaded
in a thread pool; the app caches the result behind an explicit refresh.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from google.api_core.exceptions import NotFound

from experiments.scaling_law_sweeps.olmo_bpb.olmo_bpb_tasks_set import (
    CODE_BPB,
    MATH_BPB,
    MT_MBPP_BPB,
    QA_LANG_BPB,
    QA_RC_MC_BPB,
)
from experiments.spec_explorer.catalog import EVAL_REGIONS, model_label, parse_run_name

logger = logging.getLogger(__name__)

# OLMo bpb reader-side category groups (mirrors plot_olmo_bpb_random_ladder).
OLMO_CATEGORIES: dict[str, list[str]] = {
    "Code_bpb": list(CODE_BPB) + list(MT_MBPP_BPB),
    "Math_bpb": list(MATH_BPB),
    "QA_bpb": list(QA_LANG_BPB) + list(QA_RC_MC_BPB),
    "QA_rc_bpb": list(QA_RC_MC_BPB),
    "QA_gen_bpb": list(QA_LANG_BPB),
}

# Per-family "higher is better". loss/bpb families are False.
FAMILY_HIGHER_IS_BETTER: dict[str, bool] = {
    "paloma": False,
    "uncheatable": False,
    "core_v2": True,
    "olmo_bpb": False,
    "olmes": True,
}

# Training summaries (paloma/uncheatable) are written to fixed us-central1
# prefixes, so we scan only that bucket for them (not all regions).
_LOSS_REGION = "us-central1"
_LOSS_SUBPATHS = (
    "metadata/data_curation_warc_scaling_results",
    "metadata/data_curation_fixed_model_results",
    "metadata/data_curation_10k_natural_results",
)
_CORE_SUBPATH = "metadata/data_curation_10k_core_results"
_OLMO_SUBPATH = "metadata/olmo_bpb_results"
# OLMES: one consolidated CSV (run_stem == run_name_core), not the 98 MB per-run dumps.
_OLMES_SUMMARY_CSV = "gs://marin-us-central1/metadata/olmes_base_summary.csv"


@dataclass(frozen=True)
class EvalRow:
    run_name_core: str
    method: str
    n_warcs: int
    budget: float
    dim: int
    model_label: str
    tag: str
    eval_family: str
    task: str
    value: float
    higher_is_better: bool

    def as_dict(self) -> dict:
        return {
            "run_name_core": self.run_name_core,
            "method": self.method,
            "n_warcs": self.n_warcs,
            "budget": self.budget,
            "dim": self.dim,
            "model_label": self.model_label,
            "tag": self.tag,
            "eval_family": self.eval_family,
            "task": self.task,
            "value": self.value,
            "higher_is_better": self.higher_is_better,
        }


def _client():
    from google.cloud import storage

    return storage.Client()


def _bucket(client, region: str):
    return client.bucket(f"marin-{region}")


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


# --- per-family metric extraction (json payload -> {(family, task): value}) ---


# Loss families whose diverged/near-init runs are dropped by the collapse
# post-pass (see _drop_collapsed_runs). Both collapse to a per-tokenizer init
# fingerprint that is byte-identical across unrelated methods.
_COLLAPSE_FAMILIES = ("uncheatable", "paloma")
# A run's macro within this of a collapse value is treated as the same near-init
# state. 1e-3 catches every observed single-method sibling (all within 4e-4 of a
# collapse value) while the nearest genuinely-undertrained run is >0.05 away.
_COLLAPSE_EPS = 1e-3


def _extract_loss(payload: dict) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    evald = payload.get("eval") or {}
    paloma: dict[str, float] = {}
    uncheatable: dict[str, float] = {}
    unc_macro: float | None = None
    for key, val in evald.items():
        if not _is_number(val):
            continue
        parts = key.split("/")
        # eval/paloma/<ds>/loss
        if len(parts) >= 4 and parts[1] == "paloma" and parts[-1] == "loss":
            paloma["/".join(parts[2:-1])] = float(val)
        # eval/uncheatable_eval/<ds>/bpb  and  eval/uncheatable_eval/macro_bpb
        elif len(parts) >= 3 and parts[1] == "uncheatable_eval":
            if parts[-1] == "bpb" and len(parts) >= 4:
                uncheatable["/".join(parts[2:-1])] = float(val)
            elif parts[-1] == "macro_bpb":
                unc_macro = float(val)

    for ds, v in paloma.items():
        out[("paloma", ds)] = v
    if paloma:  # reader-side macro across paloma datasets
        out[("paloma", "macro")] = sum(paloma.values()) / len(paloma)

    for ds, v in uncheatable.items():
        out[("uncheatable", ds)] = v
    if unc_macro is not None:
        out[("uncheatable", "macro")] = unc_macro
    return out


def _extract_core(payload: dict) -> dict[tuple[str, str], float]:
    dclm = payload.get("dclm") or {}
    out: dict[tuple[str, str], float] = {}
    for agg in ("Core_v2", "Core"):
        if _is_number(dclm.get(agg)):
            out[("core_v2", agg)] = float(dclm[agg])
    # Individual DCLM subtasks if present as a flat {task: score} dict.
    raw = dclm.get("raw_results")
    if isinstance(raw, dict):
        for task, score in raw.items():
            if _is_number(score):
                out[("core_v2", task)] = float(score)
    return out


def _extract_olmo(payload: dict) -> dict[tuple[str, str], float]:
    if payload.get("limit") is not None:  # capped smoke run
        return {}
    tasks = payload.get("tasks") or {}
    out: dict[tuple[str, str], float] = {}
    macro = (payload.get("averages") or {}).get("macro_bpb")
    if _is_number(macro):
        out[("olmo_bpb", "macro")] = float(macro)

    def bpb_of(task_key: str):
        b = tasks.get(task_key, {}).get("bpb")
        return float(b) if _is_number(b) else None

    for cat, keys in OLMO_CATEGORIES.items():
        vals = [b for k in keys if (b := bpb_of(k)) is not None]
        if vals:
            out[("olmo_bpb", cat)] = sum(vals) / len(vals)
    # Individual tasks too (for the "individual" plots/table).
    for task_key, metrics in tasks.items():
        b = metrics.get("bpb") if isinstance(metrics, dict) else None
        if _is_number(b):
            out[("olmo_bpb", f"task:{task_key}")] = float(b)
    return out


# --- listing + loading ---


@dataclass(frozen=True)
class _FileRef:
    region: str
    blob_name: str
    run_name_core: str


def _list_flat(client, region: str, subpath: str, strip_suffix: str) -> list[_FileRef]:
    """List ``{subpath}/<run>{strip_suffix}`` files; run = basename minus suffix."""
    refs: list[_FileRef] = []
    for blob in _bucket(client, region).list_blobs(prefix=f"{subpath}/"):
        base = blob.name.rsplit("/", 1)[-1]
        if not base.startswith("curation-") or not base.endswith(strip_suffix):
            continue
        run = base[: -len(strip_suffix)]
        refs.append(_FileRef(region, blob.name, run))
    return refs


def _list_nested(client, region: str, subpath: str) -> list[_FileRef]:
    """List ``{subpath}/<run>/`` run-dirs via a delimiter (cheap: enumerates
    run directories, not every file under them), then target ``results.json``.

    Missing ``results.json`` under a run-dir is tolerated: the download step
    quietly skips a 404. This avoids the recursive full-object listing that
    made a naive scan enumerate every eval artifact in every run.
    """
    it = _bucket(client, region).list_blobs(prefix=f"{subpath}/", delimiter="/")
    for _ in it:  # consume the page iterator so `prefixes` is populated
        pass
    refs: list[_FileRef] = []
    for run_prefix in it.prefixes:  # e.g. "metadata/olmo_bpb_results/curation-…/"
        run = run_prefix.rstrip("/").rsplit("/", 1)[-1]
        if not run.startswith("curation-"):
            continue
        refs.append(_FileRef(region, f"{run_prefix}results.json", run))
    return refs


# Groups read via the list-files + download-JSON machinery (olmes uses a CSV).
_FAMILY_EXTRACTORS = {
    "loss": _extract_loss,
    "core_v2": _extract_core,
    "olmo_bpb": _extract_olmo,
}


def _list_group(client, group: str) -> list[_FileRef]:
    refs: list[_FileRef] = []
    # loss lives only in the fixed us-central1 bucket; the others scatter.
    regions = (_LOSS_REGION,) if group == "loss" else EVAL_REGIONS
    for region in regions:
        try:
            if group == "loss":
                for sub in _LOSS_SUBPATHS:
                    refs += _list_flat(client, region, sub, ".json")
            elif group == "core_v2":
                refs += _list_flat(client, region, _CORE_SUBPATH, "_summary.json")
            elif group == "olmo_bpb":
                refs += _list_nested(client, region, _OLMO_SUBPATH)
        except Exception as e:  # a missing prefix in one region is not fatal
            logger.warning("listing %s in %s failed: %s", group, region, e)
    return refs


def _olmes_rows(client) -> list[EvalRow]:
    """OLMES macro per run from the single consolidated CSV (cheap)."""
    import csv
    import io

    bucket_name, blob_name = _OLMES_SUMMARY_CSV.replace("gs://", "").split("/", 1)
    try:
        text = client.bucket(bucket_name).blob(blob_name).download_as_text()
    except NotFound:
        logger.warning("olmes summary CSV not found at %s; skipping olmes", _OLMES_SUMMARY_CSV)
        return []
    rows: list[EvalRow] = []
    for rec in csv.DictReader(io.StringIO(text)):
        ident = parse_run_name(rec.get("run_stem", ""))
        mean = rec.get("mean_olmes")
        if ident is None or not mean:
            continue
        try:
            value = float(mean)
        except ValueError:
            continue
        rows.append(
            EvalRow(
                run_name_core=ident.run_name_core,
                method=ident.method,
                n_warcs=ident.n_warcs,
                budget=ident.budget,
                dim=ident.dim,
                model_label=model_label(ident.dim),
                tag=ident.tag,
                eval_family="olmes",
                task="macro",
                value=value,
                higher_is_better=FAMILY_HIGHER_IS_BETTER["olmes"],
            )
        )
    return rows


def _collapsed_runs(rows: list[EvalRow], family: str) -> set[str]:
    """run_name_cores whose ``family`` eval is a diverged/near-init collapse.

    A collapse value is a macro shared byte-identically across >=2 unrelated
    methods — impossible for real models trained on different data, so it is the
    tokenizer-determined init fingerprint. Runs whose macro is within
    ``_COLLAPSE_EPS`` of any collapse value (so single-method siblings that hit the
    same init state are caught too) are flagged; genuinely undertrained runs have
    unique macros far from the fingerprint and are kept.
    """
    macro_by_run: dict[str, float] = {}
    methods_by_value: dict[float, set[str]] = {}
    for r in rows:
        if r.eval_family == family and r.task == "macro":
            macro_by_run[r.run_name_core] = r.value
            methods_by_value.setdefault(r.value, set()).add(r.method)
    collapse_values = [v for v, methods in methods_by_value.items() if len(methods) >= 2]
    if not collapse_values:
        return set()
    return {run for run, macro in macro_by_run.items() if any(abs(macro - cv) < _COLLAPSE_EPS for cv in collapse_values)}


def _drop_collapsed_runs(seen: dict[tuple[str, str, str], EvalRow]) -> dict[tuple[str, str, str], EvalRow]:
    """Drop every collapse family's rows for runs whose that-family eval collapsed."""
    rows = list(seen.values())
    for family in _COLLAPSE_FAMILIES:
        collapsed = _collapsed_runs(rows, family)
        if not collapsed:
            continue
        before = len(seen)
        seen = {k: v for k, v in seen.items() if not (v.eval_family == family and v.run_name_core in collapsed)}
        logger.info("%s: dropped %d diverged/near-init runs (%d -> %d rows)", family, len(collapsed), before, len(seen))
    return seen


def build_rows(groups: tuple[str, ...] = ("loss", "core_v2", "olmo_bpb", "olmes")) -> list[dict]:
    """Scan the requested eval groups across all regions -> normalized rows.

    ``loss`` yields both the ``paloma`` and ``uncheatable`` families. Rows are
    deduped on ``(run_name_core, eval_family, task)`` (a run re-homed across
    regions writes identical values).
    """
    client = _client()
    seen: dict[tuple[str, str, str], EvalRow] = {}

    for group in groups:
        if group == "olmes":
            for row in _olmes_rows(client):
                seen.setdefault((row.run_name_core, row.eval_family, row.task), row)
            logger.info("olmes: %d macro rows from consolidated CSV", sum(1 for k in seen if k[1] == "olmes"))
            continue
        refs = _list_group(client, group)
        # Only load files whose run_name_core parses to a known method.
        keep = [(r, parse_run_name(r.run_name_core)) for r in refs]
        keep = [(r, ident) for r, ident in keep if ident is not None]
        logger.info("%s: %d files, %d with parseable run names", group, len(refs), len(keep))

        extractor = _FAMILY_EXTRACTORS[group]

        def _load(item, extractor=extractor):  # bind loop var into the closure
            ref, ident = item
            try:
                text = _bucket(client, ref.region).blob(ref.blob_name).download_as_text()
            except NotFound:
                return ident, {}  # run-dir without a results.json — expected, quiet
            except Exception as e:
                logger.warning("read %s failed: %s", ref.blob_name, e)
                return ident, {}
            import json

            try:
                metrics = extractor(json.loads(text))
            except Exception as e:
                logger.warning("parse %s failed: %s", ref.blob_name, e)
                return ident, {}
            return ident, metrics

        with ThreadPoolExecutor(max_workers=32) as ex:
            for ident, metrics in ex.map(_load, keep):
                for (family, task), value in metrics.items():
                    key = (ident.run_name_core, family, task)
                    if key in seen:
                        continue
                    seen[key] = EvalRow(
                        run_name_core=ident.run_name_core,
                        method=ident.method,
                        n_warcs=ident.n_warcs,
                        budget=ident.budget,
                        dim=ident.dim,
                        model_label=model_label(ident.dim),
                        tag=ident.tag,
                        eval_family=family,
                        task=task,
                        value=value,
                        higher_is_better=FAMILY_HIGHER_IS_BETTER[family],
                    )
    seen = _drop_collapsed_runs(seen)
    rows = [r.as_dict() for r in seen.values()]
    logger.info("built %d eval rows", len(rows))
    return rows


def summarize(rows: list[dict]) -> dict:
    """Small facets object for the frontend: methods, budgets, dims, families, tasks."""
    methods = sorted({r["method"] for r in rows})
    budgets = sorted({r["budget"] for r in rows})
    dims = sorted({r["dim"] for r in rows})
    ns = sorted({r["n_warcs"] for r in rows})
    families = sorted({r["eval_family"] for r in rows})
    tasks: dict[str, list[str]] = {}
    for r in rows:
        tasks.setdefault(r["eval_family"], [])
        if r["task"] not in tasks[r["eval_family"]]:
            tasks[r["eval_family"]].append(r["task"])
    for fam in tasks:
        tasks[fam].sort()
    return {
        "methods": methods,
        "budgets": budgets,
        "dims": dims,
        "n_warcs": ns,
        "families": families,
        "tasks": tasks,
        "row_count": len(rows),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("--groups", nargs="+", default=["loss", "core_v2", "olmo_bpb", "olmes"])
    args = p.parse_args()
    rows = build_rows(tuple(args.groups))
    print(json.dumps(summarize(rows), indent=2, default=str))


if __name__ == "__main__":
    main()
