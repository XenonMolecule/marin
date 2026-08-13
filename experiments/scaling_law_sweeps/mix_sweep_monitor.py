# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Health check for the OLMIX-mixture 10k arms (`dclm_10k_mix`, `high_quality_10k_mix`).

Emits ONE line per cycle plus an ALERT line for anything worth acting on. Written so that
**silence means healthy** only because every failure mode I can name is checked explicitly:

* jobs that failed / were cancelled (not just "how many are running")
* runs whose step count has not advanced since the previous cycle while still marked running
* non-finite or exploding loss
* the mixture silently not applying -- a mix run whose config carries one training component
  instead of the 96 grid cells would train the un-mixed corpus and look completely normal
* coverage: results landed vs. the 38-cell grid, so a coordinator that stops dispatching
  (the cursor gap that stranded 5 swarm runs) shows up as "live children < remaining work"

Usage: `python -m experiments.scaling_law_sweeps.mix_sweep_monitor [--once]`
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import fsspec
import wandb

from experiments.scaling_law_sweeps.curation_plan import METHODS as CURATION_METHODS
from experiments.scaling_law_sweeps.dclm_core.build_mix_core_manifest import _olmo_canonical_done
from experiments.scaling_law_sweeps.repair_stale_final_eval import audit_one

# Both mixture sweeps. Their run names all contain "10k_mix", so the wandb regex and the
# child-name SQL patterns already match every arm without further changes.
METHODS = (
    "dclm_10k_mix",
    "high_quality_10k_mix",
    "dclm_10k_mix_lambda0p01",
    "high_quality_10k_mix_lambda0p01",
    "dclm_10k_mix_corev2_lambda0p05",
    "dclm_10k_mix_corev2_lambda0p01",
    "high_quality_10k_mix_corev2_lambda0p05",
    "high_quality_10k_mix_corev2_lambda0p01",
    "dclm_10k_mix_blend_lambda0p01",
    "high_quality_10k_mix_blend_lambda0p01",
)
GRID_CELLS = 38
# Cells deliberately abandoned (user directive 2026-08-03: drop anything needing >24h). All
# four are the 2e+21 budget: dclm+hq d2432-L24-B256, dclm+hq d3584-L35-B128. Without this the
# sweep can never reach 2*GRID_CELLS, so the dispatch-gap check fires every cycle forever once
# training ends -- an alert that cannot be cleared trains you to ignore all of them.
# Their rolling temps stay resumable until ~2026-08-17 if they are ever revived.
DROPPED_CELLS = 4
# lambda=0.05 arms: 2*38 - 4 dropped = 72. lambda=0.01 arms launched with --max-budget 9e20,
# which excludes the same 1.8e21 rung up front, so 36 per arm = 72.
# lambda0.05: 72. lambda0.01: 72 launched (many later culled by user directive, so `done` will
# stop short of this by design). corev2: 4 arms x 24 cells under --max-budget 9e19.
SWEEP_TARGET = (
    (2 * GRID_CELLS - DROPPED_CELLS) + 2 * (GRID_CELLS - DROPPED_CELLS // 2) + 96 + 48
)  # + blend-steered sweep (2 arms x 24 cells @ --max-budget 9e19)
# Compact labels for the per-arm counts on the status line.
# Arms evaluated on BLEnD ONLY -- never Core_v2 / olmo_bpb / MMLU.
BLEND_ONLY_METHODS = (
    "dclm_10k_mix_blend_lambda0p01",
    "high_quality_10k_mix_blend_lambda0p01",
)
SHORT_NAMES = {
    "dclm_10k_mix": "dclm",
    "high_quality_10k_mix": "hq",
    "dclm_10k_mix_lambda0p01": "dclm-L01",
    "high_quality_10k_mix_lambda0p01": "hq-L01",
    "dclm_10k_mix_corev2_lambda0p05": "dclm-C05",
    "dclm_10k_mix_corev2_lambda0p01": "dclm-C01",
    "high_quality_10k_mix_corev2_lambda0p05": "hq-C05",
    "high_quality_10k_mix_corev2_lambda0p01": "hq-C01",
    "dclm_10k_mix_blend_lambda0p01": "dclm-B01",
    "high_quality_10k_mix_blend_lambda0p01": "hq-B01",
}
# Every coordinator that submits TRAINING children. Targeted relaunches run under their own
# coordinator (`olmix-mix-resume-N`), so a monitor scoped to the original one reports run=0 and
# stays silent while a whole relaunch wave dies -- the same blind spot that hid the olmo parents.
# Matched as `<prefix>%/%`, so add a prefix here whenever a new training coordinator is launched.
COORD_PREFIXES = (
    "/michaelryan/olmix-mix-10k-coord",
    "/michaelryan/olmix-mix-resume",
    "/michaelryan/olmix-lambda0p01",
    "/michaelryan/olmix-corev2-coord",
    "/michaelryan/olmix-blendmix",
)
RESULTS = "gs://marin-us-central1/metadata/data_curation_10k_natural_results/"
# Levanter block size the mix arms must be using; 2048 would mean the mixture didn't apply.
EXPECTED_BLOCK = 65535
# Alert only well outside the observed healthy tail. The worst healthy run measured 51 min
# stale and then resumed; a 30 min threshold flagged 6-19 runs per cycle, i.e. pure noise.
# Consecutive cycles with zero new completions (while jobs run) before flagging.
NO_PROGRESS_CYCLES = 4


def _iris_counts(sql: str) -> dict[str, int] | None:
    """Run one controller query. None on failure -- a failed read must not look like data."""
    try:
        p = subprocess.run(
            [
                "/Users/michaelryan/Documents/School/Stanford/Research/marin/.venv/bin/iris",
                "--cluster",
                "marin",
                "query",
                sql,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return None
    if p.returncode != 0:
        return None
    out: dict[str, int] = {}
    for line in p.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-1].isdigit() and parts[0].isdigit():
            out[parts[0]] = int(parts[-1])
    return out


def _done_counts() -> dict[str, int] | None:
    try:
        p = subprocess.run(["gcloud", "storage", "ls", RESULTS], capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired:
        return None
    if p.returncode != 0:
        return {} if "matched no objects" in (p.stderr or "").lower() else None
    # Anchor on the line end. An unanchored `...\.json` also matches `...json.bak`, and a
    # repair pass that wrote backups beside the results inflated dclm 26->35 and hq 29->41,
    # reporting a finished sweep (76/76) that was really at 55/76.
    return {
        m: sum(
            1 for line in p.stdout.splitlines() if re.search(rf"curation-{m}-expFM_natural-[^/]+\.json$", line.rstrip())
        )
        for m in METHODS
    }


def wandb_snapshot(api) -> tuple[dict[str, tuple[str, int, float, float]], list[str]]:
    """(run -> (state, step, loss, minutes_since_last_log)) plus loss alerts.

    Staleness comes from the last logged row's `_timestamp`, NOT from diffing steps between
    monitor cycles. Diffing produced constant false alarms: these runs log sparsely and in
    bursts, so a large fraction is legitimately between logs at any instant. Measured on the
    live sweep: of 57 running jobs, 37 were fresh (<10 min), 20 over 10 min, and 6 over 30 min
    -- a CONTINUOUS distribution, which is what slow bursty logging looks like. A genuinely
    hung population would sit far away from the healthy tail instead of blending into it.

    One run did go 51 minutes without logging and then resumed on its own, nearly doubling its
    step count -- so even "stale for most of an hour" is not proof of a hang here, and acting on
    it would have killed a healthy run.
    """
    alerts: list[str] = []
    runs: dict[str, tuple[str, int, float, float]] = {}
    now = time.time()
    for r in api.runs("marin-community/marin", filters={"display_name": {"$regex": "10k_mix"}}, per_page=200):
        s = r.summary
        step = int(s.get("_step") or 0)
        loss = s.get("train/loss")
        loss = float(loss) if isinstance(loss, (int, float)) else float("nan")
        ts = s.get("_timestamp")
        stale_min = (now - float(ts)) / 60.0 if isinstance(ts, (int, float)) else float("nan")
        runs[r.name] = (r.state, step, loss, stale_min)
        if loss == loss and (loss > 20 or loss < 0):  # NaN compares false; catch explosions
            alerts.append(f"ALERT loss={loss:.3f} in {r.name}")
        if loss != loss and step > 50:
            alerts.append(f"ALERT non-finite loss in {r.name} at step {step}")
    return runs, alerts


def mixture_applied(api) -> list[str]:
    """One config probe per method: the mixture must be present, or the arm is a duplicate base run."""
    alerts = []
    for m in METHODS:
        # ANCHORED: an unanchored "dclm_10k_mix" also matches "dclm_10k_mix_lambda0p01", so the
        # probe could sample the wrong arm and compare its component count against the other
        # arm's mixture -- a guaranteed false "MIXTURE DID NOT APPLY".
        rs = list(
            api.runs("marin-community/marin", filters={"display_name": {"$regex": f"^curation-{m}-expFM"}}, per_page=1)
        )
        if not rs:
            continue
        cfg = json.loads(json.dumps(rs[0].config, default=str))
        data = cfg.get("data") or {}
        grid = [k for k, v in (data.get("train_weights") or {}).items() if v and v > 0 and "__c" in k]
        block = data.get("mixture_block_size")
        # Expected count comes from the method's OWN mixture, not a constant: a more
        # concentrated solve pushes more cells under the 1/block floor, where `load_weights`
        # drops them. lambda=0.01 yields 90 (dclm) / 92 (hq) vs 96 at lambda=0.05, so a
        # hardcoded 96 would report "MIXTURE DID NOT APPLY" on a perfectly correct arm.
        want = len(CURATION_METHODS[m].load_weights())
        if len(grid) != want or block != EXPECTED_BLOCK:
            alerts.append(
                f"ALERT {m}: MIXTURE DID NOT APPLY -- grid_components={len(grid)} "
                f"(want {want}), block_size={block} (want {EXPECTED_BLOCK})"
            )
    return alerts


def core_eval_backlog() -> tuple[int, int, int] | None:
    """(scored, in_flight, failed) DCLM Core_v2 evals for the mix arms.

    Scores land in the CHECKPOINT's own bucket (in-region writes), so this has to look in every
    region rather than one central prefix. Returns None on a failed read.
    """
    scored = 0
    for bucket in ("marin-us-central1", "marin-us-east5", "marin-us-west4", "marin-eu-west4"):
        try:
            p = subprocess.run(
                ["gcloud", "storage", "ls", f"gs://{bucket}/metadata/data_curation_10k_core_results/"],
                capture_output=True,
                text=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            return None
        if p.returncode != 0 and "matched no objects" not in (p.stderr or "").lower():
            return None
        # ONLY `_summary.json`. A failed eval child also writes `<run>_error.txt` into this same
        # prefix, so a bare "10k_mix" substring match counts failures as successes -- it inflated
        # scored by 7 and drove READY to 0 while 5 runs actually still needed a wave.
        scored += sum(
            1 for line in p.stdout.splitlines() if "10k_mix" in line and line.rstrip().endswith("_summary.json")
        )
    states = _iris_counts(
        # Match the CHILD's name, not the coordinator prefix. Every new eval coordinator
        # (`olmix-mixcore-wN`, `olmix-allcore-wN`, ...) otherwise needs a pattern added here, and
        # three times now a whole wave has been invisible -- reported inflight=0 while hundreds of
        # children ran. Child names always embed `<suite>-curation-<method>`, so this is stable.
        "SELECT state, COUNT(*) n FROM jobs WHERE job_id LIKE '%/dclm-core-curation-%10k_mix%' GROUP BY state"
    )
    if states is None:
        return None
    inflight = states.get("1", 0) + states.get("3", 0)
    # Eval children are a SEPARATE job tree from the training coordinator, so the training
    # failure count says nothing about them. Seven died at once to a transient dataset-load
    # error ("Failed to load task hellaswag_zeroshot_0shot") during a network blip and the
    # monitor stayed silent, because it only ever looked at `olmix-mix-10k-coord`.
    failed = states.get("5", 0) + states.get("6", 0)
    return scored, inflight, failed


def olmo_bpb_status() -> tuple[int, int, int] | None:
    """(scored, in_flight, failed) olmo_bpb evals for the mix arms.

    A THIRD job tree (`olmix-mix-olmobpb*`, `olmix-baseline-mmlu-merge`). Each eval suite lives
    under its own coordinator, so a monitor scoped to one tree is blind to the others -- that is
    exactly how 7 Core-eval children died unnoticed. Every suite we launch gets a counter here.
    """
    scored = 0
    fs = fsspec.filesystem("gs")
    for bucket in ("marin-us-central1", "marin-us-east5", "marin-us-west4", "marin-eu-west4"):
        try:
            p = subprocess.run(
                ["gcloud", "storage", "ls", f"gs://{bucket}/metadata/olmo_bpb_results/"],
                capture_output=True,
                text=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            return None
        if p.returncode != 0 and "matched no objects" not in (p.stderr or "").lower():
            return None
        # A run dir is NOT a scored run: the MMLU `--merge` pass CREATES results.json, so a run
        # that never had the canonical suite shows up here with 4 tasks. Counting the dir hid 4
        # such runs and drove READY to 0, which would have left them permanently un-evaluated.
        for line in p.stdout.splitlines():
            if "10k_mix" not in line:
                continue
            if _olmo_canonical_done(fs, f"{line.strip().rstrip('/')}/results.json".removeprefix("gs://")):
                scored += 1
    # Mix-arm children only for the in-flight number, so READY reflects mix coverage. The
    # baseline MMLU merge is a different population and would inflate it.
    mix_states = _iris_counts(
        # Canonical-suite children only: olmobpb children whose name does NOT carry an -mmlu
        # suffix. Both suites share the `olmobpb-` prefix, so without the exclusion the MMLU
        # wave inflates the canonical in-flight count and READY goes negative-ish.
        "SELECT state, COUNT(*) n FROM jobs WHERE job_id LIKE '%/olmobpb-curation-%10k_mix%' "
        "AND job_id NOT LIKE '%-mmlu%' GROUP BY state"
    )
    # Mix arms ONLY, same as the in-flight query. Counting every olmobpb child cluster-wide
    # pulled in unrelated experiments' historical failures and reported fail=73 for a mix sweep
    # that had 7.
    all_states = _iris_counts(
        "SELECT state, COUNT(*) n FROM jobs WHERE job_id LIKE '%/olmobpb-curation-%10k_mix%' "
        "AND job_id NOT LIKE '%-mmlu%' GROUP BY state"
    )
    if mix_states is None or all_states is None:
        return None
    mix_inflight = mix_states.get("1", 0) + mix_states.get("3", 0)
    failed = all_states.get("5", 0) + all_states.get("6", 0)
    return scored, mix_inflight, failed


def mmlu_status() -> tuple[int, int, int] | None:
    """(scored, in_flight, failed) MMLU merge-pass evals for the mix arms.

    A FOURTH job tree (`olmix-mix-mmlu%`). This pass runs with `--merge`, so it folds 4 MMLU
    category tasks into a results.json that ALREADY EXISTS -- meaning `olmo_bpb_status`'s
    results.json counter cannot see it move, and a whole 58-child wave would run (or die)
    completely invisibly. Completion is the child's `_mmlu.done` marker and nothing else.
    """
    scored = 0
    for bucket in ("marin-us-central1", "marin-us-east5", "marin-us-west4", "marin-eu-west4"):
        try:
            p = subprocess.run(
                ["gcloud", "storage", "ls", f"gs://{bucket}/metadata/olmo_bpb_results/**/_mmlu.done"],
                capture_output=True,
                text=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            return None
        if p.returncode != 0 and "matched no objects" not in (p.stderr or "").lower():
            return None
        scored += sum(1 for line in p.stdout.splitlines() if "10k_mix" in line)
    states = _iris_counts(
        "SELECT state, COUNT(*) n FROM jobs WHERE job_id LIKE '%/olmobpb-curation-%10k_mix%-mmlu%' GROUP BY state"
    )
    if states is None:
        return None
    return scored, states.get("1", 0) + states.get("3", 0), states.get("5", 0) + states.get("6", 0)


def stale_summaries() -> list[str] | None:
    """Mix-arm summaries whose eval block is from a NON-final eval cycle.

    The children of `olmix-mix-10k-coord` were bundled 18.5h BEFORE the wandb-staleness fix
    landed in `run_curation_train_standalone`, so they can still write a summary holding an
    earlier eval cycle. `repair_stale_final_eval` is a one-shot batch pass: runs that finish
    after it are unchecked, which is exactly how a +0.031 BPB Paloma spike reached the
    published plots on high_quality_10k_mix 3e+19-d1536. Checking every cycle turns that from
    "spotted by eye in a figure" into an alert.
    """
    fs = fsspec.filesystem("gs")
    paths = [p for p in fs.ls(RESULTS, detail=False) if p.endswith(".json") and any(m in p for m in METHODS)]
    with ThreadPoolExecutor(16) as ex:
        results = list(ex.map(audit_one, [(fs, p) for p in paths]))
    return [r["path"].rsplit("/", 1)[-1] for r in results if r and r["status"] == "STALE"]


def blend_status() -> tuple[int, int, int] | None:
    """(scored, in_flight, failed) BLEnD evals -- mix arms AND the six un-mixed baselines.

    BLEnD is a `--merge` pass like MMLU, so completion is the `_blend.done` marker; results.json
    already exists for most of these runs. Counted across ALL methods, not just the mixture
    arms, because the baseline sweep is the point of comparison for the blend-steered arms.
    """
    scored = 0
    for bucket in ("marin-us-central1", "marin-us-east5", "marin-us-west4", "marin-eu-west4"):
        try:
            p = subprocess.run(
                ["gcloud", "storage", "ls", f"gs://{bucket}/metadata/olmo_bpb_results/**/_blend.done"],
                capture_output=True,
                text=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            return None
        if p.returncode != 0 and "matched no objects" not in (p.stderr or "").lower():
            return None
        scored += sum(1 for line in p.stdout.splitlines() if line.strip().endswith("_blend.done"))
    states = _iris_counts(
        "SELECT state, COUNT(*) n FROM jobs WHERE job_id LIKE '%/olmobpb-curation-%-blend%' GROUP BY state"
    )
    if states is None:
        return None
    return scored, states.get("1", 0) + states.get("3", 0), states.get("5", 0) + states.get("6", 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--interval", type=float, default=1800.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    checked_mixture = False
    prev_done, stale_cycles = -1, 0
    prev_core_failed: int | None = None
    prev_failed: int | None = None
    prev_olmo_failed: int | None = None
    prev_mmlu_failed: int | None = None

    while True:
        # A FRESH Api every cycle: `wandb.Api` memoises its run objects, so reusing one across
        # cycles replays the first snapshot forever. Measured: reusing it showed 0/70 runs
        # advancing over 45s while a fresh client showed 58/70. That turns the stall detector
        # into a false-alarm generator -- it fired "66 runs not advancing" while the completion
        # count was simultaneously climbing 2 -> 6, which is self-contradictory.
        stamp = time.strftime("%H:%MZ", time.gmtime())
        where = " OR ".join(f"job_id LIKE '{c}%/%'" for c in COORD_PREFIXES)
        jobs = _iris_counts(f"SELECT state, COUNT(*) n FROM jobs WHERE {where} GROUP BY state")
        done = _done_counts()

        # W&B is a THIRD-PARTY network dependency and the least reliable input here. An
        # unguarded `ConnectTimeout` to api.wandb.ai killed the whole monitor process mid-sweep,
        # so the sweep ran unwatched until a human noticed the silence -- strictly worse than a
        # cycle with no wandb data. Degrade to `wandb=?` and keep the loop alive; the Iris/GCS
        # signals (which are the ones that actually gate decisions) are unaffected.
        # The default 19s graphql timeout is what expired, hence the wider one.
        runs: dict[str, tuple[str, int, float, float]] = {}
        alerts: list[str] = []
        wandb_ok = True
        try:
            api = wandb.Api(api_key=os.environ["WANDB_API_KEY"], timeout=60)
            runs, alerts = wandb_snapshot(api)
            # The mixture only needs proving once -- config is fixed at submit time.
            if not checked_mixture:
                alerts += mixture_applied(api)
                checked_mixture = True
        except Exception as e:
            wandb_ok = False
            alerts.append(
                f"WARN wandb unavailable this cycle ({type(e).__name__}) -- job/GCS counts below are still valid"
            )

        if jobs is None or done is None:
            print(f"{stamp} WARN unreliable read (iris/gcs query failed) -- holding, not reporting counts", flush=True)
        else:
            failed = jobs.get("5", 0) + jobs.get("6", 0)
            running, pending = jobs.get("3", 0), jobs.get("1", 0)
            # Alert on NEW failures only. Firing on `failed > 0` re-reports the same dead child
            # every 30 minutes forever, which is how a real alert gets tuned out.
            if prev_failed is None:
                prev_failed = failed
            if failed > prev_failed:
                alerts.append(f"ALERT {failed - prev_failed} NEW training child failure(s) ({failed} total)")
            prev_failed = failed
            total_done = sum(done.values())
            # Core_v2 / olmo_bpb / MMLU are deliberately NOT run on the blend-steered arms (user
            # directive 2026-08-06: "we just need to eval them on Blend not OLMO or DCLM Core").
            # Their denominator is therefore total_done MINUS those arms, otherwise READY floors
            # at 48 forever and fires a wave alert every cycle that no wave can ever clear.
            bpb_only_done = sum(done.get(m, 0) for m in BLEND_ONLY_METHODS)
            canonical_done = total_done - bpb_only_done
            # Stalled = still 'running' in wandb but step unchanged since last cycle.
            # NO wandb-staleness alert. It was tried at 30 min and 90 min and produced only
            # false positives: runs whose wandb rows are 109-124 min stale were verified to have
            # written checkpoints 4-26 min earlier, and one run sat "flat" for 51 min then resumed
            # and doubled its step count. wandb liveness simply does not track training liveness
            # here, so the honest coarse check is "completions are still happening".
            if total_done == prev_done and running > 0:
                stale_cycles += 1
                if stale_cycles >= NO_PROGRESS_CYCLES:
                    alerts.append(
                        f"ALERT no completions in {stale_cycles} cycles while {running} jobs run "
                        f"-- check checkpoint mtimes, not wandb"
                    )
            else:
                stale_cycles = 0
            live = sum(1 for _, (st, _, _, _) in runs.items() if st == "running")
            remaining = SWEEP_TARGET - total_done
            # `live` is 0 when wandb simply could not be reached, which is not a dispatch gap.
            if wandb_ok and remaining > 0 and live == 0 and running == 0:
                alerts.append(f"ALERT dispatch gap: {remaining} cells left but ZERO live children")
            steps = [s for _, s, _, _ in runs.values()]
            median_step = sorted(steps)[len(steps) // 2] if steps else 0
            wandb_str = f"wandb live={live} md_step={median_step}" if wandb_ok else "wandb=UNAVAILABLE"
            core = core_eval_backlog()
            ob = olmo_bpb_status()
            if ob is None:
                ob_str = "olmo=?"
            else:
                ob_scored, ob_flight, ob_failed = ob
                ob_ready = max(0, canonical_done - ob_scored - ob_flight)
                ob_str = f"olmo scored={ob_scored} inflight={ob_flight} fail={ob_failed} READY={ob_ready}"
                if ob_ready >= 10:
                    alerts.append(f"OLMO WAVE READY: {ob_ready} mix runs awaiting olmo_bpb -- launch the next wave")
                if prev_olmo_failed is None:
                    prev_olmo_failed = ob_failed
                if ob_failed > prev_olmo_failed:
                    alerts.append(
                        f"ALERT {ob_failed - prev_olmo_failed} NEW olmo_bpb child failure(s) "
                        f"({ob_failed} total) -- they do not retry; rebuild the manifest and relaunch"
                    )
                prev_olmo_failed = ob_failed
            if core is None:
                core_str = "core=?"
            else:
                scored, inflight, core_failed = core
                ready = max(0, canonical_done - scored - inflight)
                core_str = f"core scored={scored} inflight={inflight} fail={core_failed} READY={ready}"
                # First cycle only baselines -- on restart the historical failure count is not
                # news, and re-reporting it every time the monitor restarts trains you to ignore
                # the alert that matters.
                if prev_core_failed is None:
                    prev_core_failed = core_failed
                if core_failed > prev_core_failed:
                    alerts.append(
                        f"ALERT {core_failed - prev_core_failed} NEW Core-eval child failure(s) "
                        f"({core_failed} total) -- they do not retry; rebuild the manifest and relaunch"
                    )
                prev_core_failed = core_failed
                if ready >= 10:
                    alerts.append(f"WAVE READY: {ready} mix runs awaiting DCLM Core_v2 -- launch the next wave")
            stale = stale_summaries()
            if stale:
                alerts.append(
                    f"ALERT {len(stale)} STALE mix summary/summaries (non-final eval cycle) -- "
                    f"corrupts the plots AND the OLMIX fit; run repair_stale_final_eval --apply: {stale[:4]}"
                )
            mm = mmlu_status()
            if mm is None:
                mm_str = "mmlu=?"
            else:
                mm_scored, mm_flight, mm_failed = mm
                mm_str = f"mmlu scored={mm_scored} inflight={mm_flight} fail={mm_failed}"
                if prev_mmlu_failed is None:
                    prev_mmlu_failed = mm_failed
                if mm_failed > prev_mmlu_failed:
                    alerts.append(
                        f"ALERT {mm_failed - prev_mmlu_failed} NEW MMLU child failure(s) "
                        f"({mm_failed} total) -- they do not retry; rebuild with --suite mmlu and relaunch"
                    )
                prev_mmlu_failed = mm_failed
            per_arm = " ".join(f"{SHORT_NAMES[m]}:{done.get(m, 0)}" for m in METHODS)
            bl = blend_status()
            bl_str = "blend=?" if bl is None else f"blend scored={bl[0]} inflight={bl[1]} fail={bl[2]}"
            print(
                f"{stamp} mix-sweep | done {total_done}/{SWEEP_TARGET} "
                f"({per_arm}) | "
                f"jobs run={running} pend={pending} fail={failed} | "
                f"{wandb_str} | "
                f"{core_str} | {ob_str} | {mm_str} | {bl_str}",
                flush=True,
            )

            prev_done = total_done

        for a in alerts:
            print(a, flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
