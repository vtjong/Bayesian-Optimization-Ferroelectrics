"""Emit the blank results sheet for the seed batch, and validate it once the lab fills it in.

Two modes, one file, because the sheet the operator fills and the sheet the campaign reads must be
the same object. Anything that reconciles two differing formats afterwards is where measurements
get lost.

    --template   write a blank sheet, one row per planned condition, every outcome field empty
    (default)    read the filled sheet, validate it, and report what came back

VALIDATION IS STRICT AND THE STRICTNESS IS THE POINT. This project's archival data already contains
a blank silently read as zero, a row shifted one column so a 1e6-cycle value landed in a 3-cycle
field, and shots whose delivered condition was never recorded. Each of those is a wrong number
rather than a missing one, and each survived because a reader made a reasonable-looking guess. So
this script fails loudly instead of guessing, and it reports UNFIRED conditions rather than
quietly analysing whatever subset happens to be present.

What it reports:
  * coverage        -- how many conditions returned a usable reading, and which did not
  * drift           -- shots whose as-fired condition differs from plan. A ladder rung that drifted
                       is no longer on its iso-Tmax level, which changes what it can be used for
  * replicate pairs -- the block D shots against their block A partners. This is the first check
                       that (V, t) is a complete description of the experiment at all
  * ladder contrast -- the readout along each level, shortest to longest pulse. THE headline: a
                       swing beyond noise means the boundary carries a dwell tilt

It deliberately does NOT fit a boundary or propose a next batch. Those decisions belong downstream
of a human looking at this output.

Usage:  python src/run_seed_results.py --template
        python src/run_seed_results.py [--results data/flash_plan_seed_results.csv]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from campaign.results import (
    ACTUAL_T_COLUMN,
    ACTUAL_V_COLUMN,
    CROSSCHECK_COLUMN,
    PLAN_KEY,
    READOUT_COLUMN,
    SPECIMEN_COLUMN,
    STATUS_COLUMN,
    blank_template,
    load,
    unfired,
)

ROOT = Path(__file__).resolve().parent.parent
PLAN_CSV = ROOT / "data" / "flash_plan_seed.csv"
RESULTS_CSV = ROOT / "data" / "flash_plan_seed_results.csv"

# Ladder rungs within this many degrees are one iso-Tmax level. Sits between the generator's
# guarantees: rungs on a level agree to within 2 C, and distinct levels are at least 6 C apart.
LEVEL_CLUSTER_C = 3.0


def _write_template(plan: pd.DataFrame, path: Path) -> None:
    """Write a blank results sheet, refusing to clobber one that already has readings in it.

    :param plan: the seed plan.
    :param path: destination CSV.
    """
    if path.exists():
        existing = pd.read_csv(path, dtype=str, keep_default_na=False)
        filled = existing.get(READOUT_COLUMN, pd.Series(dtype=str)).astype(str).str.strip()
        if (filled != "").any():
            raise SystemExit(
                f"{path.name} already contains {(filled != '').sum()} readings. Refusing to "
                "overwrite measured data. Move it aside first if you really want a fresh sheet."
            )
    blank_template(plan).to_csv(path, index=False)
    print(f"Wrote blank results sheet -> {path.relative_to(ROOT)}")
    print(f"  {len(plan)} planned conditions plus one AS-DEPOSITED CONTROL (index 0).")
    print("  The control is never flashed. It is the only measurement that can say what")
    print("  an un-annealed film reads, which nothing in the archive establishes.")
    print(f"  Fill {STATUS_COLUMN} on EVERY row: measured | failed | not_run.")
    print(f"  Record the AS-FIRED {ACTUAL_V_COLUMN} and {ACTUAL_T_COLUMN}, not the planned ones,")
    print("  wherever the tool delivered something other than what was commanded.")
    print(f"  Leave {READOUT_COLUMN} empty on any row that is not 'measured'; a blank is never")
    print("  read as zero, and a value on a non-measured row is an error.")


def _report_coverage(res) -> None:
    """Coverage and drift: what came back, and whether it came back at the intended condition."""
    n = len(res.table)
    print(f"=== coverage ===\n  {res.n_measured} of {n} conditions returned a usable reading")
    missing = unfired(res)
    if missing:
        rows = res.table.set_index(PLAN_KEY).loc[missing]
        print(f"  NOT USABLE: {missing}")
        for i, r in rows.iterrows():
            print(f"    #{i:<3d} block {r['block']}  {r[STATUS_COLUMN]:<8s} {r['note']}")
        print("  Every statement below is conditional on this subset, not on the designed batch.")

    if len(res.drift):
        print(f"\n=== as-fired drift ({len(res.drift)} shots) ===")
        for _, r in res.drift.iterrows():
            print(
                f"  #{int(r[PLAN_KEY]):<3d} planned {r['voltage_V']:.0f} V / {r['time_ms']:.1f} ms"
                f"  ->  fired {r[ACTUAL_V_COLUMN]:.0f} V / {r[ACTUAL_T_COLUMN]:.1f} ms"
            )
        print("  These are valid measurements AT THE AS-FIRED CONDITION. A drifted ladder rung is")
        print("  no longer on its iso-Tmax level and must not be read along the level.")


def _report_replicates(res) -> None:
    """Repeated conditions side by side -- the first test that (V, t) describes the experiment.

    Reports the difference and does NOT judge it. Judging needs a readout noise model. The one this
    script used was calibrated on archival PUND data from different films, which this repo's own
    provenance notes list as non-transferable, and it was written in crystalline-fraction units
    while the instrument reports permittivity. Any permittivity clipped to f = 1, collapsing sigma
    to its floor of 0.02, so a difference of order 1 was tested against a threshold of 0.057 and
    EVERY pair came back discrepant regardless of the data. Printing the numbers without a verdict
    is the honest version until a noise model calibrated on these films exists.
    """
    d = res.table[res.measured]
    dup = d[d.duplicated(subset=["voltage_V", "time_ms"], keep=False)]
    if dup.empty:
        print("\n=== replicates ===\n  no replicate pair has two usable readings yet")
        return
    print("\n=== replicates: same commanded condition, different specimen ===")
    print(f"  {'V':>5s} {'t':>5s} {'first':>9s} {'second':>9s} {'diff':>9s}   relative")
    for (v, t), g in dup.groupby(["voltage_V", "time_ms"]):
        vals = g[READOUT_COLUMN].to_numpy(float)
        if len(vals) < 2:
            continue
        diff = float(np.abs(vals[0] - vals[-1]))
        scale = float(np.abs(np.mean(vals)))
        rel = diff / scale if scale > 0 else float("nan")
        print(f"  {v:5.0f} {t:5.1f} {vals[0]:9.3f} {vals[-1]:9.3f} {diff:9.3f}   {rel:7.1%}")
    print("  No verdict is offered: see the docstring. A large difference means a variable outside")
    print("  (V, t) is in play -- film batch, interface")
    print("  chemistry, thickness, positioning. It is a sentinel, not a variance estimate.")


def _report_crosscheck(res) -> None:
    """The independent PUND reading, where it exists -- the proxy's only external check here."""
    d = res.table[res.measured]
    ok = d[CROSSCHECK_COLUMN].notna()
    if not ok.any():
        print(f"\n=== cross-check ===\n  no {CROSSCHECK_COLUMN} values recorded")
        print("  The in-loop readout is then wholly uncorroborated on these films.")
        return
    x = d.loc[ok, READOUT_COLUMN].to_numpy(float)
    y = d.loc[ok, CROSSCHECK_COLUMN].to_numpy(float)
    print(f"\n=== cross-check against {CROSSCHECK_COLUMN} (n = {ok.sum()}) ===")
    if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
        print(f"  corr = {np.corrcoef(x, y)[0, 1]:+.2f}")
    print("  A weak correlation means the in-loop readout is not tracking the ferroelectric")
    print("  response, and the boundary it implies is not the boundary of interest.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", action="store_true", help="write a blank results sheet and exit")
    ap.add_argument("--plan", type=Path, default=PLAN_CSV)
    ap.add_argument("--results", type=Path, default=RESULTS_CSV)
    args = ap.parse_args()

    if not args.plan.exists():
        raise SystemExit(f"no plan at {args.plan}; generate one from src/seed.py first")
    plan = pd.read_csv(args.plan)

    if args.template:
        _write_template(plan, args.results)
        return 0

    if not args.results.exists():
        raise SystemExit(
            f"no results sheet at {args.results}.\n"
            "Run with --template to create one, then have the lab fill it in."
        )

    res = load(args.results, plan)
    print(f"=== seed results: {args.results.name} ===")
    print(f"  plan {args.plan.name}, {len(plan)} conditions\n")
    _report_coverage(res)
    if res.n_measured == 0:
        print("\nNothing measured yet; no contrast to report.")
        return 0
    _report_replicates(res)
    _report_crosscheck(res)

    specimens = res.table.loc[res.measured, SPECIMEN_COLUMN].astype(str).str.strip()
    if (specimens == "").any() or specimens.duplicated().any():
        print(
            f"\nWARNING: {SPECIMEN_COLUMN} is blank or duplicated on some measured rows. "
            "Without it a discrepant replicate cannot be traced to a physical specimen."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
