"""Run the chart-comparison power study across readout types and save figures + results.

Headline question: on synthetic FLA data (the boss's Tmax(V,t) model), how many shots does
each readout need to identify the controlling quantity? Compares binary pass/fail vs
continuous crystalline-fraction readouts (XRD low-noise, Raman moderate, optical proxy
higher-noise). Also runs scenarios B (off-grid Ea) and C (two-mechanism) to confirm the
method reports the harder cases honestly.

Usage:  python src/run_chart_power_study.py [--reps N]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent))

from discovery.power import min_n_for_power, run_power
from visualization.base import save_figure

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "predictions" / "chart_power_study"

READOUT_LABEL = {
    "binary": "binary pass/fail",
    "optical": "optical proxy (σ≈0.12)",
    "raman": "Raman (σ≈0.07)",
    "xrd": "XRD fraction (σ≈0.03)",
}
READOUT_COLOR = {"binary": "#7a7a7a", "optical": "#d1772b",
                 "raman": "#2e8b57", "xrd": "#1f6fb2"}


def _curve(rows, out, metric, ylabel, title, fname, target=0.8):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    readouts = []
    for r in rows:
        if r["readout"] not in readouts:
            readouts.append(r["readout"])
    for ro in readouts:
        sub = sorted((r for r in rows if r["readout"] == ro), key=lambda r: r["n"])
        ys = [r[metric] for r in sub]
        if any(y is None for y in ys):
            continue
        ax.plot([r["n"] for r in sub], ys, "o-", color=READOUT_COLOR.get(ro),
                label=READOUT_LABEL.get(ro, ro))
    if target is not None:
        ax.axhline(target, color="green", ls="--", lw=1, label=f"target {target}")
        ax.set_ylim(0, 1.02)
    ax.set_xlabel("number of shots  n", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    save_figure(fig, str(out / fname))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--n", type=int, nargs="+", default=[40, 80, 160, 300])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("=== Scenario A (single controlling quantity, Ea=2.5 on grid) ===")
    rows = run_power("A", n_list=tuple(args.n), reps=args.reps)
    _curve(rows, OUT, "p_family",
           "P(identify the controlling quantity)",
           "How many shots to identify the mechanism? (by readout)",
           "power_identify.png")
    _curve(rows, OUT, "p_ea",
           "P(recover Ea_eff within ±0.5 eV)",
           "How many shots to recover the (effective) activation energy? (by readout)",
           "power_ea.png")

    gono_fam = min_n_for_power(rows, "p_family", 0.8)
    gono_ea = min_n_for_power(rows, "p_ea", 0.8)
    print("\nGO/NO-GO — min shots for P>=0.8:")
    for ro in READOUT_LABEL:
        print(f"  {READOUT_LABEL[ro]:24s}  identify family: {gono_fam.get(ro)}"
              f"   recover Ea_eff: {gono_ea.get(ro)}")

    # three-way diagnostic at n=300, XRD: a real order parameter beats the (V,t) control
    # chart (margin>0); a two-mechanism boundary does not (margin~0). Ea_eff error then
    # separates on-grid (A, low) from off-grid (B, higher).
    print("\n=== Scenario diagnostic (n=300, XRD): margin over (V,t) + Ea_eff error ===")
    diag = {}
    for sk in ("A", "B", "C"):
        r = run_power(sk, readouts=("xrd",), n_list=(300,), reps=args.reps, verbose=False)[0]
        diag[sk] = r
        eae = "NA" if r["p_ea"] is None else f"{r['median_ea_err']:.2f}eV"
        print(f"  Scenario {sk}: margin/(V,t)={r['median_margin_over_vt']:6.1f}   "
              f"med|Ea_eff err|={eae}")

    results = {"scenario_A": rows, "gono_family": gono_fam, "gono_ea": gono_ea,
               "diagnostic_BC": diag, "reps": args.reps, "n_list": args.n}
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved figures + results.json -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
