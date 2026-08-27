"""Generate the seed flash-condition plan for the crystallization-boundary campaign.

A Latin hypercube over the two knobs, stratified in log time, plus two replicates. That is a
deliberately plain design, and it is plain because a targeted one was tested against it and lost.

WHY NOT A TARGETED DESIGN. The obvious move is to spend most of the batch near where the
transition is believed to be, and to buy a matched-temperature dwell contrast there. That design
was built, and it is better -- IF the belief is right. Counting distinct conditions that land in
the transition, as a function of where the transition actually turns out to be:

    true onset      targeted      hypercube
       380 C               0              3
       440 C               5              3
       447 C               7              2      <- what the targeted design bets on
       480 C               0              2
       500 C               0              2

Seven against two where the bet lands, zero against two once it is 33 C out. The belief is a
bracket from six archival samples at one flash time, on a switching figure of merit, from a
different sample set, bracketed by a single voltage step. Averaged over the range that belief
honestly spans, both designs put the same 1.2 conditions in the transition -- but the hypercube
does it reliably and the targeted design does it as a coin flip.

The comparison was run over randomly drawn worlds in which the response family, the transition
location and width, the dwell dependence, a non-thermal voltage channel, the noise scale and the
readout's floor, span and saturation are all sampled independently, with only the 30 MEASURED peak
temperatures held fixed (``validation.worlds``). Worst-case misclassified area over the supported
box, at the 90th percentile: hypercube 0.25, targeted 0.58. The targeted design also lost on the
dwell question it was built for -- 0.30 power against 0.36 -- so it was dominated on both.

WHY TWO REPLICATES. Averaged over five hypercube realizations, 0 and 2 replicates are
indistinguishable on coverage (p90 0.250 vs 0.247) while 3 and 4 degrade it (0.292, 0.440). Two
therefore cost nothing measurable and buy two things: a check that (V, t) describes the experiment
at all -- archival replicates on this tool disagree by up to 2.08x -- and the repeated measurement
that separates readout noise from surface structure, which is otherwise confounded at this budget.

WHAT THE BATCH CANNOT DO. No 16-shot design answers the dwell question under honest uncertainty.
The best of everything tested calls a large tilt 47% of the time at a 16% false-positive rate;
this one manages 36% at 22%. That is not a defect of the design, it is the budget. The dwell
question has to accumulate across the campaign, informed by where the boundary actually turns out
to be -- which is what this batch is for.

Every condition is scored against the whole boundary-model ensemble and the CSV records each
member's prediction, so no single model's label is presented as the answer. The scoring is
REPORTING only; nothing in the placement uses it.

Usage:  python src/run_flash_plan.py [--n-core 14] [--seed 7]
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from campaign.plan import (
    N_DRAWN,
    T_SEARCH_HI,
    T_SEARCH_LO,
    check,
    make_plan,
)
from physics.constants import T_TRANSITION_REF_C, T_TRANSITION_SIGMA_C
from physics.kinetics import DEFAULT_MODEL
from physics.thermal_model import FLASH, T_HI, T_LO, V_HI, V_LO
from visualization.base import save_figure

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "predictions" / "flash_plan"

def _write_csv(plan: dict, path: Path) -> None:
    """Write the plan, recording every hypothesis rather than a single predicted label."""
    keys = list(plan["preds"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["index", "block", "voltage_V", "time_ms", "pred_Tmax_C"]
            + [f"pred_X_{k}" for k in keys]
            + ["model_disagreement", "tmax_node_supported", "readout", "note"]
        )
        for i in range(len(plan["V"])):
            w.writerow(
                [
                    i + 1,
                    plan["block"][i],
                    int(plan["V"][i]),
                    plan["t"][i],
                    round(float(plan["tmax"][i]), 1),
                ]
                + [round(float(plan["preds"][k][i]), 3) for k in keys]
                + [
                    round(float(plan["disagree"][i]), 3),
                    bool(plan["t"][i] >= T_SEARCH_LO),
                    "eps_r",
                    plan["note"][i],
                ]
            )


def _figure(plan: dict, path: Path) -> None:
    """Three panels: where the conditions sit, where the hypotheses disagree, and coverage."""
    models = plan["models"]
    vg = np.linspace(V_LO, V_HI, 220)
    tg = np.linspace(T_SEARCH_LO, T_SEARCH_HI, 220)
    vv, tt = np.meshgrid(vg, tg)
    colors = {"L": "#377eb8", "R": "#ff7f00"}
    labels = {"L": "hypercube draw", "R": "replicate"}

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

    a = axes[0]
    cf = a.contourf(vv, tt, FLASH.tmax(vv, tt), levels=18, cmap="inferno")
    fig.colorbar(cf, ax=a).set_label("measured T$_{max}$ (°C)")
    styles = {"isoT": ":", "ramp": "--", "lamp": "-", "diffusion": "-.", "rect": "-."}
    for key, m in models.items():
        a.contour(
            vv,
            tt,
            m.fraction_grid(vg, tg),
            levels=[0.5],
            colors="cyan",
            linewidths=2.4 if key == DEFAULT_MODEL else 1.4,
            linestyles=styles[key],
        )
    for b in "ABED":
        s = [i for i, x in enumerate(plan["block"]) if x == b]
        if s:
            a.scatter(
                plan["V"][s],
                plan["t"][s],
                c=colors[b],
                s=110,
                marker="D" if b == "D" else "o",
                edgecolors="w",
                linewidths=1.5,
                zorder=5,
                label=labels[b],
            )
    a.set_xlabel("flash voltage V (V)")
    a.set_ylabel("flash time t (ms)")
    a.set_title(
        "Seed conditions on the measured T$_{max}$ field\n"
        "cyan = X=0.5 under each hypothesis (solid = diffusion, dotted = iso-T$_{max}$)",
        fontweight="bold",
        fontsize=10,
    )
    a.legend(loc="upper left", fontsize=8, framealpha=0.9)

    a = axes[1]
    dg = np.stack([m.fraction_grid(vg, tg) for m in models.values()])
    cf = a.contourf(vv, tt, dg.max(0) - dg.min(0), levels=18, cmap="viridis")
    fig.colorbar(cf, ax=a).set_label("ensemble disagreement (max − min X)")
    for b in "ABED":
        s = [i for i, x in enumerate(plan["block"]) if x == b]
        if s:
            a.scatter(
                plan["V"][s],
                plan["t"][s],
                c=colors[b],
                s=110,
                marker="D" if b == "D" else "o",
                edgecolors="w",
                linewidths=1.5,
                zorder=5,
            )
    a.set_xlabel("flash voltage V (V)")
    a.set_ylabel("flash time t (ms)")
    a.set_title(
        "Where the hypotheses disagree\n(a shot where they agree settles nothing)",
        fontweight="bold",
        fontsize=10,
    )

    a = axes[2]
    # How much of the reachable temperature range the batch actually covers -- the property that
    # decided this design, since a targeted batch covers a narrow band superbly and the rest not
    # at all.
    tm = plan["tmax"]
    a.hist(tm, bins=12, color="#7fa8d1", edgecolor="k", linewidth=0.5)
    lo = T_TRANSITION_REF_C - T_TRANSITION_SIGMA_C
    hi = T_TRANSITION_REF_C + T_TRANSITION_SIGMA_C
    for x in (lo, hi):
        a.axvline(x, color="#b23", lw=1.4, ls="--")
    a.axvline(T_TRANSITION_REF_C, color="#b23", lw=2, label="proxy transition bracket")
    a.set_xlabel("peak temperature T$_{max}$ (°C)")
    a.set_ylabel("conditions")
    a.set_title(
        "Coverage of the reachable range\n(the batch does not bet on the bracket)",
        fontweight="bold",
        fontsize=10,
    )
    a.legend(fontsize=8.5)

    plt.tight_layout()
    save_figure(fig, str(path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--n-core", type=int, default=N_DRAWN, help="number of drawn conditions before replicates"
    )
    ap.add_argument("--seed", type=int, default=7, help="base RNG seed for the hypercube")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    plan = make_plan(args.n_core, args.seed)
    check(plan)
    keys = list(plan["preds"])

    n = len(plan["V"])
    print(f"Seed flash plan: {n} shots over V=[{V_LO:.0f},{V_HI:.0f}] V, t=[{T_LO},{T_HI}] ms")
    print(f"  restricted to t in [{T_SEARCH_LO}, {T_SEARCH_HI}] ms; the table has no node below")
    tm = plan["tmax"]
    print(f"  covers Tmax {tm.min():.0f}-{tm.max():.0f} C, and bets on none of it\n")
    head = "  #  blk    V   t(ms)   Tmax  " + "".join(f"{k:>11s}" for k in keys) + "   spread"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for i in range(n):
        row = f"{i + 1:3d}   {plan['block'][i]}   {int(plan['V'][i]):4d} {plan['t'][i]:6.1f} "
        row += f"{plan['tmax'][i]:6.1f}  "
        row += "".join(f"{plan['preds'][k][i]:11.3f}" for k in keys)
        row += f"   {plan['disagree'][i]:6.3f}"
        print(row)

    tm = plan["tmax"]
    print("\n  What the ensemble expects, as a check on coverage rather than a prediction:")
    mean_x = np.mean([plan["preds"][k] for k in keys], axis=0)
    print(f"    conditions the ensemble puts in the transition (0.05 < X < 0.95): "
          f"{int(np.sum((mean_x > 0.05) & (mean_x < 0.95)))} of {n}")
    print(f"    conditions it puts at the floor  (X < 0.05): {int(np.sum(mean_x <= 0.05))}")
    print(f"    conditions it puts at saturation (X > 0.95): {int(np.sum(mean_x >= 0.95))}")
    print("    Those counts assume the ensemble is right about WHERE the transition is. It is the")
    print("    bet this design declines to make -- but if the bracket is right, they are the cost.")

    _write_csv(plan, ROOT / "data" / "flash_plan_seed.csv")
    _figure(plan, OUT / "flash_plan.png")
    print(f"\nSaved -> data/flash_plan_seed.csv, {OUT / 'flash_plan.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
