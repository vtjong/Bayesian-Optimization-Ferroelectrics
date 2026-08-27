"""Stress-test the seed against ground truths it was NOT designed from.

THE WORRY THIS ANSWERS. Block B is restricted to where the five-member ensemble predicts a fraction
between 0.03 and 0.97, so the design chooses where to look using the models it exists to test. If
every member is wrong in the same way, the batch can systematically avoid the region that would
show it. Scoring the seed against those five members cannot detect that -- it is the same
circularity one level up.

So this scores designs against ``validation.adversarial``, which includes truths no member can
represent: a response that falls at high temperature, a second crystallization branch at long
dwell, a boundary that curves in log t, a direct voltage channel, a hard step.

METHOD. For each (design, truth, noise draw): simulate the readout at the design's conditions with
the calibrated heteroscedastic noise, fit a MODEL-AGNOSTIC surrogate -- a GP on (V, log t), the
same one the campaign's picker uses, with no knowledge of the ensemble -- and ask how much of the
supported design box it then classifies on the wrong side of the boundary. Using the ensemble to
score would reintroduce exactly the circularity being tested.

    misclassified fraction = area{ (predicted X > 1/2) XOR (true X > 1/2) } / area{ supported box }

Reported as the MEAN over noise draws and, more importantly, the WORST truth -- a seed is only as
good as its most misleading outcome, since there is no second first batch.

WHAT IT CANNOT SETTLE. The truths here are hostile but hand-written; a real surface could be
hostile in a way not enumerated. A design that wins here is robust against these failure modes,
not against all of them.

Usage:  python src/run_seed_stress.py [--draws 24] [--seed 0]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from campaign.plan import T_SEARCH_HI, T_SEARCH_LO, core_block, make_plan
from physics.constants import NOISE_BOUNDARY, NOISE_FLOOR
from physics.kinetics import build_ensemble
from physics.thermal_model import FLASH, T_HI, V_HI, V_LO
from validation.adversarial import build_truths
from visualization.base import save_figure

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "predictions" / "seed_stress"

GRID_V, GRID_T = 90, 70  # resolution of the misclassification integral

# Temperature window for the model-agnostic block. NOT the whole box: unconstrained maximin fills
# the corners (230 C, 286 C, 568 C), which duplicates the floor anchor and buys saturated points.
# This window is the transition bracket widened by three sigma either way -- wide enough to contain
# a transition the ensemble is badly wrong about, narrow enough that every shot could still land on
# a boundary. Deliberately independent of the ensemble's own 3-97% band, which is the whole point.
EXPLORE_LO_C = 350.0
EXPLORE_HI_C = 520.0
CLASS_THRESHOLD = 0.5
GP_RESTARTS = 4  # the surrogate is refitted thousands of times; keep it cheap but not degenerate


def _sigma(f: np.ndarray) -> np.ndarray:
    """Calibrated heteroscedastic readout noise."""
    return NOISE_FLOOR + NOISE_BOUNDARY * f * (1.0 - f)


def supported_grid() -> tuple:
    """Dense (V, t) grid over the region the measured table supports."""
    v = np.linspace(V_LO, V_HI, GRID_V)
    t = np.geomspace(T_SEARCH_LO, T_SEARCH_HI, GRID_T)
    return np.meshgrid(v, t)


def _features(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Normalized (V, log t) -- the coordinates the campaign's surrogate works in."""
    x = (np.asarray(v, float) - V_LO) / (V_HI - V_LO)
    lo, hi = np.log10(T_SEARCH_LO), np.log10(T_HI)
    return np.column_stack([x, (np.log10(np.asarray(t, float)) - lo) / (hi - lo)])


def fit_surrogate(v: np.ndarray, t: np.ndarray, y: np.ndarray):
    """A GP on (V, log t) with no knowledge of the ensemble.

    :param v: as-fired voltages.
    :param t: as-fired flash times (ms).
    :param y: observed readout.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        [0.4, 0.4], (0.05, 10.0), nu=2.5
    ) + WhiteKernel(1e-3, (1e-6, 1e0))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=GP_RESTARTS)
    return gp.fit(_features(v, t), y)


def misclassified(gp, truth, vv: np.ndarray, tt: np.ndarray) -> float:
    """Fraction of the supported box the fitted surrogate puts on the wrong side of the boundary.

    :param gp: fitted surrogate.
    :param truth: ground-truth response.
    :param vv: voltage grid.
    :param tt: flash-time grid.
    """
    pred = gp.predict(_features(vv.ravel(), tt.ravel())) > CLASS_THRESHOLD
    true = truth(vv.ravel(), tt.ravel()) > CLASS_THRESHOLD
    return float(np.mean(pred != true))


def _maximin(n: int, avoid_v: np.ndarray, avoid_t: np.ndarray) -> tuple:
    """Greedy maximin coverage in normalized (Tmax, log t), over the model-agnostic window.

    The window is a temperature range, not the ensemble's predicted-transition band, so this block
    is free to enter territory every member calls dead -- which is its entire purpose. It is not
    the whole box, because unconstrained maximin spends its picks on corners that no hypothesis
    puts a boundary near.

    :param n: how many conditions to place.
    :param avoid_v: voltages already spoken for.
    :param avoid_t: flash times already spoken for.
    """
    vv, tt = supported_grid()
    v, t = vv.ravel(), tt.ravel()
    tm = FLASH.tmax(v, t)
    keep = np.isfinite(tm) & (tm >= EXPLORE_LO_C) & (tm <= EXPLORE_HI_C)
    v, t, tm = v[keep], t[keep], tm[keep]

    # Coverage is judged in the THERMAL coordinates, which is the whole point of the design.
    lo, hi = tm.min(), tm.max()
    ax = (tm - lo) / max(hi - lo, 1e-9)
    ay = (np.log10(t) - np.log10(T_SEARCH_LO)) / (np.log10(T_SEARCH_HI) - np.log10(T_SEARCH_LO))
    cand = np.column_stack([ax, ay])

    chosen_v, chosen_t = list(avoid_v), list(avoid_t)
    picked = []
    for _ in range(n):
        atm = FLASH.tmax(np.asarray(chosen_v, float), np.asarray(chosen_t, float))
        bx = (atm - lo) / max(hi - lo, 1e-9)
        by = (np.log10(np.asarray(chosen_t, float)) - np.log10(T_SEARCH_LO)) / (
            np.log10(T_SEARCH_HI) - np.log10(T_SEARCH_LO)
        )
        taken = np.column_stack([bx, by])
        d = np.min(np.linalg.norm(cand[:, None, :] - taken[None, :, :], axis=2), axis=1)
        j = int(np.argmax(d))
        picked.append(j)
        chosen_v.append(float(np.round(v[j])))
        chosen_t.append(float(np.round(t[j], 1)))
    return np.round(v[picked]).astype(int), np.round(t[picked], 1)


def build_designs(n_core: int, seed: int) -> dict:
    """The committed seed, and hybrids that trade block-B coverage for model-agnostic probes.

    :param n_core: size of the committed core block.
    :param seed: RNG seed for the core hypercube.
    """
    models = build_ensemble()
    plan = make_plan(n_core=n_core, seed=seed)
    v, t, blk = plan["V"], plan["t"], np.array(plan["block"])
    designs = {"committed (A4 B7 D4 E1)": (v, t)}

    keep_ade = np.isin(blk, ["A", "D", "E"])
    v_ade, t_ade = v[keep_ade], t[keep_ade]

    for n_free in (2, 3, 4):
        n_b = n_core - n_free
        vb, tb = core_block(n_b, seed, v_ade, t_ade, models)
        ve, te = _maximin(n_free, np.concatenate([v_ade, vb]), np.concatenate([t_ade, tb]))
        designs[f"hybrid (A4 B{n_b} C{n_free} D4 E1)"] = (
            np.concatenate([v_ade, vb, ve]),
            np.concatenate([t_ade, tb, te]),
        )
    return designs


def score(designs: dict, truths: dict, draws: int, seed: int) -> dict:
    """Mean misclassified fraction for every (design, truth) pair.

    :param designs: candidate designs.
    :param truths: adversarial ground truths.
    :param draws: noise realizations per pair.
    :param seed: RNG seed.
    """
    vv, tt = supported_grid()
    out = {k: {} for k in designs}
    for dname, (v, t) in designs.items():
        for tname, truth in truths.items():
            f = truth(v, t)
            rng = np.random.default_rng(seed)
            vals = []
            for _ in range(draws):
                y = np.clip(f + rng.normal(0.0, _sigma(f)), 0.0, 1.0)
                vals.append(misclassified(fit_surrogate(v, t, y), truth, vv, tt))
            out[dname][tname] = float(np.mean(vals))
    return out


def _report(res: dict, truths: dict, draws: int) -> None:
    """Per-truth table, then the summary that actually decides: the worst truth."""
    names = list(res)
    print(f"Misclassified fraction of the supported box, mean of {draws} noise draws.")
    print("Lower is better. The WORST column is what matters -- there is no second first batch.\n")
    w = max(len(n) for n in truths) + 2
    print(f"{'truth':{w}s} {'fam':>4s} " + "".join(f"{n[:26]:>28s}" for n in names))
    for tname, truth in truths.items():
        tag = "OUT" if truth.outside_family else "in"
        row = f"{tname:{w}s} {tag:>4s} "
        best = min(res[d][tname] for d in names)
        for d in names:
            v = res[d][tname]
            row += f"{v:>27.3f}{'*' if v <= best + 1e-9 else ' '}"
        print(row)

    print()
    for label, subset in (
        ("all truths", list(truths)),
        ("outside the ensemble family", [k for k, v in truths.items() if v.outside_family]),
    ):
        print(f"  {label}:")
        for d in names:
            vals = [res[d][k] for k in subset]
            print(f"    {d:28s} mean {np.mean(vals):.3f}   WORST {np.max(vals):.3f}")
        print()
    print("  * marks the best design for that truth.")


def _figure(res: dict, truths: dict, designs: dict, path: Path) -> None:
    """Where each design puts its shots, and how the worst case compares."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    a = axes[0]
    vg = np.linspace(V_LO, V_HI, 220)
    tg = np.geomspace(T_SEARCH_LO, T_SEARCH_HI, 220)
    vv, tt = np.meshgrid(vg, tg)
    cf = a.contourf(vv, tt, FLASH.tmax(vv, tt), levels=16, cmap="inferno", alpha=0.85)
    fig.colorbar(cf, ax=a).set_label("T$_{max}$ (°C)")
    marks = ["o", "s", "^", "D"]
    for (name, (v, t)), m in zip(designs.items(), marks):
        a.scatter(v, t, s=54, marker=m, edgecolors="k", linewidths=0.6, label=name, alpha=0.9)
    a.set_yscale("log")
    a.set_xlabel("voltage V (V)")
    a.set_ylabel("flash time t (ms)")
    a.set_title("Where each design spends its shots", fontweight="bold", fontsize=10)
    a.legend(fontsize=7.5, loc="upper left")

    a = axes[1]
    names = list(res)
    outside = [k for k, v in truths.items() if v.outside_family]
    x = np.arange(len(names))
    worst_all = [max(res[d][k] for k in truths) for d in names]
    worst_out = [max(res[d][k] for k in outside) for d in names]
    a.bar(x - 0.2, worst_all, 0.4, label="worst over all truths")
    a.bar(x + 0.2, worst_out, 0.4, label="worst outside the family")
    a.set_xticks(x)
    a.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=7.5)
    a.set_ylabel("misclassified fraction of the box")
    a.set_title(
        "Worst-case misunderstanding after batch 1\n(lower is better)",
        fontweight="bold",
        fontsize=10,
    )
    a.legend(fontsize=8)
    plt.tight_layout()
    save_figure(fig, str(path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=24, help="noise realizations per (design, truth)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-core", type=int, default=7)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    truths = build_truths()
    designs = build_designs(args.n_core, seed=7)
    print("=== seed coverage stress test ===")
    print(f"  {len(designs)} designs x {len(truths)} truths x {args.draws} draws\n")
    for name, truth in truths.items():
        print(f"  {'OUT' if truth.outside_family else 'in ':>4s}  {name:20s} {truth.why}")
    print()

    res = score(designs, truths, args.draws, args.seed)
    _report(res, truths, args.draws)
    _figure(res, truths, designs, OUT / "seed_stress.png")
    print(f"Saved -> {OUT / 'seed_stress.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
