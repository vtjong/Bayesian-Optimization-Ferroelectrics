"""Power analysis for the seed design: can block A actually identify the boundary tilt?

The seed plan spends six shots on an iso-Tmax ladder -- two peak-temperature levels crossed with
three pulse widths -- because that contrast is the only thing in the design that separates "the
boundary is a pure temperature threshold" from "the boundary carries a kinetic tilt". This script
asks whether those shots, plus any replicate landing on a rung, survive the readout noise.

Method: simulate the ladder readings under each hypothesis in turn with the calibrated
heteroscedastic noise, then select a hypothesis by maximum likelihood under that same noise model,
and tabulate how often the selection is right.

SCOPE. This is closed-set model selection, not a power calculation: it assumes the truth is one of
the enumerated members, and it holds the onset fixed at its prior centre. The reported rates are
therefore conditional on both. The diffusion and rectangular members differ by ~0.2 C of tilt and
are degenerate by construction, so they are not separable from each other; the meaningful output is
the coarse verdict, zero tilt vs any tilt.

Usage:  python src/run_seed_power.py [--trials 4000] [--seed 0]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from campaign.plan import make_plan
from physics.kinetics import build_ensemble
from validation.picker import DEFAULT as NOISE_CFG
from validation.picker import noise_sigma
from visualization.base import save_figure

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "predictions" / "seed_power"

# Hypotheses that imply the same experiment-facing conclusion ("the dwell matters"), grouped so the
# headline verdict is not penalized for failing to split two models that are 1 C apart.
COARSE = {"isoT": "no tilt", "ramp": "tilt", "lamp": "tilt",
          "diffusion": "tilt", "rect": "tilt"}


def ladder_conditions(plan: dict) -> tuple:
    """Voltages and times of every shot that lands on the iso-Tmax ladder (blocks A and D).

    :param plan: seed plan from ``run_flash_plan.make_plan``.
    """
    keep = [i for i, b in enumerate(plan["block"]) if b in ("A", "D")]
    return plan["V"][keep], plan["t"][keep]


def simulate(x_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One noisy readout of a set of conditions, using the calibrated permittivity noise model.

    :param x_true: true crystalline fractions.
    :param rng: random generator.
    """
    return np.clip(x_true + rng.normal(0.0, noise_sigma(x_true, NOISE_CFG)), 0.0, 1.0)


def log_likelihood(obs: np.ndarray, pred: np.ndarray) -> float:
    """Gaussian log-likelihood of ``obs`` under predicted fractions ``pred``.

    :param obs: measured fractions.
    :param pred: predicted fractions under a hypothesis.
    """
    sd = noise_sigma(pred, NOISE_CFG)
    return float(np.sum(-0.5 * ((obs - pred) / sd) ** 2 - np.log(sd)))


def run(trials: int, seed: int, plan: dict = None) -> dict:
    """Simulate the ladder under every hypothesis and tabulate the selection frequencies.

    :param trials: noise realizations per hypothesis.
    :param seed: RNG seed.
    :param plan: seed plan to score; generated with the shipped defaults when omitted.
    """
    plan = make_plan(n_core=5, seed=7) if plan is None else plan
    v, t = ladder_conditions(plan)
    models = build_ensemble()
    keys = list(models)
    truth = {k: models[k].fraction(v, t) for k in keys}

    rng = np.random.default_rng(seed)
    confusion = np.zeros((len(keys), len(keys)), int)
    slopes = {k: np.empty(trials) for k in keys}
    log_t = np.log(t)
    for i, true_key in enumerate(keys):
        for r in range(trials):
            obs = simulate(truth[true_key], rng)
            lls = [log_likelihood(obs, truth[k]) for k in keys]
            confusion[i, int(np.argmax(lls))] += 1
            slopes[true_key][r] = np.polyfit(log_t, obs, 1)[0]
    return {
        "keys": keys,
        "confusion": confusion,
        "slopes": slopes,
        "conditions": (v, t),
        "truth": truth,
        "trials": trials,
    }


def _report(res: dict) -> None:
    """Print the confusion matrix and the coarse zero-tilt-vs-tilt verdict."""
    keys, conf, n = res["keys"], res["confusion"], res["trials"]
    v, t = res["conditions"]
    times = ", ".join(f"{x:g}" for x in sorted(set(t.tolist())))
    print(f"Ladder shots: {len(v)} at Tmax ~ constant, t = [{times}] ms")
    print("(iso-Tmax ladder plus the replicates that land on it)\n")
    print("True fraction at each ladder condition:")
    print(f"  {'t (ms)':>8s} " + "".join(f"{k:>11s}" for k in keys))
    order = np.argsort(t)
    for j in order:
        print(f"  {t[j]:8.1f} " + "".join(f"{res['truth'][k][j]:11.3f}" for k in keys))

    print(f"\nSelection frequency over {n} noise realizations (rows = truth, cols = selected):")
    print(f"  {'truth':>11s} " + "".join(f"{k:>11s}" for k in keys) + f"{'correct':>10s}")
    for i, k in enumerate(keys):
        row = "".join(f"{100 * conf[i, j] / n:10.1f}%" for j in range(len(keys)))
        print(f"  {k:>11s} {row}{100 * conf[i, i] / n:9.1f}%")

    print("\nCoarse verdict -- the question the campaign actually needs answered:")
    coarse_of = [COARSE[k] for k in keys]
    for i, k in enumerate(keys):
        want = COARSE[k]
        hit = sum(conf[i, j] for j in range(len(keys)) if coarse_of[j] == want)
        print(f"  true = {k:10s} ({want:7s}) -> correct class {100 * hit / n:6.1f}%")

    tilted = [i for i, k in enumerate(keys) if COARSE[k] == "tilt"]
    detect = np.mean([sum(conf[i, j] for j in tilted) / n for i in tilted])
    iso_i = keys.index("isoT")
    spec = conf[iso_i, iso_i] / n
    print(
        f"\n  power to DETECT a tilt when one exists : {100 * detect:.1f}%"
        f"\n  power to CONFIRM no tilt when there is none: {100 * spec:.1f}%"
    )


def _figure(res: dict, path: Path) -> None:
    """Ladder-slope distributions and the confusion matrix."""
    keys, conf, n = res["keys"], res["confusion"], res["trials"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    a = axes[0]
    for k in keys:
        a.hist(res["slopes"][k], bins=60, alpha=0.55, label=f"true = {k}")
    a.axvline(0.0, color="k", lw=1, ls="--")
    a.set_xlabel("fitted ladder slope  dX / d(ln t)   [same T$_{max}$ at every rung]")
    a.set_ylabel("count")
    a.set_title(
        "Ladder slope under each hypothesis\n(zero = boundary is a pure temperature threshold)",
        fontweight="bold",
        fontsize=10,
    )
    a.legend(fontsize=8.5)

    a = axes[1]
    m = 100.0 * conf / n
    im = a.imshow(m, cmap="viridis", vmin=0, vmax=100)
    fig.colorbar(im, ax=a).set_label("selection frequency (%)")
    a.set_xticks(range(len(keys)), keys, rotation=30)
    a.set_yticks(range(len(keys)), keys)
    a.set_xlabel("selected hypothesis")
    a.set_ylabel("true hypothesis")
    for i in range(len(keys)):
        for j in range(len(keys)):
            a.text(
                j,
                i,
                f"{m[i, j]:.0f}",
                ha="center",
                va="center",
                color="w" if m[i, j] < 55 else "k",
                fontsize=9,
                fontweight="bold",
            )
    a.set_title(
        "Maximum-likelihood selection from the ladder alone\n"
        "(diffusion/rect are ~1 °C apart — not separable by design)",
        fontweight="bold",
        fontsize=10,
    )

    plt.tight_layout()
    save_figure(fig, str(path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=4000, help="noise realizations per hypothesis")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    res = run(args.trials, args.seed)
    _report(res)
    _figure(res, OUT / "seed_power.png")
    print(f"\nSaved -> {OUT / 'seed_power.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
