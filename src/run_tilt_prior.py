"""Extract an empirical prior on the boundary tilt from the archival flash-anneal shots.

data/KHM005_KHM006_quartz_HZO_samples.csv holds 21 single-pulse flash shots on 10 nm HZO with a
peak-temperature column, flash times spanning 0.5-5 ms, and a 2Pr readout. That is a dwell contrast
at varying peak temperature -- i.e. the very measurement the campaign's iso-Tmax ladder is designed
to make. It is worth fitting before spending new film.

MODEL. The boundary is linear in ln t (this is the closed-form consequence of an Arrhenius budget;
see physics.kinetics), so

    P(crystallized)  =  sigmoid( s * [ Tmax - T0 + theta * ln(t / t_ref) ] )

with three free parameters: the onset T0, the transition sharpness s, and the tilt coefficient
theta. A tilt of zero means the boundary is a pure peak-temperature threshold.

WHAT TRANSFERS TO THE NEW SAMPLES, AND WHAT DOES NOT. These are a different sample set, so:

  * T0 does NOT transfer. Onset is governed by seeding density, interface chemistry and thickness;
    a vacuum break alone is documented to move it by ~200 C. Treat it as uninformative here.

  * theta in KELVIN does NOT transfer either, and for a second reason: it inherits whatever
    temperature scale this file's Max temperature column is on, which is not established.

  * theta / T50, which is DIMENSIONLESS, DOES transfer. Under any constant rescaling of the
    temperature rise, theta and T50 scale together and the ratio is invariant. Since
    theta = kB*T^2/Ea, the ratio is kB*T50/Ea -- a barrier-to-thermal-energy ratio, a property of
    the crystallization mechanism rather than of the batch or the calibration. This is the number
    to carry forward as a prior.

WHY THE GRADED READOUT AND NOT A PASS/FAIL ONE. Thresholding 2Pr at 12 uC/cm2 separates these 21
shots PERFECTLY in peak temperature: every amorphous shot is at or below 373 C and every
crystallized one at or above 403 C, a clean 30 C gap. Perfectly separated classes make the binary
logistic likelihood diverge to an infinitely sharp step, and the tilt term becomes unidentifiable
-- the fit returns exactly zero tilt with zero residual regardless of the data. So the binary fit
is reported only as a degeneracy check; the tilt is estimated from the graded 2Pr.

CAVEATS, stated because they bound the conclusion: the readout is 2Pr, a ferroelectric switch-on
indicator rather than crystallinity; n = 21; and the peak-temperature column's provenance is
unresolved. This is a prior, not a calibration.

Usage:  python src/run_tilt_prior.py [--bootstrap 4000]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from physics.constants import CELSIUS_TO_KELVIN, KB_EV
from visualization.base import save_figure

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "predictions" / "tilt_prior"
ARCHIVE = ROOT / "data" / "KHM005_KHM006_quartz_HZO_samples.csv"

# 2Pr (uC/cm2) above which a film is called crystallized; matches run_calibration.
PR_THRESHOLD = 12.0
T_REF_MS = 5.1  # reference flash time, so theta is the shift per e-fold of dwell


def load_archive() -> tuple:
    """Archival shots as ``(tmax_C, t_ms, crystallized_flag, graded_readout)``.

    The graded readout is 2Pr normalized by its maximum -- a stand-in for crystalline fraction.
    """
    d = pd.read_csv(ARCHIVE, index_col=0)
    d.columns = [c.strip() for c in d.columns]
    tmax = d["Max temperature (degC)"].to_numpy(float)
    t = d["Flash time (msec)"].to_numpy(float)
    pr = d["2Pr (uC/cm2), Pristine state"].to_numpy(float)
    ok = np.isfinite(tmax) & np.isfinite(t) & np.isfinite(pr) & (t > 0)
    tmax, t, pr = tmax[ok], t[ok], pr[ok]
    return tmax, t, (pr > PR_THRESHOLD).astype(float), pr / np.max(pr)


def is_separable(tmax: np.ndarray, y: np.ndarray) -> tuple:
    """Whether the binarized classes are perfectly separated in peak temperature.

    If they are, a binary logistic fit cannot identify the tilt at any sample size.

    :param tmax: peak temperatures (deg C).
    :param y: crystallized flags.
    """
    if y.sum() == 0 or y.sum() == y.size:
        return True, 0.0
    gap = float(tmax[y > 0].min() - tmax[y == 0].max())
    return gap > 0.0, gap


def sse(params: np.ndarray, tmax: np.ndarray, t: np.ndarray, x: np.ndarray) -> float:
    """Sum of squared residuals of the tilted-logistic boundary against the graded readout.

    :param params: ``(amplitude, T0, log_s, theta)``.
    :param tmax: peak temperatures (deg C).
    :param t: flash times (ms).
    :param x: graded readout, normalized to roughly [0, 1].
    """
    amp, t0, log_s, theta = params
    z = np.clip(np.exp(log_s) * (tmax - t0 + theta * np.log(t / T_REF_MS)), -500, 500)
    return float(np.sum((x - amp / (1.0 + np.exp(-z))) ** 2))


def fit(tmax: np.ndarray, t: np.ndarray, x: np.ndarray, free_tilt: bool = True) -> np.ndarray:
    """Least-squares fit to the graded readout; ``free_tilt=False`` pins the tilt to zero.

    :param tmax: peak temperatures (deg C).
    :param t: flash times (ms).
    :param x: graded readout, normalized to roughly [0, 1].
    :param free_tilt: whether the tilt is a free parameter.
    """
    from scipy.optimize import minimize

    best, best_val = None, np.inf
    for t0 in (360.0, 400.0, 460.0):
        for log_s in (-3.0, -2.0, -1.0):
            x0 = np.array([1.0, t0, log_s, 0.0])
            if free_tilt:
                r = minimize(sse, x0, args=(tmax, t, x), method="Nelder-Mead",
                             options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-10})
                cand, val = np.asarray(r.x), r.fun
            else:
                def obj(p):
                    return sse(np.array([p[0], p[1], p[2], 0.0]), tmax, t, x)

                rr = minimize(obj, x0[:3], method="Nelder-Mead",
                              options={"maxiter": 20000})
                cand, val = np.array([rr.x[0], rr.x[1], rr.x[2], 0.0]), rr.fun
            if val < best_val:
                best, best_val = cand, val
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        print(f"SKIPPED -- {ARCHIVE.name} not found.")
        return 0

    tmax, t, y, x = load_archive()
    print(f"=== archival flash shots: {ARCHIVE.name} ===")
    print(f"  n = {len(y)}  ({int(y.sum())} crystallized by 2Pr > {PR_THRESHOLD:.0f} uC/cm2)")
    print(f"  Tmax {tmax.min():.0f}-{tmax.max():.0f} C   t {t.min():g}-{t.max():g} ms")

    sep, gap = is_separable(tmax, y)
    print(f"\n  binarized classes perfectly separated in Tmax? {sep}  (gap {gap:.0f} C)")
    if sep:
        print("  -> a PASS/FAIL fit cannot identify the tilt here at any sample size.")
        print("     Estimating from the graded 2Pr instead.\n")

    tilted = fit(tmax, t, x, free_tilt=True)
    flat = fit(tmax, t, x, free_tilt=False)
    ll_t, ll_f = -sse(tilted, tmax, t, x), -sse(flat, tmax, t, x)

    rng = np.random.default_rng(args.seed)
    boot = []
    for _ in range(args.bootstrap):
        idx = rng.integers(0, len(y), len(y))
        if len(set(t[idx].tolist())) < 3:  # need dwell contrast to identify a tilt
            continue
        boot.append(fit(tmax[idx], t[idx], x[idx], free_tilt=True)[3])
    boot = np.asarray(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    tilted = np.array([tilted[1], tilted[2], tilted[3]])  # drop amplitude for reporting
    t50 = tilted[0] + CELSIUS_TO_KELVIN
    frac = tilted[2] / t50
    print("=== fit: P(cryst) = sigmoid( s * [Tmax - T0 + theta*ln(t/t_ref)] ) ===")
    print(f"  T0    = {tilted[0]:7.1f} C      <- does NOT transfer to new samples")
    width = 2 * np.log(9) / np.exp(tilted[1])
    print(f"  s     = {np.exp(tilted[1]):7.4f} /C    (10-90% width {width:.0f} C)")
    print(f"  theta = {tilted[2]:7.1f} K      <- scale-dependent; does NOT transfer as-is")
    print(f"  tilt over 2.6-10.1 ms = {tilted[2] * np.log(10.1 / 2.6):.1f} C on THIS file's scale")
    print(f"\n  bootstrap 95% CI on theta: [{lo:.1f}, {hi:.1f}] K "
          f"({100 * np.mean(boot > 0):.0f}% of resamples > 0)")
    print(f"  residual SSE: tilted {-ll_t:.4f} vs flat {-ll_f:.4f}  "
          f"(variance explained by the tilt term: {100 * (1 - (-ll_t) / max(-ll_f, 1e-12)):.1f}%)")

    print("\n=== the transferable quantity ===")
    print(f"  theta / T50 = {frac:.5f}   (dimensionless, invariant to any rescaling of the")
    print("                             temperature rise -- this is what carries forward)")
    print(f"  implied Ea = kB*T50/(theta/T50) = {KB_EV * t50 / frac:.2f} eV")
    print("\n  Use as a PRIOR on the tilt, not a calibration. Different sample set; 2Pr readout;")
    print("  n = 21; and this file's temperature scale is itself unestablished.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    a = axes[0]
    for tv in sorted(set(t.tolist())):
        m = t == tv
        a.scatter(tmax[m], y[m] + rng.normal(0, 0.015, m.sum()), s=60, label=f"{tv:g} ms")
    grid = np.linspace(tmax.min() - 20, tmax.max() + 20, 400)
    for tv in (0.5, 5.0):
        z = np.exp(tilted[1]) * (grid - tilted[0] + tilted[2] * np.log(tv / T_REF_MS))
        a.plot(grid, 1 / (1 + np.exp(-z)), lw=2, label=f"fit, {tv:g} ms")
    a.set_xlabel("peak temperature (°C, this file's scale)")
    a.set_ylabel("crystallized  (2P$_r$ > 12)")
    a.set_title(
        "Archival flash shots: does the boundary move with dwell?", fontweight="bold", fontsize=10
    )
    a.legend(fontsize=8, ncol=2)

    a = axes[1]
    a.hist(boot, bins=60, color="#377eb8", alpha=0.8)
    a.axvline(0.0, color="k", lw=1.5, ls="--", label="no tilt")
    a.axvline(tilted[2], color="#d1772b", lw=2, label=f"fit = {tilted[2]:.1f} K")
    a.set_xlabel(r"tilt coefficient $\theta$ (K per e-fold of dwell)")
    a.set_ylabel("bootstrap count")
    a.set_title(
        f"Bootstrap on $\\theta$ (n={len(boot)})\n95% CI [{lo:.0f}, {hi:.0f}] K",
        fontweight="bold",
        fontsize=10,
    )
    a.legend(fontsize=9)

    plt.tight_layout()
    save_figure(fig, str(OUT / "tilt_prior.png"))
    print(f"\nSaved -> {OUT / 'tilt_prior.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
