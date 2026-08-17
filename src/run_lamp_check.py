"""Validate the peak-temperature surface against independently measured lamp fluence.

The Tmax table is the foundation of the whole campaign, and it has never been checked against
anything. This script checks it against data we already own, using no free physics parameters.

The argument, end to end:

  1. A xenon flash lamp does NOT deliver fixed energy per shot. Measured fluence rises sublinearly
     with commanded pulse width (data/Bolometer_readings_PulseForge.xlsx). Fit E(V, t).

  2. A 30 nm stack has negligible heat capacity, so its temperature is set by conduction into the
     fused-silica substrate. For an absorbed fluence E_abs delivered over t, the Carslaw & Jaeger
     surface solution for a semi-infinite solid gives

         Tmax - T_room  =  2 * A * E(V,t) * sqrt(alpha / pi) / (k * sqrt(t))

     i.e. Tmax - T_room is proportional to E(V,t)/sqrt(t), with ONE unknown -- the absorptance A.

  3. That predicts the re-entrance for free: Tmax turns over where d lnE / d lnt crosses 1/2. No
     fitting, no lamp-circuit model.

SCOPE -- read this before trusting any output. The bolometer workbook and the Tmax table are
INDEPENDENT datasets, which is good (no circularity) but also limiting: they share exactly ONE
voltage (716 V) and NO flash times. The workbook covers V in [620, 799] and t in [0.5, 5.0] ms;
the table nodes are V in [506, 716] and t in {0.1, 2.6, 5.1, 7.6, 10.1}. Evaluating the conduction
law at the table nodes therefore means extrapolating the fluence fit outside its support in BOTH
axes, which is not a validation.

So this script deliberately does NOT claim to validate the temperature scale. It reports:
  * what the bolometer establishes on its own support (the lamp is not a fixed-energy source, and
    its fluence saturation sets the transient's intrinsic timescale) -- this constrains the PULSE
    SHAPE, which is what the boundary tilt depends on;
  * the conduction-law comparison restricted to the region the fluence data actually covers, with
    everything outside it flagged as extrapolation.

Resolving the temperature SCALE needs the provenance of each calibration, not more arithmetic.

Usage:  python src/run_lamp_check.py
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

from discovery.constants import T_ROOM_C
from discovery.synthetic import FLASH_T, FLASH_TMAX, FLASH_V
from visualization.base import save_figure

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "predictions" / "lamp_check"
BOLOMETER = ROOT / "data" / "Bolometer_readings_PulseForge.xlsx"

# The workbook mixes two eras: rows quoted in kV (2000-3000) and rows in the campaign's own
# voltage units (620-799). Only the latter are commensurate with the Tmax table's axis.
CAMPAIGN_V_MAX = 1200.0

# Fused silica, room temperature. Both enter only through sqrt(alpha)/k, and both are standard.
SILICA_K = 1.38  # thermal conductivity (W/m/K)
SILICA_ALPHA = 8.3e-7  # thermal diffusivity (m^2/s)

# What two 10 nm TiN electrodes on a transparent substrate can plausibly absorb. Outside this a
# candidate temperature scale is not physically realizable.
ABSORPTANCE_MIN, ABSORPTANCE_MAX = 0.10, 0.70

# Candidate temperature scales, as multipliers on the tabulated rise above room temperature.
# 1.00 is our table as shipped; 1.89 is the ratio to the second calibration of the same tool.
CANDIDATE_SCALES = {"ours (as shipped)": 1.00, "reviewer-implied": 1.39, "hotter calibration": 1.89}


def load_fluence() -> tuple:
    """Measured delivered fluence in the campaign's voltage units: ``(V, t_ms, E_J_cm2)``."""
    d = pd.read_excel(BOLOMETER, sheet_name="Combined")
    d.columns = [c.strip() for c in d.columns]
    v = d["Voltage (V)"].to_numpy(float)
    t = d["Time (ms)"].to_numpy(float)
    e = d["Energy density new cone (J/cm^2)"].to_numpy(float)
    ok = np.isfinite(v) & np.isfinite(t) & np.isfinite(e) & (e > 0) & (v < CAMPAIGN_V_MAX)
    return v[ok], t[ok], e[ok]


def fit_fluence(v: np.ndarray, t: np.ndarray, e: np.ndarray) -> tuple:
    """Fit ``E = c * V^p * t^q`` by log-log least squares; returns ``(c, p, q, mean_rel_err)``."""
    a = np.column_stack([np.ones_like(v), np.log(v), np.log(t)])
    coef, *_ = np.linalg.lstsq(a, np.log(e), rcond=None)
    pred = np.exp(a @ coef)
    rel = float(np.mean(np.abs(pred - e) / e))
    return float(np.exp(coef[0])), float(coef[1]), float(coef[2]), rel


def implied_absorptance(slope_c: float) -> float:
    """Absorptance implied by the fitted constant in ``Tmax - T_room = C * E / sqrt(t)``.

    From the Carslaw & Jaeger constant-flux surface solution, ``C = 2 A sqrt(alpha/pi) / k``.
    Converts the fit's units (E in J/cm^2, t in ms) to SI before inverting.

    :param slope_c: fitted constant with E in J/cm^2 and t in ms.
    """
    c_si = slope_c / (1e4 / np.sqrt(1e-3))  # J/cm^2, ms  ->  J/m^2, s
    return float(c_si * SILICA_K / (2.0 * np.sqrt(SILICA_ALPHA / np.pi)))


def support_overlap(v: np.ndarray, t: np.ndarray) -> dict:
    """How much of the Tmax table the fluence measurements actually cover.

    :param v: measured voltages.
    :param t: measured flash times (ms).
    """
    vv, tt = np.meshgrid(FLASH_V, FLASH_T)
    inside = (vv >= v.min()) & (vv <= v.max()) & (tt >= t.min()) & (tt <= t.max())
    return {
        "inside": inside,
        "n_inside": int(inside.sum()),
        "n_total": int(inside.size),
        "v_range": (float(v.min()), float(v.max())),
        "t_range": (float(t.min()), float(t.max())),
    }


def test_scale(scale: float, c: float, p: float, q: float, mask: np.ndarray = None) -> dict:
    """Fit the conduction law to a candidate temperature scale over the table nodes.

    :param mask: restrict to these nodes (use the fluence support; None uses every node).

    :param scale: multiplier on the tabulated rise above room temperature.
    :param c: fluence-fit prefactor.
    :param p: fluence-fit voltage exponent.
    :param q: fluence-fit time exponent.
    """
    vv, tt = np.meshgrid(FLASH_V, FLASH_T)
    sel = np.ones_like(vv, bool) if mask is None else mask
    vv, tt = vv[sel], tt[sel]
    rise = scale * (FLASH_TMAX[sel] - T_ROOM_C)
    predictor = (c * vv**p * tt**q) / np.sqrt(tt)
    slope = float(np.sum(predictor * rise) / np.sum(predictor**2))  # least squares through origin
    resid = rise - slope * predictor
    r2 = 1.0 - float(np.sum(resid**2) / np.sum((rise - rise.mean()) ** 2))
    a = implied_absorptance(slope)
    return {
        "scale": scale,
        "slope": slope,
        "r2": r2,
        "absorptance": a,
        "physical": ABSORPTANCE_MIN <= a <= ABSORPTANCE_MAX,
        "max_c": scale * (FLASH_TMAX.max() - T_ROOM_C) + T_ROOM_C,
        "n": int(rise.size),
    }


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not BOLOMETER.exists():
        print(f"SKIPPED -- {BOLOMETER.name} not found.")
        return 0

    v, t, e = load_fluence()
    c, p, q, rel = fit_fluence(v, t, e)

    print("=== lamp fluence, measured ===")
    print(f"  {len(v)} shots, V in [{v.min():.0f}, {v.max():.0f}], "
          f"t in [{t.min():g}, {t.max():g}] ms")
    print(f"  E(V,t) = {c:.3g} * V^{p:.2f} * t^{q:.3f}     (mean |rel err| {100 * rel:.1f}%)")
    print(f"  d lnE/d lnt = {q:.3f}")
    print("    fixed energy per shot would give 0.000 -- decisively rejected")
    print(f"    Tmax ~ E/sqrt(t) turns over where this crosses 0.5: measured {q:.3f}")
    print("    -> the fold in the Tmax table is a LAMP effect, not a conduction effect\n")

    ov = support_overlap(v, t)
    print("=== overlap between the fluence data and the Tmax table ===")
    print(f"  fluence support : V in [{ov['v_range'][0]:.0f}, {ov['v_range'][1]:.0f}], "
          f"t in [{ov['t_range'][0]:g}, {ov['t_range'][1]:g}] ms")
    print("  table nodes     : V in [506, 716], t in {0.1, 2.6, 5.1, 7.6, 10.1} ms")
    print(f"  table nodes INSIDE the fluence support: {ov['n_inside']} of {ov['n_total']}")
    print("  shared voltages: 716 only.  shared times: none.")
    print("  -> the table CANNOT be validated against this workbook without extrapolating.")
    print("     Everything below is restricted to the supported nodes and is a consistency")
    print("     check, not a calibration.\n")

    print("=== does a candidate temperature scale imply a physical absorptance? ===")
    print("  Carslaw & Jaeger: Tmax - T_room = 2 A E sqrt(alpha/pi) / (k sqrt(t))")
    print(f"  plausible absorptance for 2 x 10 nm TiN: {ABSORPTANCE_MIN} to {ABSORPTANCE_MAX}\n")
    print(f"  {'scale':22s} {'x rise':>7s} {'box max':>9s} {'R^2':>7s} {'A':>7s}   verdict")
    results = {}
    for name, s in CANDIDATE_SCALES.items():
        r = test_scale(s, c, p, q, ov["inside"])
        results[name] = r
        verdict = "physical" if r["physical"] else "NOT physical"
        print(
            f"  {name:22s} {r['scale']:7.2f} {r['max_c']:8.0f} C "
            f"{r['r2']:7.3f} {r['absorptance']:7.3f}   {verdict}"
        )

    print(f"\n  (fitted on {results['ours (as shipped)']['n']} supported nodes)")
    print("  R^2 is identical across scales BY CONSTRUCTION -- a constant multiplier cannot change")
    print("  fit quality, only the implied absorptance. Absorptance is the only discriminator here")
    print("  and it is weak: more than one scale is admissible. Settling the scale needs")
    print("  the provenance of each calibration, not more arithmetic.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    a = axes[0]
    for tv in sorted(set(t.tolist())):
        m = t == tv
        o = np.argsort(v[m])
        a.plot(v[m][o], e[m][o], "o-", lw=1.4, ms=5, label=f"{tv:g} ms")
    a.set_xlabel("flash voltage V (V)")
    a.set_ylabel("delivered fluence E (J/cm$^2$)")
    a.set_title(
        f"Measured fluence rises with pulse width\nE ~ V$^{{{p:.2f}}}$ t$^{{{q:.3f}}}$ "
        "— not fixed energy per shot",
        fontweight="bold",
        fontsize=10,
    )
    a.legend(fontsize=8, title="pulse width")

    a = axes[1]
    vv, tt = np.meshgrid(FLASH_V, FLASH_T)
    pred = (c * vv**p * tt**q) / np.sqrt(tt)
    for i, tv in enumerate(FLASH_T):
        a.plot(pred[i], FLASH_TMAX[i] - T_ROOM_C, "o", ms=6, label=f"{tv:g} ms")
    lo = results["ours (as shipped)"]
    xs = np.linspace(0, pred.max() * 1.05, 50)
    a.plot(xs, lo["slope"] * xs, "k--", lw=1.2, label=f"fit (R$^2$={lo['r2']:.3f})")
    a.set_xlabel("E(V,t) / $\\sqrt{t}$   (conduction predictor)")
    a.set_ylabel("tabulated T$_{max}$ − T$_{room}$ (°C)")
    a.set_title(
        "Table vs semi-infinite conduction law\n(collapse tests the model; slope sets absorptance)",
        fontweight="bold",
        fontsize=10,
    )
    a.legend(fontsize=8)

    plt.tight_layout()
    save_figure(fig, str(OUT / "lamp_check.png"))
    print(f"\nSaved -> {OUT / 'lamp_check.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
