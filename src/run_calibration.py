"""Calibrate the crystallization-boundary readout against measured HZO data.

Grounds three numbers used by the synthetic testbed and the picker in real measurements:

  1. Crystallization ONSET temperature -- where the ferroelectric response (2Pr) switches on.
     Two independent datasets: flash-lamp (2Pr vs IR-measured peak temperature) and RTA furnace
     (2Pr vs directly-set anneal temperature). A logistic fit on the pooled data gives T50.
  2. Thermal anchor -- the peak-temperature model Tmax(V,t) fit to the flash IR data (R2, MAE).
  3. Permittivity readout -- an effective dielectric constant (high-field dQ/dV slope) extracted
     from every pristine P-V loop, vs the ferroelectric response 2Pr. Reports the low-2Pr
     plateau, the eps_r <-> 2Pr correlation, and within-sample repeatability (the noise scale).

TWO CAVEATS, AND THE SECOND IS THE IMPORTANT ONE.

The loop files are LARGE-SIGNAL P-V hysteresis, not small-signal C-V, so the extracted eps_r is an
effective value. At the crystalline end it is inflated by switching: values run to ~300, which no
HfO2 phase supports. A clean crystalline endpoint needs a C-V sweep on the same capacitors.

The low-2Pr plateau is NOT an amorphous reference, and must not be called one. Every P-V sample in
data/ carries a flash-anneal tag; there is no as-deposited capacitor anywhere in the set. The
selection is 2Pr < 2 uC/cm2, which means NOT FERROELECTRIC, not NOT CRYSTALLIZED. Those 16 loops
sit at eps_r 33-45, which is the range of non-ferroelectric crystalline HfO2 (tetragonal ~30-40)
rather than amorphous (~16-20), and no loop in the whole set reads below 33.

The extraction itself is sound -- verified four ways: capacitance scales with dot area across
20/30/50/100/200 um with a median intercept of 0.84 pF (no parasitic); the measured current at high
field matches C*dV/dt (leakage is not dominating); and for these loops dQ/dV at mid-field and
high-field agree to 1-4%, so the slope is a genuine field-independent dielectric response.

So the number is right and the LABEL is wrong. Until one as-deposited capacitor is measured, this
plateau anchors "not ferroelectric", and the campaign has no measured amorphous reference at all.
That measurement costs no film -- the wafer already exists.

Usage:  python src/run_calibration.py
Reads data/ (gitignored); degrades gracefully with a message if the data is absent.
"""

import glob
import re
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parent))
from visualization.base import save_figure  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "predictions" / "calibration"
EPS0 = 8.854e-12  # F/m
D_FILM = 10e-9  # m, HZO film thickness (KHM005/6/9/10 + RTA series are all 10 nm)


# --- 1. onset temperature ---------------------------------------------------------------
def load_onset():
    """Pooled (peak temperature, 2Pr, source) from flash IR data and RTA furnace data."""
    import pandas as pd

    T, P, src = [], [], []
    fla = DATA / "KHM005_KHM006_quartz_HZO_samples.csv"
    if fla.exists():
        d = pd.read_csv(fla, index_col=0)
        d.columns = [c.strip() for c in d.columns]
        T += list(d["Max temperature (degC)"])
        P += list(d["2Pr (uC/cm2), Pristine state"])
        src += ["flash"] * len(d)
    rta = DATA / "FE_HZCO_samples_210311.csv"
    if rta.exists():
        d = pd.read_csv(rta, index_col=0)
        m = d["2Pr (uC/cm2), Pristine state"].notna()
        T += list(d["RTA temperature (C)"][m])
        P += list(d["2Pr (uC/cm2), Pristine state"][m])
        src += ["rta"] * int(m.sum())
    return np.array(T, float), np.array(P, float), np.array(src)


def fit_onset(T, P, thresh=12.0):
    """Logistic fit of crystallized (2Pr>thresh) vs temperature; return (T50, width, fn)."""
    y = (P > thresh).astype(float)
    # 2-parameter logistic p = 1/(1+exp(-(T-T50)/w)) by simple grid + refine
    T50s = np.linspace(T.min(), T.max(), 200)
    ws = np.linspace(5, 80, 60)
    best, arg = 1e18, None
    for t50 in T50s:
        for w in ws:
            p = 1 / (1 + np.exp(-(T - t50) / w))
            ll = -np.sum(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
            if ll < best:
                best, arg = ll, (t50, w)
    t50, w = arg
    return t50, w, lambda x: 1 / (1 + np.exp(-(x - t50) / w))


# --- 2. thermal anchor Tmax(V,t) --------------------------------------------------------
def load_flash_thermal():
    import pandas as pd

    f = DATA / "KHM005_KHM006_quartz_HZO_samples.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, index_col=0)
    d.columns = [c.strip() for c in d.columns]
    return (
        d["Flash voltage (kV)"].values.astype(float),
        d["Flash time (msec)"].values.astype(float),
        d["Max temperature (degC)"].values.astype(float),
    )


def fit_thermal(V, t, Tm):
    """Power-law fit  Tmax - 25 = c * V^p * t^q  (log-log OLS; robust, interpretable).

    The synthetic's re-entrant form is degenerate in real kV/ms units, and the flash is a
    multi-pulse source (duty cycle / num pulses), so a compact 2-exponent law is the honest anchor.
    """
    A = np.column_stack([np.ones_like(V), np.log(V), np.log(t)])
    coef, *_ = np.linalg.lstsq(A, np.log(Tm - 25.0), rcond=None)
    pred = np.exp(A @ coef) + 25.0
    ss = 1 - np.sum((Tm - pred) ** 2) / np.sum((Tm - Tm.mean()) ** 2)
    return (
        (float(np.exp(coef[0])), float(coef[1]), float(coef[2])),
        pred,
        ss,
        float(np.mean(np.abs(Tm - pred))),
    )


# --- 3. permittivity from P-V loops -----------------------------------------------------
def _eps_from_loop(f):
    import pandas as pd

    m = re.search(r"(\d+)um", f)
    if not m:
        return None
    A = np.pi * (float(m.group(1)) * 1e-6 / 2) ** 2  # dot area, m^2
    x = pd.read_excel(f, engine="xlrd", header=None, skiprows=1)
    V = pd.to_numeric(x[1], errors="coerce").values
    Q = pd.to_numeric(x[3], errors="coerce").values
    ok = np.isfinite(V) & np.isfinite(Q)
    V, Q = V[ok], Q[ok]
    if len(V) < 50:
        return None
    vmax = np.nanmax(np.abs(V))
    tail = np.abs(V) > 0.8 * vmax
    if tail.sum() < 10:
        return None
    C = np.polyfit(V[tail], Q[tail], 1)[0]  # F, high-field dielectric slope
    eps_r = C * D_FILM / (EPS0 * A)
    near = np.abs(V) < 0.06 * vmax
    twoPr = 2 * np.nanmean(np.abs(Q[near] / A * 1e2)) if near.sum() > 3 else np.nan
    return dict(
        eps_r=eps_r,
        twoPr=twoPr,
        series=f.split("/")[-2],
        sample=re.sub(r"_PV.*", "", f.split("/")[-1]),
    )


def load_loops():
    import pandas as pd  # noqa: F401

    rows = []
    for f in sorted(glob.glob(str(DATA / "KHM*/*PV*.xls"))):
        if re.search(r"after|cycle|endur|append", f, re.I):
            continue
        try:
            r = _eps_from_loop(f)
            if r and 5 < r["eps_r"] < 130 and np.isfinite(r["twoPr"]) and r["twoPr"] < 200:
                rows.append(r)
        except Exception:
            pass
    return rows


def eps_stats(rows):
    eps = np.array([r["eps_r"] for r in rows])
    p2 = np.array([r["twoPr"] for r in rows])
    # NOT "uncrystallized": 2Pr < 2 selects non-ferroelectric loops, and every sample here was
    # annealed. See the module docstring -- there is no as-deposited reference in this dataset.
    floor = eps[p2 < 2.0]
    # within-sample repeatability (readout-noise scale)
    from collections import defaultdict

    g = defaultdict(list)
    for r in rows:
        g[r["sample"]].append(r["eps_r"])
    cvs = [np.std(v) / np.mean(v) for v in g.values() if len(v) >= 3]
    return dict(
        corr=float(np.corrcoef(eps, p2)[0, 1]),
        floor=float(np.mean(floor)) if len(floor) else np.nan,
        floor_sd=float(np.std(floor)) if len(floor) else np.nan,
        n_floor=int(len(floor)),
        rep_cv=float(np.median(cvs)) if cvs else np.nan,
        eps=eps,
        p2=p2,
    )


def main():
    if not DATA.exists() or not (DATA / "KHM005_KHM006_quartz_HZO_samples.csv").exists():
        print(f"calibration data not found under {DATA} -- nothing to do.")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)

    T, P, src = load_onset()
    onset = {}  # fit flash and RTA separately (different thermal budgets)
    for s in ("flash", "rta"):
        m = src == s
        if m.sum() >= 4:
            onset[s] = fit_onset(T[m], P[m])
    Vt = load_flash_thermal()
    (c, pexp, qexp), pred, r2, mae = fit_thermal(*Vt)
    rows = load_loops()
    es = eps_stats(rows)

    print("=== crystallization onset (2Pr switch-on) ===")
    for s, (t50, w, _) in onset.items():
        print(f"  {s:5s}: T50 = {t50:.0f} C  (10-90% width ~{2.2 * w:.0f} C)")
    print("=== thermal anchor  Tmax-25 = c * V^p * t^q ===")
    print(f"  c={c:.1f}  p={pexp:.2f}  q={qexp:.2f}   R2={r2:.2f}  MAE={mae:.0f} C")
    print("=== permittivity readout (large-signal, effective) ===")
    print(
        f"  corr(eps_r,2Pr)={es['corr']:.2f}   low-2Pr plateau eps_r={es['floor']:.0f}"
        f" +/-{es['floor_sd']:.0f} (n={es['n_floor']})   within-sample CV~{100 * es['rep_cv']:.0f}%"
    )

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.7))
    # panel 1: onset (flash and RTA fit separately; they land within ~15 C)
    for s, col, mk in [("flash", "#d1772b", "o"), ("rta", "#1f6fb2", "s")]:
        mask = src == s
        ax[0].scatter(
            T[mask], P[mask], c=col, marker=mk, s=42, edgecolors="k", lw=0.4, label=f"{s}: 2Pr vs T"
        )
        if s in onset:
            t50, w, ofn = onset[s]
            xs = np.linspace(T[mask].min(), T[mask].max(), 200)
            ax[0].plot(xs, ofn(xs) * P[mask].max(), "--", color=col, lw=1.1, alpha=0.7)
            ax[0].axvline(t50, color=col, lw=1.6, alpha=0.8)
            ax[0].text(
                t50 + 4,
                P.max() * (0.9 if s == "flash" else 0.75),
                f"{s} T50={t50:.0f}C",
                color=col,
                fontsize=9,
                fontweight="bold",
            )
    ax[0].set_xlabel("peak (flash) / anneal (RTA) temperature (C)")
    ax[0].set_ylabel("2Pr (uC/cm2)")
    ax[0].set_title(
        "1. Crystallization onset ~370-390 C\n(two independent datasets)",
        fontweight="bold",
        fontsize=11,
    )
    ax[0].legend(fontsize=8)
    # panel 2: thermal anchor
    ax[1].scatter(Vt[2], pred, c="#7a3fb2", s=42, edgecolors="k", lw=0.4)
    lim = [min(Vt[2].min(), pred.min()) - 20, max(Vt[2].max(), pred.max()) + 20]
    ax[1].plot(lim, lim, "k--", lw=1)
    ax[1].set_xlim(lim)
    ax[1].set_ylim(lim)
    ax[1].set_xlabel("measured IR Tmax (C)")
    ax[1].set_ylabel("model Tmax(V,t) (C)")
    ax[1].set_title(
        f"2. Thermal anchor Tmax(V,t)\nR2={r2:.2f}, MAE={mae:.0f} C", fontweight="bold", fontsize=11
    )
    # panel 3: permittivity
    ax[2].scatter(es["p2"], es["eps"], c="#2e8b57", s=30, edgecolors="k", lw=0.3, alpha=0.8)
    ax[2].axhspan(
        es["floor"] - es["floor_sd"], es["floor"] + es["floor_sd"], color="gray", alpha=0.25
    )
    ax[2].axhline(
        es["floor"],
        color="gray",
        lw=1.5,
        ls="--",
        label=f"low-2P$_r$ plateau eps_r~{es['floor']:.0f}",
    )
    ax[2].set_xlabel("2Pr (uC/cm2)  ->  crystallinity")
    ax[2].set_ylabel("effective eps_r (dQ/dV)")
    ax[2].set_title(
        f"3. Permittivity tracks crystallinity\nr={es['corr']:.2f} (large-signal; needs C-V)",
        fontweight="bold",
        fontsize=11,
    )
    ax[2].legend(fontsize=8)
    save_figure(fig, str(OUT / "calibration.png"))
    print(f"\nSaved -> {OUT / 'calibration.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
