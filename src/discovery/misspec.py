"""Thermal-model misspecification stress test.

The baseline power study generates data and builds charts from the SAME thermal model, so it
only tests internal consistency. Here we generate data from a PERTURBED ("true") thermal
model while the analysis still builds charts from the CANONICAL model the analyst assumes.
We then measure how identification (P(family)) and Ea_eff recovery degrade with the
misspecification strength delta. delta=0 reproduces the matched case exactly.

The perturbation (strength delta) is two physically-motivated calibration errors at once:
  * the Tmax(V,t) surface constants shift   (a -> a(1+0.4 delta), b -> b(1-0.3 delta)), and
  * cooling becomes temperature-dependent   (tau -> tau / (1 + delta*(Tmax-T_room)/400)),
the latter being exactly the constant-tau artifact the kinetics reviewer flagged.
"""

from typing import Dict, List

import numpy as np

from .charts import build_charts
from .compare import compare
from .synthetic import (
    A_FIT,
    B_FIT,
    KB_EV,
    READOUT_SIGMA,
    SCENARIOS,
    T_ROOM,
    TAU_COOL,
    V0,
    _rank,
    sample_design,
)


def trace_misspec(v: float, t: float, delta: float, n: int = 240):
    """Perturbed temperature trace; delta=0 is identical to synthetic._trace."""
    a = A_FIT * (1 + 0.4 * delta)
    b = B_FIT * (1 - 0.3 * delta)
    peak = T_ROOM + (v - V0) * t / (a + b * t**2)
    tau = TAU_COOL / (1.0 + delta * (peak - T_ROOM) / 400.0)
    s = np.linspace(0.0, t + 6.0 * TAU_COOL, n)
    rise = s <= t / 2.0
    T = np.empty_like(s)
    T[rise] = T_ROOM + (peak - T_ROOM) * np.sin(np.pi * s[rise] / t)
    T[~rise] = T_ROOM + (peak - T_ROOM) * np.exp(-(s[~rise] - t / 2.0) / tau)
    return s, T


def _rank_truth(V, t, scenario, delta) -> np.ndarray:
    """Rank-space controlling quantity under the PERTURBED (true) thermal model."""
    ea = scenario.ea_true if scenario.ea_true is not None else 2.5
    tmax = np.empty(len(V))
    dwell = np.empty(len(V))
    tbac = np.empty(len(V))
    for i, (vi, ti) in enumerate(zip(V, t)):
        s, T = trace_misspec(float(vi), float(ti), delta)
        tmax[i] = T.max()
        dwell[i] = np.trapezoid((T > 600.0).astype(float), s)
        tbac[i] = np.trapezoid(np.exp(-ea / (KB_EV * (T + 273.15))), s)
    if scenario.two_mechanism:
        return np.minimum(_rank(tmax), _rank(dwell))
    return _rank(tbac)


def make_dataset_misspec(n, scenario, readout, rng, delta):
    """(V,t,y) generated from the perturbed thermal model; analysis stays canonical."""
    V, t = sample_design(n, rng)
    r = _rank_truth(V, t, scenario, delta)
    p = 1.0 / (1.0 + np.exp(-40.0 * (r - 0.5)))
    sigma = READOUT_SIGMA[readout]
    if sigma is None:
        y = (rng.uniform(size=n) < p).astype(float)
    else:
        y = np.clip(p + rng.normal(0.0, sigma, n), 0.0, 1.0)
    return V, t, y


def run_misspecification(
    scenario_key: str = "A",
    readout: str = "xrd",
    n: int = 200,
    deltas=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7),
    reps: int = 15,
    seed: int = 0,
) -> List[Dict]:
    """For each misspecification strength delta: generate from the perturbed model, analyze
    with CANONICAL charts, report P(identify family) and median Ea_eff error."""
    sc = SCENARIOS[scenario_key]
    rows: List[Dict] = []
    for di, delta in enumerate(deltas):
        fam = np.zeros(reps)
        ea_err = np.full(reps, np.nan)
        for r in range(reps):
            rng = np.random.default_rng(seed + 1000 * di + r)
            V, t, y = make_dataset_misspec(n, sc, readout, rng, delta)
            res = compare(build_charts(V, t), y, readout)  # canonical analysis
            fam[r] = res["tbac_family_won"]
            if sc.ea_true is not None:
                ea_err[r] = abs(res["recovered_ea_refined"] - sc.ea_true)
        rows.append(
            {
                "delta": float(delta),
                "p_family": float(fam.mean()),
                "median_ea_err": float(np.nanmedian(ea_err)),
            }
        )
    return rows
