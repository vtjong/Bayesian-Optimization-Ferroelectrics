"""Thermal-model misspecification stress test.

The baseline power study generates data and builds charts from the SAME thermal model, so it
only tests internal consistency. Here we generate data from a PERTURBED ("true") thermal
model while the analysis still builds charts from the CANONICAL model the analyst assumes.
We then measure how identification (P(family)) and Ea_eff recovery degrade with the
misspecification strength delta. delta=0 reproduces the matched case exactly.

The perturbation (strength delta) applies two physically-motivated calibration errors to the
canonical measured-table model at once, both vanishing at delta=0:
  * the Tmax(V,t) surface is mis-calibrated -- a smooth flash-time-dependent gain tilts the
    interpolated peak temperature (the analyst's table reads high at short t, low at long t); and
  * cooling becomes temperature-dependent   (tau -> tau / (1 + delta*(Tmax-T_room)/400)),
the latter being exactly the constant-tau artifact the kinetics reviewer flagged.
"""

from typing import Dict, List

import numpy as np

from .charts import build_charts
from .compare import compare
from .synthetic import (
    FLASH,
    KB_EV,
    SCENARIOS,
    T_HI,
    T_LO,
    T_ROOM,
    PulseShape,
    _rank,
    sample_design,
    sample_readout,
)


def trace_misspec(v: float, t: float, delta: float, n: int = 240):
    """Temperature trace under a PERTURBED thermal model; delta=0 == synthetic._trace.

    Perturbs the canonical measured-table model two ways, each scaled by delta (see the
    module docstring): a flash-time-dependent gain on the peak temperature, and a
    peak-temperature-dependent shortening of the pulse decay time.

    :param v: flash voltage.
    :param t: flash time (ms).
    :param delta: misspecification strength (0 reproduces the canonical model exactly).
    :param n: number of trace samples.
    """
    peak = float(FLASH.tmax(v, t))
    if delta == 0.0:
        return FLASH.trace(v, t, n)
    t_norm = (float(t) - T_LO) / (T_HI - T_LO)
    peak = T_ROOM + (peak - T_ROOM) * (1.0 + delta * (0.4 - 0.7 * t_norm))
    tau_decay = FLASH.shape.tau_decay / (1.0 + delta * (peak - T_ROOM) / 400.0)
    shape = PulseShape(
        plateau=FLASH.shape.plateau,
        tau_decay=tau_decay,
        t_rise=FLASH.shape.t_rise,
        duration_ms=FLASH.shape.duration_ms,
    )
    tau = np.linspace(0.0, shape.duration_ms, n)
    return tau, T_ROOM + (peak - T_ROOM) * shape(tau)


def _rank_truth(V, t, scenario, delta) -> np.ndarray:
    """Rank-space controlling quantity under the PERTURBED (true) thermal model.

    :param V: flash voltages of the shots.
    :param t: flash times of the shots (ms).
    :param scenario: the planted ground-truth crystallization rule.
    :param delta: thermal-model misspecification strength.
    """
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
    """(V,t,y) generated from the perturbed thermal model; analysis stays canonical.

    :param n: number of shots to simulate.
    :param scenario: the planted ground-truth crystallization rule.
    :param readout: metrology key selecting the readout-noise model.
    :param rng: random generator for the design draw and the readout noise.
    :param delta: thermal-model misspecification strength.
    """
    V, t = sample_design(n, rng)
    r = _rank_truth(V, t, scenario, delta)
    p = 1.0 / (1.0 + np.exp(-40.0 * (r - 0.5)))
    y = sample_readout(p, readout, rng)
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
    with CANONICAL charts, report P(identify family) and median Ea_eff error.

    :param scenario_key: key into SCENARIOS for the planted ground-truth rule.
    :param readout: metrology key selecting the readout-noise model.
    :param n: shots per repetition.
    :param deltas: misspecification strengths to sweep (0 = matched baseline).
    :param reps: repetitions per delta (Monte-Carlo average).
    :param seed: base RNG seed; each (delta, rep) gets a distinct derived seed.
    """
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
