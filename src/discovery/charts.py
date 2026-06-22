"""Candidate coordinate charts on the (V, t) manifold (boss's framework §4-5).

Each chart re-expresses the SAME shots in a different 2-D coordinate system. The chart
comparison (compare.py) then asks which coordinates make the crystallization boundary
simplest. Charts:

    (V, t)            -- raw control knobs (baseline; expected to lose)
    (Tmax, t)         -- peak-temperature control          \\
    (TB, t)           -- unweighted thermal budget          } the "thermal cluster"
    (log TBac(Ea), t) -- activated budget, Ea-indexed family /  (mutually ~degenerate)
    (dwell, t)        -- time above T* (growth-time axis)   \\
    (heat_rate, t)    -- peak heating rate dT/dt             } axes SEPARABLE from thermal
    (fluence, t)      -- V^2*t, pure (V,t) / photonic proxy /   (see mechanisms rework doc)

The last three are decorrelated from the thermal cluster (rank-corr 0.05-0.43), so they let
the comparison discover dwell-/heating-rate-/fluence-controlled mechanisms instead of only
thermal ones. (Per docs/crystallization_mechanisms_reworked.md degeneracy analysis.)

Two lessons from the boss's tutorial are implemented here:
  * use log(TBac), not TBac (TBac ~ exp(-30) underflows / spans many orders of magnitude);
  * standardize the TBac family with POOLED statistics (independent per-chart standardizing
    erases the Ea-dependent shifts the comparison needs).
"""

from typing import Dict, Tuple

import numpy as np

from .synthetic import KB_EV, T_ROOM, _trace, tmax

# Ea grid for the TBac chart family (deg: 2.25 is deliberately ABSENT so scenario B is
# "between grid points" and 2.5 is PRESENT so scenario A is resolvable).
EA_GRID = (1.5, 2.0, 2.5, 3.0, 3.5)


def _raw_descriptors(V: np.ndarray, t: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute Tmax, TB and log TBac(Ea) for every shot (one trace pass per shot)."""
    n = len(V)
    Tmax = np.empty(n)
    TB = np.empty(n)
    dwell = np.empty(n)
    heat_rate = np.empty(n)
    logTBac = {ea: np.empty(n) for ea in EA_GRID}
    for i in range(n):
        s, T = _trace(float(V[i]), float(t[i]))
        T_K = T + 273.15
        Tmax[i] = T.max()
        TB[i] = np.trapezoid(np.clip(T - T_ROOM, 0, None), s)
        dwell[i] = np.trapezoid((T > 600.0).astype(float), s)   # time above 600 C
        heat_rate[i] = np.gradient(T, s).max()                  # peak dT/dt (heating)
        for ea in EA_GRID:
            val = np.trapezoid(np.exp(-ea / (KB_EV * T_K)), s)
            logTBac[ea][i] = np.log(max(val, 1e-300))
    V = np.asarray(V, float)
    return {"Tmax": Tmax, "TB": TB, "dwell": dwell, "heat_rate": heat_rate,
            "fluence": V ** 2 * np.asarray(t, float),           # pure (V,t), no trace
            "t": np.asarray(t, float), "V": V,
            **{f"logTBac_{ea}": logTBac[ea] for ea in EA_GRID}}


def _std(col: np.ndarray, mean: float = None, std: float = None) -> np.ndarray:
    mean = col.mean() if mean is None else mean
    std = col.std() if std is None else std
    return (col - mean) / (std + 1e-12)


def build_charts(V: np.ndarray, t: np.ndarray) -> Dict[str, np.ndarray]:
    """Return {chart_name: standardized (n, 2) coordinate array} for all candidate charts."""
    d = _raw_descriptors(V, t)
    ts = _std(d["t"])
    charts: Dict[str, np.ndarray] = {
        "(V,t)": np.column_stack([_std(d["V"]), ts]),
        "(Tmax,t)": np.column_stack([_std(d["Tmax"]), ts]),
        "(TB,t)": np.column_stack([_std(d["TB"]), ts]),
        "(dwell,t)": np.column_stack([_std(d["dwell"]), ts]),
        "(heat_rate,t)": np.column_stack([_std(d["heat_rate"]), ts]),
        "(fluence,t)": np.column_stack([_std(d["fluence"]), ts]),
    }
    # pooled standardization across the whole TBac family (shared mean/std)
    pool = np.concatenate([d[f"logTBac_{ea}"] for ea in EA_GRID])
    pmean, pstd = pool.mean(), pool.std()
    for ea in EA_GRID:
        charts[f"(TBac{ea},t)"] = np.column_stack(
            [_std(d[f"logTBac_{ea}"], pmean, pstd), ts])
    return charts


def tbac_family_names() -> Tuple[str, ...]:
    """Names of the TBac charts, in Ea order (for parsing the recovered Ea)."""
    return tuple(f"(TBac{ea},t)" for ea in EA_GRID)


def ea_of_chart(name: str):
    """Recover the Ea value from a TBac chart name, or None."""
    if name.startswith("(TBac"):
        return float(name[len("(TBac"):name.index(",")])
    return None
