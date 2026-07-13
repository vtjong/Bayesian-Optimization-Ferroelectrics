"""Design-stage power study over sample size, readout type, and scenario.

For a planted scenario, repeatedly simulate n shots under a given readout, run the chart
comparison, and score whether it (a) identifies the controlling-quantity FAMILY and
(b) recovers the activation energy. Reports P(success) vs n for each readout, so we can
read off how many experiments each metrology choice needs -- the headline deliverable.
"""

from typing import Dict, List, Tuple

import numpy as np

from .charts import build_charts
from .compare import compare
from .synthetic import SCENARIOS, make_dataset


def run_power(
    scenario_key: str = "A",
    readouts: Tuple[str, ...] = ("binary", "optical", "raman", "xrd"),
    n_list: Tuple[int, ...] = (40, 80, 160, 300),
    reps: int = 20,
    ea_tol: float = 0.5,
    seed: int = 0,
    verbose: bool = True,
) -> List[Dict]:
    """Return rows of {readout, n, p_family, p_ea, median_top_weight, median_ea_err}."""
    scenario = SCENARIOS[scenario_key]
    rows: List[Dict] = []
    for ri, readout in enumerate(readouts):
        for n in n_list:
            fam = np.zeros(reps)
            ea_hit = np.zeros(reps)
            ea_err = np.full(reps, np.nan)
            margin = np.empty(reps)
            for r in range(reps):
                # deterministic, distinct seed per (readout, n, rep)
                rng = np.random.default_rng(seed + 100000 * ri + 1000 * n + r)
                V, t, y = make_dataset(n, scenario, readout, rng)
                res = compare(build_charts(V, t), y, readout)
                fam[r] = res["tbac_family_won"]
                margin[r] = res["margin_over_vt"]
                if scenario.ea_true is not None:
                    # CONTINUOUS (sub-grid) recovery error vs the planted truth
                    err = abs(res["recovered_ea_refined"] - scenario.ea_true)
                    ea_err[r] = err
                    ea_hit[r] = err <= ea_tol
            row = {
                "readout": readout,
                "n": n,
                "p_family": float(fam.mean()),
                "p_ea": float(ea_hit.mean()) if scenario.ea_true else None,
                "median_ea_err": float(np.nanmedian(ea_err)),
                "median_margin_over_vt": float(np.median(margin)),
            }
            rows.append(row)
            if verbose:
                eae = "NA" if scenario.ea_true is None else f"{row['median_ea_err']:.2f}eV"
                print(
                    f"  {readout:8s} n={n:4d}  P(family)={row['p_family']:.2f}  "
                    f"med|Ea_err|={eae}  margin/(V,t)={row['median_margin_over_vt']:.1f}"
                )
    return rows


def min_n_for_power(
    rows: List[Dict], key: str = "p_family", target: float = 0.8
) -> Dict[str, object]:
    """Smallest n reaching target for the given metric, per readout (None if never)."""
    out: Dict[str, object] = {}
    readouts = []
    for r in rows:
        if r["readout"] not in readouts:
            readouts.append(r["readout"])
    for ro in readouts:
        ns = sorted(
            r["n"] for r in rows if r["readout"] == ro and r[key] is not None and r[key] >= target
        )
        out[ro] = ns[0] if ns else None
    return out
