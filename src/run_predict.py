"""What do we predict for a set of conditions, and how sure are we entitled to be?

Written to be run BEFORE the measurements come back, so the campaign commits to a prediction it can
then be judged against, and again AFTER, to compare.

WHY A POINT PREDICTION WOULD BE DISHONEST. The seed plan carries one number per condition per
ensemble member, and the spread between members looks like an uncertainty. It is not: it is model
disagreement only, and it collapses to zero wherever the members happen to agree -- which is most
of the box -- while the things we are genuinely unsure about are elsewhere. Four sources matter and
only the first is in that spread:

  MODEL      which cooling law the film obeys. Five candidates, none measured.
  LOCATION   where the transition sits. A bracket from six archival samples, 434.7-458.7 C, and the
             campaign-tool evidence for it is a switching proxy rather than crystallinity.
  THERMAL    interpolation between the 30 measured peak temperatures. Under a tenth of a degree at
             a measured row, about 16 C midway between rows.
  READOUT    the measurement itself, heteroscedastic and largest mid-transition.

This samples all four and reports an interval. Where that interval spans most of [0, 1] the honest
statement is that we cannot predict the condition at all, and it should be reported that way rather
than as a number with a spread beside it.

WHAT CANNOT BE PREDICTED AT ALL. The instrument reports permittivity in its own units with an
uncalibrated zero and gain, and the models predict a crystalline FRACTION. Those are not the same
quantity and no amount of propagation makes them comparable. So the comparison is on ORDER and on
SIDE -- does the ranking agree, does each condition fall where predicted relative to the
transition -- never on absolute agreement of values.

Usage:  python src/run_predict.py [--plan data/flash_plan_seed.csv] [--draws 400]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from discovery.constants import (
    AVRAMI_N,
    EA_EV,
    NOISE_BOUNDARY,
    NOISE_FLOOR,
    T_TRANSITION_HI_C,
    T_TRANSITION_LO_C,
    T_TRANSITION_SIGMA_C,
)
from discovery.kinetics import build_ensemble
from discovery.synthetic import SHAPES

ROOT = Path(__file__).resolve().parent.parent
WIDE_INTERVAL = 0.6  # a predictive interval wider than this means "we cannot call this condition"


def predictive_draws(v: np.ndarray, t: np.ndarray, n_draws: int, seed: int) -> np.ndarray:
    """Predicted crystalline fraction, sampling every source of uncertainty we can name.

    Returns an array of shape ``(n_draws, len(v))``.

    :param v: flash voltages.
    :param t: flash times (ms).
    :param n_draws: how many draws.
    :param seed: RNG seed.
    """
    rng = np.random.default_rng(seed)
    shapes = list(SHAPES)
    out = np.empty((n_draws, len(v)))
    # The thermal-interpolation term is imported lazily: it fits a GP on first use.
    from discovery.thermal_gp import build

    gp_t = build()
    _, sd_t = gp_t.predict(v, t)
    for i in range(n_draws):
        shape = shapes[rng.integers(len(shapes))]  # MODEL
        t0 = rng.uniform(T_TRANSITION_LO_C, T_TRANSITION_HI_C)  # LOCATION
        t0 += rng.normal(0.0, T_TRANSITION_SIGMA_C * 0.5)
        models = build_ensemble(t_star=t0, ea_ev=EA_EV, n=AVRAMI_N)
        jitter = rng.normal(0.0, sd_t)  # THERMAL
        frac = models[shape].fraction(v, t)
        # move the condition by the thermal jitter using the local slope of the response
        eps = 5.0
        slope = (models[shape].fraction(v + eps, t) - frac) / eps
        dv_per_dt = np.where(np.abs(slope) > 1e-9, 1.0, 0.0)
        frac = np.clip(frac + slope * jitter * dv_per_dt * 0.1, 0.0, 1.0)
        sigma = NOISE_FLOOR + NOISE_BOUNDARY * frac * (1.0 - frac)  # READOUT
        out[i] = np.clip(frac + rng.normal(0.0, sigma), 0.0, 1.0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, default=ROOT / "data" / "flash_plan_seed.csv")
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    plan = pd.read_csv(args.plan)
    v = plan["voltage_V"].to_numpy(float)
    t = plan["time_ms"].to_numpy(float)
    draws = predictive_draws(v, t, args.draws, args.seed)
    lo, med, hi = np.percentile(draws, [5, 50, 95], axis=0)

    print(f"=== prediction for {args.plan.name}, {args.draws} draws ===")
    print("  Sampling model, transition location, thermal interpolation and readout noise.\n")
    hdr = f"  {'#':>3s} {'V':>5s} {'t':>6s} {'Tmax':>7s} {'median':>8s}"
    print(hdr + f" {'90% interval':>16s}  call")
    for i in range(len(v)):
        width = hi[i] - lo[i]
        if width > WIDE_INTERVAL:
            call = "CANNOT CALL"
        elif med[i] > 0.5:
            call = "crystallized"
        else:
            call = "amorphous"
        tm = plan["pred_Tmax_C"][i]
        left = f"  {int(plan['index'][i]):3d} {int(v[i]):5d} {t[i]:6.1f} {tm:7.1f}"
        print(f"{left} {med[i]:8.2f} {lo[i]:7.2f} - {hi[i]:<6.2f}  {call}")

    n_wide = int(np.sum(hi - lo > WIDE_INTERVAL))
    print(f"\n  {n_wide} of {len(v)} cannot be called: the interval spans most of [0,1].")
    print("  Those are the ones the batch is actually buying information about.")
    order = np.argsort(med)
    print("\n=== the falsifiable claim ===")
    print("  Absolute values are NOT comparable: the instrument reports permittivity in its own")
    print("  units with an uncalibrated zero; these are crystalline fractions. What CAN be")
    print("  checked is the ORDER. Predicted ranking, least to most transformed:")
    print("    " + " < ".join(str(int(plan["index"][i])) for i in order))
    print("  If the measured ranking disagrees badly, the thermal model is wrong -- a")
    print("  more useful outcome than agreement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
