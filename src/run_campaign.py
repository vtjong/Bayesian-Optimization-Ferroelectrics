"""Advance the campaign: read what came back from the lab, propose what to fire next.

One command per cycle. It reads the measured results, fits the boundary surrogate to them, reports
what the campaign now believes and how confident it is entitled to be, decides whether to stop, and
if not writes the next batch as a plan plus a blank results sheet in the same format.

    python src/run_campaign.py                 # report and propose
    python src/run_campaign.py --no-propose    # report only, fire nothing

WHAT IT WILL NOT DO. It will not fit anything to a results sheet that is ambiguous. A blank cell is
never read as a zero, a value on a row not marked measured is an error, and an unfilled sheet fails
loudly rather than reading as a batch of not-run specimens. Those rules are enforced upstream in
``campaign.results`` and every one of them is a regression test against a defect already present
in this project's archival data.

WHAT THE STOPPING RULE IS, AND WHAT IT IS NOT. The campaign stops when the surrogate's own boundary
stops moving between cycles -- measured as the area that changes side, which is a quantity a real
campaign can compute without knowing the truth. It deliberately does NOT stop on a self-reported
accuracy: at small n a model's estimate of its own error is optimistic exactly when it is most
wrong, so a threshold on it would fire early and confidently. Stability across cycles is weaker but
honest. It is a signal to review, not an automatic halt.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from active_learning.acquisition import (
    DEFAULT_ACQUISITION,
    binary_entropy,
    p_crystalline,
    select_batch,
)
from active_learning.surrogate import BoundarySurrogate
from campaign.plan import T_SEARCH_HI, T_SEARCH_LO
from campaign.reporting import boundary_conditions
from campaign.results import CONTROL_INDEX, PLAN_KEY, blank_template, load, unfired
from physics.thermal_model import FLASH
from validation.evaluate import supported_grid

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BATCH_SIZE = 3  # specimens per cycle
STABILITY_TOLERANCE = 0.01  # boundary movement between cycles that counts as converged
# Mean boundary entropy over the supported box, in nats. The lab-usable convergence proxy: it needs
# no ground truth, unlike a misclassified area, which can only be computed against a known answer
# and is therefore a synthetic-campaign diagnostic rather than something a real campaign can stop
# on. Falls as the surface resolves, because entropy is high only where the model still cannot call
# the phase. It saturates at ln 2 = 0.693 when the model is undecided everywhere.
ENTROPY_TOLERANCE = 0.05
MIN_MEASURED = 6  # below this the surrogate is not worth proposing from
# How far above the as-deposited control a reading must sit before it counts as a real change.
CONTROL_LIFT_SIGMAS = 3.0


def _plans() -> list:
    """Every plan CSV in the campaign, seed first, in the order they were fired."""
    seed = DATA / "flash_plan_seed.csv"
    # exclude the results sheets, which share the prefix and would otherwise be read as plans
    later = sorted(p for p in DATA.glob("flash_plan_batch*.csv") if not p.stem.endswith("_results"))
    return [p for p in [seed, *later] if p.exists()]


def _results_for(plan_path: Path) -> Path:
    """The results sheet belonging to a plan."""
    return plan_path.with_name(plan_path.stem + "_results.csv")


def gather() -> tuple:
    """Every measured condition across every fired batch, plus what is still outstanding.

    Returns ``(V, t, y, report)`` where the arrays hold as-fired conditions and readings.
    """
    v_all, t_all, y_all, lines = [], [], [], []
    control = None
    for plan_path in _plans():
        res_path = _results_for(plan_path)
        if not res_path.exists():
            lines.append(f"  {plan_path.name}: no results sheet yet")
            continue
        plan = pd.read_csv(plan_path)
        try:
            res = load(res_path, plan)
        except ValueError as exc:
            # An unfilled or malformed sheet is a STATE, not a crash -- the lab will hit this
            # every cycle between firing and reading. Report it and carry on with what exists.
            lines.append(f"  {plan_path.name}: results sheet not usable yet -- {exc}")
            continue
        v, t, y = res.conditions()
        is_control = res.table.loc[res.measured, PLAN_KEY].to_numpy() == CONTROL_INDEX
        if is_control.any():
            control = float(y[is_control][0])
        v_all.append(v[~is_control])
        t_all.append(t[~is_control])
        y_all.append(y[~is_control])
        missing = [i for i in unfired(res) if i != CONTROL_INDEX]
        lines.append(
            f"  {plan_path.name}: {int((~is_control).sum())} fired conditions measured"
            + (f", NOT USABLE {missing}" if missing else "")
        )
        if len(res.drift):
            lines.append(f"      {len(res.drift)} shot(s) drifted from plan; as-fired values used")
    if not v_all:
        return np.array([]), np.array([]), np.array([]), lines, control
    return (
        np.concatenate(v_all),
        np.concatenate(t_all),
        np.concatenate(y_all),
        lines,
        control,
    )


def _report_lift(y: np.ndarray, control: float) -> None:
    """Report how far the batch rises above an un-annealed film, and refuse to over-read it.

    THE FAILURE THIS IS ABOUT, which is not hypothetical. The readout has no calibrated zero, so
    the latent transform maps the observed range onto (0, 1). If every condition fired lands on the
    same side of the transition, that normalization stretches measurement scatter across the full
    range and the surrogate draws a confident boundary through noise. Observed on a simulated
    campaign: readings of 0.117 to 0.238 where the true fraction was 0.000 everywhere produced a
    surface calling most of the box crystallized, and the campaign then spent every cycle chasing
    it.

    WHY THIS ONLY REPORTS. Two candidate tests were built and both failed. The fitted
    hyperparameters cannot see it -- on that same batch the GP reported a signal ten times its own
    noise, because after normalization the scatter IS the field being fitted. And comparing the
    hottest reading to the control cannot separate them either: with one control specimen and no
    replicates, the control is a single noisy draw while the maximum of ten draws sits about 1.3
    standard deviations high by construction, so the comparison is between two noise realizations.
    Tested across six worlds it returned the same verdict for a batch whose true fraction never
    exceeded 0.002 and one that reached 1.000.

    WHAT WOULD RESOLVE IT: replicated conditions, or several control specimens. Either gives a
    noise estimate that does not contain the signal. The seed as fired has neither, so this prints
    the numbers and leaves the judgement to a reader who knows what the readout does.

    :param y: readings from the fired conditions.
    :param control: reading from the as-deposited control, or None.
    """
    print("\n=== is any of this above an un-annealed film? ===")
    if control is None:
        print("  No as-deposited control measured, so there is no reference at all and the")
        print("  question cannot be approached. Measure index 0 of the seed results sheet.")
        return
    order = np.sort(y)
    print(f"  as-deposited control      {control:.4g}")
    print(f"  lowest three readings     {', '.join(f'{x:.4g}' for x in order[:3])}")
    print(f"  highest three readings    {', '.join(f'{x:.4g}' for x in order[-3:])}")
    print(f"  hottest sits {np.max(y) - control:+.4g} from the control")
    print("  JUDGE THIS BY EYE. If the top readings are not clearly separated from the control")
    print("  and from each other, nothing crystallized and the boundary below is fitted to")
    print("  scatter -- fire hotter instead of refining it. This cannot be decided automatically")
    print("  without replicates or repeated controls; see the note in the source.")


def mean_boundary_entropy(gp: BoundarySurrogate, vv: np.ndarray, tt: np.ndarray) -> float:
    """Average binary entropy of the class call over the supported box, in nats.

    The quantity a real campaign can actually watch. High while the model is undecided over much of
    the box, falling as the boundary resolves. It says nothing about being RIGHT -- a confidently
    wrong surface scores well -- so it is a necessary condition for stopping, never a sufficient
    one.

    :param gp: fitted surrogate.
    :param vv: voltage grid.
    :param tt: flash-time grid.
    """
    mu, sd = gp.latent(vv.ravel(), tt.ravel())
    return float(np.mean(binary_entropy(p_crystalline(mu, sd))))


def _report_boundary(gp: BoundarySurrogate) -> np.ndarray:
    """Where the surrogate puts the boundary, as peak temperature against flash time."""
    v, t, tm = boundary_conditions(gp, T_SEARCH_LO, T_SEARCH_HI)
    if v.size == 0:
        print("\n=== boundary ===\n  no boundary inside the box: the surface is one-sided")
        print("  Every condition fired so far is on the same side. Fire hotter or colder before")
        print("  reading anything else here.")
        return np.array([])
    print("\n=== where the boundary sits now ===")
    print(f"  {'t (ms)':>8s} {'V':>8s} {'Tmax':>9s}")
    for i in np.linspace(0, v.size - 1, min(6, v.size)).astype(int):
        print(f"  {t[i]:8.1f} {v[i]:8.0f} {tm[i]:8.0f} C")
    print(f"  peak temperature along the boundary: {tm.min():.0f} - {tm.max():.0f} C")
    if tm.max() - tm.min() > 15.0:
        print("  It is NOT a constant-temperature contour -- the boundary moves with dwell, which")
        print("  is the campaign's central question. Treat as provisional until replicated.")
    else:
        print("  Consistent with a constant-temperature threshold over this dwell range.")
    return tm


def _report_stability(gp: BoundarySurrogate, v, t, y, vv, tt) -> float:
    """How much the boundary moved when the most recent batch was added.

    The honest convergence signal: refit without the last batch and measure the area that changes
    side. It needs no ground truth, which a self-reported accuracy silently does.
    """
    n_prev = max(len(v) - BATCH_SIZE, MIN_MEASURED)
    if n_prev >= len(v):
        return float("nan")
    prev = BoundarySurrogate().fit(v[:n_prev], t[:n_prev], y[:n_prev])
    moved = float(
        np.mean(
            prev.crystalline_side(vv.ravel(), tt.ravel())
            != gp.crystalline_side(vv.ravel(), tt.ravel())
        )
    )
    print("\n=== is it still moving? ===")
    print(f"  adding the last {len(v) - n_prev} conditions moved {moved:.1%} of the box")
    if moved < STABILITY_TOLERANCE:
        print(f"  Below the {STABILITY_TOLERANCE:.0%} stability tolerance. A candidate to stop on,")
        print("  but stability is not accuracy: a wrong boundary can be stable. Confirm by firing")
        print("  two conditions ON the predicted boundary and checking they straddle it.")
    return moved


def _write_batch(v: np.ndarray, t: np.ndarray, index: int) -> Path:
    """Write the proposed batch as a plan plus its blank results sheet."""
    path = DATA / f"flash_plan_batch{index:02d}.csv"
    plan = pd.DataFrame(
        {
            PLAN_KEY: np.arange(1, v.size + 1),
            "block": ["P"] * v.size,
            "voltage_V": v.astype(int),
            "time_ms": t,
            "pred_Tmax_C": np.round(FLASH.tmax(v, t), 1),
            "readout": "eps_r",
            "note": [f"proposed, cycle {index}"] * v.size,
        }
    )
    plan.to_csv(path, index=False)
    blank_template(plan).to_csv(_results_for(path), index=False)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--acquisition", default=DEFAULT_ACQUISITION)
    ap.add_argument("--no-propose", action="store_true", help="report only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    v, t, y, lines, control = gather()
    print("=== what has been measured ===")
    for line in lines:
        print(line)
    if v.size == 0:
        print("\nNothing measured yet. Fire the seed and fill in its results sheet.")
        return 0
    print(f"  {v.size} fired conditions in total")
    if control is None:
        print("  NO as-deposited control measured. Nothing establishes what an un-annealed film")
        print("  reads, so a low reading cannot be told from an already-transformed one.")
    else:
        print(f"  as-deposited control reads {control:.3g}")

    if v.size < MIN_MEASURED:
        print(f"\nFewer than {MIN_MEASURED} conditions; too few to propose from. Fire more.")
        return 0

    gp = BoundarySurrogate().fit(v, t, y)
    h = gp.hyperparameters
    print("\n=== the fitted surface ===")
    print(
        f"  lengthscales {h['ell_v']:.2f} (V) / {h['ell_logt']:.2f} (log t), "
        f"noise {h['sigma_n']:.3f}, amplitude {h['sigma_f']:.2f}"
    )
    if min(h["ell_v"], h["ell_logt"]) < 0.08 or max(h["ell_v"], h["ell_logt"]) > 5.0:
        print("  WARNING: a lengthscale has run to its bound. The fit is degenerate and the")
        print("  boundary below should not be trusted. Add conditions before proposing again.")

    _report_lift(y, control)
    vv, tt = supported_grid(T_SEARCH_LO, T_SEARCH_HI)
    _report_boundary(gp)
    moved = _report_stability(gp, v, t, y, vv, tt)

    entropy = mean_boundary_entropy(gp, vv, tt)
    print(f"\n=== convergence proxy ===\n  mean boundary entropy {entropy:.4f} nats "
          f"(saturates at {np.log(2):.3f}; tolerance {ENTROPY_TOLERANCE})")
    settled = entropy < ENTROPY_TOLERANCE and np.isfinite(moved) and moved < STABILITY_TOLERANCE
    if settled:
        print("  Both the entropy proxy and the between-cycle movement are below tolerance.")
        print("  STOP CANDIDATE. Neither is a measure of being right -- confirm by firing two")
        print("  conditions on the predicted boundary and checking they land either side of it.")

    if args.no_propose:
        return 0
    bv, bt = select_batch(
        gp, v, t, args.batch, T_SEARCH_LO, T_SEARCH_HI, args.acquisition, seed=args.seed
    )
    index = len(_plans())
    path = _write_batch(bv, bt, index)
    print(f"\n=== proposed next: {args.batch} conditions by {args.acquisition} ===")
    for a, b in zip(bv, bt):
        print(f"  {int(a):4d} V  {b:5.1f} ms   -> Tmax {FLASH.tmax(a, b):.0f} C")
    print(f"\nSaved -> {path.relative_to(ROOT)} and its blank results sheet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
