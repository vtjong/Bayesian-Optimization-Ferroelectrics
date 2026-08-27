"""Building a batch of conditions and checking it before anyone fires it.

Separate from ``run_flash_plan.py`` because six other entry points need to BUILD a plan without
also writing a CSV and a figure to disk. A script that other scripts import is a library with a
``main()`` bolted on, and the coupling shows up as studies breaking when the CLI is edited.

The generator is deliberately conservative about where it will place a condition: only flash times
the measured table actually supports, only voltages reachable on the rising branch of the
inversion, and only after snapping to what the operator can dial in. Every one of those is asserted
in ``check`` rather than trusted, because a design that quietly drifts outside the characterized
box still produces a perfectly plausible-looking CSV.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.stats import qmc

sys.path.append(str(Path(__file__).resolve().parents[1]))

from design_space import is_snapped, min_separation, snap, snap_all
from physics.kinetics import build_ensemble, disagreement
from physics.thermal_model import FLASH, FLASH_T, V_HI, V_LO

# Boundary-search points are restricted to flash times the measured table actually supports. The
# table has NO node between 0.1 and 2.6 ms even though Tmax climbs 189 -> 563 C across that gap and
# the surface maximum sits inside it, so any Tmax quoted there is an artifact of the spline.
T_SEARCH_LO, T_SEARCH_HI = float(FLASH_T[1]), float(FLASH_T[-1])  # 2.6 .. 10.1 ms

# The batch is restricted to flash times the measured table supports, and to nothing else. A
# targeted design that concentrated conditions near the believed transition -- two matched
# peak-temperature levels crossed with two flash times -- was built and tested against this one. It
# was dominated on both the boundary-mapping metric and the dwell question it existed for, because
# concentration only pays if the belief is precise and ours is a bracket from six archival samples.
# The machinery for that comparison has been removed rather than left dormant.

SEED_SIZE = 10  # of an 80-specimen campaign
# No replicates. The reproducibility check they would buy is deferred to a later batch, so all ten
# specimens go to distinct conditions and the seed resolves the boundary to 55 C rather than 61 C.
# What is given up is real and worth stating: archival shots on this tool show two nominally
# identical conditions disagreeing by 2.08x, and until something replicates, a discrepant reading
# cannot be told apart from a genuine feature of the surface.
N_REPLICATES = 0
N_DRAWN = SEED_SIZE - N_REPLICATES

# The hypercube is stratified over PEAK TEMPERATURE and inverted through the measured table, not
# drawn over the raw knobs. Which of those wins turns entirely on how much is assumed about where
# the transition sits, and that dependence is worth stating because it nearly went unnoticed:
#
#   transition sampled uniformly 360-620 C  ->  knobs 0.508, temperature 0.511   (a tie)
#   transition sampled 420-560 C            ->  knobs 0.455, temperature 0.301
#
# (worst-case misclassified area, 90th percentile, over randomized worlds). The second prior is the
# defensible one. Six campaign-tool samples at 5.0 ms put a non-crystallized film at 434.7 C and a
# crystallized one at 458.7 C, so a transition at 380 C is not merely unlikely, it is contradicted
# by the measurement. A uniform prior reaching down to 360 C throws that evidence away, and under
# it the raw knobs look better only because nothing else is known.
#
# Stratifying temperature is therefore NOT a bet on the bracket midpoint -- the draw spans the whole
# reachable range and the design is scored against transitions anywhere in it. It is a bet that the
# film responds to temperature rather than to voltage, which the measured table already asserts.
LHS_TRIES = 400  # realizations to score; the best by maximin separation is kept

# Coldest peak temperature worth a specimen. Crystallizing 10 nm HZO on a millisecond timescale is
# not plausible below this on any kinetics, and the campaign-tool samples put the transition far
# above it. A physical floor, not an archival one.
DRAW_LO_C = 350.0
DRAW_MARGIN_C = 6.0  # keep draws off the reachable edge, where the inversion is ill-conditioned
FLOOR_CONDITION = (V_LO, 5.1)  # coldest reachable column at a supported time






def draw_block(n: int, seed: int, n_tries: int = LHS_TRIES) -> tuple:
    """Latin hypercube over ``(Tmax, log t)``, inverted through the measured table.

    Repeats over ``n_tries`` realizations and keeps the one with the largest minimum separation in
    normalized coordinates. A hypercube guarantees each axis is stratified but says nothing about
    the pair, so a realization can still leave two conditions almost coincident; maximin removes
    that without importing any belief about where the boundary is.

    :param n: number of conditions.
    :param seed: base RNG seed.
    :param n_tries: how many realizations to score.
    """
    best, best_sep = None, -np.inf
    log_lo, log_hi = np.log10(T_SEARCH_LO), np.log10(T_SEARCH_HI)
    for k in range(n_tries):
        u = qmc.LatinHypercube(d=2, seed=seed + k).random(n)
        t = snap_all(np.zeros_like(u[:, 1]), 10.0 ** (log_lo + u[:, 1] * (log_hi - log_lo)))[1]
        v = np.empty(n)
        for i, ti in enumerate(t):
            r_lo, r_hi = FLASH.tmax_range(float(ti))
            lo_c = max(r_lo + DRAW_MARGIN_C, DRAW_LO_C)
            hi_c = r_hi - DRAW_MARGIN_C
            if hi_c <= lo_c:  # this flash time cannot reach the window at all
                lo_c, hi_c = r_lo + DRAW_MARGIN_C, r_hi - DRAW_MARGIN_C
            v[i] = FLASH.voltage_for_tmax(lo_c + u[i, 0] * (hi_c - lo_c), float(ti))
        snapped = [snap(a, b) for a, b in zip(v, t)]
        v = np.array([p[0] for p in snapped])
        t = np.array([p[1] for p in snapped], float)
        if len(set(zip(v.tolist(), t.tolist()))) < n:
            continue  # snapping collapsed two conditions onto one another
        sep = min_separation(v, t)
        if sep > best_sep:
            best_sep, best = sep, (v, t)
    if best is None:
        raise RuntimeError("no feasible hypercube found")
    return best


def replicate_indices(v: np.ndarray, t: np.ndarray, n: int) -> list:
    """Which conditions to repeat on a second specimen: spread across peak temperature.

    The tempting choice is the condition nearest mid-transition, where the readout is noisiest and
    a disagreement is most visible. That was tried and rejected for two reasons. It needs the
    ensemble to say where mid-transition IS, which is the bet this design exists to avoid; and when
    the draw happens to place nothing near the boundary it silently replicates a SATURATED point
    instead, where two specimens agree trivially and the check is worthless.

    Spreading the replicates over the temperature range asks the reproducibility question in more
    than one regime and cannot degenerate that way. It also assumes nothing.

    :param v: drawn voltages.
    :param t: drawn flash times (ms).
    :param n: how many to replicate.
    """
    tm = FLASH.tmax(v, t)
    order = np.argsort(tm)
    # evenly spaced in rank, avoiding the two extremes where a replicate is least informative
    picks = np.linspace(0, len(order) - 1, n + 2)[1:-1]
    return [int(order[int(round(p))]) for p in picks]


def make_plan(n_core: int, seed: int) -> dict:
    """Assemble the seed plan and score every condition against the model ensemble.

    :param n_core: number of drawn conditions, before replicates.
    :param seed: RNG seed for the hypercube.
    """
    models = build_ensemble()
    v_d, t_d = draw_block(n_core, seed)
    rep = replicate_indices(v_d, t_d, N_REPLICATES)

    v = np.concatenate([v_d, v_d[rep]])
    t = np.concatenate([t_d, t_d[rep]])
    block = ["L"] * len(v_d) + ["R"] * len(rep)
    note = ["hypercube draw"] * len(v_d) + [
        f"replicate of L{i + 1} (separate specimen)" for i in rep
    ]
    order = np.lexsort((v, t, np.array([("LR".index(b)) for b in block])))
    return {
        "V": v[order],
        "t": t[order],
        "block": [block[i] for i in order],
        "note": [note[i] for i in order],
        "tmax": FLASH.tmax(v[order], t[order]),
        "preds": {k: m.fraction(v[order], t[order]) for k, m in models.items()},
        "disagree": disagreement(models, v[order], t[order]),
        "models": models,
    }


def check(plan: dict) -> None:
    """Assert the design properties the plan is supposed to guarantee."""
    v, t, block = plan["V"], plan["t"], plan["block"]
    assert len(v) == SEED_SIZE, f"expected {SEED_SIZE} shots, got {len(v)}"
    assert np.all(t >= T_SEARCH_LO - 1e-9), "a condition lands in the un-noded 0.1-2.6 ms gap"
    assert np.all(t <= T_SEARCH_HI + 1e-9), "a condition exceeds the measured time range"
    assert np.all(v >= V_LO) and np.all(v <= V_HI), "a condition leaves the voltage box"
    assert is_snapped(v, t), "not snapped to the tool's resolution"
    assert block.count("R") == N_REPLICATES, "wrong number of replicates"

    pairs = list(zip(v.astype(int).tolist(), np.round(t, 1).tolist()))
    drawn = [p for p, b in zip(pairs, block) if b == "L"]
    assert len(set(drawn)) == len(drawn), "two drawn conditions collided after snapping"
    for p, b in zip(pairs, block):
        if b == "R":
            assert p in drawn, "a replicate does not match any drawn condition"


