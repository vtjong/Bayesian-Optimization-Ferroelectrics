"""Choosing where to spend the next few specimens.

Two pieces: a score that says how much a candidate condition is worth, and a rule for picking
several at once without asking the same question twice.

WHAT A GOOD SCORE HAS TO DO. The campaign is looking for a boundary, so a candidate is worth
firing when the model cannot say which side of it the condition falls on AND more data would
settle that. Those are two different things, and the distinction is what separates the acquisitions
below. A condition sitting exactly on the predicted boundary is unresolved, but if it has already
been measured, firing it again resolves nothing.

  ``targeted_variance``  the default. Posterior variance weighted toward the threshold, so it is
                         large only where the surface is BOTH uncertain and plausibly on the
                         boundary. It self-suppresses: variance collapses at a measured condition
                         and takes the whole score with it.
  ``straddle``           the level-set literature's standard, ``beta*sd - |mu|``. Also
                         self-suppressing, better known, and slightly more aggressive at the edges
                         of the box.
  ``boundary_entropy``   kept for comparison and NOT recommended. See below.

WHY BOUNDARY ENTROPY IS NOT THE DEFAULT. It scores by the binary entropy of the class probability,
``H(Phi(mu/sd))``, which is an appealing idea: it peaks where the model genuinely cannot call the
phase, and because the argument is a ratio, a point far from data with large uncertainty also
scores well, so exploration appears to come for free.

The flaw is that the argument is a ratio. At ``mu = 0`` the score equals its maximum ``ln 2`` for
ANY sd -- verified identical to machine precision from sd = 0.01 to sd = 50 -- so the entire
predicted contour is a flat plateau and the choice between a resolved boundary point and an
unexplored one is settled by whichever the optimizer happens to report first. In the founding
level-set benchmark this criterion failed to converge on three of four problems and was
indistinguishable from random search on the fourth. An exclusion radius patches the symptom by
forbidding re-measurement, but then the radius rather than the score is doing the work.

BATCHING BY FANTASY. Taking the top k candidates fails: they cluster at the single most uncertain
spot and ask the same question k times. Instead, after each pick the model is told to believe its
own posterior mean there. That collapses the local uncertainty, so the next pick lands elsewhere on
the boundary. The fantasy adds no information -- it is what the model already predicted -- and it
is discarded once the real measurements arrive.

Hyperparameters AND the constant mean are frozen through the fantasy loop. Freezing the
hyperparameters is standard; freezing the mean matters because every pick sits near the boundary,
so every fantasy value is near zero, and a mean recomputed over real-plus-fantasy data walks toward
the threshold as the batch grows -- manufacturing exploration pressure that increases with k.
"""

from typing import Callable, Dict, Tuple

import numpy as np
from scipy.stats import norm

from design_space import V_HI, V_LO, normalize, snap_all

from .surrogate import BoundarySurrogate

# No candidate within this distance of an existing or pending condition may be chosen. In the
# normalized box 0.04 is about 8 V or 0.08 of a decade in time -- comfortably inside the tool's
# resolution, so it costs nothing real. With a self-suppressing acquisition it is a backstop rather
# than the mechanism; with boundary entropy it is the only thing preventing re-measurement.
EXCLUDE_RADIUS = 0.04
STRADDLE_BETA = 1.96  # a nominal 95% interval, the literature's default
N_CANDIDATES = 4000  # random candidates beat a grid: no alignment artifacts
_EPS = 1.0e-12


def p_crystalline(mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Probability the latent field is above zero, i.e. that the condition crystallized.

    :param mu: posterior mean of the latent field.
    :param sd: posterior standard deviation of the latent field.
    """
    return norm.cdf(np.asarray(mu, float) / np.maximum(np.asarray(sd, float), _EPS))


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Binary entropy in nats, zero at both saturated ends.

    :param p: class probability.
    """
    p = np.clip(np.asarray(p, float), _EPS, 1.0 - _EPS)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def targeted_variance(mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Posterior variance weighted toward the boundary. THE DEFAULT.

    ``sd^2`` times a Gaussian in the latent mean, so the score is large only where the surface is
    both uncertain and plausibly on the boundary, and collapses to zero wherever either fails.

    :param mu: posterior mean of the latent field.
    :param sd: posterior standard deviation of the latent field.
    """
    var = np.maximum(np.asarray(sd, float) ** 2, _EPS)
    return var * np.exp(-0.5 * np.asarray(mu, float) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def straddle(mu: np.ndarray, sd: np.ndarray, beta: float = STRADDLE_BETA) -> np.ndarray:
    """Straddle utility ``beta*sd - |mu|``: does the confidence interval straddle the boundary?

    :param mu: posterior mean of the latent field.
    :param sd: posterior standard deviation of the latent field.
    :param beta: interval width in standard deviations.
    """
    return beta * np.asarray(sd, float) - np.abs(np.asarray(mu, float))


def boundary_entropy(mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Binary entropy of the class probability. Kept for comparison; see the module note.

    :param mu: posterior mean of the latent field.
    :param sd: posterior standard deviation of the latent field.
    """
    return binary_entropy(p_crystalline(mu, sd))


ACQUISITIONS: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "targeted_variance": targeted_variance,
    "straddle": straddle,
    "boundary_entropy": boundary_entropy,
}
DEFAULT_ACQUISITION = "targeted_variance"


def candidate_pool(t_lo: float, t_hi: float, n: int = N_CANDIDATES, seed: int = 0) -> tuple:
    """Random settable conditions to choose among, restricted to supported flash times.

    Drawn uniformly in the normalized box and then snapped, so the pool inherits no grid alignment.
    Conditions the tool cannot be set to are never proposed.

    :param t_lo: shortest supported flash time (ms).
    :param t_hi: longest supported flash time (ms).
    :param n: pool size.
    :param seed: RNG seed.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=(n, 2))
    lo, hi = np.log10(t_lo), np.log10(t_hi)
    v, t = snap_all(V_LO + u[:, 0] * (V_HI - V_LO), 10.0 ** (lo + u[:, 1] * (hi - lo)))
    keep = (t >= t_lo) & (t <= t_hi)
    return v[keep], t[keep]


def _nearest_distance(cand: np.ndarray, taken: np.ndarray) -> np.ndarray:
    """Distance from each candidate to the closest already-spoken-for condition."""
    if taken.size == 0:
        return np.full(cand.shape[0], np.inf)
    d2 = (
        np.sum(cand**2, 1)[:, None]
        + np.sum(taken**2, 1)[None, :]
        - 2.0 * cand @ taken.T
    )
    return np.sqrt(np.maximum(d2.min(axis=1), 0.0))


def select_batch(
    gp: BoundarySurrogate,
    fired_v: np.ndarray,
    fired_t: np.ndarray,
    k: int,
    t_lo: float,
    t_hi: float,
    acquisition: str = DEFAULT_ACQUISITION,
    exclude_radius: float = EXCLUDE_RADIUS,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Choose ``k`` conditions by acquisition, spread by fantasy conditioning.

    Raises rather than returning duplicates if the exclusion radius leaves nothing to choose. The
    silent alternative -- returning the same condition k times because every candidate was masked
    and the argmax fell through to index zero -- fabricates experiments that violate the very
    constraint that produced the state, and does it without a warning.

    :param gp: surrogate fitted to the real measurements only.
    :param fired_v: voltages already fired.
    :param fired_t: flash times already fired (ms).
    :param k: batch size.
    :param t_lo: shortest supported flash time (ms).
    :param t_hi: longest supported flash time (ms).
    :param acquisition: which score to use; see ``ACQUISITIONS``.
    :param exclude_radius: minimum separation from any existing or pending condition.
    :param seed: RNG seed for the candidate pool.
    """
    score_fn = ACQUISITIONS[acquisition]
    cand_v, cand_t = candidate_pool(t_lo, t_hi, seed=seed)
    cand = normalize(cand_v, cand_t)

    scratch = gp
    taken = normalize(np.asarray(fired_v, float), np.asarray(fired_t, float))
    picked_v, picked_t = [], []
    for _ in range(k):
        mu, sd = scratch.latent(cand_v, cand_t)
        score = score_fn(mu, sd)
        score = np.where(_nearest_distance(cand, taken) < exclude_radius, -np.inf, score)
        if not np.isfinite(score).any():
            raise RuntimeError(
                f"every candidate is within {exclude_radius} of an existing or pending condition; "
                f"reduce the exclusion radius or the batch size ({len(picked_v)} of {k} chosen)"
            )
        j = int(np.argmax(score))
        picked_v.append(float(cand_v[j]))
        picked_t.append(float(cand_t[j]))
        taken = np.vstack([taken, cand[j][None, :]])
        # believe the model's own prediction here, so the next pick goes elsewhere
        scratch = scratch.believe(
            np.array([cand_v[j]]), np.array([cand_t[j]]), scratch.latent(cand_v[j], cand_t[j])[0]
        )
    return np.array(picked_v), np.array(picked_t)
