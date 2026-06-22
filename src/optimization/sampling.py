"""Latin-hypercube initial sampling for the experimental design space.

Generates space-filling seed points over a (voltage, time) box for round 0 of the
campaign: LHS over a domain physically restricted so it brackets the crystallization
boundary. Thin wrapper over scipy's QMC LatinHypercube.
"""

import numpy as np
from scipy.stats import qmc


def latin_hypercube(bounds, n, seed=0):
    """Latin-hypercube sample of ``n`` points in a box.

    :param bounds: list of (lo, hi) per dimension, e.g. [(v_lo, v_hi), (t_lo, t_hi)]
    :param n: number of seed points
    :param seed: RNG seed (reproducible)
    :return: array (n, d) of points in physical units
    """
    bounds = np.asarray(bounds, dtype=float)
    unit = qmc.LatinHypercube(d=bounds.shape[0], seed=seed).random(n)  # (n, d) in [0,1]
    return qmc.scale(unit, bounds[:, 0], bounds[:, 1])
