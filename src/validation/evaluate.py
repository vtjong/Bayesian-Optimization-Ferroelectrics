"""Scoring a seed design: what it would let us conclude, and how wrong that could be.

This lives in the library rather than in a run script because three scripts need it and scripts
importing scripts is how a CLI flag ends up changing a result. The design-comparison scripts are
thin front ends over this module.

THE METRIC IS SCALE-FREE. With an unknown readout floor and span -- large-signal permittivity has
an uncalibrated zero and gain -- thresholding the observed number at one half would measure the
instrument rather than the design. Fitting the LATENT logit field instead makes the boundary the
zero level set, which is invariant to any affine distortion of the readout, so no normalization
convention has to be agreed in advance.
"""

from typing import Callable, Tuple

import numpy as np

from active_learning.surrogate import BoundarySurrogate
from physics.thermal_model import V_HI, V_LO

GRID_V, GRID_T = 90, 70  # resolution of the misclassification integral
CLASS_THRESHOLD = 0.5


def supported_grid(t_lo: float, t_hi: float) -> Tuple[np.ndarray, np.ndarray]:
    """Dense ``(V, t)`` mesh over the region the measured table supports.

    :param t_lo: shortest supported flash time (ms).
    :param t_hi: longest supported flash time (ms).
    """
    v = np.linspace(V_LO, V_HI, GRID_V)
    t = np.geomspace(t_lo, t_hi, GRID_T)
    return np.meshgrid(v, t)


def has_boundary(truth: Callable, vv: np.ndarray, tt: np.ndarray) -> bool:
    """Whether the true response actually crosses the threshold inside the box.

    Worlds whose boundary lies outside the design box are uninformative for EVERY design alike --
    nothing can be learned and nothing distinguishes one seed from another -- so they are excluded
    rather than averaged in, where they would inflate every design's tail equally and hide the
    differences that matter.

    :param truth: ground-truth response.
    :param vv: voltage grid.
    :param tt: flash-time grid.
    """
    x = truth(vv.ravel(), tt.ravel())
    return bool(x.max() >= CLASS_THRESHOLD and x.min() <= CLASS_THRESHOLD)


def misclassified_area(
    v: np.ndarray, t: np.ndarray, y: np.ndarray, truth: Callable, vv: np.ndarray, tt: np.ndarray
) -> float:
    """Fraction of the box a surrogate fitted to ``(v, t, y)`` puts on the wrong side.

    :param v: as-fired voltages.
    :param t: as-fired flash times (ms).
    :param y: observed readings.
    :param truth: ground-truth response.
    :param vv: voltage grid.
    :param tt: flash-time grid.
    """
    gp = BoundarySurrogate().fit(v, t, y)
    predicted = gp.crystalline_side(vv.ravel(), tt.ravel())
    actual = truth(vv.ravel(), tt.ravel()) > CLASS_THRESHOLD
    return float(np.mean(predicted != actual))
