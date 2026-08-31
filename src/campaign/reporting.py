"""Turning a fitted surface into something a person can read.

Everything here needs BOTH the learner and the thermal model, which is exactly why it is not in
either. The surrogate works in normalized coordinates on a latent field; nobody sets a lamp in
those units. Reporting is where the fitted boundary is put back into volts, milliseconds and
degrees so it can be checked against the tool and against the film.

Keeping this separate is what lets ``active_learning`` stay physics-free: the boundary is FOUND
without any reference to the thermal model, and only afterwards LABELLED with the temperature that
model predicts. If the two steps were in one function the separation would be untestable.
"""

import numpy as np

from active_learning.surrogate import BoundarySurrogate
from design_space import V_HI, V_LO
from physics.thermal_model import tmax_envelope


def boundary_conditions(gp: BoundarySurrogate, t_lo: float, t_hi: float, n: int = 240) -> tuple:
    """Where the fitted surface puts the boundary: ``(V, t, Tmax_lo, Tmax_hi)`` along it.

    Found by bisecting the latent mean in voltage at each flash time, which is what a reader
    wants to see -- the boundary as a curve in the controls the operator sets. The temperature
    is returned as the envelope of the available simulations rather than a point: the boundary
    itself is located WITHOUT any thermal model, and attaching one number to it afterwards would
    imply a precision that no measurement of this tool supports.

    :param gp: fitted surrogate.
    :param t_lo: shortest supported flash time (ms).
    :param t_hi: longest supported flash time (ms).
    :param n: how many flash times to report.
    """
    times = np.geomspace(t_lo, t_hi, n)
    out_v, out_t = [], []
    for t in times:
        lo, hi = V_LO, V_HI
        if gp.latent(np.array([lo]), np.array([t]))[0][0] > 0:
            continue  # already crystallized at the coldest voltage
        if gp.latent(np.array([hi]), np.array([t]))[0][0] < 0:
            continue  # never crystallized at this flash time
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if gp.latent(np.array([mid]), np.array([t]))[0][0] < 0:
                lo = mid
            else:
                hi = mid
        out_v.append(0.5 * (lo + hi))
        out_t.append(float(t))
    v = np.array(out_v)
    t = np.array(out_t)
    if not v.size:
        return v, t, np.array([]), np.array([])
    lo, hi = tmax_envelope(v, t)
    return v, t, lo, hi
