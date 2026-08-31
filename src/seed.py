"""Prespecified seed design for the flash-anneal campaign.

The seed covers the instrument's full settable design space. Voltage is
Latin-hypercube stratified linearly and pulse width is stratified uniformly
in log10 space.

Candidate designs are snapped to instrument resolution. Designs containing
duplicate setpoints after snapping are discarded, and the candidate with the
largest minimum pairwise distance in normalized ``(voltage, log10 pulse)``
coordinates is selected.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True, slots=True)
class InstrumentAxis:
    """Settable range for one instrument control.

    :param minimum: Minimum settable value.
    :param maximum: Maximum settable value.
    :param step: Instrument setting resolution.
    """

    minimum: float
    maximum: float
    step: float


VOLTAGE = InstrumentAxis(
    minimum=506.0,
    maximum=716.0,
    step=1.0,
)

PULSE_WIDTH = InstrumentAxis(
    minimum=0.1,
    maximum=10.1,
    step=0.1,
)

SEED_SIZE = 10
RNG_SEED = 20260831
MAXIMIN_DRAWS = 400


def unit_to_setpoints(unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map unit-square points to instrument setpoints.

    Voltage is mapped linearly. Pulse width is mapped linearly in ``log10``
    space so that the Latin hypercube stratifies pulse-width decades rather
    than raw milliseconds.

    Both quantities are then snapped to instrument resolution.

    :param unit: Unit-square points with shape ``(n, 2)``.
    :return: Flash voltages in V and pulse widths in ms.
    """
    voltage_v = VOLTAGE.minimum + unit[:, 0] * (VOLTAGE.maximum - VOLTAGE.minimum)

    log_pulse_min = np.log10(PULSE_WIDTH.minimum)
    log_pulse_max = np.log10(PULSE_WIDTH.maximum)
    log_pulse_width = log_pulse_min + unit[:, 1] * (log_pulse_max - log_pulse_min)
    pulse_width_ms = 10.0**log_pulse_width

    voltage_v = np.round(voltage_v / VOLTAGE.step) * VOLTAGE.step
    pulse_width_ms = np.round(pulse_width_ms / PULSE_WIDTH.step) * PULSE_WIDTH.step

    pulse_width_ms = np.clip(
        np.round(pulse_width_ms, 1),
        PULSE_WIDTH.minimum,
        PULSE_WIDTH.maximum,
    )

    return voltage_v, pulse_width_ms


def to_design_coordinates(
    voltage_v: np.ndarray,
    pulse_width_ms: np.ndarray,
) -> np.ndarray:
    """Map instrument setpoints to normalized surrogate coordinates.

    The surrogate operates in normalized ``(voltage, log10 pulse width)``
    coordinates. The same coordinates are used to evaluate seed separation.

    :param voltage_v: Flash voltages in V.
    :param pulse_width_ms: Pulse widths in ms.
    :return: Normalized coordinates with shape ``(n, 2)``.
    """
    voltage_coordinate = (voltage_v - VOLTAGE.minimum) / (VOLTAGE.maximum - VOLTAGE.minimum)

    log_pulse_min = np.log10(PULSE_WIDTH.minimum)
    log_pulse_max = np.log10(PULSE_WIDTH.maximum)
    pulse_coordinate = (np.log10(pulse_width_ms) - log_pulse_min) / (log_pulse_max - log_pulse_min)

    return np.column_stack((voltage_coordinate, pulse_coordinate))


def minimum_separation(
    voltage_v: np.ndarray,
    pulse_width_ms: np.ndarray,
) -> float:
    """Return the minimum pairwise distance between seed conditions.

    Distance is measured in the normalized coordinates used by the surrogate,
    making this the maximin criterion used to select among candidate designs.

    :param voltage_v: Flash voltages in V.
    :param pulse_width_ms: Pulse widths in ms.
    :return: Minimum normalized pairwise distance.
    """
    coordinates = to_design_coordinates(voltage_v, pulse_width_ms)

    distances = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :],
        axis=2,
    )
    np.fill_diagonal(distances, np.inf)

    return float(distances.min())


def make_seed(
    n: int = SEED_SIZE,
    rng_seed: int = RNG_SEED,
    maximin_draws: int = MAXIMIN_DRAWS,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the prespecified maximin Latin-hypercube seed.

    Each candidate is generated from a deterministic RNG seed, mapped onto
    instrument setpoints, and rejected only if snapping produces duplicate
    conditions. Among the remaining candidates, the design with the largest
    minimum separation is retained.

    :param n: Number of experimental conditions.
    :param rng_seed: Base RNG seed. Candidate ``k`` uses ``rng_seed + k``.
    :param maximin_draws: Number of candidate Latin hypercubes to evaluate.
    :return: Voltages and pulse widths sorted by pulse width, then voltage.
    :raises RuntimeError: If every candidate contains duplicate setpoints.
    """
    best_score = -np.inf
    best_voltage_v = None
    best_pulse_width_ms = None

    for draw_index in range(maximin_draws):
        unit = qmc.LatinHypercube(
            d=2,
            seed=rng_seed + draw_index,
        ).random(n)

        voltage_v, pulse_width_ms = unit_to_setpoints(unit)

        conditions = np.column_stack((voltage_v, pulse_width_ms))
        if np.unique(conditions, axis=0).shape[0] != n:
            continue

        score = minimum_separation(voltage_v, pulse_width_ms)
        if score > best_score:
            best_score = score
            best_voltage_v = voltage_v
            best_pulse_width_ms = pulse_width_ms

    if best_voltage_v is None or best_pulse_width_ms is None:
        raise RuntimeError(
            f"no realization produced {n} distinct settable conditions "
            f"across {maximin_draws} draws"
        )

    order = np.lexsort((best_voltage_v, best_pulse_width_ms))
    return best_voltage_v[order], best_pulse_width_ms[order]
