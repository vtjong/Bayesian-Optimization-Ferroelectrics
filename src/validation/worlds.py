"""Randomly generated worlds in which any of the campaign's assumptions may be wrong.

``adversarial.py`` varies the crystallization response while holding everything else fixed --
most importantly the measured peak-temperature table, which every truth there shares with every
design. That quietly guarantees the one thing the design most depends on, and it flatters designs
that navigate in thermal coordinates. A design that places conditions by inverting the table cannot
be penalised for the table being wrong if the truth is defined through the same table.

This module removes that guarantee. A world samples, independently:

  THERMAL         interpolation error ONLY. The 30 measured peak temperatures are treated as
                  correct -- they are the one direct measurement this campaign owns -- so a world's
                  thermal truth agrees with the table AT THE NODES and may deviate only between
                  them. The deviation is scaled by the GP's own posterior standard deviation, which
                  is 0.14 C at a node and rises to ~16 C midway between rows, so the size of the
                  allowed error is measured rather than invented.
  RESPONSE        which functional family the boundary belongs to, where it sits, how sharp it is,
                  how much it tilts with dwell, whether it curves, whether it rolls over, whether a
                  second cold branch exists at long dwell.
  NON-THERMAL     a direct dependence on voltage, breaking the assumption that (V, t) acts only
                  through temperature.
  NOISE           an overall scale, a heavy tail (occasional wild readings), and a per-specimen
                  offset -- the archival replicates disagree by 2.08x, so specimen scatter is not
                  hypothetical.
  READOUT         an unknown floor and span, and saturation, so the observed number is an affine,
                  possibly compressed function of the crystalline fraction rather than the fraction
                  itself.

Nothing here is tuned to any design. The point is to ask which seed survives when every assumption
that can be wrong is wrong at once.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from physics.constants import CELSIUS_TO_KELVIN, KB_EV, T_REF_MS
from physics.thermal_model import FLASH, FLASH_T, FLASH_V, T_HI, V_HI, V_LO

_SD_GRID = None


def _interp_sd():
    """Cached interpolator for the GP's posterior sd -- how wrong interpolation can be, in deg C.

    Built once: the GP is fitted to the same 30 nodes the design inverts, so its sd collapses at a
    measured point and grows between them. That field is exactly the licence a world has to differ
    from the table.
    """
    global _SD_GRID
    if _SD_GRID is None:
        from scipy.interpolate import RegularGridInterpolator

        from physics.thermal_gp import build

        gp = build()
        # The grid MUST contain the measured node coordinates, or the interpolated sd never sees
        # the collapse to ~0.14 C at a node and worlds drift away from Cristian's measurements.
        vg = np.unique(np.concatenate([np.linspace(V_LO, V_HI, 60), FLASH_V]))
        tg = np.unique(np.concatenate([np.geomspace(0.1, T_HI, 60), FLASH_T]))
        vv, tt = np.meshgrid(vg, tg, indexing="ij")
        sd = gp.predict(vv.ravel(), tt.ravel())[1].reshape(vv.shape)
        _SD_GRID = RegularGridInterpolator(
            (vg, np.log10(tg)), sd, bounds_error=False, fill_value=None
        )
    return _SD_GRID

# --- how wrong each assumption is allowed to be -------------------------------------------------
# The measured table is NOT one of them. Its 30 nodes are a direct measurement and are held exact;
# only interpolation between them is uncertain, and the GP posterior sd sets how uncertain. This
# multiplier says how many of those standard deviations a world may wander by.
INTERP_SIGMAS = 2.0
INTERP_MODES = 3  # smooth random field: this many Fourier modes per axis
DISTORT_CYCLES = 2.0  # how many ripples across the box

TRANSITION_LO_C, TRANSITION_HI_C = 360.0, 620.0  # where the transition may actually sit, in TRUTH
WIDTH_LO_C, WIDTH_HI_C = 6.0, 90.0  # 10-90% width
TILT_LO_K, TILT_HI_K = 0.0, 60.0  # K per e-fold of dwell
CURVATURE_MAX_C = 20.0  # per (log dwell)^2
# OVER-ANNEAL. Two-stage kinetics a -> o -> m: the film crystallizes, then the equilibrium
# monoclinic phase takes over and the ferroelectric response dies. The measured map is then a
# CLOSED BAND with TWO boundaries, not a single onset -- a topology that breaks assumptions no
# single-boundary design stresses. It is not hypothetical here: the campaign's readout is
# ferroelectric-sensitive, and the repo's own RTA series is non-monotone in thermal severity,
# with 2Pr peaking at 500 C and falling to 21.5 by 700 C. Locating the process WINDOW, rather
# than an onset, may be the real problem.
OVERANNEAL_PROB = 0.35
OVERANNEAL_ABOVE_LO_C, OVERANNEAL_ABOVE_HI_C = 40.0, 160.0  # how far above the first transition
OVERANNEAL_WIDTH_RATIO = (0.4, 1.5)  # second stage relative width; a higher barrier is sharper
BRANCH_PROB = 0.25  # a second cold branch at long dwell
VOLT_CHANNEL_MAX = 0.6  # non-thermal response per 100 V

NOISE_SCALE_LO, NOISE_SCALE_HI = 0.5, 3.0
HEAVY_TAIL_PROB = 0.06  # fraction of readings that are wild
HEAVY_TAIL_MULT = 6.0
SPECIMEN_SD_MAX = 0.12  # per-specimen offset, on the readout scale

READOUT_FLOOR_MAX = 0.4  # unknown additive floor
READOUT_SPAN_LO, READOUT_SPAN_HI = 0.5, 1.6
READOUT_SATURATE_PROB = 0.35

_BASE_NOISE_FLOOR, _BASE_NOISE_BOUNDARY = 0.02, 0.30


@dataclass
class World:
    """One sampled set of wrong assumptions.

    :param true_tmax: the peak temperature the FILM sees, which equals the measured table at its
        30 nodes and may differ between them.
    :param truth: TRUE crystalline fraction at ``(V, t)``.
    :param observe: turns a true fraction into what the instrument would report.
    :param label: short description, for reporting which worlds hurt.
    """

    truth: Callable[[np.ndarray, np.ndarray], np.ndarray]
    observe: Callable[[np.ndarray, np.random.Generator], np.ndarray]
    label: str
    true_tmax: Callable[[np.ndarray, np.ndarray], np.ndarray] = None
    # the true dwell coefficient, K per e-fold -- the quantity block A exists to measure
    tilt_k: float = 0.0


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def sample_world(rng: np.random.Generator) -> World:
    """Draw one world in which the thermal model, the response, the noise and the readout may all
    be wrong at once.

    :param rng: random generator.
    """
    # --- the thermal truth: the measured nodes are exact, only interpolation may be wrong ---
    sd_of = _interp_sd()
    # L1-normalized so the summed field cannot exceed 1 in magnitude, which keeps the deviation
    # bounded by INTERP_SIGMAS standard deviations rather than sqrt(modes) times that.
    coef = rng.normal(0.0, 1.0, (INTERP_MODES, INTERP_MODES))
    coef /= np.sum(np.abs(coef)) or 1.0
    ph = rng.uniform(0, 2 * np.pi, (INTERP_MODES, INTERP_MODES, 2))

    def true_tmax(v, t):
        v = np.asarray(v, float)
        t = np.asarray(t, float)
        u = (v - V_LO) / (V_HI - V_LO)
        w = (np.log10(t) - np.log10(0.1)) / (np.log10(T_HI) - np.log10(0.1))
        field = np.zeros_like(u, dtype=float)
        for i in range(INTERP_MODES):
            for j in range(INTERP_MODES):
                field += coef[i, j] * np.sin(
                    (i + 1) * DISTORT_CYCLES * np.pi * u + ph[i, j, 0]
                ) * np.sin((j + 1) * np.pi * w + ph[i, j, 1])
        sd = sd_of(np.column_stack([v.ravel(), np.log10(t.ravel())])).reshape(v.shape)
        return FLASH.tmax(v, t) + INTERP_SIGMAS * sd * field

    # --- the response ---
    t0 = float(rng.uniform(TRANSITION_LO_C, TRANSITION_HI_C))
    width = float(rng.uniform(WIDTH_LO_C, WIDTH_HI_C))
    sharp = 2.0 * np.log(9.0) / width
    tilt = float(rng.uniform(TILT_LO_K, TILT_HI_K))
    curv = float(rng.uniform(-CURVATURE_MAX_C, CURVATURE_MAX_C))
    volt = float(rng.uniform(-VOLT_CHANNEL_MAX, VOLT_CHANNEL_MAX))
    gompertz = bool(rng.random() < 0.5)
    over_t0 = (
        t0 + float(rng.uniform(OVERANNEAL_ABOVE_LO_C, OVERANNEAL_ABOVE_HI_C))
        if rng.random() < OVERANNEAL_PROB
        else None
    )
    over_sharp = sharp / float(rng.uniform(*OVERANNEAL_WIDTH_RATIO))
    has_branch = rng.random() < BRANCH_PROB
    branch_t = float(rng.uniform(TRANSITION_LO_C - 90.0, t0 - 30.0)) if has_branch else None

    def truth(v, t):
        v = np.asarray(v, float)
        t = np.asarray(t, float)
        tm = true_tmax(v, t)
        u = np.log(t / T_REF_MS)
        z = tm - t0 + tilt * u + curv * u**2 + volt * (v - 611.0) / 100.0 * 40.0
        if gompertz:
            # asymmetric, as activated kinetics actually give
            theta = KB_EV * (t0 + CELSIUS_TO_KELVIN) ** 2 / 2.5
            x = 1.0 - np.exp(-np.log(2.0) * np.exp(np.clip(2.5 * z / theta, -50, 50)))
        else:
            x = _sigmoid(sharp * z)
        if over_t0 is not None:
            # X = X_cr * (1 - X_m): crystallized AND not yet converted. The hot corner dies, so
            # the X = 1/2 contour has TWO branches and the latent field two zero crossings.
            x = x * (1.0 - _sigmoid(over_sharp * (tm - over_t0 + tilt * u)))
        if branch_t is not None:
            gate = _sigmoid(2.0 * (t - 7.0))
            x = np.maximum(x, _sigmoid(sharp * (tm - branch_t)) * gate)
        return np.clip(x, 0.0, 1.0)

    # --- how the instrument mangles it ---
    nscale = float(rng.uniform(NOISE_SCALE_LO, NOISE_SCALE_HI))
    spec_sd = float(rng.uniform(0.0, SPECIMEN_SD_MAX))
    floor = float(rng.uniform(0.0, READOUT_FLOOR_MAX))
    span = float(rng.uniform(READOUT_SPAN_LO, READOUT_SPAN_HI))
    saturate = float(rng.uniform(0.5, 0.9)) if rng.random() < READOUT_SATURATE_PROB else None

    def observe(x, gen):
        x = np.asarray(x, float)
        y = x if saturate is None else np.minimum(x, saturate)
        sd = nscale * (_BASE_NOISE_FLOOR + _BASE_NOISE_BOUNDARY * y * (1.0 - y))
        obs = floor + span * y
        obs = obs + gen.normal(0.0, sd) + gen.normal(0.0, spec_sd, size=np.shape(y))
        wild = gen.random(np.shape(y)) < HEAVY_TAIL_PROB
        obs = np.where(wild, obs + gen.normal(0.0, HEAVY_TAIL_MULT * sd), obs)
        return obs

    label = (
        f"T0={t0:.0f} w{width:.0f} tilt{tilt:.0f}"
        f"{' gomp' if gompertz else ''}{' BAND' if over_t0 is not None else ''}"
        f"{' branch' if branch_t is not None else ''}{' volt' if abs(volt) > 0.25 else ''}"
        f" n{nscale:.1f} spec{spec_sd:.02f}"
    )
    return World(
        truth=truth, observe=observe, label=label, true_tmax=true_tmax, tilt_k=tilt
    )
