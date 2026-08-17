"""Named constants for the crystallization-boundary campaign.

Every number the models depend on lives here, grouped by what kind of claim it makes. The grouping
is the point: a physical constant, a measured quantity, a campaign prior, and a numerical tuning
knob carry very different authority, and code that mixes them makes it impossible to tell which
results would move if a number were wrong.

  PHYSICAL   exact or standard; never fitted.
  PRIOR      campaign assumptions with real uncertainty. Anything derived from these inherits it.
  SHAPE      parameters of the candidate cooling laws. Provenance is recorded per group.
  NUMERICAL  quadrature and root-find settings. Chosen for convergence; results must not depend
             on them (``run_model_checks.py`` verifies this).
  READOUT    the measured noise model of the in-loop permittivity readout.

Measured data is NOT here: the peak-temperature table is read from ``data/flash_temp_table.csv``
so that file stays the single source of truth.
"""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FLASH_TABLE_CSV = DATA_DIR / "flash_temp_table.csv"
MEASURED_TRACE_CSV = DATA_DIR / "measured_transient.csv"  # not yet available; see run_thermal_check

# --- PHYSICAL ------------------------------------------------------------------------------
KB_EV = 8.617333e-5  # Boltzmann constant (eV/K)
CELSIUS_TO_KELVIN = 273.15
T_ROOM_C = 25.0  # ambient the film returns to between shots

# --- PRIOR ---------------------------------------------------------------------------------
# The onset is NOT a measured number for this stack. It comes from logistic fits of 2Pr vs
# temperature on two datasets that disagree: flash T50 = 388 C, RTA T50 = 357 C, fitted in a
# different voltage box and in different units, using a ferroelectric switch-on indicator rather
# than crystallinity. 380 was chosen as a round value near the flash figure. Treat the width below
# as the honest uncertainty and never quote T_ONSET_C as "the measured onset".
T_ONSET_C = 380.0  # onset peak temperature (deg C) -- a prior centre
T_ONSET_SIGMA_C = 30.0  # spread bracketing the two fits and the box mismatch
T_REF_MS = 5.1  # flash time at which the onset is quoted (a measured table row)

# Activated-kinetics parameters. Ea ~ 1 eV is the growth-limited regime appropriate to
# nanocrystallite-seeded films (pre-existing nuclei, no nucleation barrier); n between 2 and 3 is
# 2-3D growth from those seeds. Together they SET the transition width, so they must be stated
# rather than absorbed into a hardcoded sharpness.
EA_EV = 1.0  # activation energy (eV)
AVRAMI_N = 2.5  # Avrami exponent

# --- SHAPE ---------------------------------------------------------------------------------
TRACE_DURATION_MS = 320.0  # window over which the Arrhenius budget is integrated

# Legacy width-independent shape. Retained only to reproduce the earlier iso-Tmax campaign; the
# plateau is a fraction of Tmax that never decays, which is a wrong boundary condition.
LEGACY_PLATEAU_FRAC = 0.15
LEGACY_TAU_DECAY_MS = 35.0
LEGACY_RISE_MS = 2.0
LEGACY_DURATION_MS = 250.0

# Empirical two-exponential parameterization of the normalized transient. These four numbers are
# EYEBALL FITS to a normalized figure, not measurements, and they dominate the boundary tilt --
# RAMP_A_FAST / RAMP_TAU_FAST alone contributes ~12 ms of effective dwell.
RAMP_PLATEAU_FRAC = 0.17
RAMP_A_FAST = 0.60
RAMP_TAU_FAST_MS = 8.0
RAMP_A_SLOW = 0.23
RAMP_TAU_SLOW_MS = 40.0

# --- READOUT -------------------------------------------------------------------------------
# Permittivity proxy, calibrated against archival PUND data. Noise is heteroscedastic and largest
# mid-transition, where the film is a phase mixture: sigma_n(f) = FLOOR + BOUNDARY * f * (1 - f),
# peaking at ~0.095 at f = 1/2.
NOISE_FLOOR = 0.02
NOISE_BOUNDARY = 0.30

# --- NUMERICAL -----------------------------------------------------------------------------
# The Arrhenius integrand is concentrated within kB*T^2/Ea of the peak -- a fraction of a
# millisecond for the diffusion shape -- so the quadrature grid is dense there and sparse in the
# tail. Sizes are set for convergence, not for speed.
QUAD_NEAR_POINTS = 4000  # samples across the near-peak window
QUAD_FAR_POINTS = 1200  # log-spaced samples across the decaying tail
QUAD_NEAR_WINDOW_PULSES = 3.0  # near-peak window, in multiples of the pulse width
QUAD_TAU_MIN_MS = 1e-3  # lower bound of the log-spaced tail grid

BISECT_ITERS = 80  # bisection steps; 80 halvings is exact to machine precision here
BISECT_TMAX_LO_C = 100.0  # bracket for inverting a fraction to a peak temperature
BISECT_TMAX_HI_C = 1200.0
