"""Named constants for the crystallization-boundary campaign.

Every number the models depend on lives here, grouped by what kind of claim it makes. The grouping
is the point: a physical constant, a measured quantity, a campaign prior, and a numerical tuning
knob carry very different authority, and code that mixes them makes it impossible to tell which
results would move if a number were wrong.

  PHYSICAL   exact or standard; never fitted.
  PRIOR      campaign assumptions with real uncertainty. Anything derived from these inherits it.

PROVENANCE AND VINTAGE. Numbers derived from the ARCHIVAL sample set (KHM005/KHM006 flash shots,
the FE_HZCO RTA series, the PulseForge bolometer runs) are marked [ARCHIVAL]. Those films are not
1:1 with the current ones, so only SOME quantities carry over:

  * transferable  -- the transition width and the DIMENSIONLESS tilt ratio theta/T50 = kB*T50/Ea.
                     These describe the crystallization mechanism, not the batch, and theta/T50 is
                     additionally invariant to any constant rescaling of the temperature axis.
  * NOT transferable -- the onset temperature, the readout floor, and the noise model. Onset is set
                     by seeding density, interface chemistry and thickness; a vacuum break alone is
                     documented to move it ~200 C on nominally identical stacks.

An [ARCHIVAL] number is a PRIOR. Never treat one as a calibration of the current films.
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
# [ARCHIVAL] and NOT transferable -- see the provenance note above. Widened from 30 to 70 C on
# four independent grounds: the two source fits disagree by 31 C; they were taken in a different
# voltage box in different units; the readout is a ferroelectric switch-on threshold rather than
# crystallinity; and the current films are a different sample set, where interface chemistry alone
# is documented to shift the onset by ~200 C. A 30 C sigma was what made the earlier single-level
# ladder fail: it slid into saturation and reported "no tilt" when the tilt was large.
T_ONSET_C = 380.0  # onset peak temperature (deg C) -- a prior centre, NOT a measurement
T_ONSET_SIGMA_C = 70.0  # honest spread; see above
T_REF_MS = 5.1  # flash time at which the onset is quoted (a measured table row)

# Activated-kinetics parameters. Ea ~ 1 eV is the growth-limited regime appropriate to
# nanocrystallite-seeded films (pre-existing nuclei, no nucleation barrier); n between 2 and 3 is
# 2-3D growth from those seeds. Together they SET the transition width, so they must be stated
# rather than absorbed into a hardcoded sharpness.
# UNRESOLVED -- flagged by review, left at the historical values pending the temperature-axis
# decision, because Ea and the temperature scale are confounded. Evidence against 1.0 eV:
#   [ARCHIVAL] graded 2Pr fit of the KHM005/6 flash shots  -> Ea ~ 2.0 eV  (run_tilt_prior.py)
#   [ARCHIVAL] iso-conversion on the FE_HZCO RTA series    -> Ea ~ 1.5 eV
#   literature, in-situ XRD crystallization of HfO2        -> Ea ~ 2.6 +/- 0.5 eV
# The tilt scales as 1/Ea, so Ea alone spans a wider range than the whole pulse-shape ensemble:
# raising it from 1.0 to 2.0 eV roughly halves every predicted tilt. Do not quote a measured tilt
# without stating the Ea it assumes.
EA_EV = 1.0  # activation energy (eV) -- LOW vs all available evidence; see above
AVRAMI_N = 2.5  # Avrami exponent -- site saturation implies an integer, and 3D growth is
# geometrically impossible in a 10 nm film with comparable lateral grain size; evidence favours 1-2

# --- MEASURED ------------------------------------------------------------------------------
# Lamp irradiance envelope q(s) = LAMP_A*exp(-s/LAMP_TAU_FAST) + (1-LAMP_A)*exp(-s/LAMP_TAU_SLOW),
# truncated at the commanded pulse width. Obtained by jointly inverting two independent integral
# measurements we already own: the delivered fluence E(V,t) in
# data/Bolometer_readings_PulseForge.xlsx, and the peak-temperature table (which measures
# INT q(s)/sqrt(t-s) ds). Reproduce with: python src/run_lamp_check.py
#
# These numbers matter more than any other in the model. The lamp does NOT deliver fixed energy
# per shot and its irradiance is NOT a top hat: fluence rises sublinearly with pulse width, an
# intrinsic ~2 ms timescale sitting inside the 2.6-10.1 ms design range. That timescale breaks the
# self-similarity of pure conduction and reduces the predicted boundary tilt relative to a top-hat
# drive: 49.8 C for the square-drive member against 37.6 C here, as this implementation computes
# them. An independent inversion of the same fluence data gives ~12 C; that disagreement is
# unresolved and is itself a reason not to treat any single tilt figure as settled.
LAMP_A = 0.88
LAMP_TAU_FAST_MS = 2.06
LAMP_TAU_SLOW_MS = 54.6
LAMP_QUAD_NODES = 600  # nodes for the Duhamel integral after the sqrt substitution

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
V_SCAN_POINTS = 1024  # dense scan locating the spline's true max in V (it is not monotone there)
BISECT_TMAX_LO_C = 100.0  # bracket for inverting a fraction to a peak temperature
BISECT_TMAX_HI_C = 1200.0
