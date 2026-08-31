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
so that file stays the single source of truth, and the paths to it live in ``paths.py``.
"""


# --- PHYSICAL ------------------------------------------------------------------------------
KB_EV = 8.617333e-5  # Boltzmann constant (eV/K)
T_ROOM_C = 25.0  # ambient the film returns to between shots

# --- PRIOR ---------------------------------------------------------------------------------
# There is deliberately no transition-temperature prior here. The previous one, a 435-459 C bracket
# with a 447 C midpoint, does not survive its own provenance: it came from six shots on the earlier
# Sinteron at 2-3 kV rather than this tool, it was labelled by 2*Qsw, which measures ferroelectric
# switching rather than crystallinity, and the temperatures assigned to those shots were computed by
# the same CERA table the bracket was then used to vouch for. The seed fired against it returned ten
# crystallized specimens at conditions the bracket put well below onset.
#
# What replaced it is a measurement rather than a prior: the lowest delivered fluence among the
# pilot conditions observed to crystallize. That number lives with the pilot data, not here, because
# it is an observation and not an assumption.

# --- MEASURED ------------------------------------------------------------------------------
# Lamp irradiance envelope q(s) = LAMP_A*exp(-s/LAMP_TAU_FAST) + (1-LAMP_A)*exp(-s/LAMP_TAU_SLOW),
# truncated at the commanded pulse width. Obtained by jointly inverting two independent integral
# measurements we already own: the delivered fluence E(V,t) in
# data/Bolometer_readings_PulseForge.xlsx, and the peak-temperature table (which measures
# INT q(s)/sqrt(t-s) ds). The script that fitted it has been retired: it compared the table
# against constant candidate scale factors, and the live disagreement between the two thermal
# simulations of this tool is not constant -- it grows with pulse width and reverses in sign.
#
# These numbers matter more than any other in the model. The lamp does NOT deliver fixed energy
# per shot and its irradiance is NOT a top hat: fluence rises sublinearly with pulse width, an
# intrinsic ~2 ms timescale sitting inside the 2.6-10.1 ms design range. That timescale breaks the
# self-similarity of pure conduction and reduces the predicted boundary tilt relative to a top-hat
# drive. For the tilts this implementation actually produces, see the ensemble table printed by
# run_flash_plan.py -- they depend on EA_EV and on the dwell range quoted, so no single figure
# belongs in a comment.
#
# TWO LIMITS ON THESE NUMBERS. The bolometer rows they invert span V in [620, 799] V and
# t in [0.5, 5.0] ms, while the design box is V in [506, 716] and t in [0.1, 10.1] -- so the
# saturation is EXTRAPOLATED at exactly the long widths where it matters most. And 78% of the
# envelope's asymptotic energy sits in the slow component, entirely outside both the fluence data
# and the design range, so LAMP_TAU_SLOW_MS is unconstrained by either.
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
# There is deliberately no noise model here. The previous one (sigma_n = 0.02 + 0.30*f*(1-f)) was
# calibrated against archival PUND data on different films, which this file's own provenance notes
# list as NOT transferable, and it was written in crystalline-fraction units while the instrument
# reports permittivity with an uncalibrated zero and gain. A replacement must be calibrated on
# these films and in the units they are actually read in.

# --- NUMERICAL -----------------------------------------------------------------------------
# The Arrhenius integrand is concentrated within kB*T^2/Ea of the peak -- a fraction of a
# millisecond for the diffusion shape -- so the quadrature grid is dense there and sparse in the
# tail. Sizes are set for convergence, not for speed.

BISECT_ITERS = 80  # bisection steps; 80 halvings is exact to machine precision here
V_SCAN_POINTS = 1024  # dense scan locating the spline's true max in V (it is not monotone there)
