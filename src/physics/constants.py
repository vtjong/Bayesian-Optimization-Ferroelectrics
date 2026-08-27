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
CELSIUS_TO_KELVIN = 273.15
T_ROOM_C = 25.0  # ambient the film returns to between shots

# --- PRIOR ---------------------------------------------------------------------------------
# [MEASURED, campaign tool] WHERE THE PROXY CHANGES REGIME -- not a crystallization onset.
#
# What was actually observed, in Bolometer_readings_PulseForge.xlsx, is a pair of adjacent voltage
# settings at t = 5.0 ms:
#
#     650 V  ->  low  ferroelectric-switching response
#     670 V  ->  high ferroelectric-switching response
#
# Inverting the campaign's own peak-temperature table puts those two conditions at 434.7 C and
# 458.7 C. So the defensible claim is a BRACKET on where the electrical response transitions, and
# that bracket -- not its midpoint -- is what the design is entitled to rely on.
#
# THIS IS NOT A MEASURED CRYSTALLIZATION ONSET. The readout is 2*Qsw/(U+|D|), a ferroelectric
# switching figure of merit. It is phase-sensitive, it is not calibrated against total crystalline
# fraction, and the repo's own RTA series shows a comparable readout that is NON-MONOTONE in
# thermal severity. Calling 447 C "the onset" would assert more than the data supports, in exactly
# the way the earlier 380 C figure did.
#
# T_TRANSITION_REF_C is the bracket midpoint, used ONLY as a reference coordinate for placing seed
# conditions and for anchoring the kinetic ensemble. The design is built to be robust to its being
# wrong: the ladder levels are chosen by MINIMAX over the whole prior rather than optimised at the
# midpoint, and because the bracket is read through the same table the design inverts, a rescaling
# of the temperature axis leaves every commanded voltage unchanged.
T_TRANSITION_LO_C = 434.7  # hottest specimen still in the low-response regime
T_TRANSITION_HI_C = 458.7  # coldest specimen in the high-response regime
# Rounded to a whole degree deliberately. This is a design COORDINATE, not a measurement, and the
# bracket endpoints are themselves spline inversions of a 5x6 table -- carrying 446.7 would assert
# a tenth of a degree of precision that nothing here supports.
T_TRANSITION_REF_C = round(0.5 * (T_TRANSITION_LO_C + T_TRANSITION_HI_C))  # 447 C
#
# Wider than the 12 C half-bracket, because the bracket is not the only uncertainty: n = 6, all at
# one flash time; a single voltage step brackets it, so 24 C is a SAMPLING RESOLUTION rather than a
# measured width; the readout is a switching figure of merit rather than crystallinity; and
# KHM009/KHM010 may not be the stack about to be run.
T_TRANSITION_SIGMA_C = 25.0
T_REF_MS = 5.1  # flash time at which the bracket was observed (a measured table row)

# Activated-kinetics parameters. Together they SET the transition width, so they must be stated
# rather than absorbed into a hardcoded sharpness.
#
# The same six campaign-tool samples that fix the transition bracket bound the product n*Ea from
# below, because
# they fix the onset and the width on one temperature scale at once. The readout climbs from 0.044
# to 1.447 between 434.7 C and 458.7 C, so the 10-90% transition fits inside 24 C, giving
# s > 0.183 /C and hence
#   n*Ea  =  s * kB * Tc^2 / (2 ln2)  >  5.90 eV      (Tc = 447 C)
# which at n = 2.5 is Ea > 2.36 eV. Three independent off-tool lines agree on the magnitude:
#   [ARCHIVAL] graded 2Pr fit of the KHM005/6 flash shots  -> Ea ~ 2.0 eV  (run_tilt_prior.py)
#   [ARCHIVAL] JMAK refit of the FE_HZCO 350 C RTA isotherm -> Ea ~ 2.0-2.2 eV, n ~ 1.8
#   literature, in-situ XRD crystallization of HfO2        -> Ea ~ 2.6 +/- 0.5 eV
# 2.5 eV clears the on-tool bound at n = 2.5 (n*Ea = 6.25 eV) and reproduces a 23 C model width
# against the 24 C measured bracket. That agreement is weaker than it looks: 24 C is the spacing
# between the 650 V and 670 V settings that bracket the onset, so it is a SAMPLING RESOLUTION, and
# the same six points are equally consistent with a much sharper transition. The bound is a floor
# on n*Ea, not a measurement of it.
#
# The literature figure is one study -- isothermal in-situ XRD on ion-beam-sputtered UNDOPED HfO2 at
# 15-50 nm and 550-650 C -- not a spread across studies, so applying it to 10 nm ALD HZO between
# TiN at millisecond timescales is a substantial extrapolation.
#
# The bound holds only if the readout tracks crystalline fraction through the transition. A figure
# of merit that saturates before the film does would make the transition look sharper than it is,
# so 24 C is an upper bound on the width for the PROXY and Ea inherits that. It is a floor, not an
# estimate.
#
# The tilt scales as 1/Ea, so no measured tilt should be quoted without stating the Ea it assumes.
EA_EV = 2.5  # activation energy (eV); on-tool bound Ea > 2.36 at n = 2.5, literature 2.6 +/- 0.5
# Avrami exponent. In JMAK n = d/m + a (m = 1 interface-controlled, 2 diffusion-controlled;
# a = 0 site-saturated, 1 continuous nucleation). The empirical support for 2.5 is the repo's own
# 350 C RTA isotherm, which gives n = 2.2-2.6, and the in-situ XRD literature, which reports n ~ 2
# for HfO2. Note that no clean mechanism gives exactly 2.5 in a 10 nm film -- d/m + a = 2.5 would
# be 3-D diffusion-controlled growth with continuous nucleation, and HfO2 crystallization is
# polymorphic and interface-controlled. Treat 2.5 as an empirical fit, not a mechanism.
#
# n MATTERS FOR THE Ea FLOOR: the bound is on the PRODUCT n*Ea > 5.90 eV, so EA_EV = 2.5 clears it
# only for n >= 2.36. At n = 2.0 the floor is Ea > 2.95 eV, above the literature centre.
AVRAMI_N = 2.5

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
