"""Minimal-viable transient thermal model for flash-lamp annealing: (V, t) -> T(t).

Lumped-capacitance film with conductive loss to the substrate. The absorbed lamp power
scales with V^power and is delivered for the pulse duration; the film heats toward a
steady-state during the pulse and cools (quenches) afterward with time constant tau.
Heating is driven by the lamp, cooling only by conduction — so the heating and cooling
rates differ (an asymmetric heat/quench profile), which is the lever for crystallization
kinetics.

PROTOTYPE: lumped + analytic. Upgrade to a 1-D transient
through-thickness stack (film + finite quartz substrate) and VALIDATE against measured
bolometer/pyrometer traces before any quantitative mechanism claim.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThermalParams:
    """Lumped thermal-model parameters (rough, configurable; not yet calibrated)."""

    t_amb: float = 25.0  # ambient temperature (deg C)
    gain: float = 1000.0  # steady-state rise at V=1 (deg C) = absorbed_power / loss_coeff
    tau_ms: float = 2.0  # heating/cooling time constant (ms)
    v_power: float = 2.0  # lamp power ~ V^v_power


def simulate_profile(
    v_norm, t_pulse_ms, t_preheat=None, params=ThermalParams(), n=600, tail_factor=6.0
):
    """Return (t_grid_ms, T_celsius) for a normalized voltage v_norm in [0, 1].

    :param v_norm: Normalized flash voltage in [0, 1]
    :param t_pulse_ms: Pulse duration (ms)
    :param t_preheat: Substrate preheat temperature (deg C); the film starts and cools back
        to this baseline. Defaults to ambient. Varying it DECORRELATES the descriptors
        (a given Tmax can be reached with less flash power), breaking single-pulse collinearity.
    :param params: :class:`ThermalParams`
    :param n: Number of time samples
    :param tail_factor: Cooling tail length, in units of tau
    :return: (t_ms, T_C) arrays
    """
    base = params.t_amb if t_preheat is None else float(t_preheat)
    tau = params.tau_ms
    rise = params.gain * (float(v_norm) ** params.v_power)  # flash-driven rise above base
    t = np.linspace(0.0, t_pulse_ms + tail_factor * tau, n)

    heating = t <= t_pulse_ms
    T = np.empty_like(t)
    # heating phase: approach the baseline + flash steady-state
    T[heating] = base + rise * (1.0 - np.exp(-t[heating] / tau))
    # peak at the end of the pulse, then exponential cooling (quench) back to the baseline
    peak_rise = rise * (1.0 - np.exp(-t_pulse_ms / tau))
    t_cool = t[~heating] - t_pulse_ms
    T[~heating] = base + peak_rise * np.exp(-t_cool / tau)
    return t, T


@dataclass(frozen=True)
class ThermalParams1D:
    """Physical params for the 1-D through-thickness model (fused-silica substrate)."""

    k: float = 1.4            # substrate thermal conductivity (W/m/K)
    rho: float = 2200.0       # density (kg/m^3)
    cp: float = 740.0         # specific heat (J/kg/K)
    absorptance: float = 0.3  # film optical absorptance (measurable via under-sample bolometer)
    energy_max: float = 15.0  # lamp energy density at V=1 (J/cm^2)
    v_power: float = 2.0      # energy density ~ V^v_power
    depth_um: float = 150.0   # simulated substrate depth (um); a few diffusion lengths
    nz: int = 160             # spatial nodes
    t_amb: float = 25.0       # ambient / cooling floor (deg C)


def simulate_profile_1d(v_norm, t_pulse_ms, t_preheat=None, params=ThermalParams1D(),
                        nt=2000, tail_factor=6.0):
    """1-D transient through-thickness model: surface (film) temperature T(t).

    Solves rho*cp dT/dt = k d2T/dz2 in the substrate with an absorbed-flux boundary at the
    top (the film) during the pulse and an insulated deep boundary; backward-Euler implicit
    (unconditionally stable), tridiagonal solve per step. For flash annealing heating is
    global, so 1-D vertical conduction is the justified geometry. Returns (t_ms, T_surface_C).
    """
    from scipy.linalg import solve_banded

    base = params.t_amb if t_preheat is None else float(t_preheat)
    diff = params.k / (params.rho * params.cp)            # thermal diffusivity (m^2/s)
    dz = (params.depth_um * 1e-6) / (params.nz - 1)
    t_total = (t_pulse_ms * 1e-3) * (1.0 + tail_factor)   # s
    dt = t_total / nt
    r = diff * dt / dz ** 2

    energy = params.energy_max * (float(v_norm) ** params.v_power)   # J/cm^2
    flux_on = params.absorptance * (energy * 1e4) / (t_pulse_ms * 1e-3)  # W/m^2 during pulse

    # constant tridiagonal (backward Euler); Neumann flux at z=0, insulated at z=L
    nz = params.nz
    ab = np.zeros((3, nz))
    ab[0, 1:] = -r
    ab[0, 1] = -2 * r          # surface super-diagonal
    ab[1, :] = 1.0 + 2 * r
    ab[2, :-1] = -r
    ab[2, nz - 2] = -2 * r     # insulated bottom sub-diagonal

    T = np.full(nz, base)
    times = np.empty(nt + 1)
    surf = np.empty(nt + 1)
    times[0], surf[0] = 0.0, base
    for n in range(nt):
        t_cur = (n + 1) * dt
        flux = flux_on if t_cur <= t_pulse_ms * 1e-3 else 0.0
        rhs = T.copy()
        rhs[0] += 2 * r * dz * flux / params.k          # absorbed surface flux (in K)
        T = solve_banded((1, 1), ab, rhs)
        times[n + 1] = t_cur * 1e3                       # ms
        surf[n + 1] = T[0]
    return times, surf
