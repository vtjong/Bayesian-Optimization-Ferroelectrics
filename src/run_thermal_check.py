"""Validate the thermal forward model T(V, t) before it feeds the crystallization campaign.

The model factorizes as ``T(tau; V, t) = T_room + (Tmax(V, t) - T_room) * g(tau; t)``. Only the
first factor is measured. Tmax comes from a 5x6 table of measured peak temperatures; the SHAPE
g(tau; t) is asserted, and the assertion is load-bearing, because activated kinetics integrate the
trace. A shape that ignores the commanded pulse width forces the crystallization boundary to be a
plain Tmax level set; one that tracks it does not. This script checks what can be checked and says
plainly what cannot.

Checks printed:
  * node-exactness  -- Tmax at the 30 table nodes must equal the tabulated value;
  * interpolation   -- cubic-spline overshoot over the box vs the tabulated maximum;
  * table support   -- flags the time interval where the table has no nodes at all;
  * shape collapse  -- spread of normalized traces across flash times, PER SHAPE. A residual of
                       zero does not mean the tool has a universal transient; it means the shape
                       is incapable of representing pulse-width dependence, so this test cannot
                       discriminate. Reported as INCAPABLE, not as a pass.
  * effective dwell -- the quantity the boundary actually depends on, and the tilt it implies;
  * measured trace  -- SKIPPED, loudly, until a digitized T(tau) exists to compare against. This
                       is the one measurement that would settle which shape is right.

Usage:  python src/run_thermal_check.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from discovery.kinetics import EA_EV, build_ensemble, theta_kelvin, trace_times
from discovery.synthetic import (
    FLASH,
    FLASH_T,
    FLASH_TMAX,
    FLASH_V,
    KB_EV,
    SHAPES,
    T_HI,
    T_LO,
    T_ROOM,
    T_TRANSITION_REF_C,
    V_HI,
    V_LO,
    tmax,
)
from visualization.base import save_figure

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "predictions" / "thermal_check"
CUT_V = 674.0  # voltage column to show transients for (matches the measured grid)
MEASURED_TRACE = ROOT / "data" / "measured_transient.csv"  # not yet available
_C2K = 273.15


def effective_dwell(shape, tmax_c: float, t: float, ea_ev: float = EA_EV) -> float:
    """Arrhenius-weighted dwell: ``INT exp(-Ea/kB T(tau)) dtau / exp(-Ea/kB Tmax)`` in ms.

    This is the single number through which the trace shape reaches the crystallization boundary:
    the budget is ``Phi = t_eff * exp(-Ea / kB Tmax)``, so two shapes with the same t_eff(t) give
    the same boundary no matter how different they look.

    :param shape: normalized pulse shape.
    :param tmax_c: peak temperature (deg C).
    :param t: flash time / pulse width (ms).
    :param ea_ev: activation energy (eV).
    """
    tau = trace_times(t, shape.duration_ms)
    t_kelvin = T_ROOM + (tmax_c - T_ROOM) * shape(tau, t) + _C2K
    integral = np.trapezoid(np.exp(-ea_ev / (KB_EV * t_kelvin)), tau)
    return float(integral / np.exp(-ea_ev / (KB_EV * (tmax_c + _C2K))))


def _collapse_residual(shape) -> float:
    """Spread of normalized traces across flash times; 0 means the shape ignores pulse width."""
    shapes = []
    for t in FLASH_T:
        tau = np.linspace(0.0, shape.duration_ms, 4000)
        shapes.append(shape(tau, float(t)))
    return float(np.max(np.std(np.array(shapes), axis=0)))


def _validate() -> dict:
    """Node-exactness, interpolation overshoot, and the per-shape collapse residual."""
    node_err = 0.0
    for j, t in enumerate(FLASH_T):
        for i, v in enumerate(FLASH_V):
            node_err = max(node_err, abs(float(tmax(v, t)) - FLASH_TMAX[j, i]))
    vg = np.linspace(V_LO, V_HI, 300)
    tg = np.linspace(T_LO, T_HI, 300)
    vv, tt = np.meshgrid(vg, tg)
    return {
        "node_err": node_err,
        "box_max": float(tmax(vv, tt).max()),
        "table_max": float(FLASH_TMAX.max()),
        "collapse": {k: _collapse_residual(s) for k, s in SHAPES.items()},
    }


def _report(v: dict) -> None:
    """Print every check, including the ones that cannot currently be run."""
    print("=== thermal forward-model checks ===\n")
    print(f"  node-exactness   : max|spline - table| over the 30 nodes = {v['node_err']:.3g} C")
    print(
        f"  interpolation    : box max {v['box_max']:.1f} C vs table max {v['table_max']:.1f} C "
        f"(overshoot {v['box_max'] - v['table_max']:+.1f} C)"
    )

    gap_lo, gap_hi = float(FLASH_T[0]), float(FLASH_T[1])
    rise = FLASH_TMAX[1].max() - FLASH_TMAX[0].max()
    print(
        f"  table support    : NO nodes in t = ({gap_lo}, {gap_hi}) ms, across which the hottest "
        f"column rises {rise:.0f} C"
    )
    print(
        "                     and the Tmax surface reaches its maximum. Any Tmax quoted there is "
        "an\n                     artifact of the spline -- keep campaign conditions at "
        f"t >= {gap_hi} ms."
    )

    print("\n  shape collapse   : spread of normalized traces across flash times")
    for k, r in v["collapse"].items():
        verdict = "INCAPABLE (ignores pulse width)" if r < 1e-9 else "width-dependent"
        print(f"      {k:10s} {r:8.4f}   {verdict}")
    print(
        "                     A zero residual is NOT evidence of a universal transient; it means\n"
        "                     the shape cannot express width dependence, so the test is blind."
    )

    print(
        f"\n  effective dwell  : t_eff (ms) at Tmax = {T_TRANSITION_REF_C:.0f} C, "
        "and the tilt it implies"
    )
    print(f"      theta = kB*T^2/Ea = {theta_kelvin(T_TRANSITION_REF_C):.1f} K per e-fold of dwell")
    print(f"      {'shape':10s} {'t=2.6':>9s} {'t=10.1':>9s} {'ratio':>7s} {'tilt (C)':>9s}")
    models = build_ensemble()
    for k, s in SHAPES.items():
        lo = effective_dwell(s, T_TRANSITION_REF_C, 2.6)
        hi = effective_dwell(s, T_TRANSITION_REF_C, 10.1)
        print(f"      {k:10s} {lo:9.3f} {hi:9.3f} {hi / lo:7.3f} {models[k].tilt_c():9.1f}")

    print("\n  measured trace   : ", end="")
    if MEASURED_TRACE.exists():
        print(f"comparing against {MEASURED_TRACE.name}")
    else:
        print("SKIPPED -- no digitized T(tau) available.")
        print(f"                     Expected at data/{MEASURED_TRACE.name}. Until it exists the")
        print("                     cooling law is UNVALIDATED and the ensemble spans 0-50 C of")
        print("                     boundary tilt. This is the cheapest measurement in the")
        print("                     campaign and it collapses the ensemble to one member.")


def _figure(path: Path) -> None:
    """Four panels: the Tmax field, the spline vs the table, transients, and the collapse test."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    cols = plt.cm.viridis(np.linspace(0, 0.92, len(FLASH_T)))

    a = axes[0, 0]
    vg = np.linspace(V_LO, V_HI, 240)
    tg = np.linspace(T_LO, T_HI, 240)
    vv, tt = np.meshgrid(vg, tg)
    cf = a.contourf(vv, tt, tmax(vv, tt), levels=18, cmap="inferno")
    fig.colorbar(cf, ax=a).set_label("T$_{max}$ (°C)")
    a.contour(vv, tt, tmax(vv, tt), levels=[T_TRANSITION_REF_C], colors="cyan", linewidths=2)
    a.axhspan(FLASH_T[0], FLASH_T[1], color="w", alpha=0.30, zorder=2)
    a.text(
        0.5 * (V_LO + V_HI),
        0.5 * (FLASH_T[0] + FLASH_T[1]),
        "no table nodes\n(spline is unconstrained here)",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        zorder=6,
    )
    vn, tn = np.meshgrid(FLASH_V, FLASH_T)
    a.scatter(vn, tn, c="cyan", s=22, edgecolors="k", linewidths=0.4, zorder=5, label="table nodes")
    a.set_xlabel("voltage V (V)")
    a.set_ylabel("flash time t (ms)")
    a.set_title(
        f"1. Measured T$_{{max}}$(V, t)  (cyan = {T_TRANSITION_REF_C:.0f} °C)",
        fontweight="bold",
        fontsize=10,
    )
    a.legend(loc="upper left", fontsize=8)

    a = axes[0, 1]
    vs = np.linspace(V_LO, V_HI, 200)
    for j, t in enumerate(FLASH_T):
        a.plot(vs, tmax(vs, np.full_like(vs, t)), color=cols[j], lw=1.6, label=f"{t} ms")
        a.scatter(
            FLASH_V, FLASH_TMAX[j], color=cols[j], s=28, edgecolors="k", linewidths=0.4, zorder=5
        )
    a.set_xlabel("voltage V (V)")
    a.set_ylabel("T$_{max}$ (°C)")
    a.set_title("2. Spline vs table — passes through every node", fontweight="bold", fontsize=10)
    a.legend(title="flash time", fontsize=8)

    a = axes[1, 0]
    for j, t in enumerate(FLASH_T):
        tau, temp = FLASH.trace(CUT_V, float(t))
        lab = f"{t} ms  (T$_{{max}}$={tmax(CUT_V, t):.0f})"
        a.plot(tau, temp, color=cols[j], lw=1.8, label=lab)
    a.axhline(T_TRANSITION_REF_C, color="gray", ls="--", lw=1)
    a.set_xlim(0, 60)
    a.set_xlabel("time since flash τ (ms)")
    a.set_ylabel("T (°C)")
    a.set_title(
        f"3. Transients at {CUT_V:.0f} V — physics default (diffusion)",
        fontweight="bold",
        fontsize=10,
    )
    a.legend(fontsize=7.5)

    a = axes[1, 1]
    for key, style in (("isoT", ":"), ("diffusion", "-")):
        shape = SHAPES[key]
        for j, t in enumerate(FLASH_T):
            tau = np.linspace(0.0, shape.duration_ms, 3000)
            a.plot(
                tau,
                shape(tau, float(t)),
                color=cols[j],
                lw=1.5,
                ls=style,
                label=f"{key}, {t} ms" if j in (0, len(FLASH_T) - 1) else None,
            )
    a.set_xlim(0, 40)
    a.set_xlabel("time since flash τ (ms)")
    a.set_ylabel("normalized  (T − T$_{room}$)/(T$_{max}$ − T$_{room}$)")
    a.set_title(
        "4. Collapse test: dotted (iso-T) collapses BY CONSTRUCTION,\n"
        "solid (diffusion) does not — the test can now fail",
        fontweight="bold",
        fontsize=10,
    )
    a.legend(fontsize=7.5)

    plt.tight_layout()
    save_figure(fig, str(path))


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    v = _validate()
    _report(v)

    assert v["node_err"] < 1e-6, "spline does not interpolate the measured nodes"
    assert v["collapse"]["diffusion"] > 1e-3, "collapse test is blind for the default shape"
    assert v["collapse"]["isoT"] < 1e-9, "legacy shape should collapse exactly"

    _figure(OUT / "thermal_check.png")
    print(f"\nSaved -> {OUT / 'thermal_check.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
