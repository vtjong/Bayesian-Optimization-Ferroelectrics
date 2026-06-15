"""Extract physics descriptors from a thermal profile T(t) — the boss's thermal features.

Given a temperature history, compute the scalar descriptors the kinetic mechanism models
consume: peak temperature, heating/cooling rate, time-above-threshold (dwell), integrated
thermal budget, and the Arrhenius-weighted thermal budget K = ∫exp(-Ea/kB·T(t)) dt.

NOTE (identifiability): for a single pulse, every one of these is a deterministic function
of (V, t), so they are mutually collinear — useful for physics/interpretation and as
inputs to the kinetic forward models, but they add no independent *predictive* information
over (V, t) until the design is decorrelated (preheat / multi-pulse).
"""

import numpy as np

KB_EV = 8.617e-5  # Boltzmann constant, eV/K


def extract_descriptors(t_ms, T_C, ea_eV=1.5, thresholds=(400.0, 500.0)):
    """Return a dict of thermal descriptors for the profile (t_ms, T_C).

    :param t_ms: Time samples (ms)
    :param T_C: Temperature (deg C) at each sample
    :param ea_eV: Activation energy for the Arrhenius budget (eV)
    :param thresholds: Temperatures (deg C) for dwell/time-above descriptors
    """
    T_C = np.asarray(T_C, dtype=float)
    t_ms = np.asarray(t_ms, dtype=float)
    T_K = T_C + 273.15
    dTdt = np.gradient(T_C, t_ms)  # deg C / ms

    out = {
        "Tmax": float(T_C.max()),
        "heating_rate": float(dTdt.max()),  # deg C / ms
        "cooling_rate": float(-dTdt.min()),  # deg C / ms (positive)
        "thermal_integral": float(np.trapz(np.clip(T_C - T_C[0], 0.0, None), t_ms)),
        "arrhenius_K": float(np.trapz(np.exp(-ea_eV / (KB_EV * T_K)), t_ms)),
    }
    for thr in thresholds:
        out[f"t_gt_{int(thr)}"] = float(np.trapz((T_C > thr).astype(float), t_ms))
    return out
