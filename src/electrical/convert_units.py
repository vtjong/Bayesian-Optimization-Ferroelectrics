"""Convert measured charge and capacitance to polarization and permittivity of the HZO film.

Every capacitor is a square TiN / HZO / TiN stack whose side length is in its file name
(``..._30um.xls``); the HZO film is 10 nm thick.
"""

#: Vacuum permittivity, F/m.
VACUUM_PERMITTIVITY_F_PER_M = 8.854e-12
#: HZO film thickness, m (fused silica / TiN 20 nm / HZO 10 nm / TiN 10 nm).
HZO_THICKNESS_M = 10e-9
#: Area of the 30 x 30 um capacitor, used when a file name gives no side length, cm^2.
DEFAULT_AREA_CM2 = 9.0e-6


def area_cm2(side_um: float | None) -> float:
    """Return the capacitor area from its side length (30 um -> 9.0e-6 cm^2)."""
    if side_um is None:
        return DEFAULT_AREA_CM2
    return (float(side_um) * 1e-4) ** 2


def uc_cm2_per_coulomb(area: float) -> float:
    """Return the factor that turns a charge in C into a polarization in uC/cm^2 on ``area``."""
    return 1e6 / area


def kappa_from_capacitance(capacitance_f: float, area: float) -> float:
    """Return the relative permittivity of the HZO film from a capacitance on ``area`` cm^2."""
    return float(capacitance_f * HZO_THICKNESS_M / (VACUUM_PERMITTIVITY_F_PER_M * (area * 1e-4)))
