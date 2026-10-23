"""Geometry and constants of the measured capacitor stack."""

#: Vacuum permittivity, F/m.
EPS0 = 8.854e-12
#: HZO film thickness, m (fused silica / TiN 20 nm / HZO 10 nm / TiN 10 nm).
T_HZO_M = 10e-9
#: Area of the 30 x 30 um capacitor the file names default to, cm^2.
AREA_CM2_DEFAULT = 9.0e-6


def area_cm2(size_um):
    """Return the capacitor area from the side length in the file name (30 -> 9.0e-6 cm^2)."""
    if size_um is None:
        return AREA_CM2_DEFAULT
    return (float(size_um) * 1e-4) ** 2
