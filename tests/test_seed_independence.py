"""Tests that the seed design depends on nothing but the instrument.

The seed must be reproducible from its recorded parameters, cover the full
settable design space, and contain no quantity derived from a thermal model.

Import independence is asserted structurally rather than by review: a module
that cannot import a model cannot inherit a bound from one.
"""

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))

import seed  # noqa: E402

#: Modules the seed may import. Each is either the standard library or an array
#: or sampling primitive. None can express a physical assumption.
ALLOWED_IMPORT_ROOTS = frozenset({"dataclasses", "numpy", "scipy", "typing"})

#: Fraction of an axis within which the seed must reach that axis's limits.
AXIS_COVERAGE_TOLERANCE = 0.15


def imported_roots(module_path: Path) -> set[str]:
    """Return the top-level module names imported by a source file.

    :param module_path: Path to the module to inspect.
    :return: Top-level names appearing in any import statement.
    """
    tree = ast.parse(module_path.read_text())

    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)

    return roots


def test_seed_imports_no_model() -> None:
    """The seed module imports only sampling and array primitives."""
    unexpected = imported_roots(SOURCE_ROOT / "seed.py") - ALLOWED_IMPORT_ROOTS

    assert not unexpected, (
        f"seed.py imports {sorted(unexpected)}. Every bound in the seed must be "
        f"an instrument limit or a recorded design parameter."
    )


@pytest.mark.parametrize(
    ("axis", "column"),
    [
        (seed.VOLTAGE, 0),
        (seed.PULSE_WIDTH, 1),
    ],
)
def test_conditions_lie_on_the_instrument_grid(
    axis: seed.InstrumentAxis,
    column: int,
) -> None:
    """Every setpoint is within range and on the instrument's resolution.

    :param axis: Instrument axis the values belong to.
    :param column: Index of that axis in the returned conditions.
    """
    values = np.column_stack(seed.make_seed())[:, column]

    assert values.min() >= axis.minimum
    assert values.max() <= axis.maximum
    assert np.allclose(values / axis.step, np.round(values / axis.step))


@pytest.mark.parametrize(
    ("axis", "column"),
    [
        (seed.VOLTAGE, 0),
        (seed.PULSE_WIDTH, 1),
    ],
)
def test_seed_reaches_both_ends_of_each_axis(
    axis: seed.InstrumentAxis,
    column: int,
) -> None:
    """The seed spans the settable range rather than an interior subset.

    Coverage is measured in the coordinate the axis is stratified in, so pulse
    width is judged on its decades rather than on raw milliseconds.

    :param axis: Instrument axis to check coverage of.
    :param column: Index of that axis in the returned conditions.
    """
    values = np.column_stack(seed.make_seed())[:, column]

    if axis is seed.PULSE_WIDTH:
        values = np.log10(values)
        minimum, maximum = np.log10(axis.minimum), np.log10(axis.maximum)
    else:
        minimum, maximum = axis.minimum, axis.maximum

    span = maximum - minimum

    assert values.min() <= minimum + AXIS_COVERAGE_TOLERANCE * span
    assert values.max() >= maximum - AXIS_COVERAGE_TOLERANCE * span


def test_conditions_are_distinct() -> None:
    """Snapping never collapses two conditions onto one another."""
    conditions = np.column_stack(seed.make_seed())

    assert conditions.shape[0] == seed.SEED_SIZE
    assert np.unique(conditions, axis=0).shape[0] == seed.SEED_SIZE


def test_seed_is_reproducible() -> None:
    """The recorded parameters reproduce the fired design exactly."""
    first = np.column_stack(seed.make_seed())
    second = np.column_stack(seed.make_seed())

    assert np.array_equal(first, second)
