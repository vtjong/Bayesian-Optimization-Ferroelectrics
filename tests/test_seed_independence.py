"""The seed generator must not be able to reach a model.

The previous seed failed because two bounds derived from the thermal simulation -- a temperature
floor and a minimum pulse width -- decided which parts of the instrument's range were allowed into
the design. Both were invisible in every coverage diagnostic, because the hypercube was correctly
stratified inside the domain it was handed.

The structural defence is that ``seed.py`` cannot import anything capable of expressing such a
bound. If it depends only on numpy and scipy, then every number in it is either an instrument
specification or a declared design parameter, and a reader can verify that by reading one file.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
ALLOWED_ROOTS = {"numpy", "scipy", "typing"}


def _imported_roots(path: Path) -> set:
    """Top-level names imported by a module, relative imports included.

    :param path: path to the module.
    """
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            found.add((node.module or "").split(".")[0] if not node.level else "<relative>")
        elif isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
    return {f for f in found if f}


def test_the_seed_imports_nothing_but_numpy_and_scipy():
    """Anything else could carry a model-derived bound into the design."""
    leaked = _imported_roots(SRC / "seed.py") - ALLOWED_ROOTS
    assert not leaked, (
        f"seed.py imports {sorted(leaked)}. Every bound in the seed must be an instrument limit or "
        "a declared design parameter; an import is how a thermal assumption gets in."
    )


def test_the_seed_spans_the_full_settable_range():
    """A design that stops short of the instrument's limits has excluded something."""
    import sys

    sys.path.insert(0, str(SRC))
    import seed

    v, t = seed.make_seed()
    assert t.min() <= 0.2, f"shortest pulse is {t.min()} ms; the tool reaches {seed.T_LO}"
    assert t.max() >= 0.5 * seed.T_HI, f"longest pulse {t.max()} ms; tool reaches {seed.T_HI}"
    assert v.min() <= seed.V_LO + 0.15 * (seed.V_HI - seed.V_LO), "no low-voltage condition"
    assert v.max() >= seed.V_HI - 0.15 * (seed.V_HI - seed.V_LO), "no high-voltage condition"


def test_conditions_are_settable_and_distinct():
    """A plan the operator cannot dial in, or that repeats a condition, is a defect."""
    import sys

    sys.path.insert(0, str(SRC))
    import seed

    v, t = seed.make_seed()
    assert len(v) == seed.SEED_SIZE
    assert (v == v.round()).all(), "a voltage is not a whole volt"
    assert ((t * 10).round() == (t * 10)).all(), "a pulse width is not on the 0.1 ms grid"
    assert ((v >= seed.V_LO) & (v <= seed.V_HI)).all()
    assert ((t >= seed.T_LO) & (t <= seed.T_HI)).all()
    assert len(set(zip(v.tolist(), t.tolist()))) == seed.SEED_SIZE


def test_the_draw_is_reproducible():
    """The recorded RNG seed must reproduce the fired design exactly."""
    import sys

    sys.path.insert(0, str(SRC))
    import seed

    v1, t1 = seed.make_seed()
    v2, t2 = seed.make_seed()
    assert (v1 == v2).all() and (t1 == t2).all()
