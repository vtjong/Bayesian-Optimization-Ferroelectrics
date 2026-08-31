"""The learner must not be able to see the physics.

``active_learning`` fits and searches a scalar field over the design box. It is given coordinates
and readings, never a temperature, never a cooling law, never an activation energy. That is what
makes the boundary it reports evidence about the film rather than a re-drawing of the thermal model
in ``physics`` -- and it is the claim a reader is most entitled to be sceptical about, because both
halves live in one repository and share an author.

An import is the cheapest possible way for that separation to fail. One convenience import of the
thermal model inside an acquisition and the batch is being steered by the very model it exists to
test, with nothing in the output to show it. So the separation is asserted here over the TRANSITIVE
closure of every module in the package, not left to review.

``campaign.reporting`` is deliberately exempt: it labels a boundary that was already found. Two
steps, two packages, one direction.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

FORWARD = "physics"  # turns a condition into a predicted outcome
LEARNER = "active_learning"  # chooses what to fire next


def _module_name(path: Path) -> str:
    """Dotted name of a source file relative to ``src``.

    :param path: path to a module inside ``src``.
    """
    rel = path.relative_to(SRC).with_suffix("")
    parts = [p for p in rel.parts if p != "__init__"]
    return ".".join(parts)


def _modules_in(package: str) -> list:
    """Every module belonging to a package.

    :param package: top-level package name under ``src``.
    """
    return sorted(_module_name(p) for p in (SRC / package).rglob("*.py"))


def _direct_imports(module: str) -> set:
    """Dotted names ``module`` imports from within ``src``, relative imports resolved.

    :param module: dotted module name under ``src``.
    """
    path = SRC / (module.replace(".", "/") + ".py")
    if not path.exists():
        path = SRC / module.replace(".", "/") / "__init__.py"
    if not path.exists():
        return set()
    package = module.rsplit(".", 1)[0] if "." in module else ""
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            base = node.module or ""
            found.add(f"{package}.{base}" if node.level else base)
        elif isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
    return {f for f in found if f and (SRC / f.replace(".", "/")).with_suffix(".py").exists()}


def _closure(module: str) -> set:
    """Every module under ``src`` reachable from ``module``, transitively.

    :param module: dotted module name under ``src``.
    """
    seen, stack = set(), [module]
    while stack:
        for dep in _direct_imports(stack.pop()):
            if dep not in seen:
                seen.add(dep)
                stack.append(dep)
    return seen


@pytest.mark.parametrize("module", _modules_in(LEARNER))
def test_the_learner_cannot_reach_the_forward_model(module):
    """A learner that can import the thermal model can be steered by it, silently."""
    leaked = {d for d in _closure(module) if d.split(".")[0] == FORWARD}
    assert not leaked, (
        f"{module} can reach {sorted(leaked)}. The boundary it reports is then partly an echo of "
        "our own thermal model rather than evidence about the film. If the physics is needed to "
        "PRESENT a result, put that step in campaign.reporting instead."
    )


@pytest.mark.parametrize("module", _modules_in(FORWARD))
def test_the_forward_model_does_not_depend_on_the_learner(module):
    """The physics must stay usable, and testable, with no surrogate fitted."""
    leaked = {d for d in _closure(module) if d.split(".")[0] == LEARNER}
    assert not leaked, f"{module} reaches {sorted(leaked)}; the forward model is not standalone"


def test_the_design_space_stays_free_of_both():
    """Both sides depend on the box. The moment it grows an opinion, that stops being safe."""
    reached = _closure("design_space")
    assert not {d for d in reached if d.split(".")[0] in (FORWARD, LEARNER)}, (
        f"design_space reaches {sorted(reached)}; it must hold the box and nothing else"
    )


def test_the_guard_would_notice_a_violation():
    """A test that cannot fail protects nothing -- check the walk resolves real dependencies."""
    assert "active_learning.surrogate" in _closure("active_learning.acquisition")
    assert {d for d in _closure("campaign.reporting") if d.split(".")[0] == FORWARD}, (
        "campaign.reporting should reach the physics; labelling a boundary is its whole job"
    )
