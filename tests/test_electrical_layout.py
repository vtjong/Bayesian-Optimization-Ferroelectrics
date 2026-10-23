"""Each module of the feature pipeline owns one phase, and the boundaries stay where they are."""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "electrical"

#: Every module and the names a reader should expect to find in it.
OWNERSHIP = {
    "device": ("EPS0", "T_HZO_M", "area_cm2"),
    "issues": ("ValidationIssue", "IssueCode", "ElectricalError", "describe_issues"),
    "locate": ("MeasurementFile", "CapacitorRuns", "parse_file_name", "group_runs"),
    "workbook": ("PundRun", "PvRun", "EnduranceTable", "read_workbook", "parse_settings"),
    "pund": ("PundFeatures", "SplitRule", "segment_pulses", "split_ramp", "pund_features"),
    "pv": ("PvFeatures", "pv_features"),
    "endurance": ("EnduranceFeatures", "endurance_features"),
    "capacitor": ("CapacitorRecord", "CAPACITOR_COLUMNS", "capacitor_row", "capacitor_table"),
    "aggregate": ("HEADLINE_FEATURES", "SAMPLE_COLUMNS", "per_sample"),
    "accumulate": ("write_table", "update_campaign", "read_campaign"),
    "report": ("flag_outliers", "build_report", "write_report"),
    "pipeline": ("RoundResult", "process_round", "describe_round"),
}  # fmt: skip
#: Imports that belong to exactly one module.
EXCLUSIVE_IMPORTS = {
    "matplotlib": "report",
    "html": "report",
    "tempfile": "accumulate",
}
#: Calls that belong to exactly one module.
EXCLUSIVE_CALLS = {"read_excel": "workbook", "ExcelFile": "workbook", "to_csv": "accumulate"}


def _modules() -> dict[str, ast.Module]:
    return {p.stem: ast.parse(p.read_text()) for p in PACKAGE.glob("*.py") if p.stem != "__init__"}


def _defined_names(tree: ast.Module) -> set[str]:
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_the_package_has_exactly_the_modules_it_documents():
    assert set(_modules()) == set(OWNERSHIP)


@pytest.mark.parametrize("module", sorted(OWNERSHIP))
def test_each_module_defines_what_it_owns(module):
    missing = [name for name in OWNERSHIP[module] if name not in _defined_names(_modules()[module])]
    assert missing == [], f"{module}.py no longer defines {missing}"


@pytest.mark.parametrize(("imported", "owner"), sorted(EXCLUSIVE_IMPORTS.items()))
def test_some_imports_belong_to_one_module(imported, owner):
    users = sorted(
        {
            name
            for name, tree in _modules().items()
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and any((alias.name.split(".")[0] == imported) for alias in node.names)
            or (isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == imported)
        }
    )
    assert users == [owner], f"{imported} is imported by {users}; only {owner}.py may use it"


@pytest.mark.parametrize(("call", "owner"), sorted(EXCLUSIVE_CALLS.items()))
def test_some_calls_belong_to_one_module(call, owner):
    users = sorted(
        {
            name
            for name, tree in _modules().items()
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == call
        }
    )
    assert users == [owner], f"{call} is called from {users}; only {owner}.py may"
