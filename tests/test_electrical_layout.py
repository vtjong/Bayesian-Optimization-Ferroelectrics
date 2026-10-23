"""Each module of the electrical package owns one step, and the boundaries between them hold."""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "electrical"

#: Every module, in reading order, and the names a reader should expect to find in it.
OWNERSHIP = {
    "process_round": ("RoundResult", "process_round"),
    "find_measurement_files": (
        "MeasurementFile", "parse_file_name", "find_measurement_files", "sample_key",
    ),
    "read_keithley_export": (
        "KeithleyExport", "read_keithley_export", "parse_settings", "read_numeric_column",
    ),
    "read_measurement_runs": (
        "PundRun", "PvRun", "EnduranceLog", "read_runs", "assign_states", "group_runs",
    ),
    "extract_pund_features": (
        "PundFeatures", "segment_pulses", "split_ramp", "switched_polarization",
        "extract_pund_features",
    ),
    "extract_pv_features": ("PvFeatures", "extract_pv_features"),
    "extract_endurance_features": ("EnduranceFeatures", "extract_endurance_features"),
    "convert_units": ("area_cm2", "uc_cm2_per_coulomb", "kappa_from_capacitance"),
    "build_feature_tables": (
        "CAPACITOR_COLUMNS", "build_capacitor_table", "summarize_samples", "update_campaign_table",
    ),
    "flag_issues": ("Issue", "IssueCode", "describe_issues", "flag_outliers"),
    "write_qc_report": ("build_qc_page", "write_qc_report"),
}  # fmt: skip
#: Imports and calls that belong to exactly one module.
EXCLUSIVE_IMPORTS = {"matplotlib": "write_qc_report", "html": "write_qc_report"}
EXCLUSIVE_CALLS = {"read_excel": "read_keithley_export", "ExcelFile": "read_keithley_export"}


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


def _imported_roots(tree: ast.Module) -> set[str]:
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _called_attributes(tree: ast.Module) -> set[str]:
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


def test_the_package_has_exactly_the_modules_it_documents():
    assert set(_modules()) == set(OWNERSHIP)


def test_the_package_docstring_lists_every_module():
    docstring = ast.get_docstring(ast.parse((PACKAGE / "__init__.py").read_text()))
    assert [m for m in OWNERSHIP if m != "process_round" and f"``{m}``" not in docstring] == []


@pytest.mark.parametrize("module", sorted(OWNERSHIP))
def test_each_module_defines_what_it_owns(module):
    missing = [name for name in OWNERSHIP[module] if name not in _defined_names(_modules()[module])]
    assert missing == [], f"{module}.py no longer defines {missing}"


@pytest.mark.parametrize(("imported", "owner"), sorted(EXCLUSIVE_IMPORTS.items()))
def test_some_imports_belong_to_one_module(imported, owner):
    users = sorted(name for name, tree in _modules().items() if imported in _imported_roots(tree))
    assert users == [owner], f"{imported} is imported by {users}; only {owner}.py may"


@pytest.mark.parametrize(("call", "owner"), sorted(EXCLUSIVE_CALLS.items()))
def test_some_calls_belong_to_one_module(call, owner):
    users = sorted(name for name, tree in _modules().items() if call in _called_attributes(tree))
    assert users == [owner], f"{call} is called from {users}; only {owner}.py may"


def test_tables_reach_disk_only_through_the_shared_file_helpers():
    writers = sorted(
        name for name, tree in _modules().items() if "to_csv" in _called_attributes(tree)
    )
    assert writers == [], f"{writers} write CSV directly; use utils.file_io"
