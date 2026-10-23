"""The per-capacitor and per-sample column contracts are generated, unique and unit-suffixed."""

import re

import numpy as np

from electrical.aggregate import HEADLINE_FEATURES, SAMPLE_COLUMNS, SWITCHING_ONLY_FEATURES
from electrical.capacitor import (
    CAPACITOR_COLUMNS,
    IDENTITY_COLUMNS,
    TRAILING_COLUMNS,
    CapacitorRecord,
    capacitor_row,
)
from electrical.locate import CapacitorId
from electrical.pund import SPLIT_KEYS

UNIT_SUFFIXES = ("_uc_cm2", "_a_cm2", "_uc_v_cm2", "_cm2", "_um", "_v", "_s", "_a", "_c", "_f")
#: Dimensionless or categorical columns that carry no unit suffix.
UNITLESS = {
    "round", "sample", "sample_key", "sample_num", "cap", "row", "col", "state",
    "n_pulses", "pulse_pattern", "pund_n_points", "pv_n_points", "switching",
    "fraction_main", "found", "rule", "kappa_pund", "kappa_inst_u", "kappa_summary", "kappa0",
    "dkappa_rel", "kappa_neg_branch", "kappa_rise1", "pinching_index",
    "n_iterations", "max_cycles", "cycles_at_max", "wake_ratio", "wake_ratio_max",
    "fatigue_ratio", "cycles_half_wakeup", "pund_executed", "pv_executed", "n_issues",
    "has_error", "issues", "psw_over_pr", "psw_inst_over_pr",
}  # fmt: skip


SPLIT_PREFIXES = tuple(f"split_{key}_" for key in SPLIT_KEYS)


def _bare(column: str) -> str:
    for prefix in ("endurance_", *SPLIT_PREFIXES):
        if column.startswith(prefix):
            return column[len(prefix) :]
    return column


class TestCapacitorColumns:
    def test_columns_are_unique_and_bracketed_by_identity_and_findings(self):
        assert len(set(CAPACITOR_COLUMNS)) == len(CAPACITOR_COLUMNS)
        assert CAPACITOR_COLUMNS[: len(IDENTITY_COLUMNS)] == IDENTITY_COLUMNS
        assert CAPACITOR_COLUMNS[-len(TRAILING_COLUMNS) :] == TRAILING_COLUMNS

    def test_every_column_carries_a_unit_or_is_known_to_be_unitless(self):
        unexplained = [
            c for c in CAPACITOR_COLUMNS
            if not c.endswith(UNIT_SUFFIXES) and _bare(c) not in UNITLESS
        ]  # fmt: skip
        assert unexplained == []

    def test_column_names_are_snake_case(self):
        assert all(re.fullmatch(r"[a-z][a-z0-9_]*", c) for c in CAPACITOR_COLUMNS)

    def test_an_empty_record_flattens_to_exactly_the_contract(self):
        record = CapacitorRecord(
            round_label=2, sample_key="sample16", sample_name="Sample16", sample_number=16,
            capacitor=CapacitorId(row=9, col=9), state="pris", side_um=30, area_cm2=9e-6,
        )  # fmt: skip
        row = capacitor_row(record)
        assert tuple(row) == CAPACITOR_COLUMNS
        assert row["cap"] == "R9C9" and np.isnan(row["psw_uc_cm2"]) and row["n_issues"] == 0


class TestSampleColumns:
    def test_headline_features_are_capacitor_columns_or_derived_ratios(self):
        derived = {"psw_over_pr", "psw_inst_over_pr"}
        missing = [f for f in HEADLINE_FEATURES if f not in CAPACITOR_COLUMNS and f not in derived]
        assert missing == []

    def test_switching_only_features_are_headline_features(self):
        assert set(SWITCHING_ONLY_FEATURES) <= set(HEADLINE_FEATURES)

    def test_sample_columns_are_unique(self):
        assert len(set(SAMPLE_COLUMNS)) == len(SAMPLE_COLUMNS)
