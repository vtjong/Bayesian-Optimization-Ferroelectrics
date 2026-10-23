"""The per-capacitor column contract, the per-sample columns, and the campaign merge."""

import re

import numpy as np
import pandas as pd
import pytest

from electrical.build_feature_tables import (
    CAMPAIGN_COLUMNS,
    CAPACITOR_COLUMNS,
    DERIVED_COLUMNS,
    HEADLINE_FEATURES,
    IDENTITY_COLUMNS,
    SAMPLE_COLUMNS,
    SWITCHING_ONLY_FEATURES,
    TRAILING_COLUMNS,
    CapacitorRecord,
    build_capacitor_row,
    build_capacitor_table,
    flatten_endurance_features,
    flatten_pund_features,
    flatten_pv_features,
    read_campaign_table,
    update_campaign_table,
)
from electrical.extract_pund_features import SPLIT_KEYS
from electrical.find_measurement_files import CapacitorId
from utils.file_io import ContractError

UNIT_SUFFIXES = ("_uc_cm2", "_a_cm2", "_uc_v_cm2", "_cm2", "_um", "_v", "_s", "_a", "_c", "_f")
#: Dimensionless or categorical columns that carry no unit suffix.
UNITLESS = {
    "round", "sample", "sample_key", "sample_num", "cap", "row", "col", "state",
    "n_pulses", "pulse_pattern", "pund_n_points", "pv_n_points", "switching",
    "fraction_main", "found", "rule", "kappa_pund", "kappa_inst_u", "kappa_summary", "kappa0",
    "dkappa_rel", "kappa_neg_branch", "kappa_rise1", "pinching_index",
    "n_iterations", "max_cycles", "cycles_at_max", "wake_ratio", "wake_ratio_max",
    "fatigue_ratio", "cycles_half_wakeup", "pund_executed", "pv_executed", "n_issues",
    "has_error", "issues",
}  # fmt: skip
PREFIXES = ("endurance_", *(f"split_{key}_" for key in SPLIT_KEYS))


def _bare(column: str) -> str:
    return next((column[len(p) :] for p in PREFIXES if column.startswith(p)), column)


def _record(**overrides) -> CapacitorRecord:
    fields = dict(
        round_label=2, sample_key="sample16", sample_name="Sample16", sample_number=16,
        capacitor=CapacitorId(row=9, col=9), state="pris", side_um=30, area_cm2=9e-6,
    )  # fmt: skip
    return CapacitorRecord(**{**fields, **overrides})


class TestTheCapacitorContract:
    def test_columns_are_unique_and_bracketed_by_identity_and_issues(self):
        assert len(set(CAPACITOR_COLUMNS)) == len(CAPACITOR_COLUMNS)
        assert CAPACITOR_COLUMNS[: len(IDENTITY_COLUMNS)] == IDENTITY_COLUMNS
        assert CAPACITOR_COLUMNS[-len(TRAILING_COLUMNS) :] == TRAILING_COLUMNS

    def test_the_contract_is_the_identity_then_each_instrument_then_the_issues(self):
        blocks = (
            *flatten_pund_features(None),
            *flatten_pv_features(None),
            *flatten_endurance_features(None),
        )
        assert CAPACITOR_COLUMNS == (*IDENTITY_COLUMNS, *blocks, *TRAILING_COLUMNS)

    def test_every_column_carries_a_unit_or_is_known_to_be_unitless(self):
        unexplained = [
            c
            for c in CAPACITOR_COLUMNS
            if not c.endswith(UNIT_SUFFIXES) and _bare(c) not in UNITLESS
        ]
        assert unexplained == []

    def test_column_names_are_snake_case(self):
        assert all(re.fullmatch(r"[a-z][a-z0-9_]*", c) for c in CAPACITOR_COLUMNS)

    def test_a_capacitor_with_no_runs_lays_out_to_exactly_the_contract(self):
        row = build_capacitor_row(_record())
        assert tuple(row) == CAPACITOR_COLUMNS
        assert row["cap"] == "R9C9" and np.isnan(row["psw_uc_cm2"]) and row["n_issues"] == 0

    def test_the_table_adds_the_ratios_and_reads_a_missing_pund_as_not_switching(self):
        table = build_capacitor_table([_record()])
        assert list(table.columns) == [*CAPACITOR_COLUMNS, *DERIVED_COLUMNS]
        assert table["switching"].tolist() == [False] and np.isnan(table.loc[0, "psw_over_pr"])


class TestTheSampleColumns:
    def test_headline_features_are_capacitor_columns_or_ratios(self):
        assert [
            f for f in HEADLINE_FEATURES if f not in (*CAPACITOR_COLUMNS, *DERIVED_COLUMNS)
        ] == []

    def test_switching_only_features_are_headline_features(self):
        assert set(SWITCHING_ONLY_FEATURES) <= set(HEADLINE_FEATURES)

    def test_sample_columns_are_unique(self):
        assert len(set(SAMPLE_COLUMNS)) == len(SAMPLE_COLUMNS)


def _campaign_rows(
    round_label: int, sample: str, caps: list[str], state: str = "pris", value: float = 1.0
):
    frame = pd.DataFrame({c: [None] * len(caps) for c in CAMPAIGN_COLUMNS})
    frame["round"] = round_label
    frame["sample"] = sample
    frame["sample_key"] = sample.lower()
    frame["sample_num"] = int(sample.removeprefix("Sample"))
    frame["cap"] = caps
    frame["row"] = [int(c[1]) for c in caps]
    frame["col"] = [int(c[3:]) for c in caps]
    frame["state"] = state
    frame["psw_uc_cm2"] = value
    frame["outlier_flags"] = ""
    return frame


class TestTheCampaignTable:
    def test_a_rerun_round_replaces_its_own_rows_and_keeps_the_others(self, tmp_path):
        path = tmp_path / "campaign.csv"
        update_campaign_table(path, _campaign_rows(2, "Sample16", ["R9C9", "R9C8"], value=1.0))
        update_campaign_table(path, _campaign_rows(3, "Sample20", ["R9C9"], value=5.0))
        update_campaign_table(path, _campaign_rows(2, "Sample16", ["R9C9"], value=2.0))
        values = read_campaign_table(path).set_index(["round", "sample_key", "cap"])["psw_uc_cm2"]
        assert values.to_dict() == {
            (2, "sample16", "R9C8"): 1.0,
            (2, "sample16", "R9C9"): 2.0,
            (3, "sample20", "R9C9"): 5.0,
        }

    def test_pristine_and_cycled_rows_of_one_capacitor_are_kept_apart(self, tmp_path):
        path = tmp_path / "campaign.csv"
        update_campaign_table(path, _campaign_rows(2, "Sample16", ["R9C9"], state="pris"))
        merged = update_campaign_table(
            path, _campaign_rows(2, "Sample16", ["R9C9"], state="post1e6")
        )
        assert sorted(merged["state"]) == ["post1e6", "pris"]

    def test_a_table_written_by_another_version_is_refused(self, tmp_path):
        path = tmp_path / "campaign.csv"
        pd.DataFrame({"round": [2], "sample": ["Sample16"]}).to_csv(path, index=False)
        with pytest.raises(ContractError):
            update_campaign_table(path, _campaign_rows(2, "Sample16", ["R9C9"]))
