"""Settings blocks, column matching and bad cells, on synthetic sheets and real exports."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from electrical.flag_issues import IssueCode
from electrical.read_keithley_export import (
    ColumnSpec,
    InstrumentSettings,
    WorkbookShapeError,
    check_settings,
    parse_settings,
    read_keithley_export,
    read_numeric_column,
)
from paths import round_raw_dir

WHERE = {"sample_key": "sample16", "capacitor": "R9C9", "path": Path("x.xls"), "sheet": "Data"}


def _block(title: str, executed: str, module: str, second_cell: str | None = None) -> list:
    return [
        ["=" * 34, None, None, None],
        [title, second_cell, None, None],
        ["=" * 34, None, None, None],
        ["Last Executed", executed, None, None],
        ["Module Name", module, None, None],
        ["Parameter Name", "In/Out", "Type", "Value"],
        ["Vp", "Input", "DOUBLE", 3],
        ["V", "Output", "DBL_ARRAY", None],
        ["V_size", "Input", "INT", 10000],
    ]


class TestSettingsBlocks:
    def test_a_two_run_sheet_gives_a_block_per_run_sheet(self):
        frame = pd.DataFrame(
            _block("Append 1", "09/10/2026 09:36:10", "pundTest", "Latest Run")
            + _block("Initial Run", "09/10/2026 09:35:26", "pundTest")
        )
        settings = parse_settings(frame)
        assert set(settings) == {"Data", "Append1"}
        assert settings["Data"].executed_at == datetime(2026, 9, 10, 9, 35, 26)
        assert settings["Append1"].executed_at == datetime(2026, 9, 10, 9, 36, 10)
        assert set(settings["Data"].inputs) == {"Vp"}  # the *_size counters are not inputs

    def test_a_single_run_sheet_describes_the_data_sheet(self):
        settings = parse_settings(
            pd.DataFrame(_block("Latest Run", "09/10/2026 08:28:53", "doubleSweepSeg"))
        )
        assert list(settings) == ["Data"] and settings["Data"].module_name == "doubleSweepSeg"

    def test_an_unparsable_timestamp_is_kept_as_written_and_reported(self):
        settings = parse_settings(
            pd.DataFrame(_block("Latest Run", "2026-09-10T09:36", "pundTest"))
        )["Data"]
        assert settings.executed_at is None and settings.executed_raw == "2026-09-10T09:36"
        assert [i.code for i in check_settings(settings, "pundTest")] == [
            IssueCode.UNPARSED_TIMESTAMP
        ]

    def test_an_unexpected_module_is_reported(self):
        settings = InstrumentSettings(
            module_name="pundTest", executed_at=None, executed_raw=None, inputs={}
        )
        assert [i.code for i in check_settings(settings, "pundEndurance")] == [
            IssueCode.UNEXPECTED_MODULE
        ]


class TestColumns:
    def test_a_bad_cell_is_reported_and_its_row_kept(self):
        values, issues = read_numeric_column(
            pd.DataFrame({"V": [0.0, "x", 2.0]}), ColumnSpec(label="V"), WHERE
        )
        assert len(values) == 3 and np.isnan(values[1])
        assert [i.code for i in issues] == [IssueCode.BAD_CELL]
        assert issues[0].where["column"] == "V" and "row 3" in issues[0].message

    def test_an_alternative_header_is_accepted(self):
        values, issues = read_numeric_column(
            pd.DataFrame({"Time": [0.0, 5e-9]}),
            ColumnSpec(label="t", alternatives=("Time",)),
            WHERE,
        )
        assert values.tolist() == [0.0, 5e-9] and issues == []

    def test_a_missing_column_is_a_shape_error(self):
        with pytest.raises(WorkbookShapeError):
            read_numeric_column(pd.DataFrame({"V": [0.0]}), ColumnSpec(label="I"), WHERE)


class TestRealExports:
    def test_a_one_run_export_has_one_run_sheet(self):
        export = read_keithley_export(round_raw_dir(2) / "Sample14" / "PUND_pris_R9C9_30um.xls")
        assert list(export.sheets) == ["Data"] and export.settings["Data"].module_name == "pundTest"
        assert len(export.sheets["Data"]) == 3400

    def test_a_two_run_export_has_both_run_sheets_in_time_order(self):
        export = read_keithley_export(
            round_raw_dir(2) / "Sample16" / "PUND_pris_1e6_R9C12_30um.xls"
        )
        assert list(export.sheets) == ["Data", "Append1"]
        assert export.settings["Data"].executed_at < export.settings["Append1"].executed_at
