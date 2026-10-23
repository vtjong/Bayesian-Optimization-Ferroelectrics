"""Settings blocks, column matching and cell findings, on synthetic frames and two real exports."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from electrical.issues import IssueCode, WorkbookShapeError
from electrical.locate import parse_file_name
from electrical.workbook import (
    EnduranceTable,
    InstrumentSettings,
    PundRun,
    RunSource,
    build_run,
    parse_settings,
    read_workbook,
)
from paths import round_raw_dir


def _block(title: str, executed: str, module: str, second_cell: str | None = None) -> list:
    rows = [
        ["=" * 34, None, None, None],
        [title, second_cell, None, None],
        ["=" * 34, None, None, None],
    ]
    rows += [
        ["Site Coordinate", "0,0", None, None],
        ["Last Executed", executed, None, None],
        ["Module Name", module, None, None],
        [None, None, None, None],
        ["Parameter Name", "In/Out", "Type", "Value"],
        ["Vp", "Input", "DOUBLE", 3],
        ["V", "Output", "DBL_ARRAY", None],
        ["V_size", "Input", "INT", 10000],
    ]
    return rows


class TestSettingsBlocks:
    def test_a_two_run_sheet_yields_a_block_per_data_sheet(self):
        frame = pd.DataFrame(
            _block("Append 1", "09/10/2026 09:36:10", "pundTest", "Latest Run")
            + _block("Initial Run", "09/10/2026 09:35:26", "pundTest")
        )
        settings = parse_settings(frame)
        assert set(settings) == {"Data", "Append1"}
        assert settings["Data"].executed_at == datetime(2026, 9, 10, 9, 35, 26)
        assert settings["Append1"].executed_at == datetime(2026, 9, 10, 9, 36, 10)
        assert settings["Data"].module_name == "pundTest"
        assert set(settings["Data"].inputs) == {"Vp"}  # the *_size counters are not inputs

    def test_a_single_run_sheet_describes_the_data_sheet(self):
        settings = parse_settings(
            pd.DataFrame(_block("Latest Run", "09/10/2026 08:28:53", "doubleSweepSeg"))
        )
        assert list(settings) == ["Data"]
        assert settings["Data"].module_name == "doubleSweepSeg"

    def test_an_unparsable_timestamp_is_kept_as_written(self):
        settings = parse_settings(
            pd.DataFrame(_block("Latest Run", "2026-09-10T09:36", "pundTest"))
        )
        assert settings["Data"].executed_at is None
        assert settings["Data"].executed_raw == "2026-09-10T09:36"


def _source(name: str, module: str, sheet: str = "Data") -> RunSource:
    settings = InstrumentSettings(
        module_name=module, executed_at=None, executed_raw=None, inputs={}
    )
    return RunSource(file=parse_file_name(Path("Sample16") / name), sheet=sheet, settings=settings)


class TestBuildingRuns:
    def test_a_bad_cell_is_reported_and_its_row_kept(self):
        frame = pd.DataFrame(
            {
                "V": [0.0, "x", 2.0],
                "I": [0.0, 0.0, 0.0],
                "t": [0.0, 5e-9, 1e-8],
                "P": [1e-10, None, None],
            }
        )
        run = build_run(_source("PUND_pris_R9C9_30um.xls", "pundTest"), frame)
        assert isinstance(run, PundRun)
        assert run.n_points == 3 and np.isnan(run.voltage_v[1])
        assert [i.code for i in run.issues] == [IssueCode.BAD_CELL]
        assert run.issues[0].location == "Data!V"
        assert run.instrument.p_c == 1e-10 and np.isnan(run.instrument.u_c)

    def test_an_alternative_header_is_accepted(self):
        frame = pd.DataFrame({"V": [0.0, 1.0], "I": [0.0, 0.0], "Time": [0.0, 5e-9]})
        run = build_run(_source("PUND_pris_R9C9_30um.xls", "pundTest"), frame)
        assert run.dt_s == 5e-9

    def test_a_missing_required_column_is_a_shape_error(self):
        frame = pd.DataFrame({"V": [0.0, 1.0], "t": [0.0, 5e-9]})
        with pytest.raises(WorkbookShapeError):
            build_run(_source("PUND_pris_R9C9_30um.xls", "pundTest"), frame)

    def test_an_unexpected_module_is_reported(self):
        frame = pd.DataFrame(
            {
                "iteration": [0, 1],
                "P": [1.0, 1.0],
                "U": [0.0, 0.0],
                "N": [-1.0, -1.0],
                "D": [0.0, 0.0],
            }
        )
        run = build_run(_source("END_1e6_R9C9_30um.xls", "pundTest"), frame)
        assert isinstance(run, EnduranceTable)
        assert [i.code for i in run.issues] == [IssueCode.UNEXPECTED_MODULE]


class TestRealExports:
    def test_a_one_run_pund_export_reads_as_one_run(self):
        file = parse_file_name(round_raw_dir(2) / "Sample14" / "PUND_pris_R9C9_30um.xls")
        (run,) = read_workbook(file)
        assert run.source.sheet == "Data" and run.source.settings.module_name == "pundTest"
        assert run.n_points == 3400 and run.dt_s == pytest.approx(5e-9)
        assert np.isfinite(run.instrument.psw_c) and run.issues == ()

    def test_a_two_run_export_reads_as_two_runs_in_time_order(self):
        file = parse_file_name(round_raw_dir(2) / "Sample16" / "PUND_pris_1e6_R9C12_30um.xls")
        first, second = read_workbook(file)
        assert (first.source.sheet, second.source.sheet) == ("Data", "Append1")
        assert first.source.settings.executed_at < second.source.settings.executed_at
        assert first.n_points == second.n_points == 3400

    def test_an_endurance_log_reads_its_cycle_counts(self):
        file = parse_file_name(round_raw_dir(2) / "Sample16" / "END_1e6_R9C12_30um.xls")
        (table,) = read_workbook(file)
        assert table.cycles[0] == 0 and table.cycles[-1] == 1e6 and table.n_points == 9
