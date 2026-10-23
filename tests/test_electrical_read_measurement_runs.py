"""Runs read from real exports, the states they are assigned, and how they are paired."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from electrical.find_measurement_files import (
    CapacitorId,
    MeasurementKind,
    RoundLayoutError,
    parse_file_name,
)
from electrical.flag_issues import IssueCode
from electrical.read_keithley_export import InstrumentSettings
from electrical.read_measurement_runs import (
    EnduranceLog,
    InstrumentCharges,
    PundRun,
    PvRun,
    RunSource,
    assign_states,
    group_runs,
    read_runs,
)
from paths import round_raw_dir


def _run(
    name: str, sheet: str = "Data", executed: datetime | None = None, folder: str = "Sample16"
):
    """A run with empty arrays, for pairing tests that never look at the data."""
    file = parse_file_name(Path(folder) / name)
    settings = InstrumentSettings(
        module_name=None, executed_at=executed, executed_raw=None, inputs={}
    )
    source = RunSource(file=file, sheet=sheet, settings=settings)
    empty = np.zeros(3)
    if file.kind is MeasurementKind.PUND:
        return PundRun(
            source=source,
            time_s=empty,
            voltage_v=empty,
            current_a=empty,
            instrument=InstrumentCharges(),
        )
    if file.kind is MeasurementKind.PV:
        return PvRun(source=source, time_s=empty, voltage_v=empty, current_a=empty, charge_c=empty)
    return EnduranceLog(source=source, cycles=empty, p_c=empty, u_c=empty, n_c=empty, d_c=empty)


def _states(runs) -> list[str]:
    return [state for _, state in assign_states(runs)]


class TestReadingRuns:
    def test_a_one_run_pund_export_reads_as_one_run(self):
        (run,) = read_runs(
            parse_file_name(round_raw_dir(2) / "Sample14" / "PUND_pris_R9C9_30um.xls")
        )
        assert isinstance(run, PundRun) and run.source.sheet == "Data"
        assert len(run.time_s) == 3400 and run.dt_s == pytest.approx(5e-9)
        assert np.isfinite(run.instrument.psw_c) and run.issues == ()

    def test_a_two_run_export_reads_as_two_runs(self):
        first, second = read_runs(
            parse_file_name(round_raw_dir(2) / "Sample16" / "PV_pris_1e6_R9C12_30um.xls")
        )
        assert (first.source.sheet, second.source.sheet) == ("Data", "Append1")
        assert isinstance(first, PvRun) and len(first.charge_c) == len(second.charge_c) == 1000

    def test_an_endurance_log_reads_its_cycle_counts(self):
        (log,) = read_runs(
            parse_file_name(round_raw_dir(2) / "Sample16" / "END_1e6_R9C12_30um.xls")
        )
        assert isinstance(log, EnduranceLog)
        assert log.cycles[0] == 0 and log.cycles[-1] == 1e6 and len(log.cycles) == 9


class TestAssigningStates:
    def test_a_two_run_file_is_pristine_then_post_cycling(self):
        runs = [
            _run("PUND_pris_1e6_R9C9_30um.xls", "Data"),
            _run("PUND_pris_1e6_R9C9_30um.xls", "Append1"),
        ]
        assert _states(runs) == ["pris", "post1e6"]

    def test_an_endurance_log_belongs_to_the_pristine_state(self):
        assert _states([_run("END_1e6_R9C9_30um.xls")]) == ["pris"]

    def test_a_two_run_file_without_a_state_token_still_starts_pristine(self):
        runs = [_run("PUND_1e6_R9C10_30um.xls", "Data"), _run("PUND_1e6_R9C10_30um.xls", "Append1")]
        assert _states(runs) == ["pris", "post1e6"]

    def test_a_single_run_file_without_a_state_token_is_post_cycling(self):
        assert _states([_run("PV_1e6_R9C11_30um.xls")]) == ["post1e6"]

    def test_an_append_run_without_a_cycle_count_takes_it_from_the_endurance_log(self):
        runs = [
            _run("PUND_pris_R9C11_30um.xls", "Data"),
            _run("PUND_pris_R9C11_30um.xls", "Append1"),
            _run("END_1e6_R9C11_30um.xls"),
        ]
        assert _states(runs) == ["pris", "post1e6", "pris"]


class TestPairingRuns:
    def test_a_capacitor_gets_a_pristine_and_a_post_cycling_state(self):
        runs = [
            _run("PUND_pris_1e6_R9C9_30um.xls", "Data"),
            _run("PUND_pris_1e6_R9C9_30um.xls", "Append1"),
            _run("PV_pris_1e6_R9C9_30um.xls", "Data"),
            _run("PV_pris_1e6_R9C9_30um.xls", "Append1"),
            _run("END_1e6_R9C9_30um.xls"),
        ]
        groups = group_runs(assign_states(runs))
        assert [
            (g.state, g.pund is not None, g.pv is not None, g.endurance is not None) for g in groups
        ] == [
            ("post1e6", True, True, False),
            ("pris", True, True, True),
        ]
        assert all(g.capacitor == CapacitorId(row=9, col=9) and g.issues == () for g in groups)

    def test_a_missing_run_is_reported_on_its_state(self):
        groups = group_runs(
            assign_states([_run("PUND_pris_R9C11_30um.xls"), _run("PV_1e6_R9C11_30um.xls")])
        )
        assert [(g.state, {i.code for i in g.issues}) for g in groups] == [
            ("post1e6", {IssueCode.UNPAIRED_PUND, IssueCode.SUSPECT_FILE_NAME}),
            ("pris", {IssueCode.UNPAIRED_PV}),
        ]

    def test_pund_before_the_presetting_sweep_is_reported_on_the_pristine_state_only(self):
        early, late = datetime(2026, 9, 10, 9, 0), datetime(2026, 9, 10, 9, 5)
        runs = [
            _run("PUND_pris_1e6_R9C9_30um.xls", "Data", early),
            _run("PV_pris_1e6_R9C9_30um.xls", "Data", late),
            _run("PUND_pris_1e6_R9C9_30um.xls", "Append1", early),
            _run("PV_pris_1e6_R9C9_30um.xls", "Append1", late),
        ]
        by_state = {g.state: [i.code for i in g.issues] for g in group_runs(assign_states(runs))}
        assert by_state == {"pris": [IssueCode.PUND_BEFORE_PV], "post1e6": []}

    def test_a_suspect_duplicate_gives_way_to_the_tidy_file_and_is_reported(self):
        runs = [_run("PUND_pris_1e6_R9C8_30um_x.xls"), _run("PUND_pris_1e6_R9C8_30um.xls")]
        (group,) = group_runs(assign_states(runs))
        assert group.pund.source.file.path.name == "PUND_pris_1e6_R9C8_30um.xls"
        assert {i.code for i in group.issues} == {IssueCode.UNPAIRED_PV, IssueCode.DUPLICATE_RUN}

    def test_two_tidy_files_for_one_run_stop_the_round(self):
        runs = [_run("PUND_pris_1e6_R9C8_30um.xls"), _run("PUND_pris_1e6_R9C8_30um.xls")]
        with pytest.raises(RoundLayoutError):
            group_runs(assign_states(runs))

    def test_a_folder_without_a_sample_number_is_reported(self):
        (group,) = group_runs(
            assign_states([_run("PUND_pris_R9C9_30um.xls", folder="Pristine Sample")])
        )
        assert group.sample_key == "pristinesample"
        assert IssueCode.NO_SAMPLE_NUMBER in [i.code for i in group.issues]
