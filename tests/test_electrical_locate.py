"""File names, sample keys and the pairing of runs into capacitor-states; no workbook is opened."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from electrical.issues import IssueCode, RoundLayoutError
from electrical.locate import (
    CapacitorId,
    MeasurementKind,
    find_measurement_files,
    group_runs,
    parse_file_name,
    run_states,
    sample_key,
)
from electrical.workbook import (
    EnduranceTable,
    InstrumentCharges,
    InstrumentSettings,
    PundRun,
    PvRun,
    RunSource,
)


@pytest.mark.parametrize(
    ("name", "kind", "state", "cycles", "capacitor", "suspect"),
    [
        ("PUND_pris_R9C9_30um.xls", MeasurementKind.PUND, "pris", None, "R9C9", False),
        ("PV_pris_1e6_R10C12_30um.xls", MeasurementKind.PV, "pris", "1e6", "R10C12", False),
        ("END_1e6_R8C8_30um.xls", MeasurementKind.END, None, "1e6", "R8C8", False),
        ("PUND_1e6_R9C10_30um.xls", MeasurementKind.PUND, None, "1e6", "R9C10", False),
        ("END_1E6_R8C11_30um.xls", MeasurementKind.END, None, "1e6", "R8C11", True),
        ("PV_PRIS_1e6_R8C8_30um.xls", MeasurementKind.PV, "pris", "1e6", "R8C8", True),
        ("PUND_pris_1e6_R9c8_30um.xls", MeasurementKind.PUND, "pris", "1e6", "R9C8", True),
        ("PV_pris-1e6_R8C12_30um.xls", MeasurementKind.PV, "pris", "1e6", "R8C12", True),
        ("PUND_pris_1e6_R9C8_30um_x.xls", MeasurementKind.PUND, "pris", "1e6", "R9C8", True),
    ],
)
def test_every_naming_variant_seen_in_the_rounds_parses(
    name, kind, state, cycles, capacitor, suspect
):
    parsed = parse_file_name(Path("Round2/Sample16") / name)
    assert parsed is not None
    assert (parsed.kind, parsed.state_token, parsed.cycles_tag) == (kind, state, cycles)
    assert parsed.capacitor.label == capacitor
    assert parsed.side_um == 30
    assert parsed.suspect is suspect
    assert (parsed.sample_name, parsed.sample_key, parsed.sample_number) == (
        "Sample16",
        "sample16",
        16,
    )


@pytest.mark.parametrize(
    "name", ["PUND_R9C9.png", "desktop.ini", "Test_subtract_16.m", "notes.xls"]
)
def test_other_files_are_not_exports(name):
    assert parse_file_name(Path("Sample16") / name) is None


@pytest.mark.parametrize(
    ("folder", "key"),
    [
        ("Sample 11", "sample11"),
        ("Sample2_square", "sample2"),
        ("Sample4_again", "sample4"),
        ("Sample 39", "sample39"),
        ("sample_07", "sample7"),
        ("Pristine Sample", "pristinesample"),
    ],
)
def test_sample_keys_normalise_the_groups_spellings(folder, key):
    assert sample_key(folder) == key


class TestFindingFiles:
    def test_exports_are_found_in_sample_folders_and_plots_are_skipped(self, tmp_path):
        (tmp_path / "Sample16").mkdir()
        (tmp_path / "Sample16" / "PUND_pris_1e6_R9C9_30um.xls").touch()
        (tmp_path / "Sample16" / "PUND_pris_1e6_R9C9_30um.bmp").touch()
        (tmp_path / "Sample16" / "Analysis_Plots").mkdir()
        (tmp_path / "Sample16" / "Analysis_Plots" / "PUND_pris_1e6_R9C8_30um.xls").touch()
        (tmp_path / "Sample 15").mkdir()
        (tmp_path / "Sample 15" / "PV_pris_R8C8_30um.xls").touch()
        found = find_measurement_files(tmp_path)
        assert [(f.sample_key, f.path.name) for f in found] == [
            ("sample15", "PV_pris_R8C8_30um.xls"),
            ("sample16", "PUND_pris_1e6_R9C9_30um.xls"),
        ]

    def test_a_folder_without_exports_is_a_layout_error(self, tmp_path):
        (tmp_path / "Sample1").mkdir()
        (tmp_path / "Sample1" / "PUND_R9C9.png").touch()
        with pytest.raises(RoundLayoutError):
            find_measurement_files(tmp_path)


def _run(
    name: str, sheet: str = "Data", executed: datetime | None = None, folder: str = "Sample16"
):
    file = parse_file_name(Path(folder) / name)
    settings = InstrumentSettings(
        module_name=None, executed_at=executed, executed_raw=None, inputs={}
    )
    source = RunSource(file=file, sheet=sheet, settings=settings)
    arrays = np.zeros(3)
    if file.kind is MeasurementKind.PUND:
        return PundRun(
            source=source, frame=pd.DataFrame(), time_s=arrays, voltage_v=arrays, current_a=arrays,
            instrument=InstrumentCharges(),
        )  # fmt: skip
    if file.kind is MeasurementKind.PV:
        return PvRun(
            source=source, frame=pd.DataFrame(), time_s=arrays, voltage_v=arrays, current_a=arrays,
            charge_c=arrays,
        )  # fmt: skip
    return EnduranceTable(
        source=source, frame=pd.DataFrame(), cycles=arrays, p_c=arrays, u_c=arrays, n_c=arrays,
        d_c=arrays,
    )  # fmt: skip


class TestGroupingRuns:
    def test_two_run_files_give_a_pristine_and_a_post_cycling_state(self):
        runs = [
            _run("PUND_pris_1e6_R9C9_30um.xls", "Data"),
            _run("PUND_pris_1e6_R9C9_30um.xls", "Append1"),
            _run("PV_pris_1e6_R9C9_30um.xls", "Data"),
            _run("PV_pris_1e6_R9C9_30um.xls", "Append1"),
            _run("END_1e6_R9C9_30um.xls"),
        ]
        groups = group_runs(runs)
        assert [
            (g.state, g.pund is not None, g.pv is not None, g.endurance is not None) for g in groups
        ] == [
            ("post1e6", True, True, False),
            ("pris", True, True, True),
        ]
        assert all(g.capacitor == CapacitorId(row=9, col=9) for g in groups)
        assert all(g.issues == () for g in groups)

    def test_a_post_cycling_file_without_a_state_token_is_its_own_state(self):
        groups = group_runs([_run("PUND_pris_R9C11_30um.xls"), _run("PV_1e6_R9C11_30um.xls")])
        assert [(g.state, [i.code for i in g.issues]) for g in groups] == [
            ("post1e6", [IssueCode.UNPAIRED_PUND]),
            ("pris", [IssueCode.UNPAIRED_PV]),
        ]

    def test_an_append_run_without_a_cycle_count_takes_it_from_the_endurance_log(self):
        runs = [
            _run("PUND_pris_R9C11_30um.xls", "Data"),
            _run("PUND_pris_R9C11_30um.xls", "Append1"),
            _run("END_1e6_R9C11_30um.xls"),
        ]
        assert [state for _, state in run_states(runs)] == ["pris", "post1e6", "endurance1e6"]

    def test_pund_before_the_presetting_sweep_is_flagged_on_the_pristine_state_only(self):
        early, late = datetime(2026, 9, 10, 9, 0), datetime(2026, 9, 10, 9, 5)
        runs = [
            _run("PUND_pris_1e6_R9C9_30um.xls", "Data", early),
            _run("PV_pris_1e6_R9C9_30um.xls", "Data", late),
            _run("PUND_pris_1e6_R9C9_30um.xls", "Append1", early),
            _run("PV_pris_1e6_R9C9_30um.xls", "Append1", late),
        ]
        by_state = {g.state: [i.code for i in g.issues] for g in group_runs(runs)}
        assert by_state == {"pris": [IssueCode.PUND_BEFORE_PV], "post1e6": []}

    def test_a_suspect_duplicate_loses_to_the_tidy_file_and_is_reported(self):
        groups = group_runs(
            [_run("PUND_pris_1e6_R9C8_30um_x.xls"), _run("PUND_pris_1e6_R9C8_30um.xls")]
        )
        (group,) = groups
        assert group.pund.source.file.path.name == "PUND_pris_1e6_R9C8_30um.xls"
        assert {i.code for i in group.issues} == {IssueCode.UNPAIRED_PV, IssueCode.DUPLICATE_RUN}

    def test_two_tidy_files_for_one_run_stop_the_round(self):
        with pytest.raises(RoundLayoutError):
            group_runs(
                [
                    _run("PUND_pris_1e6_R9C8_30um.xls"),
                    _run("PUND_pris_1e6_R9C8_30um.xls", folder="Sample16"),
                ]
            )

    def test_a_folder_without_a_sample_number_is_reported(self):
        (group,) = group_runs([_run("PUND_pris_R9C9_30um.xls", folder="Pristine Sample")])
        assert group.sample_key == "pristinesample"
        assert IssueCode.NO_SAMPLE_NUMBER in [i.code for i in group.issues]
