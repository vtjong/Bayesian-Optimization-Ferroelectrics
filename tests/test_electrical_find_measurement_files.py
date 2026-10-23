"""File names, sample keys and the search for exports; no workbook is opened."""

from pathlib import Path

import pytest

from electrical.find_measurement_files import (
    MeasurementKind,
    RoundLayoutError,
    find_measurement_files,
    parse_file_name,
    sample_key,
)


@pytest.mark.parametrize(
    ("name", "kind", "state", "cycles", "capacitor", "suspect"),
    [
        ("PUND_pris_R9C9_30um.xls", MeasurementKind.PUND, "pris", None, "R9C9", False),
        ("PV_pris_1e6_R10C12_30um.xls", MeasurementKind.PV, "pris", "1e6", "R10C12", False),
        ("END_1e6_R8C8_30um.xls", MeasurementKind.END, None, "1e6", "R8C8", False),
        ("PUND_1e6_R9C10_30um.xls", MeasurementKind.PUND, None, "1e6", "R9C10", True),
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
    assert (parsed.kind, parsed.state_token, parsed.cycles_tag) == (kind, state, cycles)
    assert parsed.capacitor.label == capacitor and parsed.side_um == 30
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


def test_exports_are_found_in_sample_folders_and_plots_are_skipped(tmp_path):
    (tmp_path / "Sample16" / "Analysis_Plots").mkdir(parents=True)
    (tmp_path / "Sample16" / "PUND_pris_1e6_R9C9_30um.xls").touch()
    (tmp_path / "Sample16" / "PUND_pris_1e6_R9C9_30um.bmp").touch()
    (tmp_path / "Sample16" / "Analysis_Plots" / "PUND_pris_1e6_R9C8_30um.xls").touch()
    (tmp_path / "Sample 15").mkdir()
    (tmp_path / "Sample 15" / "PV_pris_R8C8_30um.xls").touch()
    found = find_measurement_files(tmp_path)
    assert [(f.sample_key, f.path.name) for f in found] == [
        ("sample15", "PV_pris_R8C8_30um.xls"),
        ("sample16", "PUND_pris_1e6_R9C9_30um.xls"),
    ]


def test_a_folder_without_exports_is_a_layout_error(tmp_path):
    (tmp_path / "Sample1").mkdir()
    (tmp_path / "Sample1" / "PUND_R9C9.png").touch()
    with pytest.raises(RoundLayoutError):
        find_measurement_files(tmp_path)
