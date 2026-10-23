"""Per-round tables are written once; the campaign table merges rounds by key, atomically."""

import pandas as pd
import pytest

from electrical.accumulate import (
    TABLE_COLUMNS,
    guard_formulas,
    merge_round,
    read_campaign,
    update_campaign,
    write_table,
)
from electrical.issues import ContractError


def _rows(
    round_label: int, sample: str, caps: list[str], state: str = "pris", value: float = 1.0
) -> pd.DataFrame:
    frame = pd.DataFrame({c: [None] * len(caps) for c in TABLE_COLUMNS})
    frame["round"] = round_label
    frame["sample"] = sample
    frame["sample_key"] = sample.lower()
    frame["sample_num"] = int(sample.replace("Sample", ""))
    frame["cap"] = caps
    frame["row"] = [int(c[1]) for c in caps]
    frame["col"] = [int(c[3:]) for c in caps]
    frame["state"] = state
    frame["psw_uc_cm2"] = value
    frame["outlier_flags"] = ""
    return frame


class TestWritingTables:
    def test_a_table_is_written_once_and_not_replaced_by_accident(self, tmp_path):
        path = write_table(tmp_path / "out" / "t.csv", _rows(2, "Sample16", ["R9C9"]))
        assert path.exists()
        with pytest.raises(FileExistsError):
            write_table(path, _rows(2, "Sample16", ["R9C9"]))
        write_table(path, _rows(2, "Sample16", ["R9C8"]), overwrite=True)
        assert pd.read_csv(path)["cap"].tolist() == ["R9C8"]

    def test_formula_like_text_is_quoted(self):
        frame = pd.DataFrame(
            {"sample": ["=cmd()", "Sample16", "-1+1"], "psw_uc_cm2": [-1.0, 2.0, 3.0]}
        )
        guarded = guard_formulas(frame)
        assert guarded["sample"].tolist() == ["'=cmd()", "Sample16", "'-1+1"]
        assert guarded["psw_uc_cm2"].tolist() == [-1.0, 2.0, 3.0]


class TestTheCampaignTable:
    def test_a_rerun_round_replaces_its_own_rows_and_keeps_the_others(self, tmp_path):
        path = tmp_path / "campaign.csv"
        update_campaign(path, _rows(2, "Sample16", ["R9C9", "R9C8"], value=1.0))
        update_campaign(path, _rows(3, "Sample20", ["R9C9"], value=5.0))
        merged = update_campaign(path, _rows(2, "Sample16", ["R9C9"], value=2.0))
        table = read_campaign(path)
        assert len(table) == len(merged) == 3
        by_key = table.set_index(["round", "sample_key", "cap"])["psw_uc_cm2"]
        assert by_key[(2, "sample16", "R9C9")] == 2.0
        assert by_key[(2, "sample16", "R9C8")] == 1.0
        assert by_key[(3, "sample20", "R9C9")] == 5.0

    def test_pristine_and_cycled_rows_of_one_capacitor_are_distinct(self):
        existing = _rows(2, "Sample16", ["R9C9"], state="pris")
        merged = merge_round(existing, _rows(2, "Sample16", ["R9C9"], state="post1e6"))
        assert sorted(merged["state"]) == ["post1e6", "pris"]

    def test_a_table_with_other_columns_is_refused(self, tmp_path):
        path = tmp_path / "campaign.csv"
        pd.DataFrame({"round": [2], "sample": ["Sample16"]}).to_csv(path, index=False)
        with pytest.raises(ContractError):
            read_campaign(path)

    def test_writing_leaves_no_temporary_file_behind(self, tmp_path):
        path = tmp_path / "campaign.csv"
        update_campaign(path, _rows(2, "Sample16", ["R9C9"]))
        assert [p.name for p in tmp_path.iterdir()] == ["campaign.csv"]
