"""Tables are written once or atomically, quoted against formulas, read back against a contract."""

import pandas as pd
import pytest

from utils.file_io import (
    ContractError,
    guard_formulas,
    read_csv_with_columns,
    write_csv,
    write_csv_atomic,
)


def test_a_table_is_written_once_and_replaced_only_when_asked(tmp_path):
    path = write_csv(tmp_path / "out" / "t.csv", pd.DataFrame({"cap": ["R9C9"]}))
    with pytest.raises(FileExistsError):
        write_csv(path, pd.DataFrame({"cap": ["R9C9"]}))
    write_csv(path, pd.DataFrame({"cap": ["R9C8"]}), overwrite=True)
    assert pd.read_csv(path)["cap"].tolist() == ["R9C8"]


def test_formula_like_text_is_quoted_and_numbers_are_left_alone():
    frame = pd.DataFrame({"sample": ["=cmd()", "Sample16", "-1+1"], "psw_uc_cm2": [-1.0, 2.0, 3.0]})
    guarded = guard_formulas(frame)
    assert guarded["sample"].tolist() == ["'=cmd()", "Sample16", "'-1+1"]
    assert guarded["psw_uc_cm2"].tolist() == [-1.0, 2.0, 3.0]


def test_an_atomic_write_leaves_no_temporary_file(tmp_path):
    write_csv_atomic(tmp_path / "campaign.csv", pd.DataFrame({"a": [1]}))
    write_csv_atomic(tmp_path / "campaign.csv", pd.DataFrame({"a": [2]}))
    assert [p.name for p in tmp_path.iterdir()] == ["campaign.csv"]
    assert pd.read_csv(tmp_path / "campaign.csv")["a"].tolist() == [2]


def test_reading_back_returns_the_contract_order(tmp_path):
    pd.DataFrame({"b": [2], "a": [1]}).to_csv(tmp_path / "t.csv", index=False)
    assert list(read_csv_with_columns(tmp_path / "t.csv", ("a", "b")).columns) == ["a", "b"]


@pytest.mark.parametrize("columns", [("a",), ("a", "b", "c")])
def test_a_table_with_other_columns_is_refused(tmp_path, columns):
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(tmp_path / "t.csv", index=False)
    with pytest.raises(ContractError):
        read_csv_with_columns(tmp_path / "t.csv", columns)
