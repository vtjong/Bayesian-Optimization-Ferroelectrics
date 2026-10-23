"""The QC page escapes what it shows, has one section per sample and state, and marks rows."""

import numpy as np
import pandas as pd
import pytest

from electrical.flag_issues import flag_outliers
from electrical.process_round import RoundResult
from electrical.write_qc_report import _cell, build_qc_page


def _capacitors(values, sample="Sample16", state="pris") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "round": 2,
            "sample": sample,
            "state": state,
            "cap": [f"R9C{i}" for i in range(len(values))],
            "switching": [v > 2 for v in values],
            "pr_uc_cm2": values,
            "psw_uc_cm2": values,
            "pulse_pattern": "PUND",
            "issues": None,
        }
    )


def _result(capacitors: pd.DataFrame) -> RoundResult:
    capacitors = flag_outliers(capacitors)
    samples = (
        capacitors.groupby(["sample", "state"], sort=False)
        .size()
        .rename("n_capacitors")
        .reset_index()
    )
    samples["n_switching"] = 0
    raw_index = pd.DataFrame(
        {
            "sample": ["<b>x</b>"],
            "cap": ["R9C9"],
            "kind": ["PUND"],
            "problems": ["BAD_CELL: <script>"],
        }
    )
    return RoundResult(
        raw_index=raw_index,
        capacitors=capacitors,
        samples=samples,
        traces={},
        endurance_curves=pd.DataFrame(),
        issues=(),
    )


def test_every_value_from_the_data_is_escaped():
    page = build_qc_page(
        _result(_capacitors([8.0, 8.1, 7.9, 8.2], sample="<script>alert(1)</script>"))
    )
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page and "BAD_CELL: &lt;script&gt;" in page


def test_there_is_one_section_per_sample_and_state():
    both = pd.concat(
        [
            _capacitors([8.0, 8.1, 7.9, 8.2], state="pris"),
            _capacitors([16.0, 16.1, 15.9, 16.2], state="post1e6"),
        ],
        ignore_index=True,
    )
    page = build_qc_page(_result(both))
    assert page.count("<h2>Sample16 — pris") == 1 and page.count("<h2>Sample16 — post1e6") == 1


def test_switching_and_outlying_rows_are_marked():
    page = build_qc_page(_result(_capacitors([8.0, 8.1, 7.9, 8.2, 8.0, 7.8, 8.1, 20.0])))
    assert "<tr class='flag'>" in page and "<tr class='sw'>" in page


@pytest.mark.parametrize(
    ("value", "rendered"),
    [
        (np.nan, "<td>–</td>"),
        (1.23456e-7, "<td>1.235e-07</td>"),
        (True, "<td class='l'>yes</td>"),
        (None, "<td class='l'></td>"),
    ],
)
def test_cells_render_numbers_blanks_and_flags(value, rendered):
    assert _cell(value) == rendered
