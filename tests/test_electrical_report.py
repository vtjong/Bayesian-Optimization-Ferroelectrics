"""Outlier flags and the QC page, on a small synthetic round."""

import numpy as np
import pandas as pd
import pytest

from electrical.pipeline import RoundResult
from electrical.report import OUTLIER_ROBUST_Z, build_report, flag_outliers


def _capacitors(values, sample="Sample16", state="pris") -> pd.DataFrame:
    n = len(values)
    return pd.DataFrame(
        {
            "round": 2,
            "sample": sample,
            "sample_key": sample.lower(),
            "sample_num": 16,
            "cap": [f"R9C{i}" for i in range(n)],
            "row": 9,
            "col": range(n),
            "state": state,
            "switching": [v > 2 for v in values],
            "pr_uc_cm2": values,
            "kappa_summary": 40.0,
            "psw_uc_cm2": values,
            "kappa_pund": 50.0,
            "j_leak_a_cm2": 1e-6,
            "pulse_pattern": "PUND",
            "issues": None,
            "has_error": False,
        }
    )


class TestOutlierFlags:
    def test_one_wild_capacitor_is_flagged_on_the_features_it_breaks(self):
        flagged = flag_outliers(_capacitors([8.0, 8.1, 7.9, 8.2, 8.0, 7.8, 8.1, 20.0]))
        assert flagged["outlier_flags"].str.len().gt(0).tolist() == [False] * 7 + [True]
        assert flagged.loc[7, "outlier_flags"].startswith("pr_uc_cm2(z=")
        assert "psw_uc_cm2(z=" in flagged.loc[7, "outlier_flags"]

    def test_a_feature_with_no_spread_flags_nothing(self):
        flagged = flag_outliers(_capacitors([8.0] * 6))
        assert (flagged["outlier_flags"] == "").all()

    def test_flags_are_judged_within_each_sample_state(self):
        pristine = _capacitors([8.0, 8.1, 7.9, 8.2, 8.0, 8.1], state="pris")
        woken = _capacitors([16.0, 16.1, 15.9, 16.2, 16.0, 16.1], state="post1e6")
        flagged = flag_outliers(pd.concat([pristine, woken], ignore_index=True))
        assert (flagged["outlier_flags"] == "").all()

    def test_the_threshold_is_the_robust_z(self):
        base = [8.0, 8.1, 7.9, 8.2, 8.0, 7.8, 8.1]
        median = np.median(base)
        mad = np.median(np.abs(np.array(base) - median)) * 1.4826
        just_inside = _capacitors(base + [median + 0.9 * OUTLIER_ROBUST_Z * mad])
        assert (flag_outliers(just_inside)["outlier_flags"] == "").all()


def _result(capacitors: pd.DataFrame) -> RoundResult:
    samples = pd.DataFrame(
        {
            "sample": capacitors["sample"].unique(),
            "state": "pris",
            "n_capacitors": len(capacitors),
            "n_switching": int(capacitors["switching"].sum()),
            "pr_uc_cm2_mean": capacitors["pr_uc_cm2"].mean(),
            "pr_uc_cm2_sd": capacitors["pr_uc_cm2"].std(),
            "inhomogeneous": False,
        }
    )
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


class TestThePage:
    def test_every_value_from_the_data_is_escaped(self):
        capacitors = flag_outliers(
            _capacitors([8.0, 8.1, 7.9, 8.2], sample="<script>alert(1)</script>")
        )
        page = build_report(_result(capacitors), capacitors)
        assert "<script>alert(1)</script>" not in page
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
        assert "BAD_CELL: &lt;script&gt;" in page

    def test_one_section_per_sample_state_with_its_capacitors(self):
        capacitors = flag_outliers(
            pd.concat(
                [
                    _capacitors([8.0, 8.1, 7.9, 8.2], state="pris"),
                    _capacitors([16.0, 16.1, 15.9, 16.2], state="post1e6"),
                ],
                ignore_index=True,
            )
        )
        page = build_report(_result(capacitors), capacitors)
        assert page.count("<h2>Sample16 — pris") == 1 and page.count("<h2>Sample16 — post1e6") == 1
        assert page.count("R9C3") == 2

    def test_switching_and_flagged_rows_are_marked(self):
        capacitors = flag_outliers(_capacitors([8.0, 8.1, 7.9, 8.2, 8.0, 7.8, 8.1, 20.0]))
        page = build_report(_result(capacitors), capacitors)
        assert "<tr class='flag'>" in page and "<tr class='sw'>" in page

    @pytest.mark.parametrize(
        "value, rendered",
        [
            (np.nan, "<td>–</td>"),
            (1.23456e-7, "<td>1.235e-07</td>"),
            (True, "<td class='l'>yes</td>"),
        ],
    )
    def test_cells_render_numbers_blanks_and_flags(self, value, rendered):
        from electrical.report import _cell

        assert _cell(value) == rendered
