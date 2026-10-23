"""Issues are described and counted; outliers are judged by robust z within a sample and state."""

import numpy as np
import pandas as pd

from electrical.flag_issues import (
    MAD_TO_SD,
    OUTLIER_ROBUST_Z,
    IssueCode,
    describe_issues,
    error,
    flag_outliers,
    has_error,
    warning,
)


def _capacitors(values, sample="Sample16", state="pris") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "round": 2,
            "sample": sample,
            "state": state,
            "cap": [f"R9C{i}" for i in range(len(values))],
            "pr_uc_cm2": values,
            "psw_uc_cm2": values,
            "kappa_summary": 40.0,
        }
    )


def test_issues_describe_themselves_and_only_errors_count_as_errors():
    issues = [warning(IssueCode.UNPAIRED_PV, "no PV run"), error(IssueCode.FEATURE_FAILURE, "boom")]
    assert describe_issues(issues) == "UNPAIRED_PV: no PV run; FEATURE_FAILURE: boom"
    assert has_error(issues) and not has_error(issues[:1])
    assert describe_issues([]) == ""


def test_one_wild_capacitor_is_flagged_on_the_features_it_breaks():
    flagged = flag_outliers(_capacitors([8.0, 8.1, 7.9, 8.2, 8.0, 7.8, 8.1, 20.0]))
    assert (flagged["outlier_flags"] != "").tolist() == [False] * 7 + [True]
    assert flagged.loc[7, "outlier_flags"].startswith("pr_uc_cm2(z=")
    assert "psw_uc_cm2(z=" in flagged.loc[7, "outlier_flags"]


def test_a_feature_with_no_spread_flags_nothing():
    assert (flag_outliers(_capacitors([8.0] * 6))["outlier_flags"] == "").all()


def test_outliers_are_judged_within_each_sample_and_state():
    both = pd.concat(
        [
            _capacitors([8.0, 8.1, 7.9, 8.2], state="pris"),
            _capacitors([16.0, 16.1, 15.9, 16.2], state="post1e6"),
        ],
        ignore_index=True,
    )
    assert (flag_outliers(both)["outlier_flags"] == "").all()


def test_a_capacitor_just_inside_the_threshold_is_not_flagged():
    base = [8.0, 8.1, 7.9, 8.2, 8.0, 7.8, 8.1]
    median = np.median(base)
    spread = np.median(np.abs(np.array(base) - median)) * MAD_TO_SD
    inside = _capacitors(base + [median + 0.9 * OUTLIER_ROBUST_Z * spread])
    assert (flag_outliers(inside)["outlier_flags"] == "").all()
