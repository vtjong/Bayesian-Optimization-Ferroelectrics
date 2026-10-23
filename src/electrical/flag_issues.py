"""Record the problems found in a round's measurements and flag capacitors that are outliers.

Issues are found while reading and reducing the exports: a missing P-V run, a hand-typed file
name, a split rule that found no shoulder. Each carries a code and its location and travels on
the row it concerns, in the ``issues`` column. An error means the affected values are lost; a
warning means they stand but deserve a look.

Outliers are capacitors far from the rest of their sample and state on a headline feature,
flagged in ``outlier_flags`` by robust z = |x - median| / (1.4826 MAD), so one wild capacitor
cannot hide itself by inflating the spread it is judged against.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

#: Features checked for outliers.
OUTLIER_COLUMNS = ("pr_uc_cm2", "kappa_summary", "psw_uc_cm2", "kappa_pund", "j_leak_a_cm2")
#: A capacitor is an outlier when its robust z exceeds this.
OUTLIER_ROBUST_Z = 3.5
#: Scale that makes the median absolute deviation a standard deviation for normal data.
MAD_TO_SD = 1.4826
#: Column the outlier flags are written to.
OUTLIER_FLAGS_COLUMN = "outlier_flags"
#: Outliers are judged within one sample in one state of one round.
SAMPLE_STATE = ("round", "sample", "state")


class Severity(Enum):
    ERROR = "error"  # the affected values are lost
    WARNING = "warning"  # the values stand, but the operator should look


class IssueCode(Enum):
    UNREADABLE_WORKBOOK = "unreadable_workbook"
    BAD_CELL = "bad_cell"
    UNEXPECTED_MODULE = "unexpected_module"
    UNPARSED_TIMESTAMP = "unparsed_timestamp"
    SUSPECT_FILE_NAME = "suspect_file_name"
    DUPLICATE_RUN = "duplicate_run"
    NO_SAMPLE_NUMBER = "no_sample_number"
    UNPAIRED_PUND = "unpaired_pund"
    UNPAIRED_PV = "unpaired_pv"
    PUND_BEFORE_PV = "pund_before_pv"
    UNEXPECTED_PULSE_PATTERN = "unexpected_pulse_pattern"
    SWEEP_NOT_POSITIVE_FIRST = "sweep_not_positive_first"
    ENDURANCE_TOO_SHORT = "endurance_too_short"
    SPLIT_NOT_FOUND = "split_not_found"
    TIMING_NOT_IN_SETTINGS = "timing_not_in_settings"
    FEATURE_FAILURE = "feature_failure"


@dataclass(frozen=True, slots=True, kw_only=True)
class Issue:
    """One problem, with enough of its location to go and look.

    :param severity: Whether the affected values are lost (error) or doubtful (warning).
    :param code: What kind of problem it is.
    :param message: One sentence for the operator.
    :param where: Location: sample, capacitor, state, file, sheet, column.
    """

    severity: Severity
    code: IssueCode
    message: str
    where: Mapping[str, object] = field(default_factory=dict)

    def describe(self) -> str:
        """Return the issue as ``CODE: message``."""
        return f"{self.code.name}: {self.message}"


def error(code: IssueCode, message: str, **where) -> Issue:
    """Record a problem that loses the affected values."""
    return Issue(severity=Severity.ERROR, code=code, message=message, where=where)


def warning(code: IssueCode, message: str, **where) -> Issue:
    """Record a problem the operator should look at; the values stand."""
    return Issue(severity=Severity.WARNING, code=code, message=message, where=where)


def has_error(issues: Iterable[Issue]) -> bool:
    """Return whether any issue is an error."""
    return any(issue.severity is Severity.ERROR for issue in issues)


def describe_issues(issues: Iterable[Issue]) -> str:
    """Return every issue as ``CODE: message; CODE: message``, empty when there are none."""
    return "; ".join(issue.describe() for issue in issues)


def flag_outliers(capacitors: pd.DataFrame) -> pd.DataFrame:
    """Return the per-capacitor table with ``outlier_flags`` added.

    Each flag names the feature and its robust z, such as ``pr_uc_cm2(z=5.3);``. Features with
    no spread in a sample flag nothing there.
    """
    capacitors = capacitors.copy()
    keys = [capacitors[column] for column in SAMPLE_STATE]
    flags = pd.Series("", index=capacitors.index, dtype=object)
    for column in OUTLIER_COLUMNS:
        if column not in capacitors.columns:
            continue
        x = pd.to_numeric(capacitors[column], errors="coerce")
        deviation = (x - x.groupby(keys).transform("median")).abs()
        scale = deviation.groupby(keys).transform("median") * MAD_TO_SD
        z = (deviation / scale).where(scale > 0)
        outlier = z > OUTLIER_ROBUST_Z
        flags = flags.where(~outlier, flags + column + "(z=" + z.round(1).astype(str) + ");")
    capacitors[OUTLIER_FLAGS_COLUMN] = flags
    return capacitors
