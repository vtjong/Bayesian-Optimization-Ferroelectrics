"""Findings collected while reading one round, and the errors that stop a run outright.

A finding never raises on its own: it is attached to the record it concerns (a run, a
capacitor, a sample) and reaches the operator through the tables and the QC report, so one
pass over a round reports every problem at once. Only a problem that makes the round
unprocessable, such as a folder with no samples, is an exception.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Severity(Enum):
    ERROR = "error"  # the affected value is missing or not to be trusted
    WARNING = "warning"  # the value stands, but the operator should look
    INFO = "info"


class IssueCode(Enum):
    UNREADABLE_WORKBOOK = "unreadable_workbook"
    MISSING_SHEET = "missing_sheet"
    MISSING_COLUMN = "missing_column"
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
    UNRELIABLE_ON_SWITCHING = "unreliable_on_switching"
    ENDURANCE_MISMATCH = "endurance_mismatch"
    FEATURE_FAILURE = "feature_failure"
    OUTLIER = "outlier"


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationIssue:
    """One finding, with enough of its location to go and look at the file.

    :param severity: Whether the affected value is lost (error) or merely doubtful (warning).
    :param code: Machine-readable kind of finding.
    :param message: One sentence for the operator.
    :param sample_key: Normalised sample the finding belongs to, if any.
    :param capacitor: Capacitor label such as ``"R9C9"``, if any.
    :param state: Measurement state such as ``"pris"``, if any.
    :param path: Workbook the finding came from, if any.
    :param location: Sheet, row or pulse inside that workbook, if any.
    """

    severity: Severity
    code: IssueCode
    message: str
    sample_key: str | None = None
    capacitor: str | None = None
    state: str | None = None
    path: Path | None = None
    location: str | None = None

    def describe(self) -> str:
        """Return the finding as ``CODE: message``."""
        return f"{self.code.name}: {self.message}"


def error(code: IssueCode, message: str, **where) -> ValidationIssue:
    """Build an error finding; ``where`` names its sample, capacitor, state, path or location."""
    return ValidationIssue(severity=Severity.ERROR, code=code, message=message, **where)


def warning(code: IssueCode, message: str, **where) -> ValidationIssue:
    """Build a warning finding; ``where`` names its sample, capacitor, state, path or location."""
    return ValidationIssue(severity=Severity.WARNING, code=code, message=message, **where)


def errors(issues: Iterable[ValidationIssue]) -> tuple[ValidationIssue, ...]:
    """Return only the findings that lose a value."""
    return tuple(issue for issue in issues if issue.severity is Severity.ERROR)


def has_error(issues: Iterable[ValidationIssue]) -> bool:
    """Return whether any finding is an error."""
    return any(issue.severity is Severity.ERROR for issue in issues)


def describe_issues(issues: Iterable[ValidationIssue]) -> str:
    """Return every finding as one ``CODE: message; CODE: message`` string, empty when clean."""
    return "; ".join(issue.describe() for issue in issues)


class ElectricalError(Exception):
    """A problem that stops the round rather than one capacitor."""


class RoundLayoutError(ElectricalError):
    """The data folder is not one round of ``Sample*/`` folders, or two files claim one run."""


class UnreadableWorkbookError(ElectricalError, ValueError):
    """The spreadsheet library could not open or parse the workbook."""

    def __init__(self, path: Path, cause: BaseException) -> None:
        super().__init__(f"{path}: {type(cause).__name__}: {cause}")
        self.path = path


class WorkbookShapeError(ElectricalError, ValueError):
    """The workbook opened, but a sheet or column the export always has is missing."""

    def __init__(self, path: Path, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path


class FeatureExtractionError(ElectricalError):
    """A run could not be reduced to features; carries the code the issue table uses."""

    def __init__(self, code: IssueCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class ContractError(ElectricalError):
    """A table on disk does not have the columns this version of the pipeline writes."""
