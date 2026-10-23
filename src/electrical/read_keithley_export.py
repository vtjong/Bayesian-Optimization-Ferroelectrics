"""Read a Keithley 4200A export: its Settings blocks and its run sheets, opened once.

A workbook is a Settings sheet plus one data sheet per run. The Settings sheet has one block
per run (``Latest Run``, ``Initial Run`` or ``Append k``) with the KTEI module name, the
``Last Executed`` timestamp and the instrument's input parameters. The first run's data is
sheet ``Data``; a run appended later is ``Append1``. Columns are matched by header label and
read as numbers; a cell that is not a number becomes NaN and an issue, and its row is kept.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from electrical.flag_issues import Issue, IssueCode, warning

SETTINGS_SHEET = "Settings"
DATA_SHEET = "Data"
APPEND_SHEET_PREFIX = "Append"
#: ``Last Executed`` is written as month/day/year.
TIMESTAMP_FORMAT = "%m/%d/%Y %H:%M:%S"


class UnreadableWorkbookError(ValueError):
    """The spreadsheet library could not open or parse the workbook."""

    def __init__(self, path: Path, cause: BaseException) -> None:
        super().__init__(f"{path}: {type(cause).__name__}: {cause}")
        self.path = path


class WorkbookShapeError(ValueError):
    """The workbook opened, but a sheet or column the export always has is missing."""

    def __init__(self, path: Path | None, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path


@dataclass(frozen=True, slots=True, kw_only=True)
class InstrumentSettings:
    """One run's block of the Settings sheet.

    :param module_name: KTEI module, such as ``"pundTest"``.
    :param executed_at: ``Last Executed`` parsed, or ``None`` if absent or unparsable.
    :param executed_raw: ``Last Executed`` as written.
    :param inputs: Instrument input parameters by name, as written.
    """

    module_name: str | None
    executed_at: datetime | None
    executed_raw: str | None
    inputs: Mapping[str, str]


#: Settings of a run sheet that has no block of its own.
NO_SETTINGS = InstrumentSettings(module_name=None, executed_at=None, executed_raw=None, inputs={})


@dataclass(frozen=True, slots=True, kw_only=True)
class KeithleyExport:
    """One workbook: the settings of each run sheet and the run sheets, ``Data`` first."""

    path: Path
    settings: Mapping[str, InstrumentSettings]
    sheets: Mapping[str, pd.DataFrame]


@dataclass(frozen=True, slots=True, kw_only=True)
class ColumnSpec:
    """A data-sheet column, matched by header label or one of the alternatives seen for it."""

    label: str
    alternatives: tuple[str, ...] = ()

    def find(self, frame: pd.DataFrame) -> str | None:
        """Return the frame's column holding this quantity, or ``None``."""
        for name in (self.label, *self.alternatives):
            if name in frame.columns:
                return name
        return None


def read_keithley_export(path: Path) -> KeithleyExport:
    """Open one export once and return its settings blocks and run sheets.

    :raises UnreadableWorkbookError: If the library cannot open or read the workbook.
    :raises WorkbookShapeError: If the Settings or Data sheet is missing.
    """
    try:
        book = pd.ExcelFile(path, engine="xlrd")
        names = list(book.sheet_names)
        if SETTINGS_SHEET not in names or DATA_SHEET not in names:
            raise WorkbookShapeError(path, f"sheets {names} lack {SETTINGS_SHEET} or {DATA_SHEET}")
        settings = parse_settings(pd.read_excel(book, sheet_name=SETTINGS_SHEET, header=None))
        runs = [DATA_SHEET, *sorted(n for n in names if n.startswith(APPEND_SHEET_PREFIX))]
        sheets = {name: pd.read_excel(book, sheet_name=name) for name in runs}
    except WorkbookShapeError:
        raise
    except Exception as exc:  # the spreadsheet library raises many kinds
        raise UnreadableWorkbookError(path, exc) from exc
    return KeithleyExport(path=path, settings=settings, sheets=sheets)


def parse_settings(frame: pd.DataFrame) -> dict[str, InstrumentSettings]:
    """Split the Settings sheet into one block per run, keyed by the sheet the run is in.

    ``Append k`` describes sheet ``Appendk``; ``Latest Run`` and ``Initial Run`` describe
    ``Data``. Inputs are the rows whose second column reads ``Input``, value in the fourth.
    """
    labels = frame.iloc[:, 0].fillna("").astype(str).str.strip()
    blocks: list[tuple[str, list[int]]] = []
    for i, label in enumerate(labels):
        if label.startswith("===="):
            continue
        if label in ("Latest Run", "Initial Run") or label.startswith(APPEND_SHEET_PREFIX):
            blocks.append((label, []))
        elif blocks:
            blocks[-1][1].append(i)

    settings = {}
    for title, rows in blocks:
        values = {labels.iloc[i]: frame.iloc[i, 1] for i in rows}
        inputs = {
            labels.iloc[i]: str(frame.iloc[i, 3])
            for i in rows
            if str(frame.iloc[i, 1]).strip() == "Input" and "size" not in labels.iloc[i]
        }
        module = values.get("Module Name")
        raw = values.get("Last Executed")
        raw = raw.strip() if isinstance(raw, str) else None
        sheet = title.replace(" ", "") if title.startswith(APPEND_SHEET_PREFIX) else DATA_SHEET
        settings[sheet] = InstrumentSettings(
            module_name=str(module).strip() if isinstance(module, str) else None,
            executed_at=parse_timestamp(raw),
            executed_raw=raw,
            inputs=inputs,
        )
    return settings


def parse_timestamp(raw: str | None) -> datetime | None:
    """Parse ``Last Executed``; ``None`` when absent or not in the instrument's format."""
    if raw is None:
        return None
    try:
        return datetime.strptime(raw, TIMESTAMP_FORMAT)
    except ValueError:
        return None


def check_settings(settings: InstrumentSettings, expected_module: str, **where) -> list[Issue]:
    """Report a run recorded by an unexpected module or with an unreadable timestamp."""
    issues = []
    if settings.module_name != expected_module:
        issues.append(
            warning(
                IssueCode.UNEXPECTED_MODULE,
                f"module {settings.module_name!r}, expected {expected_module!r}",
                **where,
            )
        )
    if settings.executed_raw and settings.executed_at is None:
        issues.append(
            warning(
                IssueCode.UNPARSED_TIMESTAMP,
                f"Last Executed {settings.executed_raw!r} is not {TIMESTAMP_FORMAT}",
                **where,
            )
        )
    return issues


def read_numeric_column(
    frame: pd.DataFrame, spec: ColumnSpec, where: Mapping[str, object]
) -> tuple[np.ndarray, list[Issue]]:
    """Read one column as floats; blank or non-numeric cells become NaN and one issue.

    :raises WorkbookShapeError: If no header matches the spec.
    """
    name = spec.find(frame)
    if name is None:
        sheet = where.get("sheet")
        raise WorkbookShapeError(where.get("path"), f"sheet {sheet} has no {spec.label!r} column")
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
    bad = np.flatnonzero(~np.isfinite(values))
    if len(bad) == 0:
        return values, []
    issue = warning(
        IssueCode.BAD_CELL,
        f"{len(bad)} cell(s) of {name!r} are blank or not numbers, first at row {bad[0] + 2}",
        **where,
        column=name,
    )
    return values, [issue]


def read_first_row_value(frame: pd.DataFrame, name: str) -> float:
    """Return a value the instrument writes once per run on the first data row, or NaN."""
    if name not in frame.columns:
        return np.nan
    return float(pd.to_numeric(frame[name].iloc[0], errors="coerce"))
