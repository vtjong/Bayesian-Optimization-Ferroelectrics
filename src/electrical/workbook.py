"""Read one Keithley 4200A export once and hand back its runs as typed records.

A workbook has a ``Settings`` sheet (one block per run: module name, timestamp, instrument
inputs) and one data sheet per run: ``Data`` for the first run and ``Append1`` for a run
appended later. Columns are matched by their header label, every cell that fails to parse is
reported and kept as NaN, and a workbook the library cannot open or that lacks a sheet the
export always has raises, so the pipeline can record it and move on.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from electrical.issues import (
    IssueCode,
    UnreadableWorkbookError,
    ValidationIssue,
    WorkbookShapeError,
    describe_issues,
    warning,
)
from electrical.locate import MeasurementFile, MeasurementKind

SETTINGS_SHEET = "Settings"
DATA_SHEET = "Data"
APPEND_SHEET_PREFIX = "Append"
#: The instrument writes ``Last Executed`` as month/day/year.
TIMESTAMP_FORMAT = "%m/%d/%Y %H:%M:%S"
#: KTEI module each export kind comes from.
EXPECTED_MODULE = {
    MeasurementKind.PUND: "pundTest",
    MeasurementKind.PV: "doubleSweepSeg",
    MeasurementKind.END: "pundEndurance",
}


@dataclass(frozen=True, slots=True, kw_only=True)
class ColumnSpec:
    """A data-sheet column, matched by header label.

    :param label: Header the instrument writes.
    :param alternatives: Other headers seen for the same quantity.
    :param required: Whether a run without it cannot be reduced.
    """

    label: str
    alternatives: tuple[str, ...] = ()
    required: bool = True

    def find(self, frame: pd.DataFrame) -> str | None:
        """Return the frame's column holding this quantity, or ``None``."""
        for name in (self.label, *self.alternatives):
            if name in frame.columns:
                return name
        return None


PUND_TIME = ColumnSpec(label="t", alternatives=("Time",))
PUND_VOLTAGE = ColumnSpec(label="V")
PUND_CURRENT = ColumnSpec(label="I")
PV_VOLTAGE = ColumnSpec(label="Vforce")
PV_CURRENT = ColumnSpec(label="Imeas")
PV_CHARGE = ColumnSpec(label="Charge")
PV_TIME = ColumnSpec(label="Time")
END_CYCLES = ColumnSpec(label="iteration")
END_P = ColumnSpec(label="P")
END_U = ColumnSpec(label="U")
END_N = ColumnSpec(label="N")
END_D = ColumnSpec(label="D")
#: Ramp-integrated charges the instrument writes on the first data row, in coulombs.
INSTRUMENT_SCALARS = ("P", "Pa", "U", "Ua", "N", "Na", "D", "Da", "Psw", "Qsw")


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


@dataclass(frozen=True, slots=True, kw_only=True)
class InstrumentCharges:
    """The instrument's own ramp-integrated charges for one PUND run, coulombs (NaN if absent)."""

    p_c: float = np.nan
    pa_c: float = np.nan
    u_c: float = np.nan
    ua_c: float = np.nan
    n_c: float = np.nan
    na_c: float = np.nan
    d_c: float = np.nan
    da_c: float = np.nan
    psw_c: float = np.nan
    qsw_c: float = np.nan


@dataclass(frozen=True, slots=True, kw_only=True)
class RunSource:
    """Where a run came from: the file, the sheet inside it and that sheet's settings block."""

    file: MeasurementFile
    sheet: str
    settings: InstrumentSettings


@dataclass(frozen=True, slots=True, kw_only=True)
class PundRun:
    """One PUND measurement: the V(t), I(t) record and the instrument's scalars."""

    source: RunSource
    frame: pd.DataFrame
    time_s: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray
    instrument: InstrumentCharges
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)

    @property
    def n_points(self) -> int:
        return len(self.time_s)

    @property
    def dt_s(self) -> float:
        return float(np.nanmedian(np.diff(self.time_s)))

    @property
    def t_end_s(self) -> float:
        return float(self.time_s[-1])


@dataclass(frozen=True, slots=True, kw_only=True)
class PvRun:
    """One P-V sweep: forced voltage, measured current and the instrument's running charge."""

    source: RunSource
    frame: pd.DataFrame
    time_s: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray
    charge_c: np.ndarray
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)

    @property
    def n_points(self) -> int:
        return len(self.time_s)

    @property
    def dt_s(self) -> float:
        return float(np.nanmedian(np.diff(self.time_s)))

    @property
    def t_end_s(self) -> float:
        return float(self.time_s[-1])


@dataclass(frozen=True, slots=True, kw_only=True)
class EnduranceTable:
    """One endurance log: ramp-integrated P, U, N, D charges at each logged cycle count."""

    source: RunSource
    frame: pd.DataFrame
    cycles: np.ndarray
    p_c: np.ndarray
    u_c: np.ndarray
    n_c: np.ndarray
    d_c: np.ndarray
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)

    @property
    def n_points(self) -> int:
        return len(self.cycles)

    @property
    def dt_s(self) -> float | None:
        return None

    @property
    def t_end_s(self) -> float | None:
        return None


Run = PundRun | PvRun | EnduranceTable


def parse_settings(frame: pd.DataFrame) -> dict[str, InstrumentSettings]:
    """Split the Settings sheet into one block per run sheet.

    Blocks are separated by ``====`` lines and titled ``Latest Run``, ``Initial Run`` or
    ``Append k`` (``Append k`` is sheet ``Appendk``; the other two are sheet ``Data``). Input
    parameters are the rows whose second column reads ``Input``, value in the fourth column.

    :param frame: The Settings sheet read without a header row.
    :return: Settings keyed by the sheet name they describe.
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
            executed_at=_parse_timestamp(raw),
            executed_raw=raw,
            inputs=inputs,
        )
    return settings


def _parse_timestamp(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    try:
        return datetime.strptime(raw, TIMESTAMP_FORMAT)
    except ValueError:
        return None


def read_workbook(file: MeasurementFile) -> list[Run]:
    """Open one export once and return every run it holds, ``Data`` first.

    :param file: The export, as ``locate.parse_file_name`` described it.
    :return: One run per data sheet.
    :raises UnreadableWorkbookError: If the library cannot open or read the workbook.
    :raises WorkbookShapeError: If a sheet or required column is missing.
    """
    try:
        book = pd.ExcelFile(file.path, engine="xlrd")
        sheets = list(book.sheet_names)
        if SETTINGS_SHEET not in sheets or DATA_SHEET not in sheets:
            raise WorkbookShapeError(
                file.path, f"sheets {sheets} lack {SETTINGS_SHEET} or {DATA_SHEET}"
            )
        settings = parse_settings(pd.read_excel(book, sheet_name=SETTINGS_SHEET, header=None))
        run_sheets = [DATA_SHEET, *sorted(s for s in sheets if s.startswith(APPEND_SHEET_PREFIX))]
        frames = {sheet: pd.read_excel(book, sheet_name=sheet) for sheet in run_sheets}
    except WorkbookShapeError:
        raise
    except Exception as exc:  # the spreadsheet library's own errors, many kinds
        raise UnreadableWorkbookError(file.path, exc) from exc

    runs = []
    for sheet, frame in frames.items():
        block = settings.get(sheet) or settings.get(DATA_SHEET)
        if block is None:
            block = InstrumentSettings(
                module_name=None, executed_at=None, executed_raw=None, inputs={}
            )
        source = RunSource(file=file, sheet=sheet, settings=block)
        runs.append(build_run(source, frame))
    return runs


def build_run(source: RunSource, frame: pd.DataFrame) -> Run:
    """Turn one data sheet and its settings block into a typed run, collecting cell findings."""
    file = source.file
    issues = []
    where = {"sample_key": file.sample_key, "capacitor": file.capacitor.label, "path": file.path}
    expected = EXPECTED_MODULE[file.kind]
    if source.settings.module_name != expected:
        issues.append(
            warning(
                IssueCode.UNEXPECTED_MODULE,
                f"module {source.settings.module_name!r}, expected {expected!r}",
                location=source.sheet,
                **where,
            )
        )
    if source.settings.executed_raw and source.settings.executed_at is None:
        issues.append(
            warning(
                IssueCode.UNPARSED_TIMESTAMP,
                f"Last Executed {source.settings.executed_raw!r} is not {TIMESTAMP_FORMAT}",
                location=source.sheet,
                **where,
            )
        )

    def column(spec: ColumnSpec) -> np.ndarray:
        name = spec.find(frame)
        if name is None:
            raise WorkbookShapeError(
                file.path, f"sheet {source.sheet} has no {spec.label!r} column"
            )
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
        bad = np.flatnonzero(~np.isfinite(values))
        if len(bad):
            issues.append(
                warning(
                    IssueCode.BAD_CELL,
                    f"{len(bad)} cell(s) of {name!r} are blank or not numbers, "
                    f"first at row {bad[0] + 2}",
                    location=f"{source.sheet}!{name}",
                    **where,
                )
            )
        return values

    if file.kind is MeasurementKind.PUND:
        time_s, voltage_v, current_a = column(PUND_TIME), column(PUND_VOLTAGE), column(PUND_CURRENT)
        scalars = {
            f"{name.lower()}_c": float(pd.to_numeric(frame[name].iloc[0], errors="coerce"))
            for name in INSTRUMENT_SCALARS
            if name in frame.columns
        }
        return PundRun(
            source=source,
            frame=frame,
            time_s=time_s,
            voltage_v=voltage_v,
            current_a=current_a,
            instrument=InstrumentCharges(**scalars),
            issues=tuple(issues),
        )
    if file.kind is MeasurementKind.PV:
        return PvRun(
            source=source,
            frame=frame,
            time_s=column(PV_TIME),
            voltage_v=column(PV_VOLTAGE),
            current_a=column(PV_CURRENT),
            charge_c=column(PV_CHARGE),
            issues=tuple(issues),
        )
    return EnduranceTable(
        source=source,
        frame=frame,
        cycles=column(END_CYCLES),
        p_c=column(END_P),
        u_c=column(END_U),
        n_c=column(END_N),
        d_c=column(END_D),
        issues=tuple(issues),
    )


def index_row(run: Run, state: str) -> dict:
    """Return one raw-index row: what the run is, when it ran, how it was sampled, its findings."""
    file, settings = run.source.file, run.source.settings
    return {
        "sample": file.sample_name,
        "sample_key": file.sample_key,
        "sample_num": file.sample_number,
        "cap": file.capacitor.label,
        "row": file.capacitor.row,
        "col": file.capacitor.col,
        "kind": file.kind.value,
        "state": state,
        "run": run.source.sheet,
        "size_um": file.side_um,
        "module": settings.module_name,
        "executed": settings.executed_at.isoformat()
        if settings.executed_at
        else settings.executed_raw,
        "n_points": run.n_points,
        "dt_s": run.dt_s,
        "t_end_s": run.t_end_s,
        "params": ";".join(f"{k}={v}" for k, v in settings.inputs.items()),
        "problems": describe_issues(run.issues),
        "path": str(file.path),
    }
