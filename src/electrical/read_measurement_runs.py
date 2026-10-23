"""Read each export's sheets as PUND, P-V or endurance runs, and pair the runs per capacitor.

A PUND or P-V workbook holds the pristine run in sheet ``Data`` and, once the capacitor has
been cycled, the run after cycling in ``Append1``; an endurance workbook holds one log. The
state of a run, ``pris`` or ``post<cycles>``, follows from its sheet, its file name and the
capacitor's endurance log. The runs of one capacitor in one state are paired so their features
can be extracted together, and anything missing or ambiguous in the pairing is recorded.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from electrical.find_measurement_files import (
    CapacitorId,
    MeasurementFile,
    MeasurementKind,
    RoundLayoutError,
)
from electrical.flag_issues import Issue, IssueCode, describe_issues, warning
from electrical.read_keithley_export import (
    DATA_SHEET,
    NO_SETTINGS,
    ColumnSpec,
    InstrumentSettings,
    check_settings,
    read_first_row_value,
    read_keithley_export,
    read_numeric_column,
)

#: State of the initial measurement on a never-cycled capacitor.
PRISTINE_STATE = "pris"
#: KTEI module each kind of export comes from.
EXPECTED_MODULE = {
    MeasurementKind.PUND: "pundTest",
    MeasurementKind.PV: "doubleSweepSeg",
    MeasurementKind.END: "pundEndurance",
}
#: Columns each kind of run is read from, by the field they fill.
RUN_COLUMNS = {
    MeasurementKind.PUND: {
        "time_s": ColumnSpec(label="t", alternatives=("Time",)),
        "voltage_v": ColumnSpec(label="V"),
        "current_a": ColumnSpec(label="I"),
    },
    MeasurementKind.PV: {
        "time_s": ColumnSpec(label="Time"),
        "voltage_v": ColumnSpec(label="Vforce"),
        "current_a": ColumnSpec(label="Imeas"),
        "charge_c": ColumnSpec(label="Charge"),
    },
    MeasurementKind.END: {
        "cycles": ColumnSpec(label="iteration"),
        "p_c": ColumnSpec(label="P"),
        "u_c": ColumnSpec(label="U"),
        "n_c": ColumnSpec(label="N"),
        "d_c": ColumnSpec(label="D"),
    },
}
#: Ramp-integrated charges the instrument writes on a PUND run's first data row, in coulombs.
INSTRUMENT_SCALARS = ("P", "Pa", "U", "Ua", "N", "Na", "D", "Da", "Psw", "Qsw")
#: Pairing issue raised when a capacitor-state lacks a run of this kind.
UNPAIRED = {
    MeasurementKind.PUND: IssueCode.UNPAIRED_PUND,
    MeasurementKind.PV: IssueCode.UNPAIRED_PV,
}


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
    """Where a run came from: the file, the sheet inside it, and that sheet's settings."""

    file: MeasurementFile
    sheet: str
    settings: InstrumentSettings

    def where(self) -> dict[str, object]:
        """Return the location of an issue found in this run."""
        return self.file.where(sheet=self.sheet)


@dataclass(frozen=True, slots=True, kw_only=True)
class PundRun:
    """One PUND measurement: V(t), I(t) and the instrument's ramp-integrated charges."""

    source: RunSource
    time_s: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray
    instrument: InstrumentCharges
    issues: tuple[Issue, ...] = ()

    @property
    def dt_s(self) -> float:
        return float(np.nanmedian(np.diff(self.time_s)))


@dataclass(frozen=True, slots=True, kw_only=True)
class PvRun:
    """One P-V sweep: forced voltage, measured current and the instrument's running charge."""

    source: RunSource
    time_s: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray
    charge_c: np.ndarray
    issues: tuple[Issue, ...] = ()

    @property
    def dt_s(self) -> float:
        return float(np.nanmedian(np.diff(self.time_s)))


@dataclass(frozen=True, slots=True, kw_only=True)
class EnduranceLog:
    """One endurance log: ramp-integrated P, U, N, D charges at each logged cycle count."""

    source: RunSource
    cycles: np.ndarray
    p_c: np.ndarray
    u_c: np.ndarray
    n_c: np.ndarray
    d_c: np.ndarray
    issues: tuple[Issue, ...] = ()


Run = PundRun | PvRun | EnduranceLog
RUN_TYPES = {
    MeasurementKind.PUND: PundRun,
    MeasurementKind.PV: PvRun,
    MeasurementKind.END: EnduranceLog,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class CapacitorRuns:
    """The runs that describe one capacitor in one state, paired, with pairing issues.

    :param state: ``"pris"`` or ``"post<cycles>"``.
    :param pund: The PUND run, if measured.
    :param pv: The P-V run, if measured.
    :param endurance: The endurance log, attached to the pristine state.
    :param issues: Pairing and naming issues for this capacitor-state.
    """

    sample_key: str
    sample_name: str
    sample_number: int | None
    capacitor: CapacitorId
    state: str
    pund: PundRun | None = None
    pv: PvRun | None = None
    endurance: EnduranceLog | None = None
    issues: tuple[Issue, ...] = ()

    @property
    def runs(self) -> tuple[Run, ...]:
        """Return the runs present: PUND, then P-V, then endurance."""
        return tuple(run for run in (self.pund, self.pv, self.endurance) if run is not None)

    @property
    def side_um(self) -> int:
        return self.runs[0].source.file.side_um


def read_runs(file: MeasurementFile) -> list[Run]:
    """Read every run in one export, ``Data`` sheet first.

    :raises UnreadableWorkbookError: If the workbook cannot be opened.
    :raises WorkbookShapeError: If a sheet or a column the run needs is missing.
    """
    export = read_keithley_export(file.path)
    runs = []
    for sheet, frame in export.sheets.items():
        settings = export.settings.get(sheet) or export.settings.get(DATA_SHEET) or NO_SETTINGS
        source = RunSource(file=file, sheet=sheet, settings=settings)
        where = source.where()
        issues = check_settings(settings, EXPECTED_MODULE[file.kind], **where)
        arrays = {}
        for field_name, spec in RUN_COLUMNS[file.kind].items():
            arrays[field_name], found = read_numeric_column(frame, spec, where)
            issues.extend(found)
        if file.kind is MeasurementKind.PUND:
            arrays["instrument"] = InstrumentCharges(
                **{
                    f"{name.lower()}_c": read_first_row_value(frame, name)
                    for name in INSTRUMENT_SCALARS
                }
            )
        runs.append(RUN_TYPES[file.kind](source=source, **arrays, issues=tuple(issues)))
    return runs


def assign_states(runs: Iterable[Run]) -> list[tuple[Run, str]]:
    """Pair every run with the state it measured.

    * An endurance log belongs to the pristine state: it starts from the pristine film.
    * Sheet ``Data`` is the state its file name says (``PUND_pris_...`` -> ``pris``). Without a
      state token it is pristine when the file holds a second run, since the instrument appends
      the post-cycling run to the file it started with; otherwise it is a post-cycling run alone.
    * Every other run is ``post<cycles>``, the cycle count taken from the file name, else from
      the capacitor's endurance log, else the sheet name.
    """
    runs = list(runs)
    runs_per_file = Counter(run.source.file.path for run in runs)
    logged_cycles: dict[tuple[str, str], set[str]] = defaultdict(set)
    for run in runs:
        file = run.source.file
        if file.kind is MeasurementKind.END and file.cycles_tag:
            logged_cycles[(file.sample_key, file.capacitor.label)].add(file.cycles_tag)

    def state_of(run: Run) -> str:
        file, sheet = run.source.file, run.source.sheet
        if file.kind is MeasurementKind.END:
            return PRISTINE_STATE
        if sheet == DATA_SHEET and file.state_token:
            return file.state_token
        if sheet == DATA_SHEET and runs_per_file[file.path] > 1:
            return PRISTINE_STATE
        logged = logged_cycles.get((file.sample_key, file.capacitor.label), set())
        cycles = file.cycles_tag or (next(iter(logged)) if len(logged) == 1 else sheet.lower())
        return f"post{cycles}"

    return [(run, state_of(run)) for run in runs]


def group_runs(assigned: Iterable[tuple[Run, str]]) -> list[CapacitorRuns]:
    """Pair the PUND, P-V and endurance runs of every capacitor and state.

    When two files describe the same run, the one with the tidy name is used and the other is
    reported. Two tidy files for one run stop the round: neither can be chosen without the
    operator.

    :param assigned: Every run of the round with its state, from ``assign_states``.
    :return: One record per (sample, capacitor, state), in sample and capacitor order.
    :raises RoundLayoutError: If two tidy files describe the same run.
    """
    chosen: dict[tuple, Run] = {}
    ignored: dict[tuple, list[Run]] = defaultdict(list)
    for run, state in assigned:
        file = run.source.file
        slot = (file.sample_key, file.capacitor.label, state, file.kind)
        other = chosen.get(slot)
        if other is None:
            chosen[slot] = run
            continue
        if not file.suspect and not other.source.file.suspect:
            raise RoundLayoutError(
                f"two files describe the same run {slot[:3]}: {other.source.file.path.name} "
                f"and {file.path.name}"
            )
        keep, drop = (
            (run, other) if other.source.file.suspect and not file.suspect else (other, run)
        )
        chosen[slot] = keep
        ignored[slot].append(drop)

    by_state: dict[tuple, dict[MeasurementKind, Run]] = defaultdict(dict)
    for (sample, capacitor, state, kind), run in chosen.items():
        by_state[(sample, capacitor, state)][kind] = run

    records = []
    for (sample, capacitor, state), by_kind in by_state.items():
        dropped = {kind: ignored.get((sample, capacitor, state, kind), []) for kind in by_kind}
        records.append(_pair_runs(state, by_kind, dropped))
    return sorted(records, key=lambda r: (*r.runs[0].source.file.order[:4], r.state))


def _pair_runs(
    state: str, by_kind: dict[MeasurementKind, Run], ignored: dict[MeasurementKind, list[Run]]
) -> CapacitorRuns:
    """Build one capacitor-state and record what is missing, doubtful or out of order."""
    file = next(iter(by_kind.values())).source.file
    where = {"sample_key": file.sample_key, "capacitor": file.capacitor.label, "state": state}
    issues = [
        warning(UNPAIRED[kind], f"no {kind.value} run for this state", **where)
        for kind in (MeasurementKind.PUND, MeasurementKind.PV)
        if kind not in by_kind
    ]
    for kind, run in by_kind.items():
        name = run.source.file.path.name
        if run.source.file.suspect:
            issues.append(
                warning(
                    IssueCode.SUSPECT_FILE_NAME, f"{name} departs from the usual naming", **where
                )
            )
        for dropped in ignored[kind]:
            issues.append(
                warning(
                    IssueCode.DUPLICATE_RUN,
                    f"{dropped.source.file.path.name} ignored in favour of {name}",
                    **where,
                )
            )
    if file.sample_number is None:
        issues.append(
            warning(
                IssueCode.NO_SAMPLE_NUMBER, f"folder {file.sample_name!r} names no sample", **where
            )
        )
    pund, pv = by_kind.get(MeasurementKind.PUND), by_kind.get(MeasurementKind.PV)
    if state == PRISTINE_STATE and pund is not None and pv is not None:
        pund_at, pv_at = pund.source.settings.executed_at, pv.source.settings.executed_at
        if pund_at is not None and pv_at is not None and pund_at < pv_at:
            issues.append(
                warning(
                    IssueCode.PUND_BEFORE_PV,
                    "PUND ran before the P-V sweep that presets the film",
                    **where,
                )
            )
    return CapacitorRuns(
        sample_key=file.sample_key,
        sample_name=file.sample_name,
        sample_number=file.sample_number,
        capacitor=file.capacitor,
        state=state,
        pund=pund,
        pv=pv,
        endurance=by_kind.get(MeasurementKind.END),
        issues=tuple(issues),
    )


def index_row(run: Run, state: str) -> dict[str, object]:
    """Describe one run for the raw index: which file and sheet, when it ran, how it was sampled."""
    file, settings = run.source.file, run.source.settings
    if isinstance(run, EnduranceLog):
        n_points, dt_s, t_end_s = len(run.cycles), None, None
    else:
        n_points, dt_s, t_end_s = len(run.time_s), run.dt_s, float(run.time_s[-1])
    return {
        **file.identity(),
        "kind": file.kind.value,
        "state": state,
        "run": run.source.sheet,
        "module": settings.module_name,
        "executed": settings.executed_at.isoformat()
        if settings.executed_at
        else settings.executed_raw,
        "n_points": n_points,
        "dt_s": dt_s,
        "t_end_s": t_end_s,
        "params": ";".join(f"{k}={v}" for k, v in settings.inputs.items()),
        "problems": describe_issues(run.issues),
        "path": str(file.path),
    }
