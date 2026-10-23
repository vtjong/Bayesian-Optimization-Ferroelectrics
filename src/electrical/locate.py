"""Find the Keithley exports of one round and group their runs per (capacitor, state).

Nothing is opened here. A file name carries the measurement kind, the state token and the
capacitor, and the folder carries the sample:

    PUND_pris_RxCy_30um.xls         one run (sheet Data)
    PUND_pris_1e6_RxCy_30um.xls     two runs: Data = pristine, Append1 = after 1e6 cycles
    PV_pris_1e6_RxCy_30um.xls       same pattern; PV_1e6_RxCy_30um.xls is a post-cycling run alone
    END_1e6_RxCy_30um.xls           endurance log, attached to the capacitor's pristine state

Names the instrument operator typed by hand vary in case, separator and a trailing ``_x`` for
a repeated measurement; all of those parse and are marked ``suspect`` so the QC report shows
them. ``group_runs`` works on the runs ``workbook.read_workbook`` returns.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from electrical.issues import (
    IssueCode,
    RoundLayoutError,
    ValidationIssue,
    warning,
)

if TYPE_CHECKING:
    from electrical.workbook import EnduranceTable, PundRun, PvRun, Run

FILE_NAME_RE = re.compile(
    r"^(?P<kind>PUND|PV|END)[_-](?P<tag>.+?)[_-](?P<cap>R\d+C\d+)[_-](?P<size>\d+)um"
    r"(?P<suffix>[_-]x)?\.xls$",
    re.IGNORECASE,
)
CAPACITOR_RE = re.compile(r"R(?P<row>\d+)C(?P<col>\d+)", re.IGNORECASE)
SAMPLE_RE = re.compile(r"sample\s*_?(?P<number>\d+)", re.IGNORECASE)
CYCLES_RE = re.compile(r"^(\d+e\d+|\d{3,})$", re.IGNORECASE)

#: State token of the initial measurement on a never-cycled capacitor.
PRISTINE_STATE = "pris"
#: Folders that never hold exports.
SKIPPED_FOLDERS = ("__MACOSX", "Analysis_Plots")


class MeasurementKind(Enum):
    PUND = "PUND"
    PV = "PV"
    END = "END"


@dataclass(frozen=True, slots=True, kw_only=True)
class CapacitorId:
    """Position of a capacitor on the die."""

    row: int
    col: int

    @property
    def label(self) -> str:
        """Return the ``RxCy`` label the files use."""
        return f"R{self.row}C{self.col}"


@dataclass(frozen=True, slots=True, kw_only=True)
class MeasurementFile:
    """One export, as its name and folder describe it.

    :param path: The workbook.
    :param kind: PUND, P-V or endurance.
    :param state_token: Lower-case state token from the name (``"pris"``), or ``None``.
    :param cycles_tag: Cycle count from the name (``"1e6"``), or ``None``.
    :param capacitor: Capacitor the file belongs to.
    :param side_um: Capacitor side length from the name, um.
    :param sample_name: Folder name as written.
    :param sample_key: Normalised sample key used for joins (``"sample16"``).
    :param sample_number: Sample number from the folder name, or ``None``.
    :param suspect: Whether the name departs from the instrument's own convention.
    """

    path: Path
    kind: MeasurementKind
    state_token: str | None
    cycles_tag: str | None
    capacitor: CapacitorId
    side_um: int
    sample_name: str
    sample_key: str
    sample_number: int | None
    suspect: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class CapacitorRuns:
    """The runs that describe one capacitor in one state, paired and checked.

    :param sample_key: Normalised sample key.
    :param sample_name: Folder name as written.
    :param sample_number: Sample number, or ``None``.
    :param capacitor: Capacitor on the die.
    :param state: ``"pris"`` or ``"post<cycles>"``.
    :param pund: The PUND run, if measured.
    :param pv: The P-V run, if measured.
    :param endurance: The endurance log, attached to the pristine state only.
    :param issues: Pairing and naming findings for this capacitor-state.
    """

    sample_key: str
    sample_name: str
    sample_number: int | None
    capacitor: CapacitorId
    state: str
    pund: PundRun | None = None
    pv: PvRun | None = None
    endurance: EnduranceTable | None = None
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)

    @property
    def runs(self) -> tuple[Run, ...]:
        """Return the runs present, PUND then P-V then endurance."""
        return tuple(run for run in (self.pund, self.pv, self.endurance) if run is not None)


def sample_key(folder_name: str) -> str:
    """Normalise a folder name for joins (``Sample 11`` -> ``sample11``).

    A folder without a sample number keeps its whole name, lower-cased and without spaces.
    """
    match = SAMPLE_RE.search(folder_name)
    if match:
        return f"sample{int(match['number'])}"
    return re.sub(r"\s+", "", folder_name).lower()


def sample_number(folder_name: str) -> int | None:
    """Return the sample number in a folder name, or ``None``."""
    match = SAMPLE_RE.search(folder_name)
    return int(match["number"]) if match else None


def parse_file_name(path: Path) -> MeasurementFile | None:
    """Describe an export from its name and folder, or return ``None`` for any other file."""
    match = FILE_NAME_RE.match(path.name)
    if match is None:
        return None
    capacitor_match = CAPACITOR_RE.match(match["cap"])
    tokens = [token for token in re.split(r"[_-]", match["tag"]) if token]
    cycles = [token for token in tokens if CYCLES_RE.match(token)]
    states = [token for token in tokens if not CYCLES_RE.match(token)]
    tidy = (
        match["suffix"] is None
        and "-" not in path.name
        and match["kind"].isupper()
        and match["cap"] == match["cap"].upper()
        and all(token == token.lower() for token in tokens)
    )
    folder = path.parent.name
    return MeasurementFile(
        path=path,
        kind=MeasurementKind(match["kind"].upper()),
        state_token=states[0].lower() if states else None,
        cycles_tag=cycles[0].lower() if cycles else None,
        capacitor=CapacitorId(row=int(capacitor_match["row"]), col=int(capacitor_match["col"])),
        side_um=int(match["size"]),
        sample_name=folder,
        sample_key=sample_key(folder),
        sample_number=sample_number(folder),
        suspect=not tidy,
    )


def find_measurement_files(data_dir: Path) -> list[MeasurementFile]:
    """Return every export under ``data_dir``, sorted by sample, capacitor and name.

    :param data_dir: A round folder (``Sample*/`` folders inside) or one sample folder.
    :raises RoundLayoutError: If no export is found.
    """
    found = []
    for path in sorted(data_dir.rglob("*.xls")):
        if any(part in SKIPPED_FOLDERS for part in path.relative_to(data_dir).parts):
            continue
        parsed = parse_file_name(path)
        if parsed is not None:
            found.append(parsed)
    if not found:
        raise RoundLayoutError(f"no PUND, PV or END .xls export under {data_dir}")
    found.sort(key=_file_order)
    return found


def _file_order(file: MeasurementFile) -> tuple:
    number = file.sample_number if file.sample_number is not None else float("inf")
    return (number, file.sample_key, file.capacitor.row, file.capacitor.col, file.path.name)


def run_state(run: Run) -> str:
    """Name the state a run measured: ``pris``, ``post<cycles>``, or ``endurance<cycles>``.

    A file without a state token, such as ``PV_1e6_RxCy``, is itself a post-cycling run. An
    ``Append`` sheet in a file whose name carries no cycle count is ``post<sheet>`` until
    ``group_runs`` resolves it from the capacitor's endurance log.
    """
    file = run.source.file
    cycles = file.cycles_tag
    if file.kind is MeasurementKind.END:
        return f"endurance{cycles or ''}"
    if run.source.sheet == "Data":
        return file.state_token if file.state_token else f"post{cycles}"
    return f"post{cycles or run.source.sheet.lower()}"


def run_states(runs: Iterable[Run]) -> list[tuple[Run, str]]:
    """Pair every run with its state; ``post<sheet>`` is resolved from the endurance log."""
    runs = list(runs)
    endurance_cycles: dict[tuple[str, str], set[str]] = {}
    for run in runs:
        file = run.source.file
        if file.kind is MeasurementKind.END and file.cycles_tag:
            endurance_cycles.setdefault((file.sample_key, file.capacitor.label), set()).add(
                file.cycles_tag
            )
    return [(run, _resolved_state(run, endurance_cycles)) for run in runs]


def group_runs(runs: Iterable[Run]) -> list[CapacitorRuns]:
    """Pair the PUND, P-V and endurance runs of every capacitor-state and record what is missing.

    Two files claiming the same run are resolved in favour of the tidy name; a tidy pair is a
    layout error, because neither can be chosen without the operator.

    :param runs: Every run read from the round's workbooks.
    :return: One record per (sample, capacitor, state), in sample and capacitor order.
    :raises RoundLayoutError: If two tidy files describe the same run.
    """
    chosen: dict[tuple, Run] = {}
    duplicates: dict[tuple, list[Run]] = {}
    for run, state in run_states(runs):
        file = run.source.file
        if file.kind is MeasurementKind.END:
            state = PRISTINE_STATE
        key = (file.sample_key, file.capacitor.label, state, file.kind)
        if key not in chosen:
            chosen[key] = run
            continue
        other = chosen[key]
        if not file.suspect and not other.source.file.suspect:
            raise RoundLayoutError(
                f"two files describe the same run {key[:3]}: {other.source.file.path.name} "
                f"and {file.path.name}"
            )
        keep, drop = (
            (run, other) if other.source.file.suspect and not file.suspect else (other, run)
        )
        chosen[key] = keep
        duplicates.setdefault(key, []).append(drop)

    groups: dict[tuple, dict] = {}
    for (sample, capacitor, state, kind), run in chosen.items():
        groups.setdefault((sample, capacitor, state), {})[kind] = run

    records = []
    for (sample, capacitor, state), by_kind in groups.items():
        any_run = next(iter(by_kind.values()))
        file = any_run.source.file
        issues = []
        where = {"sample_key": sample, "capacitor": capacitor, "state": state}
        for kind in (MeasurementKind.PUND, MeasurementKind.PV):
            if kind not in by_kind:
                code = (
                    IssueCode.UNPAIRED_PUND
                    if kind is MeasurementKind.PUND
                    else IssueCode.UNPAIRED_PV
                )
                issues.append(warning(code, f"no {kind.value} run for this state", **where))
        for kind, run in by_kind.items():
            if run.source.file.suspect:
                issues.append(
                    warning(
                        IssueCode.SUSPECT_FILE_NAME,
                        f"{run.source.file.path.name} departs from the usual naming",
                        path=run.source.file.path,
                        **where,
                    )
                )
            for dropped in duplicates.get((sample, capacitor, state, kind), []):
                issues.append(
                    warning(
                        IssueCode.DUPLICATE_RUN,
                        f"{dropped.source.file.path.name} ignored in favour of "
                        f"{run.source.file.path.name}",
                        path=dropped.source.file.path,
                        **where,
                    )
                )
        if file.sample_number is None:
            issues.append(
                warning(
                    IssueCode.NO_SAMPLE_NUMBER,
                    f"folder {file.sample_name!r} names no sample",
                    **where,
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
        records.append(
            CapacitorRuns(
                sample_key=sample,
                sample_name=file.sample_name,
                sample_number=file.sample_number,
                capacitor=file.capacitor,
                state=state,
                pund=pund,
                pv=pv,
                endurance=by_kind.get(MeasurementKind.END),
                issues=tuple(issues),
            )
        )
    records.sort(key=_record_order)
    return records


def _resolved_state(run: Run, endurance_cycles: dict) -> str:
    state = run_state(run)
    file = run.source.file
    if state.startswith("postappend"):
        tags = endurance_cycles.get((file.sample_key, file.capacitor.label), set())
        if len(tags) == 1:
            return f"post{next(iter(tags))}"
    return state


def _record_order(record: CapacitorRuns) -> tuple:
    number = record.sample_number if record.sample_number is not None else float("inf")
    return (number, record.sample_key, record.capacitor.row, record.capacitor.col, record.state)
