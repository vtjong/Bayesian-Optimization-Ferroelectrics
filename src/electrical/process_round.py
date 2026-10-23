"""Process one round of measurements: find the files, read the runs, extract features, build tables.

Every workbook is opened once. Each capacitor and state becomes one row: ``pris`` for the
initial run, ``post<cycles>`` for the run after the endurance test, with the endurance features
on the ``pris`` row. A workbook or run that cannot be reduced becomes an error on its row and
the round carries on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from electrical.build_feature_tables import (
    CapacitorRecord,
    build_capacitor_table,
    summarize_samples,
)
from electrical.convert_units import area_cm2
from electrical.extract_endurance_features import extract_endurance_features
from electrical.extract_pund_features import extract_pund_features
from electrical.extract_pv_features import extract_pv_features
from electrical.find_measurement_files import MeasurementFile, find_measurement_files
from electrical.flag_issues import Issue, IssueCode, error, flag_outliers
from electrical.read_keithley_export import UnreadableWorkbookError, WorkbookShapeError
from electrical.read_measurement_runs import (
    CapacitorRuns,
    Run,
    assign_states,
    group_runs,
    index_row,
    read_runs,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class RoundResult:
    """Everything one round produces.

    :param raw_index: One row per run read (and per file that could not be read).
    :param capacitors: One row per capacitor and state, outliers flagged.
    :param samples: Mean / sd / n / median / MAD per feature, per sample and state.
    :param traces: ``{(sample, state): {cap: {"pund", "pv", "end"}}}`` for the QC figures.
    :param endurance_curves: Psw against cycle count per capacitor; empty without END files.
    :param issues: Issues that belong to no capacitor-state, such as unreadable workbooks.
    """

    raw_index: pd.DataFrame
    capacitors: pd.DataFrame
    samples: pd.DataFrame
    traces: dict
    endurance_curves: pd.DataFrame
    issues: tuple[Issue, ...]


def process_round(data_dir: Path, round_label: int) -> RoundResult:
    """Extract every feature of every capacitor under ``data_dir``.

    :param data_dir: Folder holding one ``Sample*/`` folder per film, or one sample folder.
    :param round_label: Round number written into every row.
    :raises RoundLayoutError: If the folder holds no exports or two tidy files claim one run.
    """
    runs, unreadable = [], []
    for file in find_measurement_files(data_dir):
        try:
            runs.extend(read_runs(file))
        except (UnreadableWorkbookError, WorkbookShapeError) as exc:
            unreadable.append(
                (file, error(IssueCode.UNREADABLE_WORKBOOK, str(exc), **file.where()))
            )
    assigned = assign_states(runs)

    records, traces, curves = [], {}, []
    for group in group_runs(assigned):
        record, trace, curve = _extract_capacitor(group, round_label)
        records.append(record)
        traces.setdefault((group.sample_name, group.state), {})[group.capacitor.label] = trace
        if curve is not None:
            curves.append(curve)

    capacitors = flag_outliers(build_capacitor_table(records))
    return RoundResult(
        raw_index=_index_runs(assigned, unreadable),
        capacitors=capacitors,
        samples=summarize_samples(capacitors),
        traces=traces,
        endurance_curves=pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(),
        issues=tuple(issue for _, issue in unreadable),
    )


def _extract_capacitor(group: CapacitorRuns, round_label: int):
    """Extract one capacitor-state's features; return its record, QC traces and endurance curve."""
    issues = [*group.issues, *(issue for run in group.runs for issue in run.issues)]
    where = {
        "sample_key": group.sample_key,
        "capacitor": group.capacitor.label,
        "state": group.state,
    }

    def extract(extract_features, run):
        if run is None:
            return None, None
        try:
            return extract_features(run)
        except Exception as exc:  # the one place a surprise in the data is caught and recorded
            name = extract_features.__name__
            issues.append(
                error(IssueCode.FEATURE_FAILURE, f"{name}: {type(exc).__name__}: {exc}", **where)
            )
            return None, None

    pund, pund_traces = extract(extract_pund_features, group.pund)
    pv, pv_traces = extract(extract_pv_features, group.pv)
    endurance, endurance_curve = extract(extract_endurance_features, group.endurance)
    issues += [issue for f in (pund, pv, endurance) if f is not None for issue in f.issues]

    curve = None
    if endurance is not None:
        curve = endurance_curve.to_frame(round_label, group.sample_name, group.capacitor.label)
    trace = {
        name: value
        for name, value in (("pund", pund_traces), ("pv", pv_traces), ("end", curve))
        if value is not None
    }
    record = CapacitorRecord(
        round_label=round_label,
        sample_key=group.sample_key,
        sample_name=group.sample_name,
        sample_number=group.sample_number,
        capacitor=group.capacitor,
        state=group.state,
        side_um=group.side_um,
        area_cm2=area_cm2(group.side_um),
        pund_executed_at=group.pund.source.settings.executed_at if group.pund else None,
        pv_executed_at=group.pv.source.settings.executed_at if group.pv else None,
        pund=pund,
        pv=pv,
        endurance=endurance,
        issues=tuple(issues),
    )
    return record, trace, curve


def _index_runs(
    assigned: list[tuple[Run, str]], unreadable: list[tuple[MeasurementFile, Issue]]
) -> pd.DataFrame:
    """One row per run read, plus one per file that could not be read."""
    rows = [index_row(run, state) for run, state in assigned]
    rows += [
        {
            **file.identity(),
            "kind": file.kind.value,
            "problems": issue.describe(),
            "path": str(file.path),
        }
        for file, issue in unreadable
    ]
    order = ["sample_num", "sample", "row", "col", "kind", "run"]
    return pd.DataFrame(rows).sort_values(order).reset_index(drop=True)
