"""One round of raw exports -> raw index, per-capacitor table, per-sample table, endurance curves.

Every workbook is opened once. Each (capacitor, state) becomes one row: state ``pris`` for the
initial run and ``post<cycles>`` for the run after the endurance test, with the endurance
features on the ``pris`` row. A workbook or run that cannot be reduced becomes an error
finding on its row; the round carries on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from electrical import locate, workbook
from electrical.aggregate import add_derived, per_sample
from electrical.capacitor import CapacitorRecord, capacitor_table
from electrical.device import area_cm2
from electrical.endurance import endurance_features
from electrical.issues import (
    IssueCode,
    UnreadableWorkbookError,
    ValidationIssue,
    WorkbookShapeError,
    error,
)
from electrical.locate import CapacitorRuns, MeasurementFile
from electrical.pund import pund_features
from electrical.pv import pv_features


@dataclass
class RoundResult:
    """Everything one round produces, before the QC report flags outliers.

    :param raw_index: One row per run found: file, sheet, module, timestamp, dt, n, findings.
    :param capacitors: One row per (capacitor, state) with every feature.
    :param samples: Mean / sd / n / median / MAD per feature, per (sample, state).
    :param traces: ``{(sample, state): {cap: {"pund": ..., "pv": ..., "end": ...}}}`` for the plots.
    :param endurance_curves: Psw against cycle count per capacitor, empty without END files.
    :param issues: Findings that belong to no single capacitor, such as unreadable files.
    """

    raw_index: pd.DataFrame
    capacitors: pd.DataFrame
    samples: pd.DataFrame
    traces: dict
    endurance_curves: pd.DataFrame
    issues: tuple[ValidationIssue, ...]


def read_round(
    data_dir: Path,
) -> tuple[list[workbook.Run], list[tuple[MeasurementFile, ValidationIssue]]]:
    """Open every export under ``data_dir`` once; unreadable files become findings, not stops."""
    runs, failed = [], []
    for file in locate.find_measurement_files(data_dir):
        try:
            runs.extend(workbook.read_workbook(file))
        except (UnreadableWorkbookError, WorkbookShapeError) as exc:
            failed.append(
                (
                    file,
                    error(
                        IssueCode.UNREADABLE_WORKBOOK,
                        str(exc),
                        sample_key=file.sample_key,
                        capacitor=file.capacitor.label,
                        path=file.path,
                    ),
                )
            )
    return runs, failed


def process_round(data_dir: Path, round_label) -> RoundResult:
    """Extract every feature of every run under ``data_dir``.

    :param data_dir: Folder holding one ``Sample*/`` folder per film.
    :param round_label: Round number written into every row.
    :return: The round's tables and traces.
    :raises RoundLayoutError: If the folder holds no exports or two tidy files claim one run.
    """
    runs, failed = read_round(data_dir)
    groups = locate.group_runs(runs)
    raw_index = build_raw_index(runs, failed)
    records, traces, curves = [], {}, []
    for group in groups:
        record, trace, curve = capacitor_record(group, round_label)
        records.append(record)
        traces.setdefault((group.sample_name, group.state), {})[group.capacitor.label] = trace
        if curve is not None:
            curves.append(curve)
    capacitors = add_derived(capacitor_table(records))
    curves_df = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return RoundResult(
        raw_index=raw_index,
        capacitors=capacitors,
        samples=per_sample(capacitors),
        traces=traces,
        endurance_curves=curves_df,
        issues=tuple(issue for _, issue in failed),
    )


def build_raw_index(runs, failed) -> pd.DataFrame:
    """One row per run read, plus one per file that could not be read."""
    rows = [workbook.index_row(run, state) for run, state in locate.run_states(runs)]
    for file, issue in failed:
        rows.append(
            {
                "sample": file.sample_name,
                "sample_key": file.sample_key,
                "sample_num": file.sample_number,
                "cap": file.capacitor.label,
                "row": file.capacitor.row,
                "col": file.capacitor.col,
                "kind": file.kind.value,
                "size_um": file.side_um,
                "problems": issue.describe(),
                "path": str(file.path),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["sample_num", "sample", "row", "col", "kind", "run"]).reset_index(
        drop=True
    )


def capacitor_record(
    group: CapacitorRuns, round_label
) -> tuple[CapacitorRecord, dict, pd.DataFrame | None]:
    """Reduce one capacitor-state to its record, its QC traces and its endurance curve."""
    issues = list(group.issues)
    for run in group.runs:
        issues.extend(run.issues)
    side_um = group.runs[0].source.file.side_um
    where = {
        "sample_key": group.sample_key,
        "capacitor": group.capacitor.label,
        "state": group.state,
    }
    trace, curve = {}, None
    pund = pv = endurance = None

    if group.pund is not None:
        pund, pund_trace = _reduce(pund_features, group.pund, issues, where)
        if pund is not None:
            trace["pund"] = pund_trace
    if group.pv is not None:
        pv, pv_trace = _reduce(pv_features, group.pv, issues, where)
        if pv is not None:
            trace["pv"] = pv_trace
    if group.endurance is not None:
        endurance, curve_record = _reduce(endurance_features, group.endurance, issues, where)
        if endurance is not None:
            curve = curve_record.to_frame(round_label, group.sample_name, group.capacitor.label)
            trace["end"] = curve
    for features in (pund, pv, endurance):
        if features is not None:
            issues.extend(features.issues)

    record = CapacitorRecord(
        round_label=round_label,
        sample_key=group.sample_key,
        sample_name=group.sample_name,
        sample_number=group.sample_number,
        capacitor=group.capacitor,
        state=group.state,
        side_um=side_um,
        area_cm2=area_cm2(side_um),
        pund_executed_at=group.pund.source.settings.executed_at if group.pund else None,
        pv_executed_at=group.pv.source.settings.executed_at if group.pv else None,
        pund=pund,
        pv=pv,
        endurance=endurance,
        issues=tuple(issues),
    )
    return record, trace, curve


def _reduce(reduce, run, issues: list, where: dict):
    """Run one feature function; a failure on unexpected data becomes a finding, not a stop."""
    try:
        return reduce(run)
    except Exception as exc:  # the one place a surprise in the data is caught and recorded
        issues.append(
            error(
                IssueCode.FEATURE_FAILURE,
                f"{reduce.__name__}: {type(exc).__name__}: {exc}",
                **where,
            )
        )
        return None, None


def describe_round(result: RoundResult) -> str:
    """Return the per-sample summary lines the operator checks against the group's sheet."""
    cap, samp = result.capacitors, result.samples
    n_error = int(cap["has_error"].sum()) + len(result.issues)
    n_warn = int((cap["n_issues"] > 0).sum())
    head = (
        f"{len(result.raw_index)} runs, {len(cap)} capacitor-states, "
        f"{samp.shape[0]} sample-states; {n_error} with errors, {n_warn} with findings"
    )
    cols = [
        "sample", "state", "n_capacitors", "n_switching", "psw_inst_uc_cm2_mean",
        "psw_inst_uc_cm2_sd", "pr_uc_cm2_mean", "pr_uc_cm2_sd", "kappa_summary_mean",
        "kappa_summary_sd", "kappa_pund_mean", "endurance_wake_ratio_mean",
    ]  # fmt: skip
    table = samp[[c for c in cols if c in samp]].round(3).to_string(index=False)
    return head + "\n" + table
