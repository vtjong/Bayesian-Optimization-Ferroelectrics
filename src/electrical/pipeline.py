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
from electrical.endurance import endurance_features
from electrical.issues import (
    IssueCode,
    UnreadableWorkbookError,
    ValidationIssue,
    WorkbookShapeError,
    describe_issues,
    error,
    has_error,
)
from electrical.locate import CapacitorRuns, MeasurementFile
from electrical.pund import pund_features
from electrical.pv import pv_features

#: Columns that identify one row of the per-capacitor table across rounds.
CAPACITOR_KEY = ["round", "sample", "cap", "state"]
#: Messages the feature functions still report as a ``problem`` string, by finding.
PROBLEM_CODES = {
    "unexpected pulse pattern": IssueCode.UNEXPECTED_PULSE_PATTERN,
    "sweep is not positive-first": IssueCode.SWEEP_NOT_POSITIVE_FIRST,
    "endurance table too short": IssueCode.ENDURANCE_TOO_SHORT,
}


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
    rows, traces, curves = [], {}, []
    for group in groups:
        row, trace, curve = capacitor_row(group, round_label)
        rows.append(row)
        traces.setdefault((group.sample_name, group.state), {})[group.capacitor.label] = trace
        if curve is not None:
            curves.append(curve)
    capacitors = add_derived(pd.DataFrame(rows))
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


def capacitor_row(group: CapacitorRuns, round_label) -> tuple[dict, dict, pd.DataFrame | None]:
    """Reduce one capacitor-state to its feature row, its QC traces and its endurance curve."""
    row = {
        "round": round_label,
        "sample": group.sample_name,
        "sample_key": group.sample_key,
        "sample_num": group.sample_number,
        "cap": group.capacitor.label,
        "row": group.capacitor.row,
        "col": group.capacitor.col,
        "state": group.state,
        "size_um": None,
        "switching": False,
    }
    issues = list(group.issues)
    for run in group.runs:
        issues.extend(run.issues)
    trace, curve = {}, None
    where = {
        "sample_key": group.sample_key,
        "capacitor": group.capacitor.label,
        "state": group.state,
    }

    if group.pund is not None:
        file = group.pund.source.file
        row["size_um"] = file.side_um
        features, pund_trace = _extract(
            pund_features, group.pund.frame, file.side_um, issues, where
        )
        if features is not None:
            row.update(features)
            trace["pund"] = pund_trace
            row["pund_executed"] = group.pund.source.settings.executed_at
    if group.pv is not None:
        file = group.pv.source.file
        row["size_um"] = file.side_um
        features, pv_trace = _extract(pv_features, group.pv.frame, file.side_um, issues, where)
        if features is not None:
            pv_trace["area_cm2"] = features["area_cm2"]
            row.update(
                {k: v for k, v in features.items() if k not in ("area_cm2", "dt_s", "n_points")}
            )
            trace["pv"] = pv_trace
            row["pv_executed"] = group.pv.source.settings.executed_at
    if group.endurance is not None:
        file = group.endurance.source.file
        features, curve = _extract(
            endurance_features, group.endurance.frame, file.side_um, issues, where
        )
        if features is not None:
            row.update(features)
            curve.insert(0, "cap", group.capacitor.label)
            curve.insert(0, "sample", group.sample_name)
            curve.insert(0, "round", round_label)
            trace["end"] = curve

    row["n_issues"] = len(issues)
    row["has_error"] = has_error(issues)
    row["issues"] = describe_issues(issues) or None
    return row, trace, curve


def _extract(reduce, frame, side_um, issues: list, where: dict):
    """Run one feature function, turning its ``problem`` strings and failures into findings."""
    try:
        features, trace = reduce(frame, side_um)
    except Exception as exc:  # the feature functions do not yet classify their own failures
        issues.append(
            error(
                IssueCode.FEATURE_FAILURE,
                f"{reduce.__name__}: {type(exc).__name__}: {exc}",
                **where,
            )
        )
        return None, None
    for key in ("problem", "problem_end"):
        message = features.pop(key, None)
        if message:
            code = next(
                (code for prefix, code in PROBLEM_CODES.items() if message.startswith(prefix)),
                IssueCode.FEATURE_FAILURE,
            )
            issues.append(error(code, message, **where))
    return features, trace


def merge_into_campaign(path: Path, capacitors: pd.DataFrame) -> pd.DataFrame:
    """Replace this round's rows in the cross-round table and write it back.

    :param path: The campaign CSV; created if absent.
    :param capacitors: This round's per-capacitor rows.
    :return: The merged table.
    """
    merged = capacitors
    if path.exists():
        old = pd.read_csv(path)
        old = old[
            ~old.set_index(CAPACITOR_KEY).index.isin(capacitors.set_index(CAPACITOR_KEY).index)
        ]
        merged = pd.concat([old, capacitors], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return merged


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
        "sample", "state", "n_capacitors", "n_switching", "Psw_inst_mean", "Psw_inst_sd",
        "Pr_mean", "Pr_sd", "kappa_summary_mean", "kappa_summary_sd", "kappa_PUND_mean",
        "wake_ratio_mean",
    ]  # fmt: skip
    table = samp[[c for c in cols if c in samp]].round(3).to_string(index=False)
    return head + "\n" + table
