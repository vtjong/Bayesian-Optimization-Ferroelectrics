"""One capacitor-state as a typed record and as a row of the per-capacitor table.

The table's columns are a data contract with the per-sample statistics, the campaign CSV
and the correlation study. They are generated from the feature records' fields, so a field
added to a record appears in the table and a renamed field renames its column; the contract
test checks that the generated names and the flattened rows agree.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from enum import Enum
from typing import get_type_hints

import numpy as np
import pandas as pd

from electrical.endurance import EnduranceFeatures
from electrical.issues import ContractError, ValidationIssue, describe_issues, has_error
from electrical.locate import CapacitorId
from electrical.pund import (
    SPLIT_KEYS,
    Coercivity,
    Leakage,
    PulseCharge,
    PundDielectric,
    PundFeatures,
    PundWaveform,
    RampSplit,
    SwitchedPolarization,
    SwitchingPeak,
)
from electrical.pv import NegativeHalfLoop, PvDielectric, PvFeatures, PvSweep, Remanence
from electrical.workbook import InstrumentCharges

#: Columns that say which capacitor-state a row is.
IDENTITY_COLUMNS = (
    "round",
    "sample",
    "sample_key",
    "sample_num",
    "cap",
    "row",
    "col",
    "state",
    "side_um",
    "area_cm2",
)
#: Columns that close every row: provenance and findings.
TRAILING_COLUMNS = ("pund_executed", "pv_executed", "n_issues", "has_error", "issues")
#: Key that identifies one row across rounds.
CAPACITOR_KEY = ("round", "sample_key", "cap", "state")


@dataclass(frozen=True, slots=True, kw_only=True)
class CapacitorRecord:
    """Everything known about one capacitor in one state."""

    round_label: int
    sample_key: str
    sample_name: str
    sample_number: int | None
    capacitor: CapacitorId
    state: str
    side_um: int
    area_cm2: float
    pund_executed_at: datetime | None = None
    pv_executed_at: datetime | None = None
    pund: PundFeatures | None = None
    pv: PvFeatures | None = None
    endurance: EnduranceFeatures | None = None
    issues: tuple[ValidationIssue, ...] = ()


@dataclass(frozen=True, slots=True)
class ColumnGroup:
    """One feature record's scalar fields, flattened under a column prefix."""

    prefix: str
    record_type: type
    select: Callable[[CapacitorRecord], object | None]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.prefix + name for name in scalar_fields(self.record_type))


def scalar_fields(record_type: type) -> tuple[str, ...]:
    """Return the fields that flatten to one column each: not nested records, not findings."""
    hints = get_type_hints(record_type)
    return tuple(
        f.name
        for f in fields(record_type)
        if f.name != "issues" and not is_dataclass(hints[f.name])
    )


def _pund(select: Callable) -> Callable[[CapacitorRecord], object | None]:
    return lambda record: select(record.pund) if record.pund is not None else None


def _pv(select: Callable) -> Callable[[CapacitorRecord], object | None]:
    return lambda record: select(record.pv) if record.pv is not None else None


COLUMN_GROUPS: tuple[ColumnGroup, ...] = (
    ColumnGroup("", PundWaveform, _pund(lambda p: p.waveform)),
    ColumnGroup("inst_", InstrumentCharges, _pund(lambda p: p.instrument)),
    ColumnGroup("q_p_", PulseCharge, _pund(lambda p: p.charges.p if p.charges else None)),
    ColumnGroup("q_u_", PulseCharge, _pund(lambda p: p.charges.u if p.charges else None)),
    ColumnGroup("q_n_", PulseCharge, _pund(lambda p: p.charges.n if p.charges else None)),
    ColumnGroup("q_d_", PulseCharge, _pund(lambda p: p.charges.d if p.charges else None)),
    ColumnGroup("", SwitchedPolarization, _pund(lambda p: p.polarization)),
    *(
        ColumnGroup(f"split_{key}_", RampSplit, _pund(lambda p, key=key: p.splits.get(key)))
        for key in SPLIT_KEYS
    ),
    ColumnGroup(
        "peak_pos_", SwitchingPeak, _pund(lambda p: p.coercivity.pos if p.coercivity else None)
    ),
    ColumnGroup(
        "peak_neg_", SwitchingPeak, _pund(lambda p: p.coercivity.neg if p.coercivity else None)
    ),
    ColumnGroup("", Coercivity, _pund(lambda p: p.coercivity)),
    ColumnGroup("", Leakage, _pund(lambda p: p.leakage)),
    ColumnGroup("", PundDielectric, _pund(lambda p: p.dielectric)),
    ColumnGroup("", PvSweep, _pv(lambda v: v.sweep)),
    ColumnGroup("", Remanence, _pv(lambda v: v.remanence)),
    ColumnGroup("", NegativeHalfLoop, _pv(lambda v: v.half_loop)),
    ColumnGroup("", PvDielectric, _pv(lambda v: v.dielectric)),
    ColumnGroup("endurance_", EnduranceFeatures, lambda record: record.endurance),
)

CAPACITOR_COLUMNS: tuple[str, ...] = (
    *IDENTITY_COLUMNS,
    *(name for group in COLUMN_GROUPS for name in group.names),
    *TRAILING_COLUMNS,
)
if len(set(CAPACITOR_COLUMNS)) != len(CAPACITOR_COLUMNS):
    _dup = sorted({c for c in CAPACITOR_COLUMNS if CAPACITOR_COLUMNS.count(c) > 1})
    raise ContractError(f"capacitor columns collide: {_dup}")


def _cell(value):
    if isinstance(value, Enum):
        return value.value
    return value


def capacitor_row(record: CapacitorRecord) -> dict:
    """Flatten one record into a row keyed by ``CAPACITOR_COLUMNS``; absent groups are NaN."""
    row = {
        "round": record.round_label,
        "sample": record.sample_name,
        "sample_key": record.sample_key,
        "sample_num": record.sample_number,
        "cap": record.capacitor.label,
        "row": record.capacitor.row,
        "col": record.capacitor.col,
        "state": record.state,
        "side_um": record.side_um,
        "area_cm2": record.area_cm2,
    }
    for group in COLUMN_GROUPS:
        values = group.select(record)
        for name in scalar_fields(group.record_type):
            row[group.prefix + name] = (
                _cell(getattr(values, name)) if values is not None else np.nan
            )
    row["pund_executed"] = record.pund_executed_at
    row["pv_executed"] = record.pv_executed_at
    row["n_issues"] = len(record.issues)
    row["has_error"] = has_error(record.issues)
    row["issues"] = describe_issues(record.issues) or None
    return row


def capacitor_table(records: Iterable[CapacitorRecord]) -> pd.DataFrame:
    """Build the per-capacitor table, one row per record, in contract column order."""
    frame = pd.DataFrame.from_records(
        [capacitor_row(r) for r in records], columns=list(CAPACITOR_COLUMNS)
    )
    frame["switching"] = frame["switching"].fillna(False).astype(bool)
    return frame
