"""Assembly and serialization of a fabrication plan.

A plan is an ordered set of flash conditions grouped into blocks that record
why each condition is in the batch. Blocks carry no meaning to the surrogate;
they distinguish, for example, seed conditions from bridge conditions.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Plan CSV schema. These names are a data contract with the
# ``campaign.results``; changing one invalidates sheets already in circulation.
INDEX_COLUMN = "index"
BLOCK_COLUMN = "block"
VOLTAGE_COLUMN = "voltage_V"
PULSE_WIDTH_COLUMN = "time_ms"
READOUT_KIND_COLUMN = "readout"
NOTE_COLUMN = "note"

PLAN_COLUMNS = (
    INDEX_COLUMN,
    BLOCK_COLUMN,
    VOLTAGE_COLUMN,
    PULSE_WIDTH_COLUMN,
    READOUT_KIND_COLUMN,
    NOTE_COLUMN,
)

# Index 0 is reserved for the never-flashed reference inserted into the results
# sheet by ``campaign.results.blank_template``.
FIRST_CONDITION_INDEX = 1


@dataclass(frozen=True, slots=True)
class ConditionBlock:
    """Conditions included in the plan for the same experimental reason.

    :param label: Short block label recorded in the plan, such as ``"S"``.
    :param voltage_v: Flash voltages in V.
    :param pulse_width_ms: Pulse widths in ms.
    :param note: Reason these conditions are included, written to every row.
    :param annotations: Optional reporting columns with one value per condition.
    """

    label: str
    voltage_v: np.ndarray
    pulse_width_ms: np.ndarray
    note: str = ""
    annotations: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the block's condition and annotation columns.

        :raises ValueError: If a condition array is not one-dimensional, column
            lengths differ, or an annotation uses a reserved plan column name.
        """
        voltage_v = np.asarray(self.voltage_v)
        pulse_width_ms = np.asarray(self.pulse_width_ms)

        if voltage_v.ndim != 1:
            raise ValueError(f"block {self.label!r}: voltage_v must be one-dimensional")
        if pulse_width_ms.ndim != 1:
            raise ValueError(f"block {self.label!r}: pulse_width_ms must be one-dimensional")
        if len(pulse_width_ms) != len(voltage_v):
            raise ValueError(
                f"block {self.label!r}: {len(voltage_v)} voltages against "
                f"{len(pulse_width_ms)} pulse widths"
            )

        for name, values in self.annotations.items():
            if name in PLAN_COLUMNS:
                raise ValueError(
                    f"block {self.label!r}: annotation {name!r} conflicts with a plan column"
                )

            values = np.asarray(values)
            if values.ndim != 1:
                raise ValueError(
                    f"block {self.label!r}: annotation {name!r} must be one-dimensional"
                )
            if len(values) != len(voltage_v):
                raise ValueError(
                    f"block {self.label!r}: annotation {name!r} has "
                    f"{len(values)} values for {len(voltage_v)} conditions"
                )

    def __len__(self) -> int:
        """Return the number of conditions in the block."""
        return len(self.voltage_v)


def build_plan(
    blocks: Iterable[ConditionBlock],
    readout_kind: str,
) -> pd.DataFrame:
    """Assemble condition blocks into an indexed fabrication plan.

    Duplicate setpoints are permitted. For example, a bridge condition may
    deliberately repeat a condition fired in an earlier fabrication batch.

    :param blocks: Condition blocks in plan order.
    :param readout_kind: Instrument reading recorded for every condition.
    :return: Plan with one row per fabricated condition.
    :raises ValueError: If no block contains a condition.
    """
    frames: list[pd.DataFrame] = []

    for block in blocks:
        if not block:
            continue

        frames.append(
            pd.DataFrame(
                {
                    BLOCK_COLUMN: block.label,
                    VOLTAGE_COLUMN: block.voltage_v,
                    PULSE_WIDTH_COLUMN: block.pulse_width_ms,
                    READOUT_KIND_COLUMN: readout_kind,
                    NOTE_COLUMN: block.note,
                    **block.annotations,
                }
            )
        )

    if not frames:
        raise ValueError("a plan needs at least one condition")

    plan = pd.concat(frames, ignore_index=True)
    plan.insert(
        0,
        INDEX_COLUMN,
        np.arange(
            FIRST_CONDITION_INDEX,
            FIRST_CONDITION_INDEX + len(plan),
        ),
    )

    extra_columns = [column for column in plan.columns if column not in PLAN_COLUMNS]

    return plan[[*PLAN_COLUMNS, *extra_columns]]


def write_plan(path: Path, plan: pd.DataFrame) -> Path:
    """Write a fabrication plan without permitting replacement.

    A plan records the conditions fixed before firing. Exclusive creation makes
    replacement impossible even if another process creates the destination
    between checking and writing.

    :param path: Destination CSV path.
    :param plan: Plan to write.
    :return: Path of the written plan.
    :raises FileExistsError: If the destination already exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8", newline="") as destination:
        plan.to_csv(destination, index=False)

    return path


def describe_plan(
    plan: pd.DataFrame,
    annotations: Iterable[str] = (),
) -> str:
    """Render selected plan columns as a fixed-width terminal table.

    :param plan: Plan to render.
    :param annotations: Optional annotation columns to display.
    :return: Fixed-width representation of the plan.
    """
    columns = [
        INDEX_COLUMN,
        BLOCK_COLUMN,
        VOLTAGE_COLUMN,
        PULSE_WIDTH_COLUMN,
        *[name for name in annotations if name in plan.columns],
    ]
    return plan[columns].to_string(index=False)
