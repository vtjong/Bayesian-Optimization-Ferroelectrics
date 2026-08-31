"""Assembly and serialization of a fabrication plan.

A plan is an ordered set of flash conditions to be fabricated, grouped into
blocks that record why each condition is in the batch. Blocks carry no meaning
to the surrogate; they exist so that a reader, and the results loader, can tell
a seed condition from a bridge condition from a never-flashed reference.

Conditions arrive already snapped to instrument resolution. This module does
not choose them, does not filter them, and evaluates no physical quantity. It
assigns indices, concatenates, and writes.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

#: Column names written to the plan CSV. These are a data contract with the
#: laboratory and with ``campaign.results``; changing one invalidates every
#: sheet already in circulation.
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

#: First index assigned to a fabricated condition. Index 0 is reserved for the
#: never-flashed reference, which ``campaign.results.blank_template`` inserts.
FIRST_CONDITION_INDEX = 1


@dataclass(frozen=True, slots=True)
class ConditionBlock:
    """A group of conditions sharing a label and a reason for being fired.

    :param label: Short block label recorded in the plan, such as ``"S"``.
    :param voltage_v: Flash voltages in V.
    :param pulse_width_ms: Pulse widths in ms.
    :param note: Why these conditions are in the batch. Written to every row.
    :param annotations: Extra columns keyed by name, one value per condition.
        Reporting only; nothing downstream reads them.
    """

    label: str
    voltage_v: np.ndarray
    pulse_width_ms: np.ndarray
    note: str = ""
    annotations: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject a block whose columns disagree on length.

        :raises ValueError: If any array length differs from the voltages.
        """
        size = np.asarray(self.voltage_v).size

        if np.asarray(self.pulse_width_ms).size != size:
            raise ValueError(
                f"block {self.label!r}: {size} voltages against "
                f"{np.asarray(self.pulse_width_ms).size} pulse widths"
            )

        for name, values in self.annotations.items():
            if np.asarray(values).size != size:
                raise ValueError(
                    f"block {self.label!r}: annotation {name!r} has "
                    f"{np.asarray(values).size} values for {size} conditions"
                )

    def __len__(self) -> int:
        """Return the number of conditions in the block."""
        return int(np.asarray(self.voltage_v).size)


def build_plan(blocks: Sequence[ConditionBlock], readout_kind: str) -> pd.DataFrame:
    """Assemble blocks into a plan, numbering conditions in the order given.

    Duplicate setpoints are permitted and deliberate: a bridge condition repeats
    a setpoint already fired in an earlier batch, which is how a run-to-run
    difference is separated from a process dependence.

    :param blocks: Condition blocks, written in this order.
    :param readout_kind: Instrument reading recorded per condition, such as
        ``"eps_r"``. Constant across the plan so one sheet is read one way.
    :return: Plan with one row per condition.
    :raises ValueError: If no block contains a condition.
    """
    frames = []
    for block in blocks:
        if not len(block):
            continue

        frame = pd.DataFrame(
            {
                BLOCK_COLUMN: np.full(len(block), block.label),
                VOLTAGE_COLUMN: np.asarray(block.voltage_v),
                PULSE_WIDTH_COLUMN: np.asarray(block.pulse_width_ms),
                READOUT_KIND_COLUMN: np.full(len(block), readout_kind),
                NOTE_COLUMN: np.full(len(block), block.note),
            }
        )
        for name, values in block.annotations.items():
            frame[name] = np.asarray(values)

        frames.append(frame)

    if not frames:
        raise ValueError("a plan needs at least one condition")

    plan = pd.concat(frames, ignore_index=True)
    plan.insert(0, INDEX_COLUMN, np.arange(len(plan)) + FIRST_CONDITION_INDEX)

    ordered = [column for column in PLAN_COLUMNS if column in plan.columns]
    extra = [column for column in plan.columns if column not in ordered]

    return plan[ordered + extra]


def write_plan(path: Path, plan: pd.DataFrame) -> Path:
    """Write a plan to CSV, refusing to overwrite an existing file.

    A plan is fixed before firing and is the record of what was intended, so
    silently replacing one would destroy the only evidence of what a batch was
    supposed to be.

    :param path: Destination CSV path.
    :param plan: Plan to write.
    :return: The path written.
    :raises FileExistsError: If the destination already exists.
    """
    if path.exists():
        raise FileExistsError(
            f"{path} already exists; a fired plan is a record and is not rewritten"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(path, index=False)

    return path


def describe_plan(plan: pd.DataFrame, annotations: Iterable[str] = ()) -> str:
    """Render a plan as a fixed-width table for reading in a terminal.

    :param plan: Plan to render.
    :param annotations: Extra columns to include after the setpoints.
    :return: The rendered table.
    """
    shown = [INDEX_COLUMN, BLOCK_COLUMN, VOLTAGE_COLUMN, PULSE_WIDTH_COLUMN]
    shown += [name for name in annotations if name in plan.columns]

    widths = {
        column: max(len(column), plan[column].astype(str).str.len().max()) for column in shown
    }
    header = "  ".join(column.rjust(widths[column]) for column in shown)

    rows = [
        "  ".join(str(row[column]).rjust(widths[column]) for column in shown)
        for _, row in plan.iterrows()
    ]

    return "\n".join([header, *rows])
