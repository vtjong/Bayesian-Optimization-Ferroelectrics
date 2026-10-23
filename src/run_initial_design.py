"""Generate the campaign's initial design plan and blank laboratory results sheet.

The plan records the conditions fixed before the batch is fired. The results
sheet contains one row per planned condition plus the never-flashed reference.

Both outputs are write-once. Neither may be replaced after generation.

Usage:
    python src/run_initial_design.py
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from campaign.plan import (
    NOTE_COLUMN,
    ConditionBlock,
    build_plan,
    describe_plan,
    write_plan,
)
from campaign.results import blank_template
from design_space import in_box, is_snapped, min_separation
from physics.thermal_model import tmax_envelope

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPOSITORY_ROOT / "data"

PLAN_PATH = DATA_DIR / "flash_plan_initial.csv"
RESULTS_PATH = DATA_DIR / "flash_plan_initial_results.csv"

READOUT_KIND = "kappa_pristine"

DESIGN_BLOCK_LABEL = "S"

TEMPERATURE_LOW_COLUMN = "sim_Tmax_lo_C"
TEMPERATURE_HIGH_COLUMN = "sim_Tmax_hi_C"


@dataclass(frozen=True, slots=True)
class Condition:
    """One flash condition and the reason it is in the design.

    :param voltage_v: Flash voltage in V.
    :param pulse_width_ms: Pulse width in ms.
    :param reason: Why this condition is in the design, where it differs from
        the block's own note. Written to the plan so the justification travels
        with the row rather than living in a notebook.
    """

    voltage_v: float
    pulse_width_ms: float
    reason: str = ""


#: How the conditions below were arrived at. Recorded on every row of the plan.
DESIGN_PROVENANCE = "condition selected by the principal investigator"

#: The initial design. These conditions are selected, not generated: nothing in
#: this repository chose them, and no derived quantity may reject one. They are
#: checked against the instrument's settable box and resolution before a plan is
#: written, and their separation is reported for reading only.
CONDITIONS: tuple[Condition, ...] = (
    Condition(voltage_v=709.0, pulse_width_ms=2.8),
    Condition(voltage_v=655.0, pulse_width_ms=3.2),
    Condition(voltage_v=691.0, pulse_width_ms=3.6),
    Condition(voltage_v=572.0, pulse_width_ms=4.4),
    Condition(voltage_v=641.0, pulse_width_ms=4.9),
    Condition(voltage_v=674.0, pulse_width_ms=5.6),
    Condition(voltage_v=605.0, pulse_width_ms=5.9),
    Condition(voltage_v=626.0, pulse_width_ms=6.8),
    Condition(voltage_v=647.0, pulse_width_ms=8.7),
    Condition(voltage_v=669.0, pulse_width_ms=9.5),
    Condition(voltage_v=525.0, pulse_width_ms=1.0),
    Condition(voltage_v=525.0, pulse_width_ms=5.0),
    Condition(voltage_v=600.0, pulse_width_ms=1.0),
)


def thermal_annotations(
    voltage_v: np.ndarray,
    pulse_width_ms: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute post-design thermal-model annotations.

    The temperature band spans the available thermal models for the instrument.
    These values describe the selected conditions only; they take no part in
    choosing them.

    :param voltage_v: Flash voltages in V.
    :param pulse_width_ms: Pulse widths in ms.
    :return: Simulated temperature-band columns keyed by plan column name.
    """
    tmax_low_c, tmax_high_c = tmax_envelope(voltage_v, pulse_width_ms)

    return {
        TEMPERATURE_LOW_COLUMN: np.round(tmax_low_c, 1),
        TEMPERATURE_HIGH_COLUMN: np.round(tmax_high_c, 1),
    }


def check_settable(
    voltage_v: np.ndarray,
    pulse_width_ms: np.ndarray,
) -> None:
    """Verify every condition can actually be dialled in on the instrument.

    These are instrument facts rather than judgements, so a violation is an
    error rather than a warning. Nothing here consults a thermal model.

    :param voltage_v: Flash voltages in V.
    :param pulse_width_ms: Pulse widths in ms.
    :raises ValueError: If a condition lies outside the settable box or off the
        instrument's resolution.
    """
    outside = ~in_box(voltage_v, pulse_width_ms)
    if outside.any():
        listed = ", ".join(
            f"{v:g} V / {t:g} ms" for v, t in zip(voltage_v[outside], pulse_width_ms[outside])
        )
        raise ValueError(f"outside the settable box: {listed}")

    if not is_snapped(voltage_v, pulse_width_ms):
        raise ValueError("conditions are not on the instrument's 1 V and 0.1 ms resolution")


def build_design_block() -> ConditionBlock:
    """Build the condition block written to the plan.

    :return: The initial design as a single labelled block.
    :raises ValueError: If no condition has been declared.
    """
    if not CONDITIONS:
        raise ValueError("CONDITIONS is empty; declare the design before generating a plan")

    voltage_v = np.array([condition.voltage_v for condition in CONDITIONS], float)
    pulse_width_ms = np.array([condition.pulse_width_ms for condition in CONDITIONS], float)

    check_settable(voltage_v, pulse_width_ms)

    return ConditionBlock(
        label=DESIGN_BLOCK_LABEL,
        voltage_v=voltage_v,
        pulse_width_ms=pulse_width_ms,
        annotations={
            **thermal_annotations(voltage_v, pulse_width_ms),
        },
    )


def main() -> int:
    """Generate the immutable plan and laboratory results template.

    :return: Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    for path in (args.plan, args.results):
        if path.exists():
            print(f"{path} already exists; refusing to overwrite it")
            return 1

    block = build_design_block()
    plan = build_plan([block], READOUT_KIND)
    plan[NOTE_COLUMN] = [condition.reason or DESIGN_PROVENANCE for condition in CONDITIONS]

    write_plan(args.plan, plan)
    blank_template(plan).to_csv(args.results, index=False, mode="x")

    print(
        describe_plan(
            plan,
            annotations=(TEMPERATURE_LOW_COLUMN, TEMPERATURE_HIGH_COLUMN),
        )
    )
    separation = min_separation(block.voltage_v, block.pulse_width_ms)
    print(f"\n  closest pair separated by {separation:.3f} in normalized coordinates")
    print("  simulated temperatures are a band across thermal models, for reading only")
    print(f"\n  plan    -> {args.plan}")
    print(f"  results -> {args.results}")
    print("  the results sheet carries an as-deposited reference at index 0; do not flash it")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
