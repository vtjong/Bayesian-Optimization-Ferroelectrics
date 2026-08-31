"""Generate the campaign seed plan and the blank sheet the laboratory fills in.

Writes two files: the plan, which is the record of what this batch was intended
to be, and a results template carrying one row per condition plus the
never-flashed reference.

Neither file may be regenerated once the batch is fired. The plan is fixed
before firing so that the design cannot be adjusted in the light of its own
outcome, and ``campaign.plan.write_plan`` refuses to overwrite an existing file.

Usage:  python src/run_seed_plan.py [--no-bridge]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(Path(__file__).resolve().parent))

import seed  # noqa: E402
from campaign.plan import ConditionBlock, build_plan, describe_plan, write_plan  # noqa: E402
from campaign.results import blank_template  # noqa: E402
from physics.thermal_model import tmax_envelope  # noqa: E402

DATA_DIR = REPOSITORY_ROOT / "data"
PLAN_PATH = DATA_DIR / "campaign_seed.csv"
RESULTS_PATH = DATA_DIR / "campaign_seed_results.csv"

#: Instrument reading recorded for every condition. Constant across the plan so
#: that one sheet is read one way.
READOUT_KIND = "eps_r"

SEED_BLOCK_LABEL = "S"
BRIDGE_BLOCK_LABEL = "B"

#: The pilot condition refired alongside this seed. The pilot specimens and this
#: batch come from different fabrication runs, so without a condition common to
#: both, a wafer difference cannot be distinguished from a process dependence.
#: This is the lowest-fluence pilot condition observed to crystallize.
BRIDGE_VOLTAGE_V = 572.0
BRIDGE_PULSE_WIDTH_MS = 4.4

TEMPERATURE_LOW_COLUMN = "sim_Tmax_lo_C"
TEMPERATURE_HIGH_COLUMN = "sim_Tmax_hi_C"


def temperature_annotations(
    voltage_v: np.ndarray,
    pulse_width_ms: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the simulated peak-temperature band for a set of conditions.

    Attached to the plan for reading only. The band spans the available thermal
    models of this instrument, which differ by roughly a factor of two in
    temperature rise; a single value would state a precision no measurement
    supports. Nothing downstream reads these columns, and no design decision
    depends on them.

    :param voltage_v: Flash voltages in V.
    :param pulse_width_ms: Pulse widths in ms.
    :return: Annotation columns keyed by name.
    """
    tmax_low_c, tmax_high_c = tmax_envelope(voltage_v, pulse_width_ms)

    return {
        TEMPERATURE_LOW_COLUMN: np.round(tmax_low_c, 1),
        TEMPERATURE_HIGH_COLUMN: np.round(tmax_high_c, 1),
    }


def seed_blocks(include_bridge: bool = True) -> list[ConditionBlock]:
    """Assemble the blocks that make up the seed batch.

    :param include_bridge: Whether to refire the pilot bridge condition.
    :return: Condition blocks in the order they are written.
    """
    voltage_v, pulse_width_ms = seed.make_seed()

    blocks = [
        ConditionBlock(
            label=SEED_BLOCK_LABEL,
            voltage_v=voltage_v,
            pulse_width_ms=pulse_width_ms,
            note="prespecified maximin Latin hypercube over the full settable range",
            annotations=temperature_annotations(voltage_v, pulse_width_ms),
        )
    ]

    if include_bridge:
        bridge_voltage_v = np.array([BRIDGE_VOLTAGE_V])
        bridge_pulse_width_ms = np.array([BRIDGE_PULSE_WIDTH_MS])
        blocks.append(
            ConditionBlock(
                label=BRIDGE_BLOCK_LABEL,
                voltage_v=bridge_voltage_v,
                pulse_width_ms=bridge_pulse_width_ms,
                note="pilot bridge: refire of the lowest-fluence pilot condition that crystallized",
                annotations=temperature_annotations(bridge_voltage_v, bridge_pulse_width_ms),
            )
        )

    return blocks


def main() -> int:
    """Write the seed plan and its blank results sheet.

    :return: Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument(
        "--no-bridge",
        action="store_true",
        help="omit the pilot bridge condition",
    )
    args = parser.parse_args()

    if args.results.exists():
        print(f"{args.results} already exists; refusing to replace a sheet in circulation")
        return 1

    plan = build_plan(seed_blocks(include_bridge=not args.no_bridge), READOUT_KIND)

    write_plan(args.plan, plan)
    blank_template(plan).to_csv(args.results, index=False)

    print(
        describe_plan(
            plan,
            annotations=(TEMPERATURE_LOW_COLUMN, TEMPERATURE_HIGH_COLUMN),
        )
    )
    print(f"\n  seed drawn with RNG_SEED = {seed.RNG_SEED}, {seed.MAXIMIN_DRAWS} candidates scored")
    print("  simulated temperatures are a band across thermal models, for reading only")
    print(f"\n  plan    -> {args.plan}")
    print(f"  results -> {args.results}")
    print("  the results sheet carries an as-deposited reference at index 0; do not flash it")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
