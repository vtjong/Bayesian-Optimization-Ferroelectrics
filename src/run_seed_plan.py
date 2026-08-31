"""Generate the campaign seed plan and blank laboratory results sheet.

The plan records the conditions fixed before the batch is fired. The results
sheet contains one row per planned condition plus the never-flashed reference.

Both outputs are write-once. Neither may be replaced after generation.

Usage:
    python src/run_seed_plan.py [--no-bridge]
"""

import argparse
from pathlib import Path

import numpy as np

import seed
from campaign.plan import ConditionBlock, build_plan, describe_plan, write_plan
from campaign.results import blank_template
from physics.thermal_model import tmax_envelope

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPOSITORY_ROOT / "data"

PLAN_PATH = DATA_DIR / "campaign_seed.csv"
RESULTS_PATH = DATA_DIR / "campaign_seed_results.csv"

READOUT_KIND = "eps_r"

SEED_BLOCK_LABEL = "S"
BRIDGE_BLOCK_LABEL = "B"

# Pilot condition refired with the seed to distinguish fabrication-run effects
# from process dependence. This is the lowest-fluence pilot condition observed
# to crystallize.
BRIDGE_VOLTAGE_V = 572.0
BRIDGE_PULSE_WIDTH_MS = 4.4

TEMPERATURE_LOW_COLUMN = "sim_Tmax_lo_C"
TEMPERATURE_HIGH_COLUMN = "sim_Tmax_hi_C"


def thermal_annotations(
    voltage_v: np.ndarray,
    pulse_width_ms: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute post-design thermal-model annotations.

    The temperature band spans the available thermal models for the instrument.
    These values describe the selected conditions only; they do not influence
    seed generation or selection.

    :param voltage_v: Flash voltages in V.
    :param pulse_width_ms: Pulse widths in ms.
    :return: Simulated temperature-band columns keyed by plan column name.
    """
    tmax_low_c, tmax_high_c = tmax_envelope(
        voltage_v,
        pulse_width_ms,
    )

    return {
        TEMPERATURE_LOW_COLUMN: np.round(tmax_low_c, 1),
        TEMPERATURE_HIGH_COLUMN: np.round(tmax_high_c, 1),
    }


def build_seed_blocks(
    include_bridge: bool = True,
) -> list[ConditionBlock]:
    """Build the condition blocks written to the seed plan.

    :param include_bridge: Whether to include the pilot bridge condition.
    :return: Seed and optional bridge condition blocks.
    """
    voltage_v, pulse_width_ms = seed.make_seed()

    blocks = [
        ConditionBlock(
            label=SEED_BLOCK_LABEL,
            voltage_v=voltage_v,
            pulse_width_ms=pulse_width_ms,
            note=("prespecified maximin Latin hypercube over the " "full settable range"),
            annotations=thermal_annotations(
                voltage_v,
                pulse_width_ms,
            ),
        )
    ]

    if not include_bridge:
        return blocks

    bridge_voltage_v = np.array([BRIDGE_VOLTAGE_V])
    bridge_pulse_width_ms = np.array([BRIDGE_PULSE_WIDTH_MS])

    blocks.append(
        ConditionBlock(
            label=BRIDGE_BLOCK_LABEL,
            voltage_v=bridge_voltage_v,
            pulse_width_ms=bridge_pulse_width_ms,
            note=(
                "pilot bridge: refire of the lowest-fluence pilot " "condition that crystallized"
            ),
            annotations=thermal_annotations(
                bridge_voltage_v,
                bridge_pulse_width_ms,
            ),
        )
    )

    return blocks


def main() -> int:
    """Generate the immutable seed plan and laboratory results template.

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

    for path in (args.plan, args.results):
        if path.exists():
            print(f"{path} already exists; refusing to overwrite it")
            return 1

    plan = build_plan(
        build_seed_blocks(include_bridge=not args.no_bridge),
        READOUT_KIND,
    )

    write_plan(args.plan, plan)
    blank_template(plan).to_csv(
        args.results,
        index=False,
        mode="x",
    )

    print(
        describe_plan(
            plan,
            annotations=(
                TEMPERATURE_LOW_COLUMN,
                TEMPERATURE_HIGH_COLUMN,
            ),
        )
    )
    print(
        f"\n  seed drawn with RNG_SEED = {seed.RNG_SEED}, "
        f"{seed.MAXIMIN_DRAWS} candidates scored"
    )
    print("  simulated temperatures are a band across thermal models, " "for reading only")
    print(f"\n  plan    -> {args.plan}")
    print(f"  results -> {args.results}")
    print("  the results sheet carries an as-deposited reference at index 0; " "do not flash it")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
