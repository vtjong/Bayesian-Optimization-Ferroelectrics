"""Tests for plan assembly and serialization.

A plan is the record of what a batch was intended to be, so the properties
asserted here are about faithfulness: every condition survives, indices are
unique and contiguous, blocks keep their labels, and a written plan is never
silently replaced.
"""

import numpy as np
import pandas as pd
import pytest

from campaign.plan import (
    BLOCK_COLUMN,
    FIRST_CONDITION_INDEX,
    INDEX_COLUMN,
    PULSE_WIDTH_COLUMN,
    READOUT_KIND_COLUMN,
    VOLTAGE_COLUMN,
    ConditionBlock,
    build_plan,
    describe_plan,
    write_plan,
)

READOUT_KIND = "eps_r"


@pytest.fixture
def seed_block() -> ConditionBlock:
    """Return a three-condition block standing in for a seed."""
    return ConditionBlock(
        label="S",
        voltage_v=np.array([510.0, 620.0, 700.0]),
        pulse_width_ms=np.array([0.1, 1.5, 8.0]),
        note="seed",
    )


@pytest.fixture
def bridge_block() -> ConditionBlock:
    """Return a one-condition block standing in for a pilot bridge."""
    return ConditionBlock(
        label="B",
        voltage_v=np.array([572.0]),
        pulse_width_ms=np.array([4.4]),
        note="pilot bridge",
    )


def test_every_condition_survives_assembly(seed_block, bridge_block) -> None:
    """No condition is dropped, reordered within a block, or altered."""
    plan = build_plan([seed_block, bridge_block], READOUT_KIND)

    assert len(plan) == len(seed_block) + len(bridge_block)
    assert np.array_equal(
        plan[VOLTAGE_COLUMN].to_numpy()[: len(seed_block)],
        seed_block.voltage_v,
    )
    assert np.array_equal(
        plan[PULSE_WIDTH_COLUMN].to_numpy()[len(seed_block) :],
        bridge_block.pulse_width_ms,
    )


def test_indices_are_contiguous_and_reserve_the_control_row(seed_block, bridge_block) -> None:
    """Indices run from the first condition index with no gaps.

    Index 0 is never assigned here: it belongs to the never-flashed reference,
    which the results template inserts.
    """
    plan = build_plan([seed_block, bridge_block], READOUT_KIND)

    expected = np.arange(len(plan)) + FIRST_CONDITION_INDEX

    assert np.array_equal(plan[INDEX_COLUMN].to_numpy(), expected)
    assert plan[INDEX_COLUMN].min() > 0


def test_blocks_keep_their_labels(seed_block, bridge_block) -> None:
    """A row can always be traced back to why it was fired."""
    plan = build_plan([seed_block, bridge_block], READOUT_KIND)

    labels = plan[BLOCK_COLUMN].to_numpy()

    assert set(labels) == {"S", "B"}
    assert list(labels).count("S") == len(seed_block)


def test_a_repeated_setpoint_is_allowed(seed_block) -> None:
    """A bridge condition deliberately repeats a setpoint already fired."""
    repeat = ConditionBlock(
        label="B",
        voltage_v=seed_block.voltage_v[:1].copy(),
        pulse_width_ms=seed_block.pulse_width_ms[:1].copy(),
        note="deliberate repeat",
    )

    plan = build_plan([seed_block, repeat], READOUT_KIND)

    assert len(plan) == len(seed_block) + 1
    assert plan[INDEX_COLUMN].is_unique


def test_annotations_are_carried_through(seed_block) -> None:
    """Reporting columns reach the CSV without becoming setpoints."""
    annotated = ConditionBlock(
        label="S",
        voltage_v=seed_block.voltage_v,
        pulse_width_ms=seed_block.pulse_width_ms,
        annotations={"sim_Tmax_lo_C": np.array([100.0, 200.0, 300.0])},
    )

    plan = build_plan([annotated], READOUT_KIND)

    assert "sim_Tmax_lo_C" in plan.columns
    assert plan.columns.get_loc("sim_Tmax_lo_C") > plan.columns.get_loc(READOUT_KIND_COLUMN)


def test_a_ragged_block_is_rejected() -> None:
    """Columns of unequal length would silently misalign conditions."""
    with pytest.raises(ValueError, match="pulse widths"):
        ConditionBlock(
            label="S",
            voltage_v=np.array([510.0, 620.0]),
            pulse_width_ms=np.array([0.1]),
        )


def test_a_ragged_annotation_is_rejected() -> None:
    """An annotation of the wrong length would attach to the wrong condition."""
    with pytest.raises(ValueError, match="annotation"):
        ConditionBlock(
            label="S",
            voltage_v=np.array([510.0, 620.0]),
            pulse_width_ms=np.array([0.1, 1.0]),
            annotations={"sim_Tmax_lo_C": np.array([100.0])},
        )


def test_an_empty_plan_is_rejected(seed_block) -> None:
    """A plan with no conditions is a mistake, not an empty batch."""
    empty = ConditionBlock(
        label="S",
        voltage_v=np.array([]),
        pulse_width_ms=np.array([]),
    )

    with pytest.raises(ValueError, match="at least one condition"):
        build_plan([empty], READOUT_KIND)


def test_writing_round_trips(tmp_path, seed_block) -> None:
    """A written plan reads back with the same setpoints."""
    plan = build_plan([seed_block], READOUT_KIND)
    path = write_plan(tmp_path / "plan.csv", plan)

    reloaded = pd.read_csv(path)

    assert np.allclose(reloaded[VOLTAGE_COLUMN], plan[VOLTAGE_COLUMN])
    assert np.allclose(reloaded[PULSE_WIDTH_COLUMN], plan[PULSE_WIDTH_COLUMN])


def test_writing_refuses_to_overwrite(tmp_path, seed_block) -> None:
    """A fired plan is a record; replacing it would destroy the only evidence."""
    plan = build_plan([seed_block], READOUT_KIND)
    path = write_plan(tmp_path / "plan.csv", plan)

    with pytest.raises(FileExistsError):
        write_plan(path, plan)


def test_description_lists_every_condition(seed_block, bridge_block) -> None:
    """The terminal rendering shows one row per condition plus a header."""
    plan = build_plan([seed_block, bridge_block], READOUT_KIND)

    rendered = describe_plan(plan)

    assert len(rendered.splitlines()) == len(plan) + 1
