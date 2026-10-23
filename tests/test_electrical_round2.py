"""Round 2 end to end: the group's summary sheet and the committed golden tables.

The three summary definitions (instrument ramp-only Psw, centred Pr crossing, kappa from the
0.8-2.25 V slope) were reverse-engineered from the group's sheet. Running the real Round-2
exports on every change is what keeps a shifted window or crossing rule from drifting the
numbers the correlation study is built on.
"""

import io

import numpy as np
import pandas as pd
import pytest

from electrical.pipeline import process_round
from electrical.report import flag_outliers
from paths import SUMMARY_ROUNDS_1_2_CSV, round_raw_dir

GOLDEN_DIR = (
    __import__("pathlib").Path(__file__).resolve().parent / "data" / "electrical" / "golden"
)
REGENERATE = "scripts/regenerate_electrical_golden.sh"

#: The group's Sample14 line (n = 10, no capacitor switches), reproduced exactly.
SAMPLE14_PSW_INST = (0.081695, 0.109722)
SAMPLE14_PR = (0.022336, 0.165866)
SAMPLE14_KAPPA = (26.344252, 1.285783)


@pytest.fixture(scope="module")
def round2():
    result = process_round(round_raw_dir(2), 2)
    capacitors = flag_outliers(result.capacitors)
    return capacitors, result.samples


def _pristine(samples: pd.DataFrame, sample: str) -> pd.Series:
    rows = samples[(samples["sample"] == sample) & (samples["state"] == "pris")]
    assert len(rows) == 1, f"{sample}: expected one pristine row, found {len(rows)}"
    return rows.iloc[0]


class TestTheGroupsSheetIsReproduced:
    def test_every_round2_sample_matches_the_summary_sheet(self, round2):
        _, samples = round2
        ref = pd.read_csv(SUMMARY_ROUNDS_1_2_CSV)
        ref = ref[ref["round"] == 2].set_index("sample_key")
        checked = 0
        for key, want in ref.iterrows():
            ours = _pristine(samples, key.capitalize())
            checked += 1
            assert ours["psw_inst_uc_cm2_mean"] == pytest.approx(
                want["Pristine_Psw_Mean"], abs=1e-3
            )
            # Sample15 sits 0.023 off: one of its eleven capacitors has no pristine P-V run.
            assert ours["pr_uc_cm2_mean"] == pytest.approx(want["Pristine_Pr_Mean"], abs=0.03)
            # Sample16, the inhomogeneous film, sits 0.21 off; the others agree within 0.05.
            assert ours["kappa_summary_mean"] == pytest.approx(
                want["Pristine_Kappa_Mean"], abs=0.25
            )
            assert ours["n_capacitors"] == want["N_Capacitors"]
        assert checked == 6

    def test_sample14_reproduces_the_sheet_exactly(self, round2):
        _, samples = round2
        r = _pristine(samples, "Sample14")
        assert (r["psw_inst_uc_cm2_mean"], r["psw_inst_uc_cm2_sd"]) == pytest.approx(
            SAMPLE14_PSW_INST, abs=1e-3
        )
        assert (r["pr_uc_cm2_mean"], r["pr_uc_cm2_sd"]) == pytest.approx(SAMPLE14_PR, abs=5e-3)
        assert (r["kappa_summary_mean"], r["kappa_summary_sd"]) == pytest.approx(
            SAMPLE14_KAPPA, abs=0.02
        )
        assert r["n_capacitors"] == 10 and r["n_switching"] == 0

    def test_sample16_switches_wakes_up_and_splits_at_the_dip(self, round2):
        capacitors, _ = round2
        s16 = capacitors[(capacitors["sample"] == "Sample16") & (capacitors["state"] == "pris")]
        assert len(s16) == 10 and s16["switching"].all()
        assert s16["split_raw_n_dip_found"].all()
        assert s16["split_raw_n_dip_fraction_main"].between(0.59, 0.72).all()
        assert s16["split_raw_n_crossover_fraction_main"].between(0.62, 0.72).all()
        assert s16["split_raw_n_linear_fraction_main"].between(0.68, 0.80).all()
        assert s16["endurance_wake_ratio"].between(1.7, 2.0).all()
        assert s16["endurance_fatigue_ratio"].between(0.95, 1.05).all()


class TestNothingIsSilentlyDropped:
    def test_every_run_was_read(self, round2):
        capacitors, samples = round2
        assert len(capacitors) == 112
        assert sorted(samples["state"].unique()) == ["post1e6", "pris"]
        bad = capacitors["has_error"]
        assert not bad.any(), capacitors.loc[bad, ["sample", "cap", "state", "issues"]]

    def test_endurance_cycle_zero_agrees_with_the_pristine_pund(self, round2):
        capacitors, _ = round2
        sw = capacitors[capacitors["switching"] & (capacitors["state"] == "pris")]
        rel = (sw["endurance_psw_cycle0_uc_cm2"] - sw["psw_inst_uc_cm2"]).abs() / sw[
            "psw_inst_uc_cm2"
        ]
        assert rel.max() < 0.15, rel.describe()


class TestGoldenTables:
    @pytest.mark.parametrize("name", ["round2_per_capacitor", "round2_per_sample"])
    def test_committed_table_matches_the_pipeline(self, round2, name):
        capacitors, samples = round2
        table = capacitors if name.endswith("capacitor") else samples
        ours = pd.read_csv(io.StringIO(table.to_csv(index=False)))  # what the pipeline writes
        golden = pd.read_csv(GOLDEN_DIR / f"{name}.csv")
        assert list(golden.columns) == list(ours.columns), (
            f"columns changed; if intended run {REGENERATE} and review the diff"
        )
        for column in golden.columns:
            theirs, mine = golden[column], ours[column].reset_index(drop=True)
            if pd.api.types.is_numeric_dtype(theirs) and pd.api.types.is_numeric_dtype(mine):
                same = np.allclose(
                    theirs.fillna(np.nan), mine.fillna(np.nan), rtol=1e-6, equal_nan=True
                )
            else:
                same = theirs.fillna("").astype(str).equals(mine.fillna("").astype(str))
            assert same, (
                f"{name}.csv column {column!r} differs from the pipeline; every cell that moved "
                f"is a behaviour change. If intended, run {REGENERATE} and review the diff."
            )
