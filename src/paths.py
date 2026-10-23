"""Where the measured data lives.

Separate from ``physics.constants`` on purpose. Everything in that module is a claim about the
world -- a physical constant, a prior, a fitted shape parameter -- and its authority is the whole
reason it is grouped and annotated the way it is. A filesystem path is not a claim about the world,
and mixing the two means the design box cannot locate the measured grid without also importing
every kinetic assumption in the campaign.

Measured data is never mirrored as literals in the source: the CSVs are the single source of truth,
so the file and the code cannot drift apart.
"""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FLASH_TABLE_CSV = DATA_DIR / "flash_temp_table.csv"  # the CERA surface; the design box comes
# from its axes. A second, independent simulation of the same tool on the same 30 nodes, used
# only to bound how far apart the available thermal models are.
FLASH_TABLE_SKR_CSV = DATA_DIR / "flash_temp_table_skr.csv"
MEASURED_TRACE_CSV = DATA_DIR / "measured_transient.csv"  # not yet available

# Electrical characterisation of the flashed films: raw Keithley 4200A exports per round, the
# group's reference summaries, and the feature tables the electrical pipeline writes.
ELECTRICAL_DIR = DATA_DIR / "electrical"
ELECTRICAL_RAW_DIR = ELECTRICAL_DIR / "raw"
ELECTRICAL_REFERENCE_DIR = ELECTRICAL_DIR / "reference"
ELECTRICAL_FEATURES_DIR = ELECTRICAL_DIR / "features"
SUMMARY_ROUNDS_1_2_CSV = ELECTRICAL_REFERENCE_DIR / "summary_rounds1-2.csv"
SUMMARY_ROUND_1_EXTENDED_CSV = ELECTRICAL_REFERENCE_DIR / "summary_round1_extended.csv"
PROCESS_ROUNDS_1_2_CSV = ELECTRICAL_REFERENCE_DIR / "process_rounds1-2.csv"
CAMPAIGN_PER_CAPACITOR_CSV = ELECTRICAL_FEATURES_DIR / "campaign_per_capacitor.csv"


def round_raw_dir(round_number: int) -> Path:
    """Return the folder holding one round's raw exports, one `Sample<NN>` folder per film."""
    return ELECTRICAL_RAW_DIR / f"round{round_number}"


def round_features_dir(round_number: int) -> Path:
    """Return the folder holding one round's feature tables."""
    return ELECTRICAL_FEATURES_DIR / f"round{round_number}"
