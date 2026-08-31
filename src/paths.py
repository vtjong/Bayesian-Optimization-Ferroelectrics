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
FLASH_TABLE_CSV = DATA_DIR / "flash_temp_table.csv"
MEASURED_TRACE_CSV = DATA_DIR / "measured_transient.csv"  # not yet available
