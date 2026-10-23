"""Tables on disk: per-round outputs written once, and the campaign table updated in place.

The campaign table holds every round's per-capacitor rows and is what the correlation study
reads, so it is the one file this package rewrites: a round's rows replace any earlier rows
with the same (round, sample_key, cap, state), and the write is atomic so a crash leaves the
previous table intact. Its columns are checked against the contract on every read.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

from electrical.aggregate import DERIVED_COLUMNS
from electrical.capacitor import CAPACITOR_COLUMNS, CAPACITOR_KEY
from electrical.issues import ContractError
from electrical.report import OUTLIER_FLAGS_COLUMN

#: Columns of the per-capacitor table as written: the contract plus the ratios and the flags.
TABLE_COLUMNS: tuple[str, ...] = (*CAPACITOR_COLUMNS, *DERIVED_COLUMNS, OUTLIER_FLAGS_COLUMN)
#: Cell prefixes a spreadsheet would execute; such text is quoted before it reaches a CSV.
FORMULA_PREFIXES = ("=", "+", "-", "@")


def guard_formulas(frame: pd.DataFrame) -> pd.DataFrame:
    """Prefix text cells that a spreadsheet would read as formulas with an apostrophe."""
    frame = frame.copy()
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(
            frame[column]
        ):
            continue
        text = frame[column]
        is_formula = text.map(lambda v: isinstance(v, str) and v.startswith(FORMULA_PREFIXES))
        if is_formula.any():
            frame[column] = text.where(~is_formula, "'" + text.astype(str))
    return frame


def write_table(path: Path, frame: pd.DataFrame, overwrite: bool = False) -> Path:
    """Write one output table, refusing to replace an existing file unless asked.

    :param path: Destination CSV; its folder is created.
    :param frame: The table.
    :param overwrite: Replace an existing file instead of refusing.
    :return: The path written.
    :raises FileExistsError: If the file exists and ``overwrite`` is false.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8", newline="") as destination:
        guard_formulas(frame).to_csv(destination, index=False)
    return path


def read_campaign(path: Path) -> pd.DataFrame:
    """Read the campaign table, checking it has exactly the columns this version writes.

    :raises ContractError: If columns are missing or unexpected.
    """
    table = pd.read_csv(path)
    missing = [c for c in TABLE_COLUMNS if c not in table.columns]
    unexpected = [c for c in table.columns if c not in TABLE_COLUMNS]
    if missing or unexpected:
        raise ContractError(
            f"{path} does not match the per-capacitor contract: missing {missing}, "
            f"unexpected {unexpected}. Rebuild it by re-running every round."
        )
    return table[list(TABLE_COLUMNS)]


def merge_round(existing: pd.DataFrame, capacitors: pd.DataFrame) -> pd.DataFrame:
    """Replace the rows of ``existing`` that share a key with ``capacitors``, keeping the rest."""
    key = list(CAPACITOR_KEY)
    incoming = capacitors.set_index(key).index
    kept = existing[~existing.set_index(key).index.isin(incoming)]
    merged = pd.concat([kept, capacitors[list(TABLE_COLUMNS)]], ignore_index=True)
    return merged.sort_values(
        ["round", "sample_num", "sample_key", "row", "col", "state"]
    ).reset_index(drop=True)


def write_campaign(path: Path, table: pd.DataFrame) -> Path:
    """Write the campaign table atomically: to a temporary file beside it, then renamed over it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as destination:
            guard_formulas(table).to_csv(destination, index=False)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def update_campaign(path: Path, capacitors: pd.DataFrame) -> pd.DataFrame:
    """Merge one round's flagged per-capacitor rows into the campaign table and write it back."""
    existing = read_campaign(path) if path.exists() else pd.DataFrame(columns=list(TABLE_COLUMNS))
    merged = merge_round(existing, capacitors)
    write_campaign(path, merged)
    return merged
