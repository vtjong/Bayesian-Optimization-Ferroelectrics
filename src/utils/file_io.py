"""Write CSV tables once or atomically, and read one back against the columns it must have."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

#: Cell prefixes a spreadsheet would execute; such text is quoted before it reaches a CSV.
FORMULA_PREFIXES = ("=", "+", "-", "@")


class ContractError(ValueError):
    """A table on disk does not have the columns this version of the code writes."""


def guard_formulas(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with text cells a spreadsheet would run as formulas quoted by an apostrophe."""
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


def write_csv(path: Path, frame: pd.DataFrame, overwrite: bool = False) -> Path:
    """Write a table, refusing to replace an existing file unless asked.

    :raises FileExistsError: If the file exists and ``overwrite`` is false.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8", newline="") as destination:
        guard_formulas(frame).to_csv(destination, index=False)
    return path


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> Path:
    """Write a table to a temporary file beside the destination, then rename it over.

    A crash part-way leaves the previous file intact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as destination:
            guard_formulas(frame).to_csv(destination, index=False)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def read_csv_with_columns(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """Read a table that must have exactly ``columns``, returned in that order.

    :raises ContractError: If columns are missing or unexpected.
    """
    table = pd.read_csv(path)
    missing = [c for c in columns if c not in table.columns]
    unexpected = [c for c in table.columns if c not in columns]
    if missing or unexpected:
        raise ContractError(
            f"{path} does not match the expected columns: "
            f"missing {missing}, unexpected {unexpected}"
        )
    return table[list(columns)]
