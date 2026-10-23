"""Find the measurement files of one round and read what their names and folders say.

A round is a folder of ``Sample<NN>/`` folders. Each export's name carries the measurement
kind, a state token, the capacitor and its side length; the folder carries the sample:

    PUND_pris_RxCy_30um.xls         PUND, one run
    PUND_pris_1e6_RxCy_30um.xls     PUND, the pristine run, then the run after 1e6 cycles
    PV_pris_1e6_RxCy_30um.xls       P-V sweep, same pattern
    END_1e6_RxCy_30um.xls           endurance log over 1e6 cycles

Names typed by hand vary in case, separator and a trailing ``_x`` for a repeated measurement.
All of them parse; those that depart from the instrument's convention are marked ``suspect``.
Nothing here opens a file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

FILE_NAME_RE = re.compile(
    r"^(?P<kind>PUND|PV|END)[_-](?P<tag>.+?)[_-](?P<cap>R\d+C\d+)[_-](?P<size>\d+)um"
    r"(?P<suffix>[_-]x)?\.xls$",
    re.IGNORECASE,
)
CAPACITOR_RE = re.compile(r"R(?P<row>\d+)C(?P<col>\d+)", re.IGNORECASE)
SAMPLE_RE = re.compile(r"sample\s*_?(?P<number>\d+)", re.IGNORECASE)
CYCLES_RE = re.compile(r"^(\d+e\d+|\d{3,})$", re.IGNORECASE)
#: Folders that never hold exports.
SKIPPED_FOLDERS = ("__MACOSX", "Analysis_Plots")


class RoundLayoutError(Exception):
    """The folder is not one round of measurements, or two files claim the same run."""


class MeasurementKind(Enum):
    PUND = "PUND"
    PV = "PV"
    END = "END"


@dataclass(frozen=True, slots=True, kw_only=True)
class CapacitorId:
    """Position of a capacitor on the die."""

    row: int
    col: int

    @property
    def label(self) -> str:
        """Return the ``RxCy`` label the files use."""
        return f"R{self.row}C{self.col}"


@dataclass(frozen=True, slots=True, kw_only=True)
class MeasurementFile:
    """One export, as its name and folder describe it.

    :param path: The workbook.
    :param kind: PUND, P-V or endurance.
    :param state_token: Lower-case state token from the name (``"pris"``), or ``None``.
    :param cycles_tag: Cycle count from the name (``"1e6"``), or ``None``.
    :param capacitor: Capacitor the file belongs to.
    :param side_um: Capacitor side length from the name, um.
    :param sample_name: Folder name as written.
    :param sample_key: Normalised sample key used for joins (``"sample16"``).
    :param sample_number: Sample number from the folder name, or ``None``.
    :param suspect: Whether the name departs from the instrument's own convention.
    """

    path: Path
    kind: MeasurementKind
    state_token: str | None
    cycles_tag: str | None
    capacitor: CapacitorId
    side_um: int
    sample_name: str
    sample_key: str
    sample_number: int | None
    suspect: bool

    @property
    def order(self) -> tuple:
        """Sort key: sample number, then capacitor row and column, then file name."""
        number = self.sample_number if self.sample_number is not None else float("inf")
        return (number, self.sample_key, self.capacitor.row, self.capacitor.col, self.path.name)

    def identity(self) -> dict[str, object]:
        """Return the columns that say which sample and capacitor a table row is about."""
        return {
            "sample": self.sample_name,
            "sample_key": self.sample_key,
            "sample_num": self.sample_number,
            "cap": self.capacitor.label,
            "row": self.capacitor.row,
            "col": self.capacitor.col,
            "side_um": self.side_um,
        }

    def where(self, sheet: str | None = None) -> dict[str, object]:
        """Return the location of an issue found in this file, optionally in one sheet."""
        location = {
            "sample_key": self.sample_key,
            "capacitor": self.capacitor.label,
            "path": self.path,
        }
        if sheet is not None:
            location["sheet"] = sheet
        return location


def sample_key(folder_name: str) -> str:
    """Normalise a folder name for joins (``Sample 11`` -> ``sample11``).

    ``Sample2_square`` and ``Sample4_again`` become ``sample2`` and ``sample4``; a folder with
    no sample number keeps its whole name, lower-cased and without spaces.
    """
    match = SAMPLE_RE.search(folder_name)
    if match:
        return f"sample{int(match['number'])}"
    return re.sub(r"\s+", "", folder_name).lower()


def parse_file_name(path: Path) -> MeasurementFile | None:
    """Describe an export from its name and folder, or return ``None`` for any other file."""
    match = FILE_NAME_RE.match(path.name)
    if match is None:
        return None
    tokens = [token for token in re.split(r"[_-]", match["tag"]) if token]
    cycles = [token for token in tokens if CYCLES_RE.match(token)]
    states = [token for token in tokens if not CYCLES_RE.match(token)]
    kind = MeasurementKind(match["kind"].upper())
    tidy = (
        match["suffix"] is None
        and "-" not in path.name
        and match["kind"].isupper()
        and match["cap"] == match["cap"].upper()
        and all(token == token.lower() for token in tokens)
        and (bool(states) or kind is MeasurementKind.END)
    )
    capacitor = CAPACITOR_RE.match(match["cap"])
    folder = path.parent.name
    number = SAMPLE_RE.search(folder)
    return MeasurementFile(
        path=path,
        kind=kind,
        state_token=states[0].lower() if states else None,
        cycles_tag=cycles[0].lower() if cycles else None,
        capacitor=CapacitorId(row=int(capacitor["row"]), col=int(capacitor["col"])),
        side_um=int(match["size"]),
        sample_name=folder,
        sample_key=sample_key(folder),
        sample_number=int(number["number"]) if number else None,
        suspect=not tidy,
    )


def find_measurement_files(data_dir: Path) -> list[MeasurementFile]:
    """Return every export under ``data_dir``, in sample and capacitor order.

    :param data_dir: A round folder (``Sample*/`` folders inside) or one sample folder.
    :raises RoundLayoutError: If no export is found.
    """
    found = []
    for path in sorted(data_dir.rglob("*.xls")):
        if any(part in SKIPPED_FOLDERS for part in path.relative_to(data_dir).parts):
            continue
        parsed = parse_file_name(path)
        if parsed is not None:
            found.append(parsed)
    if not found:
        raise RoundLayoutError(f"no PUND, PV or END .xls export under {data_dir}")
    return sorted(found, key=lambda file: file.order)
