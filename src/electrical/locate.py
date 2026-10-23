"""Walk a round directory, find every PUND / P-V / endurance export, read Settings metadata.

File naming seen so far (Keithley 4200A KTEI exports):
    PUND_pris_RxCy_30um.xls          one run (sheet Data)                       Sample14
    PUND_pris_1e6_RxCy_30um.xls      two runs: Data = initial (pristine),       Round 2
                                     Append1 = after the 1e6-cycle endurance
    PV_pris_RxCy_30um.xls / PV_pris_1e6_RxCy_30um.xls   same pattern
    END_1e6_RxCy_30um.xls            pundEndurance: P/U/N/D charges vs cycle count
    PV_1e6_RxCy_30um.xls             a stand-alone post-cycling P-V

Each run becomes one record (`RawFile`) with `state`: the file's own token for the
initial run ("pris"), and "post<cycles>" for an Append sheet.  Nothing about the
timing is assumed: dt, n and the instrument parameters are read from each file and
recorded in `raw_index.csv`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

FILE_RE = re.compile(
    r"^(?P<kind>PUND|PV|END)_(?P<tag>[A-Za-z0-9_]+?)_(?P<cap>R\d+C\d+)_(?P<size>\d+)um\.xls$"
)
CAP_RE = re.compile(r"R(?P<row>\d+)C(?P<col>\d+)")
SAMPLE_RE = re.compile(r"(?i)sample\s*_?(?P<num>\d+)")
CYCLES_RE = re.compile(r"(\d+e\d+|\d{3,})")
EXPECTED_MODULE = {"PUND": "pundTest", "PV": "doubleSweepSeg", "END": "pundEndurance"}


@dataclass
class RawFile:
    path: str
    kind: str  # "PUND" | "PV" | "END"
    state: str  # "pris" | "post1e6" | ...
    run: str  # sheet name holding this run: "Data", "Append1", ...
    cap: str
    row: int
    col: int
    size_um: int
    sample: str
    sample_num: int | None
    cycles_tag: str | None = None
    module: str | None = None
    executed: datetime | None = None
    params: dict = field(default_factory=dict)
    n_points: int | None = None
    dt_s: float | None = None
    t_end_s: float | None = None
    problems: list = field(default_factory=list)

    def index_row(self):
        return {
            "sample": self.sample,
            "sample_num": self.sample_num,
            "cap": self.cap,
            "row": self.row,
            "col": self.col,
            "kind": self.kind,
            "state": self.state,
            "run": self.run,
            "size_um": self.size_um,
            "module": self.module,
            "executed": self.executed.isoformat() if self.executed else None,
            "n_points": self.n_points,
            "dt_s": self.dt_s,
            "t_end_s": self.t_end_s,
            "params": ";".join(f"{k}={v}" for k, v in self.params.items()),
            "problems": ";".join(self.problems),
            "path": self.path,
        }


def _parse_dt(raw):
    if not isinstance(raw, str):
        return None
    for fmt in ("%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
        try:
            return datetime.strptime(raw.strip(), fmt)
        except ValueError:
            pass
    return None


def parse_settings(path: str) -> dict[str, dict]:
    """{sheet: {"module", "executed", "params"}} from the Settings sheet.

    The sheet holds one block per run, separated by '====' lines and titled
    'Latest Run', 'Append 1 / Latest Run' or 'Initial Run'. 'Initial Run' is the
    Data sheet; 'Append k' is sheet Appendk; a lone 'Latest Run' is Data.
    """
    s = pd.read_excel(path, sheet_name="Settings", engine="xlrd", header=None)
    col0 = s[0].fillna("").astype(str).str.strip()
    blocks, cur = [], None
    for i in range(len(s)):
        v = col0.iloc[i]
        if v.startswith("===="):
            continue
        if v in ("Latest Run", "Initial Run") or v.startswith("Append"):
            cur = {"title": v, "rows": []}
            blocks.append(cur)
            continue
        if cur is not None:
            cur["rows"].append(i)
    out = {}
    for b in blocks:
        kv = {col0.iloc[i]: s.iloc[i, 1] for i in b["rows"]}
        params = {
            col0.iloc[i]: s.iloc[i, 3]
            for i in b["rows"]
            if str(s.iloc[i, 1]).strip() == "Input" and "size" not in col0.iloc[i]
        }
        m = re.match(r"Append\s*(\d+)", b["title"])
        sheet = f"Append{m.group(1)}" if m else "Data"
        out[sheet] = {
            "module": str(kv.get("Module Name", "")).strip() or None,
            "executed": _parse_dt(kv.get("Last Executed")),
            "params": params,
        }
    return out


def read_data(path: str, sheet: str = "Data") -> pd.DataFrame:
    """A run's sheet: header row + arrays; instrument scalars live on the first data row."""
    return pd.read_excel(path, sheet_name=sheet, engine="xlrd")


def scan(data_dir: str) -> list[RawFile]:
    """Every PUND/PV/END run under data_dir (sample = the folder the file sits in)."""
    found = []
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = [d for d in dirs if not d.startswith("__MACOSX") and d != "Analysis_Plots"]
        for f in sorted(files):
            m = FILE_RE.match(f)
            if not m:
                continue
            sample = os.path.basename(root)
            sm = SAMPLE_RE.search(sample)
            cm = CAP_RE.match(m["cap"])
            tag = m["tag"]
            cyc = CYCLES_RE.search(tag)
            cycles_tag = cyc.group(1) if cyc else None
            base_state = re.sub(r"_?(\d+e\d+|\d{3,})", "", tag).strip("_") or (
                "cycled" if m["kind"] != "END" else "end"
            )
            path = os.path.join(root, f)
            try:
                settings = parse_settings(path)
                sheets = pd.ExcelFile(path, engine="xlrd").sheet_names
            except Exception as e:  # noqa: BLE001
                rf = RawFile(
                    path,
                    m["kind"],
                    base_state,
                    "Data",
                    m["cap"],
                    int(cm["row"]),
                    int(cm["col"]),
                    int(m["size"]),
                    sample,
                    int(sm["num"]) if sm else None,
                    cycles_tag,
                )
                rf.problems.append(f"read error: {type(e).__name__}: {e}")
                found.append(rf)
                continue
            run_sheets = ["Data"] + sorted(s for s in sheets if s.startswith("Append"))
            for sheet in run_sheets:
                if m["kind"] == "END":
                    state = f"endurance{cycles_tag or ''}"
                elif sheet == "Data":
                    # PV_1e6 (no 'pris' token) is itself a post-cycling run
                    state = base_state if base_state not in ("cycled",) else f"post{cycles_tag}"
                else:
                    state = f"post{cycles_tag or sheet.lower()}"
                rf = RawFile(
                    path,
                    m["kind"],
                    state,
                    sheet,
                    m["cap"],
                    int(cm["row"]),
                    int(cm["col"]),
                    int(m["size"]),
                    sample,
                    int(sm["num"]) if sm else None,
                    cycles_tag,
                )
                meta = settings.get(sheet) or settings.get("Data") or {}
                rf.module, rf.executed, rf.params = (
                    meta.get("module"),
                    meta.get("executed"),
                    meta.get("params", {}),
                )
                try:
                    d = read_data(path, sheet)
                    tcol = "t" if "t" in d.columns else ("Time" if "Time" in d.columns else None)
                    if tcol:
                        t = d[tcol].astype(float).to_numpy()
                        rf.n_points = len(t)
                        rf.dt_s = float(pd.Series(t).diff().median())
                        rf.t_end_s = float(t[-1])
                    else:
                        rf.n_points = len(d)
                    if rf.module != EXPECTED_MODULE[m["kind"]]:
                        rf.problems.append(f"unexpected module {rf.module}")
                except Exception as e:  # noqa: BLE001
                    rf.problems.append(f"read error: {type(e).__name__}: {e}")
                found.append(rf)
    # an Append run in a file whose name carries no cycle count takes the count from the
    # capacitor's endurance file (PUND_pris_R9C11 + END_1e6_R9C11 -> post1e6)
    tags = {}
    for rf in found:
        if rf.cycles_tag:
            tags.setdefault((rf.sample, rf.cap), set()).add(rf.cycles_tag)
    for rf in found:
        if rf.state.startswith("postappend"):
            t = tags.get((rf.sample, rf.cap), set())
            if len(t) == 1:
                rf.state = f"post{next(iter(t))}"
    return found


def pair_by_capacitor(files: list[RawFile]) -> dict[tuple, dict]:
    """{(sample, cap, state): {"PUND": RawFile, "PV": RawFile, "END": RawFile}}.

    Endurance records attach to the pristine state of the same capacitor.
    """
    pairs: dict[tuple, dict] = {}
    for rf in files:
        if rf.kind == "END":
            continue
        pairs.setdefault((rf.sample, rf.cap, rf.state), {})[rf.kind] = rf
    for rf in files:
        if rf.kind == "END":
            key = next(
                (k for k in pairs if k[0] == rf.sample and k[1] == rf.cap and k[2] == "pris"), None
            )
            if key is None:
                key = (rf.sample, rf.cap, "pris")
                pairs[key] = {}
            pairs[key]["END"] = rf
    for key, p in pairs.items():
        if "PUND" not in p or "PV" not in p:
            for rf in p.values():
                rf.problems.append("unpaired (missing %s)" % ("PV" if "PV" not in p else "PUND"))
        elif (
            key[2] == "pris"
            and p["PUND"].executed
            and p["PV"].executed
            and p["PUND"].executed < p["PV"].executed
        ):
            p["PUND"].problems.append("PUND ran before PV: preset state undefined")
    return pairs


def raw_index(files: list[RawFile]) -> pd.DataFrame:
    df = pd.DataFrame([rf.index_row() for rf in files])
    return df.sort_values(["sample_num", "sample", "row", "col", "kind", "run"]).reset_index(
        drop=True
    )
