"""One round of raw exports -> raw index, per-capacitor table, per-sample table, endurance curves.

Each (capacitor, state) becomes one row: state ``pris`` for the initial run and ``post<cycles>``
for the run after the endurance test. Endurance features attach to the ``pris`` row.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from electrical import locate
from electrical.aggregate import add_derived, per_sample
from electrical.endurance import endurance_features
from electrical.pund import pund_features
from electrical.pv import pv_features

#: Columns that identify one row of the per-capacitor table across rounds.
CAPACITOR_KEY = ["round", "sample", "cap", "state"]


@dataclass
class RoundResult:
    """Everything one round produces, before the QC report flags outliers.

    :param raw_index: One row per run found: file, sheet, module, timestamp, dt, n, problems.
    :param capacitors: One row per (capacitor, state) with every feature.
    :param samples: Mean / sd / n / median / MAD per feature, per (sample, state).
    :param traces: ``{(sample, state): {cap: {"pund": ..., "pv": ..., "end": ...}}}`` for the plots.
    :param endurance_curves: Psw against cycle count per capacitor, empty without END files.
    """

    raw_index: pd.DataFrame
    capacitors: pd.DataFrame
    samples: pd.DataFrame
    traces: dict
    endurance_curves: pd.DataFrame


def process_round(data_dir: Path, round_label) -> RoundResult:
    """Extract every feature of every run under ``data_dir``.

    :param data_dir: Folder holding one ``Sample*/`` folder per film.
    :param round_label: Round number written into every row.
    :return: The round's tables and traces.
    """
    files = locate.scan(str(data_dir))
    pairs = locate.pair_by_capacitor(files)
    idx = locate.raw_index(files)
    rows, traces, curves = [], {}, []
    for (sample, cap, state), p in sorted(pairs.items(), key=lambda kv: kv[0]):
        any_rf = next(iter(p.values()))
        row = {
            "round": round_label,
            "sample": sample,
            "sample_num": any_rf.sample_num,
            "cap": cap,
            "row": any_rf.row,
            "col": any_rf.col,
            "state": state,
            "size_um": any_rf.size_um,
            "problem": None,
            "switching": False,
        }
        tr, probs = {}, []
        if "PUND" in p:
            try:
                f, t = pund_features(
                    locate.read_data(p["PUND"].path, p["PUND"].run), p["PUND"].size_um
                )
                row.update(f)
                tr["pund"] = t
                if "problem" in f:
                    probs.append(f["problem"])
                row["pund_executed"] = p["PUND"].executed
            except Exception as e:  # noqa: BLE001
                probs.append(f"PUND: {type(e).__name__}: {e}")
                traceback.print_exc()
        else:
            probs.append("no PUND run")
        if "PV" in p:
            try:
                f, t = pv_features(locate.read_data(p["PV"].path, p["PV"].run), p["PV"].size_um)
                t["area_cm2"] = f["area_cm2"]
                row.update(
                    {
                        k: v
                        for k, v in f.items()
                        if k not in ("area_cm2", "dt_s", "n_points", "problem")
                    }
                )
                if "problem" in f:
                    probs.append(f["problem"])
                tr["pv"] = t
                row["pv_executed"] = p["PV"].executed
            except Exception as e:  # noqa: BLE001
                probs.append(f"PV: {type(e).__name__}: {e}")
                traceback.print_exc()
        else:
            probs.append("no PV run")
        if "END" in p:
            try:
                f, curve = endurance_features(
                    locate.read_data(p["END"].path, p["END"].run), p["END"].size_um
                )
                row.update(f)
                curve.insert(0, "cap", cap)
                curve.insert(0, "sample", sample)
                curve.insert(0, "round", round_label)
                curves.append(curve)
                tr["end"] = curve
            except Exception as e:  # noqa: BLE001
                probs.append(f"END: {type(e).__name__}: {e}")
                traceback.print_exc()
        for rf in p.values():
            probs.extend(x for x in rf.problems if x not in probs)
        row["problem"] = "; ".join(probs) if probs else None
        rows.append(row)
        traces.setdefault((sample, state), {})[cap] = tr
    cap_df = pd.DataFrame(rows)
    if "switching" in cap_df:
        cap_df["switching"] = cap_df["switching"].fillna(False).astype(bool)
    cap_df = add_derived(cap_df)
    curves_df = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return RoundResult(idx, cap_df, per_sample(cap_df), traces, curves_df)


def merge_into_campaign(path: Path, capacitors: pd.DataFrame) -> pd.DataFrame:
    """Replace this round's rows in the cross-round table and write it back.

    :param path: The campaign CSV; created if absent.
    :param capacitors: This round's per-capacitor rows.
    :return: The merged table.
    """
    merged = capacitors
    if path.exists():
        old = pd.read_csv(path)
        old = old[
            ~old.set_index(CAPACITOR_KEY).index.isin(capacitors.set_index(CAPACITOR_KEY).index)
        ]
        merged = pd.concat([old, capacitors], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return merged


def describe_round(result: RoundResult) -> str:
    """Return the per-sample summary lines the operator checks against the group's sheet."""
    cap, samp = result.capacitors, result.samples
    n_prob = int(cap["problem"].notna().sum())
    head = (
        f"{len(result.raw_index)} runs, {len(cap)} capacitor-states, "
        f"{samp.shape[0]} sample-states; {n_prob} with problems"
    )
    cols = [
        "sample", "state", "n_capacitors", "n_switching", "Psw_inst_mean", "Psw_inst_sd",
        "Pr_mean", "Pr_sd", "kappa_summary_mean", "kappa_summary_sd", "kappa_PUND_mean",
        "wake_ratio_mean",
    ]  # fmt: skip
    table = samp[[c for c in cols if c in samp]].round(3).to_string(index=False)
    return head + "\n" + table
