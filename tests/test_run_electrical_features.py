"""The command line on one real sample: tables written once, the campaign merged, exit codes."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

from paths import round_raw_dir

SCRIPT = Path(__file__).resolve().parents[1] / "src" / "run_electrical_features.py"
SAMPLE14 = round_raw_dir(2) / "Sample14"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def test_one_sample_produces_its_tables_report_and_campaign_rows(tmp_path):
    out, campaign, report = tmp_path / "out", tmp_path / "campaign.csv", tmp_path / "qc.html"
    args = ["--data", str(SAMPLE14), "--round", "2", "--out", str(out), "--campaign", str(campaign)]
    first = _run(*args, "--report", str(report))
    assert first.returncode == 0, first.stderr
    assert {p.name for p in out.iterdir()} == {
        "raw_index.csv",
        "features_per_capacitor.csv",
        "features_per_sample.csv",
    }
    assert report.read_text(encoding="utf-8").count("<img") == 1
    assert "Sample14" in first.stdout and "0 with errors" in first.stdout

    again = _run(*args)
    assert again.returncode == 1 and "--overwrite" in again.stderr

    rerun = _run(*args, "--overwrite")
    assert rerun.returncode == 0, rerun.stderr
    assert (
        len(pd.read_csv(campaign)) == 10
    )  # the rerun replaced its rows rather than adding to them


def test_a_folder_without_exports_exits_with_a_layout_error(tmp_path):
    (tmp_path / "Sample1").mkdir()
    result = _run("--data", str(tmp_path), "--round", "1", "--out", str(tmp_path / "out"))
    assert result.returncode == 1 and "no PUND, PV or END" in result.stderr
