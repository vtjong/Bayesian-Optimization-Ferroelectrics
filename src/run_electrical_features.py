"""Turn one round of Keithley exports into feature tables and a QC report.

    python src/run_electrical_features.py --data data/electrical/raw/round2 --round 2 \
        --out data/electrical/features/round2 \
        --campaign data/electrical/features/campaign_per_capacitor.csv \
        --report predictions/electrical/round2/qc_report.html

Writes to --out: raw_index.csv, features_per_capacitor.csv, features_per_sample.csv and, when
endurance files exist, endurance_curves.csv; existing tables are kept unless --overwrite is
given. The QC report overlays every capacitor of every sample and state; read it before
trusting a sample mean. With --campaign the round's rows replace any earlier rows with the same
(round, sample_key, cap, state). --strict exits 2 when any capacitor-state carries an error.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from electrical.build_feature_tables import update_campaign_table
from electrical.find_measurement_files import RoundLayoutError
from electrical.process_round import RoundResult, process_round
from electrical.write_qc_report import write_qc_report
from utils.file_io import ContractError, write_csv

EXIT_LAYOUT = 1
EXIT_ERRORS = 2


def describe_round(result: RoundResult) -> str:
    """Return the per-sample lines the operator checks against the group's summary sheet."""
    capacitors, samples = result.capacitors, result.samples
    n_error = int(capacitors["has_error"].sum()) + len(result.issues)
    n_flagged = int((capacitors["n_issues"] > 0).sum())
    head = (
        f"{len(result.raw_index)} runs, {len(capacitors)} capacitor-states, "
        f"{len(samples)} sample-states; {n_error} with errors, {n_flagged} with issues"
    )
    columns = [
        "sample", "state", "n_capacitors", "n_switching", "psw_inst_uc_cm2_mean",
        "psw_inst_uc_cm2_sd", "pr_uc_cm2_mean", "pr_uc_cm2_sd", "kappa_summary_mean",
        "kappa_summary_sd", "kappa_pund_mean", "endurance_wake_ratio_mean",
    ]  # fmt: skip
    return head + "\n" + samples[columns].round(3).to_string(index=False)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data", type=Path, required=True, help="folder with one Sample*/ folder per film"
    )
    ap.add_argument("--out", type=Path, required=True, help="folder for the feature tables")
    ap.add_argument("--round", type=int, required=True, help="round number written into every row")
    ap.add_argument(
        "--campaign", type=Path, help="CSV accumulating per-capacitor rows across rounds"
    )
    ap.add_argument("--report", type=Path, help="path of the QC HTML report")
    ap.add_argument("--overwrite", action="store_true", help="replace tables already in --out")
    ap.add_argument(
        "--strict", action="store_true", help="exit 2 if any capacitor-state has an error"
    )
    args = ap.parse_args()

    if not args.data.is_dir():
        raise SystemExit(f"no such data folder: {args.data}")
    try:
        result = process_round(args.data, args.round)
        write_csv(args.out / "raw_index.csv", result.raw_index, args.overwrite)
        write_csv(args.out / "features_per_capacitor.csv", result.capacitors, args.overwrite)
        write_csv(args.out / "features_per_sample.csv", result.samples, args.overwrite)
        if len(result.endurance_curves):
            write_csv(args.out / "endurance_curves.csv", result.endurance_curves, args.overwrite)
        if args.report:
            write_qc_report(args.report, result)
        if args.campaign:
            update_campaign_table(args.campaign, result.capacitors)
    except (RoundLayoutError, ContractError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_LAYOUT
    except FileExistsError as exc:
        print(f"error: {exc.filename} exists; pass --overwrite to replace it", file=sys.stderr)
        return EXIT_LAYOUT

    print(describe_round(result))
    if args.strict and (result.capacitors["has_error"].any() or result.issues):
        return EXIT_ERRORS
    return 0


if __name__ == "__main__":
    sys.exit(main())
