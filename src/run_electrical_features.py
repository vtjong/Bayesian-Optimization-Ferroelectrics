"""Turn one round of Keithley exports into feature tables and a QC report.

    python src/run_electrical_features.py --data data/electrical/raw/round2 --round 2 \
        --out data/electrical/features/round2 \
        --campaign data/electrical/features/campaign_per_capacitor.csv \
        --report predictions/electrical/round2/qc_report.html

Writes to --out: raw_index.csv, features_per_capacitor.csv, features_per_sample.csv and,
when endurance files exist, endurance_curves.csv; existing tables are kept unless
--overwrite is given. The QC report overlays every capacitor of every sample-state; read it
before trusting a sample mean. With --campaign the round's rows replace any earlier rows
with the same (round, sample_key, cap, state) in that file. --strict exits 2 when any
capacitor-state carries an error finding.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from electrical.accumulate import update_campaign, write_table
from electrical.issues import ContractError, RoundLayoutError
from electrical.pipeline import describe_round, process_round
from electrical.report import flag_outliers, write_report

EXIT_LAYOUT = 1
EXIT_FINDINGS = 2


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
        capacitors = (
            write_report(args.report, result) if args.report else flag_outliers(result.capacitors)
        )
        write_table(args.out / "raw_index.csv", result.raw_index, args.overwrite)
        write_table(args.out / "features_per_capacitor.csv", capacitors, args.overwrite)
        write_table(args.out / "features_per_sample.csv", result.samples, args.overwrite)
        if len(result.endurance_curves):
            write_table(args.out / "endurance_curves.csv", result.endurance_curves, args.overwrite)
        if args.campaign:
            update_campaign(args.campaign, capacitors)
    except (RoundLayoutError, ContractError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_LAYOUT
    except FileExistsError as exc:
        print(f"error: {exc.filename} exists; pass --overwrite to replace it", file=sys.stderr)
        return EXIT_LAYOUT

    print(describe_round(result))
    if args.strict and (capacitors["has_error"].any() or result.issues):
        return EXIT_FINDINGS
    return 0


if __name__ == "__main__":
    sys.exit(main())
