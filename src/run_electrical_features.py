"""Turn one round of Keithley exports into feature tables and a QC report.

    python src/run_electrical_features.py --data data/electrical/raw/round2 --round 2 \
        --out data/electrical/features/round2 \
        --campaign data/electrical/features/campaign_per_capacitor.csv \
        --report predictions/electrical/round2/qc_report.html

Writes to --out: raw_index.csv, features_per_capacitor.csv, features_per_sample.csv and,
when endurance files exist, endurance_curves.csv. The QC report overlays every capacitor of
every sample; read it before trusting a sample mean. With --campaign the round's rows replace
any earlier rows with the same (round, sample, cap, state) in that file.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from electrical.pipeline import describe_round, merge_into_campaign, process_round
from electrical.report import write_report


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
    args = ap.parse_args()

    if not args.data.is_dir():
        raise SystemExit(f"no such data folder: {args.data}")
    args.out.mkdir(parents=True, exist_ok=True)

    result = process_round(args.data, args.round)
    result.raw_index.to_csv(args.out / "raw_index.csv", index=False)
    capacitors = result.capacitors
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        capacitors = write_report(
            str(args.report), capacitors, result.samples, result.traces, result.raw_index
        )
    capacitors.to_csv(args.out / "features_per_capacitor.csv", index=False)
    result.samples.to_csv(args.out / "features_per_sample.csv", index=False)
    if len(result.endurance_curves):
        result.endurance_curves.to_csv(args.out / "endurance_curves.csv", index=False)
    if args.campaign:
        merge_into_campaign(args.campaign, capacitors)
    print(describe_round(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
