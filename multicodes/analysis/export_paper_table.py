"""Export the nine-test panel in the paper's Table 1 format."""

import csv
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "qwen_recognition" / "final"

PANEL = [
    ("Frequency", "frequency"),
    ("BlockFrequency", "block frequency"),
    ("CumulativeSums", "cumulative sums"),
    ("Runs", "runs"),
    ("LongestRun", "longest run of ones"),
    ("Rank", "rank"),
    ("FFT", "discrete Fourier transform"),
    ("ApproximateEntropy", "approximate entropy"),
    ("Serial", "serial"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        default=str(ROOT / "nist_sts/runs/apen_m4"),
        help="archived STS run directory",
    )
    parser.add_argument(
        "--csv",
        default=str(OUT / "table1_paper_style.csv"),
    )
    args = parser.parse_args()

    run = Path(args.run)
    summary = json.loads((run / "nist_summary.json").read_text(encoding="utf-8"))
    by_name = {t["test"]: t for t in summary["tests"]}

    rows = []
    for key, label in PANEL:
        test = by_name[key]
        for index, detail in enumerate(test["details"], start=1):
            rows.append(
                {
                    "test": label if len(test["details"]) == 1 else f"{label} #{index}",
                    "p_value_of_p_values": detail["pooled_p_value"],
                    "proportion": detail["proportion"],
                    "result": "PASS" if detail["pass"] else "FAIL",
                    "starred": detail.get("starred", False),
                }
            )

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'test':<28}{'p(p-values)':>14}{'proportion':>12}{'result':>8}")
    for row in rows:
        print(
            f"{row['test']:<28}{row['p_value_of_p_values']:>14}"
            f"{row['proportion']:>12}{row['result']:>8}"
        )
    print()
    print(f"streams   : {summary['streams']}")
    print(f"length    : {summary['stream_length']} bits each")
    print(f"total bits: {summary['streams'] * summary['stream_length']:,}")
    print(f"alpha     : {summary['alpha']}")
    print(f"written to {csv_path}")


if __name__ == "__main__":
    main()
