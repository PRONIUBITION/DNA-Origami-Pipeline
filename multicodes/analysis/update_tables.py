"""Rebuild the paper-style tables with the rank p-values filled in.

Two outputs:
  1. final/table1_final.csv         56 x 12,888 single-block protocol
  2. final/table_three_parts.csv    three independent blocks, rank per block
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "nist_sts/runs"
OUT = ROOT / "outputs/qwen_recognition/final"

PANEL = [
    ("Frequency", "frequency"),
    ("BlockFrequency", "block frequency"),
    ("CumulativeSums", "cumulative sums"),
    ("Runs", "runs"),
    ("LongestRun", "longest run of ones"),
    ("FFT", "discrete Fourier transform"),
    ("ApproximateEntropy", "approximate entropy"),
    ("Serial", "serial"),
]


def rank_single_p(part):
    path = RUNS / f"three_parts_part{part}_single" / "algorithm_testing_logs/Rank/results.txt"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            return float(line.strip())
    raise ValueError(path)


def multi_rows(run_name):
    summary = json.loads((RUNS / run_name / "nist_summary.json").read_text(encoding="utf-8"))
    by_name = {t["test"]: t for t in summary["tests"]}
    rows = []
    for key, label in PANEL:
        details = by_name[key]["details"]
        for index, item in enumerate(details, start=1):
            name = label if len(details) == 1 else f"{label} #{index}"
            rows.append(
                {
                    "test": name,
                    "p_value": (
                        "----"
                        if item["pooled_p_value"] is None
                        else f"{item['pooled_p_value']:.6f}"
                    ),
                    "proportion": item["proportion"],
                    "result": "PASS" if item["pass"] else "FAIL",
                    "note": "56 sequences x 4,296 bits",
                }
            )
    return rows


def safe_open(path):
    """Open for writing, falling back to a _v2 suffix if the file is locked."""
    try:
        return path.open("w", newline="", encoding="utf-8-sig"), path
    except PermissionError:
        alternate = path.with_name(path.stem + "_v2" + path.suffix)
        return alternate.open("w", newline="", encoding="utf-8-sig"), alternate


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # --- three independent blocks ---------------------------------------
    three = []
    for part in (1, 2, 3):
        for row in multi_rows(f"three_parts_part{part}_56x4296"):
            three.append({"block": f"part {part}", **row})

        p = rank_single_p(part)
        three.append(
            {
                "block": f"part {part}",
                "test": "rank",
                "p_value": f"{p:.6f}",
                "proportion": "1/1",
                "result": "PASS" if p > 0.001 else "FAIL",
                "note": "single 240,576-bit stream, 234 matrices",
            }
        )

    path = OUT / "table_three_parts.csv"
    handle, actual_three = safe_open(path)
    with handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["block", "test", "p_value", "proportion", "result", "note"],
        )
        writer.writeheader()
        writer.writerows(three)

    print("=== three independent blocks ===")
    print(f"{'block':<8}{'test':<26}{'p':>11}{'proportion':>12}{'result':>8}")
    for row in three:
        print(
            f"{row['block']:<8}{row['test']:<26}{row['p_value']:>11}"
            f"{row['proportion']:>12}{row['result']:>8}"
        )

    # --- final single-block 56 x 12,888 table ---------------------------
    summary = json.loads((RUNS / "strict_m4_s8" / "nist_summary.json").read_text(encoding="utf-8"))
    by_name = {t["test"]: t for t in summary["tests"]}
    rows = []
    for key, label in PANEL + [("Rank", "rank")]:
        details = by_name[key]["details"]
        for index, item in enumerate(details, start=1):
            name = label if len(details) == 1 else f"{label} #{index}"
            rows.append(
                {
                    "test": name,
                    "p_value": (
                        "----"
                        if item["pooled_p_value"] is None
                        else f"{item['pooled_p_value']:.6f}"
                    ),
                    "proportion": item["proportion"],
                    "result": "PASS" if item["pass"] else "FAIL",
                }
            )

    path = OUT / "table1_final.csv"
    handle, actual_final = safe_open(path)
    with handle:
        writer = csv.DictWriter(handle, fieldnames=["test", "p_value", "proportion", "result"])
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=== 56 x 12,888 (single block) ===")
    print(f"{'test':<26}{'p':>11}{'proportion':>12}{'result':>8}")
    for row in rows:
        print(f"{row['test']:<26}{row['p_value']:>11}{row['proportion']:>12}{row['result']:>8}")

    print()
    print(f"written to {actual_three}")
    print(f"written to {actual_final}")


if __name__ == "__main__":
    main()
