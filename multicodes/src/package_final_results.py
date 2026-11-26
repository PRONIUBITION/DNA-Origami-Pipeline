"""Package the final three-part result: bitstream text, tables and charts."""

import csv
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
BASE = ROOT / "outputs/qwen_recognition"
FINAL = BASE / "final/image_partition_final"
RUNS = ROOT / "nist_sts/runs"
PYTHON = __import__("sys").executable
PART_JSON = BASE / "final/image_partition_final.json"


def bits_text(path, limit=None):
    data = np.fromfile(path, dtype=np.uint8)
    text = "".join(map(str, np.unpackbits(data).tolist()))
    return text[:limit] if limit else text


def rank_p_value(part):
    path = RUNS / f"finalpart_part{part}_rank_single" / "algorithm_testing_logs/Rank/results.txt"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            return float(line.strip())
    raise ValueError(path)


def rank_matrices(part):
    import re

    path = RUNS / f"finalpart_part{part}_rank_single" / "algorithm_testing_logs/Rank/stats.txt"
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"# of matrices\s*=\s*(\d+)", text)
    return int(match.group(1)) if match else None


def write_table(part, summary, length, rank, matrices, out_path):
    rows = []
    for test in summary["tests"]:
        details = test["details"]
        for index, item in enumerate(details, start=1):
            rows.append(
                {
                    "part": part,
                    "test": test["test"],
                    "subtest": index,
                    "streams": summary["streams"],
                    "sequence_length": length,
                    "p_value": (
                        "" if item["pooled_p_value"] is None else f"{item['pooled_p_value']:.6f}"
                    ),
                    "proportion": item["proportion"],
                    "result": "PASS" if item["pass"] else "FAIL",
                    "starred": item["starred"],
                    "note": "",
                }
            )
    rows.append(
        {
            "part": part,
            "test": "Rank (single stream)",
            "subtest": 1,
            "streams": 1,
            "sequence_length": "",
            "p_value": f"{rank:.6f}",
            "proportion": "1/1",
            "result": "PASS" if rank > 0.001 else "FAIL",
            "starred": False,
            "note": f"{matrices} matrices",
        }
    )
    with out_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    partition = json.loads(PART_JSON.read_text(encoding="utf-8"))
    combined = []

    for part_meta in partition["parts"]:
        part = part_meta["part"]
        folder = FINAL / f"part{part}"
        folder.mkdir(parents=True, exist_ok=True)

        test_bin = FINAL / f"part{part}.bin"
        full_bin = FINAL / f"part{part}_full.bin"
        test_txt = folder / f"part{part}_bits.txt"
        full_txt = folder / f"part{part}_full_bits.txt"
        summary_path = next(RUNS.glob(f"finalpart_part{part}_56x*/nist_summary.json"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        length = summary["stream_length"]

        test_txt.write_text(bits_text(test_bin, length * summary["streams"]), encoding="ascii")
        full_txt.write_text(bits_text(full_bin, part_meta["bits"]), encoding="ascii")

        rank = rank_p_value(part)
        matrices = rank_matrices(part)

        rows = write_table(
            part,
            summary,
            length,
            rank,
            matrices,
            folder / f"nist_results_part{part}.csv",
        )
        combined.extend(rows)

        subprocess.run(
            [
                PYTHON,
                str(SRC / "plot_nist_charts.py"),
                "--run",
                str(summary_path.parent),
                "--prefix",
                f"part{part}",
                "--rank-p",
                str(rank),
                "--out",
                str(BASE / "final/charts"),
            ],
            cwd=str(ROOT),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        print(
            f"part {part}: {len(test_txt.read_text(encoding='ascii')):,} test bits, "
            f"{len(full_txt.read_text(encoding='ascii')):,} full bits, "
            f"rank p={rank:.6f} ({matrices} matrices)"
        )

    combined_path = FINAL / "nist_results_all_parts.csv"
    with combined_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    print(f"\ncombined table: {combined_path}")
    print(f"charts updated in {BASE / 'final/charts'}")


if __name__ == "__main__":
    main()
