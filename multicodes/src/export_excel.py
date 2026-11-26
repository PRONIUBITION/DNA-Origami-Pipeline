"""Export combined and three-part NIST results into one Excel workbook."""

import json
import re
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "nist_sts/runs"
OUT = ROOT / "outputs/qwen_recognition/final/NIST_results_all.xlsx"
PART_JSON = ROOT / "outputs/qwen_recognition/final/image_partition_final.json"

PANEL = [
    ("Frequency", "Frequency"),
    ("BlockFrequency", "Block frequency"),
    ("CumulativeSums", "Cumulative sums"),
    ("Runs", "Runs"),
    ("LongestRun", "Longest run"),
    ("Rank", "Rank"),
    ("FFT", "DFT"),
    ("ApproximateEntropy", "Approximate entropy"),
    ("Serial", "Serial"),
]

DATASETS = [
    {
        "name": "Combined (no ccc)",
        "summary": RUNS / "combined_excl_ccc/nist_summary.json",
        "rank": RUNS / "combined_excl_ccc_rank/algorithm_testing_logs/Rank/results.txt",
        "rank_stats": RUNS / "combined_excl_ccc_rank/algorithm_testing_logs/Rank/stats.txt",
        "note": "single interleaved stream excluding ccc",
    },
    {
        "name": "Part 1",
        "summary": RUNS / "finalpart_part1_56x4075/nist_summary.json",
        "rank": RUNS / "finalpart_part1_rank_single/algorithm_testing_logs/Rank/results.txt",
        "rank_stats": RUNS / "finalpart_part1_rank_single/algorithm_testing_logs/Rank/stats.txt",
        "note": "10 source images",
    },
    {
        "name": "Part 2",
        "summary": RUNS / "finalpart_part2_56x3873/nist_summary.json",
        "rank": RUNS / "finalpart_part2_rank_single/algorithm_testing_logs/Rank/results.txt",
        "rank_stats": RUNS / "finalpart_part2_rank_single/algorithm_testing_logs/Rank/stats.txt",
        "note": "11 source images",
    },
    {
        "name": "Part 3",
        "summary": RUNS / "finalpart_part3_56x4306/nist_summary.json",
        "rank": RUNS / "finalpart_part3_rank_single/algorithm_testing_logs/Rank/results.txt",
        "rank_stats": RUNS / "finalpart_part3_rank_single/algorithm_testing_logs/Rank/stats.txt",
        "note": "12 source images",
    },
]


def first_value(path):
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            return float(line.strip())
    raise ValueError(path)


def matrix_count(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"# of matrices\s*=\s*(\d+)", text)
    return int(match.group(1)) if match else None


def main():
    wb = Workbook()
    summary_ws = wb.active
    summary_ws.title = "Summary"
    detail_ws = wb.create_sheet("Details")
    info_ws = wb.create_sheet("Dataset info")

    header_font = Font(bold=True)
    pass_fill = PatternFill("solid", fgColor="C6EFCE")
    fail_fill = PatternFill("solid", fgColor="FFC7CE")

    summary_header = ["Dataset", "Streams", "Sequence length"]
    for _, label in PANEL:
        summary_header.extend([f"{label} proportion", f"{label} p-value", f"{label} result"])
    summary_header.append("All pass")
    summary_ws.append(summary_header)

    detail_header = [
        "Dataset",
        "Test",
        "Subtest",
        "Streams",
        "Sequence length",
        "p-value",
        "Proportion",
        "Result",
        "Starred",
        "Note",
    ]
    detail_ws.append(detail_header)

    info_ws.append(
        [
            "Dataset",
            "Source",
            "Images",
            "Crosses",
            "Total bits",
            "Streams",
            "Sequence length",
            "Used bits",
            "Discarded bits",
        ]
    )
    partition = json.loads(PART_JSON.read_text(encoding="utf-8"))
    part_meta = {f"Part {p['part']}": p for p in partition["parts"]}
    combined_meta = json.loads(
        (ROOT / "outputs/qwen_recognition/final/combined_excl_ccc_summary.json").read_text(
            encoding="utf-8"
        )
    )

    for dataset in DATASETS:
        summary = json.loads(dataset["summary"].read_text(encoding="utf-8"))
        by_name = {t["test"]: t for t in summary["tests"]}
        streams = summary["streams"]
        length = summary["stream_length"]

        row = [dataset["name"], streams, length]
        all_pass = True
        for key, label in PANEL:
            if key == "Rank":
                p = first_value(dataset["rank"])
                matrices = matrix_count(dataset["rank_stats"])
                ok = p > 0.001
                proportion = "1/1"
                note = f"{matrices} matrices, single full stream"
                pooled = p
            else:
                item = by_name[key]["details"][0]
                ok = bool(item["pass"])
                proportion = item["proportion"]
                pooled = item["pooled_p_value"]
                note = ""
            all_pass &= ok
            row.extend(
                [
                    proportion,
                    "" if pooled is None else round(float(pooled), 6),
                    "PASS" if ok else "FAIL",
                ]
            )
            detail_ws.append(
                [
                    dataset["name"],
                    label,
                    1,
                    streams if key != "Rank" else 1,
                    length if key != "Rank" else "",
                    "" if pooled is None else round(float(pooled), 6),
                    proportion,
                    "PASS" if ok else "FAIL",
                    False,
                    note if note else dataset["note"],
                ]
            )
            if key != "Rank":
                for index, item in enumerate(by_name[key]["details"][1:], start=2):
                    detail_ws.append(
                        [
                            dataset["name"],
                            label,
                            index,
                            streams,
                            length,
                            (
                                ""
                                if item["pooled_p_value"] is None
                                else round(float(item["pooled_p_value"]), 6)
                            ),
                            item["proportion"],
                            "PASS" if item["pass"] else "FAIL",
                            item["starred"],
                            dataset["note"],
                        ]
                    )
        row.append("PASS" if all_pass else "FAIL")
        summary_ws.append(row)

        if dataset["name"] == "Combined (no ccc)":
            info_ws.append(
                [
                    dataset["name"],
                    "combined_excl_ccc.bin",
                    "",
                    253,
                    combined_meta["total_bits"],
                    combined_meta["streams"],
                    combined_meta["sequence_length"],
                    combined_meta["used_bits"],
                    combined_meta["discarded_bits"],
                ]
            )
        else:
            meta = part_meta.get(dataset["name"], {})
            bits = meta.get("bits", "")
            used = length * streams
            info_ws.append(
                [
                    dataset["name"],
                    f"{len(meta.get('images', []))} source images",
                    len(meta.get("images", [])),
                    meta.get("crosses", ""),
                    bits,
                    streams,
                    length,
                    used,
                    bits - used if isinstance(bits, int) else "",
                ]
            )

    for ws in (summary_ws, detail_ws, info_ws):
        for cell in ws[1]:
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", wrap_text=True)
        ws.freeze_panes = "A2"

    for row in summary_ws.iter_rows(min_row=2):
        for cell in row:
            if cell.value == "PASS":
                cell.fill = pass_fill
            elif cell.value == "FAIL":
                cell.fill = fail_fill
    for row in detail_ws.iter_rows(min_row=2):
        result_cell = row[7]
        if result_cell.value == "PASS":
            result_cell.fill = pass_fill
        elif result_cell.value == "FAIL":
            result_cell.fill = fail_fill

    widths = {
        "A": 20,
        "B": 12,
        "C": 16,
        "D": 8,
        "E": 16,
        "F": 12,
        "G": 12,
        "H": 10,
        "I": 10,
        "J": 32,
    }
    for column, width in widths.items():
        detail_ws.column_dimensions[column].width = width

    for column in range(1, summary_ws.max_column + 1):
        summary_ws.column_dimensions[get_column_letter(column)].width = 18

    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        wb.save(OUT)
        saved = OUT
    except PermissionError:
        saved = OUT.with_name(OUT.stem + "_v2" + OUT.suffix)
        wb.save(saved)
    print(f"written to {saved}")
    print(f"summary rows {summary_ws.max_row - 1}, detail rows {detail_ws.max_row - 1}")


if __name__ == "__main__":
    main()
