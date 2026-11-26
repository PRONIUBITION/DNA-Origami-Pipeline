"""Run the official NIST SP 800-22 STS (C version) on a binary bitstream.

The official assess.exe is an interactive console program and its file dialog
uses scanf("%s"), so the input path must not contain spaces. This wrapper
stages the bitstream inside sts/data/ and drives assess.exe non-interactively.

Usage:
    python run_official_sts.py --input <bitstream.bin> --name 111
    python run_official_sts.py --input <bitstream.bin> --name 111 --streams 4

Notes:
    * --streams N means the file is split into N consecutive streams of
      --length bits each; the official STS expects that layout.
    * Results are written to nist_sts/runs/<timestamp>_<name>/ and never
      overwrite a previous run. The shared experiments/AlgorithmTesting
      scratch directory is archived into each run folder.
"""

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STS_DIR = ROOT / "nist_sts" / "source" / "NIST-Statistical-Test-Suite-master" / "sts"
ASSESS = STS_DIR / "assess.exe"
RUNS_DIR = ROOT / "nist_sts" / "runs"
SCRATCH = STS_DIR / "experiments" / "AlgorithmTesting"
STAGE_NAME = "_sts_stage.bin"
ALPHA = 0.01
P_THRESHOLD = 0.0001

TEST_ORDER_NAMES = [
    "Frequency",
    "BlockFrequency",
    "CumulativeSums",
    "Runs",
    "LongestRun",
    "Rank",
    "FFT",
    "NonOverlappingTemplate",
    "OverlappingTemplate",
    "Universal",
    "ApproximateEntropy",
    "RandomExcursions",
    "RandomExcursionsVariant",
    "Serial",
    "LinearComplexity",
]

REPORT_ROW = re.compile(
    r"^\s*((?:\d+\s+){10})\s+(\S+?)\s*(\*?)\s+(\S+?)\s*(\*?)\s+"
    r"([A-Za-z]+)\s*$"
)
SAMPLE_SIZE = re.compile(r"sample size =\s*(\d+)\s+binary sequences")

# Order matches the sequence assess.exe writes results.txt files in.
TEST_ORDER = [
    "Frequency",
    "BlockFrequency",
    "CumulativeSums",
    "Runs",
    "LongestRun",
    "Rank",
    "FFT",
    "NonOverlappingTemplate",
    "OverlappingTemplate",
    "Universal",
    "ApproximateEntropy",
    "RandomExcursions",
    "RandomExcursionsVariant",
    "Serial",
    "LinearComplexity",
]


def read_p_values(test_dir):
    """Return the float p-values written by one test, ignoring headers."""
    path = test_dir / "results.txt"
    if not path.exists():
        return []
    values = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            continue
    return values


def parse_official_report(report_path):
    """Parse finalAnalysisReport.txt into ordered rows.

    Each row is the STS verdict for one subtest: the p-value decile histogram,
    the pooled p-value column, the passing proportion, and the test name.
    'PROPORTION' is <passed>/<total> sequences. A proportion of '------' means
    the subtest could not be evaluated at all (for example RandomExcursions
    needs at least 500 cycles), which is not a failure.

    The pass threshold follows what assess.c itself uses:
        threshold = p_hat - 3 * sqrt(p_hat * alpha / m),  p_hat = 1 - alpha
    with m the number of sequences contributing to that row. This is applied
    per row, because RandomExcursions rows only count the sequences that had
    enough cycles. For a single sequence the threshold is 0.6915, so '1/1'
    passes and '0/1' fails, which is the correct reading. The number printed
    at the bottom of the official report is that same value truncated to an
    integer count, so it is kept only as a cross-check.
    """
    text = report_path.read_text(encoding="utf-8", errors="replace")

    sample = SAMPLE_SIZE.search(text)
    sample_size = int(sample.group(1)) if sample else None
    p_hat = 1.0 - ALPHA

    rows = []
    for line in text.splitlines():
        match = REPORT_ROW.match(line)
        if not match:
            continue
        (
            histogram,
            p_value_col,
            p_star,
            proportion,
            prop_star,
            test_name,
        ) = match.groups()
        if test_name not in TEST_ORDER:
            continue

        if "/" in proportion:
            num, den = (int(v) for v in proportion.split("/", 1))
            ratio = num / den if den else None
        else:
            num = den = 0
            ratio = None

        pooled = None if p_value_col == "----" else float(p_value_col)
        starred = bool(p_star or prop_star)

        if den:
            threshold = p_hat - 3.0 * math.sqrt(p_hat * ALPHA / den)
            verdict = ratio >= threshold
            # SP 800-22 also requires the p-values themselves to be uniform.
            # assess.exe flags a failure with '*'; a pooled p-value below
            # 0.0001 means the histogram is not uniform.
            if verdict and (starred or (pooled is not None and pooled < P_THRESHOLD)):
                verdict = False
        else:
            threshold = None
            verdict = None

        rows.append(
            {
                "test": test_name,
                "histogram": [int(v) for v in histogram.split()],
                "pooled_p_value": pooled,
                "proportion": proportion,
                "starred": starred,
                "sequences_passed": num,
                "sequences_total": den,
                "pass_ratio": ratio,
                "min_pass_rate": threshold,
                "pass": verdict,
            }
        )
    return rows, sample_size


def detect_format(data):
    """Return the STS input-mode code: 0 for ASCII 0/1 text, 1 for packed bytes."""
    sample = data[:4096]
    if sample and all(byte in (0x30, 0x31, 0x0A, 0x0D, 0x20, 0x09) for byte in sample):
        return 0
    return 1


def run_assess(stream_length, num_streams, input_mode, params=None, selection=None):
    """Drive assess.exe: input file, all tests, default params, binary mode."""
    if not ASSESS.exists():
        sys.exit(f"assess.exe not found at {ASSESS}")

    lines = [
        "0",  # generator: user prescribed input file
        f"data/{STAGE_NAME}",  # relative path, no spaces
    ]
    if selection is None:
        lines.append("1")  # apply all 15 statistical tests
    else:
        lines.append("0")  # choose tests individually
        lines.append(selection)  # 15 digits, one flag per test
    params = params or {}
    if not params:
        lines.append("0")  # keep default test parameters
    else:
        # parameter menu indices: 1 block freq, 2 non-overlap, 3 overlap,
        # 4 approximate entropy, 5 serial, 6 linear complexity
        for index in sorted(params):
            lines.extend([str(index), str(int(params[index]))])
        lines.append("0")
    lines.extend(
        [
            str(num_streams),
            str(input_mode),  # 0 = ASCII 0/1 text, 1 = packed binary
            "",
        ]
    )
    stdin = "\n".join(lines)

    proc = subprocess.run(
        [str(ASSESS), str(stream_length)],
        cwd=str(STS_DIR),
        input=stdin,
        capture_output=True,
        text=True,
        errors="replace",
    )
    return proc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="binary bitstream file")
    parser.add_argument("--name", required=True, help="label for the run")
    parser.add_argument(
        "--length",
        type=int,
        default=1_000_000,
        help="bits per individual stream (default 1000000)",
    )
    parser.add_argument(
        "--streams",
        type=int,
        default=1,
        help="number of consecutive streams in the file (default 1)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="explicit output directory (default: nist_sts/runs/<timestamp>_<name>)",
    )
    parser.add_argument(
        "--apen-m",
        type=int,
        default=None,
        help="ApproximateEntropy block length (default: STS value 10)",
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=0.0001,
        help=(
            "minimum p-value of p-values for a test to pass "
            "(STS default 0.0001; the referenced paper uses 0.001)"
        ),
    )
    parser.add_argument(
        "--serial-m",
        type=int,
        default=None,
        help="Serial block length (default: STS value 16)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="run only the named tests, e.g. --only FFT Rank",
    )
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"input not found: {src}")

    data = src.read_bytes()
    # assess.exe reads packed input in 4-byte chunks, so the file must cover a
    # whole number of 32-bit groups or it reports "Insufficient data".
    total_bits = args.length * args.streams
    min_bytes = -(-total_bits // 8)
    chunked_bytes = -(-total_bits // 32) * 4
    if len(data) < min_bytes:
        sys.exit(
            f"input too small: have {len(data)} bytes, "
            f"need {min_bytes} for {args.streams} x {args.length} bits"
        )

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = RUNS_DIR / f"{stamp}_{args.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    staged = STS_DIR / "data" / STAGE_NAME
    staged.parent.mkdir(parents=True, exist_ok=True)
    # pad to the 4-byte chunk boundary assess.exe requires
    payload = bytearray(data[:min_bytes])
    if len(payload) < chunked_bytes:
        payload.extend(bytes(chunked_bytes - len(payload)))
    staged.write_bytes(bytes(payload))

    input_mode = detect_format(data)
    global P_THRESHOLD
    P_THRESHOLD = args.p_threshold

    params = {}
    if args.apen_m is not None:
        params[4] = args.apen_m
    if args.serial_m is not None:
        params[5] = args.serial_m
    selection = None
    if args.only:
        unknown = [n for n in args.only if n not in TEST_ORDER_NAMES]
        if unknown:
            sys.exit(f"unknown test(s): {unknown}")
        flags = ["0"] * len(TEST_ORDER_NAMES)
        for name in args.only:
            flags[TEST_ORDER_NAMES.index(name)] = "1"
        selection = "".join(flags)

    proc = run_assess(args.length, args.streams, input_mode, params, selection)
    (out_dir / "assess_console.txt").write_text(
        proc.stdout + "\n" + proc.stderr, encoding="utf-8", errors="replace"
    )

    report_src = SCRATCH / "finalAnalysisReport.txt"
    if not report_src.exists():
        sys.exit(f"no finalAnalysisReport.txt produced; see {out_dir / 'assess_console.txt'}")
    shutil.copyfile(report_src, out_dir / "finalAnalysisReport.txt")

    logs_dir = out_dir / "algorithm_testing_logs"
    if logs_dir.exists():
        shutil.rmtree(logs_dir)
    shutil.copytree(SCRATCH, logs_dir)

    report_rows, sample_size = parse_official_report(report_src)
    by_test = {}
    for row in report_rows:
        by_test.setdefault(row["test"], []).append(row)

    tests = []
    total_pass = 0
    total_sub = 0
    total_skipped = 0
    for test_name in TEST_ORDER:
        rows = by_test.get(test_name, [])
        subtests = [
            {
                "index": i + 1,
                "proportion": row["proportion"],
                "pass_ratio": row["pass_ratio"],
                "min_pass_ratio": row["min_pass_rate"],
                "pooled_p_value": row["pooled_p_value"],
                "histogram": row["histogram"],
                "starred": row["starred"],
                "pass": row["pass"],
            }
            for i, row in enumerate(rows)
        ]
        passed = sum(1 for s in subtests if s["pass"] is True)
        failed = sum(1 for s in subtests if s["pass"] is False)
        skipped = sum(1 for s in subtests if s["pass"] is None)
        total_pass += passed
        total_sub += len(subtests)
        total_skipped += skipped
        status = "PASS" if failed == 0 else "FAIL"
        tests.append(
            {
                "test": test_name,
                "status": status,
                "subtests": len(subtests),
                "passed": passed,
                "failed": failed,
                "not_evaluated": skipped,
                "raw_p_values": read_p_values(SCRATCH / test_name),
                "details": subtests,
            }
        )

    summary = {
        "name": args.name,
        "input": str(src),
        "stream_length": args.length,
        "streams": args.streams,
        "alpha": ALPHA,
        "engine": "official NIST SP 800-22 STS (C, assess.exe)",
        "official_report": str(out_dir / "finalAnalysisReport.txt"),
        "report_sample_size": sample_size,
        "totals": {
            "tests": len(TEST_ORDER),
            "tests_failed": sum(1 for t in tests if t["status"] == "FAIL"),
            "subtests": total_sub,
            "passed": total_pass,
            "failed": total_sub - total_pass - total_skipped,
            "not_evaluated": total_skipped,
        },
        "tests": tests,
    }
    (out_dir / "nist_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Official NIST SP 800-22 STS summary for <{args.name}>")
    print(f"  bitstream: {src.name}  ({args.streams} x {args.length} bits, alpha={ALPHA})")
    # Thresholds are computed per row; show the one that applies to the
    # bulk of the rows so the printed number is meaningful.
    common = [
        s["min_pass_ratio"] for t in tests for s in t["details"] if s["min_pass_ratio"] is not None
    ]
    if common:
        print(f"  minimum pass proportion: {min(common):.4f}")
    print(f"  {'test':<28}{'passed':>8}{'failed':>8}{'n/a':>6}")
    for item in tests:
        print(
            f"  {item['test']:<28}{item['passed']:>8}{item['failed']:>8}{item['not_evaluated']:>6}"
        )
    totals = summary["totals"]
    print(f"  {'TOTAL':<28}{totals['passed']:>8}{totals['failed']:>8}{totals['not_evaluated']:>6}")
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
