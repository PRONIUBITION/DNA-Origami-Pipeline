"""Combined single-stream test after excluding ccc."""

import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
BASE = ROOT / "outputs/qwen_recognition"
FINAL = BASE / "final"
PYTHON = __import__("sys").executable
STREAMS = 56


def main():
    stream = FINAL / "combined_excl_ccc.bin"
    subprocess.run(
        [
            PYTHON,
            str(SRC / "build_random_stream.py"),
            "--features",
            str(BASE / "all_features_v2.json"),
            "--order",
            "interleave",
            "--exclude",
            "ccc",
            "--output",
            str(stream),
        ],
        cwd=str(ROOT),
        check=True,
    )

    data = np.fromfile(stream, dtype=np.uint8)
    bits = np.unpackbits(data)
    length = bits.size // STREAMS
    used = length * STREAMS
    print(
        f"total bits {bits.size:,}  ->  {STREAMS} x {length:,} = {used:,} "
        f"(discard {bits.size - used})"
    )

    test = FINAL / "combined_excl_ccc_56.bin"
    test.write_bytes(np.packbits(bits[:used]).tobytes())

    summary_dir = ROOT / "nist_sts/runs/combined_excl_ccc"
    summary = subprocess.run(
        [
            PYTHON,
            str(SRC / "run_official_sts.py"),
            "--input",
            str(test),
            "--name",
            "combined_excl_ccc",
            "--length",
            str(length),
            "--streams",
            str(STREAMS),
            "--apen-m",
            "4",
            "--serial-m",
            "8",
            "--p-threshold",
            "0.001",
            "--output-dir",
            str(summary_dir),
        ],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    )
    print(summary.stdout[-1200:])

    rank_dir = ROOT / "nist_sts/runs/combined_excl_ccc_rank"
    subprocess.run(
        [
            PYTHON,
            str(SRC / "run_official_sts.py"),
            "--input",
            str(stream),
            "--name",
            "combined_excl_ccc_rank",
            "--length",
            str(bits.size),
            "--output-dir",
            str(rank_dir),
        ],
        cwd=str(ROOT),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    rank_path = rank_dir / "algorithm_testing_logs/Rank/results.txt"
    rank_p = None
    for line in rank_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rank_p = float(line.strip())
            break
    print(f"rank single-stream p = {rank_p}")

    (FINAL / "combined_excl_ccc_summary.json").write_text(
        json.dumps(
            {
                "total_bits": int(bits.size),
                "streams": STREAMS,
                "sequence_length": int(length),
                "used_bits": int(used),
                "discarded_bits": int(bits.size - used),
                "rank_single_p": rank_p,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
