#!/usr/bin/env python3
"""DNA Origami White Dot Detection - Python Entry Point
Usage: python datapredict/run.py"""

import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIMER_ROOT = Path(__file__).resolve().parent.parent
YOLO_WEIGHTS = Path(
    os.environ.get("YOLO_WEIGHTS", PROJECT_ROOT / "models" / "yolo_best.pt")
).resolve()
HRNET_WEIGHTS = Path(
    os.environ.get("HRNET_WEIGHTS", PROJECT_ROOT / "models" / "hrnet_best.pth")
).resolve()
INPUT_DIR = Path(os.environ.get("INPUT_DIR", DIMER_ROOT / "datapredict" / "input")).resolve()
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", DIMER_ROOT / "datapredict" / "outputs")).resolve()


def main():
    print("=" * 50)
    print("""DNA Origami White Dot Detection - Automatic Run""")
    print("=" * 50)
    print()

    # Check if the file exists
    if not YOLO_WEIGHTS.exists():
        print(f"""[Error] YOLO weights not found: {YOLO_WEIGHTS}""")
        return 1
    if not HRNET_WEIGHTS.exists():
        print(f"""[Error] HRNet weights not found: {HRNET_WEIGHTS}""")
        return 1
    if not INPUT_DIR.exists():
        print(f"""[Error] Input directory not found: {INPUT_DIR}""")
        return 1

    print(f"""YOLO weights: {YOLO_WEIGHTS}""")
    print(f"""HRNet weights: {HRNET_WEIGHTS}""")
    print(f"""Input Directory: {INPUT_DIR}""")
    print(f"""Output directory: {OUTPUT_DIR}""")
    print()
    print("=" * 50)
    print("""Starting to run the pipeline...""")
    print("=" * 50)
    print()

    # Build command
    cmd = [
        sys.executable,  # Use the current Python interpreter
        str(DIMER_ROOT / "pipeline.py"),
        "--yolo-weights",
        str(YOLO_WEIGHTS),
        "--hrnet-weights",
        str(HRNET_WEIGHTS),
        "--input",
        str(INPUT_DIR),
        "--output",
        str(OUTPUT_DIR),
    ]

    # Run
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    print()
    print("=" * 50)
    if result.returncode == 0:
        print("""[Success] Run completed!""")
    else:
        print(f"""[Error] Run failed (Code: {result.returncode} )""")
    print("=" * 50)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
