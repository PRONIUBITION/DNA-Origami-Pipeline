#!/usr/bin/env python3
"""Unified entry point for two-point and three-point monomer inference."""

import argparse
import os
import runpy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = Path(__file__).resolve().parent
MONO_TYPES = ("0", "00", "01", "1", "10", "11")


def build_parser():
    parser = argparse.ArgumentParser(description="Run YOLOv5-seg and HRNet monomer inference.")
    parser.add_argument("--type", choices=MONO_TYPES, required=True)
    parser.add_argument("--input", required=True, help="Input image directory")
    parser.add_argument("--yolo-weights", required=True)
    parser.add_argument("--hrnet-weights", required=True)
    parser.add_argument("--yolov5-dir", required=True)
    parser.add_argument("--output", default="outputs/monomer")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--font", default="arial.ttf")
    return parser


def main():
    args = build_parser().parse_args()
    input_dir = Path(args.input).expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    implementation = (
        "pipeline_two_point.py" if args.type in {"0", "1"} else "pipeline_three_point.py"
    )
    os.environ.update(
        {
            "MONOMER_TYPE": args.type,
            "TEST_DIR": str(input_dir),
            "YOLO_WEIGHTS": str(Path(args.yolo_weights).expanduser().resolve()),
            "HRNET_WEIGHTS": str(Path(args.hrnet_weights).expanduser().resolve()),
            "YOLOV5_DIR": str(Path(args.yolov5_dir).expanduser().resolve()),
            "OUTPUT_DIR": str(Path(args.output).expanduser().resolve()),
            "DEVICE": args.device,
            "FONT_PATH": args.font,
        }
    )
    runpy.run_path(str(PIPELINE_DIR / implementation), run_name="__main__")


if __name__ == "__main__":
    main()
