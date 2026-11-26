"""Verify that a YOLOv5-Seg weight file can be loaded."""

import argparse
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolov5-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.yolov5_dir.resolve()))
    from models.common import DetectMultiBackend

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetectMultiBackend(
        str(args.weights.resolve()),
        device=device,
        dnn=False,
        fp16=False,
    )
    model.model.eval()
    print(f"loaded={args.weights.name} device={device} stride={int(model.stride)}")


if __name__ == "__main__":
    main()
