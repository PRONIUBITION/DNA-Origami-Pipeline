"""Repository-relative paths for the dimer stage-one utilities."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DNA_DATA_ROOT", REPO_ROOT / "data")).resolve()
OUTPUT_ROOT = Path(os.environ.get("DNA_OUTPUT_ROOT", REPO_ROOT / "outputs")).resolve()
YOLOV5_DIR = Path(os.environ.get("YOLOV5_DIR", REPO_ROOT / "third_party" / "yolov5")).resolve()
YOLO_WEIGHTS = Path(os.environ.get("YOLO_WEIGHTS", REPO_ROOT / "models" / "yolo_best.pt")).resolve()
HRNET_DATA_ROOT = Path(
    os.environ.get("HRNET_DATA_ROOT", PROJECT_ROOT / "stage2_hrnet" / "data")
).resolve()


def data_path(*parts):
    """Return a path below the configured data root."""
    return DATA_ROOT.joinpath(*parts)


def output_path(*parts):
    """Return a path below the configured output root."""
    return OUTPUT_ROOT.joinpath(*parts)
