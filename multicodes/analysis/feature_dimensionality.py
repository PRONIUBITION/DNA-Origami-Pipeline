"""Compute PCA effective dimensionality for the final feature table."""

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = ROOT / "outputs" / "qwen_recognition" / "all_features_v2.json"
META = {"image", "group", "crop", "code", "confidence"}


def effective_dimensions(matrix, levels=(0.90, 0.95, 0.99)):
    standardized = (matrix - matrix.mean(axis=0)) / np.where(
        matrix.std(axis=0) == 0, 1.0, matrix.std(axis=0)
    )
    _, singular, _ = np.linalg.svd(
        standardized - standardized.mean(axis=0),
        full_matrices=False,
    )
    explained = singular**2 / np.sum(singular**2)
    cumulative = np.cumsum(explained)
    return {
        level: int(np.searchsorted(cumulative, level) + 1)
        for level in levels
    }


def summarize(records, names, label):
    matrix = np.asarray([[record[name] for name in names] for record in records])
    dimensions = effective_dimensions(matrix)
    print(
        f"{label}: n={len(records)}, features={len(names)}, "
        f"90%={dimensions[0.90]}, 95%={dimensions[0.95]}, "
        f"99%={dimensions[0.99]}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=str(DEFAULT_FEATURES))
    args = parser.parse_args()

    records = json.loads(Path(args.features).read_text(encoding="utf-8"))
    names = [name for name in records[0] if name not in META]
    summarize(records, names, "all crosses")
    summarize(
        [record for record in records if record["group"] != "ccc"],
        names,
        "final dataset excluding ccc",
    )


if __name__ == "__main__":
    main()
