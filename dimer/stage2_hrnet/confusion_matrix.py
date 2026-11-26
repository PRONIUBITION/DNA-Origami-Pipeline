"""Validation Result Confusion Matrix Generation Tool

Usage:
 python confusion_matrix.py --pred <val_results directory> --anno <annotation directory> [--threshold <distance threshold>]

Example:
 python confusion_matrix.py \\
 --pred outputs/val_results/val_train_20260607_153853_20260616_162801 \\
 --anno data/annotations/val

Output:
 - The console prints the confusion matrix + various metrics
 - Save confusion_matrix.png to the prediction results directory"""

import json
import os
import sys
import numpy as np
from pathlib import Path


def compute_gt_class(
    points_dict, image_width, image_height, distance_threshold=55, input_size=(256, 256)
):
    """Calculate the true classification results from the annotated keypoints"""
    if len(points_dict) < 4:
        return "N/A", None, None

    for label in ["left_dot_1", "left_dot_2", "right_dot_1", "right_dot_2"]:
        if label not in points_dict:
            return "N/A", None, None

    scale_x = input_size[1] / image_width

    ld = np.linalg.norm(points_dict["left_dot_1"] - points_dict["left_dot_2"]) * scale_x
    rd = np.linalg.norm(points_dict["right_dot_1"] - points_dict["right_dot_2"]) * scale_x
    lt = 1 if ld < distance_threshold else 0
    rt = 1 if rd < distance_threshold else 0
    cls = f"{lt}{rt}"
    return cls, ld, rd


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="""Generate a confusion matrix for the validation results"""
    )
    parser.add_argument(
        "--pred",
        type=str,
        required=True,
        help="""Output directory of predict.py (including val_results.json)""",
    )
    parser.add_argument(
        "--anno", type=str, default="data/annotations/val", help="""Labelme annotation directory"""
    )
    parser.add_argument(
        "--threshold", type=float, default=55, help="""Distance threshold (256x256 scale)"""
    )
    parser.add_argument("--save", action="store_true", help="""Save confusion matrix image""")
    args = parser.parse_args()

    pred_dir = Path(args.pred)
    anno_dir = Path(args.anno)
    threshold = args.threshold

    # Read prediction results
    pred_path = pred_dir / "val_results.json"
    if not pred_path.exists():
        print(f"""[ERROR] Prediction result not found: {pred_path}""")
        sys.exit(1)
    with open(pred_path, encoding="utf-8") as f:
        pred_results = json.load(f)

    # Index by image_name
    pred_map = {}
    for r in pred_results:
        name = r.get("image_name", "")
        pred_map[name] = r

    # Build a confusion matrix
    gt_labels = ["00", "01", "10", "11", "N/A"]
    conf = {a: {b: 0 for b in gt_labels} for a in gt_labels}

    matched = 0
    for fname in sorted(os.listdir(anno_dir)):
        if not fname.endswith(".json"):
            continue
        stem = fname[:-5]
        if stem not in pred_map:
            continue

        # Read annotations
        with open(os.path.join(anno_dir, fname), encoding="utf-8") as f:
            anno = json.load(f)
        iw = anno.get("imageWidth", 1)
        ih = anno.get("imageHeight", 1)

        # Parse keypoints
        pts = {}
        for shape in anno.get("shapes", []):
            label = shape.get("label", "")
            if label not in ["left_dot_1", "left_dot_2", "right_dot_1", "right_dot_2"]:
                continue
            xy = shape["points"][0]
            if len(xy) != 2 or xy == [-1, -1]:
                continue
            pts[label] = np.array(xy, dtype=np.float32)

        # Ground truth classification
        gt_cls, gt_ld, gt_rd = compute_gt_class(pts, iw, ih, threshold)

        # Predicted classification
        r = pred_map[stem]
        n_dots = r.get("num_dots", 0)
        if n_dots == 4:
            pd_cls = r.get("classification", "N/A")
        else:
            pd_cls = "N/A"

        conf[gt_cls][pd_cls] += 1
        matched += 1

    print(f"""\n Number of matching samples: {matched}""")
    print(f"""Distance threshold: {threshold} px \n""")

    # Print Confusion Matrix
    col_labels = ["GT\\Pred"] + gt_labels
    header = "".join(f"{c:>6}" for c in col_labels)
    print("=" * len(header))
    print("""Confusion Matrix (Rows = True Labels, Columns = Predicted Results)""")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for gt in gt_labels:
        row = f"{gt:>8}" + "".join(f"{conf[gt][c]:>6}" for c in gt_labels)
        print(row)
    print("=" * len(header))

    # Calculate various indicators
    total = sum(sum(v.values()) for v in conf.values())
    correct_4dot = sum(conf[c][c] for c in ["00", "01", "10", "11"])
    total_4dot_gt = sum(
        conf[c]["00"] + conf[c]["01"] + conf[c]["10"] + conf[c]["11"]
        for c in ["00", "01", "10", "11"]
    )
    total_4dot_pred = sum(
        conf["00"][c] + conf["01"][c] + conf["10"][c] + conf["11"][c]
        for c in ["00", "01", "10", "11"]
    )

    print()
    print("""--- Core Metrics ---""")
    print(f"""Number of 4-point samples (actual): {total_4dot_gt}""")
    print(f"""Number of 4-point samples (predicted): {total_4dot_pred}""")
    prec = correct_4dot / total_4dot_pred * 100 if total_4dot_pred > 0 else 0
    rec = correct_4dot / total_4dot_gt * 100 if total_4dot_gt > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"""4-point classification correct: {correct_4dot} / {total_4dot_gt}""")
    print(f"""Classification precision: {prec:.1f}%""")
    print(f"""Classification recall rate: {rec:.1f} %""")
    print(f"""Classification F1 score: {f1:.1f}%""")

    # Accuracy for each category
    print()
    print("""--- Accuracy Rate for Each Category ---""")
    for c in ["00", "01", "10", "11"]:
        total_c = sum(conf[c].values())
        if total_c > 0:
            acc = conf[c][c] / total_c * 100
            print(f"""Category {c} : {conf[c][c]} / {total_c} = {acc:.1f} %""")

    # N/A Statistics
    na_pred = sum(conf[c]["N/A"] for c in gt_labels)
    na_gt = sum(conf["N/A"][c] for c in gt_labels)
    print(f"""N/A (Forecast not 4 points): {na_pred}""")
    print(f"""N/A (Actually not 4 points): {na_gt}""")

    # Confusion Matrix Visualization
    if args.save:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        labels = ["00", "01", "10", "11", "N/A"]
        matrix = np.array([[conf[gt][pd] for pd in labels] for gt in labels])

        row_sums = matrix.sum(axis=1, keepdims=True)
        norm_matrix = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix, dtype=float),
            where=row_sums > 0,
        )

        im = ax.imshow(
            norm_matrix,
            interpolation="nearest",
            cmap=plt.cm.Blues,
            vmin=0,
            vmax=1,
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="""Ratio""")

        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_ylabel("True Class", fontsize=12)
        ax.set_title(f"Confusion Matrix (threshold={threshold}px)", fontsize=14)

        for i in range(5):
            for j in range(5):
                val = matrix[i, j]
                pct = norm_matrix[i, j] * 100
                color = "white" if norm_matrix[i, j] > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{int(val)}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=10,
                )

        save_path = pred_dir / "confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"""\n [OK] Confusion matrix saved: {save_path}""")


if __name__ == "__main__":
    main()
