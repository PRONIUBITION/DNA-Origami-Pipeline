"""Visualize the distribution of distances between two points on the cross to help determine the threshold."""

import json
import numpy as np
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def load_annotations(annotation_dir):
    """Load all annotation files"""
    annotation_files = list(Path(annotation_dir).glob("*.json"))
    annotations = []
    for f in annotation_files:
        with open(f, "r", encoding="utf-8") as file:
            data = json.load(file)
            annotations.append(data)
    return annotations


def parse_keypoints(data):
    """Parse keypoint coordinates"""
    keypoints = []
    shapes = data.get("shapes", [])
    for shape in shapes:
        if shape.get("shape_type") == "point":
            label = shape.get("label", "")
            points = shape.get("points", [])
            if len(points) > 0:
                x, y = points[0]
                keypoints.append({"label": label, "x": x, "y": y})
    return keypoints


def group_by_cross(keypoints):
    """Group points into crosses based on labels"""
    groups = {"left": [], "right": []}
    for kp in keypoints:
        label = kp["label"].lower()
        if "left" in label:
            groups["left"].append(kp)
        elif "right" in label:
            groups["right"].append(kp)
    complete_groups = {k: v for k, v in groups.items() if len(v) == 2}
    return complete_groups


def calculate_distances(complete_groups, image_width, image_height, input_size=256):
    """Calculate the distance between two points within each group and normalize it to a 256x256 scale."""
    distances = []
    scale_x = input_size / image_width
    scale_y = input_size / image_height

    for cross_id, points in complete_groups.items():
        p1, p2 = points[0], points[1]
        dx_orig = p1["x"] - p2["x"]
        dy_orig = p1["y"] - p2["y"]
        dist_orig = np.sqrt(dx_orig**2 + dy_orig**2)

        dx_norm = dx_orig * scale_x
        dy_norm = dy_orig * scale_y
        dist_norm = np.sqrt(dx_norm**2 + dy_norm**2)

        distances.append({"cross_id": cross_id, "dist_original": dist_orig, "dist_256": dist_norm})
    return distances


def create_visualization(all_distances, output_path):
    """Create a threshold visualization chart"""
    dists_256 = np.array([d["dist_256"] for d in all_distances])

    # Statistical measures
    mean_val = np.mean(dists_256)
    median_val = np.median(dists_256)
    std_val = np.std(dists_256)
    q25 = np.percentile(dists_256, 25)
    q75 = np.percentile(dists_256, 75)
    min_val = np.min(dists_256)
    max_val = np.max(dists_256)

    # Create Chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram + Threshold Line
    ax1 = axes[0, 0]
    n, bins, patches = ax1.hist(dists_256, bins=40, alpha=0.7, color="skyblue", edgecolor="black")

    # Add Threshold Line
    thresholds = [
        (25, "red", "Current (25px)"),
        (40, "orange", "Previous (40px)"),
        (q25, "green", f"25% percentile ({q25:.1f}px)"),
        (median_val, "blue", f"Median ({median_val:.1f}px)"),
        (mean_val, "purple", f"Mean ({mean_val:.1f}px)"),
        (q75, "brown", f"75% percentile ({q75:.1f}px)"),
    ]

    for val, color, label in thresholds:
        ax1.axvline(val, color=color, linestyle="--", linewidth=2, label=label)

    ax1.set_xlabel("Distance (px) @ 256x256 scale", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Distance Distribution with Threshold Candidates", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Box plot
    ax2 = axes[0, 1]
    bp = ax2.boxplot([dists_256], vert=True, patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")

    # Label values on the box plot
    ax2.text(1.15, q25, f"Q1: {q25:.1f}", fontsize=9, va="center")
    ax2.text(1.15, median_val, f"Median: {median_val:.1f}", fontsize=9, va="center", color="blue")
    ax2.text(1.15, q75, f"Q3: {q75:.1f}", fontsize=9, va="center")
    ax2.text(1.15, mean_val, f"Mean: {mean_val:.1f}", fontsize=9, va="center", color="purple")

    ax2.set_ylabel("Distance (px)", fontsize=11)
    ax2.set_title("Box Plot of Distances", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # 3. Cumulative Distribution Plot
    ax3 = axes[1, 0]
    sorted_dists = np.sort(dists_256)
    cumulative = np.arange(1, len(sorted_dists) + 1) / len(sorted_dists) * 100
    ax3.plot(sorted_dists, cumulative, linewidth=2, color="blue")

    # Annotate the cumulative percentage at the key threshold
    for val, color, label in thresholds:
        pct = np.sum(dists_256 < val) / len(dists_256) * 100
        ax3.axvline(val, color=color, linestyle="--", alpha=0.5)
        ax3.axhline(pct, color=color, linestyle=":", alpha=0.5)
        ax3.scatter([val], [pct], color=color, s=50, zorder=5)
        ax3.text(
            val + 2, pct, f"{label}\n{pct:.1f}% < threshold", fontsize=8, va="center", color=color
        )

    ax3.set_xlabel("Distance (px)", fontsize=11)
    ax3.set_ylabel("Cumulative Percentage (%)", fontsize=11)
    ax3.set_title("Cumulative Distribution", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, max_val * 1.1)
    ax3.set_ylim(0, 100)

    # 4. Threshold Comparison Table
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Prepare table data
    table_data = []
    for val, color, label in thresholds:
        pct_type1 = np.sum(dists_256 < val) / len(dists_256) * 100
        pct_type0 = 100 - pct_type1
        table_data.append(
            [
                label.replace(f" ({val:.1f}px)", ""),
                f"{val:.1f}px",
                f"{pct_type1:.1f}%",
                f"{pct_type0:.1f}%",
            ]
        )

    table = ax4.table(
        cellText=table_data,
        colLabels=["Threshold", "Value", "Type 1 (close)", "Type 0 (far)"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0.3, 1, 0.6],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Highlight Table Header
    for i in range(4):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    ax4.set_title("Threshold Comparison", fontsize=12, fontweight="bold", y=0.95)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nVisualization saved: {output_path}")


def main():
    # Load Data
    dimer_root = Path(__file__).resolve().parents[1]
    train_dir = dimer_root / "stage2_hrnet" / "data" / "annotations" / "train"
    val_dir = dimer_root / "stage2_hrnet" / "data" / "annotations" / "val"

    all_distances = []

    for dir_path in [train_dir, val_dir]:
        if Path(dir_path).exists():
            annotations = load_annotations(dir_path)
            for data in annotations:
                keypoints = parse_keypoints(data)
                groups = group_by_cross(keypoints)
                img_width = data.get("imageWidth", 256)
                img_height = data.get("imageHeight", 256)
                distances = calculate_distances(groups, img_width, img_height)
                all_distances.extend(distances)

    if all_distances:
        # Create visualization
        output_dir = dimer_root / "stage2_hrnet" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        create_visualization(all_distances, output_dir / "threshold_analysis.png")

        # Print Statistics
        dists_256 = [d["dist_256"] for d in all_distances]
        print("\n" + "=" * 60)
        print("Distance Statistics (256x256 scale)")
        print("=" * 60)
        print(f"Sample count: {len(dists_256)}")
        print(f"Mean: {np.mean(dists_256):.2f}px")
        print(f"Median: {np.median(dists_256):.2f}px")
        print(f"Std: {np.std(dists_256):.2f}px")
        print(f"Min: {np.min(dists_256):.2f}px")
        print(f"Max: {np.max(dists_256):.2f}px")
        print(f"25% percentile: {np.percentile(dists_256, 25):.2f}px")
        print(f"75% percentile: {np.percentile(dists_256, 75):.2f}px")

        # Recommended Threshold
        print("\n" + "=" * 60)
        print("Recommended Thresholds")
        print("=" * 60)
        print(f"Conservative (75% percentile): {np.percentile(dists_256, 75):.1f}px")
        print(f"Moderate (mean): {np.mean(dists_256):.1f}px")
        print(f"Aggressive (median): {np.median(dists_256):.1f}px")
    else:
        print("No valid distance data found!")


if __name__ == "__main__":
    main()
