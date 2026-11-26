"""Analyze the distribution of distances between two points on the same cross in the annotated data
Used to determine the appropriate classification threshold"""

import json
import numpy as np
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_annotations(annotation_dir):
    """Load all annotation files"""
    annotation_files = list(Path(annotation_dir).glob("*.json"))
    print(f"""{len(annotation_files)} annotation files found""")

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
    """Group points into crosses based on labels
    Annotation format: left_dot_1, left_dot_2, right_dot_1, right_dot_2
    left_* belongs to the same cross, right_* belongs to another cross"""
    groups = {"left": [], "right": []}

    for kp in keypoints:
        label = kp["label"].lower()
        if "left" in label:
            groups["left"].append(kp)
        elif "right" in label:
            groups["right"].append(kp)

    # Filter out groups with exactly 2 points (complete crosses)
    complete_groups = {k: v for k, v in groups.items() if len(v) == 2}
    return complete_groups


def calculate_distances(complete_groups, image_width, image_height, input_size=256):
    """Calculate the distance between the two points within each group and normalize it to a 256x256 scale."""
    distances = []

    scale_x = input_size / image_width
    scale_y = input_size / image_height

    for cross_id, points in complete_groups.items():
        p1 = points[0]
        p2 = points[1]

        # Distance on the original image
        dx_orig = p1["x"] - p2["x"]
        dy_orig = p1["y"] - p2["y"]
        dist_orig = np.sqrt(dx_orig**2 + dy_orig**2)

        # Distance standardized to a 256x256 scale
        dx_norm = dx_orig * scale_x
        dy_norm = dy_orig * scale_y
        dist_norm = np.sqrt(dx_norm**2 + dy_norm**2)

        distances.append(
            {
                "cross_id": cross_id,
                "dist_original": dist_orig,
                "dist_256": dist_norm,
                "p1": (p1["x"], p1["y"]),
                "p2": (p2["x"], p2["y"]),
            }
        )

    return distances


def analyze_distances(all_distances):
    """Statistical analysis of distance distribution"""
    if len(all_distances) == 0:
        print("""No valid two-point combination found!""")
        return

    dists_256 = [d["dist_256"] for d in all_distances]
    dists_orig = [d["dist_original"] for d in all_distances]

    print("\n" + "=" * 60)
    print(
        """Statistical analysis of the distance distribution between two points on the same cross"""
    )
    print("=" * 60)

    print("""\n [Original Image Scale]""")
    print(f"""Number of samples: {len(dists_orig)}""")
    print(f"""Mean: {np.mean(dists_orig):.2f} px""")
    print(f"""Median: {np.median(dists_orig):.2f} px""")
    print(f"""Standard Deviation: {np.std(dists_orig):.2f} px""")
    print(f"""Minimum: {np.min(dists_orig):.2f} px""")
    print(f"""Maximum: {np.max(dists_orig):.2f} px""")
    print(f"""25th percentile: {np.percentile(dists_orig, 25):.2f} px""")
    print(f"""75th percentile: {np.percentile(dists_orig, 75):.2f} px""")

    print("""\n [256x256 Standardized Scale]""")
    print(f"""Number of samples: {len(dists_256)}""")
    print(f"""Mean: {np.mean(dists_256):.2f} px""")
    print(f"""Median: {np.median(dists_256):.2f} px""")
    print(f"""Standard Deviation: {np.std(dists_256):.2f} px""")
    print(f"""Minimum: {np.min(dists_256):.2f} px""")
    print(f"""Maximum: {np.max(dists_256):.2f} px""")
    print(f"""25th percentile: {np.percentile(dists_256, 25):.2f} px""")
    print(f"""75th percentile: {np.percentile(dists_256, 75):.2f} px""")

    # Recommended threshold
    mean_256 = np.mean(dists_256)
    std_256 = np.std(dists_256)

    print("""\n [Recommended threshold (based on 256x256 scale)]""")
    print(f"""Conservative threshold (mean +0.5 std): {mean_256 + 0.5 * std_256:.1f} px""")
    print(f"""Moderate threshold (mean): {mean_256:.1f} px""")
    print(f"""Loose threshold (mean - 0.5 std): {mean_256 - 0.5 * std_256:.1f} px""")
    print(f"""75th percentile threshold: {np.percentile(dists_256, 75):.1f} px""")

    return dists_256, dists_orig


def plot_distribution(dists_256, dists_orig, output_path):
    """Plot distance distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 256x256 scale
    ax1 = axes[0]
    ax1.hist(dists_256, bins=30, alpha=0.7, color="blue", edgecolor="black")
    ax1.axvline(
        np.mean(dists_256),
        color="red",
        linestyle="--",
        label=f"""Mean: {np.mean(dists_256):.1f} px""",
    )
    ax1.axvline(
        np.median(dists_256),
        color="green",
        linestyle="--",
        label=f"""Median: {np.median(dists_256):.1f} px""",
    )
    ax1.set_xlabel("""Distance (px)""", fontsize=12)
    ax1.set_ylabel("""Frequency""", fontsize=12)
    ax1.set_title("""Distance distribution at 256x256 scale""", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Original scale
    ax2 = axes[1]
    ax2.hist(dists_orig, bins=30, alpha=0.7, color="orange", edgecolor="black")
    ax2.axvline(
        np.mean(dists_orig),
        color="red",
        linestyle="--",
        label=f"""Mean: {np.mean(dists_orig):.1f} px""",
    )
    ax2.axvline(
        np.median(dists_orig),
        color="green",
        linestyle="--",
        label=f"""Median: {np.median(dists_orig):.1f} px""",
    )
    ax2.set_xlabel("""Distance (px)""", fontsize=12)
    ax2.set_ylabel("""Frequency""", fontsize=12)
    ax2.set_title("""Distance distribution at the original image scale""", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"""\n Distribution map saved: {output_path}""")


def main():
    # Annotation File Directory
    dimer_root = Path(__file__).resolve().parents[1]
    train_dir = dimer_root / "stage2_hrnet" / "data" / "annotations" / "train"
    val_dir = dimer_root / "stage2_hrnet" / "data" / "annotations" / "val"

    all_distances = []

    # Process training set
    if Path(train_dir).exists():
        print("""\n Processing training set annotations...""")
        annotations = load_annotations(train_dir)
        for data in annotations:
            keypoints = parse_keypoints(data)
            groups = group_by_cross(keypoints)

            # Get image dimensions
            img_width = data.get("imageWidth", 256)
            img_height = data.get("imageHeight", 256)

            distances = calculate_distances(groups, img_width, img_height)
            all_distances.extend(distances)

    # Process validation set
    if Path(val_dir).exists():
        print("""\n Processing validation set annotations...""")
        annotations = load_annotations(val_dir)
        for data in annotations:
            keypoints = parse_keypoints(data)
            groups = group_by_cross(keypoints)

            img_width = data.get("imageWidth", 256)
            img_height = data.get("imageHeight", 256)

            distances = calculate_distances(groups, img_width, img_height)
            all_distances.extend(distances)

    # Analysis Results
    if all_distances:
        dists_256, dists_orig = analyze_distances(all_distances)

        # Save distribution chart
        output_dir = dimer_root / "stage2_hrnet" / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_distribution(dists_256, dists_orig, output_dir / "distance_distribution.png")

        # Save detailed data
        with open(output_dir / "distance_analysis.json", "w") as f:
            json.dump(all_distances, f, indent=2)
        print(f"""Detailed data saved: {output_dir / "distance_analysis.json"}""")
    else:
        print("""No valid distance data found!""")


if __name__ == "__main__":
    main()
