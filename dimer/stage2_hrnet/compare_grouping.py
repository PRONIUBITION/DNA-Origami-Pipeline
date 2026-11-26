"""Comparison of two grouping methods: PCA projection vs. distance grouping
Test and compare results on the validation set"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import cv2
import json
from tqdm import tqdm
import argparse

from models import SimpleHRNet
import yaml


def load_model(checkpoint_path, config_path, device="cuda"):
    """Load Model"""
    from models import HRNet, SimpleHRNet, hrnet_w32_cfg

    # Load configuration (supports relative paths)
    config_path = Path(config_path)
    if not config_path.is_absolute():
        # Relative to the current working directory
        config_path = Path.cwd() / config_path

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = config["model"].get("hrnet_type", "SimpleHRNet")
    num_keypoints = config["data"]["num_keypoints"]

    # Create model
    if model_type == "SimpleHRNet":
        model = SimpleHRNet(num_keypoints=num_keypoints, in_channels=1)
    elif model_type == "HRNet-W32":
        cfg = hrnet_w32_cfg.copy()
        cfg["NUM_HEATMAPS"] = num_keypoints
        model = HRNet(cfg)
    else:
        raise ValueError(f"""Unknown model type: {model_type}""")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def preprocess(img_path, input_size=(256, 256)):
    """Preprocess image"""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, None
    original_size = img.shape[:2]
    img_resized = cv2.resize(img, (input_size[1], input_size[0]))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_normalized = (img_normalized - 0.5) / 0.5
    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0).float()
    return img_tensor, original_size, img


def heatmap_to_coordinates(heatmaps, threshold=0.1):
    """Decode coordinates from heat map"""
    coords = []
    for i in range(heatmaps.shape[0]):
        heatmap = heatmaps[i]
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        conf = heatmap[y, x]
        if conf > threshold:
            coords.append((x, y, conf))
        else:
            coords.append((0, 0, 0))
    return coords


def classify_pca_grouping(keypoints, distance_threshold=50):
    """PCA Projection Grouping (Legacy Method)"""
    valid_indices = np.where(keypoints[:, 2] > 0)[0]
    valid_points = keypoints[valid_indices][:, :2]
    num_dots = len(valid_points)

    if num_dots != 4:
        return {"classification": "N/A", "method": "PCA", "num_dots": num_dots}

    # PCA
    centroid = np.mean(valid_points, axis=0)
    centered = valid_points - centroid
    cov_matrix = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    main_axis = eigenvectors[:, np.argmax(eigenvalues)]
    split_direction = np.array([-main_axis[1], main_axis[0]])

    # Projection onto the grouping of separation directions
    projections = np.dot(centered, split_direction)
    sorted_indices = np.argsort(projections)

    side1_indices = sorted_indices[:2]
    side2_indices = sorted_indices[2:]

    side1_points = valid_points[side1_indices]
    side2_points = valid_points[side2_indices]

    side1_dist = np.linalg.norm(side1_points[0] - side1_points[1])
    side2_dist = np.linalg.norm(side2_points[0] - side2_points[1])

    type1 = 1 if side1_dist < distance_threshold else 0
    type2 = 1 if side2_dist < distance_threshold else 0

    # Determine left and right
    main_projections = np.dot(centered, main_axis)
    side1_main = np.mean(main_projections[side1_indices])
    side2_main = np.mean(main_projections[side2_indices])

    if side1_main < side2_main:
        left_type, right_type = type1, type2
    else:
        left_type, right_type = type2, type1

    return {
        "classification": f"{left_type}{right_type}",
        "method": "PCA",
        "num_dots": 4,
        "side1_dist": float(side1_dist),
        "side2_dist": float(side2_dist),
        "left_type": left_type,
        "right_type": right_type,
    }


def classify_distance_grouping(keypoints, distance_threshold=50):
    """Distance grouping (new method)"""
    valid_indices = np.where(keypoints[:, 2] > 0)[0]
    valid_points = keypoints[valid_indices][:, :2]
    num_dots = len(valid_points)

    if num_dots != 4:
        return {"classification": "N/A", "method": "Distance", "num_dots": num_dots}

    # PCA (only for finding the principal axis and drawing the dividing line)
    centroid = np.mean(valid_points, axis=0)
    centered = valid_points - centroid
    cov_matrix = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    main_axis = eigenvectors[:, np.argmax(eigenvalues)]

    # Distance-based grouping
    from itertools import combinations

    pairs = list(combinations(range(4), 2))
    pair_dists = []
    for i, j in pairs:
        dist = np.linalg.norm(valid_points[i] - valid_points[j])
        pair_dists.append((dist, i, j))

    pair_dists.sort()
    closest_pair = pair_dists[0]
    side1_indices = [closest_pair[1], closest_pair[2]]

    all_indices = set(range(4))
    side2_indices = list(all_indices - set(side1_indices))

    side1_points = valid_points[side1_indices]
    side2_points = valid_points[side2_indices]

    side1_dist = np.linalg.norm(side1_points[0] - side1_points[1])
    side2_dist = np.linalg.norm(side2_points[0] - side2_points[1])

    type1 = 1 if side1_dist < distance_threshold else 0
    type2 = 1 if side2_dist < distance_threshold else 0

    # Determine left and right
    main_projections = np.dot(centered, main_axis)
    side1_main = np.mean(main_projections[side1_indices])
    side2_main = np.mean(main_projections[side2_indices])

    if side1_main < side2_main:
        left_type, right_type = type1, type2
    else:
        left_type, right_type = type2, type1

    return {
        "classification": f"{left_type}{right_type}",
        "method": "Distance",
        "num_dots": 4,
        "side1_dist": float(side1_dist),
        "side2_dist": float(side2_dist),
        "left_type": left_type,
        "right_type": right_type,
    }


def visualize_comparison(img, keypoints, original_size, result_pca, result_dist, output_path):
    """Two methods for visual comparison"""
    h, w = img.shape[:2]

    # Create a comparison chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Coordinates are mapped back from a 256x256 scale to the original image scale.
    scale_x = w / 256.0
    scale_y = h / 256.0

    valid_points = []
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0:
            # Scale to original image size
            orig_x = int(x * scale_x)
            orig_y = int(y * scale_y)
            valid_points.append((orig_x, orig_y))

    # Left: PCA method
    ax1 = axes[0]
    ax1.imshow(img, cmap="gray")
    ax1.set_title(
        f"""PCA grouping: {result_pca["classification"]} \n"""
        f"d1={result_pca['side1_dist']:.1f}, d2={result_pca['side2_dist']:.1f}"
    )
    for i, (x, y) in enumerate(valid_points):
        ax1.plot(x, y, "ro", markersize=10)
        ax1.text(x + 3, y - 3, str(i + 1), color="yellow", fontsize=12)
    ax1.axis("off")

    # Right figure: Distance-based grouping method
    ax2 = axes[1]
    ax2.imshow(img, cmap="gray")
    ax2.set_title(
        f"""Distance-based clustering: {result_dist["classification"]} \n"""
        f"d1={result_dist['side1_dist']:.1f}, d2={result_dist['side2_dist']:.1f}"
    )
    for i, (x, y) in enumerate(valid_points):
        ax2.plot(x, y, "bo", markersize=10)
        ax2.text(x + 3, y - 3, str(i + 1), color="yellow", fontsize=12)
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="""HRNet weight path""")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="""HRNet configuration file path""",
    )
    parser.add_argument(
        "--input", type=str, required=True, help="""Validation Set Image Directory"""
    )
    parser.add_argument(
        "--output", type=str, default="outputs/comparison", help="""Output Directory"""
    )
    parser.add_argument("--threshold", type=float, default=50, help="""Distance threshold""")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"""Device in use: {device}""")

    # Load Model
    print("""Loading model...""")
    model = load_model(args.weights, args.config, device)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    diff_dir = output_dir / "differences"
    diff_dir.mkdir(exist_ok=True)

    # Load image (relative path supported)
    input_dir = Path(args.input)
    if not input_dir.is_absolute():
        input_dir = Path.cwd() / input_dir

    image_files = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.jpg"))
    print(f"""Input directory: {input_dir}""")
    print(f"""{len(image_files)} images found""")

    results = []
    differences = []

    for img_path in tqdm(image_files, desc="""Process images"""):
        img_tensor, original_size, img = preprocess(img_path)
        if img_tensor is None:
            continue

        # Inference
        with torch.no_grad():
            heatmaps = model(img_tensor.to(device))[0].cpu().numpy()

        # Decode coordinates (at 256x256 scale)
        coords = heatmap_to_coordinates(heatmaps)
        keypoints = np.zeros((4, 3))
        for i, (x, y, conf) in enumerate(coords):
            keypoints[i] = [x * 4, y * 4, conf]  # 64->256

        num_dots = int(np.sum(keypoints[:, 2] > 0.1))
        if num_dots != 4:
            continue

        # Classification of the two methods
        result_pca = classify_pca_grouping(keypoints, args.threshold)
        result_dist = classify_distance_grouping(keypoints, args.threshold)

        result = {
            "image": img_path.name,
            "pca": result_pca,
            "distance": result_dist,
            "same": result_pca["classification"] == result_dist["classification"],
        }
        results.append(result)

        # Record differences
        if not result["same"]:
            differences.append(result)
            # Save difference visualization
            visualize_comparison(
                img,
                keypoints,
                original_size,
                result_pca,
                result_dist,
                diff_dir / f"{img_path.stem}_diff.png",
            )

    # Statistics
    total = len(results)
    if total == 0:
        print("""\n No 4-point samples found!""")
        return

    same_count = sum(1 for r in results if r["same"])
    diff_count = total - same_count

    print(f"\n{'=' * 60}")
    print(f"""Total of 4 samples: {total}""")
    print(f"""Consistent classification: {same_count} ( {same_count / total * 100:.1f} %)""")
    print(f"""Inconsistent classification: {diff_count} ( {diff_count / total * 100:.1f} %)""")
    print(f"{'=' * 60}")

    # Save detailed results
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save detailed list of inconsistencies
    if differences:
        print("""\n Samples with inconsistent classifications:""")
        for d in differences:
            print(
                f"  {d['image']}: PCA={d['pca']['classification']} vs Distance={d['distance']['classification']}"
            )

        with open(output_dir / "differences.json", "w") as f:
            json.dump(differences, f, indent=2)

    print(f"""\n Results saved to: {output_dir}""")


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main()
