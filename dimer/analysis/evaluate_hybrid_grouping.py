"""Evaluate the accuracy of the hybrid grouping strategy on the validation set
Compare the predicted classification with the actual classification annotated by Labelme"""

import sys
import os
import json
import yaml
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

DIMER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIMER_ROOT / "stage2_hrnet"))
from models import HRNet, SimpleHRNet, hrnet_w32_cfg


def cv2_imread_unicode(path):
    with open(path, "rb") as f:
        img_array = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)


def load_model(checkpoint_path, config_path, device="cuda"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = config["model"].get("hrnet_type", "SimpleHRNet")
    num_keypoints = config["data"]["num_keypoints"]
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
    return model, config


def preprocess(img_path, input_size=(256, 256)):
    img = cv2_imread_unicode(str(img_path))
    if img is None:
        return None, None, None
    original_size = img.shape[:2]
    img_resized = cv2.resize(img, (input_size[1], input_size[0]))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_normalized = (img_normalized - 0.5) / 0.5
    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0).float()
    return img_tensor, original_size, img


def heatmap_to_coordinates(heatmaps, threshold=0.1):
    coords = []
    for i in range(heatmaps.shape[0]):
        hm = heatmaps[i]
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        conf = hm[y, x]
        if conf > threshold:
            coords.append((x, y, conf))
        else:
            coords.append((0, 0, 0))
    return coords


def classify_from_annotation(
    annotation_path, original_size, input_size=(256, 256), distance_threshold=50
):
    """Calculate the true classification from the Labelme annotation (calculate the distance at a 256x256 input scale)"""
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    points = {}
    for shape in data.get("shapes", []):
        label = shape["label"]
        pt = shape["points"][0]
        points[label] = np.array(pt, dtype=np.float32)

    # Required keypoints
    required = ["left_dot_1", "left_dot_2", "right_dot_1", "right_dot_2"]
    valid_points = {k: v for k, v in points.items() if k in required}

    if len(valid_points) != 4:
        return {"num_dots": len(valid_points), "classification": "N/A"}

    # Map coordinates from the original image scale to the 256x256 input scale
    scale_x = input_size[1] / original_size[1]
    scale_y = input_size[0] / original_size[0]

    left1 = valid_points["left_dot_1"] * np.array([scale_x, scale_y])
    left2 = valid_points["left_dot_2"] * np.array([scale_x, scale_y])
    right1 = valid_points["right_dot_1"] * np.array([scale_x, scale_y])
    right2 = valid_points["right_dot_2"] * np.array([scale_x, scale_y])

    left_dist = np.linalg.norm(left1 - left2)
    right_dist = np.linalg.norm(right1 - right2)

    left_type = 1 if left_dist < distance_threshold else 0
    right_type = 1 if right_dist < distance_threshold else 0

    return {
        "num_dots": 4,
        "classification": f"{left_type}{right_type}",
        "left_type": left_type,
        "right_type": right_type,
        "left_distance": float(left_dist),
        "right_distance": float(right_dist),
    }


def classify_hybrid(keypoints, distance_threshold=50):
    """Mixed grouping strategy (ported from pipeline)"""
    valid_indices = np.where(keypoints[:, 2] > 0)[0]
    valid_points = keypoints[valid_indices][:, :2]
    num_dots = len(valid_points)

    if num_dots != 4:
        return {"num_dots": num_dots, "classification": "N/A", "method": "N/A"}

    # PCA finds the main direction
    centroid = np.mean(valid_points, axis=0)
    centered = valid_points - centroid
    cov_matrix = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    main_axis = eigenvectors[:, np.argmax(eigenvalues)]

    # Distance grouping
    from itertools import combinations

    pairs = list(combinations(range(4), 2))
    pair_dists = []
    for i, j in pairs:
        dist = np.linalg.norm(valid_points[i] - valid_points[j])
        pair_dists.append((dist, i, j))
    pair_dists.sort()

    min_dist = pair_dists[0][0]
    second_min_dist = pair_dists[1][0]
    ratio = second_min_dist / (min_dist + 1e-6)

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

    main_projections = np.dot(centered, main_axis)
    side1_main = np.mean(main_projections[side1_indices])
    side2_main = np.mean(main_projections[side2_indices])

    if side1_main < side2_main:
        left_type, right_type = type1, type2
    else:
        left_type, right_type = type2, type1

    dist_cls = f"{left_type}{right_type}"

    # Hybrid Strategy Decision-Making
    if ratio > 1.5:
        method = "distance_high_conf"
        final_cls = dist_cls
    elif ratio >= 1.0:
        # PCA projection grouping as a comparison
        split_direction = np.array([-main_axis[1], main_axis[0]])
        projections = np.dot(centered, split_direction)
        sorted_idx = np.argsort(projections)
        pca_side1 = sorted_idx[:2]
        pca_side2 = sorted_idx[2:]
        pca_s1 = valid_points[pca_side1]
        pca_s2 = valid_points[pca_side2]
        pca_d1 = np.linalg.norm(pca_s1[0] - pca_s1[1])
        pca_d2 = np.linalg.norm(pca_s2[0] - pca_s2[1])
        pca_t1 = 1 if pca_d1 < distance_threshold else 0
        pca_t2 = 1 if pca_d2 < distance_threshold else 0
        pca_main_proj1 = np.mean(main_projections[pca_side1])
        pca_main_proj2 = np.mean(main_projections[pca_side2])
        if pca_main_proj1 < pca_main_proj2:
            pca_left, pca_right = pca_t1, pca_t2
        else:
            pca_left, pca_right = pca_t2, pca_t1
        pca_cls = f"{pca_left}{pca_right}"

        if pca_cls == dist_cls:
            method = "consensus"
            final_cls = dist_cls
        else:
            method = "distance_fallback"
            final_cls = dist_cls
    else:
        # ratio < 1.0, revert to PCA
        split_direction = np.array([-main_axis[1], main_axis[0]])
        projections = np.dot(centered, split_direction)
        sorted_idx = np.argsort(projections)
        pca_side1 = sorted_idx[:2]
        pca_side2 = sorted_idx[2:]
        pca_s1 = valid_points[pca_side1]
        pca_s2 = valid_points[pca_side2]
        pca_d1 = np.linalg.norm(pca_s1[0] - pca_s1[1])
        pca_d2 = np.linalg.norm(pca_s2[0] - pca_s2[1])
        pca_t1 = 1 if pca_d1 < distance_threshold else 0
        pca_t2 = 1 if pca_d2 < distance_threshold else 0
        pca_main_proj1 = np.mean(main_projections[pca_side1])
        pca_main_proj2 = np.mean(main_projections[pca_side2])
        if pca_main_proj1 < pca_main_proj2:
            pca_left, pca_right = pca_t1, pca_t2
        else:
            pca_left, pca_right = pca_t2, pca_t1
        method = "pca_fallback"
        final_cls = f"{pca_left}{pca_right}"

    return {
        "num_dots": 4,
        "classification": final_cls,
        "method": method,
        "confidence_ratio": float(ratio),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default=str(
            Path(
                os.environ.get(
                    "HRNET_WEIGHTS", DIMER_ROOT / "stage2_hrnet" / "runs" / "best_model.pth"
                )
            )
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DIMER_ROOT / "stage2_hrnet" / "configs" / "config.yaml"),
    )
    parser.add_argument(
        "--images",
        type=str,
        default=str(DIMER_ROOT / "stage2_hrnet" / "data" / "processed" / "val"),
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default=str(DIMER_ROOT / "stage2_hrnet" / "data" / "annotations" / "val"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"""Device: {device}""")

    model, config = load_model(args.weights, args.config, device)
    input_size = tuple(config["data"]["input_size"])
    heatmap_size = tuple(config["heatmap"]["output_size"])
    threshold = config["inference"]["heatmap_threshold"]
    distance_threshold = config["data"].get("distance_threshold", 50)

    image_dir = Path(args.images)
    ann_dir = Path(args.annotations)
    image_files = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))

    print(f"""Validation set images: {len(image_files)} images""")
    print(f"""Annotation directory: {ann_dir}""")
    print("=" * 60)

    results = []
    four_dot_count = 0
    correct_count = 0
    method_counts = {}

    for img_path in tqdm(image_files, desc="""Evaluation"""):
        ann_path = ann_dir / f"{img_path.stem}.json"
        if not ann_path.exists():
            continue

        # Forecast
        img_tensor, original_size, img = preprocess(img_path, input_size)
        if img_tensor is None:
            continue

        # True classification (calculated at 256x256 scale)
        true_result = classify_from_annotation(
            ann_path, original_size, input_size, distance_threshold
        )
        true_cls = true_result["classification"]

        with torch.no_grad():
            heatmaps = model(img_tensor.to(device))[0].cpu().numpy()

        coords = heatmap_to_coordinates(heatmaps, threshold)
        pred_keypoints = np.zeros((4, 3))
        for i, (x, y, conf) in enumerate(coords):
            if conf > 0:
                pred_keypoints[i, 0] = x * (input_size[1] / heatmap_size[1])
                pred_keypoints[i, 1] = y * (input_size[0] / heatmap_size[0])
                pred_keypoints[i, 2] = conf

        num_dots = int(np.sum(pred_keypoints[:, 2] > threshold))
        if num_dots != 4:
            pred_cls = "N/A"
            method = "non4dot"
            ratio = 0.0
        else:
            pred_result = classify_hybrid(pred_keypoints, distance_threshold)
            pred_cls = pred_result["classification"]
            method = pred_result["method"]
            ratio = pred_result["confidence_ratio"]

        is_correct = (pred_cls == true_cls) and (true_cls != "N/A")

        if true_cls != "N/A":
            four_dot_count += 1
            if is_correct:
                correct_count += 1

        method_counts[method] = method_counts.get(method, 0) + 1

        results.append(
            {
                "image": img_path.name,
                "true": true_cls,
                "pred": pred_cls,
                "correct": is_correct,
                "method": method,
                "ratio": ratio,
                "num_dots": num_dots,
            }
        )

    # Statistics
    print("\n" + "=" * 60)
    print(f"""Total number of 4-point samples: {four_dot_count}""")
    print(
        f"""Correct classification: {correct_count} ( {correct_count / four_dot_count * 100:.1f} %)"""
    )
    print(f"=" * 60)

    print("""\n Number of times each method was used:""")
    for method, count in sorted(method_counts.items()):
        print(f"  {method}: {count}")

    # Confusion Matrix
    print("""\n Confusion Matrix (Actual Predicted):""")
    classes = ["00", "01", "10", "11"]
    for true_c in classes:
        row = []
        for pred_c in classes:
            cnt = sum(1 for r in results if r["true"] == true_c and r["pred"] == pred_c)
            row.append(f"{cnt:2d}")
        print(f"""Actual {true_c} :""" + " | ".join(f"{c}:{v}" for c, v in zip(classes, row)))

    # Incorrect samples
    errors = [r for r in results if not r["correct"] and r["true"] != "N/A"]
    if errors:
        print(f"""\n Incorrect samples ({len(errors)}):""")
        for e in errors:
            print(
                f"""{e["image"]}: Actual= {e["true"]}, Predicted= {e["pred"]}, Method= {e["method"]}, Ratio= {e["ratio"]:.2f}"""
            )

    # Save detailed results
    output_path = Path("datapredict/outputs/hybrid_evaluation.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "total_images": len(image_files),
                    "four_dot_samples": four_dot_count,
                    "correct": correct_count,
                    "accuracy": correct_count / four_dot_count if four_dot_count > 0 else 0,
                    "method_counts": method_counts,
                },
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"""\n Detailed results saved to: {output_path}""")


if __name__ == "__main__":
    main()
