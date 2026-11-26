"""HRNet Inference Script
Used for white spot detection on new DNA origami images"""

import os
import sys
import argparse
import yaml
import json
import shutil
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from models import HRNet, SimpleHRNet, hrnet_w32_cfg
from utils import heatmap_to_coordinates, visualize_predictions, visualize_heatmap


def load_model(checkpoint_path, config, device="cuda"):
    """Load the trained model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    model_type = config["model"].get("hrnet_type", "SimpleHRNet")
    num_keypoints = config["data"]["num_keypoints"]

    if model_type == "SimpleHRNet":
        model = SimpleHRNet(num_keypoints=num_keypoints, in_channels=1)
    elif model_type == "HRNet-W32":
        cfg = hrnet_w32_cfg.copy()
        cfg["NUM_HEATMAPS"] = num_keypoints
        model = HRNet(cfg, output_size=tuple(config["heatmap"]["output_size"]))
    else:
        raise ValueError(f"""Unknown model type: {model_type}""")

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def preprocess_image(image_path, input_size=(256, 256)):
    """Preprocess a single image"""
    # Read image
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"""Unable to read image: {image_path}""")

    original_size = img.shape[:2]

    # Resize
    img_resized = cv2.resize(img, (input_size[1], input_size[0]))

    # Normalization
    img_normalized = img_resized.astype(np.float32) / 255.0

    # Standardization
    mean = 0.5
    std = 0.5
    img_normalized = (img_normalized - mean) / std

    # Convert to tensor [1, H, W]
    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).float()

    return img_tensor, original_size, img


def classify_dots(keypoints, image_width, image_height, distance_threshold, input_size=(256, 256)):
    """Classify based on the position of the small white dots (00, 01, 10, 11)
    The dividing line is perpendicular to the main axis of the DNA dimer and passes through the center of the structure.

    Args:
    keypoints: [N, 3] array of (x, y, conf)
    image_width: Image width (original scale)
    image_height: Image height (original scale)
    distance_threshold: Distance threshold (256x256 normalized scale)
    input_size: Input size [H, W]

    Returns:
    classification: dict with classification results"""
    # Filter valid points (confidence > 0)
    valid_points = keypoints[keypoints[:, 2] > 0][:, :2]  # [M, 2]
    num_dots = len(valid_points)

    # Calculate the scaling factor from the original image to the 256x256 input
    scale_x = input_size[1] / image_width
    scale_y = input_size[0] / image_height

    def normalized_distance(p1, p2):
        dx = (p1[0] - p2[0]) * scale_x
        dy = (p1[1] - p2[1]) * scale_y
        return float(np.sqrt(dx * dx + dy * dy))

    if num_dots == 0:
        return {
            "num_dots": 0,
            "left_dots": 0,
            "right_dots": 0,
            "left_type": -1,
            "right_type": -1,
            "classification": "N/A",
            "left_points": [],
            "right_points": [],
            "split_line": None,
        }

    # For 4-point samples, use distance grouping (the two points on the same cross are closest to each other)
    if num_dots == 4:
        # Calculate the centroid of four points
        centroid = np.mean(valid_points, axis=0)

        # Use PCA to find the principal direction (only for determining left and right)
        centered = valid_points - centroid
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        main_axis = eigenvectors[:, np.argmax(eigenvalues)]
        if main_axis[0] < 0:
            main_axis = -main_axis
        split_direction = np.array([-main_axis[1], main_axis[0]])

        # Key change: Use distance-based grouping instead of PCA projection-based grouping
        # Calculate the distance between all pairs of points
        from itertools import combinations

        pairs = list(combinations(range(4), 2))
        pair_dists = []
        for i, j in pairs:
            dist = np.linalg.norm(valid_points[i] - valid_points[j])
            pair_dists.append((dist, i, j))

        # Sort by distance
        pair_dists.sort()

        # The two closest points belong to the same cross.
        closest_pair = pair_dists[0]
        side1_indices = [closest_pair[1], closest_pair[2]]

        # The other two points belong to another cross.
        all_indices = set(range(4))
        side2_indices = list(all_indices - set(side1_indices))

        side1_points = valid_points[side1_indices]
        side2_points = valid_points[side2_indices]

        # Calculate the distance on both sides (original image scale)
        side1_dist_orig = np.linalg.norm(side1_points[0] - side1_points[1])
        side2_dist_orig = np.linalg.norm(side2_points[0] - side2_points[1])

        # Convert to 256x256 normalized scale
        side1_dist = normalized_distance(
            side1_points[0], side1_points[1]
        )  # Use x-direction scaling (or averaging can be used)
        side2_dist = normalized_distance(side2_points[0], side2_points[1])

        # Judgment Type
        type1 = 1 if side1_dist < distance_threshold else 0
        type2 = 1 if side2_dist < distance_threshold else 0

        # Sort by spatial position (side1 is "left", side2 is "right")
        # Determine left and right based on the projection in the direction of the principal axis
        main_projections = np.dot(centered, main_axis)
        side1_main_proj = np.mean(main_projections[side1_indices])
        side2_main_proj = np.mean(main_projections[side2_indices])

        if side1_main_proj < side2_main_proj:
            left_points, right_points = side1_points, side2_points
            left_type, right_type = type1, type2
            left_dist, right_dist = side1_dist, side2_dist
        else:
            left_points, right_points = side2_points, side1_points
            left_type, right_type = type2, type1
            left_dist, right_dist = side2_dist, side1_dist

        classification = f"{left_type}{right_type}"

        return {
            "num_dots": 4,
            "left_dots": 2,
            "right_dots": 2,
            "left_type": left_type,
            "right_type": right_type,
            "left_distance": float(left_dist),
            "right_distance": float(right_dist),
            "left_distance_orig": float(left_dist / scale_x),
            "right_distance_orig": float(right_dist / scale_x),
            "classification": classification,
            "left_points": left_points.tolist(),
            "right_points": right_points.tolist(),
            "centroid": centroid.tolist(),
            "main_axis": main_axis.tolist(),
            "split_direction": split_direction.tolist(),
        }

    # Non-4-point samples: Divided along the midline of the image (as a backup)
    mid_x = image_width / 2
    left_points = valid_points[valid_points[:, 0] < mid_x]
    right_points = valid_points[valid_points[:, 0] >= mid_x]

    left_dots = len(left_points)
    right_dots = len(right_points)

    # Determine the type on the left side: If there are 2 points and the distance is less than the threshold, it is 1; otherwise, it is 0.
    left_type = 0
    left_distance = None
    left_distance_256 = None
    if left_dots == 2:
        dist_orig = np.linalg.norm(left_points[0] - left_points[1])
        dist_256 = normalized_distance(left_points[0], left_points[1])
        left_distance = float(dist_orig)
        left_distance_256 = float(dist_256)
        if dist_256 < distance_threshold:
            left_type = 1

    # Determine the type on the right side
    right_type = 0
    right_distance = None
    right_distance_256 = None
    if right_dots == 2:
        dist_orig = np.linalg.norm(right_points[0] - right_points[1])
        dist_256 = normalized_distance(right_points[0], right_points[1])
        right_distance = float(dist_orig)
        right_distance_256 = float(dist_256)
        if dist_256 < distance_threshold:
            right_type = 1

    # Combined classification results (only meaningful when there are 2+2=4 points)
    if num_dots == 4:
        classification = f"{left_type}{right_type}"
    else:
        classification = "N/A"  # Samples with fewer than 4 points are not classified

    return {
        "num_dots": num_dots,
        "left_dots": left_dots,
        "right_dots": right_dots,
        "left_type": left_type,
        "right_type": right_type,
        "classification": classification,
        "left_distance": left_distance,
        "right_distance": right_distance,
        "left_distance_256": left_distance_256,
        "right_distance_256": right_distance_256,
        "left_points": left_points.tolist(),
        "right_points": right_points.tolist(),
    }


def predict_single(model, image_path, config, device="cuda", nms_distance=None):
    """Predict a single image

    Args:
    model: Loaded model
    image_path: Image path
    config: Configuration dictionary
    device: Computing device
    nms_distance: NMS distance threshold (overrides configuration)

    Returns:
    result: dict containing prediction results"""
    input_size = tuple(config["data"]["input_size"])
    num_keypoints = config["data"]["num_keypoints"]
    heatmap_size = tuple(config["heatmap"]["output_size"])
    threshold = config["inference"]["heatmap_threshold"]
    distance_threshold = config["data"].get("distance_threshold", 44)

    # NMS distance threshold (using command-line arguments or configuration)
    if nms_distance is None:
        nms_distance = config["inference"].get("nms_distance", 5)

    # Preprocessing
    img_tensor, original_size, original_img = preprocess_image(image_path, input_size)
    img_tensor = img_tensor.unsqueeze(0).to(device)  # [1, 1, H, W]

    # Inference
    with torch.no_grad():
        heatmaps_pred = model(img_tensor)  # [1, num_keypoints, H, W]

    heatmaps_pred = heatmaps_pred[0].cpu().numpy()

    # Extract keypoints from heatmaps (with NMS)
    pred_coords = heatmap_to_coordinates(
        heatmaps_pred,
        threshold=threshold,
        nms_distance=nms_distance,
        cross_channel_nms=(num_keypoints != 6),
    )

    # Convert back to the original image coordinate system
    pred_keypoints = np.zeros((num_keypoints, 3))
    scale_x = original_size[1] / input_size[1]
    scale_y = original_size[0] / input_size[0]

    for i, (x, y, conf) in enumerate(pred_coords):
        if conf > 0:
            # Heatmap -> Input Size -> Original Size
            input_x = x * (input_size[1] / heatmap_size[1])  # 64->256
            input_y = y * (input_size[0] / heatmap_size[0])
            pred_keypoints[i, 0] = input_x * scale_x  # 256->Original
            pred_keypoints[i, 1] = input_y * scale_y
            pred_keypoints[i, 2] = conf

    # Calculate the number of small white dots (dots with a confidence level greater than the threshold)
    num_dots = int(np.sum(pred_keypoints[:, 2] > threshold))

    # Perform classification (based on left and right positions)
    classification_result = classify_dots(
        pred_keypoints,
        image_width=original_size[1],
        image_height=original_size[0],
        distance_threshold=distance_threshold,
        input_size=input_size,
    )

    result = {
        "image_path": str(image_path),
        "image_name": Path(image_path).stem,
        "keypoints": pred_keypoints,  # [num_keypoints, 3]
        "num_dots": num_dots,
        "heatmap": heatmaps_pred,
        "original_size": original_size,
        "classification": classification_result,
    }

    return result


def predict_batch(
    model,
    image_dir,
    config,
    device="cuda",
    output_dir=None,
    val_mode=False,
    checkpoint_path=None,
    nms_distance=None,
):
    """Batch prediction of all images in the folder

    Args:
    model: Model
    image_dir: Image folder path
    config: Configuration
    device: Device
    output_dir: Output directory (optional)
    val_mode: Whether it is in validation set mode (creates a folder with a timestamp)
    checkpoint_path: Model path (used in val_mode)
    nms_distance: NMS distance threshold (overrides configuration)

    Returns:
    results: List of all prediction results"""
    image_dir = Path(image_dir)

    # Get all image files
    image_files = sorted(
        [
            f
            for f in image_dir.iterdir()
            if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
        ]
    )

    print(f"""{len(image_files)} images found""")

    results = []

    for img_path in tqdm(image_files, desc="""Predicting"""):
        try:
            result = predict_single(model, img_path, config, device, nms_distance=nms_distance)
            results.append(result)
        except Exception as e:
            print(f"""Error processing {img_path.name}: {e}""")

    # Save Results
    if output_dir:
        # Validation Set Mode: Create a subfolder with a timestamp
        if val_mode and checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
            train_name = checkpoint_path.parent.parent.name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(output_dir) / f"val_{train_name}_{timestamp}"
        else:
            output_dir = Path(output_dir)
            train_name = None
            timestamp = None

        output_dir.mkdir(parents=True, exist_ok=True)

        # Validation Set Mode: Save Model and README
        if val_mode and checkpoint_path:
            model_dir = output_dir / "model"
            model_dir.mkdir(exist_ok=True)
            shutil.copy(checkpoint_path, model_dir / "best_model.pth")
            with open(model_dir / "README.txt", "w", encoding="utf-8") as f:
                f.write(f"Training Run: {train_name}\n")
                f.write(f"Checkpoint: {checkpoint_path}\n")
                f.write(f"Val Date: {timestamp}\n")
                f.write(f"Val Data: {image_dir}\n")
                f.write("=" * 50 + "\n")
                f.write("Model Configuration:\n")
                f.write(f"  - HRNet Type: {config['model'].get('hrnet_type', 'N/A')}\n")
                f.write(f"  - Input Size: {config['data']['input_size']}\n")
                f.write(f"  - Heatmap Sigma: {config['heatmap']['sigma']}\n")
                f.write(f"  - Heatmap Threshold: {config['inference']['heatmap_threshold']}\n")

        # Create a visualization directory
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Create a subdirectory for 4-point samples
        four_dots_dir = viz_dir / "four_dots"
        four_dots_dir.mkdir(exist_ok=True)

        # Save Visualization
        for r in results:
            num_dots = r["num_dots"]
            if num_dots == 4:
                # 4-point sample: Save to subfolder, show categories only
                vis_img = visualize_prediction_simple(r, show_class_only=True)
                viz_path = four_dots_dir / f"{r['image_name']}_pred.png"
            else:
                # Non-4-point sample: Save to the main folder and display the number of points
                vis_img = visualize_prediction_simple(r, show_class_only=False)
                viz_path = viz_dir / f"{r['image_name']}_pred.png"
            cv2.imwrite(str(viz_path), vis_img)

        # Save as JSON format
        json_results = []
        for r in results:
            json_result = {
                "image_name": r["image_name"],
                "num_dots": r["num_dots"],
                "keypoints": r["keypoints"].tolist(),
                "original_size": r["original_size"],
                "classification": r["classification"]["classification"],
                "left_dots": r["classification"]["left_dots"],
                "right_dots": r["classification"]["right_dots"],
                "left_type": r["classification"]["left_type"],
                "right_type": r["classification"]["right_type"],
            }
            # Add distance information (if available)
            left_dist = r["classification"].get("left_distance")
            right_dist = r["classification"].get("right_distance")
            if left_dist is not None:
                json_result["left_distance"] = left_dist
            if right_dist is not None:
                json_result["right_distance"] = right_dist
            json_results.append(json_result)

        with open(output_dir / "val_results.json", "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

        # Save statistics
        num_dots_list = [r["num_dots"] for r in results]
        from collections import Counter

        dot_distribution = Counter(num_dots_list)

        # Statistical classification results
        class_counts = Counter([r["classification"]["classification"] for r in results])

        with open(output_dir / "val_statistics.txt", "w", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write("""Validation Result Statistics \n""")
            f.write("=" * 50 + "\n\n")
            if val_mode:
                f.write(f"""Training model: {train_name} \n""")
                f.write(f"""Validation time: {timestamp} \n""")
            f.write(f"""Verification Data: {image_dir} \n \n""")
            f.write("""Distribution of the number of white spots: \n""")
            for num_dots in sorted(dot_distribution.keys()):
                count = dot_distribution[num_dots]
                f.write(
                    f"""{num_dots} dots: {count} samples ( {count / len(results) * 100:.1f} %) \n"""
                )
            f.write(f"""\n Total: {len(results)} samples \n \n""")

            f.write("=" * 50 + "\n")
            f.write("""Classification statistics (4-point samples) \n""")
            f.write("=" * 50 + "\n")
            for cls in ["00", "01", "10", "11"]:
                count = class_counts.get(cls, 0)
                four_dot_count = sum(
                    1 for r in results if r["classification"]["classification"] == cls
                )
                f.write(f"""Type {cls}: {count} samples \n""")
            f.write(f"""N/A (not 4-point): {class_counts.get("N/A", 0)} samples \n""")

        # Count of 4-point samples
        four_dot_count = sum(1 for r in results if r["num_dots"] == 4)

        # Save 4-point sample category sequence
        four_dot_results = [r for r in results if r["num_dots"] == 4]
        if four_dot_results:
            class_sequence = " ".join(
                [r["classification"]["classification"] for r in four_dot_results]
            )
            with open(four_dots_dir / "class_sequence.txt", "w", encoding="utf-8") as f:
                f.write(class_sequence)
            print(f"""Category sequence saved: {four_dots_dir / "class_sequence.txt"}""")

        print(f"""\n Results saved to: {output_dir}""")
        if val_mode:
            print(f"""- Model file: {model_dir} /best_model.pth""")
            print(f"""- Model description: {model_dir} /README.txt""")
        print(f"""- Detailed results: {output_dir} /val_results.json""")
        print(f"""- Visualization (not 4-point): {viz_dir}""")
        print(
            f"""- Visualization (4-point sample): {four_dots_dir} / ( {four_dot_count} samples)"""
        )
        print(f"""- Statistics: {output_dir} /val_statistics.txt""")

    return results


def visualize_prediction_simple(result, show_class_only=False):
    """Simplified visualization (not dependent on matplotlib)

    Args:
    result: Prediction result
    show_class_only: If True, only the class is displayed (for 4-point samples), with smaller, centered text."""
    img = cv2.imread(str(result["image_path"]), cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    keypoints = result["keypoints"]
    # Use more distinctive colors
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # Red, Green, Blue, Yellow

    h, w = img.shape[:2]
    cls = result.get("classification", {})

    # Draw dividing lines
    if show_class_only and "centroid" in cls and "split_direction" in cls:
        # 4-point sample: Draw a dividing line perpendicular to the principal axis
        centroid = np.array(cls["centroid"])
        split_dir = np.array(cls["split_direction"])

        # The dividing line passes through the centroid, and its direction is split_direction.
        # Calculate the intersection of the line and the image boundary
        # Parametric equation: point = centroid + t * split_dir

        # Normalized direction vector
        split_dir = split_dir / np.linalg.norm(split_dir)

        # Calculate the intersection with the image boundary
        intersections = []

        # Intersection with the left boundary (x=0)
        if abs(split_dir[0]) > 1e-6:
            t = (0 - centroid[0]) / split_dir[0]
            y = centroid[1] + t * split_dir[1]
            if 0 <= y <= h:
                intersections.append((0, int(y)))

        # Intersection with the right boundary (x=w)
        if abs(split_dir[0]) > 1e-6:
            t = (w - centroid[0]) / split_dir[0]
            y = centroid[1] + t * split_dir[1]
            if 0 <= y <= h:
                intersections.append((w, int(y)))

        # Intersection with the upper boundary (y=0)
        if abs(split_dir[1]) > 1e-6:
            t = (0 - centroid[1]) / split_dir[1]
            x = centroid[0] + t * split_dir[0]
            if 0 <= x <= w:
                intersections.append((int(x), 0))

        # Intersection with the lower boundary (y=h)
        if abs(split_dir[1]) > 1e-6:
            t = (h - centroid[1]) / split_dir[1]
            x = centroid[0] + t * split_dir[0]
            if 0 <= x <= w:
                intersections.append((int(x), h))

        # Draw dividing lines
        if len(intersections) >= 2:
            cv2.line(img, intersections[0], intersections[1], (128, 128, 128), 1)

        # Draw centroid
        cv2.circle(img, (int(centroid[0]), int(centroid[1])), 2, (255, 0, 255), -1)
    else:
        # Non-4-point sample: Draw the centerline of the image
        mid_x = w // 2
        cv2.line(img, (mid_x, 0), (mid_x, h), (128, 128, 128), 1)

    # Draw points (do not show numbers)
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0:
            cx, cy = int(x), int(y)
            color = colors[i % len(colors)]
            # Draw dots
            cv2.circle(img, (cx, cy), 2, color, -1)
            cv2.circle(img, (cx, cy), 3, (255, 255, 255), 1)

    if show_class_only:
        # 4-point sample: Only the category is displayed, the font size is reduced by half, and it is centered slightly lower.
        cls = result.get("classification", {})
        class_label = cls.get("classification", "N/A")

        # Calculate text position (slightly below center)
        text = class_label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35  # Half of the original 0.7
        thickness = 1  # Half of the original 2
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Centered, slightly lower (at the bottom 1/4 of the image)
        text_x = (w - text_width) // 2
        text_y = h - 10  # 10 pixels from the bottom

        # Draw text (with black background to improve readability)
        cv2.rectangle(
            img,
            (text_x - 2, text_y - text_height - 2),
            (text_x + text_width + 2, text_y + 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 255, 255), thickness)
    else:
        # Non-4-point sample: The number of points is displayed in the top-left corner
        cv2.putText(
            img,
            f"dots: {result['num_dots']}",
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )

    return img


def visualize_result(result, save_path=None):
    """Visualize the prediction results of a single image"""
    # Read the original image
    img = cv2.imread(str(result["image_path"]), cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    keypoints = result["keypoints"]
    cls = result.get("classification", {})

    h, w = img.shape[:2]
    mid_x = w // 2

    # Draw the midline
    cv2.line(img, (mid_x, 0), (mid_x, h), (128, 128, 128), 1)

    # Draw key points
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0:
            cx, cy = int(x), int(y)
            color = colors[i % len(colors)]
            cv2.circle(img, (cx, cy), 2, color, -1)
            cv2.circle(img, (cx, cy), 3, (255, 255, 255), 1)
            side = "L" if x < mid_x else "R"
            cv2.putText(
                img, f"{i + 1}({side})", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

    # Add title information
    class_label = cls.get("classification", "N/A")
    cv2.putText(
        img,
        f"Type: {class_label}, Dots: {result['num_dots']}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # Display left and right information
    left_text = f"L: {cls.get('left_dots', 0)}d, T={cls.get('left_type', -1)}"
    if cls.get("left_distance"):
        left_text += f", d={cls['left_distance']:.1f}px"
    cv2.putText(img, left_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    right_text = f"R: {cls.get('right_dots', 0)}d, T={cls.get('right_type', -1)}"
    if cls.get("right_distance"):
        right_text += f", d={cls['right_distance']:.1f}px"
    cv2.putText(img, right_text, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display or Save
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f"{result['image_name']} - Type {class_label} ({result['num_dots']} dots)")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"""Visualization results saved: {save_path}""")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Predict DNA dot keypoints using HRNet")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="path to config file"
    )
    parser.add_argument("--weights", type=str, required=True, help="path to model weights")
    parser.add_argument("--input", type=str, required=True, help="input image path or directory")
    parser.add_argument(
        "--output", type=str, default="outputs/predictions", help="output directory"
    )
    parser.add_argument("--device", type=str, default="cuda", help="device to use")
    parser.add_argument("--viz", action="store_true", help="visualize results")
    parser.add_argument(
        "--val", action="store_true", help="validation mode: save to val_results with timestamp"
    )
    parser.add_argument(
        "--nms-distance",
        type=int,
        default=None,
        help="NMS distance threshold (default: use config value)",
    )
    args = parser.parse_args()

    # Load Configuration
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"""Device in use: {device}""")

    # Load Model
    print(f"""Load model: {args.weights}""")
    model = load_model(args.weights, config, device)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine whether the input is a file or a directory
    input_path = Path(args.input)

    if input_path.is_file():
        # Single image prediction
        print(f"""Predict a single image: {input_path}""")
        result = predict_single(model, input_path, config, device, nms_distance=args.nms_distance)

        cls = result["classification"]

        print(f"""\n Prediction result:""")
        print(f"""Image: {result["image_name"]}""")
        print(f"""Number of white spots: {result["num_dots"]}""")
        print(f"""Classification result: {cls["classification"]}""")
        print(f"""Left side: {cls["left_dots"]} spots, Type= {cls["left_type"]}""", end="")
        if cls["left_distance"] is not None:
            print(f""", Distance= {cls["left_distance"]:.1f} px""")
        else:
            print()
        print(f"""Right side: {cls["right_dots"]} points, Type= {cls["right_type"]}""", end="")
        if cls["right_distance"] is not None:
            print(f""", Distance= {cls["right_distance"]:.1f} px""")
        else:
            print()
        print(f"""Key point coordinates:""")
        for i, (x, y, conf) in enumerate(result["keypoints"]):
            if conf > 0:
                side = """Left""" if x < result["original_size"][1] / 2 else """Right"""
                print(f"""Point {i + 1}: ({x:.1f}, {y:.1f}), Confidence= {conf:.3f}, [{side}]""")

        if args.viz:
            viz_path = output_dir / f"{result['image_name']}_pred.png"
            visualize_result(result, save_path=viz_path)

    elif input_path.is_dir():
        # Batch Prediction
        print(f"""Batch prediction directory: {input_path}""")

        # Validation Set Mode: Use the outputs/val_results directory
        if args.val:
            output_dir = "outputs/val_results"
            print(
                """[Validation Set Mode] Results will be saved to outputs/val_results/val_ {train_name} _ {timestamp} /"""
            )

        results = predict_batch(
            model,
            input_path,
            config,
            device,
            output_dir,
            val_mode=args.val,
            checkpoint_path=args.weights,
            nms_distance=args.nms_distance,
        )

        # Print statistics
        num_dots_list = [r["num_dots"] for r in results]
        from collections import Counter

        dot_distribution = Counter(num_dots_list)

        print("""\n Distribution of the number of white spots:""")
        for num_dots in sorted(dot_distribution.keys()):
            count = dot_distribution[num_dots]
            print(f"""{num_dots} points: {count} samples ({count / len(results) * 100:.1f}%)""")

        # Print Classification Statistics
        class_counts = Counter([r["classification"]["classification"] for r in results])
        print("""\n Classification Result Statistics:""")
        for cls in ["00", "01", "10", "11"]:
            count = class_counts.get(cls, 0)
            print(f"""Type {cls}: {count} samples ({count / len(results) * 100:.1f}%)""")
        print(f"""N/A (not 4 points): {class_counts.get("N/A", 0)} samples""")

    else:
        print(f"""Error: Input path does not exist: {input_path}""")


if __name__ == "__main__":
    main()
