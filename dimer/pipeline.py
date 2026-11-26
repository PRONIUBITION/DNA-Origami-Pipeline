"""DNA Origami White Dot Detection and Classification - All-in-One Integrated Script

Integrates Stage 1 (YOLOv5 segmentation) and Stage 2 (HRNet classification),
Achieve a fully automated process from raw images to category sequence codes.

Example usage:
 python pipeline.py \
 --input "D:/data/raw_afm_images" \
 --yolo-weights "models/yolo_best.pt" \
 --hrnet-weights "models/hrnet_best.pth" \
 --output "outputs/pipeline_results"

Output:
 - output/clean_images/ # Clean images output from Stage 1
 - output/sequences.txt # Class sequence code (4-point sample)
 - output/detailed_results.json # Detailed results (including all samples)
 - output/visualizations/ # Visualization results"""

import os
import sys
import yaml
import json
import shutil
import subprocess
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from collections import Counter

# ========== Configuration ==========

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
YOLOV5_DIR = Path(os.environ.get("YOLOV5_DIR", WORKSPACE_ROOT / "third_party" / "yolov5")).resolve()

# Defaults are repository-relative. Deployment-specific paths should be passed
# through command-line arguments or environment variables.
DEFAULT_CONFIG = {
    # YOLOv5 Configuration
    "yolov5_dir": str(YOLOV5_DIR),
    "yolo_img_size": 640,
    "yolo_conf": 0.45,
    "target_class": 0,  # 0=dimmer
    "margin_ratio": 0.10,
    "min_margin": 20,
    "dilate_pixels": 6,
    # HRNet Configuration
    "hrnet_config": str(PROJECT_ROOT / "stage2_hrnet" / "configs" / "config.yaml"),
    "input_size": [256, 256],
    "heatmap_size": [64, 64],
    "heatmap_threshold": 0.1,
    "distance_threshold": 50,
}


# ========== Utility Functions ==========


def log(message, level="INFO"):
    """Print logs with timestamps"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def cv2_imread_unicode(path):
    """cv2.imread with support for Chinese file paths"""
    with open(path, "rb") as f:
        img_array = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)


def cv2_imwrite_unicode(path, img):
    """cv2.imwrite with support for Chinese file paths"""
    ext = Path(path).suffix.lower()
    if ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    elif ext in [".jpg", ".jpeg"]:
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    else:
        params = []

    success, buffer = cv2.imencode(ext, img, params)
    if success:
        with open(path, "wb") as f:
            f.write(buffer)
        return True
    return False


def compute_iou(box1, box2):
    """Calculate the IOU of two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / (union_area + 1e-6)


# ========== Stage 1: YOLOv5 Segmentation ==========


class Stage1YOLOv5:
    """Stage 1: YOLOv5 Object Detection and Segmentation"""

    def __init__(self, config):
        self.config = config
        self.yolov5_dir = Path(config["yolov5_dir"])

    def prepare_images(self, input_dir, output_dir, img_size=640):
        """Prepare image: convert to grayscale, resize"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
        log(f"""{len(image_files)} original images found""")

        resize_info = {}

        for img_path in tqdm(image_files, desc="""[Stage 1] Prepare images"""):
            img = cv2_imread_unicode(str(img_path))
            if img is None:
                log(f"""Unable to read: {img_path.name}""", "WARN")
                continue

            # Convert to Grayscale
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img

            h, w = img_gray.shape

            # Calculate the zoom ratio (maintain the aspect ratio)
            scale = min(img_size / w, img_size / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            # Zoom
            resized = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Create a square canvas (filled with black borders)
            square = np.zeros((img_size, img_size), dtype=np.uint8)
            y_offset = (img_size - new_h) // 2
            x_offset = (img_size - new_w) // 2
            square[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

            # Save
            output_path = output_dir / img_path.name
            cv2.imwrite(str(output_path), square)

            # Record zoom information
            resize_info[img_path.stem] = {
                "original_size": (w, h),
                "scale": scale,
                "offset": (x_offset, y_offset),
                "resized_size": (new_w, new_h),
            }

        # Save zoom information
        with open(output_dir / "resize_info.json", "w") as f:
            json.dump(resize_info, f, indent=2)

        log(f"""Image preparation complete. Saved to: {output_dir}""")
        return output_dir, resize_info

    def run_inference(self, weights, source, img_size=640, conf=0.45):
        """Run YOLOv5 inference"""
        # Convert to an absolute path before changing the directory to avoid the relative path becoming invalid under yolov5_dir.
        weights = str(Path(weights).resolve())
        source = str(Path(source).resolve())

        cmd = [
            sys.executable,
            "segment/predict.py",
            f"--weights={weights}",
            f"--source={source}",
            f"--imgsz={img_size}",
            f"--conf-thres={conf}",
            "--save-txt",
            "--save-conf",
            "--exist-ok",
        ]

        log(f"""Running YOLOv5 inference...""")
        log(f"""Command: {" ".join(cmd)}""")

        original_dir = os.getcwd()
        os.chdir(self.yolov5_dir)

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            log("""YOLOv5 inference completed""")
        except subprocess.CalledProcessError as e:
            log(f"""YOLOv5 inference failed: {e}""", "ERROR")
            os.chdir(original_dir)
            return None
        finally:
            os.chdir(original_dir)

        # Found the latest exp directory
        runs_dir = self.yolov5_dir / "runs" / "predict-seg"
        exp_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("exp")]

        if not exp_dirs:
            log("""Prediction results directory not found""", "ERROR")
            return None

        latest_exp = max(exp_dirs, key=lambda d: d.stat().st_mtime)
        log(f"""Inference result: {latest_exp}""")

        return latest_exp

    def convert_masks(
        self,
        labels_dir,
        original_images,
        output_dir,
        resize_info,
        target_class=0,
        margin_ratio=0.10,
        min_margin=20,
        iou_threshold=0.5,
    ):
        """Convert YOLO output to a mask and cropped image for a single instance"""
        labels_dir = Path(labels_dir)
        original_images = Path(original_images)
        output_dir = Path(output_dir)

        out_images = output_dir / "images"
        out_masks = output_dir / "masks"
        out_images.mkdir(parents=True, exist_ok=True)
        out_masks.mkdir(parents=True, exist_ok=True)

        txt_files = list(labels_dir.glob("*.txt"))
        log(f"""Processing {len(txt_files)} label files""")

        instance_count = 0
        total_filtered = 0

        for txt_file in tqdm(txt_files, desc="""[Stage 1] Convert mask"""):
            img_stem = txt_file.stem

            # Read the original image
            img_path = original_images / f"{img_stem}.png"
            if not img_path.exists():
                img_path = original_images / f"{img_stem}.jpg"
            if not img_path.exists():
                continue

            img = cv2_imread_unicode(str(img_path))
            if img is None:
                continue

            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            h, w = img.shape

            # Read label
            with open(txt_file, "r") as f:
                lines = f.readlines()

            # Collect candidate detections
            candidates = []
            for line_idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                if class_id != target_class:
                    continue

                points = []
                for i in range(1, len(parts), 2):
                    if i + 1 < len(parts):
                        x = float(parts[i]) * w
                        y = float(parts[i + 1]) * h
                        points.append([int(x), int(y)])

                if len(points) < 3:
                    continue

                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                bw, bh = x2 - x1, y2 - y1

                # Filter out areas that are too small
                if bw * bh < 3000:
                    continue

                # Filter out areas that are too flat or too long
                aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)
                if aspect_ratio > 2.0:
                    continue

                candidates.append(
                    {
                        "points": points,
                        "bbox": (x1, y1, x2, y2),
                        "area": bw * bh,
                        "line_idx": line_idx,
                    }
                )

            # IOU deduplication
            candidates.sort(key=lambda x: x["area"], reverse=True)
            kept = []
            for cand in candidates:
                is_duplicate = False
                for kept_cand in kept:
                    iou = compute_iou(cand["bbox"], kept_cand["bbox"])
                    if iou > iou_threshold:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    kept.append(cand)

            total_filtered += len(candidates) - len(kept)

            # Process retained detections
            for idx, cand in enumerate(kept):
                points = cand["points"]
                x1, y1, x2, y2 = cand["bbox"]
                bw, bh = x2 - x1, y2 - y1

                # Expand bounding box
                margin_x = max(int(bw * margin_ratio), min_margin)
                margin_y = max(int(bh * margin_ratio), min_margin)

                x1 = max(0, x1 - margin_x)
                y1 = max(0, y1 - margin_y)
                x2 = min(w, x2 + margin_x)
                y2 = min(h, y2 + margin_y)

                # Create mask
                full_mask = np.zeros((h, w), dtype=np.uint8)
                pts = np.array(points, np.int32)
                cv2.fillPoly(full_mask, [pts], 255)

                # Crop
                mask_crop = full_mask[y1:y2, x1:x2]
                img_crop = img[y1:y2, x1:x2]

                # Save
                base_name = f"{img_stem}_inst{idx}"

                cv2_imwrite_unicode(str(out_images / f"{base_name}_img.png"), img_crop)
                cv2_imwrite_unicode(str(out_masks / f"{base_name}_mask.png"), mask_crop)

                mask_np = (mask_crop > 0).astype(np.float32)
                np.save(str(out_masks / f"{base_name}_mask.npy"), mask_np)

                instance_count += 1

        log(f"""Generate {instance_count} instances (filter duplicates: {total_filtered})""")
        return output_dir

    def clean_images(self, input_img_dir, input_mask_dir, output_clean_dir, dilate_pixels=6):
        """Apply mask and remove background after dilation"""
        input_img_dir = Path(input_img_dir)
        input_mask_dir = Path(input_mask_dir)
        output_clean_dir = Path(output_clean_dir)
        output_clean_dir.mkdir(parents=True, exist_ok=True)

        img_files = list(input_img_dir.glob("*_img.png"))

        for img_path in tqdm(img_files, desc="""[Stage 1] Clean image"""):
            stem = img_path.stem.replace("_img", "")
            mask_path = input_mask_dir / f"{stem}_mask.npy"

            if not mask_path.exists():
                continue

            try:
                img = cv2_imread_unicode(str(img_path))
                if img is None:
                    continue
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mask = np.load(str(mask_path))

                if img.shape != mask.shape:
                    mask = cv2.resize(
                        mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
                    )

                # Dilate mask
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (dilate_pixels, dilate_pixels)
                )
                mask_dilated = cv2.dilate(mask, kernel, iterations=1)
                mask_dilated = (mask_dilated > 0).astype(np.float32)

                # Apply mask
                clean_img = (img * mask_dilated).astype(np.uint8)

                # Save
                cv2_imwrite_unicode(str(output_clean_dir / f"{stem}_clean.png"), clean_img)

            except Exception as e:
                log(f"""Processing {stem} failed: {e}""", "WARN")

        log(f"""Clean image saved to: {output_clean_dir}""")
        return output_clean_dir

    def run(self, input_dir, weights, temp_dir):
        """Run the complete Phase 1 process"""
        log("=" * 60)
        log("""Start Phase 1: YOLOv5 Object Detection and Segmentation""")
        log("=" * 60)

        # Step 1: Prepare the image
        prepared_dir, resize_info = self.prepare_images(
            input_dir, temp_dir / "prepared", self.config["yolo_img_size"]
        )

        # Step 2: YOLOv5 Inference
        exp_dir = self.run_inference(
            weights, str(prepared_dir), self.config["yolo_img_size"], self.config["yolo_conf"]
        )

        if exp_dir is None:
            raise RuntimeError("""YOLOv5 inference failed""")

        # Step 3: Convert mask
        labels_dir = exp_dir / "labels"
        converted_dir = self.convert_masks(
            labels_dir,
            prepared_dir,
            temp_dir / "converted",
            resize_info,
            self.config["target_class"],
            self.config["margin_ratio"],
            self.config["min_margin"],
        )

        # Step 4: Clean the image
        clean_dir = self.clean_images(
            converted_dir / "images",
            converted_dir / "masks",
            temp_dir / "clean_images",
            self.config["dilate_pixels"],
        )

        log("""Stage 1 completed""")
        return clean_dir


# ========== Stage 2: HRNet Classification ==========


class Stage2HRNet:
    """Stage 2: HRNet Keypoint Detection and Classification"""

    def __init__(self, config_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load configuration
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Update some configurations
        self.input_size = tuple(self.config["data"]["input_size"])
        self.heatmap_size = tuple(self.config["heatmap"]["output_size"])
        self.num_keypoints = self.config["data"]["num_keypoints"]
        self.threshold = self.config["inference"]["heatmap_threshold"]
        self.distance_threshold = self.config["data"].get("distance_threshold", 50)

    def load_model(self, checkpoint_path):
        """Load trained model"""
        sys.path.insert(0, str(Path(__file__).parent / "stage2_hrnet"))

        from models import HRNet, SimpleHRNet, hrnet_w32_cfg

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model_type = self.config["model"].get("hrnet_type", "SimpleHRNet")

        if model_type == "SimpleHRNet":
            model = SimpleHRNet(num_keypoints=self.num_keypoints, in_channels=1)
        elif model_type == "HRNet-W32":
            cfg = hrnet_w32_cfg.copy()
            cfg["NUM_HEATMAPS"] = self.num_keypoints
            model = HRNet(cfg)
        else:
            raise ValueError(f"""Unknown model type: {model_type}""")

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        self.model = model
        log(f"""HRNet model loaded: {checkpoint_path}""")

    def preprocess(self, image_path):
        """Preprocess image"""
        img = cv2_imread_unicode(str(image_path))
        if img is None:
            raise ValueError(f"""Unable to read image: {image_path}""")

        # Ensure it is a grayscale image
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        original_size = img.shape[:2]

        # Resize
        img_resized = cv2.resize(img, (self.input_size[1], self.input_size[0]))

        # Normalization
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_normalized = (img_normalized - 0.5) / 0.5

        # Convert to tensor
        img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).float()

        return img_tensor, original_size, img

    def heatmap_to_coordinates(self, heatmaps, threshold=0.1):
        """Extract keypoint coordinates from the heatmap"""
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

    def classify_pca_grouping(self, keypoints, distance_threshold):
        """PCA projection grouping (fallback strategy)"""
        valid_indices = np.where(keypoints[:, 2] > 0)[0]
        valid_points = keypoints[valid_indices][:, :2]
        num_dots = len(valid_points)

        if num_dots != 4:
            return {"num_dots": num_dots, "classification": "N/A"}

        centroid = np.mean(valid_points, axis=0)
        centered = valid_points - centroid
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        main_axis = eigenvectors[:, np.argmax(eigenvalues)]
        split_direction = np.array([-main_axis[1], main_axis[0]])

        # PCA projection grouping
        projections = np.dot(centered, split_direction)
        sorted_indices = np.argsort(projections)

        side1_indices = sorted_indices[:2]
        side2_indices = sorted_indices[2:]

        side1_points = valid_points[side1_indices]
        side2_points = valid_points[side2_indices]
        side1_original_indices = valid_indices[side1_indices].tolist()
        side2_original_indices = valid_indices[side2_indices].tolist()

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

        return {
            "num_dots": 4,
            "classification": f"{left_type}{right_type}",
            "left_type": left_type,
            "right_type": right_type,
            "centroid": centroid.tolist(),
            "split_direction": split_direction.tolist(),
            "main_axis": main_axis.tolist(),
            "side1_indices": side1_original_indices,
            "side2_indices": side2_original_indices,
            "side1_coords": side1_points.tolist(),
            "side2_coords": side2_points.tolist(),
        }

    def classify_distance_grouping(self, keypoints, distance_threshold):
        """Distance grouping (main strategy)"""
        valid_indices = np.where(keypoints[:, 2] > 0)[0]
        valid_points = keypoints[valid_indices][:, :2]
        num_dots = len(valid_points)

        if num_dots != 4:
            return {"num_dots": num_dots, "classification": "N/A"}

        # PCA finds the main direction (used only to determine left and right and to draw dividing lines)
        centroid = np.mean(valid_points, axis=0)
        centered = valid_points - centroid
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        main_axis = eigenvectors[:, np.argmax(eigenvalues)]
        split_direction = np.array([-main_axis[1], main_axis[0]])

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
        confidence_ratio = second_min_dist / (min_dist + 1e-6)

        closest_pair = pair_dists[0]
        side1_indices = [closest_pair[1], closest_pair[2]]
        all_indices = set(range(4))
        side2_indices = list(all_indices - set(side1_indices))

        side1_points = valid_points[side1_indices]
        side2_points = valid_points[side2_indices]
        side1_original_indices = valid_indices[side1_indices].tolist()
        side2_original_indices = valid_indices[side2_indices].tolist()

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

        return {
            "num_dots": 4,
            "classification": f"{left_type}{right_type}",
            "left_type": left_type,
            "right_type": right_type,
            "centroid": centroid.tolist(),
            "split_direction": split_direction.tolist(),
            "main_axis": main_axis.tolist(),
            "side1_indices": side1_original_indices,
            "side2_indices": side2_original_indices,
            "side1_coords": side1_points.tolist(),
            "side2_coords": side2_points.tolist(),
            "confidence_ratio": float(confidence_ratio),
            "method": "distance",
        }

    def classify_dots(self, keypoints, image_width):
        """Hybrid grouping strategy: Distance grouping is primary, PCA is fallback"""
        dist_result = self.classify_distance_grouping(keypoints, self.distance_threshold)

        if dist_result.get("num_dots") != 4:
            return dist_result

        ratio = dist_result.get("confidence_ratio", 999.0)

        if ratio > 1.5:
            # High confidence; directly use distance grouping
            dist_result["method"] = "distance_high_conf"
            return dist_result
        elif ratio >= 1.0:
            # Ambiguous region: Apply both methods and check for consistency.
            pca_result = self.classify_pca_grouping(keypoints, self.distance_threshold)
            if pca_result["classification"] == dist_result["classification"]:
                log(
                    f"""Mixed grouping: ratio= {ratio:.2f} Ambiguous region, distance/PCA consensus= {dist_result["classification"]}"""
                )
                dist_result["method"] = "consensus"
                return dist_result
            else:
                log(
                    f"""Mixed grouping warning: ratio= {ratio:.2f} ambiguous region, distance= {dist_result["classification"]} vs PCA= {pca_result["classification"]}, distance grouping used"""
                )
                dist_result["method"] = "distance_fallback"
                dist_result["pca_classification"] = pca_result["classification"]
                return dist_result
        else:
            # Ratio < 1.0, distance grouping failed, revert to PCA
            pca_result = self.classify_pca_grouping(keypoints, self.distance_threshold)
            log(
                f"""Mixed grouping: ratio= {ratio:.2f} < 1.0, distance grouping failed, revert to PCA projection grouping= {pca_result["classification"]}"""
            )
            pca_result["method"] = "pca_fallback"
            pca_result["distance_classification"] = dist_result["classification"]
            pca_result["confidence_ratio"] = ratio
            return pca_result

    def predict_single(self, image_path):
        """Predict a single image"""
        img_tensor, original_size, original_img = self.preprocess(image_path)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            heatmaps_pred = self.model(img_tensor)

        heatmaps_pred = heatmaps_pred[0].cpu().numpy()
        pred_coords = self.heatmap_to_coordinates(heatmaps_pred, threshold=self.threshold)

        pred_keypoints_input = np.zeros((self.num_keypoints, 3))
        pred_keypoints_orig = np.zeros((self.num_keypoints, 3))

        scale_x = original_size[1] / self.input_size[1]
        scale_y = original_size[0] / self.input_size[0]

        for i, (x, y, conf) in enumerate(pred_coords):
            if conf > 0:
                input_x = x * (self.input_size[1] / self.heatmap_size[1])
                input_y = y * (self.input_size[0] / self.heatmap_size[0])
                pred_keypoints_input[i, 0] = input_x
                pred_keypoints_input[i, 1] = input_y
                pred_keypoints_input[i, 2] = conf
                pred_keypoints_orig[i, 0] = input_x * scale_x
                pred_keypoints_orig[i, 1] = input_y * scale_y
                pred_keypoints_orig[i, 2] = conf

        num_dots = int(np.sum(pred_keypoints_input[:, 2] > self.threshold))
        classification = self.classify_dots(pred_keypoints_input, self.input_size[1])

        # Map the classification coordinates from the 256x256 input scale back to the original image scale for visualization.
        scale_x = original_size[1] / self.input_size[1]
        scale_y = original_size[0] / self.input_size[0]
        if "centroid" in classification:
            classification["centroid"] = [
                classification["centroid"][0] * scale_x,
                classification["centroid"][1] * scale_y,
            ]
        if "side1_coords" in classification:
            classification["side1_coords"] = [
                [c[0] * scale_x, c[1] * scale_y] for c in classification["side1_coords"]
            ]
        if "side2_coords" in classification:
            classification["side2_coords"] = [
                [c[0] * scale_x, c[1] * scale_y] for c in classification["side2_coords"]
            ]

        return {
            "image_path": str(image_path),
            "image_name": Path(image_path).stem,
            "keypoints": pred_keypoints_orig,
            "keypoints_input": pred_keypoints_input,
            "num_dots": num_dots,
            "classification": classification,
            "original_size": original_size,
        }

    def visualize_result(self, result, output_path):
        """Visualize single-slide results, including PCA dividing lines"""
        img = cv2_imread_unicode(str(result["image_path"]))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        keypoints = result["keypoints"]
        h, w = img.shape[:2]

        cls_info = result["classification"]
        if result["num_dots"] == 4 and "side1_coords" in cls_info:
            side1_coords = cls_info["side1_coords"]
            side2_coords = cls_info["side2_coords"]
            for coord in side1_coords:
                x, y = int(coord[0]), int(coord[1])
                cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
                cv2.circle(img, (x, y), 6, (255, 255, 255), 1)
            for coord in side2_coords:
                x, y = int(coord[0]), int(coord[1])
                cv2.circle(img, (x, y), 4, (255, 0, 0), -1)
                cv2.circle(img, (x, y), 6, (255, 255, 255), 1)

            if "centroid" in cls_info and "split_direction" in cls_info:
                centroid = np.array(cls_info["centroid"])
                split_dir = np.array(cls_info["split_direction"])
                split_dir = split_dir / (np.linalg.norm(split_dir) + 1e-6)
                dx, dy = split_dir[0], split_dir[1]
                points = []
                if abs(dx) > 1e-6:
                    t = (0 - centroid[0]) / dx
                    y = centroid[1] + t * dy
                    if 0 <= y <= h:
                        points.append((0, int(y)))
                    t = (w - centroid[0]) / dx
                    y = centroid[1] + t * dy
                    if 0 <= y <= h:
                        points.append((w, int(y)))
                if abs(dy) > 1e-6:
                    t = (0 - centroid[1]) / dy
                    x = centroid[0] + t * dx
                    if 0 <= x <= w:
                        points.append((int(x), 0))
                    t = (h - centroid[1]) / dy
                    x = centroid[0] + t * dx
                    if 0 <= x <= w:
                        points.append((int(x), h))
                if len(points) >= 2:
                    cv2.line(img, points[0], points[1], (0, 255, 255), 2)
                cv2.circle(img, (int(centroid[0]), int(centroid[1])), 3, (0, 255, 0), -1)

            if len(side1_coords) == 2 and len(side2_coords) == 2:
                p1 = (int(side1_coords[0][0]), int(side1_coords[0][1]))
                p2 = (int(side1_coords[1][0]), int(side1_coords[1][1]))
                cv2.line(img, p1, p2, (0, 0, 255), 2)
                p3 = (int(side2_coords[0][0]), int(side2_coords[0][1]))
                p4 = (int(side2_coords[1][0]), int(side2_coords[1][1]))
                cv2.line(img, p3, p4, (255, 0, 0), 2)
        else:
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > 0:
                    cx, cy = int(x), int(y)
                    color = colors[i % len(colors)]
                    cv2.circle(img, (cx, cy), 4, color, -1)
                    cv2.circle(img, (cx, cy), 6, (255, 255, 255), 1)

        cls = cls_info["classification"]
        text = cls if result["num_dots"] == 4 else f"dots: {result['num_dots']}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5 if result["num_dots"] == 4 else 0.4
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = (w - text_width) // 2
        text_y = h - 10
        cv2.rectangle(
            img,
            (text_x - 2, text_y - text_height - 2),
            (text_x + text_width + 2, text_y + 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 255, 255), thickness)

        cv2_imwrite_unicode(str(output_path), img)

    def run(self, clean_dir, output_dir):
        """Run the complete Phase 2 process"""
        log("=" * 60)
        log("""Start Phase 2: HRNet Keypoint Detection and Classification""")
        log("=" * 60)

        clean_dir = Path(clean_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        classified_dir = output_dir / "classified_images"
        four_dots_dir = classified_dir / "four_dots"
        non_four_dots_dir = classified_dir / "non_four_dots"
        classified_dir.mkdir(exist_ok=True)
        four_dots_dir.mkdir(exist_ok=True)
        non_four_dots_dir.mkdir(exist_ok=True)

        image_files = sorted(
            [f for f in clean_dir.iterdir() if f.suffix.lower() == ".png" and "_clean" in f.name]
        )

        log(f"""{len(image_files)} clean images found""")

        results = []
        for img_path in tqdm(image_files, desc="""[Stage 2] Prediction"""):
            try:
                result = self.predict_single(img_path)
                results.append(result)

                if result["num_dots"] == 4:
                    self.visualize_result(
                        result, four_dots_dir / f"{result['image_name']}_classified.png"
                    )
                else:
                    self.visualize_result(
                        result,
                        non_four_dots_dir / f"{result['image_name']}_dots{result['num_dots']}.png",
                    )
            except Exception as e:
                log(f"""Processing {img_path.name} failed: {e}""", "WARN")

        four_dot_results = [r for r in results if r["num_dots"] == 4]
        class_sequence = " ".join([r["classification"]["classification"] for r in four_dot_results])

        dot_distribution = Counter([r["num_dots"] for r in results])
        class_counts = Counter([r["classification"]["classification"] for r in results])

        log(f"""\n Test Result Statistics:""")
        for num_dots in sorted(dot_distribution.keys()):
            count = dot_distribution[num_dots]
            log(f"""{num_dots} points: {count} samples ({count / len(results) * 100:.1f}%)""")

        log(f"""\n Classification Results Statistics (4-point samples):""")
        for cls in ["00", "01", "10", "11"]:
            count = class_counts.get(cls, 0)
            log(f"""Type {cls}: {count} samples""")
        log(f"""N/A (not 4 points): {class_counts.get("N/A", 0)} samples""")

        with open(output_dir / "class_sequence.txt", "w", encoding="utf-8") as f:
            f.write(class_sequence)

        json_results = []
        for r in results:
            json_result = {
                "image_name": r["image_name"],
                "num_dots": r["num_dots"],
                "classification": r["classification"]["classification"],
                "keypoints": r["keypoints"].tolist(),
            }
            if r["classification"].get("left_type") is not None:
                json_result["left_type"] = r["classification"]["left_type"]
                json_result["right_type"] = r["classification"]["right_type"]
            json_results.append(json_result)

        with open(output_dir / "detailed_results.json", "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

        with open(output_dir / "statistics.txt", "w", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write("""DNA origami white dot detection classification results \n""")
            f.write("=" * 50 + "\n\n")
            f.write(f"""Processing time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} \n""")
            f.write(f"""Total number of samples: {len(results)} \n""")
            f.write(f"""Number of 4-point samples: {len(four_dot_results)} \n \n""")

            f.write("""Distribution of small white spots: \n""")
            for num_dots in sorted(dot_distribution.keys()):
                count = dot_distribution[num_dots]
                f.write(
                    f"""{num_dots} dots: {count} samples ( {count / len(results) * 100:.1f} %) \n"""
                )

            f.write("""\n Classification Statistics: \n""")
            for cls in ["00", "01", "10", "11"]:
                count = class_counts.get(cls, 0)
                f.write(f"""Type {cls}: {count} samples \n""")
            f.write(f"""N/A (not 4-point): {class_counts.get("N/A", 0)} samples \n""")

            f.write("\n" + "=" * 50 + "\n")
            f.write("""Category sequence code (4-point sample): \n""")
            f.write(class_sequence + "\n")

        log(f"""\n Results saved to: {output_dir}""")
        log(f"""- Category sequence: {output_dir / "class_sequence.txt"}""")
        log(f"""- Detailed results: {output_dir / "detailed_results.json"}""")
        log(f"""- Statistics: {output_dir / "statistics.txt"}""")
        log(f"""- Classified images: {classified_dir} /""")
        log(f"""- 4-point sample: {four_dots_dir} /""")
        log(f"""- Non-4-point sample: {non_four_dots_dir} /""")

        return results, class_sequence


# ========== Main Process ==========


def main():
    parser = argparse.ArgumentParser(
        description="""DNA Origami White Dot Detection Classification - One-Stop Integrated Script""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example:
 # Basic Usage
 python pipeline.py \\
 --input "D:/data/raw_afm_images" \\
 --yolo-weights "models/yolo_best.pt" \\
 --hrnet-weights "models/hrnet_best.pth"

 # Specify the output directory
 python pipeline.py \\
 --input "D:/data/raw_afm_images" \\
 --yolo-weights "..." \\
 --hrnet-weights "..." \\
 --output "my_results"

 # Run using the CPU
 python pipeline.py \\
 --input "..." \\
 --yolo-weights "..." \\
 --hrnet-weights "..." \\
 --device cpu""",
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="""Enter the directory of the original AFM images""",
    )
    parser.add_argument(
        "--yolo-weights", type=str, required=True, help="""YOLOv5 Model Weight Path"""
    )
    parser.add_argument(
        "--hrnet-weights", type=str, required=True, help="""HRNet model weight path"""
    )
    parser.add_argument(
        "--hrnet-config",
        type=str,
        default=DEFAULT_CONFIG["hrnet_config"],
        help="""HRNet configuration file path""",
    )
    parser.add_argument(
        "--output", type=str, default="datapredict/outputs", help="""Output results directory"""
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="""Running device (CUDA/CPU)""",
    )
    parser.add_argument(
        "--keep-temp", action="store_true", help="""Keep temporary files (for debugging)"""
    )

    args = parser.parse_args()

    # Create an output directory with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"pipeline_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a temporary directory
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    log("=" * 60)
    log("""DNA Origami White Dot Detection Classification - One-Stop Processing""")
    log("=" * 60)
    log(f"""Input directory: {args.input}""")
    log(f"""YOLOv5 Weights: {args.yolo_weights}""")
    log(f"""HRNet weights: {args.hrnet_weights}""")
    log(f"""Output directory: {output_dir}""")
    log(f"""Device: {args.device}""")

    try:
        # ========== Stage 1: YOLOv5 Segmentation ==========
        stage1 = Stage1YOLOv5(DEFAULT_CONFIG)
        clean_dir = stage1.run(args.input, args.yolo_weights, temp_dir)

        # Copy the clean image to the output directory
        final_clean_dir = output_dir / "clean_images"
        final_clean_dir.mkdir(exist_ok=True)
        for f in clean_dir.glob("*_clean.png"):
            shutil.copy2(f, final_clean_dir / f.name)
        log(f"""Clean image copied to: {final_clean_dir}""")

        # ========== Stage 2: HRNet Classification ==========
        stage2 = Stage2HRNet(args.hrnet_config, device=args.device)
        stage2.load_model(args.hrnet_weights)
        results, class_sequence = stage2.run(final_clean_dir, output_dir)

        # ========== Completed ==========
        log("=" * 60)
        log("""Processing complete!""")
        log("=" * 60)

        # Display Category Sequence
        if class_sequence:
            log(f"""\n Category Sequence Code (4-point sample):""")
            log(f"  {class_sequence}")
        else:
            log("""\n No 4-point sample detected""")

        log(f"""\n All results saved in: {output_dir}""")

        # Clean up temporary files
        if not args.keep_temp:
            log("""Cleaning up temporary files...""")
            shutil.rmtree(temp_dir)

        log("""Done!""")

        return 0

    except Exception as e:
        log(f"""Error: {e}""", "ERROR")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
