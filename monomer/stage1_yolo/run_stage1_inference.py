"""Stage 1: YOLOv5 Inference + Post-Processing Workflow
From the original AFM image clean_images (prepare to input into HRNet)"""

import os
import sys
import subprocess
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
from settings import (
    HRNET_DATA_ROOT,
    YOLOV5_DIR,
    YOLO_WEIGHTS,
    data_path,
    output_path,
)

# ========== Configuration Path ==========
CONFIG = {
    # Input: Original Image Directory
    "input_dir": os.environ.get("STAGE1_INPUT_DIR", data_path("stage1", "raw")),
    # Output root directory (new version with IOU deduplication)
    "output_base": os.environ.get("STAGE1_OUTPUT_DIR", output_path("stage1")),
    # YOLOv5 Configuration
    "yolov5_dir": YOLOV5_DIR,
    "weights": YOLO_WEIGHTS,
    "img_size": 640,  # YOLOv5 Inference Size
    "conf": 0.45,  # Confidence Threshold
    # Category (only dimmer is retained)
    "target_class": 0,
    # BBox extended margin (to prevent cutting off the small white dot)
    "margin_ratio": 0.10,  # Increased to 20%; the original 5% was too small.
    "min_margin": 20,  # Increased to 30 pixels; the original 10 pixels were too small.
    # clean_image dilation parameter
    "dilate_pixels": 6,
}


def cv2_imread_unicode(path):
    """cv2.imread with support for Chinese paths"""
    with open(path, "rb") as f:
        img_array = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)


def cv2_imwrite_unicode(path, img):
    """cv2.imwrite with support for Chinese paths"""
    # Get file extension
    ext = Path(path).suffix.lower()
    if ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    elif ext in [".jpg", ".jpeg"]:
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    else:
        params = []

    # Encode image
    success, buffer = cv2.imencode(ext, img, params)
    if success:
        with open(path, "wb") as f:
            f.write(buffer)
        return True
    return False


def prepare_images(input_dir, output_dir, img_size=640):
    """Prepare images for YOLOv5 inference:
    1. Convert to grayscale
    2. Resize to img_size x img_size (keep aspect ratio, black border padding)"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
    print(f"""{len(image_files)} images found""")

    resize_info = {}

    for img_path in tqdm(image_files, desc="""Prepare images"""):
        # Read image (Chinese path supported)
        img = cv2_imread_unicode(str(img_path))
        if img is None:
            print(f"""[!] Unable to read: {img_path.name}""")
            continue

        # Convert to grayscale
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        h, w = img_gray.shape

        # Calculate zoom ratio (preserve aspect ratio)
        scale = min(img_size / w, img_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Zoom
        resized = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create a square canvas (black border fill)
        square = np.zeros((img_size, img_size), dtype=np.uint8)
        y_offset = (img_size - new_h) // 2
        x_offset = (img_size - new_w) // 2
        square[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        # Save
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), square)

        # Record zoom information (for later restoration)
        resize_info[img_path.stem] = {
            "original_size": (w, h),
            "scale": scale,
            "offset": (x_offset, y_offset),
            "resized_size": (new_w, new_h),
        }

    # Save zoom information
    import json

    with open(output_dir / "resize_info.json", "w") as f:
        json.dump(resize_info, f, indent=2)

    print(f"""[OK] Image is ready and saved to: {output_dir}""")
    return output_dir, resize_info


def run_yolov5_inference(yolov5_dir, weights, source, img_size=640, conf=0.25):
    """Run YOLOv5 inference"""
    yolov5_dir = Path(yolov5_dir)

    # Build command
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

    print("\n" + "=" * 50)
    print("""Running YOLOv5 inference...""")
    print("=" * 50)
    print(f"""Command: {" ".join(cmd)}""")

    # Switch to the yolov5 directory
    original_dir = os.getcwd()
    os.chdir(yolov5_dir)

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("""Warning:""", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"""YOLOv5 inference failed: {e}""")
        print(e.stdout)
        print(e.stderr)
        os.chdir(original_dir)
        return None
    finally:
        os.chdir(original_dir)

    # Find the latest exp directory (sorted by modification time)
    runs_dir = yolov5_dir / "runs" / "predict-seg"
    exp_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("exp")]

    if not exp_dirs:
        print("""[!] Prediction results directory not found""")
        return None

    # Sort by modification time, with the most recent first
    latest_exp = max(exp_dirs, key=lambda d: d.stat().st_mtime)
    labels_dir = latest_exp / "labels"

    if not labels_dir.exists():
        print(f"""[!] Label directory not found: {labels_dir}""")
        return None

    print(f"""[OK] YOLOv5 inference completed. Results are at: {latest_exp}""")
    return latest_exp


def compute_iou(box1, box2):
    """Calculate the IOU (Intersection over Union) of two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate the intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Calculate the union
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / (union_area + 1e-6)


def convert_masks(
    labels_dir,
    original_images,
    output_dir,
    resize_info,
    target_class=0,
    margin_ratio=0.05,
    min_margin=10,
    iou_threshold=0.5,
):
    """Convert YOLO output to a mask and cropped image for a single instance
    New: IOU deduplication to prevent the same object from being detected multiple times"""
    labels_dir = Path(labels_dir)
    original_images = Path(original_images)
    output_dir = Path(output_dir)

    out_images = output_dir / "images"
    out_masks = output_dir / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    txt_files = list(labels_dir.glob("*.txt"))
    print(f"""\n Found {len(txt_files)} label files""")
    print(
        f"""IOU deduplication threshold: {iou_threshold} (IOU> {iou_threshold} is considered a duplicate)"""
    )

    instance_count = 0
    metadata = []
    total_filtered = 0  # Count the number of duplicates filtered

    for txt_file in tqdm(txt_files, desc="""Process mask"""):
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
        # Convert to grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = img.shape

        # Read label
        with open(txt_file, "r") as f:
            lines = f.readlines()

        # ========== Step 1: Collect all candidate detections ==========
        candidates = []
        for line_idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            if class_id != target_class:
                continue

            # Parse polygon points (normalized coordinates)
            points = []
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    x = float(parts[i]) * w
                    y = float(parts[i + 1]) * h
                    points.append([int(x), int(y)])

            if len(points) < 3:
                continue

            # Calculate the original bounding box
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            bw, bh = x2 - x1, y2 - y1

            # Filter out areas that are too small (possibly fragments)
            min_area = 3000
            if bw * bh < min_area:
                continue

            # Filter out areas that are too flat or too long
            aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)
            if aspect_ratio > 2.0:
                continue

            # Save candidate (do not expand; expansion is done before processing)
            candidates.append(
                {"points": points, "bbox": (x1, y1, x2, y2), "area": bw * bh, "line_idx": line_idx}
            )

        # ========== Step 2: IOU deduplication ==========
        # Sort by area from largest to smallest (retain the largest)
        candidates.sort(key=lambda x: x["area"], reverse=True)

        kept = []  # Retained detections
        for cand in candidates:
            is_duplicate = False
            for kept_cand in kept:
                iou = compute_iou(cand["bbox"], kept_cand["bbox"])
                if iou > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append(cand)

        filtered_count = len(candidates) - len(kept)
        total_filtered += filtered_count
        if filtered_count > 0:
            print(
                f"""{img_stem}: Filter {filtered_count} duplicate detections (from {len(candidates)} to {len(kept)})"""
            )

        # ========== Step 3: Process the retained detections ==========
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

            # Save in NumPy format (0-1)
            mask_np = (mask_crop > 0).astype(np.float32)
            np.save(str(out_masks / f"{base_name}_mask.npy"), mask_np)

            metadata.append(
                {
                    "name": base_name,
                    "source": str(img_path),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "class": "dimmer",
                }
            )

            instance_count += 1

    # Save metadata
    import json

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(
            {"total": instance_count, "filtered_duplicates": total_filtered, "instances": metadata},
            f,
            indent=2,
        )

    print(f"""\n [Statistics] Original detections: {instance_count + total_filtered}""")
    print(f"""[Statistics] Duplicates filtered: {total_filtered}""")
    print(f"""[OK] Final output: {instance_count} instances""")
    return instance_count


def clean_images(input_img_dir, input_mask_dir, output_clean_dir, dilate_pixels=6):
    """Apply mask, remove background after dilation, and retain small white dots"""
    input_img_dir = Path(input_img_dir)
    input_mask_dir = Path(input_mask_dir)
    output_clean_dir = Path(output_clean_dir)
    output_clean_dir.mkdir(parents=True, exist_ok=True)

    img_files = list(input_img_dir.glob("*_img.png"))
    print(f"""\n Process {len(img_files)} images for cleaning...""")

    for img_path in tqdm(img_files, desc="""Clean image"""):
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
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_pixels, dilate_pixels))
            mask_dilated = cv2.dilate(mask, kernel, iterations=1)
            mask_dilated = (mask_dilated > 0).astype(np.float32)

            # Apply mask
            clean_img = (img * mask_dilated).astype(np.uint8)

            # Save
            cv2_imwrite_unicode(str(output_clean_dir / f"{stem}_clean.png"), clean_img)

        except Exception as e:
            print(f"[!] {stem}: {e}")

    print(f"""[OK] Clean image saved to: {output_clean_dir}""")
    return output_clean_dir


def copy_to_hrnet(clean_dir, hrnet_raw_dir):
    """Copy the clean image to the HRNet data directory"""
    clean_dir = Path(clean_dir)
    hrnet_raw_dir = Path(hrnet_raw_dir)
    hrnet_raw_dir.mkdir(parents=True, exist_ok=True)

    clean_files = list(clean_dir.glob("*_clean.png"))

    for f in tqdm(clean_files, desc="""Copy to HRNet"""):
        shutil.copy2(f, hrnet_raw_dir / f.name)

    print(f"""[OK] Copy {len(clean_files)} images to: {hrnet_raw_dir}""")
    return len(clean_files)


def main():
    print("=" * 60)
    print("""Stage 1: YOLOv5 Inference + Post-Processing HRNet Input""")
    print("=" * 60)

    # Check weight file
    weights_path = Path(CONFIG["weights"])
    if not weights_path.exists():
        print(f"""[!] Weight file does not exist: {weights_path}""")
        return

    # Ensure that the output path is a Path object
    output_base = Path(CONFIG["output_base"])

    # Step 1: Prepare the image
    print("\n" + "-" * 60)
    print("""Step 1: Prepare the grayscale image and adjust the size...""")
    print("-" * 60)
    prepared_dir, resize_info = prepare_images(
        CONFIG["input_dir"], output_base / "prepared", CONFIG["img_size"]
    )

    # Step 2: YOLOv5 Inference
    print("\n" + "-" * 60)
    print("""Step 2: YOLOv5 inference...""")
    print("-" * 60)
    exp_dir = run_yolov5_inference(
        CONFIG["yolov5_dir"],
        CONFIG["weights"],
        str(prepared_dir),
        CONFIG["img_size"],
        CONFIG["conf"],
    )

    if exp_dir is None:
        print("""[!] YOLOv5 inference failed, exiting""")
        return

    labels_dir = exp_dir / "labels"

    # Step 3: Convert mask
    print("\n" + "-" * 60)
    print("""Step 3: Convert mask...""")
    print("-" * 60)
    convert_output = output_base / "converted"
    convert_masks(
        labels_dir,
        prepared_dir,
        convert_output,
        resize_info,
        CONFIG["target_class"],
        CONFIG["margin_ratio"],
        CONFIG["min_margin"],
    )

    # Step 4: Clean the image
    print("\n" + "-" * 60)
    print("""Step 4: Clean the image (remove the background)...""")
    print("-" * 60)
    clean_dir = clean_images(
        convert_output / "images",
        convert_output / "masks",
        convert_output / "clean_images",
        CONFIG["dilate_pixels"],
    )

    # Step 5: Copy to HRNet
    print("\n" + "-" * 60)
    print("""Step 5: Copy to HRNet data directory...""")
    print("-" * 60)
    hrnet_raw = HRNET_DATA_ROOT / "raw"
    count = copy_to_hrnet(clean_dir, hrnet_raw)

    # Completed
    print("\n" + "=" * 60)
    print("""Phase 1 complete!""")
    print("=" * 60)
    print(f"""[OK] {count} dimmer instances processed""")
    print(f"""[OK] Clean image saved to: {hrnet_raw}""")
    print(f"""\n Next step: Run HRNet data preparation""")
    print(f"  cd dimer/stage2_hrnet")
    print(f"  python prepare_data.py --split")
    print(f"  python prepare_data.py --create_template")


if __name__ == "__main__":
    main()
