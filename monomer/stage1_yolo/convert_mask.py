import cv2
import numpy as np
import json
from pathlib import Path
from settings import data_path, output_path

# ========== Configuration ==========
LABELS_DIR = output_path("yolo_runs", "predict-seg", "labels")
ORIGINAL_IMAGES = data_path("stage1", "images")
OUTPUT_DIR = output_path("stage1", "masks_and_crops")

TARGET_CLASS = 0  # 0=dimmer, keep only this category

# ========== Added: Margin Configuration ==========
MARGIN_RATIO = (
    0.05  # Expansion ratio: 25% (if you feel it is not enough, you can change it to 0.3 or 0.4)
)
MIN_MARGIN = (
    10  # Minimum margin: At least 30 pixels (to prevent insufficient expansion of small targets)
)


# ==================================


def parse_yolo_line(line, img_w, img_h):
    """Parse a line of YOLO txt and return the class and pixel coordinates"""
    parts = line.strip().split()
    if len(parts) < 5:
        return None, None

    class_id = int(parts[0])
    points = []
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            x = float(parts[i]) * img_w
            y = float(parts[i + 1]) * img_h
            points.append([int(x), int(y)])

    return class_id, points


def get_bbox_from_points(points):
    """Calculate the bounding box from the polygon points"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


# ========== Added: Extended bbox function ==========
def expand_bbox(x1, y1, x2, y2, img_w, img_h, ratio=MARGIN_RATIO, min_margin=MIN_MARGIN):
    """Expand the bounding box and add margins around it to prevent cutting off the small white dots.
    ratio: Expand in proportion to the size of the target (0.25=25%)
    min_margin: Minimum margin (pixels) to prevent insufficient expansion of small targets"""
    w = x2 - x1
    h = y2 - y1

    # Calculate the amount of expansion (take the larger of the ratio and the minimum value)
    margin_x = max(int(w * ratio), min_margin)
    margin_y = max(int(h * ratio), min_margin)

    # Expand in all directions
    new_x1 = max(0, x1 - margin_x)
    new_y1 = max(0, y1 - margin_y)
    new_x2 = min(img_w, x2 + margin_x)
    new_y2 = min(img_h, y2 + margin_y)

    return new_x1, new_y1, new_x2, new_y2


def create_single_mask(img_shape, points, bbox):
    """Create a mask for a single instance (full image size), then crop it.
    Note: The current bounding box is already expanded."""
    h, w = img_shape[:2]
    full_mask = np.zeros((h, w), dtype=np.uint8)

    if len(points) >= 3:
        pts = np.array(points, np.int32)
        cv2.fillPoly(full_mask, [pts], 255)

    x1, y1, x2, y2 = bbox
    # Ensure that it does not go beyond the boundaries (double check)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    mask_crop = full_mask[y1:y2, x1:x2]
    return mask_crop, (x1, y1, x2, y2)


def main():
    labels_dir = Path(LABELS_DIR)
    img_dir = Path(ORIGINAL_IMAGES)
    out_dir = Path(OUTPUT_DIR)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)

    metadata = []
    instance_count = 0

    for txt_file in labels_dir.glob("*.txt"):
        img_stem = txt_file.stem
        img_path = img_dir / (img_stem + ".jpg")
        if not img_path.exists():
            img_path = img_dir / (img_stem + ".png")
        if not img_path.exists():
            print(f"""[!] Skip: Original image not found {img_stem}""")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape[:2]

        with open(txt_file, "r") as f:
            lines = f.readlines()

        for line_idx, line in enumerate(lines):
            class_id, points = parse_yolo_line(line, w, h)

            if class_id is None or len(points) < 3:
                continue

            if class_id != TARGET_CLASS:
                continue

            # 1. Calculate the original bbox
            x1, y1, x2, y2 = get_bbox_from_points(points)

            if x2 <= x1 or y2 <= y1:
                continue

            # ========== 2. Key modification: Expand the bbox and leave margins! ==========
            x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, w, h)
            # Now (x1, y1, x2, y2) already includes the extended margins.

            # 3. Create a mask (using the expanded bounding box)
            mask_crop, final_bbox = create_single_mask(img.shape, points, (x1, y1, x2, y2))

            # 4. Crop the original image (same area)
            fx1, fy1, fx2, fy2 = final_bbox
            img_crop = img[fy1:fy2, fx1:fx2]

            # Save the file (the logic for generating the file name remains unchanged)
            base_name = f"{img_stem}_inst{line_idx}"

            mask_path = out_dir / "masks" / f"{base_name}_mask.png"
            cv2.imwrite(str(mask_path), mask_crop)

            img_path_out = out_dir / "images" / f"{base_name}_img.png"
            cv2.imwrite(str(img_path_out), img_crop)

            # Save in NumPy format (0-1 floating-point numbers)
            mask_np = (mask_crop > 0).astype(np.float32)
            np.save(str(out_dir / "masks" / f"{base_name}_mask.npy"), mask_np)

            # Record metadata (added a comparison between the original bounding box and the expanded one for easy debugging)
            metadata.append(
                {
                    "name": base_name,
                    "source_image": str(img_path),
                    "bbox": [int(fx1), int(fy1), int(fx2), int(fy2)],
                    "bbox_size": [int(fx2 - fx1), int(fy2 - fy1)],  # Added: Cropped size
                    "class": int(class_id),
                    "class_name": "dimmer",
                    "mask_pixels": int(np.sum(mask_crop) // 255),
                }
            )

            instance_count += 1
            print(
                f"""[] {base_name}: Cropped size = {fx2 - fx1} x {fy2 - fy1}, Mask area = {np.sum(mask_crop) // 255} px"""
            )

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "total_instances": instance_count,
                "margin_ratio": MARGIN_RATIO,
                "min_margin": MIN_MARGIN,
                "instances": metadata,
            },
            f,
            indent=2,
        )

    print(f"""\n [] Done! A total of {instance_count} individual instances were generated""")
    print(f"""Output Directory: {OUTPUT_DIR}""")
    print(f"""Margin settings: Ratio {MARGIN_RATIO * 100}% + Minimum {MIN_MARGIN} pixels""")


if __name__ == "__main__":
    main()
