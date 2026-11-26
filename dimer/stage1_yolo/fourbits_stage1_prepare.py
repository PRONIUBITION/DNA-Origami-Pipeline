import json
import math
from pathlib import Path

import cv2
import numpy as np
from settings import PROJECT_ROOT, data_path, output_path


YOLO_LABELS_DIR = output_path("yolo_runs", "predict-seg", "fourbits_stage1", "labels")
ORIGINAL_DIR = data_path("fourbits", "test")
OUTPUT_DIR = PROJECT_ROOT / "stage2_hrnet" / "data" / "fourbits" / "segments_and_annotations"

TARGET_CLASS = 0
MARGIN_RATIO = 0.10
MIN_MARGIN = 20
DILATE_PIXELS = 6
IOU_THRESHOLD = 0.5


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def imwrite_unicode(path, image):
    ext = Path(path).suffix.lower()
    if ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    elif ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    else:
        params = []
    ok, buffer = cv2.imencode(ext, image, params)
    if not ok:
        raise RuntimeError(f"failed to encode image: {path}")
    with open(path, "wb") as f:
        f.write(buffer)


def parse_yolo_line(line, img_w, img_h):
    parts = line.strip().split()
    if len(parts) < 5:
        return None, None, None

    class_id = int(parts[0])
    values = [float(x) for x in parts[1:]]

    # The final value is confidence. The remaining values form normalized x/y pairs.
    if len(values) % 2 == 1:
        confidence = values[-1]
        coord_values = values[:-1]
    else:
        confidence = None
        coord_values = values

    points = []
    for i in range(0, len(coord_values), 2):
        x = coord_values[i] * img_w
        y = coord_values[i + 1] * img_h
        points.append([int(round(x)), int(round(y))])

    return class_id, points, confidence


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / (union + 1e-6)


def write_labelme_template(image_path):
    h, w = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE).shape
    annotation = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [],
        "imagePath": image_path.name,
        "imageHeight": h,
        "imageWidth": w,
    }
    json_path = image_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2, ensure_ascii=False)


def main():
    labels_dir = Path(YOLO_LABELS_DIR)
    original_dir = Path(ORIGINAL_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    total_kept = 0
    total_duplicates = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_PIXELS, DILATE_PIXELS))

    for txt_path in sorted(labels_dir.glob("*.txt")):
        stem = txt_path.stem
        image_path = original_dir / f"{stem}.png"
        if not image_path.exists():
            image_path = original_dir / f"{stem}.jpg"
        if not image_path.exists():
            continue

        image = imread_unicode(image_path)
        if image is None:
            continue

        img_h, img_w = image.shape[:2]
        candidates = []

        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line_idx, line in enumerate(lines):
            class_id, points, confidence = parse_yolo_line(line, img_w, img_h)
            if class_id != TARGET_CLASS or len(points) < 3:
                continue

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            if x2 <= x1 or y2 <= y1:
                continue

            area = (x2 - x1) * (y2 - y1)
            if area < 100:
                continue

            candidates.append(
                {
                    "points": points,
                    "bbox": (x1, y1, x2, y2),
                    "area": area,
                    "confidence": confidence,
                    "line_idx": line_idx,
                }
            )

        candidates.sort(key=lambda item: item["area"], reverse=True)
        kept = []

        for candidate in candidates:
            duplicate = False
            for existing in kept:
                if bbox_iou(candidate["bbox"], existing["bbox"]) > IOU_THRESHOLD:
                    duplicate = True
                    break
            if duplicate:
                total_duplicates += 1
            else:
                kept.append(candidate)

        print(f"{stem}: {len(candidates)} candidates -> {len(kept)} kept")

        for idx, candidate in enumerate(kept):
            points = candidate["points"]
            x1, y1, x2, y2 = candidate["bbox"]
            bw = x2 - x1
            bh = y2 - y1

            margin_x = max(int(round(bw * MARGIN_RATIO)), MIN_MARGIN)
            margin_y = max(int(round(bh * MARGIN_RATIO)), MIN_MARGIN)
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(img_w, x2 + margin_x)
            y2 = min(img_h, y2 + margin_y)

            full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            polygon = np.array(points, dtype=np.int32)
            cv2.fillPoly(full_mask, [polygon], 255)

            mask_crop = full_mask[y1:y2, x1:x2]
            image_crop = image[y1:y2, x1:x2]
            mask_dilated = cv2.dilate(mask_crop, kernel, iterations=1)
            clean_image = (image_crop * (mask_dilated > 0)).astype(np.uint8)

            base_name = f"{stem}_inst{idx}_clean"
            out_image = output_dir / f"{base_name}.png"
            imwrite_unicode(out_image, clean_image)
            write_labelme_template(out_image)

            metadata.append(
                {
                    "name": base_name,
                    "source_image": str(image_path),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "crop_size": [int(x2 - x1), int(y2 - y1)],
                    "yolo_confidence": candidate["confidence"],
                    "yolo_line": candidate["line_idx"],
                }
            )
            total_kept += 1

    metadata_path = output_dir / "_segmentation_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "yolo_run": "fourbits_stage1",
                "target_class": TARGET_CLASS,
                "margin_ratio": MARGIN_RATIO,
                "min_margin": MIN_MARGIN,
                "dilate_pixels": DILATE_PIXELS,
                "iou_threshold": IOU_THRESHOLD,
                "total_instances": total_kept,
                "duplicates_removed": total_duplicates,
                "instances": metadata,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Generated {total_kept} dimmer instances")
    print(f"Removed {total_duplicates} duplicate boxes")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
