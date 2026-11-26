import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from settings import PROJECT_ROOT, output_path


OUTPUT_DIR = PROJECT_ROOT / "stage2_hrnet" / "data" / "fourbits" / "segments_and_annotations"
METADATA_PATH = OUTPUT_DIR / "_segmentation_metadata.json"
YOLO_RUN = output_path("yolo_runs", "predict-seg", "fourbits_stage1")
MARGIN_RATIO = 0.10
MIN_MARGIN = 20


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def imwrite_unicode(path, image):
    ext = Path(path).suffix.lower()
    if ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    else:
        params = []
    ok, buffer = cv2.imencode(ext, image, params)
    if not ok:
        raise RuntimeError(f"failed to encode image: {path}")
    with open(path, "wb") as f:
        f.write(buffer)


def parse_yolo_line(line, img_w, img_h):
    parts = line.strip().split()
    class_id = int(parts[0])
    values = [float(x) for x in parts[1:]]
    confidence = values[-1] if len(values) % 2 == 1 else None
    coords = values[:-1] if len(values) % 2 == 1 else values
    points = []
    for i in range(0, len(coords), 2):
        x = coords[i] * img_w
        y = coords[i + 1] * img_h
        points.append([int(round(x)), int(round(y))])
    return class_id, points, confidence


def write_labelme_template(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
    annotation = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [],
        "imagePath": image_path.name,
        "imageHeight": img.shape[0],
        "imageWidth": img.shape[1],
    }
    json_path = image_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", required=True)
    parser.add_argument("--dilation-pct", type=float, default=0.25)
    parser.add_argument("--kernel-size", type=int, default=None)
    args = parser.parse_args()

    metadata = json.loads(Path(METADATA_PATH).read_text(encoding="utf-8"))
    entry = next(
        (item for item in metadata["instances"] if item["name"] == args.instance),
        None,
    )
    if entry is None:
        raise SystemExit(f"instance not found: {args.instance}")

    source_image = Path(entry["source_image"])
    labels_path = Path(YOLO_RUN) / "labels" / f"{source_image.stem}.txt"
    lines = labels_path.read_text(encoding="utf-8").splitlines()
    if entry["yolo_line"] >= len(lines):
        raise SystemExit("YOLO line index is out of range")

    image = imread_unicode(source_image)
    if image is None:
        raise SystemExit(f"cannot read source image: {source_image}")
    img_h, img_w = image.shape[:2]

    class_id, points, confidence = parse_yolo_line(lines[entry["yolo_line"]], img_w, img_h)
    if class_id != 0 or len(points) < 3:
        raise SystemExit("unexpected YOLO polygon")

    x1, y1, x2, y2 = entry["bbox"]

    full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(full_mask, [np.array(points, dtype=np.int32)], 255)
    mask_crop = full_mask[y1:y2, x1:x2]
    image_crop = image[y1:y2, x1:x2]

    crop_h, crop_w = mask_crop.shape[:2]
    if args.kernel_size is not None:
        kernel_size = max(3, args.kernel_size)
    else:
        kernel_size = max(3, int(min(crop_h, crop_w) * args.dilation_pct) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_dilated = cv2.dilate(mask_crop, kernel, iterations=1)
    clean_image = (image_crop * (mask_dilated > 0)).astype(np.uint8)

    output_image = Path(OUTPUT_DIR) / f"{args.instance}.png"
    imwrite_unicode(output_image, clean_image)

    if not output_image.with_suffix(".json").exists():
        write_labelme_template(output_image)

    if args.kernel_size is not None:
        entry["mask_dilation_pixels"] = kernel_size
        entry.pop("mask_dilation_pct", None)
    else:
        entry["mask_dilation_pct"] = args.dilation_pct
        entry.pop("mask_dilation_pixels", None)
    entry["mask_kernel_size"] = kernel_size
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Remasked {args.instance}")
    print(f"  crop: {crop_w}x{crop_h}")
    if args.kernel_size is not None:
        print(f"  fixed kernel: {kernel_size}")
    else:
        print(f"  dilation: {args.dilation_pct:.2f}, kernel: {kernel_size}")
    print(f"  output: {output_image}")


if __name__ == "__main__":
    main()
