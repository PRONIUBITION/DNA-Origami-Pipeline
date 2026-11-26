"""Re-run feature extraction and report quantities that are computed but
thrown away, plus shape/height descriptors the pipeline never looks at.

Everything here is derived from data that already exists on disk, so it
shows what could be added without a single new AFM image.
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dot_features import (
    canonical_points,
    detect_dots_at_expected_sites,
    estimate_cross_geometry,
    imread_unicode,
    similarity_transform,
)

ROOT = Path(__file__).resolve().parents[1]


def rotation_angle(matrix):
    """Extract the rotation angle in degrees from a 2x2 rotation matrix."""
    return math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))


def shape_features(gray, center, scale):
    """Descriptors of the whole cross outline and of the height field."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = labels == label

    area = float(mask.sum())
    ys, xs = np.nonzero(mask)
    points = np.column_stack([xs, ys]).astype(np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(points)
    rect_area = float(w * h)
    hull = cv2.convexHull(points)
    hull_area = float(cv2.contourArea(hull))
    perimeter = float(cv2.arcLength(hull, True))

    # Orientation of the fitted rectangle: this rotates with how the molecule
    # happened to land on the surface.
    rect = cv2.minAreaRect(points)
    orientation = float(rect[2])

    return {
        "outline_area": area,
        "min_area_rect_w": float(w),
        "min_area_rect_h": float(h),
        "rect_fill_ratio": area / rect_area if rect_area else 0.0,
        "convexity": area / hull_area if hull_area else 0.0,
        "hull_perimeter": perimeter,
        "compactness": (4 * math.pi * area / (perimeter**2) if perimeter else 0.0),
        "rect_orientation_deg": orientation,
        "centroid_offset_x": float(xs.mean() - center[0]),
        "centroid_offset_y": float(ys.mean() - center[1]),
        "mask_intensity_mean": float(gray[mask].mean()),
        "mask_intensity_std": float(gray[mask].std()),
        "mask_intensity_p25": float(np.percentile(gray[mask], 25)),
        "mask_intensity_p50": float(np.percentile(gray[mask], 50)),
        "mask_intensity_p75": float(np.percentile(gray[mask], 75)),
        "intensity_range": float(gray[mask].max() - gray[mask].min()),
    }


def main():
    rows = []
    for group in ("111", "222"):
        codes = [
            record["code"]
            for record in json.loads(
                (
                    ROOT
                    / f"outputs/qwen_recognition/{group}/thinking_clean"
                    / "random_expand/features.json"
                ).read_text(encoding="utf-8")
            )
        ]
        crops = ROOT / f"outputs/qwen_recognition/{group}/thinking_clean/input"
        for index, code in enumerate(codes, start=1):
            path = crops / f"cross_{index:02d}.png"
            image = imread_unicode(path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            center, scale, _ = estimate_cross_geometry(gray)
            target = canonical_points(code)
            expected = center + scale * (target - np.array([207.0, 199.0]))
            dots = detect_dots_at_expected_sites(gray, expected, scale)
            actual = np.array([[d["subpixel_x"], d["subpixel_y"]] for d in dots], float)
            fit_scale, fit_rotation, _ = similarity_transform(actual, target)
            row = {
                "group": group,
                "index": index,
                "code": code,
                "fit_scale": float(fit_scale),
                "fit_rotation_deg": rotation_angle(fit_rotation),
            }
            row.update(shape_features(gray, center, scale))
            rows.append(row)

    keys = [k for k in rows[0] if k not in ("group", "index", "code")]
    print(f"crosses: {len(rows)}")
    print()
    print(f"{'quantity':<26}{'min':>10}{'max':>10}{'std':>10}{'uniq':>7}   varies?")
    for key in keys:
        values = np.array([row[key] for row in rows], float)
        distinct = len(set(np.round(values, 6).tolist()))
        print(
            f"{key:<26}{values.min():>10.3f}{values.max():>10.3f}"
            f"{values.std():>10.3f}{distinct:>7}   "
            f"{'yes' if values.std() > 0 else 'CONSTANT'}"
        )

    out = ROOT / "outputs" / "discarded_features_audit.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
