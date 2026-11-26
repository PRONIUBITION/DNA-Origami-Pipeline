"""White-dot localization, alignment, and base feature extraction."""

import math

import cv2
import numpy as np

# Coordinates from reference/canonical_cross_clean.png.
# Each bit uses one of two mutually exclusive white-dot positions.
CANONICAL_PAIRS = {
    0: {"1": (131.0, 31.0), "0": (132.0, 360.0)},
    1: {"1": (266.0, 33.0), "0": (266.0, 360.0)},
    2: {"1": (381.0, 153.0), "0": (32.0, 157.0)},
    3: {"1": (359.0, 257.0), "0": (32.0, 240.0)},
}


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def estimate_cross_geometry(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        raise ValueError("Cannot detect cross foreground")
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = int(stats[label, cv2.CC_STAT_LEFT])
    y = int(stats[label, cv2.CC_STAT_TOP])
    w = int(stats[label, cv2.CC_STAT_WIDTH])
    h = int(stats[label, cv2.CC_STAT_HEIGHT])
    center = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float64)
    return center, w / 414.0, (x, y, w, h)


def detect_dots_at_expected_sites(gray, expected_points, scale):
    small_sigma = max(1.0, 31.5 * scale * 0.35)
    large_sigma = small_sigma * 2.5
    small = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), small_sigma)
    large = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), large_sigma)
    dog = small - large

    points = []
    search_radius = max(6, int(42 * scale))
    for expected_x, expected_y in expected_points:
        x1 = max(0, int(round(expected_x)) - search_radius)
        y1 = max(0, int(round(expected_y)) - search_radius)
        x2 = min(gray.shape[1], int(round(expected_x)) + search_radius + 1)
        y2 = min(gray.shape[0], int(round(expected_y)) + search_radius + 1)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Expected dot search window is outside image")

        window = dog[y1:y2, x1:x2]
        local_y, local_x = np.unravel_index(np.argmax(window), window.shape)
        x = x1 + int(local_x)
        y = y1 + int(local_y)
        refined_x, refined_y = refine_peak_subpixel(dog, x, y)
        response = float(dog[y, x])

        radius = max(4, int(31.5 * scale))
        ox1 = max(0, x - radius)
        oy1 = max(0, y - radius)
        ox2 = min(gray.shape[1], x + radius + 1)
        oy2 = min(gray.shape[0], y + radius + 1)
        local = gray[oy1:oy2, ox1:ox2].astype(np.float32)
        peak = float(gray[y, x])
        background = float(np.percentile(local, 25))
        threshold = background + 0.5 * (peak - background)
        area = int((local >= threshold).sum())
        points.append(
            {
                "x": float(x),
                "y": float(y),
                "subpixel_x": refined_x,
                "subpixel_y": refined_y,
                "response": response,
                "peak": peak,
                "background": background,
                "contrast": peak - background,
                "area": area,
            }
        )
    return points


def refine_peak_subpixel(heatmap, x, y, radius=2):
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(heatmap.shape[1], x + radius + 1)
    y2 = min(heatmap.shape[0], y + radius + 1)
    patch = heatmap[y1:y2, x1:x2].astype(np.float64)
    if patch.shape[0] < 3 or patch.shape[1] < 3:
        return float(x), float(y)

    rows, cols = np.indices(patch.shape)
    xx = cols - (x - x1)
    yy = rows - (y - y1)
    design = np.column_stack(
        [
            xx.ravel() ** 2,
            yy.ravel() ** 2,
            (xx * yy).ravel(),
            xx.ravel(),
            yy.ravel(),
            np.ones(xx.size),
        ]
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, patch.ravel(), rcond=None)
    a, b, c, d, e, _ = coefficients
    matrix = np.array([[2 * a, c], [c, 2 * b]], dtype=np.float64)
    rhs = np.array([-d, -e], dtype=np.float64)
    try:
        offset = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return float(x), float(y)
    offset = np.clip(offset, -0.75, 0.75)
    return float(x + offset[0]), float(y + offset[1])


def canonical_points(code):
    return np.array(
        [CANONICAL_PAIRS[index][bit] for index, bit in enumerate(code)],
        dtype=np.float64,
    )


def similarity_transform(source, target):
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = target_centered.T @ source_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt

    variance = np.sum(source_centered**2)
    scale = float(singular_values.sum() / (variance + 1e-12))
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def extract_cross_features(image_path, code):
    image = imread_unicode(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    target_points = canonical_points(code)
    center, scale, bbox = estimate_cross_geometry(gray)
    canonical_center = np.array([207.0, 199.0], dtype=np.float64)
    expected_points = center + scale * (target_points - canonical_center)
    ordered_detected = detect_dots_at_expected_sites(gray, expected_points, scale)

    actual_points = np.array(
        [[point["subpixel_x"], point["subpixel_y"]] for point in ordered_detected],
        dtype=np.float64,
    )
    fit_scale, fit_rotation, fit_translation = similarity_transform(actual_points, target_points)
    mapped_to_canonical = fit_scale * (actual_points @ fit_rotation.T) + fit_translation
    residual = mapped_to_canonical - target_points
    rms = float(np.sqrt(np.mean(residual**2)))

    # The similarity fit also pins down how the molecule happens to sit on the
    # surface. That orientation is an independent degree of freedom and was
    # previously computed and then discarded, so it is recorded here.
    rotation_deg = float(math.degrees(math.atan2(fit_rotation[1, 0], fit_rotation[0, 0])))

    feature_vector = []
    all_features = []
    per_dot = []
    for index, point in enumerate(ordered_detected):
        dx = float(residual[index, 0])
        dy = float(residual[index, 1])
        row = {
            "dot_index": index,
            "bit": int(code[index]),
            "detected": point,
            "residual_x": dx,
            "residual_y": dy,
            "residual_distance": float(math.hypot(dx, dy)),
        }
        per_dot.append(row)
        all_features.extend(
            [
                dx,
                dy,
                point["contrast"],
                point["area"],
                point["response"],
            ]
        )
        feature_vector.extend(
            [
                dx,
                dy,
                point["contrast"],
            ]
        )

    all_features.extend([rms, fit_scale, rotation_deg])
    return {
        "image": str(image_path),
        "code": code,
        "alignment_rms": rms,
        "scale": fit_scale,
        "fit_rotation": fit_rotation.tolist(),
        "fit_rotation_deg": rotation_deg,
        "fit_translation": fit_translation.tolist(),
        "initial_scale": scale,
        "bbox": list(bbox),
        "features": feature_vector,
        "feature_names": [
            f"dot{dot}_{name}" for dot in range(4) for name in ("dx", "dy", "contrast")
        ],
        "all_features": all_features,
        "all_feature_names": [
            f"dot{dot}_{name}"
            for dot in range(4)
            for name in ("dx", "dy", "contrast", "area", "response")
        ]
        + ["rms", "scale", "rotation_deg"],
        "dots": per_dot,
    }
