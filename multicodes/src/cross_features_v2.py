"""Extended per-cross feature extraction."""

import math

import cv2
import numpy as np

from dot_features import (
    estimate_cross_geometry,
    extract_cross_features,
    imread_unicode,
)

ANGLE_BINS = 720


def largest_component_mask(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        raise ValueError("no foreground")
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label


def radial_profile(mask, center, bins=ANGLE_BINS):
    ys, xs = np.nonzero(mask)
    dx = xs - center[0]
    dy = ys - center[1]
    radius = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx) % (2 * math.pi)
    index = (angle / (2 * math.pi) * bins).astype(int) % bins
    profile = np.zeros(bins)
    np.maximum.at(profile, index, radius)
    return profile


def smooth_circular(values, window=15):
    kernel = np.ones(window) / window
    padded = np.concatenate([values[-window:], values, values[:window]])
    return np.convolve(padded, kernel, mode="same")[window:-window]


def arm_features(mask, center):
    """Locate the two arm axes and measure how far they deviate from design.

    A cross has four arms along two axes that should sit at 90 degrees. Rather
    than hunting for peaks, take the direction of greatest extent as axis one,
    then search the perpendicular half of the circle for axis two.
    """
    profile = radial_profile(mask, center)
    n = len(profile)
    smoothed = smooth_circular(profile, window=31)
    degrees = np.arange(n) / n * 360.0

    index_1 = int(np.argmax(smoothed))
    index_2 = (index_1 + n // 2) % n
    span = n // 8
    # axis two is perpendicular to axis one, so search around index_1 +/- 90deg
    perpendicular = (index_1 + n // 4) % n
    window = [(perpendicular + offset) % n for offset in range(-span, span + 1)]
    index_3 = max(window, key=lambda k: smoothed[k])
    index_4 = (index_3 + n // 2) % n

    # axis one runs index_1 <-> index_2, axis two runs index_3 <-> index_4
    angle_1 = float(degrees[index_1])
    angle_3 = float(degrees[index_3])

    difference = abs(angle_1 - angle_3) % 180.0
    axis_angle = float(min(difference, 180.0 - difference))

    lengths = [
        float(smoothed[index_1]),
        float(smoothed[index_2]),
        float(smoothed[index_3]),
        float(smoothed[index_4]),
    ]

    values = {
        "radial_max": float(profile.max()),
        "radial_std": float(profile.std()),
        "radial_mean_over_max": float(profile.mean() / profile.max()),
        "axis_angle_deg": axis_angle,
        "axis_deviation_from_90_deg": abs(axis_angle - 90.0),
        "axis1_angle_deg": angle_1,
    }
    for slot, length in enumerate(lengths, start=1):
        values[f"arm_len_{slot}"] = length
    values["arm_len_asymmetry"] = float(
        abs(lengths[0] - lengths[1]) / max(lengths[0], lengths[1], 1e-9)
    )
    values["arm_len_asymmetry_2"] = float(
        abs(lengths[2] - lengths[3]) / max(lengths[2], lengths[3], 1e-9)
    )
    values["arm_len_ratio"] = float((lengths[0] + lengths[1]) / max(lengths[2] + lengths[3], 1e-9))
    return values


def shape_features(mask, gray, center):
    ys, xs = np.nonzero(mask)
    points = np.column_stack([xs, ys]).astype(np.float32)
    area = float(mask.sum())
    (_, _), (w, h), _ = cv2.minAreaRect(points)
    rect_area = float(w * h)
    hull = cv2.convexHull(points)
    hull_area = float(cv2.contourArea(hull))
    perimeter = float(cv2.arcLength(hull, True))
    return {
        "outline_area": area,
        "rect_w": float(w),
        "rect_h": float(h),
        "rect_aspect": float(max(w, h) / max(1.0, min(w, h))),
        "rect_fill_ratio": area / rect_area if rect_area else 0.0,
        "convexity": area / hull_area if hull_area else 0.0,
        "hull_perimeter": perimeter,
        "compactness": (4 * math.pi * area / (perimeter**2) if perimeter else 0.0),
        "centroid_offset_x": float(xs.mean() - center[0]),
        "centroid_offset_y": float(ys.mean() - center[1]),
        "centroid_offset_r": float(np.hypot(xs.mean() - center[0], ys.mean() - center[1])),
    }


def height_features(mask, gray):
    values = gray[mask].astype(np.float64)
    ys, xs = np.nonzero(mask)
    design = np.column_stack([xs.astype(np.float64), ys.astype(np.float64), np.ones(xs.size)])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    plane = design @ coefficients
    detrended = values - plane

    features = {
        "height_mean": float(values.mean()),
        "height_std": float(values.std()),
        "height_p10": float(np.percentile(values, 10)),
        "height_p50": float(np.percentile(values, 50)),
        "height_p90": float(np.percentile(values, 90)),
        "height_range": float(values.max() - values.min()),
        "height_iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "height_skew": float(((values - values.mean()) ** 3).mean() / (values.std() ** 3 + 1e-12)),
        "roughness_rms": float(np.sqrt((detrended**2).mean())),
        "tilt_x": float(coefficients[0]),
        "tilt_y": float(coefficients[1]),
        "tilt_magnitude": float(np.hypot(coefficients[0], coefficients[1])),
    }

    # Height contrast between the body of the cross and the background ring.
    dilated = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1).astype(
        bool
    )
    ring = dilated & ~mask
    if ring.any():
        features["height_vs_background"] = float(values.mean() - gray[ring].mean())
        features["background_mean"] = float(gray[ring].mean())
        features["background_std"] = float(gray[ring].std())
    else:
        features["height_vs_background"] = float("nan")
        features["background_mean"] = float("nan")
        features["background_std"] = float("nan")
    return features


def extract_extended(image_path, code):
    base = extract_cross_features(image_path, code)
    image = imread_unicode(image_path)
    gray_u8 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray_u8.astype(np.float64)
    mask = largest_component_mask(gray_u8)
    center, scale, _ = estimate_cross_geometry(gray_u8)

    record = {
        "image": str(image_path),
        "code": code,
    }
    # keep the original per-dot features under their own names
    for name, value in zip(base["all_feature_names"], base["all_features"]):
        record[name] = float(value)
    record.update(shape_features(mask, gray, center))
    record.update(height_features(mask, gray))
    record.update(arm_features(mask, center))
    return record
