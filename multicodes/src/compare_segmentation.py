import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from pipeline import (
    imread_unicode,
    imwrite_unicode,
    normalize_binary,
    segment_by_erosion,
)


def find_local_peaks(values, min_distance):
    peaks = []
    for index in range(1, len(values) - 1):
        if values[index] < values[index - 1] or values[index] < values[index + 1]:
            continue
        if any(abs(index - peak) < min_distance for peak in peaks):
            continue
        peaks.append(index)
    return peaks


def box_from_segment(axis_start, axis_end, horizontal, image_shape, padding):
    h, w = image_shape[:2]
    if horizontal:
        return (
            max(0, axis_start - padding),
            0,
            min(w, axis_end + padding),
            h,
        )
    return (
        0,
        max(0, axis_start - padding),
        w,
        min(h, axis_end + padding),
    )


def segment_period(image, reference_shape, padding):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = normalize_binary(gray)
    h, w = gray.shape[:2]
    horizontal = w >= h
    length = w if horizontal else h

    profile = binary.sum(axis=0 if horizontal else 1).astype(np.float32)
    kernel = np.ones(7, dtype=np.float32) / 7.0
    smooth = np.convolve(profile, kernel, mode="same")
    centered = smooth - smooth.mean()

    autocorr = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    if autocorr[0] > 0:
        autocorr = autocorr / autocorr[0]

    reference_length = reference_shape[1] if horizontal else reference_shape[0]
    min_lag = max(8, int(reference_length * 0.55))
    max_lag = min(len(autocorr) - 2, int(reference_length * 1.8))
    min_lag = min(min_lag, max_lag - 1)

    peaks = find_local_peaks(autocorr, max(2, reference_length // 4))
    candidate_peaks = [p for p in peaks if min_lag <= p <= max_lag]
    if candidate_peaks:
        period = max(candidate_peaks, key=lambda p: autocorr[p])
    else:
        period = max(min_lag, int(round(reference_length)))

    count = max(1, int(round(length / period)))
    expected_boundaries = np.linspace(0, length, count + 1)
    boundaries = [0]

    for expected in expected_boundaries[1:-1]:
        radius = max(3, int(period * 0.2))
        start = max(1, int(expected - radius))
        end = min(length - 1, int(expected + radius))
        if end <= start:
            boundaries.append(int(expected))
        else:
            local = smooth[start:end]
            boundaries.append(start + int(np.argmin(local)))
    boundaries.append(length)
    boundaries = sorted(set(boundaries))

    boxes = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < reference_length * 0.45:
            continue
        boxes.append(box_from_segment(start, end, horizontal, image.shape, padding))

    return (
        boxes,
        binary,
        {
            "period_px": int(period),
            "count": len(boxes),
            "profile": smooth.tolist(),
            "autocorr": autocorr.tolist(),
        },
    )


def segment_template(
    image,
    template,
    padding,
    threshold,
    min_distance_ratio,
    max_crosses,
    count_hint=None,
    min_score=0.15,
):
    chain_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    if template_gray.shape[0] > chain_gray.shape[0] or template_gray.shape[1] > chain_gray.shape[1]:
        return [], None, {"error": "template_larger_than_chain"}

    result = cv2.matchTemplate(chain_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    result = np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)

    h, w = template_gray.shape[:2]
    min_distance = max(4, int(min(w, h) * min_distance_ratio))
    boxes = []
    scores = []
    work = result.copy()

    target_count = min(max_crosses, count_hint) if count_hint else max_crosses
    while len(boxes) < target_count:
        _, max_value, _, max_location = cv2.minMaxLoc(work)
        effective_threshold = threshold if count_hint is None else min_score
        if max_value < effective_threshold:
            break
        x, y = max_location
        boxes.append(
            (
                max(0, x - padding),
                max(0, y - padding),
                min(image.shape[1], x + w + padding),
                min(image.shape[0], y + h + padding),
            )
        )
        scores.append(float(max_value))

        x1 = max(0, x - min_distance)
        y1 = max(0, y - min_distance)
        x2 = min(work.shape[1], x + min_distance + 1)
        y2 = min(work.shape[0], y + min_distance + 1)
        work[y1:y2, x1:x2] = -1.0

    boxes.sort(key=lambda box: (box[1], box[0]))
    return (
        boxes,
        result,
        {
            "count": len(boxes),
            "scores": scores,
            "threshold": threshold,
            "count_hint": count_hint,
            "template_size": [int(w), int(h)],
        },
    )


def draw_boxes(image, boxes, color):
    vis = image.copy()
    for index, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            str(index),
            (x1 + 3, min(y2 - 4, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def add_header(image, title):
    header = np.zeros((36, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header,
        title,
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([header, image])


def save_method(
    method_dir,
    method_name,
    image,
    boxes,
    method_color,
    summary,
):
    method_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = method_dir / "crops"
    crops_dir.mkdir(exist_ok=True)

    vis = draw_boxes(image, boxes, method_color)
    imwrite_unicode(method_dir / "boxes.png", vis)

    for index, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = box
        crop = image[y1:y2, x1:x2]
        imwrite_unicode(crops_dir / f"cross_{index:02d}.png", crop)

    summary[method_name] = {
        **summary.get(method_name, {}),
        "count": len(boxes),
        "boxes": [[int(v) for v in box] for box in boxes],
    }


def main():
    parser = argparse.ArgumentParser(description="Compare cross-chain segmentation methods.")
    parser.add_argument(
        "--single",
        default="input/single",
        help="single-cross template image or directory",
    )
    parser.add_argument(
        "--multi",
        default="input/multi",
        help="cross-chain image or directory",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--erode-radius", type=int, default=4)
    parser.add_argument("--min-area", type=int, default=800)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--template-threshold", type=float, default=0.45)
    parser.add_argument("--min-distance-ratio", type=float, default=0.5)
    parser.add_argument("--max-crosses", type=int, default=20)
    args = parser.parse_args()

    if args.output is None:
        output_dir = (
            Path(__file__).parent
            / "outputs"
            / f"segmentation_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    else:
        output_dir = Path(args.output)

    single_path = Path(args.single)
    single_files = (
        [single_path]
        if single_path.is_file()
        else sorted(
            p
            for p in single_path.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        )
    )
    template = imread_unicode(single_files[0]) if single_files else None
    if template is None:
        raise SystemExit("No single-cross template image found")

    multi_path = Path(args.multi)
    multi_files = (
        [multi_path]
        if multi_path.is_file()
        else sorted(
            p
            for p in multi_path.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        )
    )

    summary = {
        "template": str(single_files[0]),
        "template_size": [template.shape[1], template.shape[0]],
        "methods": {},
    }

    for image_path in multi_files:
        image = imread_unicode(image_path)
        if image is None:
            continue
        base_dir = output_dir / image_path.stem

        boxes_erode, _, _ = segment_by_erosion(
            image, args.erode_radius, args.min_area, args.padding
        )
        save_method(
            base_dir / "erode",
            "erode",
            image,
            boxes_erode,
            (0, 0, 255),
            summary["methods"],
        )

        boxes_period, _, period_info = segment_period(image, template.shape, args.padding)
        save_method(
            base_dir / "period",
            "period",
            image,
            boxes_period,
            (0, 255, 0),
            summary["methods"],
        )
        summary["methods"]["period"].update(
            {key: value for key, value in period_info.items() if key not in ("profile", "autocorr")}
        )

        boxes_template, _, template_info = segment_template(
            image,
            template,
            args.padding,
            args.template_threshold,
            args.min_distance_ratio,
            args.max_crosses,
        )
        save_method(
            base_dir / "template",
            "template",
            image,
            boxes_template,
            (255, 0, 0),
            summary["methods"],
        )
        summary["methods"]["template"].update(template_info)

        boxes_hybrid, _, hybrid_info = segment_template(
            image,
            template,
            args.padding,
            args.template_threshold,
            args.min_distance_ratio,
            args.max_crosses,
            count_hint=len(boxes_period),
            min_score=0.15,
        )
        save_method(
            base_dir / "hybrid",
            "hybrid",
            image,
            boxes_hybrid,
            (255, 255, 0),
            summary["methods"],
        )
        summary["methods"]["hybrid"].update(hybrid_info)

        comparison = np.vstack(
            [
                add_header(
                    draw_boxes(image, boxes_erode, (0, 0, 255)),
                    f"erode: {len(boxes_erode)}",
                ),
                add_header(
                    draw_boxes(image, boxes_period, (0, 255, 0)),
                    f"period: {len(boxes_period)} (T={period_info['period_px']}px)",
                ),
                add_header(
                    draw_boxes(image, boxes_template, (255, 0, 0)),
                    f"template: {len(boxes_template)}",
                ),
                add_header(
                    draw_boxes(image, boxes_hybrid, (255, 255, 0)),
                    f"hybrid: {len(boxes_hybrid)}",
                ),
            ]
        )
        imwrite_unicode(base_dir / "methods_comparison.png", comparison)

        print(
            f"{image_path.name}: "
            f"erode={len(boxes_erode)}, "
            f"period={len(boxes_period)}, "
            f"template={len(boxes_template)}, "
            f"hybrid={len(boxes_hybrid)}"
        )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
