"""Draw Qwen recognition results, matching the 111/222 output layout.

Produces two things per group:
  visualizations/cross_XX_result.png   per-crop image with box and code
  visualizations/<group>_chain.png     whole chain with every box and code
"""

import json
from pathlib import Path

import cv2

from compare_segmentation import segment_template
from pairing import template_path
from pipeline import imread_unicode, imwrite_unicode
from segment_chains import fit_template

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "qwen_recognition"


def chain_boxes(stem):
    """Boxes on the chain for a group, from the union summary or matching."""
    for name in ("period_centers_summary.json", "union_summary.json"):
        path = BASE / name
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if stem in data:
            return [tuple(item["box"]) for item in data[stem]["crops"]]
    chain = imread_unicode(ROOT / "input" / "multi" / f"{stem}.png")
    template = imread_unicode(template_path(stem))
    template, _ = fit_template(template, chain)
    boxes, _, _ = segment_template(chain, template, 0, 0.30, 0.75, 40, None, 0.15)
    return boxes


def draw_box_label(image, box, text, colour=(0, 255, 0), scale=0.5):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    tx = x1
    ty = max(th + 4, y1 - 4)
    cv2.rectangle(image, (tx, ty - th - 3), (tx + tw + 4, ty + 3), (0, 0, 0), -1)
    cv2.putText(image, text, (tx + 2, ty), font, scale, colour, thickness, cv2.LINE_AA)


def main():
    for group_dir in sorted(p for p in BASE.iterdir() if p.is_dir()):
        stem = group_dir.name
        run_dir = group_dir / "thinking_clean"
        results_path = run_dir / "merged_results.json"
        if not results_path.exists():
            continue
        results = json.loads(results_path.read_text(encoding="utf-8"))
        results = [
            item
            for item in results
            if len(str(item.get("code", ""))) == 4
            and not set(str(item.get("code", ""))) - {"0", "1"}
        ]
        if not results:
            continue

        viz_dir = run_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        # per-crop images, same naming as the 111/222 runs
        for number, item in enumerate(results, start=1):
            image_path = Path(str(item.get("image", "")))
            if not image_path.exists():
                image_path = run_dir / "input" / f"cross_{number:02d}.png"
            if not image_path.exists():
                continue
            image = imread_unicode(image_path)
            h, w = image.shape[:2]
            vis = image.copy()
            draw_box_label(
                vis,
                (1, 1, w - 2, h - 2),
                f"{number}: {item['code']}",
                (0, 255, 255),
                0.45,
            )
            imwrite_unicode(viz_dir / f"cross_{number:02d}_result.png", vis)

        # whole-chain overview
        chain_path = ROOT / "input" / "multi" / f"{stem}.png"
        if not chain_path.exists():
            continue
        chain = imread_unicode(chain_path)
        boxes = chain_boxes(stem)
        if len(boxes) != len(results):
            print(
                f"{stem}: box count {len(boxes)} != results {len(results)}, skipping chain overview"
            )
            continue
        ordered = sorted(zip(boxes, results), key=lambda pair: pair[0][0])
        vis = chain.copy()
        scale = max(0.5, chain.shape[1] / 1800.0)
        for number, (box, item) in enumerate(ordered, start=1):
            draw_box_label(vis, box, f"{number}: {item['code']}", (0, 255, 0), scale)
        imwrite_unicode(viz_dir / f"{stem}_chain.png", vis)
        print(f"{stem}: {len(results)} crops + chain overview")


if __name__ == "__main__":
    main()
