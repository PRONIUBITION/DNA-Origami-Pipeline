"""Segment curved chains by period, cropping a template-width window per centre.

Template matching fails when a chain curves, because each cross is rotated by a
different amount. Period segmentation only relies on the repeating spacing, so
it still finds every cross. The crop is then taken as a fixed-width window
centred on each period, which keeps every cross whole and every crop the same
size for feature extraction.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from compare_segmentation import draw_boxes, segment_period
from pairing import chain_path, template_path
from pipeline import imread_unicode, imwrite_unicode
from segment_chains import fit_template

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="*", required=True)
    args = parser.parse_args()

    summary = {}
    for stem in args.groups:
        chain = imread_unicode(chain_path(stem))
        template = imread_unicode(template_path(stem))
        template, trimmed = fit_template(template, chain)
        th, tw = template.shape[:2]
        height, width = chain.shape[:2]

        period_boxes, _, info = segment_period(chain, template.shape, 0)

        # template response, used only as a per-crop quality score
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        chain_gray = cv2.cvtColor(chain, cv2.COLOR_BGR2GRAY)
        response = np.nan_to_num(
            cv2.matchTemplate(chain_gray, template_gray, cv2.TM_CCOEFF_NORMED),
            nan=-1.0,
            posinf=-1.0,
            neginf=-1.0,
        )

        crops = []
        for x1, y1, x2, y2 in period_boxes:
            centre = (x1 + x2) // 2
            left = centre - tw // 2
            left = max(0, min(left, width - tw))
            right = left + tw
            score = float(response[:, left:right].max())
            crops.append(
                {
                    "box": [int(left), 0, int(right), int(height)],
                    "centre": int(centre),
                    "template_score": round(score, 3),
                }
            )

        run_dir = ROOT / "outputs" / "qwen_recognition" / stem / "thinking_clean"
        crops_dir = run_dir / "input"
        crops_dir.mkdir(parents=True, exist_ok=True)
        for stale in crops_dir.glob("cross_*.png"):
            stale.unlink()
        for number, item in enumerate(crops, start=1):
            x1, y1, x2, y2 = item["box"]
            item["crop"] = f"cross_{number:02d}.png"
            imwrite_unicode(crops_dir / item["crop"], chain[y1:y2, x1:x2])

        imwrite_unicode(
            run_dir / f"{stem}_boxes.png",
            draw_boxes(chain, [c["box"] for c in crops], (0, 255, 0)),
        )
        scores = [c["template_score"] for c in crops]
        summary[stem] = {
            "chain_size": [width, height],
            "template_size": [tw, th],
            "template_trimmed": trimmed,
            "period_px": info.get("period_px"),
            "count": len(crops),
            "template_score_median": round(float(np.median(scores)), 3),
            "template_score_min": round(float(np.min(scores)), 3),
            "crops": crops,
        }
        print(
            f"{stem}: {len(crops):>2} crops  period={info.get('period_px')}  "
            f"score median={np.median(scores):.2f} min={np.min(scores):.2f}"
        )

    out = ROOT / "outputs/qwen_recognition/period_centers_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    total = sum(v["count"] for v in summary.values())
    print(f"\ntotal crops: {total}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
