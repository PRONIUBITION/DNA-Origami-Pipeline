"""Combine template matching with period segmentation for the new chains.

Template matching gives high-confidence boxes but misses crosses whose
appearance differs from the template. Period segmentation tiles the chain
evenly and covers those gaps, but can cut where there is no cross. This script
takes template boxes as primary and adds period-based centres only in regions
the template did not cover; Qwen then decides whether each crop is a cross.
"""

import json
from pathlib import Path

import numpy as np

from compare_segmentation import draw_boxes, segment_period, segment_template
from pipeline import imread_unicode, imwrite_unicode
from segment_chains import fit_template

ROOT = Path(__file__).resolve().parents[1]


def main():
    groups = ["333", "444", "555", "666", "777", "888", "999"]
    summary = {}
    for stem in groups:
        index = int(stem[0])
        chain = imread_unicode(ROOT / "input/multi" / f"{stem}.png")
        template = imread_unicode(ROOT / "input/single" / f"{index}.png")
        template, _ = fit_template(template, chain)
        th, tw = template.shape[:2]

        boxes, _, info = segment_template(chain, template, 0, 0.30, 0.75, 40, None, 0.15)
        period_boxes, _, pinfo = segment_period(chain, template.shape, 0)

        covered = np.zeros(chain.shape[1], dtype=bool)
        for x1, _, x2, _ in boxes:
            covered[x1:x2] = True

        added = []
        for x1, _, x2, _ in period_boxes:
            centre = (x1 + x2) // 2
            if covered[max(0, centre - tw // 4) : centre + tw // 4].any():
                continue
            left = max(0, centre - tw // 2)
            right = min(chain.shape[1], left + tw)
            left = right - tw
            added.append((left, 0, right, chain.shape[0]))

        combined = sorted(
            [(b, "template") for b in boxes] + [(b, "period") for b in added],
            key=lambda item: item[0][0],
        )

        out_dir = ROOT / "outputs/qwen_recognition" / stem / "thinking_clean"
        crops_dir = out_dir / "input"
        crops_dir.mkdir(parents=True, exist_ok=True)
        for stale in crops_dir.glob("cross_*.png"):
            stale.unlink()

        records = []
        for number, (box, source) in enumerate(combined, start=1):
            x1, y1, x2, y2 = box
            name = f"cross_{number:02d}.png"
            imwrite_unicode(crops_dir / name, chain[y1:y2, x1:x2])
            records.append(
                {
                    "crop": name,
                    "source": source,
                    "box": [int(v) for v in box],
                }
            )
        imwrite_unicode(
            out_dir / f"{stem}_boxes.png",
            draw_boxes(chain, [r["box"] for r in records], (0, 255, 0)),
        )
        summary[stem] = {
            "template_boxes": len(boxes),
            "period_boxes": len(period_boxes),
            "added_from_period": len(added),
            "total": len(records),
            "period_px": pinfo.get("period_px"),
            "template_size": [tw, th],
            "chain_size": [chain.shape[1], chain.shape[0]],
            "crops": records,
        }
        print(f"{stem}: template {len(boxes)} + period {len(added)} = {len(records)} crops")

    (ROOT / "outputs/qwen_recognition/union_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    total = sum(v["total"] for v in summary.values())
    print(f"total new crops: {total}")


if __name__ == "__main__":
    main()
