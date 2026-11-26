"""Segment every chain image with its matching single-cross template.

input/single/N.png is the template for input/multi/NNN.png. Template matching
is what produced the existing 111/222 crops, so the new groups use the same
method and the same threshold.
"""

import argparse
import json
from pathlib import Path


from compare_segmentation import draw_boxes, segment_template
from pipeline import imread_unicode, imwrite_unicode

ROOT = Path(__file__).resolve().parents[1]


def fit_template(template, chain):
    """Template must fit inside the chain; trim evenly if it is one pixel over."""
    th, tw = template.shape[:2]
    ch, cw = chain.shape[:2]
    if th <= ch and tw <= cw:
        return template, False
    top = max(0, (th - ch) // 2)
    left = max(0, (tw - cw) // 2)
    cropped = template[top : top + min(th, ch), left : left + min(tw, cw)]
    return cropped, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--groups", nargs="*", default=None, help="chain stems to process, e.g. 333 444"
    )
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--max-crosses", type=int, default=40)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument(
        "--min-distance-ratio",
        type=float,
        default=0.75,
        help="suppression radius as a fraction of the template short side",
    )
    args = parser.parse_args()

    chains = sorted(ROOT.glob("input/multi/*.png"))
    targets = args.groups or [p.stem for p in chains]
    summary = {}

    for stem in targets:
        chain_path = ROOT / "input/multi" / f"{stem}.png"
        template_index = int(stem[0])
        template_path = ROOT / "input/single" / f"{template_index}.png"
        if not chain_path.exists() or not template_path.exists():
            print(f"[skip] missing pair for {stem}")
            continue

        chain = imread_unicode(chain_path)
        template = imread_unicode(template_path)
        template, trimmed = fit_template(template, chain)
        boxes, _, info = segment_template(
            chain,
            template,
            args.padding,
            args.threshold,
            args.min_distance_ratio,
            args.max_crosses,
            count_hint=None,
            min_score=0.15,
        )

        out_dir = ROOT / "outputs" / "qwen_recognition" / stem / "thinking_clean"
        crops_dir = out_dir / "input"
        crops_dir.mkdir(parents=True, exist_ok=True)
        for existing in crops_dir.glob("cross_*.png"):
            existing.unlink()

        for index, box in enumerate(boxes, start=1):
            x1, y1, x2, y2 = box
            imwrite_unicode(crops_dir / f"cross_{index:02d}.png", chain[y1:y2, x1:x2])
        imwrite_unicode(out_dir / f"{stem}_boxes.png", draw_boxes(chain, boxes, (0, 255, 0)))

        scores = info.get("scores", [])
        summary[stem] = {
            "template": f"single/{template_index}.png",
            "template_trimmed": trimmed,
            "template_size": info.get("template_size"),
            "chain_size": [chain.shape[1], chain.shape[0]],
            "threshold": args.threshold,
            "count": len(boxes),
            "scores": [round(float(s), 3) for s in scores],
            "boxes": [[int(v) for v in b] for b in boxes],
        }
        print(
            f"{stem}: {len(boxes)} crosses from {chain.shape[1]}x"
            f"{chain.shape[0]}  scores "
            f"{[round(float(s), 2) for s in scores]}"
        )

    (ROOT / "outputs" / "qwen_recognition" / "segmentation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v["count"] for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
