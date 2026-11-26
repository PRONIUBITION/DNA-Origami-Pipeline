"""Run Qwen thinking-mode recognition over the segmented crops.

Writes the same layout as the existing 111/222 runs so downstream feature
extraction and encoding can treat every group identically.
"""

import argparse
import json
import os
import time
from pathlib import Path

from pipeline import (
    call_qwen,
    extract_pptx_reference,
    imread_unicode,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="*", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--pause", type=float, default=0.5)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="keep crops that already have a valid 4-bit code",
    )
    args = parser.parse_args()

    config_path = ROOT / "config" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig")) if config_path.exists() else {}
    api_key = (
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("QWEN_API_KEY")
        or config.get("api_key")
    )
    base_url = config.get("base_url") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model = config.get("model") or "qwen3.8-max"
    if not api_key:
        raise SystemExit("Set DASHSCOPE_API_KEY, QWEN_API_KEY, or multicodes/config/config.json")

    for stem in args.groups:
        run_dir = ROOT / "outputs/qwen_recognition" / stem / "thinking_clean"
        crops = sorted((run_dir / "input").glob("cross_*.png"))
        if args.limit:
            crops = crops[: args.limit]
        if not crops:
            print(f"[skip] no crops for {stem}")
            continue

        reference_text, reference_images = extract_pptx_reference(
            ROOT / "assets" / "reference", run_dir / "reference"
        )
        all_results = []
        for number, crop_path in enumerate(crops, start=1):
            result_path = run_dir / f"cross_{number:02d}_results.json"
            if args.skip_existing and result_path.exists():
                previous = json.loads(result_path.read_text(encoding="utf-8"))[0]
                code = str(previous.get("code", ""))
                if len(code) == 4 and not set(code) - {"0", "1"}:
                    all_results.append(previous)
                    print(f"{stem} [{number:>2}/{len(crops)}] kept {code}")
                    continue

            image = imread_unicode(crop_path)
            result = None
            for attempt in range(1, args.retries + 1):
                try:
                    result = call_qwen(
                        image,
                        api_key,
                        base_url,
                        model,
                        args.timeout,
                        reference_text=reference_text,
                        reference_images=reference_images,
                        enable_thinking=True,
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"  {stem} {crop_path.name} attempt {attempt} failed: {exc}")
                    if attempt == args.retries:
                        result = {
                            "code": "ERROR",
                            "confidence": None,
                            "reason": str(exc),
                        }
                    else:
                        time.sleep(3)

            item = {
                "image": str(crop_path),
                "cross_index": number,
                "bbox": [0, 0, image.shape[1], image.shape[0]],
                "crop": crop_path.name,
                **result,
            }
            all_results.append(item)
            print(
                f"{stem} [{number:>2}/{len(crops)}] {crop_path.name}: "
                f"{item.get('code')} conf={item.get('confidence')}"
            )
            result_path.write_text(
                json.dumps([item], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            time.sleep(args.pause)

        (run_dir / "all_results.json").write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_dir / "merged_results.json").write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        codes = [item.get("code", "ERROR") for item in all_results]
        (run_dir / "codes.txt").write_text("\n".join(codes) + "\n", encoding="utf-8")
        print(f"{stem}: {codes}")


if __name__ == "__main__":
    main()
