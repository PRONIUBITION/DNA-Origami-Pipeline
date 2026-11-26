"""Extract the extended feature set for every group in one pass.

Reads each group's merged_results.json for the Qwen code, pairs it with the
crop named by cross_index, and writes one combined feature table.
"""

import json
import argparse
from pathlib import Path

from cross_features_v2 import extract_extended

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "qwen_recognition"
OUT = BASE / "all_features_v2.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()

    records = []
    for group_dir in sorted(p for p in BASE.iterdir() if p.is_dir()):
        if args.groups and group_dir.name not in args.groups:
            continue
        run_dir = group_dir / "thinking_clean"
        results_path = run_dir / "merged_results.json"
        if not results_path.exists():
            continue
        results = json.loads(results_path.read_text(encoding="utf-8"))
        used = 0
        for item in results:
            code = str(item.get("code", ""))
            if len(code) != 4 or set(code) - {"0", "1"}:
                continue
            # merged_results.json records the source image; cross_index is not
            # a reliable crop number for the older runs (all entries are 1).
            crop = Path(str(item.get("image", "")))
            if not crop.exists():
                index = int(item.get("cross_index", 0))
                crop = run_dir / "input" / f"cross_{index:02d}.png"
            if not crop.exists():
                print(f"  [missing] {group_dir.name} {item.get('image')}")
                continue
            try:
                record = extract_extended(crop, code)
            except Exception as exc:  # noqa: BLE001
                print(f"  [failed] {group_dir.name} {crop.name}: {exc}")
                continue
            record["group"] = group_dir.name
            record["crop"] = crop.name
            record["confidence"] = item.get("confidence")
            records.append(record)
            used += 1
        print(f"{group_dir.name}: {used} crosses")

    Path(args.output).write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\ntotal crosses: {len(records)}")
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
